"""Static, layered triangle index. No simulator imports or global XY floor assumption."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SurfaceHit:
    kind: str
    z: float
    source: str = ''
    reason: str = ''
    bounds: tuple = ()


class TerrainIndex:
    def __init__(self, vertices=(), kinds=(), sources=(), *, grid_size=1.0,
                 z_tolerance=0.28, flat_max_slope=0.08, obstacles=(), infer_allowed=None,
                 query_padding=0.075, body_min_z=.22, body_max_z=.65):
        self.grid_size = grid_size
        self.z_tolerance = z_tolerance
        self.query_padding = query_padding
        self.body_min_z, self.body_max_z = body_min_z, body_max_z
        self.obstacles = ()
        self.obstacle_sources = ()
        v = np.asarray(vertices, dtype=float).reshape(-1, 3, 3)
        self.kinds = np.asarray(kinds, dtype='U8')
        self.sources = list(sources) if len(sources) else [''] * len(v)
        if len(v) != len(self.kinds):
            raise ValueError('Triangle/kind count mismatch')
        self.vertices = v
        self.infer_allowed = np.ones(len(v), dtype=bool) if infer_allowed is None else np.asarray(infer_allowed)
        self.buckets = defaultdict(list)
        self.obstacle_buckets = defaultdict(list)
        self.global_surfaces = []
        self.global_obstacles = []
        if len(v):
            self.origin = v[:, 0]
            self.u, self.w = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
            self.det = self.u[:, 0] * self.w[:, 1] - self.u[:, 1] * self.w[:, 0]
            if (np.abs(self.det) < 1e-10).any() or not np.isfinite(v).all():
                raise ValueError('Degenerate or nonfinite support geometry')
            normals = np.cross(self.u, self.w)
            slope = np.linalg.norm(normals[:, :2], axis=1) / np.abs(normals[:, 2])
            self.kinds[(slope > flat_max_slope) & (self.kinds == 'FLAT')] = 'UNKNOWN'
            self.bounds = np.stack([v[:, :, 0].min(1), v[:, :, 0].max(1),
                                    v[:, :, 1].min(1), v[:, :, 1].max(1)], 1)
            for i, box in enumerate(self.bounds):
                if (box[1]-box[0])*(box[3]-box[2]) > 10000 * grid_size**2:
                    self.global_surfaces.append(i)
                    continue
                for key in self._keys(box):
                    self.buckets[key].append(i)
        self.add_obstacles(obstacles)
        self.buckets = {key: np.asarray(ids) for key, ids in self.buckets.items()}

    def add_obstacles(self, boxes, sources=None):
        start = len(self.obstacles)
        self.obstacles += tuple(boxes)
        self.obstacle_sources += tuple(sources) if sources is not None else ('',)*len(boxes)
        for i in range(start, len(self.obstacles)):
            box = self.obstacles[i]
            if (box[1]-box[0])*(box[3]-box[2]) > 10000 * self.grid_size**2:
                self.global_obstacles.append(i)
                continue
            for key in self._keys(box[:4]):
                self.obstacle_buckets[key].append(i)

    def _keys(self, box):
        x0, x1, y0, y1 = box
        for x in range(math.floor(x0 / self.grid_size), math.floor(x1 / self.grid_size) + 1):
            for y in range(math.floor(y0 / self.grid_size), math.floor(y1 / self.grid_size) + 1):
                yield x, y

    def surface_at(self, x, y, reference_z):
        key = (math.floor(x / self.grid_size), math.floor(y / self.grid_size))
        ids = np.asarray(list(self.buckets.get(key, ())) + self.global_surfaces, dtype=int)
        if not len(ids):
            return None
        px, py = x - self.origin[ids, 0], y - self.origin[ids, 1]
        a = (px * self.w[ids, 1] - py * self.w[ids, 0]) / self.det[ids]
        b = (self.u[ids, 0] * py - self.u[ids, 1] * px) / self.det[ids]
        z = self.origin[ids, 2] + a * self.u[ids, 2] + b * self.w[ids, 2]
        inside = (a >= -1e-6) & (b >= -1e-6) & (a + b <= 1 + 1e-6)
        # A too-high step/low overhead surface must not expose the ground beneath it.
        if (inside & (z > reference_z+self.z_tolerance) & (z < reference_z+self.body_max_z)).any():
            i = int(ids[np.flatnonzero(inside & (z > reference_z+self.z_tolerance) & (z < reference_z+self.body_max_z))[0]])
            return SurfaceHit('UNKNOWN',reference_z,self.sources[i],'height_or_overhead_conflict')
        mask = inside & (np.abs(z - reference_z) <= self.z_tolerance)
        if not mask.any():
            return None
        selected = ids[mask]
        heights = z[mask]
        # Highest compatible support masks a floor beneath a stair or object.
        top = heights.max()
        candidates = selected[np.abs(heights - top) < 1e-4]
        i = next((int(i) for i in candidates if self.kinds[i] == 'STAIRS'), int(candidates[0]))
        kind = self.kinds[i]
        for j in list(self.obstacle_buckets.get(key, ())) + self.global_obstacles:
            x0, x1, y0, y1, z0, z1 = self.obstacles[j]
            if x0 - .02 <= x <= x1 + .02 and y0 - .02 <= y <= y1 + .02 and z1 > top + self.body_min_z and z0 < top + self.body_max_z:
                return SurfaceHit('UNKNOWN',float(top),self.obstacle_sources[j] or self.sources[i],f'obstacle_at_support:{j}',tuple(self.obstacles[j]))
        return SurfaceHit(str(kind), float(top), self.sources[i],
                          'unclassified_or_sloped_support' if kind == 'UNKNOWN' else 'collision_support')

    def footprint(self, x, y, reference_z, radius):
        center = self.surface_at(x, y, reference_z)
        if center is None:
            key = (math.floor(x/self.grid_size), math.floor(y/self.grid_size))
            ids = list(self.buckets.get(key, ()))+self.global_surfaces
            if not ids:
                return SurfaceHit('UNKNOWN',reference_z,reason='no_support_in_xy_bucket')
            ids=np.asarray(ids,dtype=int)
            px,py=x-self.origin[ids,0],y-self.origin[ids,1]
            a=(px*self.w[ids,1]-py*self.w[ids,0])/self.det[ids]
            b=(self.u[ids,0]*py-self.u[ids,1]*px)/self.det[ids]
            inside=(a>=-1e-6)&(b>=-1e-6)&(a+b<=1+1e-6)
            if inside.any():
                z=self.origin[ids,2]+a*self.u[ids,2]+b*self.w[ids,2]
                candidates=np.flatnonzero(inside);j=int(candidates[np.argmin(abs(z[inside]-reference_z))])
                return SurfaceHit('UNKNOWN',reference_z,self.sources[int(ids[j])],
                    f'support_z_mismatch:reference={reference_z:.3f}:nearest={z[j]:.3f}:tolerance={self.z_tolerance:.3f}')
            return SurfaceHit('UNKNOWN',reference_z,reason='xy_outside_support_triangles')
        if center.kind == 'UNKNOWN':
            return center
        # Check the whole footprint against obstacles, not only sampled perimeter points.
        clearance = radius + self.query_padding
        candidates = set(self.global_obstacles)
        for key in self._keys((x-clearance,x+clearance,y-clearance,y+clearance)):
            candidates.update(self.obstacle_buckets.get(key, ()))
        for i in candidates:
            x0,x1,y0,y1,z0,z1 = self.obstacles[i]
            if (x0-clearance <= x <= x1+clearance and y0-clearance <= y <= y1+clearance
                    and z1 > center.z+self.body_min_z and z0 < center.z+self.body_max_z):
                return SurfaceHit('UNKNOWN',center.z,self.obstacle_sources[i] or center.source,f'footprint_obstacle_overlap:{i}',tuple(self.obstacles[i]))
        hits = [center]
        for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius),
                       (.707*radius, .707*radius), (-.707*radius, .707*radius),
                       (.707*radius, -.707*radius), (-.707*radius, -.707*radius)):
            hit = self.surface_at(x + dx, y + dy, center.z)
            if hit is None or hit.kind == 'UNKNOWN':
                return SurfaceHit('UNKNOWN', center.z,
                    hit.source if hit else center.source,
                    'footprint_edge:'+hit.reason if hit else 'footprint_edge_missing_support', hit.bounds if hit else ())
            hits.append(hit)
        return SurfaceHit('STAIRS' if any(h.kind == 'STAIRS' for h in hits) else 'FLAT', center.z, center.source)

    def trace(self, position, direction, length, radius, step):
        """Follow support Z continuously; returns first hazard and body-clearance distance."""
        x, y, z = position
        distances = np.r_[np.arange(0, length, step), length]
        for distance in distances:
            cx, cy = x + direction[0] * distance, y + direction[1] * distance
            hit = self.footprint(cx, cy, z, radius)
            if hit.kind != 'FLAT':
                lo, hi = max(0., float(distance)-step), float(distance)
                kind = hit.kind
                # Refine the boundary so a coarse query grid does not create cap jumps.
                for _ in range(6):
                    mid = (lo+hi)/2
                    probe = self.footprint(x+direction[0]*mid, y+direction[1]*mid, z, radius)
                    if probe.kind == 'FLAT':
                        lo = mid
                    else:
                        hi, kind = mid, probe.kind
                return kind, lo, hit.z
            z = hit.z
        return 'FLAT', math.inf, z

    def confirm_geometric_stairs(self):
        """Confirm three aligned, distinct horizontal tread levels; names alone never qualify."""
        if not len(self.vertices):
            return 0
        normals = np.cross(self.u, self.w)
        horizontal = (np.linalg.norm(normals[:, :2], axis=1) <= .02 * np.abs(normals[:, 2])) & self.infer_allowed
        area = .5*np.abs(self.det)
        longest = np.maximum.reduce([np.linalg.norm(self.u[:,:2],axis=1),
            np.linalg.norm(self.w[:,:2],axis=1),np.linalg.norm((self.u-self.w)[:,:2],axis=1)])
        horizontal &= (area >= .02) & (2*area/np.maximum(longest,1e-9) >= .1)
        centers = np.stack([(self.bounds[:, 0] + self.bounds[:, 1]) / 2,
                            (self.bounds[:, 2] + self.bounds[:, 3]) / 2,
                            self.vertices[:, :, 2].mean(1)], 1)
        confirmed = set()
        # Nearby centroid buckets, independent of prim naming or world position.
        bins = defaultdict(list)
        for i, point in enumerate(centers):
            if horizontal[i]:
                bins[(math.floor(point[0]), math.floor(point[1]))].append(i)
        for i, (x, y, z) in enumerate(centers):
            if not horizontal[i]:
                continue
            cell = math.floor(x), math.floor(y)
            ids = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                   for j in bins.get((cell[0]+dx, cell[1]+dy), ())]
            delta = centers[ids] - centers[i]
            distance = np.linalg.norm(delta[:, :2], axis=1)
            valid = (distance >= .12) & (distance <= .75) & (np.abs(delta[:, 2]) >= .04) & (np.abs(delta[:, 2]) <= .25)
            near = np.asarray(ids)[valid]
            lower = near[centers[near, 2] < z]
            upper = near[centers[near, 2] > z]
            for j in lower:
                for k in upper:
                    a, b = centers[i, :2] - centers[j, :2], centers[k, :2] - centers[i, :2]
                    if np.dot(a, b) > .9 * np.linalg.norm(a) * np.linalg.norm(b):
                        confirmed.update((i, int(j), int(k)))
        # Annotated structural obstacles never become treads by geometric inference.
        count = 0
        for i in confirmed:
            if self.kinds[i] in ('UNKNOWN', 'FLAT'):
                self.kinds[i] = 'STAIRS'
                count += 1
        return count

    def summary(self):
        return {'triangles': len(self.vertices), 'flat': int((self.kinds == 'FLAT').sum()),
                'stairs': int((self.kinds == 'STAIRS').sum()),
                'unknown': int((self.kinds == 'UNKNOWN').sum()), 'obstacles': len(self.obstacles)}
