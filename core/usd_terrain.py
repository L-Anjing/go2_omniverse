"""Build and invalidate a terrain index from the existing composed USD Stage."""
from __future__ import annotations

import itertools
import math

import numpy as np
from pxr import Gf, Tf, Usd, UsdGeom, UsdPhysics

from core.semantic_labels import (canonical_semantic_class, infer_semantic_class,
                                  direct_railing_semantic_class, direct_door_semantic_class)
from core.spatial_geometry import is_walkable_triangle
from core.terrain_index import TerrainIndex


def semantic_class(prim):
    path = str(prim.GetPath())
    if direct_railing_semantic_class(path) or direct_door_semantic_class(path):
        return 'wall'
    current = prim
    while current and not current.IsPseudoRoot():
        for attr in current.GetAttributes():
            if 'semanticData' in attr.GetName() and attr.Get():
                return canonical_semantic_class(str(attr.Get()))
        current = current.GetParent()
    inferred = infer_semantic_class(path)
    return 'stairs_hint' if inferred == 'stairs' else inferred


def static_collision(prim):
    """Require real collision support; exclude robot/dynamic and disabled shapes."""
    collision = False
    current = prim
    while current and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(current)
            if api.GetRigidBodyEnabledAttr().Get() and not api.GetKinematicEnabledAttr().Get():
                return False
        if current.HasAPI(UsdPhysics.CollisionAPI):
            if UsdPhysics.CollisionAPI(current).GetCollisionEnabledAttr().Get() is False:
                return False
            collision = True
        current = current.GetParent()
    return collision


def build_terrain_index(stage, roots, config, semantic_resolver=None):
    resolver = semantic_resolver or semantic_class
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'proxy', 'render'])
    vertices, kinds, sources, infer_allowed, obstacles = [], [], [], [], []
    seen = set()
    solid_candidates = []
    obstacle_sources = []
    for root in roots:
        prim = stage.GetPrimAtPath(root)
        if not prim or not prim.IsValid():
            continue
        for leaf in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            path = str(leaf.GetPath())
            if path in seen or not leaf.IsActive() or '/Robot' in path or not static_collision(leaf):
                continue
            seen.add(path)
            transform = xforms.GetLocalToWorldTransform(leaf)
            semantic = resolver(leaf)
            points, faces = [], []
            plane_world = None
            orientation = 1.
            if leaf.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(leaf)
                points = mesh.GetPointsAttr().Get() or []
                counts = mesh.GetFaceVertexCountsAttr().Get() or []
                indices = mesh.GetFaceVertexIndicesAttr().Get() or []
                holes = set(mesh.GetHoleIndicesAttr().Get() or [])
                orientation = -1. if mesh.GetOrientationAttr().Get() == 'leftHanded' else 1.
                offset = 0
                for face_index, count in enumerate(counts):
                    if face_index not in holes:
                        faces.append(list(indices[offset:offset+count]))
                    offset += count
            elif leaf.IsA(UsdGeom.Cube):
                size = float(UsdGeom.Cube(leaf).GetSizeAttr().Get()) / 2
                points = [Gf.Vec3d(x*size, y*size, z*size)
                          for x, y, z in itertools.product((-1, 1), repeat=3)]
                # Outward-oriented faces, including vertical collision barriers.
                faces = [[1, 5, 7, 3], [0, 2, 6, 4], [0, 4, 5, 1],
                         [2, 3, 7, 6], [0, 1, 3, 2], [4, 6, 7, 5]]
            elif leaf.GetTypeName() == 'Plane':
                # USD physics planes are infinite; use a finite, explicit indexing horizon.
                # Nothing outside this horizon is certified as flat.
                axis = str(leaf.GetAttribute('axis').Get() or 'Z')
                e1,e2 = {'X':((0,1,0),(0,0,1)), 'Y':((0,0,1),(1,0,0)),
                         'Z':((1,0,0),(0,1,0))}[axis]
                origin = np.asarray(tuple(transform.Transform(Gf.Vec3d(0,0,0))))
                a = np.asarray(tuple(transform.Transform(Gf.Vec3d(*e1))))-origin
                b = np.asarray(tuple(transform.Transform(Gf.Vec3d(*e2))))-origin
                a = a/np.linalg.norm(a)*config.plane_index_half_extent
                b = b/np.linalg.norm(b)*config.plane_index_half_extent
                plane_world = np.asarray([origin-a-b,origin+a-b,origin+a+b,origin-a+b])
                points = [Gf.Vec3d(0,0,0)]*4
                faces = [[0,1,2,3]]
                semantic = 'floor'
            if not len(points):
                if leaf.IsA(UsdGeom.Gprim):
                    bounds = bounds_cache.ComputeWorldBound(leaf).ComputeAlignedRange()
                    lo, hi = bounds.GetMin(), bounds.GetMax()
                    values = (float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1]),float(lo[2]),float(hi[2]))
                    if not np.isfinite(values).all():
                        raise ValueError(f'Unknown collision shape without valid bounds: {path}')
                    obstacles.append(values)
                    obstacle_sources.append(path)
                continue
            world = plane_world if plane_world is not None else np.asarray([tuple(transform.Transform(Gf.Vec3d(*p))) for p in points])
            if semantic not in ('floor','stairs','ramp'):
                lo, hi = world.min(0), world.max(0)
                if hi[2]-lo[2] > .22:
                    solid_candidates.append(((lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]),path))
            determinant_sign = -1 if transform.GetDeterminant() < 0 else 1
            orientation *= determinant_sign
            for face in faces:
                if len(face) < 3:
                    continue
                for corner in range(1, len(face)-1):
                    tri = world[[face[0], face[corner], face[corner+1]]]
                    if not np.isfinite(tri).all():
                        continue
                    normal = np.cross(tri[1]-tri[0], tri[2]-tri[0]) * orientation
                    norm = np.linalg.norm(normal)
                    if norm < 1e-8:
                        continue
                    up = normal[2] / norm
                    if abs(up) < .5 or up < -.55:
                        lo, hi = tri.min(0), tri.max(0)
                        obstacles.append((lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]))
                        obstacle_sources.append(path)
                    if not is_walkable_triangle(tri, min_up_normal=.55,
                            min_projected_area_m2=1e-6, min_projected_width_m=.001,
                            orientation_sign=orientation):
                        continue
                    kind = 'STAIRS' if semantic == 'stairs' else ('FLAT' if semantic == 'floor' else 'UNKNOWN')
                    vertices.append(tri)
                    kinds.append(kind)
                    sources.append(path)
                    infer_allowed.append(semantic in ('floor', 'stairs', 'stairs_hint', 'misc', 'ramp'))
    index = TerrainIndex(vertices, kinds, sources, grid_size=config.grid_size,
                         z_tolerance=config.surface_z_tolerance, flat_max_slope=config.flat_max_slope,
                         obstacles=obstacles, infer_allowed=infer_allowed,query_padding=config.query_step/2,
                         body_min_z=config.body_min_z,body_max_z=config.body_max_z)
    inferred = index.confirm_geometric_stairs()
    index.obstacle_sources = tuple(obstacle_sources)
    terrain_by_source = {}
    for kind, path in zip(index.kinds,index.sources):
        terrain_by_source.setdefault(path,set()).add(str(kind))
    remaining = [(box,path) for box,path in solid_candidates if terrain_by_source.get(path) != {'STAIRS'}]
    index.add_obstacles([box for box,path in remaining],[path for box,path in remaining])
    print(f'[terrain-index] roots={roots} {index.summary()} geometry_confirmed={inferred}', flush=True)
    return index


class StageTerrainIndex:
    def __init__(self, stage, roots, config, semantic_resolver=None):
        self.stage, self.roots, self.config = stage, roots, config
        self.resolver = semantic_resolver
        self.dirty = True
        self.index = TerrainIndex()
        self.notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._changed, stage)
        self.refresh()

    def _changed(self, notice, sender):
        for path in list(notice.GetResyncedPaths()) + list(notice.GetChangedInfoOnlyPaths()):
            text = str(path)
            prim_path = str(path.GetPrimPath())
            if '/Robot' in prim_path:
                continue
            if not any(prim_path == r or prim_path.startswith(r + '/') or r.startswith(prim_path + '/') for r in self.roots):
                continue
            if not path.IsPropertyPath() or any(token in text for token in (
                'xformOp', '.points', '.faceVertex', '.holeIndices', '.orientation', '.semantic',
                '.physics:collisionEnabled', '.physics:rigidBodyEnabled', '.physics:kinematicEnabled',
                '.size', '.extent', '.axis', '.radius', '.height')):
                self.dirty = True
                break

    def refresh(self):
        if not self.dirty:
            return False
        try:
            self.index = build_terrain_index(self.stage, self.roots, self.config, self.resolver)
        except Exception as error:
            print(f'[terrain-index] failed; conservative UNKNOWN: {error}', flush=True)
            self.index = TerrainIndex()
        self.dirty = False
        return True

    def close(self):
        self.notice.Revoke()
