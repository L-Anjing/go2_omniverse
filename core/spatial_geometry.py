"""Pure geometry predicates used by collision-aware spatial map extraction."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence


Point3 = tuple[float, float, float]
Bounds3 = tuple[float, float, float, float, float, float]


def is_walkable_triangle(
    vertices: Sequence[Point3],
    *,
    min_up_normal: float = 0.55,
    min_projected_area_m2: float = 0.02,
    min_projected_width_m: float = 0.10,
    orientation_sign: float = 1.0,
) -> bool:
    """Reject downward/vertical faces and narrow rail-top triangles."""
    if len(vertices) != 3:
        return False
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = vertices
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    projected_area = 0.5 * abs(nz)
    projected_edges = (
        math.hypot(bx - ax, by - ay),
        math.hypot(cx - bx, cy - by),
        math.hypot(ax - cx, ay - cy),
    )
    longest_edge = max(projected_edges)
    projected_width = 2.0 * projected_area / longest_edge if longest_edge > 1e-6 else 0.0
    return bool(
        norm >= 1e-6
        and orientation_sign * nz / norm >= min_up_normal
        and projected_area >= min_projected_area_m2
        and projected_width >= min_projected_width_m
    )


def has_body_clearance_from_obstacles(
    point: Point3,
    obstacles: Iterable[Bounds3],
    *,
    horizontal_clearance_m: float = 0.32,
    body_height_m: float = 0.70,
) -> bool:
    """Conservatively erode walkable surfaces around rails, walls, and props."""
    x, y, floor_z = point
    body_min_z = floor_z + 0.05
    body_max_z = floor_z + body_height_m
    for x_min, x_max, y_min, y_max, z_min, z_max in obstacles:
        if (
            x_min - horizontal_clearance_m <= x <= x_max + horizontal_clearance_m
            and y_min - horizontal_clearance_m <= y <= y_max + horizontal_clearance_m
            and body_min_z <= z_max
            and body_max_z >= z_min
        ):
            return False
    return True


def find_traversal_entry_spawn_points(
    spawn_points: Sequence[Point3],
    traversal_points: Sequence[Point3],
    *,
    neighbor_radius_m: float = 0.45,
    max_step_height_m: float = 0.30,
    min_component_elevation_m: float = 0.80,
    low_band_m: float = 0.25,
    min_entry_xy_distance_m: float = 0.0,
    max_entry_xy_distance_m: float = 1.50,
    max_entry_z_distance_m: float = 0.35,
    min_component_points: int = 5,
) -> list[Point3]:
    """Find stable floor spawns near the low end of each 3D route component."""
    if not spawn_points or not traversal_points:
        return []
    radius = max(float(neighbor_radius_m), 0.05)
    points = [tuple(float(value) for value in point) for point in traversal_points]
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (x, y, _) in enumerate(points):
        key = (math.floor(x / radius), math.floor(y / radius))
        buckets.setdefault(key, []).append(index)

    visited = [False] * len(points)
    low_ends: list[Point3] = []
    for start in range(len(points)):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            x, y, z = points[current]
            cell = (math.floor(x / radius), math.floor(y / radius))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for candidate in buckets.get((cell[0] + dx, cell[1] + dy), ()):
                        if visited[candidate]:
                            continue
                        cx, cy, cz = points[candidate]
                        horizontal = math.hypot(cx - x, cy - y)
                        if (
                            0.02 < horizontal <= radius
                            and abs(cz - z) <= max_step_height_m
                        ):
                            visited[candidate] = True
                            queue.append(candidate)

        elevations = [points[index][2] for index in component]
        if (
            len(component) < min_component_points
            or max(elevations) - min(elevations) < min_component_elevation_m
        ):
            continue
        low_limit = min(elevations) + low_band_m
        low_ends.extend(
            points[index] for index in component if points[index][2] <= low_limit
        )

    result: list[Point3] = []
    for raw_spawn in spawn_points:
        spawn = tuple(float(value) for value in raw_spawn)
        compatible_distances = [
            math.hypot(spawn[0] - entry[0], spawn[1] - entry[1])
            for entry in low_ends
            if abs(spawn[2] - entry[2]) <= max_entry_z_distance_m
        ]
        nearest = min(compatible_distances, default=math.inf)
        if min_entry_xy_distance_m <= nearest <= max_entry_xy_distance_m:
            result.append(spawn)
    return result
