"""Pure geometry predicates used by collision-aware spatial map extraction."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


Point3 = tuple[float, float, float]
Bounds3 = tuple[float, float, float, float, float, float]


def is_walkable_triangle(
    vertices: Sequence[Point3],
    *,
    min_up_normal: float = 0.55,
    min_projected_area_m2: float = 0.02,
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
    return bool(
        norm >= 1e-6
        and orientation_sign * nz / norm >= min_up_normal
        and projected_area >= min_projected_area_m2
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
