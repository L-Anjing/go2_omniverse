# Copyright (c) 2024, RoboVerse community
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""Script to deploy an RSL-RL checkpoint in Isaac Sim."""

from __future__ import annotations


"""Launch Isaac Sim Simulator first."""
import argparse
import copy
import json
import re
from pathlib import Path
from isaaclab.app import AppLauncher


from core import cli_args
from core.semantic_labels import (
    canonical_semantic_class,
    infer_semantic_class,
    is_traversal_surface_class,
)
from core.spatial_geometry import (
    find_traversal_entry_spawn_points,
    has_body_clearance_from_obstacles,
    is_walkable_triangle,
)
import time
import os
import threading


# add argparse arguments
parser = argparse.ArgumentParser(description="Deploy an RL agent with RSL-RL.")
# parser.add_argument("--device", type=str, default="cpu", help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Unitree-Go2-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--custom_env", type=str, default="", help="Setup the environment"
)
parser.add_argument("--robot", type=str, default="go2", help="Setup the robot")
parser.add_argument(
    "--robot_amount", type=int, default=1, help="Setup the robot amount"
)
parser.add_argument(
    "--cmd_source",
    type=str,
    choices=("keyboard", "ros2"),
    default="keyboard",
    help=(
        "Select who writes high-level base commands: "
        "'keyboard' for manual teleop, 'ros2' for /robot{N}/cmd_vel subscribers."
    ),
)
parser.add_argument(
    "--viewer_follow",
    type=str,
    choices=("auto", "on", "off"),
    default="auto",
    help=(
        "Viewer tracking mode for the main camera. "
        "'auto' keeps the previous behavior (follow only for single-robot play), "
        "'on' always follows the robot root, and 'off' leaves the viewer free."
    ),
)
parser.add_argument(
    "--deploy_architecture",
    type=str,
    choices=("cts", "moe_cts"),
    default="cts",
    help=(
        "Preferred Go2 deployment family. "
        "'cts' defaults to the standard CTS experiment folder; "
        "'moe_cts' defaults to the reference-style MoE-CTS folder."
    ),
)
parser.add_argument(
    "--camera_mode",
    type=str,
    choices=("single", "double"),
    default="single",
    help=(
        "Camera configuration: 'single' uses one front_cam, "
        "'double' uses a stereo pair (front_cam_left / front_cam_right)."
    ),
)
parser.add_argument(
    "--enable_depth",
    action="store_true",
    default=False,
    help="Publish depth images alongside RGB (default: False).",
)


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import omni


ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

# FOR VR SUPPORT
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.core", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.steamvr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.simulatedxr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.openxr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.telemetry", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.profile.vr", True)


"""Rest everything follows."""
import gymnasium as gym
import torch
import carb


from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import omni.appwindow
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from rsl_rl.runners import OnPolicyRunner


import rclpy
from core.ros2 import (
    RobotBaseNode,
    add_camera,
    add_copter_camera,
    add_rtx_lidar,
    pub_robo_data_ros2
)
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger


from configs.agent_cfg import unitree_go2_agent_cfg, unitree_g1_agent_cfg
from configs.custom_rl_env import (
    UnitreeGo2CustomEnvCfg,
    Go2StairDeployCfg,
    Go2FullSceneDeployCfg,
    G1RoughEnvCfg,
    GO2_MODEL_JOINT_NAMES,
)
import configs.custom_rl_env as custom_rl_env

from assets.robots.copter.config import CRAZYFLIE_CFG
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from omni.physx import get_physx_scene_query_interface
from pxr import Semantics, Usd, UsdGeom


from core.omnigraph import create_front_cam_omnigraph


REPO_ROOT = Path(__file__).resolve().parent.parent
_CUSTOM_SCENE_BOUNDS: tuple[float, float, float, float] | None = None
_CUSTOM_FLOOR_BOUNDS: list[tuple[float, float, float, float, float]] = []
_CUSTOM_OBSTACLE_BOUNDS: list[
    tuple[float, float, float, float, float, float]
] = []
_CUSTOM_TRAVERSAL_TRIANGLES: list[
    tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
] = []
_VALID_SPAWN_POINTS: list[tuple[float, float, float]] | None = None
_VALID_TRAVERSAL_POINTS: list[tuple[float, float, float]] = []
_INITIAL_SPAWN_POINTS: list[tuple[float, float, float]] = []
_INITIAL_SPAWN_HEADINGS: list[float] = []
_SPATIAL_MAP_METADATA: dict[str, object] = {}
_CUSTOM_SCENE_PRIM_PATH = ""
_EPISODE_RESET_REQUESTED = threading.Event()
_EPISODE_RESET_COMPLETED = threading.Event()
_EPISODE_RESET_LOCK = threading.Lock()
_EPISODE_RESET_ERROR = ""


def _direct_semantic_class(prim) -> str:
    for attribute in prim.GetAttributes():
        if "semanticData" not in attribute.GetName():
            continue
        value = attribute.Get()
        if value:
            return canonical_semantic_class(str(value))
    return "misc"


def _spatial_semantic_class(prim, profile_hint: str = "") -> str:
    """Resolve geometry leaf-first, then fall back to authored ancestor labels."""
    inferred = infer_semantic_class(str(prim.GetPath()), profile_hint)
    if inferred != "misc":
        return inferred
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        authored = _direct_semantic_class(current)
        if authored != "misc":
            return authored
        current = current.GetParent()
    return "misc"


def _spatial_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _record_custom_scene_bounds(stage, prim_path: str) -> None:
    """Cache the loaded scene bounds and its walkable floor mesh bounds."""
    global _CUSTOM_SCENE_BOUNDS, _CUSTOM_FLOOR_BOUNDS, _CUSTOM_OBSTACLE_BOUNDS
    global _CUSTOM_TRAVERSAL_TRIANGLES
    global _CUSTOM_SCENE_PRIM_PATH, _VALID_SPAWN_POINTS
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[WARN] Cannot compute spatial bounds: invalid prim {prim_path}")
        return
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    minimum, maximum = aligned.GetMin(), aligned.GetMax()
    scene_bounds = tuple(
        float(value) for value in (minimum[0], maximum[0], minimum[1], maximum[1])
    )
    floor_bounds: list[tuple[float, float, float, float, float]] = []
    obstacle_bounds: list[tuple[float, float, float, float, float, float]] = []
    traversal_triangles: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = []
    ignored_names = ("collisionplane", "worldgrid", "groundplane")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for descendant in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not descendant.IsActive() or not descendant.IsA(UsdGeom.Mesh):
            continue
        path = str(descendant.GetPath())
        lowered = path.lower()
        mesh_range = cache.ComputeWorldBound(descendant).ComputeAlignedRange()
        mesh_min, mesh_max = mesh_range.GetMin(), mesh_range.GetMax()
        semantic_class = _spatial_semantic_class(
            descendant, os.environ.get("DATA_COLLECTION_PROFILE", args_cli.custom_env)
        )
        if semantic_class not in {"floor", "ramp", "stairs", "pit"}:
            obstacle = tuple(
                float(value)
                for value in (
                    mesh_min[0],
                    mesh_max[0],
                    mesh_min[1],
                    mesh_max[1],
                    mesh_min[2],
                    mesh_max[2],
                )
            )
            if (
                all(torch.isfinite(torch.tensor(obstacle)).tolist())
                and obstacle[1] - obstacle[0] > 0.03
                and obstacle[3] - obstacle[2] > 0.03
                and obstacle[5] - obstacle[4] > 0.03
            ):
                obstacle_bounds.append(obstacle)
        if semantic_class != "floor":
            continue
        if any(name in lowered for name in ignored_names):
            continue
        values = tuple(
            float(value)
            for value in (
                mesh_min[0],
                mesh_max[0],
                mesh_min[1],
                mesh_max[1],
                mesh_max[2],
            )
        )
        if not all(torch.isfinite(torch.tensor(values)).tolist()):
            continue
        if values[1] - values[0] < 0.25 or values[3] - values[2] < 0.25:
            continue
        floor_bounds.append(values)

    for descendant in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not descendant.IsActive() or not descendant.IsA(UsdGeom.Mesh):
            continue
        semantic_class = _spatial_semantic_class(
            descendant,
            os.environ.get("DATA_COLLECTION_PROFILE", args_cli.custom_env),
        )
        if not is_traversal_surface_class(semantic_class):
            continue
        mesh = UsdGeom.Mesh(descendant)
        orientation_sign = (
            -1.0
            if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
            else 1.0
        )
        local_points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if not local_points or not counts or not indices:
            continue
        transform = xform_cache.GetLocalToWorldTransform(descendant)
        world_points = [transform.Transform(point) for point in local_points]
        offset = 0
        for count in counts:
            count = int(count)
            face = [world_points[int(index)] for index in indices[offset : offset + count]]
            offset += count
            if len(face) < 3:
                continue
            for corner in range(1, len(face) - 1):
                vertices = (face[0], face[corner], face[corner + 1])
                ax, ay, az = (float(value) for value in vertices[0])
                bx, by, bz = (float(value) for value in vertices[1])
                cx, cy, cz = (float(value) for value in vertices[2])
                triangle = ((ax, ay, az), (bx, by, bz), (cx, cy, cz))
                if not is_walkable_triangle(
                    triangle, orientation_sign=orientation_sign
                ):
                    continue
                traversal_triangles.append(triangle)

    if floor_bounds:
        bounds = (
            min(item[0] for item in floor_bounds),
            max(item[1] for item in floor_bounds),
            min(item[2] for item in floor_bounds),
            max(item[3] for item in floor_bounds),
        )
    else:
        bounds = scene_bounds
    if not all(torch.isfinite(torch.tensor(bounds)).tolist()):
        print(f"[WARN] Ignoring non-finite scene bounds for {prim_path}: {bounds}")
        return
    _CUSTOM_SCENE_BOUNDS = bounds
    _CUSTOM_FLOOR_BOUNDS = floor_bounds
    _CUSTOM_OBSTACLE_BOUNDS = obstacle_bounds
    _CUSTOM_TRAVERSAL_TRIANGLES = traversal_triangles
    _CUSTOM_SCENE_PRIM_PATH = prim_path.rstrip("/")
    _VALID_SPAWN_POINTS = None
    print(
        f"[INFO] Dynamic scene XY bounds {prim_path}: {bounds} "
        f"({len(floor_bounds)} floor meshes, "
        f"{len(obstacle_bounds)} obstacle meshes, "
        f"{len(traversal_triangles)} traversal triangles)"
    )
    for floor in floor_bounds[:12]:
        print(f"[INFO] Walkable floor bounds: {floor}")
    if not floor_bounds:
        print("[WARN] No named floor meshes found; falling back to scene bounds and raycasts")


def _scene_floor_height(x: float, y: float) -> float | None:
    """Return a horizontal floor hit belonging to the custom scene."""
    if not _CUSTOM_SCENE_PRIM_PATH:
        return None
    z_min = _spatial_float_env("SPATIAL_FLOOR_Z_MIN", -0.25)
    z_max = _spatial_float_env("SPATIAL_FLOOR_Z_MAX", 0.50)
    hits: list[float] = []

    def report(hit) -> bool:
        collision = str(getattr(hit, "collision", ""))
        position = getattr(hit, "position", None)
        normal = getattr(hit, "normal", None)
        if (
            collision.startswith(_CUSTOM_SCENE_PRIM_PATH + "/")
            and position is not None
            and normal is not None
            and float(normal.z) > 0.65
            and z_min <= float(position.z) <= z_max
        ):
            hits.append(float(position.z))
        return True

    query = get_physx_scene_query_interface()
    origin_z = z_max + 3.0
    query.raycast_all(
        carb.Float3(x, y, origin_z),
        carb.Float3(0.0, 0.0, -1.0),
        origin_z - z_min + 1.0,
        report,
    )
    if hits:
        return max(hits)
    containing = [
        bounds
        for bounds in _CUSTOM_FLOOR_BOUNDS
        if bounds[0] <= x <= bounds[1]
        and bounds[2] <= y <= bounds[3]
        and z_min <= bounds[4] <= z_max
    ]
    if not containing:
        return None

    # Some Isaac environment USDs contain several vertically stacked display
    # sets. Select the uppermost full-coverage floor, not narrow wall tops and
    # not an inactive lower set. This also handles a world-space USD offset
    # when the configured floor range is expressed in asset-local coordinates.
    areas = [
        max(bounds[1] - bounds[0], 0.0) * max(bounds[3] - bounds[2], 0.0)
        for bounds in _CUSTOM_FLOOR_BOUNDS
    ]
    dominant_area = max(areas, default=0.0)
    dominant_heights = [
        bounds[4]
        for bounds, area in zip(_CUSTOM_FLOOR_BOUNDS, areas)
        if dominant_area > 0.0 and area >= 0.80 * dominant_area
    ]
    reference_z = max(dominant_heights) if dominant_heights else min(item[4] for item in containing)
    return min(containing, key=lambda item: abs(item[4] - reference_z))[4]


def _spawn_clearance_is_free(x: float, y: float, floor_z: float) -> bool:
    """Check a Go2-sized box above the floor for scene collisions."""
    blocked = False

    def report(hit) -> bool:
        nonlocal blocked
        path = str(getattr(hit, "rigid_body", "") or getattr(hit, "collision", ""))
        if path.startswith(_CUSTOM_SCENE_PRIM_PATH + "/"):
            blocked = True
            return False
        return True

    query = get_physx_scene_query_interface()
    half_x = _spatial_float_env("SPATIAL_CLEARANCE_X", 0.42)
    half_y = _spatial_float_env("SPATIAL_CLEARANCE_Y", 0.28)
    half_z = _spatial_float_env("SPATIAL_CLEARANCE_Z", 0.28)
    body_min_z = floor_z + 0.08
    body_max_z = floor_z + 2.0 * half_z + 0.08
    if any(
        x + half_x >= bound[0]
        and x - half_x <= bound[1]
        and y + half_y >= bound[2]
        and y - half_y <= bound[3]
        and body_max_z >= bound[4]
        and body_min_z <= bound[5]
        for bound in _CUSTOM_OBSTACLE_BOUNDS
    ):
        return False
    query.overlap_box(
        carb.Float3(half_x, half_y, half_z),
        carb.Float3(x, y, floor_z + half_z + 0.08),
        carb.Float4(0.0, 0.0, 0.0, 1.0),
        report,
        False,
    )
    return not blocked


def _write_spatial_map(
    points: list[tuple[float, float, float]],
    traversal_points: list[tuple[float, float, float]],
    grid_resolution: float,
) -> None:
    path_value = os.environ.get("SPATIAL_BOUNDS_FILE", "").strip()
    if not path_value or _CUSTOM_SCENE_BOUNDS is None:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "spatial_semantics_version": 3,
        "profile": os.environ.get("DATA_COLLECTION_PROFILE", args_cli.custom_env),
        "scene_prim_path": _CUSTOM_SCENE_PRIM_PATH,
        "xy_bounds": list(_CUSTOM_SCENE_BOUNDS),
        "valid_points": [list(point) for point in points],
        "traversal_points": [list(point) for point in traversal_points],
        "grid_resolution": grid_resolution,
        "traversal_grid_resolution": _spatial_float_env(
            "SPATIAL_TRAVERSAL_GRID_RESOLUTION", 0.30
        ),
        "elevation_range": (
            [
                min(point[2] for point in points + traversal_points),
                max(point[2] for point in points + traversal_points),
            ]
            if points or traversal_points
            else []
        ),
        **_SPATIAL_MAP_METADATA,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    elevation = document["elevation_range"]
    print(
        f"[INFO] Spatial map written: {path} ({len(points)} spawn points, "
        f"{len(traversal_points)} traversal points, "
        f"elevation={elevation})"
    )


def _triangle_height_at_xy(
    triangle: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    x: float,
    y: float,
) -> float | None:
    a, b, c = triangle
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denominator) < 1e-8:
        return None
    wa = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
    wb = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
    wc = 1.0 - wa - wb
    if min(wa, wb, wc) < -1e-5:
        return None
    return wa * a[2] + wb * b[2] + wc * c[2]


def _build_traversal_points() -> list[tuple[float, float, float]]:
    if not _CUSTOM_TRAVERSAL_TRIANGLES:
        return []
    resolution = max(_spatial_float_env("SPATIAL_TRAVERSAL_GRID_RESOLUTION", 0.30), 0.15)
    z_min = _spatial_float_env("SPATIAL_FLOOR_Z_MIN", -1.0)
    z_max = _spatial_float_env("SPATIAL_FLOOR_Z_MAX", 4.5)
    traversal_clearance = _spatial_float_env(
        "SPATIAL_TRAVERSAL_CLEARANCE", 0.32
    )
    points: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for triangle in _CUSTOM_TRAVERSAL_TRIANGLES:
        x_min = min(vertex[0] for vertex in triangle)
        x_max = max(vertex[0] for vertex in triangle)
        y_min = min(vertex[1] for vertex in triangle)
        y_max = max(vertex[1] for vertex in triangle)
        x = x_min
        while x <= x_max + 1e-6:
            y = y_min
            while y <= y_max + 1e-6:
                z = _triangle_height_at_xy(triangle, x, y)
                point = (x, y, z) if z is not None else None
                if (
                    point is not None
                    and z_min <= z <= z_max
                    and has_body_clearance_from_obstacles(
                        point,
                        _CUSTOM_OBSTACLE_BOUNDS,
                        horizontal_clearance_m=traversal_clearance,
                    )
                ):
                    key = (round(x / resolution), round(y / resolution), round(z / 0.05))
                    points[key] = point
                y += resolution
            x += resolution
        centroid = tuple(sum(vertex[axis] for vertex in triangle) / 3.0 for axis in range(3))
        if (
            z_min <= centroid[2] <= z_max
            and has_body_clearance_from_obstacles(
                centroid,
                _CUSTOM_OBSTACLE_BOUNDS,
                horizontal_clearance_m=traversal_clearance,
            )
        ):
            key = (
                round(centroid[0] / resolution),
                round(centroid[1] / resolution),
                round(centroid[2] / 0.05),
            )
            points[key] = centroid
    return list(points.values())


def _build_valid_spawn_points() -> list[tuple[float, float, float]]:
    global _VALID_SPAWN_POINTS, _VALID_TRAVERSAL_POINTS
    global _CUSTOM_SCENE_BOUNDS, _SPATIAL_MAP_METADATA
    if _VALID_SPAWN_POINTS is not None:
        return _VALID_SPAWN_POINTS
    if _CUSTOM_SCENE_BOUNDS is None:
        return []
    x_min, x_max, y_min, y_max = _CUSTOM_SCENE_BOUNDS
    margin = _spatial_float_env("SPATIAL_BOUNDS_MARGIN", 0.75)
    resolution = max(_spatial_float_env("SPATIAL_GRID_RESOLUTION", 0.75), 0.25)
    x_min, x_max = x_min + margin, x_max - margin
    y_min, y_max = y_min + margin, y_max - margin
    max_candidates = 2500
    nx = max(int((x_max - x_min) / resolution) + 1, 1)
    ny = max(int((y_max - y_min) / resolution) + 1, 1)
    if nx * ny > max_candidates:
        resolution *= (nx * ny / max_candidates) ** 0.5
    points: list[tuple[float, float, float]] = []
    sampling_mode = "collision_checked"
    candidate_count = 0
    floor_supported_count = 0
    clearance_rejected_count = 0
    x = x_min
    while x <= x_max + 1e-6:
        y = y_min
        while y <= y_max + 1e-6:
            candidate_count += 1
            floor_z = _scene_floor_height(x, y)
            if floor_z is not None:
                floor_supported_count += 1
                if _spawn_clearance_is_free(x, y, floor_z):
                    points.append((x, y, floor_z))
                else:
                    clearance_rejected_count += 1
            y += resolution
        x += resolution
    traversal_points = _build_traversal_points()
    fallback_z_raw = os.environ.get("SPATIAL_FALLBACK_FLOOR_Z", "").strip()
    if len(points) < 16 and fallback_z_raw:
        if points:
            print(
                f"[WARN] Only {len(points)} dynamically extracted spawn points; "
                "replacing them with the configured fallback grid"
            )
        points = []
        fallback_z = float(fallback_z_raw)
        fallback_x = _parse_range_env("RANDOM_START_X_RANGE")
        fallback_y = _parse_range_env("RANDOM_START_Y_RANGE")
        fallback_mode = os.environ.get(
            "SPATIAL_FALLBACK_MODE", "collision_checked"
        ).strip().lower()
        sampling_mode = f"fallback_{fallback_mode}"
        fallback_resolution = max(
            _spatial_float_env("SPATIAL_GRID_RESOLUTION", 0.75), 0.25
        )
        if fallback_x and fallback_y:
            x = fallback_x[0] + margin
            while x <= fallback_x[1] - margin + 1e-6:
                y = fallback_y[0] + margin
                while y <= fallback_y[1] - margin + 1e-6:
                    if (
                        fallback_mode == "static_range"
                        or _spawn_clearance_is_free(x, y, fallback_z)
                    ):
                        points.append((x, y, fallback_z))
                    y += fallback_resolution
                x += fallback_resolution
            resolution = fallback_resolution
            if points:
                _CUSTOM_SCENE_BOUNDS = (
                    fallback_x[0], fallback_x[1], fallback_y[0], fallback_y[1]
                )
                print(
                    f"[INFO] Static-floor fallback ({fallback_mode}): "
                    f"z={fallback_z:.2f}, points={len(points)}"
                )
    if not points and traversal_points:
        minimum_z = min(point[2] for point in traversal_points)
        points = [point for point in traversal_points if point[2] <= minimum_z + 0.08]
        print(
            f"[INFO] Using {len(points)} lowest traversal-surface points "
            "as spawn candidates"
        )
    pre_component_count = len(points)
    component_count = 0
    if points:
        keyed = {
            (round(point[0] / resolution), round(point[1] / resolution)): point
            for point in points
        }
        remaining = set(keyed)
        components: list[list[tuple[float, float, float]]] = []
        while remaining:
            start = remaining.pop()
            queue = [start]
            component = [keyed[start]]
            while queue:
                cell = queue.pop()
                current = keyed[cell]
                for neighbor in (
                    (cell[0] - 1, cell[1]),
                    (cell[0] + 1, cell[1]),
                    (cell[0], cell[1] - 1),
                    (cell[0], cell[1] + 1),
                ):
                    if (
                        neighbor in remaining
                        and abs(keyed[neighbor][2] - current[2]) <= 0.30
                    ):
                        remaining.remove(neighbor)
                        queue.append(neighbor)
                        component.append(keyed[neighbor])
            components.append(component)
        component_count = len(components)
        points = max(components, key=len)
        if len(points) < pre_component_count:
            print(
                f"[INFO] Spatial connectivity filter: kept {len(points)}/"
                f"{pre_component_count} points from {component_count} components"
            )
    _VALID_SPAWN_POINTS = points
    _VALID_TRAVERSAL_POINTS = traversal_points
    _SPATIAL_MAP_METADATA.update(
        {
            "sampling_mode": sampling_mode,
            "candidate_count": candidate_count,
            "floor_supported_count": floor_supported_count,
            "clearance_rejected_count": clearance_rejected_count,
            "valid_candidate_fraction": (
                len(points) / candidate_count if candidate_count else 0.0
            ),
            "pre_component_valid_count": pre_component_count,
            "component_count": component_count,
            "dropped_disconnected_count": pre_component_count - len(points),
        }
    )
    _write_spatial_map(points, traversal_points, resolution)
    print(f"[INFO] Obstacle-aware spatial sampling: {len(points)} valid points")
    return points


def _traversal_entry_spawn_points(
    spawn_points: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    traversal_resolution = _spatial_float_env(
        "SPATIAL_TRAVERSAL_GRID_RESOLUTION", 0.25
    )
    result = find_traversal_entry_spawn_points(
        spawn_points,
        _VALID_TRAVERSAL_POINTS,
        neighbor_radius_m=max(0.35, 1.8 * traversal_resolution),
        min_component_elevation_m=_spatial_float_env(
            "SPATIAL_TRAVERSAL_ENTRY_MIN_ELEVATION", 0.80
        ),
        min_entry_xy_distance_m=_spatial_float_env(
            "SPATIAL_TRAVERSAL_ENTRY_MIN_DISTANCE", 0.0
        ),
    )
    maximum_fraction = _spatial_float_env(
        "SPATIAL_MAX_TRAVERSAL_ENTRY_FRACTION", 1.0
    )
    fraction = len(result) / max(len(spawn_points), 1)
    if fraction > maximum_fraction:
        raise RuntimeError(
            "Traversal-entry extraction is over-broad: "
            f"{len(result)}/{len(spawn_points)} points ({fraction:.1%}) exceed "
            f"the configured {maximum_fraction:.1%}. Check traversal semantics."
        )
    if _SPATIAL_MAP_METADATA.get("traversal_entry_count") != len(result):
        _SPATIAL_MAP_METADATA.update(
            {
                "traversal_entry_count": len(result),
                "traversal_entry_fraction": fraction,
                "spawn_points": [list(point) for point in result],
            }
        )
        _write_spatial_map(
            list(_VALID_SPAWN_POINTS or spawn_points),
            list(_VALID_TRAVERSAL_POINTS),
            max(_spatial_float_env("SPATIAL_GRID_RESOLUTION", 0.75), 0.25),
        )
    print(f"[INFO] Traversal-entry spawn candidates: {len(result)}")
    return result


def _reset_root_state_spatially_valid(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot roots on scene floor points with body clearance."""
    points = _build_valid_spawn_points()
    if not points:
        raise RuntimeError(
            "Dynamic spatial sampling is enabled but no collision-free spawn points "
            "were found. Fix the profile floor range or configure fallback_floor_z."
        )
    spawn_headings: list[float] | None = None
    fixed_spawn_json = os.environ.get("SPATIAL_FIXED_SPAWN_POINTS", "").strip()
    fixed_heading_json = os.environ.get("SPATIAL_FIXED_SPAWN_HEADINGS", "").strip()
    if fixed_spawn_json:
        fixed_points = [
            tuple(float(value) for value in point)
            for point in json.loads(fixed_spawn_json)
        ]
        fixed_headings = [float(value) for value in json.loads(fixed_heading_json)]
        if not fixed_points or len(fixed_points) != len(fixed_headings):
            raise RuntimeError("Fixed spatial spawns and headings must be non-empty and aligned")
        invalid = [
            point for point in fixed_points
            if not _spawn_clearance_is_free(point[0], point[1], point[2])
        ]
        if invalid:
            raise RuntimeError(f"Fixed spatial spawn points lack body clearance: {invalid}")
        points = fixed_points
        spawn_headings = fixed_headings
        _SPATIAL_MAP_METADATA.update(
            {
                "spawn_mode": "fixed_verified",
                "spawn_points": [list(point) for point in points],
                "spawn_headings": spawn_headings,
                "traversal_entry_count": len(points),
                "traversal_entry_fraction": len(points) / max(len(_VALID_SPAWN_POINTS or []), 1),
            }
        )
        _write_spatial_map(
            list(_VALID_SPAWN_POINTS or points),
            list(_VALID_TRAVERSAL_POINTS),
            max(_spatial_float_env("SPATIAL_GRID_RESOLUTION", 0.75), 0.25),
        )
        print(f"[INFO] Using {len(points)} fixed traversal-entry spawn candidates")
    elif _INITIAL_SPAWN_POINTS:
        points = _INITIAL_SPAWN_POINTS
        if len(_INITIAL_SPAWN_HEADINGS) == len(points):
            spawn_headings = _INITIAL_SPAWN_HEADINGS
        print(f"[INFO] Using {len(points)} structure-aware initial spawn candidates")
    elif os.environ.get("SPATIAL_SPAWN_MODE", "").lower() == "traversal_entry":
        traversal_entries = _traversal_entry_spawn_points(points)
        if traversal_entries:
            points = traversal_entries
        else:
            print("[WARN] No traversal entry points found; using all spawn points")
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    point_tensor = torch.tensor(points, device=asset.device, dtype=root_states.dtype)
    point_ids = torch.randint(0, len(points), (len(env_ids),), device=asset.device)
    selected = point_tensor[point_ids]
    positions = root_states[:, 0:3]
    positions[:, 0:2] = selected[:, 0:2]
    positions[:, 2] = selected[:, 2] + root_states[:, 2]

    if spawn_headings is not None:
        heading_tensor = torch.tensor(
            spawn_headings, device=asset.device, dtype=root_states.dtype
        )
        yaw = heading_tensor[point_ids]
        jitter = _spatial_float_env(
            "SPATIAL_FIXED_SPAWN_YAW_JITTER",
            _spatial_float_env("PROCEDURAL_SPAWN_YAW_JITTER", 0.30),
        )
        yaw += torch.empty(len(env_ids), device=asset.device).uniform_(-jitter, jitter)
    else:
        yaw_range = pose_range.get("yaw", (-3.14, 3.14))
        yaw = torch.empty(len(env_ids), device=asset.device).uniform_(*yaw_range)
    zeros = torch.zeros_like(yaw)
    orientation_delta = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

    velocity_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in velocity_keys],
        device=asset.device,
        dtype=root_states.dtype,
    )
    velocities = root_states[:, 7:13] + math_utils.sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def _default_go2_deploy_experiment_name() -> str:
    if args_cli.deploy_architecture == "moe_cts":
        return "unitree_go2_fullscene_moe_cts"
    return "unitree_go2_fullscene_cts"


def resolve_agent_cfg(base_cfg: RslRlOnPolicyRunnerCfg) -> RslRlOnPolicyRunnerCfg:
    """Return a copy of the base agent config with CLI overrides applied."""
    agent_cfg = copy.deepcopy(base_cfg)

    if args_cli.robot == "go2":
        agent_cfg["experiment_name"] = _default_go2_deploy_experiment_name()
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    if args_cli.experiment_name is not None:
        agent_cfg["experiment_name"] = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg["run_name"] = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg["logger"] = args_cli.logger
    if args_cli.log_project_name is not None:
        agent_cfg["wandb_project"] = args_cli.log_project_name
        agent_cfg["neptune_project"] = args_cli.log_project_name
    if args_cli.resume is not None:
        agent_cfg["resume"] = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg["load_run"] = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg["load_checkpoint"] = args_cli.checkpoint

    return agent_cfg


def configure_policy_from_checkpoint(
    agent_cfg: RslRlOnPolicyRunnerCfg, checkpoint_path: str
) -> None:
    """Align policy config with the saved checkpoint parameterization."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model_state_dict", {})
    policy_cfg = dict(agent_cfg["policy"])

    if "log_std" in model_state:
        policy_cfg["noise_std_type"] = "log"
    elif "std" in model_state:
        policy_cfg["noise_std_type"] = "scalar"

    agent_cfg["policy"] = policy_cfg


def _is_cts_checkpoint(path: str) -> bool:
    """Return True if the checkpoint was produced by CTSRunner."""
    ckpt = torch.load(path, map_location="cpu")
    return any(k.startswith("teacher_encoder") for k in ckpt.get("model_state_dict", {}))


def _infer_hidden_dims_from_mlp(state_dict: dict[str, torch.Tensor], prefix: str) -> list[int]:
    weights = sorted(k for k in state_dict if k.startswith(prefix) and k.endswith(".weight"))
    if len(weights) < 2:
        return []
    return [state_dict[key].shape[0] for key in weights[:-1]]


def _infer_cts_checkpoint_dims(checkpoint_path: str, map_location: str = "cpu") -> dict[str, int]:
    """Infer CTS model dimensions directly from checkpoint tensor shapes."""
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state_dict = ckpt["model_state_dict"]

    enc_w = sorted(k for k in state_dict if k.startswith("teacher_encoder.") and k.endswith(".weight"))
    latent_dim = state_dict[enc_w[-1]].shape[0]

    act_w = sorted(k for k in state_dict if k.startswith("actor.") and k.endswith(".weight"))
    num_actor_obs = state_dict[act_w[0]].shape[1] - latent_dim
    num_actions = state_dict[act_w[-1]].shape[0]

    crit_w = sorted(k for k in state_dict if k.startswith("critic.") and k.endswith(".weight"))
    num_critic_obs = state_dict[crit_w[0]].shape[1] - latent_dim

    actor_hidden_dims = _infer_hidden_dims_from_mlp(state_dict, "actor.")
    critic_hidden_dims = _infer_hidden_dims_from_mlp(state_dict, "critic.")

    if any(k.startswith("student_moe_encoder.") for k in state_dict):
        checkpoint_kind = "moe_cts"
        teacher_hidden_dims = _infer_hidden_dims_from_mlp(state_dict, "teacher_encoder.0.")
        gating_w = sorted(
            k
            for k in state_dict
            if k.startswith("student_moe_encoder.moe.gating_network.0.network.")
            and k.endswith(".weight")
        )
        backbone_w = sorted(
            k
            for k in state_dict
            if k.startswith("student_moe_encoder.moe.experts.backbone.network.")
            and k.endswith(".weight")
        )
        if not gating_w or not backbone_w:
            raise RuntimeError("Unsupported MoE-CTS checkpoint layout.")
        expert_num = state_dict[gating_w[-1]].shape[0]
        history_length = state_dict[backbone_w[0]].shape[1] // num_actor_obs
        expert_hidden_dim = state_dict[backbone_w[-1]].shape[0] // expert_num
        student_hidden_dims = [*([state_dict[key].shape[0] for key in gating_w[:-1]]), expert_hidden_dim]
    else:
        checkpoint_kind = "cts"
        teacher_hidden_dims = _infer_hidden_dims_from_mlp(state_dict, "teacher_encoder.")
        stu_w = sorted(k for k in state_dict if k.startswith("student_encoder.") and k.endswith(".weight"))
        if not stu_w:
            raise RuntimeError("Unsupported CTS checkpoint layout.")
        history_length = state_dict[stu_w[0]].shape[1] // num_actor_obs
        student_hidden_dims = _infer_hidden_dims_from_mlp(state_dict, "student_encoder.")
        expert_num = 0

    return {
        "checkpoint_kind": checkpoint_kind,
        "latent_dim": latent_dim,
        "num_actor_obs": num_actor_obs,
        "num_critic_obs": num_critic_obs,
        "num_actions": num_actions,
        "history_length": history_length,
        "actor_hidden_dims": actor_hidden_dims,
        "critic_hidden_dims": critic_hidden_dims,
        "teacher_encoder_hidden_dims": teacher_hidden_dims,
        "student_encoder_hidden_dims": student_hidden_dims,
        "expert_num": expert_num,
    }


def load_cts_policy(checkpoint_path: str, env, device: str):
    """Load a CTS-family checkpoint and return a student-only inference callable.

    All model dimensions are inferred directly from the checkpoint weights so
    the function is robust to different training configurations:
      - latent_dim   : output of last linear in teacher_encoder
      - num_actor_obs: actor first-layer input minus latent_dim
      - num_critic_obs: critic first-layer input minus latent_dim
      - num_actions  : actor last-layer output
      - history_length: student encoder first-layer input / num_actor_obs
    """
    from rsl_rl_cts import ActorCriticCTS, ActorCriticMoECTS

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    dims = _infer_cts_checkpoint_dims(checkpoint_path, map_location=device)
    checkpoint_kind = dims["checkpoint_kind"]
    latent_dim = dims["latent_dim"]
    num_actor_obs = dims["num_actor_obs"]
    num_actions = dims["num_actions"]
    num_critic_obs = dims["num_critic_obs"]
    history_length = dims["history_length"]

    print(
        f"[CTS] Checkpoint type: {checkpoint_kind}"
    )
    print(f"[CTS] Checkpoint dims: actor_obs={num_actor_obs}, critic_obs={num_critic_obs}, "
          f"actions={num_actions}, latent={latent_dim}, history={history_length}")
    print(f"[CTS] Env reports:     num_obs={env.num_obs}, num_actions={env.num_actions}")
    if env.num_obs != num_actor_obs:
        raise RuntimeError(
            f"Observation dimension mismatch: checkpoint expects {num_actor_obs}D obs "
            f"but env produces {env.num_obs}D.  Ensure the deployment env cfg matches "
            f"the training env cfg (height_scanner size, observation terms, etc.)."
        )

    if checkpoint_kind == "moe_cts":
        model = ActorCriticMoECTS(
            num_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_envs=env.num_envs,
            history_length=history_length,
            actor_hidden_dims=dims["actor_hidden_dims"],
            critic_hidden_dims=dims["critic_hidden_dims"],
            teacher_encoder_hidden_dims=dims["teacher_encoder_hidden_dims"],
            student_encoder_hidden_dims=dims["student_encoder_hidden_dims"],
            expert_num=dims["expert_num"],
            latent_dim=latent_dim,
        ).to(device)
    else:
        model = ActorCriticCTS(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_envs=env.num_envs,
            history_length=history_length,
            actor_hidden_dims=dims["actor_hidden_dims"],
            critic_hidden_dims=dims["critic_hidden_dims"],
            teacher_encoder_hidden_dims=dims["teacher_encoder_hidden_dims"],
            student_encoder_hidden_dims=dims["student_encoder_hidden_dims"],
            latent_dim=latent_dim,
        ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[CTS] Loaded checkpoint from: {checkpoint_path}")
    return model.act_inference


def resolve_checkpoint_path(
    log_root_path: str, load_run: str, load_checkpoint: str
) -> str:
    """Resolve a checkpoint whether models live in the experiment root or a run subdir."""
    root_checkpoints = [
        entry
        for entry in os.listdir(log_root_path)
        if os.path.isfile(os.path.join(log_root_path, entry))
        and re.match(load_checkpoint, entry)
    ]
    if root_checkpoints:
        root_checkpoints.sort(key=lambda name: f"{name:0>15}")
        return os.path.join(log_root_path, root_checkpoints[-1])

    return get_checkpoint_path(log_root_path, load_run, load_checkpoint)


# Keyboard teleop speed settings (m/s for linear, rad/s for angular).
# These defaults are safe for general Go2 PPO play. 45D stair checkpoints are
# narrowed at runtime to match their actual training command envelope.
CMD_LIN_VEL = 1.5   # forward / backward / strafe speed  [m/s]
CMD_ANG_VEL = 1.5   # yaw rotation speed                 [rad/s]


def _viewer_should_follow() -> bool:
    if args_cli.viewer_follow == "on":
        return True
    if args_cli.viewer_follow == "off":
        return False
    return args_cli.robot_amount == 1


def _base_command_tensor(num_envs: int, device: torch.device | str) -> torch.Tensor:
    cmds = torch.zeros(num_envs, 3, device=device)
    for i in range(num_envs):
        value = custom_rl_env.base_command.get(str(i))
        if value is not None:
            cmds[i] = torch.tensor(value, device=device)
    return cmds


def sub_keyboard_event(event, *args, **kwargs) -> bool:

    if len(custom_rl_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "W":
                custom_rl_env.base_command["0"] = [CMD_LIN_VEL, 0, 0]
            if event.input.name == "S":
                custom_rl_env.base_command["0"] = [-CMD_LIN_VEL, 0, 0]
            if event.input.name == "A":
                custom_rl_env.base_command["0"] = [0, CMD_LIN_VEL, 0]
            if event.input.name == "D":
                custom_rl_env.base_command["0"] = [0, -CMD_LIN_VEL, 0]
            if event.input.name == "Q":
                custom_rl_env.base_command["0"] = [0, 0, CMD_ANG_VEL]
            if event.input.name == "E":
                custom_rl_env.base_command["0"] = [0, 0, -CMD_ANG_VEL]

            if len(custom_rl_env.base_command) > 1:
                if event.input.name == "I":
                    custom_rl_env.base_command["1"] = [CMD_LIN_VEL, 0, 0]
                if event.input.name == "K":
                    custom_rl_env.base_command["1"] = [-CMD_LIN_VEL, 0, 0]
                if event.input.name == "J":
                    custom_rl_env.base_command["1"] = [0, CMD_LIN_VEL, 0]
                if event.input.name == "L":
                    custom_rl_env.base_command["1"] = [0, -CMD_LIN_VEL, 0]
                if event.input.name == "U":
                    custom_rl_env.base_command["1"] = [0, 0, CMD_ANG_VEL]
                if event.input.name == "O":
                    custom_rl_env.base_command["1"] = [0, 0, -CMD_ANG_VEL]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(custom_rl_env.base_command)):
                custom_rl_env.base_command[str(i)] = [0, 0, 0]
    return True


def move_copter(copter):

    # TODO tmp solution for test
    if custom_rl_env.base_command["0"] == [0, 0, 0]:
        copter_move_cmd = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device="cuda:0"
        )

    if custom_rl_env.base_command["0"] == [1, 0, 0]:
        copter_move_cmd = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device="cuda:0"
        )

    if custom_rl_env.base_command["0"] == [-1, 0, 0]:
        copter_move_cmd = torch.tensor(
            [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device="cuda:0"
        )

    if custom_rl_env.base_command["0"] == [0, 1, 0]:
        copter_move_cmd = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], device="cuda:0"
        )

    if custom_rl_env.base_command["0"] == [0, -1, 0]:
        copter_move_cmd = torch.tensor(
            [[0.0, -1.0, 0.0, 0.0, 0.0, 0.0]], device="cuda:0"
        )

    if custom_rl_env.base_command["0"] == [0, 0, 1]:
        copter_move_cmd = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device="cuda:0"
        )

    if custom_rl_env.base_command["0"] == [0, 0, -1]:
        copter_move_cmd = torch.tensor(
            [[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]], device="cuda:0"
        )

    copter.write_root_velocity_to_sim(copter_move_cmd)
    copter.write_data_to_sim()


def setup_custom_env():
    """Load a custom visual environment USD into the stage.

    This must be called BEFORE env.get_observations() (i.e. before the first
    policy-driven physics step) so that collision meshes and materials are
    fully composed before PhysX runs its first broad-phase query.

    The hospital USD bundles Materials/ and Props/ as relative sub-layers; USD
    resolves them automatically from the file's own directory — no explicit
    sub-layer loading is required here.

    Coordinate alignment:
      • hospital.usd has its ground floor at z = 0, matching IsaacLab's
        default terrain plane (/World/ground).  Go2 spawns ~0.5 m above that
        plane, so no spawn-offset adjustment is needed for a single-env run.
      • The hospital prim is anchored at World origin (0, 0, 0).  With
        env_spacing = 2.5 m and a single env the robot spawns near (0, 0),
        placing it at the hospital entrance corridor.  Adjust the translation
        below if a different spawn region is preferred.
    """
    try:
        if args_cli.custom_env == "office":
            cfg_scene = sim_utils.UsdFileCfg(usd_path=str(REPO_ROOT / "assets" / "env" / "office.usd"))
            cfg_scene.func("/World/office", cfg_scene, translation=(0.0, 0.0, 0.0))

        if args_cli.custom_env == "hospital":
            # Absolute path to the hospital asset bundle.
            # Materials/ and Props/ are referenced as relative layers inside
            # hospital.usd and are resolved automatically by the USD runtime.
            _assets_root = os.environ.get(
                "ISAAC_ASSETS_ROOT",
                "/media/user/data1/isaac-sim-assets/merged/Assets/Isaac/4.5",
            )
            HOSPITAL_USD = os.path.join(
                _assets_root, "Isaac/Environments/Hospital/hospital.usd"
            )
            cfg_scene = sim_utils.UsdFileCfg(usd_path=HOSPITAL_USD)
            cfg_scene.func("/World/hospital", cfg_scene, translation=(0.0, 0.0, 0.0))
            # The hospital USD contains a CollisionPlane prim at z=0 which is
            # exactly coincident with IsaacLab's /World/ground plane.  Two
            # overlapping PhysX collision planes produce conflicting contact
            # normals that lock the robot in place.  Deactivate it so only
            # /World/ground drives foot contact.
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            num_planes = _deactivate_hospital_collision_planes(stage)
            if num_planes == 0:
                print("[WARN] Hospital CollisionPlane prim not found — check path")
            print("[INFO] Hospital environment loaded at /World/hospital")

    except Exception as e:
        print(
            f"Error loading custom environment ({e}). "
            "For warehouse/office, download envs from: "
            "https://drive.google.com/drive/folders/1vVGuO1KIX1K6mD6mBHDZGm9nk2vaRyj3?usp=sharing"
        )


def _deactivate_hospital_collision_planes(stage) -> int:
    """Disable duplicate ground-plane prims under the imported hospital subtree.

    The labeled hospital USD does not reliably expose the original
    ``/World/hospital/Root/CollisionPlane`` prim. In practice the duplicate
    hospital-local ground may appear as either ``CollisionPlane`` or
    ``WorldGrid`` after export/composition, so resolve by a small allowlist of
    prim names instead of relying on one hard-coded path.
    """
    hospital_root = stage.GetPrimAtPath("/World/hospital")
    if not hospital_root or not hospital_root.IsValid():
        return 0

    candidate_names = {"CollisionPlane", "WorldGrid"}
    deactivated = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/hospital/"):
            continue
        if prim.GetName() not in candidate_names:
            continue
        prim.SetActive(False)
        deactivated += 1
        print(f"[INFO] Hospital duplicate ground prim deactivated: {path}")
    return deactivated


def _deactivate_warehouse_collision_planes(stage) -> int:
    """Disable duplicate ground-plane prims under the imported warehouse."""
    warehouse_root = stage.GetPrimAtPath("/World/warehouse")
    if not warehouse_root or not warehouse_root.IsValid():
        return 0

    candidate_names = {"CollisionPlane", "GroundPlane", "WorldGrid"}
    deactivated = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/warehouse/"):
            continue
        if prim.GetName() not in candidate_names:
            continue
        prim.SetActive(False)
        deactivated += 1
        print(f"[INFO] Warehouse duplicate ground prim deactivated: {path}")
    return deactivated


def _spawn_hospital_usd(prim_path: str, cfg, translation=None, orientation=None):
    """Spawn hospital USD and immediately deactivate its duplicate ground plane.

    Using this as the spawn func inside AssetBaseCfg lets us load the hospital
    BEFORE gym.make() (so robot and scene appear simultaneously) while still
    removing the CollisionPlane before PhysX runs its first broad-phase reset.

    Without this, two coincident collision planes exist during PhysX init and
    crash scene loading.  Without pre-gym.make loading, hospital collision meshes
    appear mid-episode and trigger base_contact every step.
    """
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    import omni.usd
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    stage = omni.usd.get_context().get_stage()
    repaired_labels = _reconcile_navigation_semantics(stage, prim_path)
    annotation_count = _count_semantic_annotations(stage, prim_path)
    if annotation_count == 0:
        raise RuntimeError(f"Labeled hospital USD has no semantic annotations: {cfg.usd_path}")
    num_planes = _deactivate_hospital_collision_planes(stage)
    _record_custom_scene_bounds(stage, prim_path)
    if num_planes == 0:
        print("[WARN] Hospital CollisionPlane not found during spawn — check USD prim path")
    else:
        print(
            f"[INFO] Hospital spawned with {annotation_count} semantic annotations; "
            f"{num_planes} duplicate ground prim(s) deactivated; "
            f"{repaired_labels} navigation label(s) reconciled"
        )
    return prim


def _spawn_warehouse_usd(prim_path: str, cfg, translation=None, orientation=None):
    """Spawn the labeled warehouse USD.

    Uses the pre-labeled warehouse scene so that semantic segmentation is
    available from the first frame.  Loading BEFORE gym.make() ensures the
    warehouse collision geometry is present during PhysX initialisation and
    avoids mid-episode geometry streaming glitches.
    """
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    import omni.usd
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    stage = omni.usd.get_context().get_stage()
    repaired_labels = _reconcile_navigation_semantics(stage, prim_path)
    annotation_count = _count_semantic_annotations(stage, prim_path)
    if annotation_count == 0:
        raise RuntimeError(f"Labeled warehouse USD has no semantic annotations: {cfg.usd_path}")
    num_planes = _deactivate_warehouse_collision_planes(stage)
    _record_custom_scene_bounds(stage, prim_path)
    if num_planes == 0:
        print("[WARN] Warehouse GroundPlane not found during spawn — check USD prim path")
    else:
        print(
            f"[INFO] Warehouse spawned with {annotation_count} semantic annotations; "
            f"{num_planes} duplicate ground prim(s) deactivated; "
            f"{repaired_labels} navigation label(s) reconciled"
        )
    return prim


def _has_direct_semantic_annotation(prim) -> bool:
    return any("semanticData" in attr.GetName() for attr in prim.GetAttributes())


def _reconcile_navigation_semantics(stage, root_path: str) -> int:
    """Repair stale structural/walkable labels using leaf-first prim names."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    repaired = 0
    walkable = {"floor", "ramp", "stairs"}
    for prim in Usd.PrimRange(root):
        expected = infer_semantic_class(str(prim.GetPath()))
        if expected == "misc":
            continue
        updated = False
        for attribute in prim.GetAttributes():
            if "semanticData" in attribute.GetName():
                current = canonical_semantic_class(str(attribute.Get() or ""))
                if (current in walkable) != (expected in walkable):
                    attribute.Set(expected)
                    updated = True
        if updated:
            repaired += 1
    return repaired


def _count_semantic_annotations(stage, root_path: str) -> int:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    return sum(
        1
        for prim in Usd.PrimRange(root)
        if _has_direct_semantic_annotation(prim)
    )


def _label_unannotated_geometry(stage, root_path: str, profile_hint: str) -> dict[str, int]:
    """Attach inferred class labels to every unlabeled renderable USD prim."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return {}
    counts: dict[str, int] = {}
    preserved = 0
    for prim in Usd.PrimRange(root):
        # Instance proxies cannot be authored directly. Label their instance
        # roots so referenced geometry inherits the class at render time.
        is_renderable = prim.IsA(UsdGeom.Gprim) or prim.IsInstance()
        if not prim.IsActive() or not is_renderable:
            continue
        if _has_direct_semantic_annotation(prim):
            preserved += 1
            continue
        class_name = infer_semantic_class(str(prim.GetPath()), profile_hint)
        if _set_semantic_class(prim, class_name):
            counts[class_name] = counts.get(class_name, 0) + 1
    if preserved:
        counts["preserved"] = preserved
    return counts


def _spawn_generic_usd(prim_path: str, cfg, translation=None, orientation=None):
    """Spawn and label a Data_collection-provided USD before sensors start."""
    global _SPATIAL_MAP_METADATA
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    label_mode = os.environ.get("SEMANTIC_LABEL_MODE", "").strip()
    if label_mode != "runtime_auto":
        raise RuntimeError(
            f"External USD profile must use SEMANTIC_LABEL_MODE=runtime_auto, got '{label_mode}'"
        )
    counts = _label_unannotated_geometry(
        stage,
        prim_path,
        os.environ.get("DATA_COLLECTION_PROFILE", args_cli.custom_env),
    )
    if sum(counts.values()) == 0:
        raise RuntimeError(f"No renderable geometry was labeled under {prim_path}: {cfg.usd_path}")
    _record_custom_scene_bounds(stage, prim_path)
    _SPATIAL_MAP_METADATA = {"semantic_counts": counts}
    summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    print(f"[INFO] Generic custom USD labeled before sensor startup: {summary}")
    print(f"[INFO] Generic custom USD spawned at {prim_path}: {cfg.usd_path}")
    return prim


def _spawn_procedural_layout(prim_path: str, cfg, translation=None, orientation=None):
    """Spawn a deterministic obstacle layout and publish its free-space grid."""
    global _CUSTOM_SCENE_BOUNDS, _CUSTOM_SCENE_PRIM_PATH, _VALID_SPAWN_POINTS
    global _VALID_TRAVERSAL_POINTS, _INITIAL_SPAWN_POINTS
    global _INITIAL_SPAWN_HEADINGS, _SPATIAL_MAP_METADATA

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, prim_path)
    kind = os.environ.get("PROCEDURAL_LAYOUT_KIND", "obstacles")
    seed = int(os.environ.get("PROCEDURAL_LAYOUT_SEED", "42"))
    rng = __import__("random").Random(seed)
    rectangles: list[tuple[float, float, float, float, float]] = []

    def add_box(
        index: int,
        x: float,
        y: float,
        sx: float,
        sy: float,
        height: float,
        color: tuple[float, float, float] | None = None,
        center_z: float | None = None,
        object_orientation: tuple[float, float, float, float] | None = None,
        semantic_class: str = "wall",
    ) -> None:
        box_cfg = sim_utils.CuboidCfg(
            size=(sx, sy, height),
            semantic_tags=[("class", semantic_class)],
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color or (0.28 + 0.12 * (index % 3), 0.34, 0.42),
                roughness=0.92,
                metallic=0.0,
            ),
        )
        spawn_kwargs = {
            "translation": (x, y, height * 0.5 if center_z is None else center_z)
        }
        if object_orientation is not None:
            spawn_kwargs["orientation"] = object_orientation
        box_cfg.func(
            f"{prim_path}/obstacle_{index:03d}", box_cfg, **spawn_kwargs
        )
        rectangles.append((x, y, sx * 0.5, sy * 0.5, height))

    def add_cylinder(
        index: int,
        x: float,
        y: float,
        radius: float,
        height: float,
        color: tuple[float, float, float] | None = None,
        center_z: float | None = None,
        semantic_class: str = "column",
    ) -> None:
        cylinder_cfg = sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            semantic_tags=[("class", semantic_class)],
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color or (0.32, 0.40 + 0.08 * (index % 3), 0.30),
                roughness=0.88,
                metallic=0.0,
            ),
        )
        cylinder_cfg.func(
            f"{prim_path}/obstacle_{index:03d}",
            cylinder_cfg,
            translation=(x, y, height * 0.5 if center_z is None else center_z),
        )
        # The free-space test uses a conservative axis-aligned footprint.
        rectangles.append((x, y, radius, radius, height))

    scene_spec_path = os.environ.get("PROCEDURAL_SCENE_SPEC", "").strip()
    if scene_spec_path:
        document = json.loads(Path(scene_spec_path).read_text(encoding="utf-8"))
        if document.get("layout") != kind or int(document.get("seed", -1)) != seed:
            raise ValueError(
                f"Procedural scene spec does not match runtime layout/seed: {scene_spec_path}"
            )
        for index, obj in enumerate(document.get("objects", [])):
            x, y, center_z = (float(value) for value in obj["position"])
            color = tuple(float(value) for value in obj.get("color", (0.4, 0.4, 0.4)))
            semantic_class = str(obj.get("semantic_class", "misc"))
            if obj["shape"] == "box":
                sx, sy, height = (float(value) for value in obj["size"])
                object_orientation = obj.get("orientation")
                add_box(
                    index,
                    x,
                    y,
                    sx,
                    sy,
                    height,
                    color=color,
                    center_z=center_z,
                    object_orientation=(
                        tuple(float(value) for value in object_orientation)
                        if object_orientation is not None
                        else None
                    ),
                    semantic_class=semantic_class,
                )
            elif obj["shape"] == "cylinder":
                add_cylinder(
                    index, x, y, float(obj["radius"]), float(obj["height"]),
                    color=color, center_z=center_z, semantic_class=semantic_class,
                )
            else:
                raise ValueError(f"Unknown procedural object shape: {obj['shape']}")

        free_points = [tuple(float(value) for value in point) for point in document["valid_points"]]
        traversal_points = [
            tuple(float(value) for value in point)
            for point in document.get("traversal_points", [])
        ]
        initial_points = [
            tuple(float(value) for value in point) for point in document["spawn_points"]
        ]
        headings = [float(value) for value in document["spawn_headings"]]
        bounds = tuple(float(value) for value in document["bounds"])
        resolution = float(document["grid_resolution"])
        _CUSTOM_SCENE_BOUNDS = bounds
        _CUSTOM_SCENE_PRIM_PATH = prim_path
        _VALID_SPAWN_POINTS = free_points
        _VALID_TRAVERSAL_POINTS = traversal_points
        _INITIAL_SPAWN_POINTS = initial_points
        _INITIAL_SPAWN_HEADINGS = headings
        _SPATIAL_MAP_METADATA = {
            "sampling_mode": "scene_spec",
            "scene_spec_format_version": int(document.get("format_version", 1)),
            "scene_spec": scene_spec_path,
            "layout": kind,
            "seed": seed,
            "event_types": document.get("event_types", []),
            "event_anchors": document.get("event_anchors", []),
            "event_waypoints": document.get("event_waypoints", []),
            "metrics": document.get("metrics", {}),
            "spawn_points": [list(point) for point in initial_points],
            "terrain_parameters": document.get("terrain_parameters", {}),
        }
        _write_spatial_map(free_points, traversal_points, resolution)
        print(
            f"[INFO] Procedural scene spec loaded: {len(document['objects'])} objects, "
            f"{len(free_points)} connected free points, {len(traversal_points)} traversal points, "
            f"{len(initial_points)} initial spawns"
        )
        return stage.GetPrimAtPath(prim_path)

    if kind == "maze":
        index = 0
        for x, y, sx, sy in ((0, -5, 10, .2), (0, 5, 10, .2), (-5, 0, .2, 10), (5, 0, .2, 10)):
            add_box(index, x, y, sx, sy, 1.2)
            index += 1
        for wall_index, x in enumerate((-3.0, -1.0, 1.0, 3.0)):
            gap = 3.2 if wall_index % 2 == 0 else -3.2
            lower_length = max(gap - 0.7 - (-5.0), 0.1)
            upper_length = max(5.0 - (gap + 0.7), 0.1)
            add_box(index, x, -5.0 + lower_length * 0.5, .18, lower_length, 1.2)
            index += 1
            add_box(index, x, gap + 0.7 + upper_length * 0.5, .18, upper_length, 1.2)
            index += 1
        for x, y, sx, sy in ((-4, 0, 1.8, .18), (0, 0, 1.8, .18), (4, 0, 1.8, .18)):
            add_box(index, x, y, sx, sy, 1.0)
            index += 1
    elif kind in {"obstacles", "sparse"}:
        obstacle_count = 20 if kind == "obstacles" else 7
        for index in range(obstacle_count):
            for _ in range(100):
                size_max = 1.35 if kind == "obstacles" else 0.90
                sx = rng.uniform(0.45, size_max)
                sy = rng.uniform(0.45, size_max)
                position_limit = 4.5 if kind == "obstacles" else 3.8
                x = rng.uniform(-position_limit, position_limit)
                y = rng.uniform(-position_limit, position_limit)
                if abs(x) < 1.2 and abs(y) < 1.2:
                    continue
                if all(abs(x - item[0]) > (sx * .5 + item[2] + .45) or abs(y - item[1]) > (sy * .5 + item[3] + .45) for item in rectangles):
                    break
            height = rng.uniform(0.5, 1.4)
            if index % 3 == 0:
                add_cylinder(index, x, y, max(sx, sy) * 0.5, height)
            else:
                add_box(index, x, y, sx, sy, height)
        if kind == "obstacles":
            add_box(100, -2.2, 0.0, .18, 5.5, 1.0)
            add_box(101, 2.2, 0.0, .18, 5.5, 1.0)

    resolution = 0.5
    free_points: list[tuple[float, float, float]] = []
    x = -4.5
    while x <= 4.5 + 1e-6:
        y = -4.5
        while y <= 4.5 + 1e-6:
            blocked = any(
                abs(x - cx) <= hx + 0.85 and abs(y - cy) <= hy + 0.65
                for cx, cy, hx, hy, _ in rectangles
            )
            if not blocked:
                free_points.append((x, y, 0.0))
            y += resolution
        x += resolution

    _CUSTOM_SCENE_BOUNDS = (-5.0, 5.0, -5.0, 5.0)
    _CUSTOM_SCENE_PRIM_PATH = prim_path
    _VALID_SPAWN_POINTS = free_points
    _VALID_TRAVERSAL_POINTS = []
    _write_spatial_map(free_points, [], resolution)
    print(f"[INFO] Procedural {kind} layout: {len(rectangles)} obstacles, {len(free_points)} free points")
    return stage.GetPrimAtPath(prim_path)


def _set_semantic_class(prim, class_name: str) -> bool:
    """Attach one Replicator class label to a prim if it exists."""
    if not prim or not prim.IsValid():
        return False
    instance_name = f"class_{class_name.replace(' ', '_')}"
    semantics = Semantics.SemanticsAPI.Apply(prim, instance_name)
    semantics.CreateSemanticTypeAttr().Set("class")
    semantics.CreateSemanticDataAttr().Set(class_name)
    return True


def _configure_default_ground_for_custom_usd(stage) -> bool:
    """Hide the fallback terrain when the imported USD provides its own floor.

    The plane remains active for conservative foot-contact support, while USD
    visibility inheritance removes it from RGB, depth, and semantic rendering.
    """
    global _SPATIAL_MAP_METADATA

    if os.environ.get("HIDE_DEFAULT_GROUND", "false").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    ground = stage.GetPrimAtPath("/World/ground")
    if not ground or not ground.IsValid():
        raise RuntimeError("HIDE_DEFAULT_GROUND=true but /World/ground is missing")
    if not _CUSTOM_FLOOR_BOUNDS:
        raise RuntimeError(
            "Refusing to hide /World/ground: the custom USD exposes no floor meshes"
        )
    imageables = []
    for prim in Usd.PrimRange(ground):
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        imageables.append((prim, imageable))
    if not imageables:
        raise RuntimeError(
            "HIDE_DEFAULT_GROUND=true but /World/ground has no imageable prims"
        )

    # Author visibility on the root and concrete descendants. Root visibility
    # is normally sufficient through inheritance, while descendant opinions
    # make the invariant robust to imported terrain assets with local values.
    for _, imageable in imageables:
        imageable.MakeInvisible()

    still_visible = [
        str(prim.GetPath())
        for prim, imageable in imageables
        if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
    ]
    if still_visible:
        raise RuntimeError(
            "Default ground is still sensor-visible after hiding: "
            + ", ".join(still_visible[:8])
        )
    print(
        f"[INFO] Default /World/ground hidden from sensors "
        f"({len(imageables)} imageable prims verified invisible); "
        f"the custom USD supplies {len(_CUSTOM_FLOOR_BOUNDS)} rendered floor meshes"
    )
    _SPATIAL_MAP_METADATA["default_ground_sensor_visibility"] = "invisible"
    _SPATIAL_MAP_METADATA["custom_floor_mesh_count"] = len(_CUSTOM_FLOOR_BOUNDS)
    return True


def _apply_procedural_terrain_from_env(env_cfg) -> None:
    global _CUSTOM_SCENE_BOUNDS, _CUSTOM_SCENE_PRIM_PATH, _VALID_SPAWN_POINTS
    global _VALID_TRAVERSAL_POINTS
    kind = os.environ.get("PROCEDURAL_TERRAIN_KIND", "").strip()
    if not kind:
        return
    if kind == "flat":
        sub_terrains = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
    elif kind == "ramp":
        sub_terrains = {
            "ramp": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=1.0, slope_range=(0.12, 0.28), platform_width=2.0, border_width=0.25
            )
        }
    elif kind == "stairs":
        sub_terrains = {
            "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0, step_height_range=(0.10, 0.18), step_width=0.30,
                platform_width=2.0, border_width=0.5, holes=False
            )
        }
    elif kind == "pits":
        sub_terrains = {
            "pit": terrain_gen.MeshPitTerrainCfg(
                proportion=0.6, pit_depth_range=(0.35, 0.9), platform_width=2.0, double_pit=False
            ),
            "gap": terrain_gen.MeshGapTerrainCfg(
                proportion=0.4, gap_width_range=(0.35, 0.8), platform_width=2.0
            ),
        }
    else:
        raise ValueError(f"Unknown PROCEDURAL_TERRAIN_KIND={kind}")
    generator = TerrainGeneratorCfg(
        seed=int(os.environ.get("PROCEDURAL_LAYOUT_SEED", "42")),
        size=(10.0, 10.0), border_width=2.0, num_rows=1, num_cols=1,
        curriculum=False, use_cache=False, sub_terrains=sub_terrains,
    )
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", terrain_generator=generator,
        collision_group=-1, physics_material=env_cfg.scene.terrain.physics_material,
        debug_vis=False,
    )
    env_cfg.sim.physics_material = env_cfg.scene.terrain.physics_material
    env_cfg.scene.height_scanner.mesh_prim_paths = ["/World/ground"]
    if kind == "flat":
        # The generated tile is 10x10 m. Keep one metre of stopping margin so
        # a goal-conditioned controller cannot walk off its finite edge.
        resolution = 0.5
        free_points = [
            (x * resolution, y * resolution, 0.0)
            for x in range(-8, 9)
            for y in range(-8, 9)
        ]
        _CUSTOM_SCENE_BOUNDS = (-5.0, 5.0, -5.0, 5.0)
        _CUSTOM_SCENE_PRIM_PATH = "/World/ground"
        _VALID_SPAWN_POINTS = free_points
        _VALID_TRAVERSAL_POINTS = []
        _write_spatial_map(free_points, [], resolution)
    print(f"[INFO] Procedural terrain enabled: {kind}")


def cmd_vel_cb(msg, num_robot):
    x = msg.linear.x
    y = msg.linear.y
    z = msg.angular.z
    custom_rl_env.base_command[str(num_robot)] = [x, y, z]


def _request_episode_reset(_request, response):
    """Queue a simulation reset for the main Isaac thread and wait for it."""
    global _EPISODE_RESET_ERROR
    with _EPISODE_RESET_LOCK:
        if _EPISODE_RESET_REQUESTED.is_set():
            response.success = False
            response.message = "episode reset already pending"
            return response
        _EPISODE_RESET_ERROR = ""
        _EPISODE_RESET_COMPLETED.clear()
        _EPISODE_RESET_REQUESTED.set()
    timeout = _spatial_float_env("EPISODE_RESET_SERVICE_TIMEOUT_SEC", 15.0)
    if not _EPISODE_RESET_COMPLETED.wait(timeout=timeout):
        response.success = False
        response.message = f"simulation reset timed out after {timeout:.1f}s"
        return response
    response.success = not _EPISODE_RESET_ERROR
    response.message = _EPISODE_RESET_ERROR or "simulation reset complete"
    return response


def add_cmd_sub(num_envs):
    node_test = rclpy.create_node("position_velocity_publisher")
    for i in range(num_envs):
        node_test.create_subscription(
            Twist, f"robot{i}/cmd_vel", lambda msg, i=i: cmd_vel_cb(msg, str(i)), 10
        )
        node_test.create_service(Trigger, f"robot{i}/reset_episode", _request_episode_reset)
    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node_test,), daemon=True)
    thread.start()


def specify_cmd_for_robots(numv_envs):
    for i in range(numv_envs):
        custom_rl_env.base_command[str(i)] = [0, 0, 0]


def _parse_range_env(name: str) -> tuple[float, float] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        low, high = [float(part.strip()) for part in raw.split(",", 1)]
    except Exception:
        print(f"[WARN] Ignoring malformed {name}='{raw}', expected 'low,high'")
        return None
    if low > high:
        low, high = high, low
    return (low, high)


def _apply_random_start_from_env(env_cfg) -> None:
    enabled = os.environ.get("RANDOM_START_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return

    reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
    if reset_base is None:
        print("[WARN] RANDOM_START_ENABLED=true but env_cfg.events.reset_base is unavailable")
        return

    pose_range = dict(reset_base.params.get("pose_range", {}))
    mapping = {
        "x": _parse_range_env("RANDOM_START_X_RANGE"),
        "y": _parse_range_env("RANDOM_START_Y_RANGE"),
        "yaw": _parse_range_env("RANDOM_START_YAW_RANGE"),
    }
    for key, value in mapping.items():
        if value is not None:
            pose_range[key] = value
    reset_base.params["pose_range"] = pose_range
    dynamic = os.environ.get("DYNAMIC_SPATIAL_BOUNDS", "false").lower() in {
        "1", "true", "yes", "on"
    }
    if dynamic:
        reset_base.func = _reset_root_state_spatially_valid
        print(f"[INFO] Obstacle-aware random start enabled; fallback pose_range={pose_range}")
    else:
        print(f"[INFO] Random start pose_range enabled: {pose_range}")


def run_sim():
    global _EPISODE_RESET_ERROR
    if args_cli.cmd_source == "keyboard":
        # Only subscribe to keyboard teleop in manual mode so it never races
        # with the navigation stack over the same base_command buffer.
        _input = carb.input.acquire_input_interface()
        _appwindow = omni.appwindow.get_default_app_window()
        _keyboard = _appwindow.get_keyboard()
        _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    """Run deploy-time inference with an RSL-RL checkpoint."""
    # ── Resolve agent cfg and checkpoint path FIRST ───────────────────────────
    # We must know the checkpoint type before calling gym.make() because
    # CTS-family checkpoints use dedicated deploy envs whose actor observation
    # size can differ from the default PPO play env (for example 45D
    # stair-student or 48D full-scene student).
    agent_cfg: RslRlOnPolicyRunnerCfg = resolve_agent_cfg(unitree_go2_agent_cfg)
    if args_cli.robot == "g1":
        agent_cfg = resolve_agent_cfg(unitree_g1_agent_cfg)

    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"]))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = resolve_checkpoint_path(
        log_root_path, agent_cfg["load_run"], agent_cfg["load_checkpoint"])
    configure_policy_from_checkpoint(agent_cfg, resume_path)
    print(
        "[INFO] Resolved agent config: "
        f"deploy_architecture={args_cli.deploy_architecture}, "
        f"experiment={agent_cfg['experiment_name']}, "
        f"checkpoint={agent_cfg['load_checkpoint']}, "
        f"noise_std_type={agent_cfg['policy'].get('noise_std_type', 'default')}"
    )

    # ── Choose env cfg to match the checkpoint's observation space ────────────
    _cts = _is_cts_checkpoint(resume_path)
    _cts_dims = _infer_cts_checkpoint_dims(resume_path) if _cts else None
    if _cts and args_cli.robot == "go2":
        actual_arch = _cts_dims["checkpoint_kind"]
        if actual_arch != args_cli.deploy_architecture:
            print(
                "[WARN] Requested deploy architecture "
                f"'{args_cli.deploy_architecture}' but checkpoint is '{actual_arch}'. "
                "Proceeding with the checkpoint's actual structure."
            )
    if args_cli.robot == "g1":
        env_cfg = G1RoughEnvCfg()
    elif _cts:
        actor_obs_dim = _cts_dims["num_actor_obs"]
        checkpoint_label = "MoE-CTS" if _cts_dims["checkpoint_kind"] == "moe_cts" else "CTS"
        if actor_obs_dim == 45:
            env_cfg = Go2StairDeployCfg()
            print(f"[INFO] {checkpoint_label} checkpoint detected — using Go2StairDeployCfg (45D student obs)")
        elif actor_obs_dim == 48:
            env_cfg = Go2FullSceneDeployCfg()
            print(f"[INFO] {checkpoint_label} checkpoint detected — using Go2FullSceneDeployCfg (48D velocity-aware student obs)")
        else:
            raise RuntimeError(
                f"Unsupported CTS-family actor observation size {actor_obs_dim}. "
                "Add a matching deploy env cfg before running this checkpoint."
            )
    else:
        env_cfg = UnitreeGo2CustomEnvCfg()

    _apply_procedural_terrain_from_env(env_cfg)
    _apply_random_start_from_env(env_cfg)

    global CMD_LIN_VEL, CMD_ANG_VEL
    if _cts and args_cli.robot != "g1":
        actor_obs_dim = _cts_dims["num_actor_obs"]
        if actor_obs_dim == 45:
            CMD_LIN_VEL = 1.0
            CMD_ANG_VEL = 1.5
            print(
                "[INFO] 45D stair deploy command limits enabled: "
                f"lin={CMD_LIN_VEL:.1f} m/s, yaw={CMD_ANG_VEL:.1f} rad/s"
            )
        elif actor_obs_dim == 48:
            CMD_LIN_VEL = 1.0
            CMD_ANG_VEL = 1.0
            print(
                "[INFO] 48D full-scene deploy command limits enabled: "
                f"lin={CMD_LIN_VEL:.1f} m/s, yaw={CMD_ANG_VEL:.1f} rad/s"
            )

    # Use a robot-following camera for single-robot play so visibility is not
    # dominated by the hospital/world origin framing.
    if _viewer_should_follow():
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (2.8, 2.8, 1.6)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.45)
        print("[INFO] Viewer set to follow robot root")
    else:
        print("[INFO] Viewer follow disabled; using free camera")

    # add N robots to env
    env_cfg.scene.num_envs = args_cli.robot_amount
    specify_cmd_for_robots(env_cfg.scene.num_envs)

    # --- Custom USD envs: inject BEFORE gym.make() via custom spawners ---------
    # _spawn_hospital_usd deactivates the hospital's internal CollisionPlane
    # immediately after loading the USD, so PhysX never sees two coincident
    # ground planes during its first broad-phase reset (which happens inside
    # gym.make).  This gives correct load order (scene + robot appear together)
    # and avoids the base_contact-every-step bug from post-gym.make loading.
    custom_env_usd_path = os.environ.get("CUSTOM_ENV_USD_PATH", "").strip()
    custom_env_prim_path = os.environ.get("CUSTOM_ENV_PRIM_PATH", "").strip()
    procedural_layout = os.environ.get("PROCEDURAL_LAYOUT_KIND", "").strip()

    if procedural_layout:
        layout_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
        layout_cfg.func = _spawn_procedural_layout
        env_cfg.scene.custom_env = AssetBaseCfg(
            prim_path=f"/World/procedural_{procedural_layout}",
            spawn=layout_cfg,
        )

    elif args_cli.custom_env == "hospital":
        expected_hospital = (REPO_ROOT / "assets" / "env" / "hospital_labeled.usd").resolve()
        requested_hospital = Path(custom_env_usd_path).resolve() if custom_env_usd_path else expected_hospital
        if requested_hospital != expected_hospital or not expected_hospital.is_file():
            raise RuntimeError(f"Hospital must use labeled asset: {expected_hospital}")
        HOSPITAL_USD = str(expected_hospital)
        hospital_spawn_cfg = sim_utils.UsdFileCfg(usd_path=HOSPITAL_USD)
        hospital_spawn_cfg.func = _spawn_hospital_usd
        env_cfg.scene.custom_env = AssetBaseCfg(
            prim_path="/World/hospital",
            spawn=hospital_spawn_cfg,
        )

    # Reuse the hospital loading path for warehouse: registering the USD as an
    # AssetBaseCfg makes IsaacLab compose the scene together with the robot.
    elif args_cli.custom_env == "warehouse":
        expected_warehouse = (REPO_ROOT / "assets" / "env" / "full_warehouse_labeled.usd").resolve()
        requested_warehouse = Path(custom_env_usd_path).resolve() if custom_env_usd_path else expected_warehouse
        if requested_warehouse != expected_warehouse or not expected_warehouse.is_file():
            raise RuntimeError(f"Warehouse must use labeled asset: {expected_warehouse}")
        WAREHOUSE_USD = str(expected_warehouse)
        warehouse_spawn_cfg = sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD)
        warehouse_spawn_cfg.func = _spawn_warehouse_usd
        env_cfg.scene.custom_env = AssetBaseCfg(
            prim_path="/World/warehouse",
            spawn=warehouse_spawn_cfg,
        )

    elif custom_env_usd_path:
        prim_name = re.sub(r"[^0-9A-Za-z_]+", "_", args_cli.custom_env or "custom_env").strip("_")
        generic_prim_path = custom_env_prim_path or f"/World/{prim_name or 'custom_env'}"
        generic_spawn_cfg = sim_utils.UsdFileCfg(usd_path=custom_env_usd_path)
        generic_spawn_cfg.func = _spawn_generic_usd
        env_cfg.scene.custom_env = AssetBaseCfg(
            prim_path=generic_prim_path,
            spawn=generic_spawn_cfg,
        )
        print(f"[INFO] Data_collection USD env: {custom_env_usd_path} -> {generic_prim_path}")

    # Backward-compatible direct launch path. Data_collection's office_asset
    # profile uses CUSTOM_ENV_USD_PATH and therefore the generic AssetBaseCfg
    # branch above.
    elif args_cli.custom_env == "office":
        setup_custom_env()

    # create isaac environment (scene load + robot spawn; physics starts here)
    env = gym.make(args_cli.task, cfg=env_cfg)
    # The default/imported terrain otherwise has no Replicator label and is
    # merged with every other unlabeled prim in semantic segmentation.
    import omni.usd

    semantic_stage = omni.usd.get_context().get_stage()
    default_ground_hidden = _configure_default_ground_for_custom_usd(semantic_stage)
    if (
        not default_ground_hidden
        and _set_semantic_class(semantic_stage.GetPrimAtPath("/World/ground"), "floor")
    ):
        print("[INFO] Semantic class applied: /World/ground -> floor")
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    if args_cli.robot == "go2" and _cts:
        robot = env.unwrapped.scene["robot"]
        ordered_ids, ordered_names = robot.find_joints(GO2_MODEL_JOINT_NAMES, preserve_order=True)
        print(f"[INFO] Go2 asset joint order: {robot.joint_names}")
        print(f"[INFO] Go2 deploy joint order: {ordered_names} {ordered_ids}")

    # ── Load policy ───────────────────────────────────────────────────────────
    _device = agent_cfg["device"]
    if _cts:
        policy = load_cts_policy(resume_path, env, _device)
    else:
        ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=_device)
        ppo_runner.load(resume_path)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # initialize ROS2 node
    rclpy.init()
    base_node = RobotBaseNode(env_cfg.scene.num_envs)
    if args_cli.cmd_source == "ros2":
        add_cmd_sub(env_cfg.scene.num_envs)
        print("[INFO] Command source: ROS2 (/robot{i}/cmd_vel -> base_command)")
    else:
        print("[INFO] Command source: keyboard (W/S/A/D/Q/E)")

    # Attach sensors and create render products AFTER the full scene
    # (including custom env) is on stage, but before the first obs step.
    UnitreeL1_annotator_lst = add_rtx_lidar(env_cfg.scene.num_envs, args_cli.robot, "UnitreeL1", False)
    ExtraLidar_annotator_lst = add_rtx_lidar(env_cfg.scene.num_envs, args_cli.robot, "Extra", False)
    annotator_lst = UnitreeL1_annotator_lst + ExtraLidar_annotator_lst
    add_camera(env_cfg.scene.num_envs, args_cli.robot, camera_mode=args_cli.camera_mode)
    # add_copter_camera()

    # create ros2 camera stream omnigraph (rgb + depth + semantic_segmentation)
    for i in range(env_cfg.scene.num_envs):
        create_front_cam_omnigraph(
            i,
            camera_mode=args_cli.camera_mode,
            enable_depth=args_cli.enable_depth,
        )

    # First observation collection — triggers the first policy-driven step.
    # Everything above must be fully on stage before this call.
    obs, _ = env.get_observations()

    # Log initial observation stats before any policy step
    # print(f"[DBG init] obs min={obs.min().item():.4f}  max={obs.max().item():.4f}  "
    #       f"mean={obs.mean().item():.4f}  finite={torch.isfinite(obs).all().item()}")
    # print(f"[DBG init] obs={obs[0].tolist()}")

    def _require_finite_runtime(name, value):
        if torch.is_tensor(value):
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite {name}; Isaac runtime is unhealthy")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                _require_finite_runtime(f"{name}.{key}", item)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                _require_finite_runtime(f"{name}[{index}]", item)

    # simulate environment
    _dbg_step = 0
    _dbg_resets = 0
    _dbg_timeouts = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if _EPISODE_RESET_REQUESTED.is_set():
                try:
                    for robot_index in range(env_cfg.scene.num_envs):
                        custom_rl_env.base_command[str(robot_index)] = [0, 0, 0]
                    obs, _ = env.reset()
                    _require_finite_runtime("reset observation", obs)
                    _require_finite_runtime(
                        "reset robot root state",
                        env.unwrapped.scene["robot"].data.root_state_w,
                    )
                    _dbg_resets += env_cfg.scene.num_envs
                    print(
                        "[INFO] Data-collection episode reset complete; "
                        "robot respawned from collision-free spatial candidates",
                        flush=True,
                    )
                except Exception as error:
                    _EPISODE_RESET_ERROR = str(error)
                    print(f"[ERROR] Episode reset failed: {error}", flush=True)
                finally:
                    _EPISODE_RESET_REQUESTED.clear()
                    _EPISODE_RESET_COMPLETED.set()
                continue
            # agent stepping
            _require_finite_runtime("policy observation", obs)
            actions = policy(obs)
            _require_finite_runtime("policy action", actions)
            # env stepping
            obs, _, terminated, extras = env.step(actions)
            _require_finite_runtime("step observation", obs)
            _require_finite_runtime(
                "step robot root state",
                env.unwrapped.scene["robot"].data.root_state_w,
            )
            _dbg_resets += terminated.sum().item()
            if "time_outs" in extras:
                _dbg_timeouts += extras["time_outs"].sum().item()
            pub_robo_data_ros2(
                args_cli.robot,
                env_cfg.scene.num_envs,
                base_node,
                env,
                annotator_lst,
                base_commands=custom_rl_env.base_command,
                timestamp_seconds=omni.timeline.get_timeline_interface().get_current_time(),
            )
            # move_copter(copter)

            # --- debug: print early root state so we can distinguish
            # physics-state corruption from render-only issues ---
            _dbg_step += 1
            _should_log_dbg = (
                (_dbg_step <= 5)
                or (_dbg_step <= 50 and _dbg_step % 10 == 0)
                or (_dbg_step <= 200 and _dbg_step % 50 == 0)
            )
            if _should_log_dbg:
                robot = env.unwrapped.scene["robot"]
                base_pos    = robot.data.root_pos_w[0].tolist()
                base_quat   = robot.data.root_quat_w[0].tolist()
                base_vel    = robot.data.root_lin_vel_w[0].tolist()
                cmd_obs_start = 6 if obs.shape[1] == 45 else 9
                obs_cmd     = obs[0, cmd_obs_start:cmd_obs_start + 3].tolist()
                act_max     = actions.abs().max().item()
                max_torque  = robot.data.applied_torque[0].abs().max().item()
                quat_norm   = float(robot.data.root_quat_w[0].norm().item())
                state_finite = bool(
                    torch.isfinite(robot.data.root_pos_w[0]).all()
                    and torch.isfinite(robot.data.root_quat_w[0]).all()
                    and torch.isfinite(robot.data.root_lin_vel_w[0]).all()
                )
                # Sample first annotator to print beamId range (only first 2 steps)
                try:
                    _raw = annotator_lst[0]["annotator_object"].get_data()["data"]
                    if _raw is not None and _raw.size > 0 and _raw.dtype.names and "beamId" in _raw.dtype.names:
                        _beam = _raw["beamId"]
                        _beam_info = f"  beamId=[{int(_beam.min())},{int(_beam.max())}]"
                    else:
                        _beam_info = "  beamId=N/A"
                except Exception:
                    _beam_info = ""
                # print(
                #     f"[DBG step={_dbg_step}]"
                #     f"  cmd={custom_rl_env.base_command}"
                #     f"  obs_cmd={[round(v,3) for v in obs_cmd]}"
                #     f"  pos={[round(v,3) for v in base_pos]}"
                #     f"  quat={[round(v,3) for v in base_quat]}"
                #     f"  quat_norm={quat_norm:.4f}"
                #     f"  pos_z={base_pos[2]:.4f}"
                #     f"  vel_xy={[round(v,3) for v in base_vel[:2]]}"
                #     f"  max_action={act_max:.4f}"
                #     f"  max_torque={max_torque:.4f}"
                #     f"  finite={state_finite}"
                #     f"  total_resets={int(_dbg_resets)}"
                #     f"  total_timeouts={int(_dbg_timeouts)}"
                #     + _beam_info
                # )
                # if _dbg_step <= 5:
                #     print(f"[DBG step={_dbg_step}] actions={[round(v,4) for v in actions[0].tolist()]}")
                #     print(f"[DBG step={_dbg_step}] obs={[round(v,4) for v in obs[0].tolist()]}")

    env.close()
