"""B2W navigation evaluation in the labeled Hospital USD scene."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Sequence

import torch
import numpy as np
from scipy.spatial import cKDTree

from isaaclab.managers import CommandTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, transform_points

from isaaclab_nav_task.navigation.config.b2w.navigation_env_cfg import (
    B2WNavigationEnvCfg,
)
from isaaclab_nav_task.navigation.mdp.navigation.goal_commands import (
    RobotNavigationGoalCommand,
    SuccessRateTracker,
)
from isaaclab_nav_task.navigation.mdp.navigation.goal_commands_cfg import (
    RobotNavigationGoalCommandCfg,
)


def _points(document: dict, key: str, fallback: str) -> torch.Tensor:
    raw = document.get(key) or document.get(fallback) or []
    value = torch.as_tensor(raw, dtype=torch.float32)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value):
        raise ValueError(f"spatial map has no usable {key}")
    return value


def _connected_stair_routes(points: torch.Tensor) -> list[torch.Tensor]:
    """Extract locally connected traversal surfaces with meaningful z span."""
    if len(points) < 2:
        return []
    values = points.detach().cpu().numpy()
    tree = cKDTree(values[:, :2])
    parent = np.arange(len(values), dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first, second in tree.query_pairs(r=0.38):
        if abs(values[first, 2] - values[second, 2]) <= 0.20:
            union(first, second)
    components: dict[int, list[int]] = {}
    for index in range(len(values)):
        components.setdefault(find(index), []).append(index)
    routes = []
    for indices in components.values():
        route = points[indices]
        if len(route) >= 20 and float(route[:, 2].max() - route[:, 2].min()) >= 0.50:
            routes.append(route)
    return routes


def _verified_stair_entries(
    route: torch.Tensor, spawn_points: torch.Tensor, max_distance: float = 1.5
) -> torch.Tensor:
    """Return collision-checked ground spawns near the bottom of a stair route."""
    z_min = route[:, 2].min()
    low = route[route[:, 2] <= z_min + 0.12]
    distances = torch.cdist(spawn_points, low).amin(dim=1)
    return spawn_points[distances <= max_distance]


def _load_pct_goals(path: Path) -> torch.Tensor:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("frame_id") != "map" or document.get("tomo_file") != "hospital":
        raise ValueError(f"PCT goals do not describe the Hospital map: {path}")
    raw = document.get("goals") or []
    points = torch.tensor(
        [[item["x"], item["y"], item["z"]] for item in raw], dtype=torch.float32
    )
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"PCT goal file has no usable points: {path}")
    return points


def validate_hospital_inputs(stairs: bool = False, pct_stairs: bool = False) -> dict:
    """Validate external Hospital assets without constructing the simulation."""
    map_path = Path(os.environ["SRU_HOSPITAL_SPATIAL_MAP"]).expanduser()
    document = json.loads(map_path.read_text(encoding="utf-8"))
    goal_points = _points(document, "goal_points", "valid_points")
    spawn_points = _points(document, "spawn_points", "valid_points")
    traversal_points = torch.as_tensor(
        document.get("traversal_points") or [], dtype=torch.float32
    ).reshape(-1, 3)
    stair_routes = _connected_stair_routes(traversal_points)
    verified_stair_routes = [
        route
        for route in stair_routes
        if len(_verified_stair_entries(route, spawn_points))
    ]
    if stairs and not stair_routes:
        raise ValueError("Hospital spatial map has no connected stair traversal route")
    if stairs and not verified_stair_routes:
        raise ValueError("Hospital stair routes have no nearby verified spawn point")

    pct_goal_count = 0
    matched_route_count = 0
    if pct_stairs:
        pct_path = Path(os.environ["SRU_PCT_HOSPITAL_GOALS"]).expanduser()
        pct_goals = _load_pct_goals(pct_path)
        pct_goal_count = len(pct_goals)
        for route in stair_routes:
            z_min = route[:, 2].min()
            low = route[route[:, 2] <= z_min + 0.12]
            if float(torch.cdist(pct_goals[:, :2], low[:, :2]).min()) <= 4.0:
                matched_route_count += 1
        if not matched_route_count:
            raise ValueError("No Hospital stair entrance is near a PCT goal anchor")

    return {
        "spawn_points": len(spawn_points),
        "goal_points": len(goal_points),
        "traversal_points": len(traversal_points),
        "stair_routes": len(stair_routes),
        "verified_stair_routes": len(verified_stair_routes),
        "pct_goals": pct_goal_count,
        "pct_matched_stair_routes": matched_route_count,
    }


class HospitalGoalCommand(RobotNavigationGoalCommand):
    """Use precomputed collision-free Hospital points instead of terrain masks."""

    def __init__(self, cfg: RobotNavigationGoalCommandCfg, env):
        CommandTerm.__init__(self, cfg, env)
        self.env = env
        self.robot = env.scene[cfg.asset_name]
        self.num_terrain_rows = 1
        self.num_terrain_cols = 1
        self.terrain_size = 1.0
        self._init_command_buffers()
        self._init_tracking_buffers()
        self._init_metrics()

        map_path = Path(os.environ["SRU_HOSPITAL_SPATIAL_MAP"]).expanduser()
        document = json.loads(map_path.read_text(encoding="utf-8"))
        self._goal_points = _points(document, "goal_points", "valid_points").to(self.device)
        self._spawn_points = _points(document, "spawn_points", "valid_points").to(self.device)
        self._min_goal_distance = float(os.environ.get("SRU_MIN_GOAL_DISTANCE", "3.0"))
        self._max_goal_distance = float(os.environ.get("SRU_MAX_GOAL_DISTANCE", "15.0"))
        self._stairs = os.environ.get("SRU_STAIRS_EPISODES", "0") == "1"
        self._pct_stairs = os.environ.get("SRU_PCT_STAIRS_EPISODES", "0") == "1"
        traversal = document.get("traversal_points") or []
        self._traversal_points = torch.as_tensor(
            traversal, dtype=torch.float32, device=self.device
        ).reshape(-1, 3)
        connected_routes = _connected_stair_routes(self._traversal_points)
        self._stair_routes = [
            route
            for route in connected_routes
            if len(_verified_stair_entries(route, self._spawn_points))
        ]
        if self._stairs and not self._stair_routes:
            raise ValueError(
                "Hospital spatial map has no connected stair route with a verified entrance"
            )
        self._pct_goals = None
        if self._pct_stairs:
            pct_path = Path(os.environ["SRU_PCT_HOSPITAL_GOALS"]).expanduser()
            self._pct_goals = _load_pct_goals(pct_path).to(self.device)
            self._pct_stair_routes = []
            for route in self._stair_routes:
                z_min = route[:, 2].min()
                low = route[route[:, 2] <= z_min + 0.12]
                nearest = float(torch.cdist(self._pct_goals[:, :2], low[:, :2]).min())
                if nearest <= 4.0:
                    self._pct_stair_routes.append(route)
            if not self._pct_stair_routes:
                raise ValueError("No Hospital stair entrance is near a PCT goal anchor")
            high_count = int((self._pct_goals[:, 2] >= 0.5).sum())
            print(
                f"[sru] PCT anchors: {len(self._pct_goals)} "
                f"(high-z goals in source file: {high_count}); "
                f"matched stair routes: {len(self._pct_stair_routes)}",
                flush=True,
            )
        if self._stairs:
            spans = [float(route[:, 2].max() - route[:, 2].min()) for route in self._stair_routes]
            print(
                f"[sru] Connected stair routes: {len(spans)}, "
                f"z_span={min(spans):.2f}-{max(spans):.2f} m",
                flush=True,
            )

    def _sample_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._stairs:
            routes = self._pct_stair_routes if self._pct_stairs else self._stair_routes
            route = routes[
                torch.randint(len(routes), (1,), device=self.device).item()
            ]
            z_min, z_max = route[:, 2].min(), route[:, 2].max()
            low = route[route[:, 2] <= z_min + 0.12]
            high = route[route[:, 2] >= z_max - 0.12]
            starts = _verified_stair_entries(route, self._spawn_points)
            goals = high
            if self._pct_stairs:
                # PCT points choose the entrance; the actual B2W pose is snapped
                # to a collision-checked spawn near the bottom of the stairs.
                anchor_distances = torch.cdist(self._pct_goals[:, :2], starts[:, :2])
                anchor_index, start_index = torch.unravel_index(
                    torch.argmin(anchor_distances), anchor_distances.shape
                )
                self.last_pct_anchor = self._pct_goals[anchor_index].clone()
                spawn = starts[start_index]
            else:
                self.last_pct_anchor = None
                spawn = starts[torch.randint(len(starts), (1,), device=self.device).item()]
            distances = torch.linalg.vector_norm(goals[:, :2] - spawn[:2], dim=1)
            candidates = goals[
                (distances >= 1.0) & (distances <= self._max_goal_distance)
            ]
            if not len(candidates):
                candidates = goals
            goal = candidates[
                torch.randint(len(candidates), (1,), device=self.device).item()
            ]
            return spawn, goal

        spawn = self._spawn_points[
            torch.randint(len(self._spawn_points), (1,), device=self.device).item()
        ]
        goals = self._goal_points
        distances = torch.linalg.vector_norm(goals[:, :2] - spawn[:2], dim=1)
        candidates = goals[
            (distances >= self._min_goal_distance)
            & (distances <= self._max_goal_distance)
        ]
        if not len(candidates):
            candidates = goals[distances >= 1.0]
        goal = candidates[
            torch.randint(len(candidates), (1,), device=self.device).item()
        ]
        return spawn, goal

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        self._reset_tracking_state(ids)
        for env_id in ids.tolist():
            spawn, goal = self._sample_pair()
            self.env.scene.terrain.env_origins[env_id, :2] = spawn[:2]
            self.env.scene.terrain.env_origins[env_id, 2] = spawn[2] + 0.05
            self.spawn_position_world[env_id] = torch.stack(
                (spawn[0], spawn[1], spawn[2] + 0.05)
            )
            self.goal_position_world[env_id] = torch.stack(
                (goal[0], goal[1], goal[2] + 0.35)
            )
            direction = goal[:2] - spawn[:2]
            self.spawn_heading_world[env_id] = torch.atan2(direction[1], direction[0])
        distance = torch.linalg.vector_norm(
            self.goal_position_world[ids] - self.spawn_position_world[ids], dim=1
        )
        self.initial_distance_to_goal[ids] = distance
        self.closest_distance_to_goal[ids] = distance


@configclass
class HospitalGoalCommandCfg(RobotNavigationGoalCommandCfg):
    class_type: type = HospitalGoalCommand


@configclass
class B2WSRUHospitalEnvCfg(B2WNavigationEnvCfg):
    """Evaluation-only config; the standard Go2 and SRU configs stay untouched."""

    def __post_init__(self):
        print("[sru] Applying upstream B2W navigation config...", flush=True)
        super().__post_init__()
        print("[sru] Upstream B2W config ready; applying isolated Hospital overrides...", flush=True)
        hospital = os.environ["SRU_HOSPITAL_USD"]
        self.scene.num_envs = 1
        self.scene.env_spacing = 0.0
        self.scene.replicate_physics = False
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=hospital,
            env_spacing=0.0,
            collision_group=-1,
            visual_material=None,
            debug_vis=False,
        )
        # Ray-casting against the imported root includes its descendant meshes.
        self.scene.raycast_camera.mesh_prim_paths = ["/World/ground"]
        self.scene.height_scanner_critic.mesh_prim_paths = ["/World/ground"]
        self.scene.height_scanner_critic.attach_yaw_only = True
        self.commands.robot_goal = HospitalGoalCommandCfg(
            asset_name="robot", resampling_time_range=(1e9, 1e9), debug_vis=True
        )
        self.episode_length_s = float(os.environ.get("SRU_EPISODE_SECONDS", "60"))
        self.observations.policy.enable_corruption = False
        # The critic is training-only and initializes a CUDA height encoder.
        # Evaluation needs policy, metrics, and low-level locomotion groups only.
        self.observations.critic = None
        self.events.randomize_camera_height = None
        self.events.randomize_action_scale = None
        self.events.randomize_low_pass_filter_alpha = None
        self.events.push_robot = None
        self.curriculum.disable_backward_penalty = None
        print("[sru] Hospital overrides ready.", flush=True)
