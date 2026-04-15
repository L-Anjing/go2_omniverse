# Copyright (c) 2024, RoboVerse community
# BSD-2-Clause License

"""Train the Go2 locomotion model for stair climbing using CTS.

Algorithm: Concurrent Teacher-Student (CTS)  https://arxiv.org/abs/2405.10830
---------------------------------------------------------------------------
Pure PPO cannot learn to climb stairs (confirmed by go2_rl_gym UPDATE.md v0.1.1).
CTS solves this by splitting environments into:
  • Teacher envs (75%): receive privileged observations (clean height scan +
    clean velocities + foot contact forces) → accurate value function.
  • Student envs (25%): receive only noisy proprioceptive obs history →
    deployment-realistic.
  A distillation loss keeps the student encoder aligned to the teacher encoder,
  so the student implicitly encodes terrain from its motion history.

Deployment
----------
Only the student encoder + actor are needed at runtime.
omniverse_sim.py loads the checkpoint via CTSRunner.get_inference_policy()
which returns model.act_inference (student path, history maintained internally).

Observation spaces
------------------
Policy obs  (actor, 48D, with noise):
  base_lin_vel(3) + base_ang_vel(3) + proj_gravity(3) + vel_cmd(3) +
  joint_pos(12) + joint_vel(12) + actions(12)

Critic obs  (teacher encoder, 263D, clean):
  policy_obs(48) + foot_contact_forces_norm(4) +
  dof_torques(12) + dof_acc(12) + height_scan(187)

Usage
-----
    python train_stairs.py --headless --num_envs 4096 --max_iterations 150000 \
        --experiment_name unitree_go2_fullscene_cts
    python train_stairs.py --headless --num_envs 4096 --max_iterations 150000 \
        --experiment_name unitree_go2_fullscene_cts \
        --checkpoint logs/rsl_rl/unitree_go2_fullscene_cts/seed_123/model_14000.pt
"""

from __future__ import annotations

import argparse
from dataclasses import field
from isaaclab.app import AppLauncher

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Go2 stair climbing with CTS")
parser.add_argument("--num_envs",        type=int,   default=4096)
parser.add_argument("--max_iterations",  type=int,   default=150000)
parser.add_argument("--checkpoint",      type=str,   default="",
                    help="Resume from a CTS checkpoint (.pt)")
parser.add_argument("--seed",            type=int,   default=42)
parser.add_argument("--experiment_name", type=str,   default="unitree_go2_fullscene_cts")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

MAX_SAFE_ENVS_PER_GPU = 4096
if args_cli.num_envs > MAX_SAFE_ENVS_PER_GPU:
    raise ValueError(
        f"--num_envs={args_cli.num_envs} is too large for the current Go2 full-scene CTS setup. "
        f"Use <= {MAX_SAFE_ENVS_PER_GPU} envs per GPU to avoid CUDA OOM."
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after IsaacSim app launch) ──────────────────────────────────────
import os
import sys
import gymnasium as gym
import torch

# Make the CTS package importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))
from rsl_rl_cts import CTSRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.utils.math as math_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import mdp as _base_mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)


# ── Custom observation functions ──────────────────────────────────────────────

def foot_contact_force_penalty(
    env, sensor_cfg: SceneEntityCfg, threshold: float = 147.0
) -> torch.Tensor:
    """Penalise excessive foot contact force magnitude (stomping / hard landings).

    go2_rl_gym v0.1.4: "真机上有明显的跺脚动作, 加上feet_contact_forces惩罚接触力大小,
    设置接触力阈值大小为go2的重量15*9.8=147, 系数为-1".

    Returns sum of force magnitudes above `threshold` (body-weight of Go2 ≈ 147 N).
    Normalised so that a 147 N excess gives ~1.0 penalty.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 4, 3)
    norms  = torch.norm(forces, dim=-1)                            # (N, 4)
    excess = torch.clamp(norms - threshold, min=0.0)               # (N, 4)
    return torch.nan_to_num(excess.sum(dim=-1) / threshold,        # (N,)
                            nan=0.0, posinf=1.0)


def foot_contact_forces_norm(
    env, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Foot contact force magnitudes scaled exactly like go2_rl_gym CTS."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 4, 3)
    norms  = torch.norm(forces, dim=-1)                            # (N, 4)
    return torch.nan_to_num(norms * 1.0e-3, nan=0.0, posinf=1.0)


def scaled_base_ang_vel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Match go2_rl_gym student/teacher angular velocity scaling."""
    return torch.nan_to_num(mdp.base_ang_vel(env, asset_cfg=asset_cfg) * 0.25, nan=0.0)


def scaled_base_lin_vel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Match go2_rl_gym privileged linear velocity scaling."""
    return torch.nan_to_num(mdp.base_lin_vel(env, asset_cfg=asset_cfg) * 2.0, nan=0.0)


def scaled_velocity_commands(
    env,
    command_name: str,
) -> torch.Tensor:
    """Scale commands like go2_rl_gym CTS: [2.0, 2.0, 0.25]."""
    commands = mdp.generated_commands(env, command_name=command_name)[:, :3]
    scale = torch.tensor([2.0, 2.0, 0.25], device=commands.device, dtype=commands.dtype)
    return torch.nan_to_num(commands * scale, nan=0.0)


def scaled_joint_vel_rel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Match go2_rl_gym joint velocity scaling."""
    return torch.nan_to_num(mdp.joint_vel_rel(env, asset_cfg=asset_cfg) * 0.05, nan=0.0)


def scaled_height_scan(
    env,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Match go2_rl_gym privileged height scaling after clipping."""
    heights = mdp.height_scan(env, sensor_cfg=sensor_cfg).clamp(-1.0, 1.0)
    return torch.nan_to_num(heights * 2.5, nan=0.0, posinf=2.5, neginf=-2.5)


# ── Mixed stair terrain ───────────────────────────────────────────────────────
STAIRS_MIXED_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.05,
            amplitude_range=(0.01, 0.06),
            num_waves=6,
            border_width=0.25,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),
        "rough_slope": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.05,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.02, 0.22),
            step_width=0.30,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.02, 0.22),
            step_width=0.30,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_width_range=(0.30, 1.00),
            obstacle_height_range=(0.02, 0.10),
            num_obstacles=20,
            platform_width=2.0,
            border_width=0.25,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
    },
)


def _curriculum_value(current_iter: int, cfg: dict[str, float | int]) -> float:
    """Linearly interpolate a curriculum value from iteration count."""
    start_iter = int(cfg["start_iter"])
    end_iter = int(cfg["end_iter"])
    start_value = float(cfg["start_value"])
    end_value = float(cfg["end_value"])
    if current_iter <= start_iter:
        return start_value
    if current_iter >= end_iter:
        return end_value
    alpha = (current_iter - start_iter) / max(end_iter - start_iter, 1)
    return start_value + alpha * (end_value - start_value)


LIN_VEL_Z_REWARD_CURRICULUM = {
    "start_iter": 0,
    "end_iter": 1500,
    "start_value": 1.0,
    "end_value": 0.0,
}

BASE_HEIGHT_REWARD_CURRICULUM = {
    "start_iter": 0,
    "end_iter": 5000,
    "start_value": 1.0,
    "end_value": 10.0,
}

DYNAMIC_TRACKING_SIGMA_CFG = {
    "default_sigma": 0.20,  # tightened from 0.25 — reduces cmd/error_vel_xy
    "min_lin_vel": 0.5,
    "max_lin_vel": 1.5,
    "min_ang_vel": 1.0,
    "max_ang_vel": 2.0,
    # [wave, slope, rough_slope, stairs_up, stairs_down, obstacles, flat]
    "max_sigma": [5 / 12, 1 / 4, 1 / 4, 1 / 2, 1 / 2, 3 / 4, 1 / 4],
}


def _current_training_iter(env, default_steps_per_iteration: int = 48) -> int:
    """Resolve the current training iteration from the command term when possible."""
    steps_per_iteration = default_steps_per_iteration
    command_manager = getattr(env, "command_manager", None)
    if command_manager is not None:
        command_term = command_manager.get_term("base_velocity")
        steps_per_iteration = max(
            int(getattr(getattr(command_term, "cfg", None), "steps_per_iteration", default_steps_per_iteration)),
            1,
        )
    return int(env.common_step_counter // steps_per_iteration)


def _reward_curriculum_scale(env, cfg: dict[str, float | int]) -> float:
    """Return the reference-aligned curriculum scale for the current training step."""
    return _curriculum_value(_current_training_iter(env), cfg)


def _terrain_dynamic_sigma(
    env,
    target_vel_abs: torch.Tensor,
    *,
    default_sigma: float,
    min_vel: float,
    max_vel: float,
    max_sigma: list[float],
) -> torch.Tensor:
    """Dynamic tracking sigma from go2_rl_gym, adapted to IsaacLab terrain IDs."""
    sigma = torch.full_like(target_vel_abs, default_sigma)
    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is None:
        return sigma

    max_sigma_tensor = torch.tensor(max_sigma, device=target_vel_abs.device, dtype=target_vel_abs.dtype)
    terrain_ids = terrain_types.to(dtype=torch.long).clamp_(0, len(max_sigma) - 1)
    target_sigma = max_sigma_tensor[terrain_ids]

    if max_vel > min_vel:
        interp_mask = (target_vel_abs >= min_vel) & (target_vel_abs < max_vel)
        ratio = (target_vel_abs[interp_mask] - min_vel) / (max_vel - min_vel)
        sigma[interp_mask] = default_sigma + ratio * (target_sigma[interp_mask] - default_sigma)
    sigma[target_vel_abs >= max_vel] = target_sigma[target_vel_abs >= max_vel]

    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain_levels is not None:
        level_scale = torch.clamp(torch.exp((terrain_levels.float() + 1.0) / 10.0) - 1.0, max=1.0)
        sigma = default_sigma + level_scale.to(dtype=target_vel_abs.dtype) * (sigma - default_sigma)

    return sigma


class StairVelocityCommand(UniformVelocityCommand):
    """Velocity command generator with go2_rl_gym-style curricula."""

    cfg: "StairVelocityCommandCfg"

    def __init__(self, cfg: "StairVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.commanded_planar_displacement = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_is_limit_vel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_command_ranges = {
            "lin_vel_x": tuple(cfg.ranges.lin_vel_x),
            "lin_vel_y": tuple(cfg.ranges.lin_vel_y),
            "ang_vel_z": tuple(cfg.ranges.ang_vel_z),
            "heading": tuple(cfg.ranges.heading) if cfg.ranges.heading is not None else None,
        }
        self.env_command_ranges = {
            "lin_vel_x": torch.zeros(self.num_envs, 2, device=self.device),
            "lin_vel_y": torch.zeros(self.num_envs, 2, device=self.device),
            "ang_vel_z": torch.zeros(self.num_envs, 2, device=self.device),
        }
        if self.cfg.heading_command:
            self.env_command_ranges["heading"] = torch.zeros(self.num_envs, 2, device=self.device)
        combos = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for yaw in (-1, 0, 1):
                    if x == 0 and y == 0 and yaw == 0:
                        continue
                    combos.append((x, y, yaw))
        self.limit_vel_combinations = torch.tensor(combos, device=self.device, dtype=torch.long)
        self.max_lin_vel = 0.0
        self._last_range_signature = None
        self._apply_command_range_curriculum(force=True)
        self.metrics["command_resamples"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["zero_command_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["front_air_ratio_moving"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["double_front_air_ratio_moving"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rear_air_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rear_air_ratio_moving"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rear_air_ratio_standing"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["double_rear_air_ratio_moving"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["front_rear_contact_imbalance_moving"] = torch.zeros(self.num_envs, device=self.device)

        metric_names = (
            "front_air_ratio_moving",
            "double_front_air_ratio_moving",
            "rear_air_ratio",
            "rear_air_ratio_moving",
            "rear_air_ratio_standing",
            "double_rear_air_ratio_moving",
            "front_rear_contact_imbalance_moving",
        )
        self._gait_metric_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in metric_names
        }
        self._gait_metric_counts = {
            name: torch.zeros(self.num_envs, device=self.device) for name in metric_names
        }

        contact_sensor = self._env.scene.sensors["contact_forces"]
        self._diag_foot_body_ids, _ = contact_sensor.find_bodies(
            ["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
        )

    def reset(self, env_ids=None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self._finalize_gait_metrics(env_ids)
        self.commanded_planar_displacement[env_ids] = 0.0
        self.last_is_limit_vel[env_ids] = False
        for metric_name in self._gait_metric_sums:
            self._gait_metric_sums[metric_name][env_ids] = 0.0
            self._gait_metric_counts[metric_name][env_ids] = 0.0
        return super().reset(env_ids)

    def _finalize_gait_metrics(self, env_ids):
        for metric_name in self._gait_metric_sums:
            sums = self._gait_metric_sums[metric_name][env_ids]
            counts = self._gait_metric_counts[metric_name][env_ids]
            self.metrics[metric_name][env_ids] = torch.where(
                counts > 0.0,
                sums / counts.clamp(min=1.0),
                torch.zeros_like(sums),
            )

    def _current_iter(self) -> int:
        steps_per_iter = max(int(self.cfg.steps_per_iteration), 1)
        return int(self._env.common_step_counter // steps_per_iter)

    def _current_zero_command_prob(self) -> float:
        return _curriculum_value(self._current_iter(), self.cfg.zero_command_curriculum)

    def _apply_command_range_curriculum(self, force: bool = False):
        current_iter = self._current_iter()
        ranges = {
            "lin_vel_x": tuple(self.cfg.ranges.lin_vel_x),
            "lin_vel_y": tuple(self.cfg.ranges.lin_vel_y),
            "ang_vel_z": tuple(self.cfg.ranges.ang_vel_z),
            "heading": tuple(self.cfg.ranges.heading) if self.cfg.ranges.heading is not None else None,
        }
        for cfg in self.cfg.command_range_curriculum:
            if current_iter >= int(cfg["iter"]):
                ranges["lin_vel_x"] = tuple(cfg["lin_vel_x"])
                ranges["lin_vel_y"] = tuple(cfg["lin_vel_y"])
                ranges["ang_vel_z"] = tuple(cfg["ang_vel_z"])
                if self.cfg.heading_command and cfg.get("heading") is not None:
                    ranges["heading"] = tuple(cfg["heading"])

        signature = (
            ranges["lin_vel_x"],
            ranges["lin_vel_y"],
            ranges["ang_vel_z"],
            ranges["heading"],
        )
        if not force and signature == self._last_range_signature:
            return
        self.current_command_ranges = ranges
        self.max_lin_vel = max(
            abs(ranges["lin_vel_x"][0]),
            abs(ranges["lin_vel_x"][1]),
            abs(ranges["lin_vel_y"][0]),
            abs(ranges["lin_vel_y"][1]),
        )
        self._last_range_signature = signature
        self._update_env_command_ranges()
        print(
            "[INFO] Stair command range update:"
            f" iter={current_iter}"
            f" x={ranges['lin_vel_x']}"
            f" y={ranges['lin_vel_y']}"
            f" yaw={ranges['ang_vel_z']}"
        )

    def _update_env_command_ranges(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            bounds = torch.tensor(self.current_command_ranges[key], device=self.device)
            self.env_command_ranges[key][env_ids] = bounds
        if self.cfg.heading_command:
            self.env_command_ranges["heading"][env_ids] = torch.tensor(
                self.current_command_ranges["heading"], device=self.device
            )

        terrain = getattr(self._env.scene, "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is None or not self.cfg.terrain_max_command_ranges:
            return

        env_terrain_types = terrain_types[env_ids]
        for terrain_id, terrain_cfg in enumerate(self.cfg.terrain_max_command_ranges):
            terrain_env_ids = env_ids[env_terrain_types == terrain_id]
            if len(terrain_env_ids) == 0:
                continue
            for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                base = self.env_command_ranges[key][terrain_env_ids]
                cap = torch.tensor(terrain_cfg[key], device=self.device)
                base[:, 0] = torch.maximum(base[:, 0], cap[0])
                base[:, 1] = torch.minimum(base[:, 1], cap[1])
                self.env_command_ranges[key][terrain_env_ids] = base
            if self.cfg.heading_command and "heading" in terrain_cfg:
                base = self.env_command_ranges["heading"][terrain_env_ids]
                cap = torch.tensor(terrain_cfg["heading"], device=self.device)
                base[:, 0] = torch.maximum(base[:, 0], cap[0])
                base[:, 1] = torch.minimum(base[:, 1], cap[1])
                self.env_command_ranges["heading"][terrain_env_ids] = base

    def _sample_axis(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand_like(lower)

    def _sample_planar_with_lower_bound(
        self,
        env_ids,
        min_planar_speed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_bounds = self.env_command_ranges["lin_vel_x"][env_ids]
        y_bounds = self.env_command_ranges["lin_vel_y"][env_ids]
        x = self._sample_axis(x_bounds[:, 0], x_bounds[:, 1])
        y = self._sample_axis(y_bounds[:, 0], y_bounds[:, 1])

        max_x = torch.maximum(x_bounds[:, 0].abs(), x_bounds[:, 1].abs())
        max_y = torch.maximum(y_bounds[:, 0].abs(), y_bounds[:, 1].abs())
        max_planar = torch.sqrt(max_x.pow(2) + max_y.pow(2))
        min_planar_speed = torch.minimum(min_planar_speed, max_planar * 0.95)

        for _ in range(4):
            norm = torch.sqrt(x.pow(2) + y.pow(2))
            mask = norm < min_planar_speed
            if not mask.any():
                return x, y
            x[mask] = self._sample_axis(x_bounds[mask, 0], x_bounds[mask, 1])
            y[mask] = self._sample_axis(y_bounds[mask, 0], y_bounds[mask, 1])

        mask = torch.sqrt(x.pow(2) + y.pow(2)) < min_planar_speed
        if mask.any():
            fallback_ids = mask.nonzero(as_tuple=False).flatten()
            x_cap = max_x[fallback_ids]
            y_cap = max_y[fallback_ids]
            use_x = x_cap >= y_cap
            sign = torch.where(torch.rand(len(fallback_ids), device=self.device) < 0.5, -1.0, 1.0)
            x_mag = torch.minimum(min_planar_speed[fallback_ids], x_cap)
            y_mag = torch.minimum(min_planar_speed[fallback_ids], y_cap)
            x[fallback_ids] = torch.where(use_x, sign * x_mag, torch.zeros_like(x_mag))
            y[fallback_ids] = torch.where(use_x, torch.zeros_like(y_mag), sign * y_mag)
            x[fallback_ids] = x[fallback_ids].clamp(x_bounds[fallback_ids, 0], x_bounds[fallback_ids, 1])
            y[fallback_ids] = y[fallback_ids].clamp(y_bounds[fallback_ids, 0], y_bounds[fallback_ids, 1])

        return x, y

    def _apply_limit_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        change_env_ids = env_ids
        if self.cfg.limit_vel_invert_when_continuous:
            was_limited = self.last_is_limit_vel[env_ids]
            invert_env_ids = env_ids[was_limited]
            if len(invert_env_ids) > 0:
                self.vel_command_b[invert_env_ids, :3] *= -1.0
            change_env_ids = env_ids[~was_limited]

        if len(change_env_ids) > 0:
            combo_ids = torch.randint(
                0, self.limit_vel_combinations.shape[0], (len(change_env_ids),), device=self.device
            )
            combos = self.limit_vel_combinations[combo_ids]
            for axis, key in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
                bounds = self.env_command_ranges[key][change_env_ids]
                values = torch.where(combos[:, axis] < 0, bounds[:, 0], bounds[:, 1])
                values = torch.where(combos[:, axis] == 0, torch.zeros_like(values), values)
                self.vel_command_b[change_env_ids, axis] = values

        self.last_is_limit_vel[env_ids] = True

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return

        self._apply_command_range_curriculum()
        self._update_env_command_ranges(env_ids)

        n = len(env_ids)
        if self.cfg.heading_command:
            self.is_heading_env[env_ids] = torch.rand(n, device=self.device) <= self.cfg.rel_heading_envs
            self.heading_target[env_ids] = self._sample_axis(
                self.env_command_ranges["heading"][env_ids, 0],
                self.env_command_ranges["heading"][env_ids, 1],
            )
        self.is_standing_env[env_ids] = torch.rand(n, device=self.device) <= self.cfg.rel_standing_envs

        remaining_episode_time = (
            (self._env.max_episode_length - self._env.episode_length_buf[env_ids]).float() * self._env.step_dt
        ).clamp(min=self._env.step_dt)
        remaining_distance = (
            self.cfg.dynamic_resample_target_distance
            - torch.norm(self.commanded_planar_displacement[env_ids], dim=1)
        ).clamp(min=0.0)
        min_planar_speed = torch.zeros(n, device=self.device)
        if self.cfg.dynamic_resample_commands:
            min_planar_speed = remaining_distance / remaining_episode_time

        vx, vy = self._sample_planar_with_lower_bound(env_ids, min_planar_speed)
        self.vel_command_b[env_ids, 0] = vx
        self.vel_command_b[env_ids, 1] = vy
        self.vel_command_b[env_ids, 2] = self._sample_axis(
            self.env_command_ranges["ang_vel_z"][env_ids, 0],
            self.env_command_ranges["ang_vel_z"][env_ids, 1],
        )

        rand_prob = torch.rand(n, device=self.device)
        zero_command_prob = self._current_zero_command_prob()
        self.metrics["zero_command_prob"][env_ids] = zero_command_prob

        limit_mask = rand_prob < self.cfg.limit_vel_prob
        limit_env_ids = env_ids[limit_mask]
        self.last_is_limit_vel[env_ids] = False
        if len(limit_env_ids) > 0:
            self._apply_limit_commands(limit_env_ids)

        min_prob = self.cfg.limit_vel_prob
        max_prob = min_prob + zero_command_prob
        if zero_command_prob > 0.0:
            next_zero_time = torch.clamp(
                remaining_episode_time
                - remaining_distance / (self.cfg.zero_command_speed_fraction * max(self.max_lin_vel, 1e-6)),
                min=0.0,
                max=self.cfg.resampling_time_range[1],
            )
            zero_mask = (rand_prob >= min_prob) & (rand_prob < max_prob) & (next_zero_time > 0.0)
            zero_env_ids = env_ids[zero_mask]
            if len(zero_env_ids) > 0:
                self.vel_command_b[zero_env_ids, :2] = 0.0
                self.vel_command_b[zero_env_ids, 2] = 0.0
                self.time_left[zero_env_ids] = next_zero_time[zero_mask]
                add_yaw_mask = (
                    torch.rand(len(zero_env_ids), device=self.device)
                    < self.cfg.limit_ang_vel_at_zero_command_prob
                )
                add_yaw_env_ids = zero_env_ids[add_yaw_mask]
                if len(add_yaw_env_ids) > 0:
                    yaw_bounds = self.env_command_ranges["ang_vel_z"][add_yaw_env_ids]
                    direction = torch.where(
                        torch.rand(len(add_yaw_env_ids), device=self.device) < 0.5,
                        yaw_bounds[:, 0],
                        yaw_bounds[:, 1],
                    )
                    self.vel_command_b[add_yaw_env_ids, 2] = direction

        standing_env_ids = env_ids[self.is_standing_env[env_ids]]
        if len(standing_env_ids) > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0

        self.commanded_planar_displacement[env_ids] += (
            self.vel_command_b[env_ids, :2] * self.time_left[env_ids].unsqueeze(1)
        )
        self.metrics["command_resamples"][env_ids] += 1.0

    def _update_command(self):
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                heading_error = math_utils.wrap_to_pi(
                    self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
                )
                self.vel_command_b[env_ids, 2] = torch.clamp(
                    self.cfg.heading_control_stiffness * heading_error,
                    min=self.env_command_ranges["ang_vel_z"][env_ids, 0],
                    max=self.env_command_ranges["ang_vel_z"][env_ids, 1],
                )
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0

    def _update_metrics(self):
        super()._update_metrics()

        contact_sensor = self._env.scene.sensors["contact_forces"]
        contact_time = contact_sensor.data.current_contact_time[:, self._diag_foot_body_ids]
        in_contact = contact_time > 0.0

        front_contacts = in_contact[:, :2].sum(dim=1).float()
        rear_contacts = in_contact[:, 2:].sum(dim=1).float()
        front_air_ratio = (~in_contact[:, :2]).float().mean(dim=1)
        rear_air_ratio = (~in_contact[:, 2:]).float().mean(dim=1)
        double_front_air = (front_contacts == 0.0).float()
        double_rear_air = (rear_contacts == 0.0).float()
        front_rear_imbalance = torch.abs(front_contacts - rear_contacts) / 2.0

        cmd_norm = torch.linalg.norm(self.vel_command_b[:, :2], dim=1)
        body_vel = torch.linalg.norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)
        moving = cmd_norm > 0.1
        standing = (cmd_norm < 0.1) & (body_vel < 0.2)

        self._gait_metric_sums["front_air_ratio_moving"] += front_air_ratio * moving.float()
        self._gait_metric_counts["front_air_ratio_moving"] += moving.float()

        self._gait_metric_sums["double_front_air_ratio_moving"] += double_front_air * moving.float()
        self._gait_metric_counts["double_front_air_ratio_moving"] += moving.float()

        self._gait_metric_sums["rear_air_ratio"] += rear_air_ratio
        self._gait_metric_counts["rear_air_ratio"] += 1.0

        self._gait_metric_sums["rear_air_ratio_moving"] += rear_air_ratio * moving.float()
        self._gait_metric_counts["rear_air_ratio_moving"] += moving.float()

        self._gait_metric_sums["rear_air_ratio_standing"] += rear_air_ratio * standing.float()
        self._gait_metric_counts["rear_air_ratio_standing"] += standing.float()

        self._gait_metric_sums["double_rear_air_ratio_moving"] += double_rear_air * moving.float()
        self._gait_metric_counts["double_rear_air_ratio_moving"] += moving.float()

        self._gait_metric_sums["front_rear_contact_imbalance_moving"] += (
            front_rear_imbalance * moving.float()
        )
        self._gait_metric_counts["front_rear_contact_imbalance_moving"] += moving.float()


@configclass
class StairVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Reference-style command curriculum for stair CTS training."""

    class_type: type = StairVelocityCommand
    steps_per_iteration: int = 48
    dynamic_resample_commands: bool = True
    dynamic_resample_target_distance: float = 5.0
    limit_vel_prob: float = 0.2
    limit_vel_invert_when_continuous: bool = True
    limit_ang_vel_at_zero_command_prob: float = 0.2
    zero_command_speed_fraction: float = 0.8
    zero_command_curriculum: dict[str, float | int] = field(
        default_factory=lambda: {
            "start_iter": 0,
            "end_iter": 1500,
            "start_value": 0.0,
            "end_value": 0.1,
        }
    )
    command_range_curriculum: list[dict[str, object]] = field(
        default_factory=lambda: [
            {
                # Delayed from 20000→40000: the sudden speed expansion at iter 20000
                # caused terrain_level to jump 2.9→5.5 (noise_std 1.0→2.3) in both
                # seeds, collapsing training. Give the policy 40k iters at ±0.5 m/s
                # to consolidate gait quality before extending the command range.
                "iter": 40000,
                "lin_vel_x": (-1.0, 1.0),
                "lin_vel_y": (-1.0, 1.0),
                "ang_vel_z": (-1.5, 1.5),
                "heading": (-1.57, 1.57),
            },
            {
                "iter": 80000,  # delayed from 50000 proportionally
                "lin_vel_x": (-2.0, 2.0),
                "lin_vel_y": (-1.0, 1.0),
                "ang_vel_z": (-2.0, 2.0),
                "heading": (-1.57, 1.57),
            },
        ]
    )
    terrain_max_command_ranges: list[dict[str, tuple[float, float]]] = field(
        default_factory=lambda: [
            {"lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.5, 1.5), "heading": (-1.57, 1.57)},  # wave
            {"lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.5, 1.5), "heading": (-1.57, 1.57)},  # slope
            {"lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.5, 1.5), "heading": (-1.57, 1.57)},  # rough_slope
            {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.5, 1.5), "heading": (-1.57, 1.57)},  # stairs_up
            {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.5, 1.5), "heading": (-1.57, 1.57)},  # stairs_down
            {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.5, 1.5), "heading": (-1.57, 1.57)},  # obstacles
            {"lin_vel_x": (-2.0, 2.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-2.0, 2.0), "heading": (-1.57, 1.57)},  # flat
        ]
    )


def _terrain_levels_by_command_progress(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain curriculum that compares actual progress against commanded progress."""
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command_term = env.command_manager.get_term("base_velocity")
    actual_distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    commanded_distance = torch.norm(command_term.commanded_planar_displacement[env_ids], dim=1)
    # Lowered threshold from /2 (4 m) to /3 (2.67 m): seed_42 terrain stayed at
    # level 0.9 for the entire run because the robot rarely walked 4 m from spawn.
    # /3 gives more frequent upgrade opportunities without being too aggressive.
    move_up = actual_distance > terrain.cfg.terrain_generator.size[0] / 3
    move_down = (commanded_distance > 0.5) & (actual_distance < 0.5 * commanded_distance)
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


# ── Dynamic foot clearance reward ─────────────────────────────────────────────
def _dynamic_foot_clearance_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    base_margin: float,
    max_additional: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward feet for lifting in proportion to upcoming terrain height."""
    sensor = env.scene.sensors[sensor_cfg.name]
    asset  = env.scene[asset_cfg.name]

    has_hits = False
    if hasattr(sensor.data, "ray_hits_w"):
        ray_z    = sensor.data.ray_hits_w[:, :, 2]
        has_hits = True
    elif hasattr(sensor.data, "ray_hits"):
        ray_z    = sensor.data.ray_hits[:, :, 2]
        has_hits = True

    if has_hits:
        ray_z        = torch.nan_to_num(ray_z, nan=0.0, posinf=0.0, neginf=0.0)
        base_height  = asset.data.root_pos_w[:, 2].unsqueeze(1)
        rel_height   = ray_z - base_height
        mean_rel     = torch.mean(torch.clamp(rel_height, min=0.0), dim=1)
        target_height = base_margin + torch.clamp(mean_rel, 0.0, max_additional)
    else:
        target_height = torch.full(
            (env.num_envs,), base_margin,
            device=env.device, dtype=asset.data.body_pos_w.dtype,
        )

    foot_z   = torch.nan_to_num(asset.data.body_pos_w[:, asset_cfg.body_ids, 2],
                                 nan=0.0, posinf=0.0, neginf=0.0)
    foot_vel = torch.nan_to_num(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2],
                                 nan=0.0, posinf=0.0, neginf=0.0)
    foot_z_err    = torch.square(foot_z - target_height.unsqueeze(1))
    foot_vel_tanh = torch.tanh(tanh_mult * torch.norm(foot_vel, dim=2))
    reward        = foot_z_err * foot_vel_tanh
    exponent      = torch.clamp(torch.sum(reward, dim=1) / std, max=20.0)
    return torch.exp(-exponent)


def _finite_ray_ground_height(
    env,
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Estimate local terrain height from finite scanner rays only."""
    sensor = env.scene.sensors[sensor_cfg.name]

    if hasattr(sensor.data, "ray_hits_w"):
        ray_z = sensor.data.ray_hits_w[:, :, 2]
    elif hasattr(sensor.data, "ray_hits"):
        ray_z = sensor.data.ray_hits[:, :, 2]
    else:
        return None

    valid_mask = torch.isfinite(ray_z)
    safe_ray_z = torch.where(valid_mask, ray_z, torch.zeros_like(ray_z))
    valid_count = valid_mask.sum(dim=1)
    ground_height = safe_ray_z.sum(dim=1) / valid_count.clamp(min=1)
    return ground_height, valid_count


def _safe_terrain_base_height_l2(
    env,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height relative to local terrain while ignoring invalid rays.

    IsaacLab's built-in ``base_height_l2`` becomes unsafe on stairs if any ray
    hit is ``+inf``. Replacing the missing values with zeros is also wrong
    because it turns the reward into an absolute world-height penalty, which
    punishes successful ascent. Here we estimate local ground height from only
    finite ray hits and fall back to zero penalty if a frame has no valid hits.
    """
    asset = env.scene[asset_cfg.name]

    ground = _finite_ray_ground_height(env, sensor_cfg)
    if ground is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=asset.data.root_pos_w.dtype)

    ground_height, valid_count = ground
    # If all rays are invalid for a frame, fall back to zero penalty instead of
    # inventing an arbitrary world-frame target.
    fallback_ground = asset.data.root_pos_w[:, 2] - target_height
    ground_height = torch.where(valid_count > 0, ground_height, fallback_ground)

    adjusted_target = target_height + ground_height
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target)


def _curriculum_scaled_lin_vel_z_l2(
    env,
    curriculum_cfg: dict[str, float | int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference reward curriculum: fade out vertical-velocity penalty early."""
    scale = _reward_curriculum_scale(env, curriculum_cfg)
    return mdp.lin_vel_z_l2(env, asset_cfg=asset_cfg) * scale


def _curriculum_scaled_base_height_l2(
    env,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    curriculum_cfg: dict[str, float | int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference reward curriculum: ramp terrain-relative base-height constraint."""
    scale = _reward_curriculum_scale(env, curriculum_cfg)
    return _safe_terrain_base_height_l2(
        env,
        target_height=target_height,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
    ) * scale


def _track_lin_vel_xy_dynamic_exp(
    env,
    command_name: str,
    sigma_cfg: dict[str, float | list[float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Velocity tracking with go2_rl_gym dynamic sigma for hard terrain / fast commands."""
    asset = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)
    vel_error_sq = torch.square(commands[:, :2] - asset.data.root_lin_vel_b[:, :2])
    sigma_x = _terrain_dynamic_sigma(
        env,
        torch.abs(commands[:, 0]),
        default_sigma=float(sigma_cfg["default_sigma"]),
        min_vel=float(sigma_cfg["min_lin_vel"]),
        max_vel=float(sigma_cfg["max_lin_vel"]),
        max_sigma=list(sigma_cfg["max_sigma"]),
    )
    sigma_y = _terrain_dynamic_sigma(
        env,
        torch.abs(commands[:, 1]),
        default_sigma=float(sigma_cfg["default_sigma"]),
        min_vel=float(sigma_cfg["min_lin_vel"]),
        max_vel=float(sigma_cfg["max_lin_vel"]),
        max_sigma=list(sigma_cfg["max_sigma"]),
    )
    return torch.nan_to_num(torch.exp(-(vel_error_sq[:, 0] / sigma_x + vel_error_sq[:, 1] / sigma_y)), nan=0.0)


def _track_ang_vel_z_dynamic_exp(
    env,
    command_name: str,
    sigma_cfg: dict[str, float | list[float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw tracking with go2_rl_gym dynamic sigma for hard terrain / fast commands."""
    asset = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)
    ang_vel_error_sq = torch.square(commands[:, 2] - asset.data.root_ang_vel_b[:, 2])
    sigma = _terrain_dynamic_sigma(
        env,
        torch.abs(commands[:, 2]),
        default_sigma=float(sigma_cfg["default_sigma"]),
        min_vel=float(sigma_cfg["min_ang_vel"]),
        max_vel=float(sigma_cfg["max_ang_vel"]),
        max_sigma=list(sigma_cfg["max_sigma"]),
    )
    return torch.nan_to_num(torch.exp(-ang_vel_error_sq / sigma), nan=0.0)


def _feet_regulation_reward(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    base_height_target: float,
) -> torch.Tensor:
    """Penalise high horizontal foot velocity when feet are close to the ground.

    CTS causes feet to move with reduced lateral clearance (close-distance gait).
    This reward, from go2_rl_gym, forces the swing phase to lift the foot:
      penalty = Σ_feet  ||vel_xy||² · exp(-foot_z / decay)

    The original go2_rl_gym reward uses foot height relative to local terrain:
      feet_height = foot_z - ground_z
      penalty = Σ_feet ||vel_xy||² · exp(-feet_height / (0.025 * base_height))

    Our previous IsaacLab port used absolute world-frame foot_z instead of
    terrain-relative height. That made the raw term 1-2 orders of magnitude too
    large on stairs and reintroduced a dying strategy via bad_orientation.

    Fixes:
    1. Reconstruct local ground height from finite height-scanner rays only.
    2. Restore the original decay scale 0.025 * base_height_target.
    3. Keep a velocity-command gate so standing commands do not incentivize hover.
    """
    asset = env.scene[asset_cfg.name]
    ground = _finite_ray_ground_height(env, sensor_cfg)
    if ground is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=asset.data.root_pos_w.dtype)

    ground_height, valid_count = ground
    fallback_ground = asset.data.root_pos_w[:, 2]
    ground_height = torch.where(valid_count > 0, ground_height, fallback_ground)

    foot_z = torch.nan_to_num(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2], nan=0.0)  # (N, 4)
    feet_height = torch.clamp(foot_z - ground_height.unsqueeze(1), min=0.0)
    vel_xy = torch.nan_to_num(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], nan=0.0)  # (N, 4, 2)
    vel_sq = vel_xy.pow(2).sum(dim=-1)  # (N, 4)
    decay = 0.025 * base_height_target
    penalty = (vel_sq * torch.exp(-feet_height / decay)).sum(dim=-1)  # (N,)

    # Only penalise when the robot is commanded to move (mirrors feet_air_time gate)
    cmd_xy = env.command_manager.get_command("base_velocity")[:, :2]
    moving = (torch.norm(cmd_xy, dim=1) > 0.1).float()
    return penalty * moving


def _action_smoothness_penalty(env) -> torch.Tensor:
    """Penalize action jerk to suppress twitchy gait limit cycles.

    IsaacLab already has ``action_rate_l2`` for first-order action changes.
    The CTS failure mode we observe here is more severe: the policy converges
    to a front-leg-dominant hover gait with rapid, oscillatory hind-leg
    commands. A second-order penalty makes those limit cycles more expensive
    without constraining the nominal action scale itself.
    """
    curr = env.action_manager.action
    prev = env.action_manager.prev_action
    if not hasattr(env, "_reward_prev_prev_action"):
        env._reward_prev_prev_action = torch.zeros_like(curr)
    prev_prev = env._reward_prev_prev_action

    jerk = curr - 2.0 * prev + prev_prev
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        jerk = jerk.clone()
        jerk[reset_mask] = 0.0

    env._reward_prev_prev_action = prev.clone()
    return torch.sum(jerk.square(), dim=1)


def _stand_still_joint_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    cmd_threshold: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize joint drift from default only when the robot should be standing.

    This addresses the deploy symptom where the policy keeps one or two hind
    legs lifted even at near-zero command. We intentionally gate this term by
    both command magnitude and actual base speed so it does not fight stair
    climbing or normal crouched locomotion during motion.
    """
    asset = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_dev = torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    should_stand = (cmd < cmd_threshold) & (body_vel < velocity_threshold)
    return joint_dev * should_stand.float()


def _joint_power_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L1 joint power penalty: sum(|τ · q̇|). Matches go2_rl_gym dof_power reward."""
    asset = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.nan_to_num((torques * vel).abs().sum(dim=1), nan=0.0)


def _airborne_torque_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize joint torques on legs that are currently airborne.

    A healthy gait naturally minimises torque on swing legs — the leg is
    unloaded and needs little effort.  Penalising airborne torque encourages
    the policy to relax any leg it has lifted rather than holding it rigid,
    which indirectly closes the hover/wheelbarrow local optima:

    * Rear legs held up rigid → penalised by this term → policy learns to
      either plant them or at least keep them relaxed.
    * Front legs doing the same → equally penalised (symmetric, no front/rear
      bias that could shift the problem from rear to front).

    Applies to ALL legs symmetrically.  Use in combination with
    ``feet_regulation`` (penalises dragging) and ``hip_deviation`` (prevents
    lateral sprawl) for a complete, unbiased gait constraint.
    """
    asset   = env.scene[asset_cfg.name]
    sensor  = env.scene.sensors[sensor_cfg.name]

    # contact_time > 0 means the foot has been in contact this step
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0  # (N, 4)

    # joint torques grouped per leg: (N, 4, dof_per_leg)
    dof_per_leg = len(asset_cfg.joint_ids) // 4
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]          # (N, 12)
    torques_per_leg = torques.view(torques.shape[0], 4, dof_per_leg)     # (N, 4, 3)

    torque_sq_per_leg = torques_per_leg.pow(2).sum(dim=-1)               # (N, 4)
    airborne = ~in_contact                                                # (N, 4)

    return (torque_sq_per_leg * airborne.float()).sum(dim=1)


def _leg_airborne_duration_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    cmd_threshold: float = 0.1,
    duration_threshold: float = 0.3,
) -> torch.Tensor:
    """Penalize any leg for accumulated airborne time above a threshold.

    Unlike ``feet_air_time`` (which fires only on re-contact), this term applies
    every step a foot stays airborne beyond ``duration_threshold`` seconds.
    Applied symmetrically to all four legs so the policy cannot game the penalty
    by shifting the hover exploit from rear legs to front legs.

    Observed failure modes this closes:
      - Persistent single rear-leg hover  (rear_air_ratio_moving = 58.6%)
      - Front-leg wheelbarrow gait        (front_air_ratio_moving = 36.5%)
      - Bilateral synchronised jumps      (double_*_air_ratio > 10%)
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    # current_air_time > 0 accumulates while the foot is off the ground
    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]  # (N, 4)
    excess = torch.clamp(air_time - duration_threshold, min=0.0)     # (N, 4)
    penalty = excess.sum(dim=1)                                       # (N,)
    moving = torch.linalg.norm(
        env.command_manager.get_command("base_velocity")[:, :2], dim=1
    ) > cmd_threshold
    return penalty * moving.float()


def _front_rear_support_balance_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    cmd_threshold: float,
) -> torch.Tensor:
    """Penalize front/rear support imbalance during motion.

    The current failure mode is a wheelbarrow-like gait: front legs carry most
    of the support while one or both hind legs stay airborne for long stretches.
    Velocity tracking alone does not forbid this local optimum. This term makes
    persistent front-vs-rear contact imbalance expensive while still allowing
    normal single-foot transitions and stair phases.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    front_contacts = in_contact[:, :2].sum(dim=1).float()
    rear_contacts = in_contact[:, 2:].sum(dim=1).float()
    imbalance = torch.abs(front_contacts - rear_contacts)
    moving = torch.linalg.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1) > cmd_threshold
    return imbalance * moving.float()


# ── Custom critic obs functions ───────────────────────────────────────────────

def joint_applied_torques_norm(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Applied torques normalized by the robot's joint effort limits."""
    asset = env.scene[asset_cfg.name]
    torque_limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids].clamp(min=1.0e-6)
    return torch.nan_to_num(
        asset.data.applied_torque[:, asset_cfg.joint_ids] / torque_limits,
        nan=0.0, posinf=1.0, neginf=-1.0,
    )


def joint_acc_norm(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint accelerations scaled like go2_rl_gym CTS privileged obs."""
    asset = env.scene[asset_cfg.name]
    return torch.nan_to_num(
        -asset.data.joint_acc[:, asset_cfg.joint_ids] * 1.0e-4,
        nan=0.0, posinf=1.0, neginf=-1.0,
    )


# ── Observation groups ────────────────────────────────────────────────────────

@configclass
class Go2FullScenePolicyCfg(ObsGroup):
    """Velocity-aware student observations for full-scene deploy.

    The previous 45D proprio-only student worked for open procedural stairs but
    transferred poorly to richer mixed scenes, where the controller must handle
    flat areas and stair regions under more varied contact changes. We add back
    base linear velocity so the actor can disambiguate slip/stall dynamics,
    while still keeping height_scan teacher-only:

      base_lin_vel(3) + base_ang_vel(3) + gravity(3) + cmd(3)
      + joint_pos(12) + joint_vel(12) + actions(12) = 48D
    """
    base_lin_vel      = ObsTerm(func=scaled_base_lin_vel)
    base_ang_vel      = ObsTerm(func=scaled_base_ang_vel,
                                noise=Unoise(n_min=-0.05, n_max=0.05))
    projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=scaled_velocity_commands,
                                params={"command_name": "base_velocity"})
    joint_pos         = ObsTerm(func=mdp.joint_pos_rel,
                                noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel         = ObsTerm(func=scaled_joint_vel_rel,
                                noise=Unoise(n_min=-0.075, n_max=0.075))
    actions           = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class Go2FullSceneCriticObsCfg(ObsGroup):
    """Privileged observations for the CTS teacher encoder (263D).

    Actor gets 48D velocity-aware observations; teacher keeps the full terrain
    and actuation-specific advantages:

      policy_obs(48) + foot_contacts(4) + dof_torques(12)
      + dof_acc(12) + height_scan(187) = 263D
    """
    # Clean actor-aligned observations (same order as policy, but no noise)
    base_lin_vel      = ObsTerm(func=scaled_base_lin_vel)
    base_ang_vel      = ObsTerm(func=scaled_base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=scaled_velocity_commands,
                                params={"command_name": "base_velocity"})
    joint_pos         = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel         = ObsTerm(func=scaled_joint_vel_rel)
    actions           = ObsTerm(func=mdp.last_action)
    # Privileged signals still kept teacher-only
    foot_contact_forces = ObsTerm(                                   # 4D
        func=foot_contact_forces_norm,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    dof_torques       = ObsTerm(                                     # 12D
        func=joint_applied_torques_norm,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_acc           = ObsTerm(                                     # 12D
        func=joint_acc_norm,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    height_scan       = ObsTerm(
        func=scaled_height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
    )

    def __post_init__(self):
        self.enable_corruption = False   # no noise for privileged obs
        self.concatenate_terms = True


# ── Training environment ──────────────────────────────────────────────────────
@configclass
class Go2StairTrainCfg(UnitreeGo2RoughEnvCfg):
    """Go2 stair-climbing training env using CTS.

    Inherits proper training infrastructure from UnitreeGo2RoughEnvCfg:
      - reset_base → correct terrain-relative spawning
      - reset_robot_joints → diverse initial joint states
      - UniformVelocityCommand → random heading/velocity commands
      - Domain randomisation (mass, friction)

    Adds:
      - Stair-focused mixed terrain aligned to go2_rl_gym CTS proportions
      - Full-scene actor observations aligned with deploy-time sensing
      - Privileged "critic" observation group for CTS teacher encoder
      - go2_rl_gym-style command curricula
      - Terrain-relative base-height and feet regulation rewards
    """

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 25.0

        # ── Observations: actor gets base linear velocity for scene transfer ─
        # Student sees 48D velocity-aware obs so the controller can cover flat
        # areas and stair zones in mixed scenes, while still learning terrain
        # transitions from history rather than direct height_scan.
        # Teacher keeps the full 263D privileged view.
        self.observations.policy = Go2FullScenePolicyCfg()
        self.observations.critic = Go2FullSceneCriticObsCfg()

        # ── Terrain ───────────────────────────────────────────────────────────
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=STAIRS_MIXED_TERRAIN_CFG,
            max_init_terrain_level=2,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )
        # Full-size height scanner: teacher encoder needs the wide 1.6×1.0
        # (187-ray) scan for accurate terrain value estimates.
        # Student no longer reads height scan, so pattern size doesn't affect
        # student obs dimensionality.
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground"]

        # ── Domain randomisation: wider floor friction for mixed scenes ───────
        self.events.physics_material.params["static_friction_range"]  = (0.2, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 2.0)
        self.events.physics_material.params["restitution_range"]      = (0.0, 0.5)

        # ── Commands ─────────────────────────────────────────────────────────
        self.commands.base_velocity = StairVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 5.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
                heading=(-1.57, 1.57),
            ),
        )

        # ── Curriculum ───────────────────────────────────────────────────────
        self.curriculum.terrain_levels = CurrTerm(
            func=_terrain_levels_by_command_progress,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ── Disturbances ─────────────────────────────────────────────────────
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 4.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "velocity_range": {
                    "x": (-0.4, 0.4),
                    "y": (-0.4, 0.4),
                    "yaw": (-0.6, 0.6),
                },
            },
        )

        # ── Terminations ─────────────────────────────────────────────────────
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.2},
        )

        # ── Rewards ───────────────────────────────────────────────────────────
        # ── Penalty calibration: go2_rl_gym effective values (scale × dt = scale × 0.02)
        #
        # go2_rl_gym (legged_gym) multiplies every reward by dt=0.02 each step.
        # IsaacLab does NOT.  A direct copy of go2_rl_gym scale values gives 50×
        # stronger per-step penalties.
        #
        # The IsaacLab defaults work for flat/rough terrain because the robot quickly
        # learns to avoid high-penalty states (low ω, low vz).  On STAIR terrain the
        # robot inevitably collides with step edges and stumbles at high angular
        # velocity even with a competent policy.  With IsaacLab-strength penalties:
        #
        #   ang_vel_xy=-0.05 × ω²=(5 rad/s)² = -1.25/step
        #   undesired_contacts=-0.5 (thigh/calf on stair edge, nearly every step)
        #   → net per-step reward ≈ -3 to -6, far outweighing is_alive=+0.1
        #
        # PPO then discovers that dying in ≈9 steps yields only -0.5 total, which is
        # much better than living 500 steps at -3/step = -1500 total.  The robot
        # converges to the "dying strategy": T_len collapses to 6-9 and training
        # becomes useless (and eventually NaN).
        #
        # Fix: use go2_rl_gym effective values (scale × 0.02) for ALL penalties so
        # that the per-step negative reward stays well below the positive tracking
        # signal even with bad early-training behaviour on stairs.
        #
        # flat_orientation_l2: left at default 0.0 (disabled).
        #   Go2RoughEnvCfg disables it; stair climbing requires body pitch → penalising
        #   it directly opposes the target task.
        # dof_pos_limits: go2_rl_gym uses -2.0 (effective -0.04), but enabling at
        #   high weight causes large penalties when joints start at random positions
        #   during reset, so kept at low safe weight.
        self.rewards.dof_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=-0.04,  # go2_rl_gym: -2.0 × 0.02
        )
        self.rewards.lin_vel_z_l2 = RewTerm(
            func=_curriculum_scaled_lin_vel_z_l2,
            weight=-0.04,  # go2_rl_gym: -2.0 × 0.02
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "curriculum_cfg": LIN_VEL_Z_REWARD_CURRICULUM,
            },
        )
        self.rewards.ang_vel_xy_l2.weight   = -0.001   # go2_rl_gym: -0.05 × 0.02
        self.rewards.dof_torques_l2.weight  = -2.0e-6  # go2_rl_gym: -1e-4 × 0.02
        self.rewards.dof_acc_l2.weight      = -5.0e-9  # go2_rl_gym: -2.5e-7 × 0.02
        self.rewards.action_rate_l2.weight  = -0.0002  # go2_rl_gym: -0.01 × 0.02
        self.rewards.action_smoothness = RewTerm(
            func=_action_smoothness_penalty,
            weight=-0.0002,
        )

        # Velocity tracking with go2_rl_gym dynamic sigma.
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=_track_lin_vel_xy_dynamic_exp,
            weight=1.5,
            params={
                "command_name": "base_velocity",
                "sigma_cfg": DYNAMIC_TRACKING_SIGMA_CFG,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=_track_ang_vel_z_dynamic_exp,
            weight=0.75,
            params={
                "command_name": "base_velocity",
                "sigma_cfg": DYNAMIC_TRACKING_SIGMA_CFG,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Positive air-time shaping encouraged the front-leg sync-hop exploit.
        # Reference CTS does not rely on this term, so keep it off.
        self.rewards.feet_air_time = None

        # Reference CTS does not use extra survival bonus or feet-slide penalty.
        self.rewards.feet_slide = None

        # Thigh/calf collision penalty.
        # go2_rl_gym: collision=-1.0 effective=-0.02.
        # UnitreeGo2RoughEnvCfg disables undesired_contacts entirely; we keep a tiny
        # deterrent but small enough not to cause dying strategy on stair terrain.
        self.rewards.undesired_contacts = RewTerm(
            func=_base_mdp.undesired_contacts,
            weight=-0.02,   # go2_rl_gym: -1.0 × 0.02; was -0.5 (dying strategy)
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*_thigh", ".*_calf"]),
                "threshold": 1.0,
            },
        )

        # Hip deviation penalty — go2_rl_gym original scale -0.05 (no ×dt here).
        # Prevents lateral hip sprawl / cross-leg gait on all four legs equally.
        self.rewards.hip_deviation = RewTerm(
            func=_base_mdp.joint_deviation_l1,
            weight=-0.05,   # go2_rl_gym: -0.05 (was -0.001 = go2_rl_gym×0.02, too weak)
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
        )

        # Airborne torque penalty — symmetric across all 4 legs.
        # Penalises keeping an airborne leg rigid/active; encourages the policy
        # to relax any lifted leg rather than holding it in tension. This closes
        # the hover/wheelbarrow local optima for BOTH front and rear without
        # introducing front/rear asymmetry that could merely shift the problem.
        self.rewards.airborne_torque = RewTerm(
            func=_airborne_torque_penalty,
            weight=-5e-4,   # increased from -2e-4: rear_air_moving was 0.59, target ≤0.52
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            },
        )

        # Keep the reference CTS effective weight for terrain-relative dragging.
        self.rewards.feet_regulation = RewTerm(
            func=_feet_regulation_reward,
            weight=-0.001,
            params={
                "asset_cfg":          SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg":         SceneEntityCfg("height_scanner"),
                "base_height_target": 0.38,
            },
        )

        # Avoid front/rear-specific shaping. It fought valid stair support phases
        # and was not used by either reference repository.
        self.rewards.leg_airborne_duration = None
        self.rewards.front_rear_support_balance = None

        self.rewards.stand_still_joint_posture = RewTerm(
            func=_stand_still_joint_penalty,
            weight=-0.001,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]),
                "cmd_threshold": 0.1,
                "velocity_threshold": 0.2,
            },
        )

        self.rewards.base_height = RewTerm(
            func=_curriculum_scaled_base_height_l2,
            weight=-0.02,
            params={
                "target_height": 0.38,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "curriculum_cfg": BASE_HEIGHT_REWARD_CURRICULUM,
            },
        )

        # go2_rl_gym: dof_power=-2e-5 (effective -4e-7) — penalise joint power τ·q̇.
        # dof_torques_l2 covers τ² but not the velocity-weighted part; add it.
        self.rewards.dof_power = RewTerm(
            func=_joint_power_l1,
            weight=-4e-7,   # go2_rl_gym: -2e-5 × 0.02
        )


# ── CTS training configuration ────────────────────────────────────────────────
def _make_cts_cfg(num_envs: int, max_iterations: int, experiment_name: str) -> dict:
    """Build the CTS training config dict."""
    return {
        # Architecture
        "history_length": 5,
        "policy": {
            "actor_hidden_dims":           [512, 256, 128],
            "critic_hidden_dims":          [512, 256, 128],
            "teacher_encoder_hidden_dims": [512, 256],
            "student_encoder_hidden_dims": [512, 256],
            "activation":     "elu",
            "init_noise_std": 1.0,
            "latent_dim":     32,
            "norm_type":      "l2norm",
        },
        # PPO + distillation hyperparameters
        "algorithm": {
            "value_loss_coef":              1.0,
            "use_clipped_value_loss":       True,
            "clip_param":                   0.2,
            "entropy_coef":                 0.01,
            "num_learning_epochs":          5,
            "num_mini_batches":             4,
            "learning_rate":                1e-3,
            "student_encoder_learning_rate": 1e-3,
            "schedule":        "adaptive",
            "gamma":           0.99,
            "lam":             0.95,
            "desired_kl":      0.01,
            "max_grad_norm":   1.0,
            "teacher_env_ratio": 0.75,
        },
        # Runner settings
        # go2_rl_gym note: at 4096 envs, num_steps_per_env=48 gives similar
        # sample efficiency to 8192 envs with 24 steps.
        "num_steps_per_env": 48 if num_envs <= 4096 else 24,
        "max_iterations":    max_iterations,
        "save_interval":     500,
        "experiment_name":   experiment_name,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    env_cfg = Go2StairTrainCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed           = args_cli.seed

    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO] num_obs={env.num_obs}  num_privileged_obs={env.num_privileged_obs}  "
          f"num_actions={env.num_actions}  num_envs={env.num_envs}")
    assert env.num_privileged_obs > 0, \
        "Critic obs group not detected — check Go2StairCriticObsCfg is registered."

    train_cfg = _make_cts_cfg(
        args_cli.num_envs,
        args_cli.max_iterations,
        args_cli.experiment_name,
    )
    log_dir   = os.path.abspath(
        os.path.join(
            "logs",
            "rsl_rl",
            train_cfg["experiment_name"],
            f"seed_{args_cli.seed}",
        )
    )

    runner = CTSRunner(env, train_cfg, log_dir=log_dir, device="cuda:0")

    # ── Optional checkpoint resume ────────────────────────────────────────────
    ckpt = args_cli.checkpoint
    if ckpt and os.path.isfile(ckpt):
        runner.load(ckpt)
    elif ckpt:
        print(f"[WARN] Checkpoint not found ({ckpt}), training from scratch.")

    runner.learn(
        num_learning_iterations=args_cli.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
