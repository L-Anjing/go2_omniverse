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
Policy obs  (actor, 235D, with noise):
  base_lin_vel(3) + base_ang_vel(3) + proj_gravity(3) + vel_cmd(3) +
  joint_pos(12) + joint_vel(12) + actions(12) + height_scan(187)

Critic obs  (teacher encoder, 239D, clean):
  Same as above without noise + foot_contact_forces_norm(4)

Usage
-----
    python train_stairs.py --headless --num_envs 4096 --max_iterations 15000
    python train_stairs.py --headless --num_envs 4096 --max_iterations 15000 \
        --checkpoint logs/rsl_rl/unitree_go2_stairs/model_5000.pt
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Go2 stair climbing with CTS")
parser.add_argument("--num_envs",        type=int,   default=4096)
parser.add_argument("--max_iterations",  type=int,   default=15000)
parser.add_argument("--checkpoint",      type=str,   default="",
                    help="Resume from a CTS checkpoint (.pt)")
parser.add_argument("--seed",            type=int,   default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

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
    """Foot contact force magnitudes normalised to ~1.0 at body-weight load.

    Go2 weighs ~15 kg → body weight ≈ 147 N.  Each foot carries ~37 N at
    nominal stance, so scaling by 1e-3 maps a 37 N contact to 0.037 (small
    but non-zero); scaling by 1/147 maps it to ~0.25.  We use 1/147 so
    large contacts (jumps / stair landings) don't blow up the critic input.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 4, 3)
    norms  = torch.norm(forces, dim=-1)                            # (N, 4)
    return torch.nan_to_num(norms / 147.0, nan=0.0, posinf=1.0)


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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),
        # Primary target: row 0 ≈ 0.02 m steps, row 9 ≈ 0.22 m (hospital stairs)
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.02, 0.22),
            step_width=0.30,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.02, 0.22),
            step_width=0.30,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10,
            grid_width=0.45,
            grid_height_range=(0.02, 0.10),
            platform_width=2.0,
        ),
    },
)


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


def _feet_regulation_reward(
    env,
    asset_cfg: SceneEntityCfg,
    base_height_target: float,
) -> torch.Tensor:
    """Penalise high horizontal foot velocity when feet are close to the ground.

    CTS causes feet to move with reduced lateral clearance (close-distance gait).
    This reward, from go2_rl_gym, forces the swing phase to lift the foot:
      penalty = Σ_feet  ||vel_xy||² · exp(-foot_z / decay)

    Bug fix (v1.0): original decay = 0.0095 m was computed assuming foot_z is
    in the base frame.  In IsaacLab body_pos_w is world-frame, so on flat
    ground foot_z ≈ 0-0.01 m → exp(-0/0.0095) = 1 → maximum penalty for ANY
    micro-movement → robot permanently hovers feet to escape penalty → phantom
    diagonal trot even at zero command.

    Fixes applied:
    1. decay = 0.10 m (world-frame: penalty fades from ground to ~30 cm above).
    2. Gate by velocity command: only active when commanded to move (||cmd_xy|| > 0.1).
       This prevents the standing-hover behaviour entirely.
    """
    asset   = env.scene[asset_cfg.name]
    foot_z  = torch.nan_to_num(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2], nan=0.0)   # (N, 4)
    vel_xy  = torch.nan_to_num(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], nan=0.0)  # (N, 4, 2)
    vel_sq  = vel_xy.pow(2).sum(dim=-1)                               # (N, 4)
    decay   = 0.10                                                    # 10 cm world-frame
    penalty = (vel_sq * torch.exp(-foot_z / decay)).sum(dim=-1)      # (N,)
    # Only penalise when the robot is commanded to move (mirrors feet_air_time gate)
    cmd_xy  = env.command_manager.get_command("base_velocity")[:, :2]
    moving  = (torch.norm(cmd_xy, dim=1) > 0.1).float()
    return penalty * moving


# ── Custom critic obs functions ───────────────────────────────────────────────

def joint_applied_torques_norm(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Applied joint torques normalised by nominal peak torque (~50 N·m for Go2)."""
    asset = env.scene[asset_cfg.name]
    return torch.nan_to_num(
        asset.data.applied_torque[:, asset_cfg.joint_ids] / 50.0,
        nan=0.0, posinf=1.0, neginf=-1.0,
    )


def joint_acc_norm(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint accelerations normalised by a typical peak (~100 rad/s²)."""
    asset = env.scene[asset_cfg.name]
    return torch.nan_to_num(
        asset.data.joint_acc[:, asset_cfg.joint_ids] / 100.0,
        nan=0.0, posinf=1.0, neginf=-1.0,
    )


# ── Observation groups ────────────────────────────────────────────────────────

@configclass
class Go2StairPolicyCfg(ObsGroup):
    """Student / policy observations (45D, noise-corrupted).

    Matches go2_rl_gym CTS design exactly:
      - NO base_lin_vel  (not directly measurable on real robot)
      - NO height_scan   (student must infer terrain from motion history)
    This forces the student encoder to encode terrain implicitly from 5-frame
    history, rather than reading it from a sensor.  The teacher (critic) gets
    the full privileged signals including height_scan.
    """
    base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,
                                noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands,
                                params={"command_name": "base_velocity"})
    joint_pos         = ObsTerm(func=mdp.joint_pos_rel,
                                noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel         = ObsTerm(func=mdp.joint_vel_rel,
                                noise=Unoise(n_min=-1.5, n_max=1.5))
    actions           = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class Go2StairCriticObsCfg(ObsGroup):
    """Privileged observations for the CTS teacher encoder (263D).

    Matches go2_rl_gym: policy_obs(45) + base_lin_vel(3) + foot_contacts(4)
    + dof_torques(12) + dof_acc(12) + height_scan(187) = 263D.

    All terms noise-free so the teacher provides accurate value estimates.
    The 218D extra vs student is the information the student encoder must
    learn to approximate from motion history via distillation.
    """
    # Clean proprioception (matches policy group but no noise)
    base_ang_vel      = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands,
                                params={"command_name": "base_velocity"})
    joint_pos         = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel         = ObsTerm(func=mdp.joint_vel_rel)
    actions           = ObsTerm(func=mdp.last_action)
    # Privileged signals (not available to student at deploy time)
    base_lin_vel      = ObsTerm(func=mdp.base_lin_vel)              # 3D
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
    # Clean full height scan (187D) — THE key terrain signal for stairs
    height_scan       = ObsTerm(                                     # 187D
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
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
      - Stair-focused mixed terrain (graduated 2 cm → 22 cm steps)
      - Privileged "critic" observation group for CTS teacher encoder
      - Foot clearance reward for stair-edge stepping
      - Base-height and orientation regularization
    """

    def __post_init__(self):
        super().__post_init__()

        # ── Observations: override policy + inject critic ─────────────────────
        # Student sees 45D proprioceptive-only obs (no height_scan, no lin_vel).
        # Teacher sees 263D privileged obs (full height scan + extra signals).
        # This matches go2_rl_gym CTS design and gives the teacher enough
        # information advantage for the distillation to be meaningful.
        self.observations.policy = Go2StairPolicyCfg()
        self.observations.critic = Go2StairCriticObsCfg()

        # ── Terrain ───────────────────────────────────────────────────────────
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=STAIRS_MIXED_TERRAIN_CFG,
            max_init_terrain_level=0,
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

        # ── Domain randomisation: wider friction for hospital floors ──────────
        self.events.physics_material.params["static_friction_range"]  = (0.2, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 2.0)
        self.events.physics_material.params["restitution_range"]      = (0.0, 0.5)

        # ── Commands ─────────────────────────────────────────────────────────
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.rel_standing_envs     = 0.10
        # go2_rl_gym v0.1.4: narrow early-training range to [-0.5, 0.5] to
        # prevent high-speed gait artifacts (high leg lifts at v > 0.8 m/s).
        self.commands.base_velocity.ranges.lin_vel_x      = (-0.5, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y      = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z      = (-0.6, 0.6)
        self.commands.base_velocity.ranges.heading        = (-1.57, 1.57)

        # ── Terminations ─────────────────────────────────────────────────────
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.87},  # ~50°
        )

        # ── Rewards ───────────────────────────────────────────────────────────
        # Regularisation penalties
        # torques: match go2_rl_gym (-1e-4) — 10× stronger than default
        # reduces energy waste and jerky motion on stairs
        self.rewards.dof_torques_l2.weight       = -1.0e-4
        self.rewards.dof_acc_l2.weight           = -2.5e-7
        self.rewards.action_rate_l2.weight       = -0.01
        self.rewards.lin_vel_z_l2.weight         = -2.0
        self.rewards.ang_vel_xy_l2.weight        = -0.05
        self.rewards.flat_orientation_l2.weight  = -0.5
        self.rewards.dof_pos_limits.weight       = -2.0

        # Velocity tracking (main task signal)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight  = 0.75

        # Stepping incentive (modest – avoids hopping from scratch)
        self.rewards.feet_air_time.weight = 0.15
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"

        # Survival bonus (grows episode length in early training)
        self.rewards.is_alive = RewTerm(func=_base_mdp.is_alive, weight=0.1)

        # Foot slip penalty
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg":  SceneEntityCfg("robot",          body_names=".*_foot"),
            },
        )

        # Thigh/calf collision penalty
        self.rewards.undesired_contacts = RewTerm(
            func=_base_mdp.undesired_contacts,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*_thigh", ".*_calf"]),
                "threshold": 1.0,
            },
        )

        # Hip deviation penalty — keeps hips near default position.
        # go2_rl_gym explicitly added this to fix CTS-induced close-feet gait.
        self.rewards.hip_deviation = RewTerm(
            func=_base_mdp.joint_deviation_l1,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
        )

        # All-joint default deviation (similar_to_default).
        # go2_rl_gym v0.1.4: adding similar_to_default / hip_to_default fixes the
        # phantom diagonal trot: without this, the policy has no cost for holding
        # joints away from default while standing still.
        # Weight -0.1 (go2_rl_gym ablation range 0.01–0.05 for hip-only;
        # we use -0.1 for all joints since IsaacLab L1 sums over 12 joints).
        self.rewards.joint_deviation_all = RewTerm(
            func=_base_mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Feet regulation — go2_rl_gym's specific CTS abnormal-gait fix.
        # Penalises horizontal foot velocity when feet are near the ground.
        # Forces the robot to lift feet during swing phase instead of shuffling.
        # v1.0 fix: added command gating + world-frame decay (see function docstring).
        self.rewards.feet_regulation = RewTerm(
            func=_feet_regulation_reward,
            weight=-0.05,
            params={
                "asset_cfg":          SceneEntityCfg("robot", body_names=".*_foot"),
                "base_height_target": 0.38,
            },
        )

        # Foot contact force penalty — prevents stomping/hard landings.
        # go2_rl_gym v0.1.4: "真机上有明显的跺脚动作, 加上feet_contact_forces惩罚
        # 接触力大小, 设置接触力阈值大小为go2的重量15*9.8=147, 系数为-1".
        self.rewards.foot_contact_force = RewTerm(
            func=foot_contact_force_penalty,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold":  147.0,
            },
        )

        # Dynamic foot clearance (core stair-climbing incentive)
        self.rewards.foot_clearance = RewTerm(
            func=_dynamic_foot_clearance_reward,
            weight=0.03,
            params={
                "sensor_cfg":    SceneEntityCfg("height_scanner"),
                "asset_cfg":     SceneEntityCfg("robot", body_names=".*_foot"),
                "base_margin":   0.03,
                "max_additional": 0.12,
                "std":           0.10,
                "tanh_mult":     2.0,
            },
        )

        # Terrain-relative base height (go2_rl_gym insight: 0.38 m target)
        self.rewards.base_height = RewTerm(
            func=_base_mdp.base_height_l2,
            weight=-0.5,
            params={
                "target_height": 0.38,
                "sensor_cfg":    SceneEntityCfg("height_scanner"),
            },
        )


# ── CTS training configuration ────────────────────────────────────────────────
def _make_cts_cfg(num_envs: int, max_iterations: int) -> dict:
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
        "save_interval":     200,
        "experiment_name":   "unitree_go2_stairs",
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

    train_cfg = _make_cts_cfg(args_cli.num_envs, args_cli.max_iterations)
    log_dir   = os.path.abspath(
        os.path.join("logs", "rsl_rl", train_cfg["experiment_name"]))

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
