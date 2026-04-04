# Copyright (c) 2024, RoboVerse community
# BSD-2-Clause License

"""Train the Go2 locomotion model for stair climbing from scratch.

Root-cause fix (Version 7)
--------------------------
Previous versions inherited from UnitreeGo2CustomEnvCfg (the *deployment* env)
which has a stripped-down EventCfg with no reset_base event.  Without
reset_root_state_uniform, robots are reset to their URDF default position which
ignores the actual terrain height → they spawn in the air or inside the stair
mesh → huge collision impulse → immediate ep_len=1.

This version inherits from the official UnitreeGo2RoughEnvCfg which has all
the proper training infrastructure:
  - reset_base (reset_root_state_uniform) → correct terrain-relative spawning
  - reset_robot_joints → diverse initial joint configurations
  - add_base_mass → domain randomization
  - Proper command manager (UniformVelocityCommand)

Reward improvements (from go2_rl_gym analysis)
----------------------------------------------
  - dynamic_foot_clearance: rewards feet lifting based on terrain scan height
  - base_height_l2: keep robot at correct body height on stairs
  - is_alive: survival bonus to grow ep_len
  - All L2 penalty terms zeroed (blow up on stair contact impulses)

Terrain strategy
----------------
Mixed terrain so robots are not 100% on stairs early in training:
  15% flat + 15% rough + 10% slope + 30% pyramid_stairs + 20% pyramid_stairs_inv + 10% boxes
Curriculum starts at level 0 (easiest, ~2 cm steps) and promotes up.

Usage
-----
    python train_stairs.py --headless --num_envs 4096 --max_iterations 15000
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fine-tune Go2 for stair climbing")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=15000)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="",
)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (after app launch) ────────────────────────────────────────────────
import os
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import mdp as _base_mdp

# ── Official Go2 rough-terrain training env ───────────────────────────────────
# This base class has all proper training infrastructure:
#   reset_base / reset_robot_joints events, UniformVelocityCommand, etc.
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

from agent_cfg import unitree_go2_agent_cfg


# ── dynamic_foot_clearance_reward ─────────────────────────────────────────────
# Borrowed from unitree_go2 reference repo and go2_rl_gym insights.
# Rewards the robot for lifting feet based on height_scan terrain data so the
# robot learns to clear step edges rather than shuffling along the ground.
def _dynamic_foot_clearance_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    base_margin: float,
    max_additional: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    has_hits = False
    if hasattr(sensor.data, "ray_hits_w"):
        ray_z = sensor.data.ray_hits_w[:, :, 2]
        has_hits = True
    elif hasattr(sensor.data, "ray_hits"):
        ray_z = sensor.data.ray_hits[:, :, 2]
        has_hits = True

    if has_hits:
        ray_z = torch.nan_to_num(ray_z, nan=0.0, posinf=0.0, neginf=0.0)
        base_height = asset.data.root_pos_w[:, 2].unsqueeze(1)
        rel_height = ray_z - base_height
        max_rel, _ = torch.max(rel_height, dim=1)
        target_height = base_margin + torch.clamp(max_rel, 0.0, max_additional)
    else:
        target_height = torch.full(
            (env.num_envs,), base_margin, device=env.device,
            dtype=asset.data.body_pos_w.dtype,
        )

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_z_err = torch.square(foot_z - target_height.unsqueeze(1))
    foot_vel_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_err * foot_vel_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


# ── Mixed stair terrain config ────────────────────────────────────────────────
# Mixed terrain for from-scratch training: bias toward flat / easy rough tiles
# so the policy first learns a stable gait, then graduates to stairs through
# terrain curriculum. Row 0 = easiest (2 cm steps) → row 9 ≈ hospital stairs.
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
        # From-scratch training needs more easy tiles to first discover gait.
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
        # Primary target: pyramid stairs up and down.
        # step_height_range: row 0 ≈ 0.02 m, row 9 ≈ 0.22 m (hospital stairs).
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
        # Box obstacles (random grid heights): bridges rough terrain and stairs
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10,
            grid_width=0.45,
            grid_height_range=(0.02, 0.10),
            platform_width=2.0,
        ),
    },
)


# ── Training environment ──────────────────────────────────────────────────────
@configclass
class Go2StairTrainCfg(UnitreeGo2RoughEnvCfg):
    """Go2 stair-climbing fine-tune environment.

    Inherits from UnitreeGo2RoughEnvCfg (the official IsaacLab training config)
    to get proper training infrastructure:
      - reset_base event  → robots placed at correct terrain height after each reset
      - reset_robot_joints event → diverse initial joint states
      - UniformVelocityCommand → random heading/velocity commands during training
      - Domain randomisation (mass, friction)

    Overrides only: terrain, rewards, terminations.
    """

    def __post_init__(self):
        # Apply full rough-training defaults (events, obs, commands, rewards...)
        super().__post_init__()

        # ── Replace terrain with stair-focused mixed terrain ──────────────
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=STAIRS_MIXED_TERRAIN_CFG,
            # Start at level 0 (easiest ~2 cm steps).
            # Curriculum promotes to harder tiles based on forward velocity.
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
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground"]

        # ── Easier command distribution for from-scratch learning ─────────
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.ranges.heading = (-1.57, 1.57)

        # ── Terminations ──────────────────────────────────────────────────
        # base_contact stays at 1.0 N (inherited from UnitreeGo2RoughEnvCfg).
        # With proper reset_base, no more free-fall spawning → no spurious
        # contact impulses at reset time.
        # Add bad_orientation (~50°) as the main "fallen" signal for stairs.
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.87},
        )

        # ── Rewards: keep stair incentives, but restore enough regularization
        # for from-scratch policies to first learn a stable walk.
        self.rewards.dof_torques_l2.weight      = -1.0e-5
        self.rewards.dof_acc_l2.weight          = -2.5e-7
        self.rewards.action_rate_l2.weight      = -0.002
        self.rewards.lin_vel_z_l2.weight        = -0.5
        self.rewards.ang_vel_xy_l2.weight       = -0.05
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.dof_pos_limits.weight      = -0.2

        # Forward tracking remains the main positive signal, but less aggressive
        # than the previous from-scratch settings so the policy can discover
        # stable stance before over-optimizing for speed.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight  = 0.75

        # Keep stepping incentives modest; too much lift reward from scratch
        # tends to create hopping or flailing rather than usable gait.
        self.rewards.feet_air_time.weight = 0.15
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"

        # Small survival bonus helps early learning without dominating gait.
        self.rewards.is_alive = RewTerm(func=_base_mdp.is_alive, weight=0.1)

        # Penalize foot slip while in contact so the policy settles into a more
        # realistic walking pattern before tackling tall steps.
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )

        # Penalize thigh/calf collisions, which commonly appear when a policy
        # over-lifts or folds the legs while still unstable.
        self.rewards.undesired_contacts = RewTerm(
            func=_base_mdp.undesired_contacts,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*_thigh", ".*_calf"]
                ),
                "threshold": 1.0,
            },
        )

        # Keep hip joints near their nominal stance to reduce splayed-leg gaits.
        self.rewards.hip_deviation = RewTerm(
            func=_base_mdp.joint_deviation_l1,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
        )

        # Dynamic foot clearance: rewards lifting feet based on terrain height
        # scan data. On stairs the robot learns to step over edges instead of
        # shuffling/scraping.  body_names=".*_foot" → Go2 foot bodies.
        self.rewards.foot_clearance = RewTerm(
            func=_dynamic_foot_clearance_reward,
            weight=0.08,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "base_margin":    0.05,
                "max_additional": 0.20,
                "std":            0.10,
                "tanh_mult":      2.0,
            },
        )

        # Base height (go2_rl_gym insight): penalise crouching or over-extending.
        # Uses height_scanner to compute terrain-relative body height.
        self.rewards.base_height = RewTerm(
            func=_base_mdp.base_height_l2,
            weight=-0.5,
            params={
                "target_height": 0.36,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    env_cfg = Go2StairTrainCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Sanitize observations: height-scan rays at terrain boundaries can return
    # NaN/inf, which propagate through the network and corrupt gradients.
    _orig_step  = env.step
    _orig_reset = env.reset
    def _safe_step(actions):
        obs, rew, done, info = _orig_step(actions)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=100.0, neginf=-100.0)
        return obs, rew, done, info
    def _safe_reset():
        obs, info = _orig_reset()
        obs = torch.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        return obs, info
    env.step  = _safe_step
    env.reset = _safe_reset

    agent_cfg = dict(unitree_go2_agent_cfg)
    agent_cfg["max_iterations"]  = args_cli.max_iterations
    agent_cfg["experiment_name"] = "unitree_go2_stairs"
    agent_cfg["run_name"]        = f"hospital_stairs_v8_seed{args_cli.seed}"
    agent_cfg["save_interval"]   = 200

    # Use log_std parametrization: exp(log_std) is always positive, PPO gradients
    # can never push it to NaN or negative.  Saved checkpoints have key "log_std"
    # which is incompatible with omniverse_sim, but training saves to
    # logs/rsl_rl/unitree_go2_stairs/ (simulation loads from unitree_go2_rough)
    # so there is no conflict.
    agent_cfg["policy"] = dict(agent_cfg["policy"])
    agent_cfg["policy"]["noise_std_type"] = "log"

    # PPO hyperparameters (conservative LR for numerical stability)
    agent_cfg["algorithm"]["learning_rate"] = 3e-4
    agent_cfg["algorithm"]["desired_kl"]    = 0.01
    agent_cfg["algorithm"]["entropy_coef"]  = 0.01

    log_dir = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"])
    )

    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device="cuda:0")

    # ── Minimal single-line logging ───────────────────────────────────────
    _max_iter   = args_cli.max_iterations
    _PRINT_EVERY = 50
    _state       = {"t0": None, "start_it": None}

    def _minimal_log(self, locs, width=80, pad=35):
        import statistics
        import time as _time

        it = locs.get("it", 0)
        if _state["t0"] is None:
            _state["t0"]       = _time.time()
            _state["start_it"] = it

        done = it - _state["start_it"]

        rew_buf = locs.get("rewbuffer", None)
        len_buf = locs.get("lenbuffer", None)

        # Always write to TensorBoard every iteration (same as original RSL-RL)
        if self.log_dir is not None and rew_buf and len(rew_buf) > 0:
            self.writer.add_scalar("Train/mean_reward",         statistics.mean(rew_buf), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(len_buf), it)
            # Loss scalars
            if "mean_value_loss" in locs:
                self.writer.add_scalar("Loss/value_function",   locs["mean_value_loss"],   it)
            if "mean_surrogate_loss" in locs:
                self.writer.add_scalar("Loss/surrogate",         locs["mean_surrogate_loss"], it)
            if "mean_action_noise_std" in locs:
                self.writer.add_scalar("Policy/mean_noise_std", locs["mean_action_noise_std"], it)
            if "mean_approx_kl" in locs:
                self.writer.add_scalar("Loss/approx_kl",        locs["mean_approx_kl"],    it)
            if "learning_rate" in locs:
                self.writer.add_scalar("Loss/learning_rate",    locs["learning_rate"],      it)
            # Per-reward breakdown (Metrics/*)
            ep_infos = locs.get("ep_infos", [])
            if ep_infos:
                for key in ep_infos[-1]:
                    vals = [ep_info[key] for ep_info in ep_infos if key in ep_info]
                    if not vals:
                        continue
                    if isinstance(vals[0], torch.Tensor):
                        infotensor = torch.stack(vals)
                        if infotensor.is_floating_point():
                            self.writer.add_scalar("Metrics/" + key, infotensor.mean(), it)
                    elif isinstance(vals[0], (int, float)):
                        self.writer.add_scalar("Metrics/" + key, sum(vals) / len(vals), it)

        # Print one clean line every _PRINT_EVERY iters
        if done % _PRINT_EVERY != 0:
            return
        if not rew_buf or len(rew_buf) == 0:
            return

        mean_rew = statistics.mean(rew_buf)
        mean_len = statistics.mean(len_buf) if len_buf and len(len_buf) > 0 else float("nan")
        elapsed  = _time.time() - _state["t0"]
        steps_m  = locs.get("tot_timesteps", 0) / 1e6
        remaining = _max_iter - done
        eta_s    = (elapsed / max(done, 1)) * max(remaining, 0)
        eta_str  = f"{int(eta_s // 3600)}h{int((eta_s % 3600) // 60):02d}m"

        print(
            f"[TRAIN {done}/{_max_iter}] "
            f"reward={mean_rew:+.3f}  ep_len={mean_len:.0f}  "
            f"steps={steps_m:.2f}M  ETA={eta_str}",
            flush=True,
        )

    import types
    runner.log = types.MethodType(_minimal_log, runner)

    # ── Guard against NaN/Inf gradients corrupting parameters ─────────────
    # If any parameter gradient is NaN (from NaN observations or rewards),
    # skip the optimizer step entirely to preserve the last valid weights.
    _orig_opt_step = runner.alg.optimizer.step
    def _safe_opt_step(closure=None):
        for p in runner.alg.policy.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print("[WARN] NaN/Inf gradient — skipping optimizer step", flush=True)
                runner.alg.optimizer.zero_grad()
                return None
        return _orig_opt_step(closure)
    runner.alg.optimizer.step = _safe_opt_step

    # ── Load checkpoint with scalar-std → log-std conversion ──────────────
    # model_7850.pt was trained with noise_std_type="scalar" (key: "std").
    # We now train with noise_std_type="log" (key: "log_std") for stability.
    # Convert on load; the simulation still uses the original model_7850.pt.
    ckpt = args_cli.checkpoint
    if ckpt and os.path.isfile(ckpt):
        loaded = torch.load(ckpt, map_location="cpu")
        model_sd = loaded.get("model_state_dict", loaded)
        if "std" in model_sd and "log_std" not in model_sd:
            model_sd["log_std"] = torch.log(model_sd.pop("std").clamp(min=1e-3))
            print("[INFO] Converted checkpoint: std → log_std")
        # Load only the policy weights; start optimizer fresh to avoid
        # momentum state mismatch from the parameter rename.
        runner.alg.policy.load_state_dict(model_sd, strict=False)
        print(f"[INFO] Fine-tuning from: {ckpt}")
    else:
        print(f"[WARN] Checkpoint not found ({ckpt}), training from scratch")

    runner.learn(
        num_learning_iterations=args_cli.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
