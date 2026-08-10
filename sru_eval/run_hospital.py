#!/usr/bin/env python3
"""Run the official recurrent SRU policy with B2W in the Hospital scene."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--stairs", action="store_true")
parser.add_argument("--pct-stairs", action="store_true")
parser.add_argument("--output", type=Path)
parser.add_argument("--config-only", action="store_true")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1:
    parser.error("--episodes must be at least 1")
args.enable_cameras = True
if args.output is None:
    mode = "pct_stairs" if args.pct_stairs else "stairs" if args.stairs else "normal"
    args.output = Path(f"logs/sru_hospital_{mode}_metrics.csv")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

if str(args.device).startswith("cuda"):
    torch.cuda.set_device(args.device)

import isaaclab_tasks  # noqa: F401,E402

# sru-navigation-sim targets Isaac Lab 2.1.1. The local 2.1.0 RayCaster
# behaves identically for this use but does not yet accept the declarative
# ray_alignment field used only by the critic height scanner.
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import math as math_utils

if "ray_alignment" not in RayCasterCfg.__dataclass_fields__:
    _ray_caster_init = RayCasterCfg.__init__

    def _ray_caster_init_210(self, *values, ray_alignment=None, **kwargs):
        _ray_caster_init(self, *values, **kwargs)

    RayCasterCfg.__init__ = _ray_caster_init_210

import isaaclab_nav_task  # noqa: F401,E402

from isaaclab_nav_task.navigation import NavigationEnv
from sru_eval.hospital_env_cfg import (
    B2WSRUHospitalEnvCfg,
    validate_hospital_inputs,
)


def reset_b2w_to_sampled_spawn(env, command):
    """Apply the command sampler's pose after Isaac Lab's reset events."""
    robot = env.scene["robot"]
    root_state = robot.data.default_root_state[:1].clone()
    spawn = command.spawn_position_world[:1]
    root_state[:, :2] = spawn[:, :2]
    root_state[:, 2] += spawn[:, 2]
    yaw = command.spawn_heading_world[:1]
    zeros = torch.zeros_like(yaw)
    yaw_quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    root_state[:, 3:7] = math_utils.quat_mul(root_state[:, 3:7], yaw_quat)
    root_state[:, 7:13] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:13])
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    env.sim.render()
    command.previous_position[:] = robot.data.root_pos_w
    command.initial_distance_to_goal[:] = torch.linalg.vector_norm(
        robot.data.root_pos_w - command.goal_position_world, dim=1
    )
    command.closest_distance_to_goal[:] = command.initial_distance_to_goal
    command._update_command()
    return env.observation_manager.compute()


def main() -> int:
    print(f"[sru] Python: {sys.executable}", flush=True)
    print(f"[sru] Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', '<none>')}", flush=True)
    if args.stairs:
        os.environ["SRU_STAIRS_EPISODES"] = "1"
    if args.pct_stairs:
        os.environ["SRU_STAIRS_EPISODES"] = "1"
        os.environ["SRU_PCT_STAIRS_EPISODES"] = "1"
    print("[sru] Building B2W Hospital environment...", flush=True)
    cfg = B2WSRUHospitalEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    if args.device == "cpu":
        # Upstream initializes this global encoder on CUDA during config setup.
        # Keep it aligned with the requested environment device for CPU PhysX.
        from isaaclab_nav_task.navigation.mdp import observations as nav_observations

        nav_observations.DEPTH_NOISE_GENERATOR = (
            nav_observations.DEPTH_NOISE_GENERATOR.to("cpu")
        )
        nav_observations.JIT_DEPTH_NOISE_GENERATOR = (
            nav_observations.DEPTH_NOISE_GENERATOR
        )
    if args.config_only:
        inputs = validate_hospital_inputs(
            stairs=args.stairs, pct_stairs=args.pct_stairs
        )
        print(f"[sru] Hospital inputs: {inputs}", flush=True)
        print("[sru] Config-only check passed.", flush=True)
        return 0
    env = NavigationEnv(cfg=cfg)
    policy_path = Path(os.environ["SRU_NAV_POLICY_TORCHSCRIPT"])
    print(f"[sru] Loading recurrent policy: {policy_path}", flush=True)
    policy = torch.jit.load(str(policy_path), map_location=env.device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    try:
        for episode in range(args.episodes):
            print(f"[sru] Reset episode {episode + 1}/{args.episodes}", flush=True)
            obs, _ = env.reset()
            command = env.command_manager._terms["robot_goal"]
            obs = reset_b2w_to_sampled_spawn(env, command)
            start = env.scene["robot"].data.root_pos_w[0].clone()
            expected_start = command.spawn_position_world[0].clone()
            expected_start[2] += env.scene["robot"].data.default_root_state[0, 2]
            spawn_error = float(torch.linalg.vector_norm(start - expected_start))
            if spawn_error > 0.15:
                raise RuntimeError(
                    f"B2W reset missed sampled spawn by {spawn_error:.3f} m"
                )
            initial = float(command.initial_distance_to_goal[0])
            episode_goal = command.goal_position_world[0].clone()
            path_length = 0.0
            previous = start
            success = False
            reason = "timeout"

            remaining = initial
            for step in range(env.max_episode_length):
                pre_step_position = env.scene["robot"].data.root_pos_w[0].clone()
                pre_step_remaining = float(torch.linalg.vector_norm(
                    pre_step_position[:2] - episode_goal[:2]
                ))
                policy_obs = obs["policy"]
                with torch.inference_mode():
                    raw_action = policy(policy_obs, step == 0)
                obs, _, terminated, truncated, _ = env.step(
                    raw_action
                )
                if bool(terminated[0]) or bool(truncated[0]):
                    # ManagerBasedRLEnv has already reset this environment here.
                    # Keep the last pre-reset sample instead of mixing episodes.
                    remaining = pre_step_remaining
                    active_terms = [
                        name
                        for name, values in env.termination_manager.get_active_iterable_terms(0)
                        if values[0] > 0.5
                    ]
                    if "early_termination" in active_terms or pre_step_remaining <= 0.5:
                        success, reason = True, "goal"
                    elif "time_out" in active_terms or step + 1 >= env.max_episode_length:
                        reason = "timeout"
                    elif active_terms:
                        reason = "+".join(active_terms)
                    elif bool(terminated[0]):
                        reason = "terminated"
                    else:
                        reason = "truncated"
                    break
                position = env.scene["robot"].data.root_pos_w[0]
                path_length += float(torch.linalg.vector_norm(position - previous))
                previous = position.clone()
                remaining = float(torch.linalg.vector_norm(
                    position[:2] - episode_goal[:2]
                ))
                if remaining <= 0.5:
                    success, reason = True, "goal"
                    break

            spl = (initial / max(initial, path_length)) if success else 0.0
            row = {
                "episode": episode,
                "stairs": int(args.stairs),
                "success": int(success),
                "reason": reason,
                "initial_distance_m": round(initial, 3),
                "spawn_error_m": round(spawn_error, 4),
                "final_distance_m": round(remaining, 3),
                "path_length_m": round(path_length, 3),
                "spl": round(spl, 4),
                "steps": step + 1,
                "spawn_x": round(float(start[0]), 3),
                "spawn_y": round(float(start[1]), 3),
                "spawn_z": round(float(start[2]), 3),
                "goal_x": round(float(episode_goal[0]), 3),
                "goal_y": round(float(episode_goal[1]), 3),
                "goal_z": round(float(episode_goal[2]), 3),
            }
            anchor = getattr(command, "last_pct_anchor", None)
            row["pct_anchor"] = (
                "" if anchor is None else ",".join(f"{float(value):.3f}" for value in anchor)
            )
            rows.append(row)
            print(f"[sru] {row}", flush=True)
    finally:
        env.close()

    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    success_count = sum(row["success"] for row in rows)
    summary = {
        "episodes": len(rows),
        "successes": success_count,
        "success_rate": round(success_count / len(rows), 4),
        "mean_spl": round(sum(row["spl"] for row in rows) / len(rows), 4),
        "mean_initial_distance_m": round(
            sum(row["initial_distance_m"] for row in rows) / len(rows), 3
        ),
        "mean_final_distance_m": round(
            sum(row["final_distance_m"] for row in rows) / len(rows), 3
        ),
        "mean_path_length_m": round(
            sum(row["path_length_m"] for row in rows) / len(rows), 3
        ),
        "mean_abs_elevation_change_m": round(
            sum(abs(row["goal_z"] - row["spawn_z"]) for row in rows) / len(rows), 3
        ),
    }
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[sru] Metrics written to {args.output}")
    print(f"[sru] Summary written to {summary_path}: {summary}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
