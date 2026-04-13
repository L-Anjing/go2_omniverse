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


"""Script to play a checkpoint if an RL agent from RSL-RL."""

from __future__ import annotations


"""Launch Isaac Sim Simulator first."""
import argparse
import copy
import re
from pathlib import Path
from isaaclab.app import AppLauncher


from core import cli_args
import time
import os
import threading


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
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
    "--action_smoothing",
    type=float,
    default=0.15,
    help=(
        "EMA factor applied to policy actions before env.step(). "
        "0 disables smoothing; larger values keep more of the previous action."
    ),
)
parser.add_argument(
    "--zero_cmd_stance_blend",
    type=float,
    default=0.35,
    help=(
        "When |cmd| is near zero, blend this fraction of the action back to 0 "
        "(default joint pose) to suppress stand-still twitching."
    ),
)
parser.add_argument(
    "--zero_cmd_threshold",
    type=float,
    default=0.05,
    help="Command-norm threshold below which zero-command stance blending is enabled.",
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
import omni.appwindow
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


from configs.agent_cfg import unitree_go2_agent_cfg, unitree_g1_agent_cfg
from configs.custom_rl_env import (
    UnitreeGo2CustomEnvCfg,
    Go2StairDeployCfg,
    Go2FullSceneDeployCfg,
    G1RoughEnvCfg,
)
import configs.custom_rl_env as custom_rl_env

from assets.robots.copter.config import CRAZYFLIE_CFG
from isaaclab.assets import Articulation, AssetBaseCfg


from core.omnigraph import create_front_cam_omnigraph


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_agent_cfg(base_cfg: RslRlOnPolicyRunnerCfg) -> RslRlOnPolicyRunnerCfg:
    """Return a copy of the base agent config with CLI overrides applied."""
    agent_cfg = copy.deepcopy(base_cfg)

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

    stu_w = sorted(k for k in state_dict if k.startswith("student_encoder.") and k.endswith(".weight"))
    history_length = state_dict[stu_w[0]].shape[1] // num_actor_obs

    return {
        "latent_dim": latent_dim,
        "num_actor_obs": num_actor_obs,
        "num_critic_obs": num_critic_obs,
        "num_actions": num_actions,
        "history_length": history_length,
    }


def load_cts_policy(checkpoint_path: str, env, device: str):
    """Load a CTS checkpoint and return a student-only inference callable.

    All model dimensions are inferred directly from the checkpoint weights so
    the function is robust to different training configurations:
      - latent_dim   : output of last linear in teacher_encoder
      - num_actor_obs: actor first-layer input minus latent_dim
      - num_critic_obs: critic first-layer input minus latent_dim
      - num_actions  : actor last-layer output
      - history_length: student_encoder first-layer input / num_actor_obs
    """
    from rsl_rl_cts import ActorCriticCTS

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    dims = _infer_cts_checkpoint_dims(checkpoint_path, map_location=device)
    latent_dim = dims["latent_dim"]
    num_actor_obs = dims["num_actor_obs"]
    num_actions = dims["num_actions"]
    num_critic_obs = dims["num_critic_obs"]
    history_length = dims["history_length"]

    print(f"[CTS] Checkpoint dims: actor_obs={num_actor_obs}, critic_obs={num_critic_obs}, "
          f"actions={num_actions}, latent={latent_dim}, history={history_length}")
    print(f"[CTS] Env reports:     num_obs={env.num_obs}, num_actions={env.num_actions}")
    if env.num_obs != num_actor_obs:
        raise RuntimeError(
            f"Observation dimension mismatch: checkpoint expects {num_actor_obs}D obs "
            f"but env produces {env.num_obs}D.  Ensure the deployment env cfg matches "
            f"the training env cfg (height_scanner size, observation terms, etc.)."
        )

    model = ActorCriticCTS(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        num_envs=env.num_envs,
        history_length=history_length,
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
# These defaults are safe for general Go2 PPO play. CTS stair checkpoints are
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


def _stabilize_policy_actions(
    raw_actions: torch.Tensor,
    prev_actions: torch.Tensor | None,
    cmd_tensor: torch.Tensor,
) -> torch.Tensor:
    actions = raw_actions

    smoothing = min(max(args_cli.action_smoothing, 0.0), 0.999)
    if prev_actions is not None and smoothing > 0.0:
        actions = prev_actions * smoothing + raw_actions * (1.0 - smoothing)

    zero_cmd_blend = min(max(args_cli.zero_cmd_stance_blend, 0.0), 1.0)
    zero_cmd_threshold = max(args_cli.zero_cmd_threshold, 0.0)
    if zero_cmd_blend > 0.0:
        zero_mask = torch.linalg.vector_norm(cmd_tensor, dim=1) <= zero_cmd_threshold
        if torch.any(zero_mask):
            actions = actions.clone()
            actions[zero_mask] *= 1.0 - zero_cmd_blend

    return actions


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
        if args_cli.custom_env == "warehouse":
            cfg_scene = sim_utils.UsdFileCfg(usd_path=str(REPO_ROOT / "assets" / "env" / "warehouse.usd"))
            cfg_scene.func("/World/warehouse", cfg_scene, translation=(0.0, 0.0, 0.0))

        if args_cli.custom_env == "office":
            cfg_scene = sim_utils.UsdFileCfg(usd_path=str(REPO_ROOT / "assets" / "env" / "office.usd"))
            cfg_scene.func("/World/office", cfg_scene, translation=(0.0, 0.0, 0.0))

        if args_cli.custom_env == "hospital":
            # Absolute path to the hospital asset bundle.
            # Materials/ and Props/ are referenced as relative layers inside
            # hospital.usd and are resolved automatically by the USD runtime.
            HOSPITAL_USD = (
                "/media/user/data1/isaac-sim-assets/merged/Assets/Isaac/4.5/"
                "Isaac/Environments/Hospital/hospital.usd"
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
    num_planes = _deactivate_hospital_collision_planes(stage)
    if num_planes == 0:
        print("[WARN] Hospital CollisionPlane not found during spawn — check USD prim path")
    else:
        print(f"[INFO] Hospital spawned and {num_planes} duplicate ground prim(s) deactivated")
    return prim


def cmd_vel_cb(msg, num_robot):
    x = msg.linear.x
    y = msg.linear.y
    z = msg.angular.z
    custom_rl_env.base_command[str(num_robot)] = [x, y, z]


def add_cmd_sub(num_envs):
    node_test = rclpy.create_node("position_velocity_publisher")
    for i in range(num_envs):
        node_test.create_subscription(
            Twist, f"robot{i}/cmd_vel", lambda msg, i=i: cmd_vel_cb(msg, str(i)), 10
        )
    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node_test,), daemon=True)
    thread.start()


def specify_cmd_for_robots(numv_envs):
    for i in range(numv_envs):
        custom_rl_env.base_command[str(i)] = [0, 0, 0]


def run_sim():
    if args_cli.cmd_source == "keyboard":
        # Only subscribe to keyboard teleop in manual mode so it never races
        # with the navigation stack over the same base_command buffer.
        _input = carb.input.acquire_input_interface()
        _appwindow = omni.appwindow.get_default_app_window()
        _keyboard = _appwindow.get_keyboard()
        _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    """Play with RSL-RL agent."""
    # ── Resolve agent cfg and checkpoint path FIRST ───────────────────────────
    # We must know the checkpoint type before calling gym.make() because CTS
    # checkpoints use dedicated deploy envs whose actor observation size can
    # differ from the default PPO play env (for example 45D stair-student or
    # 48D full-scene student).
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
        f"experiment={agent_cfg['experiment_name']}, "
        f"checkpoint={agent_cfg['load_checkpoint']}, "
        f"noise_std_type={agent_cfg['policy'].get('noise_std_type', 'default')}"
    )

    # ── Choose env cfg to match the checkpoint's observation space ────────────
    _cts = _is_cts_checkpoint(resume_path)
    _cts_dims = _infer_cts_checkpoint_dims(resume_path) if _cts else None
    if args_cli.robot == "g1":
        env_cfg = G1RoughEnvCfg()
    elif _cts:
        actor_obs_dim = _cts_dims["num_actor_obs"]
        if actor_obs_dim == 45:
            env_cfg = Go2StairDeployCfg()
            print("[INFO] CTS checkpoint detected — using Go2StairDeployCfg (45D student obs)")
        elif actor_obs_dim == 48:
            env_cfg = Go2FullSceneDeployCfg()
            print("[INFO] CTS checkpoint detected — using Go2FullSceneDeployCfg (48D velocity-aware student obs)")
        else:
            raise RuntimeError(
                f"Unsupported CTS actor observation size {actor_obs_dim}. "
                "Add a matching deploy env cfg before running this checkpoint."
            )
    else:
        env_cfg = UnitreeGo2CustomEnvCfg()

    global CMD_LIN_VEL, CMD_ANG_VEL
    if _cts and args_cli.robot != "g1":
        actor_obs_dim = _cts_dims["num_actor_obs"]
        if actor_obs_dim == 45:
            CMD_LIN_VEL = 0.5
            CMD_ANG_VEL = 0.6
            print(
                "[INFO] CTS stair deploy command limits enabled: "
                f"lin={CMD_LIN_VEL:.1f} m/s, yaw={CMD_ANG_VEL:.1f} rad/s"
            )
        elif actor_obs_dim == 48:
            CMD_LIN_VEL = 1.0
            CMD_ANG_VEL = 1.0
            print(
                "[INFO] CTS full-scene deploy command limits enabled: "
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

    # --- Hospital: inject BEFORE gym.make() via custom spawner -----------------
    # _spawn_hospital_usd deactivates the hospital's internal CollisionPlane
    # immediately after loading the USD, so PhysX never sees two coincident
    # ground planes during its first broad-phase reset (which happens inside
    # gym.make).  This gives correct load order (scene + robot appear together)
    # and avoids the base_contact-every-step bug from post-gym.make loading.
    if args_cli.custom_env == "hospital":
        HOSPITAL_USD = str(REPO_ROOT / "assets" / "env" / "hospital_labeled.usd")
        hospital_spawn_cfg = sim_utils.UsdFileCfg(usd_path=HOSPITAL_USD)
        hospital_spawn_cfg.func = _spawn_hospital_usd
        env_cfg.scene.custom_env = AssetBaseCfg(
            prim_path="/World/hospital",
            spawn=hospital_spawn_cfg,
        )

    # create isaac environment (scene load + robot spawn; physics starts here)
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Warehouse / office: still loaded post-gym.make (visual-only assets,
    # no robot-overlapping collision geometry).
    if args_cli.custom_env in {"warehouse", "office"}:
        setup_custom_env()

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
    add_camera(env_cfg.scene.num_envs, args_cli.robot)
    # add_copter_camera()

    # create ros2 camera stream omnigraph (rgb + semantic_segmentation)
    for i in range(env_cfg.scene.num_envs):
        create_front_cam_omnigraph(i)

    # First observation collection — triggers the first policy-driven step.
    # Everything above must be fully on stage before this call.
    obs, _ = env.get_observations()
    prev_actions = None

    # Log initial observation stats before any policy step
    print(f"[DBG init] obs min={obs.min().item():.4f}  max={obs.max().item():.4f}  "
          f"mean={obs.mean().item():.4f}  finite={torch.isfinite(obs).all().item()}")
    print(f"[DBG init] obs={obs[0].tolist()}")

    # simulate environment
    _dbg_step = 0
    _dbg_resets = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            raw_actions = policy(obs)
            cmd_tensor = _base_command_tensor(env_cfg.scene.num_envs, raw_actions.device)
            actions = _stabilize_policy_actions(raw_actions, prev_actions, cmd_tensor)
            prev_actions = actions
            # env stepping
            obs, _, terminated, _ = env.step(actions)
            _dbg_resets += terminated.sum().item()
            pub_robo_data_ros2(
                args_cli.robot,
                env_cfg.scene.num_envs,
                base_node,
                env,
                annotator_lst,
                base_commands=custom_rl_env.base_command,
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
                obs_cmd     = obs[0, 9:12].tolist()
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
                print(
                    f"[DBG step={_dbg_step}]"
                    f"  cmd={custom_rl_env.base_command}"
                    f"  obs_cmd={[round(v,3) for v in obs_cmd]}"
                    f"  pos={[round(v,3) for v in base_pos]}"
                    f"  quat={[round(v,3) for v in base_quat]}"
                    f"  quat_norm={quat_norm:.4f}"
                    f"  pos_z={base_pos[2]:.4f}"
                    f"  vel_xy={[round(v,3) for v in base_vel[:2]]}"
                    f"  max_action={act_max:.4f}"
                    f"  max_torque={max_torque:.4f}"
                    f"  finite={state_finite}"
                    f"  total_resets={int(_dbg_resets)}"
                    + _beam_info
                )
                if _dbg_step <= 5:
                    print(f"[DBG step={_dbg_step}] actions={[round(v,4) for v in actions[0].tolist()]}")
                    print(f"[DBG step={_dbg_step}] obs={[round(v,4) for v in obs[0].tolist()]}")

    env.close()
