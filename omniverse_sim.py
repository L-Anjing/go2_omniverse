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
from isaaclab.app import AppLauncher


import cli_args
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
from ros2 import (
    RobotBaseNode,
    add_camera,
    add_copter_camera,
    add_rtx_lidar,
    pub_robo_data_ros2
)
from geometry_msgs.msg import Twist


from agent_cfg import unitree_go2_agent_cfg, unitree_g1_agent_cfg
from custom_rl_env import UnitreeGo2CustomEnvCfg, G1RoughEnvCfg
import custom_rl_env

from robots.copter.config import CRAZYFLIE_CFG
from isaaclab.assets import Articulation, AssetBaseCfg


from omnigraph import create_front_cam_omnigraph


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


def sub_keyboard_event(event, *args, **kwargs) -> bool:

    if len(custom_rl_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "W":
                custom_rl_env.base_command["0"] = [1, 0, 0]
            if event.input.name == "S":
                custom_rl_env.base_command["0"] = [-1, 0, 0]
            if event.input.name == "A":
                custom_rl_env.base_command["0"] = [0, 1, 0]
            if event.input.name == "D":
                custom_rl_env.base_command["0"] = [0, -1, 0]
            if event.input.name == "Q":
                custom_rl_env.base_command["0"] = [0, 0, 1]
            if event.input.name == "E":
                custom_rl_env.base_command["0"] = [0, 0, -1]

            if len(custom_rl_env.base_command) > 1:
                if event.input.name == "I":
                    custom_rl_env.base_command["1"] = [1, 0, 0]
                if event.input.name == "K":
                    custom_rl_env.base_command["1"] = [-1, 0, 0]
                if event.input.name == "J":
                    custom_rl_env.base_command["1"] = [0, 1, 0]
                if event.input.name == "L":
                    custom_rl_env.base_command["1"] = [0, -1, 0]
                if event.input.name == "U":
                    custom_rl_env.base_command["1"] = [0, 0, 1]
                if event.input.name == "O":
                    custom_rl_env.base_command["1"] = [0, 0, -1]
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
            cfg_scene = sim_utils.UsdFileCfg(usd_path="env/warehouse.usd")
            cfg_scene.func("/World/warehouse", cfg_scene, translation=(0.0, 0.0, 0.0))

        if args_cli.custom_env == "office":
            cfg_scene = sim_utils.UsdFileCfg(usd_path="env/office.usd")
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
            col_plane = stage.GetPrimAtPath("/World/hospital/Root/CollisionPlane")
            if col_plane and col_plane.IsValid():
                col_plane.SetActive(False)
                print("[INFO] Hospital CollisionPlane deactivated (duplicate of /World/ground)")
            else:
                print("[WARN] Hospital CollisionPlane prim not found — check path")
            print("[INFO] Hospital environment loaded at /World/hospital")

    except Exception as e:
        print(
            f"Error loading custom environment ({e}). "
            "For warehouse/office, download envs from: "
            "https://drive.google.com/drive/folders/1vVGuO1KIX1K6mD6mBHDZGm9nk2vaRyj3?usp=sharing"
        )


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
    # parse configuration

    env_cfg = UnitreeGo2CustomEnvCfg()

    if args_cli.robot == "g1":
        env_cfg = G1RoughEnvCfg()

    # TODO need to think about better copter integration.
    # copter_cfg = CRAZYFLIE_CFG
    # copter_cfg.spawn.func(
    #     "/World/Crazyflie/Robot_1", copter_cfg.spawn, translation=(1.5, 0.5, 2.42)
    # )

    # # create handles for the robots
    # copter = Articulation(copter_cfg.replace(prim_path="/World/Crazyflie/Robot.*"))

    # add N robots to env
    env_cfg.scene.num_envs = args_cli.robot_amount

    specify_cmd_for_robots(env_cfg.scene.num_envs)

    agent_cfg: RslRlOnPolicyRunnerCfg = resolve_agent_cfg(unitree_go2_agent_cfg)

    if args_cli.robot == "g1":
        agent_cfg = resolve_agent_cfg(unitree_g1_agent_cfg)

    # --- Hospital: inject into scene config BEFORE gym.make() ----------------
    # PhysX composes all collision geometry during scene setup (inside gym.make).
    # Loading hospital AFTER gym.make adds collision meshes to a running PhysX,
    # which triggers penetration-correction forces every step → base_contact
    # terminates every episode. Adding it as an AssetBaseCfg ensures it is in
    # stage before the first physics broad-phase.
    if args_cli.custom_env == "hospital":
        HOSPITAL_USD = (
            "/media/user/data1/isaac-sim-assets/merged/Assets/Isaac/4.5/"
            "Isaac/Environments/Hospital/hospital.usd"
        )
        env_cfg.scene.custom_env = AssetBaseCfg(
            prim_path="/World/hospital",
            spawn=sim_utils.UsdFileCfg(usd_path=HOSPITAL_USD),
        )

    # create isaac environment (scene load + robot spawn; physics starts here)
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Deactivate the hospital's built-in CollisionPlane (it duplicates
    # IsaacLab's /World/ground plane; two coincident planes cause
    # conflicting contact normals that lock the robot).
    if args_cli.custom_env == "hospital":
        import omni.usd as _omni_usd  # noqa: PLC0415 — available only at runtime
        stage = _omni_usd.get_context().get_stage()
        col_plane = stage.GetPrimAtPath("/World/hospital/Root/CollisionPlane")
        if col_plane and col_plane.IsValid():
            col_plane.SetActive(False)
            print("[INFO] Hospital CollisionPlane deactivated")
        else:
            print("[WARN] Hospital CollisionPlane not found — check USD prim path")

    # Warehouse / office: still loaded post-gym.make (visual-only assets,
    # no robot-overlapping collision geometry).
    if args_cli.custom_env in {"warehouse", "office"}:
        setup_custom_env()

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = resolve_checkpoint_path(
        log_root_path, agent_cfg["load_run"], agent_cfg["load_checkpoint"]
    )
    configure_policy_from_checkpoint(agent_cfg, resume_path)
    print(
        "[INFO] Resolved agent config: "
        f"experiment={agent_cfg['experiment_name']}, "
        f"load_run={agent_cfg['load_run']}, "
        f"checkpoint={agent_cfg['load_checkpoint']}, "
        f"noise_std_type={agent_cfg['policy'].get('noise_std_type', 'default')}"
    )

    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, agent_cfg, log_dir=None, device=agent_cfg["device"]
    )
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

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

    # simulate environment
    _dbg_step = 0
    _dbg_resets = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
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

            # --- debug: print every 200 steps (~1 s at decimation=4, dt=0.005) ---
            _dbg_step += 1
            if _dbg_step % 200 == 0 and _dbg_step <= 400:
                robot = env.unwrapped.scene["robot"]
                base_pos    = robot.data.root_pos_w[0].tolist()
                base_vel    = robot.data.root_lin_vel_w[0].tolist()
                obs_cmd     = obs[0, 9:12].tolist()
                act_max     = actions.abs().max().item()
                max_torque  = robot.data.applied_torque[0].abs().max().item()
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
                    f"  pos_z={base_pos[2]:.4f}"
                    f"  vel_xy={[round(v,3) for v in base_vel[:2]]}"
                    f"  max_action={act_max:.4f}"
                    f"  max_torque={max_torque:.4f}"
                    f"  total_resets={int(_dbg_resets)}"
                    + _beam_info
                )

    env.close()
