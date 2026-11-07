"""B2W physical simulation and locomotion; no high-level navigation model."""
import os
import omni.timeline

import torch
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import RayCasterCameraCfg, patterns
from isaaclab.utils import configclass

from assets.robots.b2w.config import B2W_CFG
from configs.custom_rl_env import LocomotionVelocityRoughEnvCfg, base_command, command_received_at
from core.ground_friction import GroundFriction


def b2w_commands(env):
    now = omni.timeline.get_timeline_interface().get_current_time()
    values = [base_command.get(str(i), [0., 0., 0.])
              if os.environ.get('SIM_COMMAND_SOURCE', 'ros2') == 'keyboard' or (str(i) in command_received_at and 0 <= now-command_received_at[str(i)] < 1.0) else [0., 0., 0.]
              for i in range(env.num_envs)]
    requested = torch.tensor(values, device=env.device, dtype=torch.float32)
    limits = torch.tensor(env.cfg.lab_command_limits, device=env.device)
    applied = torch.clamp(requested, -limits, limits)
    env.lab_requested_command = requested.clone()
    env.lab_applied_command = applied.clone()
    return applied


@configclass
class B2WObservations:
    @configclass
    class Policy(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        velocity_commands = ObservationTermCfg(func=b2w_commands)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: Policy = Policy()


@configclass
class B2WActions:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"],
        scale=0.5, use_default_offset=True,
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[".*foot_joint"], scale=5.0, use_default_offset=True,
    )


@configclass
class NoRewards:
    pass


@configclass
class B2WDeployCfg(LocomotionVelocityRoughEnvCfg):
    observations: B2WObservations = B2WObservations()
    actions: B2WActions = B2WActions()
    rewards: NoRewards = NoRewards()
    lab_command_limits: tuple = (1.5, 1.0, 1.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = B2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None
        self.scene.raycast_camera = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            mesh_prim_paths=["/World/SRU_DepthGeometry"], update_period=0.2,
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=(0.387, 0., 0.28), rot=(0.9848078, 0., 0.1736482, 0.), convention="world",
            ),
            data_types=["distance_to_image_plane"], max_distance=11.0,
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=1.0, horizontal_aperture=64.0/(72.7025/3),
                vertical_aperture=40.0/(72.7025/3), width=64, height=40,
                horizontal_aperture_offset=(94.4457/3-32)/(72.7025/3),
                vertical_aperture_offset=(62.5424/3-20)/(72.7025/3),
            ),
        )
        friction = GroundFriction.from_env()
        self.scene.terrain.physics_material.static_friction = friction.static
        self.scene.terrain.physics_material.dynamic_friction = friction.dynamic
        self.sim.physics_material = self.scene.terrain.physics_material
        self.events.physics_material = None
        self.events.reset_base.params["pose_range"] = {k:(0.,0.) for k in ("x","y","yaw")}
        self.terminations.base_contact = None
        self.terminations.bad_orientation = None
        self.episode_length_s = 100000.0
