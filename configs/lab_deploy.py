"""Opt-in Go2 RL Lab student configuration for Isaac Sim 4.5 / Isaac Lab 2.1."""
import torch
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from configs.custom_rl_env import UnitreeGo2CustomEnvCfg, GO2_MODEL_JOINT_NAMES, constant_commands
from core.lab_actuator import LabGo2ActuatorCfg
from core.ground_friction import GroundFriction


def lab_commands(env):
    commands = constant_commands(env)
    limits = torch.tensor(env.cfg.lab_command_limits, device=env.device)
    adapter = getattr(env, 'terrain_command_adapter', None)
    applied = adapter.commands(commands) if adapter is not None else torch.clamp(commands, -limits, limits)
    # Snapshot the exact commands used to construct this observation; ROS may
    # receive a new request while the following physics step is executing.
    env.lab_requested_command = commands.detach().clone()
    env.lab_applied_command = applied.detach().clone()
    return applied


@configclass
class LabPolicyObservations(ObservationGroupCfg):
    # Clip each raw term BEFORE normalization, as in the upstream ObsManager.
    base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
    projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, clip=(-100.0, 100.0))
    velocity_commands = ObservationTermCfg(func=lab_commands, clip=(-100.0, 100.0), scale=(0.5, 1.0, 1.0))
    joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, clip=(-100.0, 100.0), params={
        'asset_cfg': SceneEntityCfg('robot', joint_names=GO2_MODEL_JOINT_NAMES, preserve_order=True)})
    joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, clip=(-100.0, 100.0), scale=0.05, params={
        'asset_cfg': SceneEntityCfg('robot', joint_names=GO2_MODEL_JOINT_NAMES, preserve_order=True)})
    actions = ObservationTermCfg(func=mdp.last_action, clip=(-100.0, 100.0))

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        # The official TorchScript module owns the ten-frame term-wise history.
        self.history_length = 0


@configclass
class LabObservations:
    policy: LabPolicyObservations = LabPolicyObservations()


@configclass
class Go2LabDeployCfg(UnitreeGo2CustomEnvCfg):
    observations: LabObservations = LabObservations()
    # Match the default flat-terrain envelope; the terrain limiter applies
    # lower caps for stairs and unknown surfaces when enabled.
    lab_command_limits: tuple = (3.5, 1.0, 1.5)

    def __post_init__(self):
        super().__post_init__()
        ground_friction = GroundFriction.from_env()
        self.scene.terrain.physics_material.static_friction = ground_friction.static
        self.scene.terrain.physics_material.dynamic_friction = ground_friction.dynamic
        self.sim.physics_material = self.scene.terrain.physics_material
        self.events.physics_material = None
        self.events.reset_base.params['pose_range'] = {k: (0., 0.) for k in ('x','y','yaw')}
        self.events.reset_base.params['velocity_range'] = {k: (0., 0.) for k in ('x','y','z','roll','pitch','yaw')}
        self.episode_length_s = 100000.
        self.actions.joint_pos.joint_names = GO2_MODEL_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

    def configure(self, deployment, command_limits=None):
        """Apply the exported Lab interface and actuator settings."""
        d = deployment
        if d['policy_joint_names'] != GO2_MODEL_JOINT_NAMES:
            raise ValueError('Unexpected Lab policy joint mapping')
        limits = list(command_limits if command_limits is not None else d['max_velocity_command'])
        if len(limits) != 3 or not all(torch.isfinite(torch.tensor(limits))) or min(limits) < 0:
            raise ValueError('Command limits must contain three finite nonnegative values')
        if any(v > min(hi, -lo) for v, lo, hi in zip(
                limits, d['training_velocity_min'], d['training_velocity_max'])):
            raise ValueError('Command limits exceed the source training envelope')
        self.lab_command_limits = tuple(limits)
        self.sim.dt = d['simulation_dt']
        self.decimation = d['control_decimation']
        self.sim.render_interval = self.decimation
        # Uniform PD in this release; fail instead of silently losing per-joint gains.
        if len(set(d['kps'])) != 1 or len(set(d['kds'])) != 1:
            raise ValueError('This actuator configuration expects uniform PD gains')
        self.scene.robot.actuators = {'Go2HV': LabGo2ActuatorCfg(
            joint_names_expr=['.*'], stiffness=d['kps'][0], damping=d['kds'][0],
            effort_limit=d['motor']['Y2'], effort_limit_sim=1e9,
            velocity_limit=1e9, velocity_limit_sim=1e9,
            friction=d['joint_friction'], armature=d['armature'],
            min_delay=d['actuator_delay_steps'], max_delay=d['actuator_delay_steps'],
            **{k: d['motor'][k] for k in ('X1', 'X2', 'Y1', 'Y2')})}
        self.scene.robot.init_state.joint_pos = dict(zip(GO2_MODEL_JOINT_NAMES, d['default_angles']))
        self.scene.robot.init_state.joint_vel = {'.*': 0.0}
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.4)
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
        self.actions.joint_pos.scale = d['action_scale']
        self.actions.joint_pos.clip = {'.*': (-100.0, 100.0)}
        # Raw action clipping is done by LabStudentPolicy, separately from target clipping.
        self.scene.height_scanner = None
        self.terminations.base_contact = None
        self.terminations.bad_orientation = None
        policy = self.observations.policy
        policy.base_ang_vel.scale = d['ang_vel_scale']
        policy.velocity_commands.scale = tuple(d['command_scale'])
        policy.joint_pos.scale = d['dof_pos_scale']
        policy.joint_vel.scale = d['dof_vel_scale']
