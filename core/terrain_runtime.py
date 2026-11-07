"""Per-robot command limiter bridge for the existing Lab observation term."""
from dataclasses import replace
import math
import time

import torch

from core.terrain_speed import TerrainAwareVelocityLimiter, align_navigation_timestamp
from core.usd_terrain import StageTerrainIndex


class TerrainCommandAdapter:
    def __init__(self, env, stage, roots, config, semantic_resolver=None, state_reader=None, log_enabled=True, ros_time_provider=None):
        self.env = env
        self.log_enabled = log_enabled
        self.ros_time_provider = ros_time_provider
        self.state_reader = state_reader
        # Retain a caller's more restrictive explicit global caps.
        vx, vy, wz = env.cfg.lab_command_limits
        self.config = replace(config, flat_max_vx=min(config.flat_max_vx, vx),
            stairs_max_vx=min(config.stairs_max_vx, vx), unknown_max_vx=min(config.unknown_max_vx, vx),
            flat_max_vy=min(config.flat_max_vy, vy), stairs_max_vy=min(config.stairs_max_vy, vy),
            unknown_max_vy=min(config.unknown_max_vy, vy), max_yaw_rate=min(config.max_yaw_rate, wz))
        self.stage_index = StageTerrainIndex(stage, roots, self.config, semantic_resolver)
        self.index_revision = 1
        limiter_class = TerrainAwareVelocityLimiter
        if config.behavior_version == 'legacy':
            from core.terrain_speed_legacy import TerrainAwareVelocityLimiter as limiter_class
        self.limiters = [limiter_class(self.stage_index.index, self.config) for _ in range(env.num_envs)]
        self.results = []
        self.last_timestamp = None
        self.last_log = -math.inf
        self.cached = None
        self.cached_raw = None

    def reset(self):
        for limiter in self.limiters:
            limiter.reset()
        self.last_timestamp, self.cached, self.cached_raw, self.results = None, None, None, []

    def commands(self, raw):
        now = float(self.env.sim.current_time)
        raw_values = raw.tolist()
        navigation = getattr(self.env, 'navigation_speed_constraints', {})
        navigation_values = [navigation.get(str(i)) for i in range(self.env.num_envs)]
        if self.ros_time_provider is not None and self.config.navigation_constraints_enabled:
            ros_now=float(self.ros_time_provider())
            navigation_values=[align_navigation_timestamp(nav,now,ros_now) for nav in navigation_values]
        same_index = all(l.index is self.stage_index.index for l in self.limiters)
        if now == self.last_timestamp and self.cached is not None and raw_values == self.cached_raw and navigation_values == getattr(self,'cached_navigation',None) and not self.stage_index.dirty and same_index:
            return self.cached.clone()
        rebuilt = self.stage_index.refresh() or not same_index
        if rebuilt:
            self.index_revision += 1
            for limiter in self.limiters:
                limiter.index = self.stage_index.index
                if self.config.behavior_version == 'legacy':
                    limiter.reset()
                else:
                    # A static-map refresh must not erase the last command
                    # and manufacture an instantaneous zero-speed request.
                    limiter.clear_since = None
        robot = self.env.scene['robot']
        pos, quat = robot.data.root_pos_w.tolist(), robot.data.root_quat_w.tolist()
        measured = torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], 1).tolist()
        if self.state_reader is not None:
            pos, quat, measured = self.state_reader(pos, quat, measured)
        self.results = [limiter.update(cmd, p, q, v, now, **({'navigation':nav} if self.config.behavior_version != 'legacy' else {}))
                        for limiter, cmd, p, q, v, nav in zip(self.limiters, raw_values, pos, quat, measured, navigation_values)]
        self.cached_navigation = navigation_values
        for i,result in enumerate(self.results):
            if getattr(result,'diagnostics',None) is not None:
                result.diagnostics['index_revision'] = self.index_revision
                result.diagnostics['index_rebuilt'] = rebuilt
                result.diagnostics['angular_velocity_body'] = robot.data.root_ang_vel_b[i].tolist()
                result.diagnostics['linear_velocity_body'] = robot.data.root_lin_vel_b[i].tolist()
        self.cached = torch.tensor([r.command for r in self.results], device=raw.device, dtype=raw.dtype)
        self.last_timestamp = now
        self.cached_raw = raw_values
        wall_now = time.monotonic()
        if self.log_enabled and self.config.log_interval > 0 and wall_now-self.last_log >= self.config.log_interval:
            for i, result in enumerate(self.results):
                print(f'[terrain robot{i}] t={now:.2f} state={result.state} raw={result.raw} '
                      f'limited={result.command} actual={tuple(round(v,3) for v in measured[i])} '
                      f'cap={result.speed_limit:.3f} stairs_distance={result.distance_to_stairs:.3f} '
                      f'brake_distance={result.brake_distance:.3f} reason={result.reason}', flush=True)
            self.last_log = wall_now
        return self.cached.clone()

    def close(self):
        self.stage_index.close()
