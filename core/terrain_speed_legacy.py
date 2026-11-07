"""Terrain-aware command shaping, independent of policy inference and ROS."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path

import numpy as np
import yaml


@dataclass
class TerrainSpeedConfig:
    enabled: bool = True
    flat_max_vx: float = 3.5
    stairs_max_vx: float = .7
    unknown_max_vx: float = .5
    flat_max_vy: float = .5
    stairs_max_vy: float = .2
    unknown_max_vy: float = .2
    max_yaw_rate: float = 1.5
    accel_limit: float = 1.0
    decel_limit: float = 2.0
    braking_decel_factor: float = .75
    yaw_accel_limit: float = 2.0
    brake_margin: float = .5
    lookahead_min: float = 1.0
    lookahead_max: float = 8.0
    terrain_state_hysteresis: float = .3
    body_radius: float = .4
    nominal_base_height: float = .35
    body_min_z: float = .22
    body_max_z: float = .65
    surface_z_tolerance: float = .28
    query_step: float = .15
    grid_size: float = 1.0
    flat_max_slope: float = .08
    plane_index_half_extent: float = 100.0
    max_timestamp_gap: float = .25
    log_interval: float = 1.0

    def __post_init__(self):
        for item in fields(self):
            if item.name != 'enabled' and (not math.isfinite(getattr(self, item.name)) or getattr(self, item.name) <= 0):
                raise ValueError(f'{item.name} must be finite and positive')
        if self.lookahead_min > self.lookahead_max:
            raise ValueError('Invalid lookahead interval')
        if not self.unknown_max_vx <= self.stairs_max_vx <= self.flat_max_vx:
            raise ValueError('Expected unknown <= stairs <= flat vx limits')
        if self.braking_decel_factor > 1:
            raise ValueError('Braking reserve factor must not exceed 1')
        if self.body_min_z >= self.body_max_z:
            raise ValueError('Invalid body clearance height range')

    @classmethod
    def load(cls, path):
        return cls(**yaml.safe_load(Path(path).read_text())['terrain_speed'])


@dataclass
class LimitResult:
    command: tuple
    state: str
    speed_limit: float
    reason: str
    distance_to_stairs: float
    brake_distance: float
    raw: tuple
    timestamp: float
    changed: bool

    def as_dict(self):
        return asdict(self)


class TerrainAwareVelocityLimiter:
    def __init__(self, index, config=None):
        self.index = index
        self.cfg = config or TerrainSpeedConfig()
        self.reset()

    def reset(self):
        self.state = 'UNKNOWN'
        self.last_time = None
        self.last_command = np.zeros(3)
        self.clear_since = None
        self.last_speed_limit = self.cfg.unknown_max_vx

    @staticmethod
    def braking_distance(speed, target_speed, decel):
        return max(0.0, (speed * speed - target_speed * target_speed) / (2 * decel))

    @staticmethod
    def directional_limit(direction, vx, vy):
        return 1.0 / math.sqrt((direction[0] / vx)**2 + (direction[1] / vy)**2)

    def update(self, raw, position, quaternion_wxyz, actual_body_velocity, timestamp):
        c = self.cfg
        raw = np.asarray(raw, dtype=float)
        pos, q, actual = map(lambda x: np.asarray(x, dtype=float),
                             (position, quaternion_wxyz, actual_body_velocity))
        previous_state = self.state
        raw_valid = raw.shape == (3,) and np.isfinite(raw).all()
        if not raw_valid:
            raw = np.zeros(3)
        valid = (raw_valid and pos.shape == (3,) and q.shape == (4,) and actual.shape == (3,)
                 and all(np.isfinite(x).all() for x in (pos, q, actual))
                 and .98 < np.linalg.norm(q) < 1.02 and math.isfinite(timestamp))
        if valid and 1-2*(q[1]**2+q[2]**2) < .7:
            valid = False
        if not c.enabled:
            return LimitResult(tuple(raw), 'DISABLED', math.inf, 'disabled', math.inf, 0,
                               tuple(raw), timestamp, False)
        gap = timestamp - self.last_time if self.last_time is not None and math.isfinite(timestamp) else .02
        if gap < 0 or gap > c.max_timestamp_gap:
            valid = False
        dt = min(max(gap, 0), .05)
        speed = float(np.linalg.norm(raw[:2]))
        direction = raw[:2] / speed if speed > 1e-8 else np.array([1., 0.])
        flat = self.directional_limit(direction, c.flat_max_vx, c.flat_max_vy)
        stairs = self.directional_limit(direction, c.stairs_max_vx, c.stairs_max_vy)
        unknown = self.directional_limit(direction, c.unknown_max_vx, c.unknown_max_vy)
        measured = float(np.linalg.norm(actual[:2])) if actual.shape == (3,) and np.isfinite(actual).all() else 0.0
        brake = self.braking_distance(measured, stairs, c.decel_limit)
        planning_decel = c.decel_limit*c.braking_decel_factor
        distance = math.inf
        limit, reason, state = unknown, 'invalid_pose_velocity_or_time', 'UNKNOWN'
        if valid:
            qw, qx, qy, qz = q
            # Project the full body-to-world rotation, including roll/pitch on stairs.
            rot = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz)],
                            [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz)]])
            def world_direction(body_direction):
                projected = rot @ body_direction
                return projected / max(np.linalg.norm(projected), 1e-8)
            ground = pos.copy()
            ground[2] -= c.nominal_base_height
            current = self.index.footprint(*ground, c.body_radius)
            length = min(c.lookahead_max, max(c.lookahead_min,
                self.braking_distance(max(measured, min(speed, flat)), stairs, planning_decel)
                + c.brake_margin + c.body_radius))
            # Check both intended motion and residual actual motion while stopping/turning.
            directions = []
            if speed > 1e-8:
                directions.append(world_direction(direction))
            if measured > .05:
                actual_direction = world_direction(actual[:2] / measured)
                if not directions or np.dot(directions[0], actual_direction) < .995:
                    directions.append(actual_direction)
            hazards = [self.index.trace(ground, d, length, c.body_radius, c.query_step) for d in directions]
            stair_distances = [d for kind, d, _ in hazards if kind == 'STAIRS']
            distance = min(stair_distances, default=math.inf)
            if current.kind == 'UNKNOWN' or any(kind == 'UNKNOWN' for kind, _, _ in hazards):
                state, limit, reason = 'UNKNOWN', unknown, 'unverified_support_or_clearance'
                self.clear_since = None
            elif current.kind == 'STAIRS':
                state, limit, reason = 'ON_STAIRS', stairs, 'body_footprint_on_stairs'
                self.clear_since = None
            elif math.isfinite(distance):
                state, reason = 'APPROACHING_STAIRS', 'lookahead_stair_braking'
                limit = min(flat, math.sqrt(stairs**2 + 2*planning_decel*max(0, distance-c.brake_margin)))
                self.clear_since = None
            else:
                if self.clear_since is None:
                    self.clear_since = timestamp
                if previous_state in ('ON_STAIRS', 'LEAVING_STAIRS') and timestamp-self.clear_since < c.terrain_state_hysteresis:
                    state, limit, reason = 'LEAVING_STAIRS', stairs, 'waiting_for_full_body_clearance'
                elif previous_state == 'APPROACHING_STAIRS' and timestamp-self.clear_since < c.terrain_state_hysteresis:
                    state, limit, reason = 'APPROACHING_STAIRS', min(flat, self.last_speed_limit), 'lookahead_clearance_hysteresis'
                elif previous_state == 'UNKNOWN' and timestamp-self.clear_since < c.terrain_state_hysteresis:
                    state, limit, reason = 'UNKNOWN', unknown, 'confirming_flat_support'
                else:
                    state, limit, reason = 'FLAT', flat, 'verified_flat_corridor'
        else:
            self.clear_since = None
        yaw_target = float(np.clip(raw[2], -c.max_yaw_rate, c.max_yaw_rate))
        # Ellipsoidal budget prevents simultaneous full translation and full yaw.
        budget = math.sqrt((speed / limit)**2 + (yaw_target / c.max_yaw_rate)**2)
        factor = 1.0 / max(1.0, budget)
        target = speed * factor
        previous = float(np.linalg.norm(self.last_command[:2]))
        slewed = np.clip(target, max(0., previous-c.decel_limit*dt), previous+c.accel_limit*dt)
        # Raw reductions/stop and newly discovered hard terrain limits win over slew.
        output_speed = min(speed, target, float(slewed))
        if target < previous-c.decel_limit*dt-1e-6:
            reason += ':operator_or_hard_cap_over_slew'
        yaw_target *= factor
        yaw_output = float(np.clip(yaw_target, self.last_command[2]-c.yaw_accel_limit*dt,
                                   self.last_command[2]+c.yaw_accel_limit*dt))
        if yaw_output * raw[2] < 0 or raw[2] == 0:
            yaw_output = 0.
        yaw_output = math.copysign(min(abs(yaw_output), abs(yaw_target)), yaw_target)
        output = np.array([*(direction * output_speed), yaw_output])
        if not raw_valid:
            output[:] = 0
        self.state = state
        self.last_time = timestamp if math.isfinite(timestamp) else None
        self.last_command, self.last_speed_limit = output, limit
        return LimitResult(tuple(output), state, limit, reason, distance, brake, tuple(raw),
                           timestamp, state != previous_state)
