"""Port of the upstream Go2HV torque-speed envelope to Isaac Lab 2.1.

Reference: go2_rl_lab/assets/go2_actuator.py (MIT, Tianyang Wu, 2026).
Uses the existing explicit PD path, not a second articulation controller.
"""
import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass


class LabGo2Actuator(DelayedPDActuator):
    def compute(self, control_action, joint_pos, joint_vel):
        self._motor_joint_vel = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort):
        c = self.cfg
        peak = torch.where(self._motor_joint_vel * effort > 0, c.Y1, c.Y2)
        fraction = ((c.X2 - self._motor_joint_vel.abs()) / (c.X2 - c.X1)).clamp(0.0, 1.0)
        limit = peak * fraction
        return torch.clamp(effort, min=-limit, max=limit)


@configclass
class LabGo2ActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = LabGo2Actuator
    X1: float = 13.5
    X2: float = 30.0
    Y1: float = 20.2
    Y2: float = 23.4
