"""Standalone Lab student loading; each robot owns its TorchScript history."""
from pathlib import Path

import torch
import yaml


def load_deployment_config(path):
    config = yaml.safe_load(Path(path).read_text())
    model, deploy = config['deployment_model'], config['deployment']
    if (model['single_observation_size'], model['actor_observation_size'],
        model['observation_history_length'], model['action_size']) != (45, 450, 10, 12):
        raise ValueError('Expected Lab 45D/10-frame student with 12 actions')
    if deploy['command_kind'] != 'command/body/velocity':
        raise ValueError('Only body-frame velocity commands are supported')
    if abs(deploy['simulation_dt'] * deploy['control_decimation'] - model['control_dt']) > 1e-8:
        raise ValueError('Inconsistent deployment control period')
    return deploy


class LabStudentPolicy:
    def __init__(self, path, num_envs, device, config):
        self.modules = [torch.jit.load(str(path), map_location=device).eval() for _ in range(num_envs)]
        self.action_clip = config['action_clip']
        for module in self.modules:
            if (module.num_single_obs, module.history_len, module.num_actions) != (45, 10, 12):
                raise ValueError('TorchScript interface differs from metadata')
        self.reset()

    def reset(self, dones=None):
        for i, module in enumerate(self.modules):
            if dones is None or bool(dones[i]):
                module.reset()

    def __call__(self, obs):
        if obs.shape != (len(self.modules), 45) or not torch.isfinite(obs).all():
            raise ValueError(f'Invalid Lab observation: {obs.shape}')
        actions = torch.cat([module(obs[i:i+1]) for i, module in enumerate(self.modules)])
        if actions.shape != (len(self.modules), 12) or not torch.isfinite(actions).all():
            raise RuntimeError('Invalid Lab policy output')
        # Official MuJoCo deployment clips raw actions before forming targets.
        return actions.clamp(-self.action_clip, self.action_clip)
