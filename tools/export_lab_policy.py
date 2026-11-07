"""Export the local Lab student without importing/installing Isaac Lab 3.

Run: python tools/export_lab_policy.py
Uses upstream network components and the unmodified upstream JIT exporter.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StudentEncoder(nn.Module):
    """Same composition as upstream StudentMoEEncoder; no training dependencies."""

    def __init__(self, moe, normalizer):
        super().__init__()
        self.moe = moe
        self.norm_layer = normalizer

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, weights = self.moe(obs)
        return self.norm_layer(latent), weights


def linear_shapes(state, prefix):
    keys = sorted((k for k in state if k.startswith(prefix) and k.endswith('.weight')),
                  key=lambda k: int(k[len(prefix):].split('.')[0]))
    return [tuple(state[k].shape) for k in keys]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'go2_rl_lab')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/moe_cts_lab.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints/moe_cts_lab')
    args = parser.parse_args()
    source, checkpoint = args.source.resolve(), args.checkpoint.resolve()
    rsl = source / 'source/rsl_rl/rsl_rl'
    networks = load_source('go2_lab_export_networks', rsl / 'moe_cts/modules/moe.py')
    exporter = load_source('go2_lab_official_exporter', rsl / 'moe_cts/exporter.py')
    metadata = load_source('go2_lab_official_metadata', rsl / 'utils/export_model_config.py')
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    actor = linear_shapes(state, 'actor.network.')
    teacher = linear_shapes(state, 'teacher_encoder.0.network.')
    gating = linear_shapes(state, 'student_moe_encoder.moe.gating_network.0.network.')
    backbone = linear_shapes(state, 'student_moe_encoder.moe.experts.backbone.network.')
    latent, actions, history_dim = teacher[-1][0], actor[-1][0], backbone[0][1]
    single_dim = actor[0][1] - latent
    experts = gating[-1][0]
    # Nonlinearities are not encoded in a state_dict. Verify this known release
    # against the local task's ELU/L2Norm configuration, and record that provenance.
    config_source = (source / 'source/go2_rl_lab/go2_rl_lab/tasks/mixed/rsl_rl_cfg.py').read_text()
    if 'activation="elu"' not in config_source or 'norm_type="l2norm"' not in config_source:
        raise ValueError('Local task changed activation/latent normalization; review export configuration.')
    if (single_dim, history_dim, actions, experts, latent) != (45, 450, 12, 8, 32):
        raise ValueError(f'Unsupported release dimensions: {single_dim, history_dim, actions, experts, latent}')
    if any('normalizer' in key for key in state):
        raise ValueError('Checkpoint has learned observation normalization; use matching upstream runner export.')
    policy = SimpleNamespace(
        actor=networks.MLP(actor[0][1], actions, [s[0] for s in actor[:-1]], 'elu'),
        student_moe_encoder=StudentEncoder(networks.MoE(
            experts, history_dim, [s[0] for s in backbone[:-1]] + [backbone[-1][0] // experts],
            latent, 'elu'), networks.L2Norm()),
        state_dependent_std=False, num_actions=actions,
        num_single_obs=single_dim, num_actor_obs=history_dim,
    )
    for name in ('actor', 'student_moe_encoder'):
        prefix = name + '.'
        getattr(policy, name).load_state_dict(
            {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        getattr(policy, name).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    exporter.export_policy_as_jit(policy, None, None, str(args.output))
    config_path = metadata.export_model_config(
        str(args.output), str(checkpoint), ckpt['iter'], 'go2-mixed-moe-cts', 'go2-mixed-moe-cts',
        num_single_obs=single_dim, num_actor_obs=history_dim, num_actions=actions,
        control_dt=0.02)
    config = yaml.safe_load(Path(config_path).read_text())
    deploy_source = source / 'deploy/deploy_mujoco/config.yaml'
    deploy = yaml.safe_load(deploy_source.read_text())
    command_tree = ast.parse((source / 'source/go2_rl_lab/go2_rl_lab/cmd_utils/rl_gym_cmd.py').read_text())
    command_class = next(n for n in command_tree.body if isinstance(n, ast.ClassDef) and n.name == 'RLGymVelCommandCfg')
    terrain_limits = ast.literal_eval(next(
        n.value for n in command_class.body if isinstance(n, ast.AnnAssign)
        and n.target.id == 'terrain_max_command_ranges'))
    flat_ranges = [terrain_limits['flat'][k] for k in ('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw')]
    keys = ('simulation_dt', 'control_decimation', 'kps', 'kds', 'motor', 'default_angles',
            'ang_vel_scale', 'dof_pos_scale', 'dof_vel_scale', 'command_scale', 'action_scale',
            'action_clip', 'num_actions', 'num_obs', 'command_kind', 'policy_joint_names')
    config['deployment'] = {k: deploy[k] for k in keys}
    config['deployment'].update(max_velocity_command=[1.0, 1.0, 1.5],
                                training_velocity_min=[r[0] for r in flat_ranges],
                                training_velocity_max=[r[1] for r in flat_ranges],
                                actuator_delay_steps=0, joint_friction=0.01, armature=0.0)
    config['provenance'] = {
        'repository': str(source),
        'commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'branch': subprocess.check_output(['git', '-C', str(source), 'branch', '--show-current'], text=True).strip(),
        'exporter': 'source/rsl_rl/rsl_rl/moe_cts/exporter.py:export_policy_as_jit',
        'deployment_config_sha256': hashlib.sha256(deploy_source.read_bytes()).hexdigest(),
        'training_snapshots_available': (checkpoint.parent / 'params/env.yaml').is_file(),
        'note': 'Deployment settings from local lab branch; original training snapshots absent unless noted. '
                'Identity observation normalizers (none in checkpoint). Zero delay matches official MuJoCo deploy; '
                'training randomizes 0..4 physics steps. Existing Isaac Sim 4.5 Go2 asset retained.',
    }
    config['deployment_model']['onnx_file'] = None
    config['deployment_model']['command_observation_scale'] = deploy['command_scale']
    Path(config_path).write_text(yaml.safe_dump(config, sort_keys=False))
    module = torch.jit.load(str(args.output / 'policy.pt')).eval()
    torch.set_num_threads(1)
    with torch.inference_mode():
        result = module(torch.zeros(1, single_dim))
        assert result.shape == (1, actions) and torch.isfinite(result).all()
        module.reset()
    print(f'Exported student: {args.output.resolve()} ({single_dim}D x {history_dim // single_dim}, {actions} actions)')


if __name__ == '__main__':
    main()
