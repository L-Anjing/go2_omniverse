# MoE-CTS actor-critic network for inference-time compatibility with
# go2_rl_gym checkpoints such as go2_moe_cts_*.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic_cts import L2Norm, SimNorm, _get_activation


class MLP(nn.Module):
    def __init__(
        self,
        dims: list[int],
        activation: str = "elu",
        last_activation: bool = False,
    ):
        super().__init__()
        act = _get_activation(activation)
        layers: list[nn.Module] = []
        last_dim = dims[0]
        for hidden_dim in dims[1:-1]:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act)
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, dims[-1]))
        if last_activation:
            layers.append(act)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Experts(nn.Module):
    def __init__(
        self,
        expert_num: int,
        input_dim: int,
        backbone_hidden_dims: list[int],
        expert_hidden_dim: int,
        output_dim: int,
        activation: str = "elu",
    ):
        super().__init__()
        self.expert_num = expert_num
        self.output_dim = output_dim
        self.backbone = MLP(
            [input_dim, *backbone_hidden_dims, expert_num * expert_hidden_dim],
            activation=activation,
            last_activation=True,
        )
        self.experts = nn.Conv1d(
            in_channels=expert_num * expert_hidden_dim,
            out_channels=expert_num * output_dim,
            kernel_size=1,
            groups=expert_num,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.backbone(x).unsqueeze(-1)
        expert_outputs = self.experts(shared_features).squeeze(-1)
        return expert_outputs.reshape(-1, self.expert_num, self.output_dim)


class MoE(nn.Module):
    def __init__(
        self,
        expert_num: int,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "elu",
    ):
        super().__init__()
        self.experts = Experts(
            expert_num=expert_num,
            input_dim=input_dim,
            backbone_hidden_dims=hidden_dims[:-1],
            expert_hidden_dim=hidden_dims[-1],
            output_dim=output_dim,
            activation=activation,
        )
        self.gating_network = nn.Sequential(
            MLP([input_dim, *hidden_dims[:-1], expert_num], activation=activation),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.gating_network(x)
        expert_outputs = self.experts(x)
        output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output, weights


class StudentMoEEncoder(nn.Module):
    def __init__(
        self,
        expert_num: int,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "elu",
        norm_type: str = "l2norm",
    ):
        super().__init__()
        self.norm_layer = L2Norm() if norm_type == "l2norm" else SimNorm()
        self.moe = MoE(
            expert_num=expert_num,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, weights = self.moe(obs)
        return self.norm_layer(latent), weights


class ActorCriticMoECTS(nn.Module):
    """MoE-CTS actor-critic compatible with go2_rl_gym checkpoints."""

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_envs: int,
        history_length: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        teacher_encoder_hidden_dims: list[int] | None = None,
        student_encoder_hidden_dims: list[int] | None = None,
        expert_num: int = 8,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        latent_dim: int = 32,
        norm_type: str = "l2norm",
        **kwargs,
    ):
        if kwargs:
            print(f"ActorCriticMoECTS: ignoring unexpected kwargs: {list(kwargs)}")
        assert norm_type in ("l2norm", "simnorm"), f"Unknown norm_type: {norm_type}"
        super().__init__()

        actor_hidden_dims = actor_hidden_dims or [512, 256, 128]
        critic_hidden_dims = critic_hidden_dims or [512, 256, 128]
        teacher_encoder_hidden_dims = teacher_encoder_hidden_dims or [512, 256]
        student_encoder_hidden_dims = student_encoder_hidden_dims or [512, 256, 256]

        self.num_actions = num_actions
        self.num_actor_obs = num_obs
        self.history_length = history_length

        self.register_buffer(
            "history",
            torch.zeros((num_envs, history_length, num_obs)),
            persistent=False,
        )

        self.teacher_encoder = nn.Sequential(
            MLP(
                [num_critic_obs, *teacher_encoder_hidden_dims, latent_dim],
                activation=activation,
            ),
            L2Norm() if norm_type == "l2norm" else SimNorm(),
        )
        self.student_moe_encoder = StudentMoEEncoder(
            expert_num=expert_num,
            input_dim=num_obs * history_length,
            hidden_dims=student_encoder_hidden_dims,
            output_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
        )
        self.actor = MLP(
            [latent_dim + num_obs, *actor_hidden_dims, num_actions],
            activation=activation,
        )
        self.critic = MLP(
            [latent_dim + num_critic_obs, *critic_hidden_dims, 1],
            activation=activation,
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        self._had_nonfinite_action_mean = False
        Normal.set_default_validate_args = False

        print(f"[MoE-CTS] Actor:              {self.actor}")
        print(f"[MoE-CTS] Critic:             {self.critic}")
        print(f"[MoE-CTS] Teacher encoder:    {self.teacher_encoder}")
        print(f"[MoE-CTS] Student MoE encoder:{self.student_moe_encoder}")

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is not None:
            self.history[dones > 0] = 0.0

    def forward(self):
        raise NotImplementedError

    def consume_nonfinite_action_flag(self) -> bool:
        had_issue = self._had_nonfinite_action_mean
        self._had_nonfinite_action_mean = False
        return had_issue

    def _update_distribution(self, latent_and_obs: torch.Tensor) -> None:
        mean = self.actor(latent_and_obs)
        if not torch.isfinite(mean).all():
            self._had_nonfinite_action_mean = True
            mean = torch.nan_to_num(mean, nan=0.0, posinf=20.0, neginf=-20.0)
        mean = mean.clamp(-100.0, 100.0)
        std = torch.nan_to_num(self.std, nan=1.0, posinf=1.0, neginf=1.0).clamp(1e-6, 10.0)
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(
        self,
        obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        history: torch.Tensor,
        is_teacher: bool,
        **kwargs,
    ) -> torch.Tensor:
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent, _ = self.student_moe_encoder(history)
            latent = latent.detach()
        self._update_distribution(torch.cat([latent, obs], dim=1))
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(
        self,
        privileged_obs: torch.Tensor,
        history: torch.Tensor,
        is_teacher: bool,
        **kwargs,
    ) -> torch.Tensor:
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent, _ = self.student_moe_encoder(history)
        return self.critic(torch.cat([latent.detach(), privileged_obs], dim=1))

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        n = obs.shape[0]
        if self.history.shape[0] != n or self.history.device != obs.device:
            self.history = torch.zeros(
                n,
                self.history_length,
                self.num_actor_obs,
                device=obs.device,
                dtype=obs.dtype,
            )
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        latent, _ = self.student_moe_encoder(self.history.flatten(1))
        return self.actor(torch.cat([latent, obs], dim=1))
