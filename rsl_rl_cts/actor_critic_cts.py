# Concurrent Teacher-Student actor-critic network.
# Ported from go2_rl_gym/rsl_rl/rsl_rl/modules/actor_critic_cts.py
# Original author: wty-yy  https://arxiv.org/abs/2405.10830
#
# Changes vs original:
#   - act_inference auto-resizes history if num_envs changes (training vs deploy)
#   - uses rsl_rl.utils.resolve_nn_activation for activation lookup

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _get_activation(act_name: str) -> nn.Module:
    try:
        from rsl_rl.utils import resolve_nn_activation
        return resolve_nn_activation(act_name)
    except ImportError:
        pass
    mapping = {
        "elu": nn.ELU(), "selu": nn.SELU(), "relu": nn.ReLU(),
        "crelu": nn.ReLU(), "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
    }
    act = mapping.get(act_name)
    if act is None:
        raise ValueError(f"Unknown activation: {act_name}")
    return act


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int,
               activation: nn.Module) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), activation]
    for i in range(len(hidden_dims) - 1):
        layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), activation]
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*layers)


class L2Norm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=-1)


class SimNorm(nn.Module):
    """Simplicial normalization (https://arxiv.org/abs/2204.00616)."""
    dim = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(shp)

    def __repr__(self) -> str:
        return f"SimNorm(dim={self.dim})"


def _build_encoder(input_dim: int, hidden_dims: list[int], latent_dim: int,
                   activation: nn.Module, norm_type: str) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), activation]
    for i in range(len(hidden_dims) - 1):
        layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), activation]
    layers.append(nn.Linear(hidden_dims[-1], latent_dim))
    if norm_type == "l2norm":
        layers.append(L2Norm())
    elif norm_type == "simnorm":
        layers.append(SimNorm())
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
    return nn.Sequential(*layers)


class ActorCriticCTS(nn.Module):
    """Concurrent Teacher-Student actor-critic.

    Architecture
    ------------
    Teacher encoder : privileged_obs (critic obs)  →  latent (32D, l2-normed)
    Student encoder : history (T×obs_dim flattened) →  latent (32D, l2-normed)
    Actor           : [latent, obs]                 →  action mean
    Critic          : [latent.detach(), privileged_obs] → value

    Training
    --------
    75% of envs use the *teacher* path (privileged obs → teacher latent).
    25% use the *student* path (observation history → student latent).
    Both paths share actor/critic weights; student encoder is aligned to
    teacher encoder via an MSE distillation loss updated on a separate
    optimizer.

    Deployment
    ----------
    Only student encoder + actor are needed.  Call ``act_inference(obs)``
    which auto-updates the internal history buffer.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_envs: int,
        history_length: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        teacher_encoder_hidden_dims: list[int] | None = None,
        student_encoder_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        latent_dim: int = 32,
        norm_type: str = "l2norm",
        value_limit: float = 1000.0,
        **kwargs,
    ):
        if kwargs:
            print(f"ActorCriticCTS: ignoring unexpected kwargs: {list(kwargs)}")
        assert norm_type in ("l2norm", "simnorm"), f"Unknown norm_type: {norm_type}"
        super().__init__()

        actor_hidden_dims = actor_hidden_dims or [512, 256, 128]
        critic_hidden_dims = critic_hidden_dims or [512, 256, 128]
        teacher_encoder_hidden_dims = teacher_encoder_hidden_dims or [512, 256]
        student_encoder_hidden_dims = student_encoder_hidden_dims or [512, 256]

        act = _get_activation(activation)

        self.num_actions = num_actions
        self.num_actor_obs = num_actor_obs
        self.history_length = history_length
        self.value_limit = value_limit

        # Encoders
        self.teacher_encoder = _build_encoder(
            num_critic_obs, teacher_encoder_hidden_dims, latent_dim, act, norm_type)
        self.student_encoder = _build_encoder(
            num_actor_obs * history_length, student_encoder_hidden_dims, latent_dim, act, norm_type)

        # Actor: [latent, obs] → action
        self.actor = _build_mlp(latent_dim + num_actor_obs, actor_hidden_dims, num_actions, act)

        # Critic: [latent, privileged_obs] → value
        self.critic = _build_mlp(latent_dim + num_critic_obs, critic_hidden_dims, 1, act)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False
        self._had_nonfinite_action_mean = False

        # History buffer (not saved in state_dict – reinitialized at deploy time)
        self._history_num_envs = num_envs
        self.register_buffer(
            "history",
            torch.zeros(num_envs, history_length, num_actor_obs),
            persistent=False,
        )

        print(f"[CTS] Actor:           {self.actor}")
        print(f"[CTS] Critic:          {self.critic}")
        print(f"[CTS] Teacher encoder: {self.teacher_encoder}")
        print(f"[CTS] Student encoder: {self.student_encoder}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    # ── Interface ─────────────────────────────────────────────────────────────

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset history for completed environments."""
        if dones is not None:
            self.history[dones > 0] = 0.0

    def forward(self):
        raise NotImplementedError

    def consume_nonfinite_action_flag(self) -> bool:
        """Return and clear the actor-output non-finite flag."""
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
    ) -> torch.Tensor:
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent = self.student_encoder(history).detach()
        self._update_distribution(torch.cat([latent, obs], dim=1))
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(
        self,
        privileged_obs: torch.Tensor,
        history: torch.Tensor,
        is_teacher: bool,
    ) -> torch.Tensor:
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent = self.student_encoder(history)
        value = self.critic(torch.cat([latent.detach(), privileged_obs], dim=1))
        # Keep critic outputs in a numerically safe range. Healthy returns in this
        # task are O(10-100), so a wide tanh bound preserves learning signal while
        # preventing runaway bootstrap targets from exploding the PPO value loss.
        if self.value_limit is not None and self.value_limit > 0.0:
            value = self.value_limit * torch.tanh(value / self.value_limit)
        return value

    # ── Inference (student only) ──────────────────────────────────────────────

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Student-only inference.  Maintains internal rolling history buffer.

        Safe to call with a different batch size than used during training:
        if the batch dimension changes the history is reinitialized to zeros.
        """
        n = obs.shape[0]
        if self.history.shape[0] != n or self.history.device != obs.device:
            self.history = torch.zeros(
                n, self.history_length, self.num_actor_obs,
                device=obs.device, dtype=obs.dtype,
            )
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        latent = self.student_encoder(self.history.flatten(1))
        return self.actor(torch.cat([latent, obs], dim=1))
