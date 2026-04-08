# Rollout storage for CTS.
# Ported from go2_rl_gym/rsl_rl/rsl_rl/storage/rollout_storage_cts.py
# Adapted: imports from system rsl_rl instead of go2_rl_gym's rsl_rl.

import torch
from rsl_rl.utils import split_and_pad_trajectories  # noqa: F401 (used by parent)


class RolloutStorageCTS:
    """Trajectory buffer that keeps teacher and student samples interleaved."""

    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.history = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        teacher_num_envs: int,
        history_length: int,
        num_transitions_per_env: int,
        obs_shape: tuple,
        privileged_obs_shape: tuple,
        actions_shape: tuple,
        device: str = "cpu",
    ):
        self.device = device
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.teacher_num_envs = teacher_num_envs
        self.student_num_envs = num_envs - teacher_num_envs
        self.history_length = history_length
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
        self.history = torch.zeros(
            num_transitions_per_env, num_envs,
            history_length * obs_shape[0], device=device)

        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)

        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        self.step = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_transitions(self, transition: "RolloutStorageCTS.Transition") -> None:
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.history[self.step].copy_(transition.history)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    # ── Returns / advantages ──────────────────────────────────────────────────

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        value_limit: float | None = None,
    ) -> None:
        def _safe(x: torch.Tensor) -> torch.Tensor:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if value_limit is not None and value_limit > 0.0:
                x = x.clamp(-value_limit, value_limit)
            return x

        last_values = _safe(last_values)
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 \
                else self.values[step + 1]
            next_values = _safe(next_values)
            rewards = torch.clamp(
                torch.nan_to_num(self.rewards[step], nan=0.0, posinf=0.0, neginf=0.0),
                min=-100.0,
                max=100.0,
            )
            values = _safe(self.values[step])
            not_terminal = 1.0 - self.dones[step].float()
            delta = rewards + not_terminal * gamma * next_values - values
            advantage = delta + not_terminal * gamma * lam * advantage
            advantage = _safe(advantage)
            self.returns[step] = _safe(advantage + values)

        self.advantages = self.returns - self.values
        self.advantages = torch.nan_to_num(self.advantages, nan=0.0, posinf=0.0, neginf=0.0)
        self.advantages = (self.advantages - self.advantages.mean()) / \
                          (self.advantages.std() + 1e-8)
        self.advantages = torch.nan_to_num(self.advantages, nan=0.0, posinf=0.0, neginf=0.0)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((
            flat_dones.new_tensor([-1], dtype=torch.int64),
            flat_dones.nonzero(as_tuple=False)[:, 0],
        ))
        traj_lengths = done_indices[1:] - done_indices[:-1]
        return traj_lengths.float().mean(), self.rewards.mean()

    # ── Mini-batch generator ──────────────────────────────────────────────────

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        T = self.teacher_num_envs * self.num_transitions_per_env
        S = self.student_num_envs * self.num_transitions_per_env
        t_batch = T // num_mini_batches
        s_batch = S // num_mini_batches

        t_idx = torch.randperm(T, device=self.device)
        s_idx = T + torch.randperm(S, device=self.device)

        # Flatten env-first (teacher envs first, then student envs)
        def _flat(t: torch.Tensor) -> torch.Tensor:
            dims = list(range(2, t.dim()))
            return t.permute(1, 0, *dims).flatten(0, 1)

        obs       = _flat(self.observations)
        priv_obs  = _flat(self.privileged_observations) \
            if self.privileged_observations is not None else obs
        actions   = _flat(self.actions)
        history   = _flat(self.history)
        values    = _flat(self.values)
        returns   = _flat(self.returns)
        log_prob  = _flat(self.actions_log_prob)
        adv       = _flat(self.advantages)
        mu        = _flat(self.mu)
        sigma     = _flat(self.sigma)

        def _gather(data, slice_t, slice_s):
            i1, i2 = slice_t
            j1, j2 = slice_s
            return torch.cat([data[t_idx[i1:i2]], data[s_idx[j1:j2]]], dim=0).detach()

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                sl_t = (i * t_batch, (i + 1) * t_batch)
                sl_s = (i * s_batch, (i + 1) * s_batch)
                yield (
                    _gather(obs,      sl_t, sl_s),
                    _gather(priv_obs, sl_t, sl_s),
                    _gather(actions,  sl_t, sl_s),
                    _gather(history,  sl_t, sl_s),
                    _gather(values,   sl_t, sl_s),
                    _gather(adv,      sl_t, sl_s),
                    _gather(returns,  sl_t, sl_s),
                    _gather(log_prob, sl_t, sl_s),
                    _gather(mu,       sl_t, sl_s),
                    _gather(sigma,    sl_t, sl_s),
                    (None, None),  # hidden states placeholder
                    None,          # masks placeholder
                )
