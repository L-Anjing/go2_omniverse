# CTS (Concurrent Teacher-Student) PPO algorithm.
# Ported from go2_rl_gym/rsl_rl/rsl_rl/algorithms/cts.py
# Removed legged_gym dependency; uses only torch + rsl_rl_cts internals.

from __future__ import annotations

import copy
import itertools
import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic_cts import ActorCriticCTS
from .rollout_storage_cts import RolloutStorageCTS


class CTS:
    """Concurrent Teacher-Student PPO.

    75% teacher envs use privileged obs → better value/advantage estimates.
    25% student envs use observation history only (deployment-realistic).
    Student encoder is aligned to teacher encoder via an MSE distillation
    loss, trained on a *separate* optimizer so it doesn't interfere with
    the PPO gradient.
    """

    model: ActorCriticCTS

    def __init__(
        self,
        model: ActorCriticCTS,
        num_envs: int,
        history_length: int,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        student_encoder_learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        teacher_env_ratio: float = 0.75,
        value_target_limit: float = 1000.0,
        device: str = "cpu",
        **kwargs,
    ):
        if kwargs:
            print(f"[CTS] Ignoring unexpected kwargs: {list(kwargs)}")

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.history_length = history_length
        self.value_target_limit = value_target_limit

        self.model = model.to(device)
        self.storage: RolloutStorageCTS | None = None
        self.transition = RolloutStorageCTS.Transition()

        # PPO + teacher-encoder + actor/critic optimizer
        params1 = [
            {"params": self.model.teacher_encoder.parameters()},
            {"params": self.model.critic.parameters()},
            {"params": self.model.actor.parameters()},
            {"params": [self.model.std]},
        ]
        self.optimizer1 = optim.Adam(params1, lr=learning_rate)
        # Student encoder distillation optimizer (separate)
        self.optimizer2 = optim.Adam(
            self.model.student_encoder.parameters(),
            lr=student_encoder_learning_rate,
        )

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Split envs: every (1/student_ratio)-th env is a student env
        student_env_ratio = 1.0 - teacher_env_ratio
        stride = max(1, round(1.0 / student_env_ratio))
        self.teacher_env_idxs = torch.tensor(
            [i for i in range(num_envs) if i % stride != 0], device=device)
        self.student_env_idxs = torch.tensor(
            [i for i in range(num_envs) if i % stride == 0], device=device)
        self.teacher_num_envs = len(self.teacher_env_idxs)
        self.student_num_envs = len(self.student_env_idxs)

    # ── Storage ───────────────────────────────────────────────────────────────

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list,
        critic_obs_shape: list,
        action_shape: list,
    ) -> None:
        self.storage = RolloutStorageCTS(
            num_envs, self.teacher_num_envs, self.history_length,
            num_transitions_per_env,
            actor_obs_shape, critic_obs_shape, action_shape,
            self.device,
        )

    def test_mode(self):
        self.model.eval()

    def train_mode(self):
        self.model.train()

    def snapshot_state(self) -> dict:
        """Capture the current train state so a bad PPO step can be rolled back."""
        return {
            "model": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
            "optimizer1": copy.deepcopy(self.optimizer1.state_dict()),
            "optimizer2": copy.deepcopy(self.optimizer2.state_dict()),
            "learning_rate": self.learning_rate,
        }

    def restore_state(self, snapshot: dict) -> None:
        """Restore a previously captured train state."""
        self.model.load_state_dict(snapshot["model"])
        self.optimizer1.load_state_dict(snapshot["optimizer1"])
        self.optimizer2.load_state_dict(snapshot["optimizer2"])
        self.learning_rate = snapshot["learning_rate"]
        for group in self.optimizer1.param_groups:
            group["lr"] = self.learning_rate

    def get_nonfinite_state_names(self) -> list[str]:
        """Return parameter/buffer names that contain NaN or inf."""
        bad_names: list[str] = []
        for name, tensor in itertools.chain(
            self.model.named_parameters(),
            self.model.named_buffers(),
        ):
            if not torch.isfinite(tensor).all():
                bad_names.append(name)
        return bad_names

    # ── Rollout collection ────────────────────────────────────────────────────

    def act(
        self,
        obs: torch.Tensor,
        privileged_obs: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """Choose actions for all envs; store transition data."""
        history = history.clone()
        ti, si = self.teacher_env_idxs, self.student_env_idxs

        def _run(o, p, h, is_t):
            acts = self.model.act(o, p, h, is_t).detach()
            vals = self.model.evaluate(p, h, is_t).detach()
            lp   = self.model.get_actions_log_prob(acts).detach()
            mu   = self.model.action_mean.detach()
            sig  = self.model.action_std.detach()
            return acts, vals, lp, mu, sig

        t_r = _run(obs[ti], privileged_obs[ti], history[ti], True)
        s_r = _run(obs[si], privileged_obs[si], history[si], False)

        # Interleave results: teacher rows first, then student rows
        merged = [torch.cat([x1, x2], dim=0) for x1, x2 in zip(t_r, s_r)]
        self.transition.actions          = merged[0]
        self.transition.values           = merged[1]
        self.transition.actions_log_prob = merged[2]
        self.transition.action_mean      = merged[3]
        self.transition.action_sigma     = merged[4]
        self.transition.history          = torch.cat([history[ti], history[si]], dim=0)
        self.transition.observations     = torch.cat([obs[ti], obs[si]], dim=0)
        self.transition.critic_observations = torch.cat([privileged_obs[ti], privileged_obs[si]], dim=0)

        # Re-order back to env index order for env.step()
        real_actions = torch.zeros_like(self.transition.actions)
        real_actions[ti] = self.transition.actions[:self.teacher_num_envs]
        real_actions[si] = self.transition.actions[self.teacher_num_envs:]
        return real_actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
    ) -> None:
        ti, si = self.teacher_env_idxs, self.student_env_idxs
        rew = torch.cat([rewards[ti], rewards[si]], dim=0).clone()
        dns = torch.cat([dones[ti], dones[si]], dim=0)
        self.transition.rewards = rew
        self.transition.dones   = dns
        if "time_outs" in infos:
            to = torch.cat([infos["time_outs"][ti], infos["time_outs"][si]], dim=0)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * to.unsqueeze(1).to(self.device), 1)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.model.reset(dones)

    def compute_returns(
        self,
        last_privileged_obs: torch.Tensor,
        last_history: torch.Tensor,
    ) -> None:
        ti, si = self.teacher_env_idxs, self.student_env_idxs
        last_vals = torch.cat([
            self.model.evaluate(last_privileged_obs[ti], last_history[ti], True).detach(),
            self.model.evaluate(last_privileged_obs[si], last_history[si], False).detach(),
        ], dim=0)
        last_vals = torch.nan_to_num(last_vals, nan=0.0, posinf=0.0, neginf=0.0)
        last_vals = last_vals.clamp(-self.value_target_limit, self.value_target_limit)
        self.storage.compute_returns(
            last_vals,
            self.gamma,
            self.lam,
            self.value_target_limit,
        )

    # ── Learning ──────────────────────────────────────────────────────────────

    def update(self) -> tuple[float, float, float, float]:
        mean_value_loss = mean_surrogate_loss = mean_entropy_loss = mean_latent_loss = 0.0
        num_ppo_updates = 0
        num_latent_updates = 0
        assert not self.model.is_recurrent

        data = list(self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs))
        t_batch = self.teacher_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        s_batch = self.student_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches

        # ── PPO pass (teacher + student actions, teacher/student values) ──────
        for sample in data:
            (obs_b, priv_obs_b, act_b, hist_b,
             tgt_val_b, adv_b, ret_b,
             old_lp_b, old_mu_b, old_sig_b, _, _) = sample

            tgt_val_b = torch.nan_to_num(tgt_val_b, nan=0.0, posinf=0.0, neginf=0.0)
            ret_b = torch.nan_to_num(ret_b, nan=0.0, posinf=0.0, neginf=0.0)
            adv_b = torch.nan_to_num(adv_b, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
            old_lp_b = torch.nan_to_num(old_lp_b, nan=0.0, posinf=0.0, neginf=0.0)
            old_mu_b = torch.nan_to_num(old_mu_b, nan=0.0, posinf=0.0, neginf=0.0)
            old_sig_b = torch.nan_to_num(old_sig_b, nan=1.0, posinf=1.0, neginf=1.0).clamp(1e-6, 10.0)
            tgt_val_b = tgt_val_b.clamp(-self.value_target_limit, self.value_target_limit)
            ret_b = ret_b.clamp(-self.value_target_limit, self.value_target_limit)

            def _fwd(start, end, is_t):
                # Sanitize inputs before forward pass — NaN in obs/history
                # (from dying strategy + short-episode rollouts) can propagate
                # into network weights and cause Normal(loc=NaN) crash.
                obs_in  = torch.nan_to_num(obs_b[start:end],      nan=0.0)
                priv_in = torch.nan_to_num(priv_obs_b[start:end], nan=0.0)
                hist_in = torch.nan_to_num(hist_b[start:end],     nan=0.0)
                self.model.act(obs_in, priv_in, hist_in, is_t)
                lp  = torch.nan_to_num(
                    self.model.get_actions_log_prob(act_b[start:end]),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                val = torch.nan_to_num(
                    self.model.evaluate(priv_in, hist_in, is_t),
                    nan=0.0, posinf=0.0, neginf=0.0,
                ).clamp(-self.value_target_limit, self.value_target_limit)
                mu = torch.nan_to_num(
                    self.model.action_mean,
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                sig = torch.nan_to_num(
                    self.model.action_std,
                    nan=1.0, posinf=1.0, neginf=1.0,
                ).clamp(1e-6, 10.0)
                ent = torch.nan_to_num(
                    self.model.entropy,
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                return lp, val, mu, sig, ent

            t_res = _fwd(0, t_batch, True)
            s_res = _fwd(t_batch, t_batch + s_batch, False)
            results = [torch.cat([x1, x2], dim=0) for x1, x2 in zip(t_res, s_res)]
            lp_b, val_b, mu_b, sig_b, ent_b = results

            # KL-adaptive LR
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sig_b / old_sig_b + 1e-5)
                        + (old_sig_b.pow(2) + (old_mu_b - mu_b).pow(2))
                        / (2.0 * sig_b.pow(2)) - 0.5,
                        dim=-1,
                    )
                    kl_mean = kl.mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for g in self.optimizer1.param_groups:
                        g["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(lp_b - old_lp_b.squeeze())
            adv_sq = adv_b.squeeze()
            surr      = -adv_sq * ratio
            surr_clip = -adv_sq * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
            surrogate_loss = torch.max(surr, surr_clip).mean()

            # Value loss
            if self.use_clipped_value_loss:
                val_clip = tgt_val_b + (val_b - tgt_val_b).clamp(-self.clip_param, self.clip_param)
                v_loss = torch.max((val_b - ret_b).pow(2), (val_clip - ret_b).pow(2)).mean()
            else:
                v_loss = (ret_b - val_b).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * v_loss \
                   - self.entropy_coef * ent_b.mean()

            if not torch.isfinite(loss):
                print("[CTS] Warning: skipping non-finite PPO batch loss.", flush=True)
                self.optimizer1.zero_grad(set_to_none=True)
                continue

            self.optimizer1.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                itertools.chain.from_iterable(g["params"] for g in self.optimizer1.param_groups),
                self.max_grad_norm,
            )
            if not torch.isfinite(torch.as_tensor(grad_norm, device=loss.device)):
                print("[CTS] Warning: skipping PPO batch with non-finite grad norm.", flush=True)
                self.optimizer1.zero_grad(set_to_none=True)
                continue
            self.optimizer1.step()

            mean_value_loss     += v_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss   += ent_b.mean().item()
            num_ppo_updates += 1

        # ── Distillation pass (student encoder only) ──────────────────────────
        for sample in data:
            (obs_b, priv_obs_b, _, hist_b, *_rest) = sample
            hist_student = torch.nan_to_num(hist_b[t_batch:], nan=0.0, posinf=0.0, neginf=0.0)
            priv_student = torch.nan_to_num(priv_obs_b[t_batch:], nan=0.0, posinf=0.0, neginf=0.0)
            student_latent = self.model.student_encoder(hist_student)
            with torch.no_grad():
                teacher_latent = self.model.teacher_encoder(priv_student)
            latent_loss = (teacher_latent - student_latent).pow(2).mean()

            if not torch.isfinite(latent_loss):
                print("[CTS] Warning: skipping non-finite latent batch loss.", flush=True)
                self.optimizer2.zero_grad(set_to_none=True)
                continue

            self.optimizer2.zero_grad()
            latent_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.student_encoder.parameters(), self.max_grad_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm, device=latent_loss.device)):
                print("[CTS] Warning: skipping latent batch with non-finite grad norm.", flush=True)
                self.optimizer2.zero_grad(set_to_none=True)
                continue
            self.optimizer2.step()
            mean_latent_loss += latent_loss.item()
            num_latent_updates += 1

        n_ppo = max(num_ppo_updates, 1)
        n_latent = max(num_latent_updates, 1)
        # Important: reset rollout cursor for the next collection phase.
        # Without this, the next iteration keeps writing past
        # num_transitions_per_env and triggers "Rollout buffer overflow".
        self.storage.clear()
        return (
            mean_value_loss / n_ppo,
            mean_surrogate_loss / n_ppo,
            mean_entropy_loss / n_ppo,
            mean_latent_loss / n_latent,
        )
