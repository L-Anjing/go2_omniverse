# MoE-CTS PPO algorithm.
# Structured to match go2_rl_gym/rsl_rl/rsl_rl/algorithms/moe_cts.py while
# keeping the numerical safeguards added in this IsaacLab port.

from __future__ import annotations

import torch
import torch.nn as nn

from .actor_critic_moe_cts import ActorCriticMoECTS
from .cts_algorithm import CTS


class MoECTS(CTS):
    """CTS with a MoE student encoder and load-balance regularization."""

    model: ActorCriticMoECTS

    def __init__(
        self,
        model: ActorCriticMoECTS,
        num_envs: int,
        history_length: int,
        load_balance_coef: float = 0.01,
        **kwargs,
    ):
        self.load_balance_coef = load_balance_coef
        super().__init__(
            model=model,
            num_envs=num_envs,
            history_length=history_length,
            **kwargs,
        )
        self.optimizer2 = torch.optim.Adam(
            self.model.student_moe_encoder.parameters(),
            lr=kwargs.get("student_encoder_learning_rate", 1e-3),
        )

    def _student_encoder_parameters(self):
        return self.model.student_moe_encoder.parameters()

    def update(self) -> tuple[float, float, float, float, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy_loss = 0.0
        mean_latent_loss = 0.0
        mean_load_balance_loss = 0.0
        num_ppo_updates = 0
        num_student_updates = 0
        assert not self.model.is_recurrent

        data = list(self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs))
        t_batch = self.teacher_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        s_batch = self.student_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches

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

            def _fwd(start: int, end: int, is_teacher: bool):
                obs_in = torch.nan_to_num(obs_b[start:end], nan=0.0)
                priv_in = torch.nan_to_num(priv_obs_b[start:end], nan=0.0)
                hist_in = torch.nan_to_num(hist_b[start:end], nan=0.0)
                self.model.act(obs_in, priv_in, hist_in, is_teacher)
                lp = torch.nan_to_num(
                    self.model.get_actions_log_prob(act_b[start:end]),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                val = torch.nan_to_num(
                    self.model.evaluate(priv_in, hist_in, is_teacher),
                    nan=0.0, posinf=0.0, neginf=0.0,
                ).clamp(-self.value_target_limit, self.value_target_limit)
                mu = torch.nan_to_num(self.model.action_mean, nan=0.0, posinf=0.0, neginf=0.0)
                sig = torch.nan_to_num(
                    self.model.action_std, nan=1.0, posinf=1.0, neginf=1.0
                ).clamp(1e-6, 10.0)
                ent = torch.nan_to_num(self.model.entropy, nan=0.0, posinf=0.0, neginf=0.0)
                return lp, val, mu, sig, ent

            t_res = _fwd(0, t_batch, True)
            s_res = _fwd(t_batch, t_batch + s_batch, False)
            results = [torch.cat([x1, x2], dim=0) for x1, x2 in zip(t_res, s_res)]
            lp_b, val_b, mu_b, sig_b, ent_b = results

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
                    for group in self.optimizer1.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(lp_b - old_lp_b.squeeze())
            adv_sq = adv_b.squeeze()
            surr = -adv_sq * ratio
            surr_clip = -adv_sq * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
            surrogate_losses = torch.max(surr, surr_clip)
            teacher_surrogate_loss = surrogate_losses[:t_batch].mean()
            student_surrogate_loss = surrogate_losses[t_batch:].mean()
            surrogate_loss = teacher_surrogate_loss + student_surrogate_loss

            if self.use_clipped_value_loss:
                val_clip = tgt_val_b + (val_b - tgt_val_b).clamp(-self.clip_param, self.clip_param)
                v_loss = torch.max((val_b - ret_b).pow(2), (val_clip - ret_b).pow(2)).mean()
            else:
                v_loss = (ret_b - val_b).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * v_loss - self.entropy_coef * ent_b.mean()

            self.optimizer1.zero_grad(set_to_none=True)
            loss.backward()
            params_to_clip = []
            for group in self.optimizer1.param_groups:
                params = group["params"]
                if isinstance(params, torch.Tensor):
                    params_to_clip.append(params)
                else:
                    params_to_clip.extend(params)
            nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)
            self.optimizer1.step()

            mean_value_loss += v_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += ent_b.mean().item()
            num_ppo_updates += 1

        for sample in data:
            (_, priv_obs_b, _, hist_b,
             _, _, _, _, _, _, _, _) = sample

            hist_student = torch.nan_to_num(hist_b[t_batch:], nan=0.0)
            priv_student = torch.nan_to_num(priv_obs_b[t_batch:], nan=0.0)
            student_latent, gating_weights = self.model.student_moe_encoder(hist_student)
            with torch.no_grad():
                teacher_latent = self.model.teacher_encoder(priv_student)

            latent_loss = torch.nan_to_num(
                (teacher_latent - student_latent).pow(2).mean(),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            mean_usage = torch.mean(gating_weights, dim=0)
            target_usage = torch.full_like(mean_usage, 1.0 / gating_weights.shape[1])
            load_balance_loss = torch.nan_to_num(
                torch.mean((mean_usage - target_usage).pow(2)),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            student_loss = latent_loss + self.load_balance_coef * load_balance_loss

            self.optimizer2.zero_grad(set_to_none=True)
            student_loss.backward()
            nn.utils.clip_grad_norm_(self.model.student_moe_encoder.parameters(), self.max_grad_norm)
            self.optimizer2.step()

            mean_latent_loss += latent_loss.item()
            mean_load_balance_loss += load_balance_loss.item()
            num_student_updates += 1

        mean_value_loss /= max(num_ppo_updates, 1)
        mean_surrogate_loss /= max(num_ppo_updates, 1)
        mean_entropy_loss /= max(num_ppo_updates, 1)
        mean_latent_loss /= max(num_student_updates, 1)
        mean_load_balance_loss /= max(num_student_updates, 1)
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_entropy_loss,
            mean_latent_loss,
            mean_load_balance_loss,
        )
