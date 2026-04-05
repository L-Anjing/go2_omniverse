# CTS on-policy runner adapted for IsaacLab's RslRlVecEnvWrapper.
#
# Key differences from go2_rl_gym's OnPolicyRunnerCTS:
#   - env.step() returns (obs, rew, dones, extras) [4 values, not 5]
#     Privileged obs live in extras["observations"]["critic"].
#   - No legged_gym dependency.
#   - Checkpoint saves/loads model_state_dict + both optimizers.
#   - get_inference_policy() returns model.act_inference (student only).

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from .actor_critic_cts import ActorCriticCTS
from .cts_algorithm import CTS
from .rollout_storage_cts import RolloutStorageCTS


class CTSRunner:
    """On-policy CTS runner for IsaacLab environments."""

    def __init__(
        self,
        env,                   # RslRlVecEnvWrapper
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ):
        self.device = device
        self.env = env
        self.log_dir = log_dir

        # ── Config ────────────────────────────────────────────────────────────
        policy_cfg    = train_cfg["policy"]
        alg_cfg       = train_cfg["algorithm"]
        self.history_length       = train_cfg.get("history_length", 5)
        self.num_steps_per_env    = train_cfg.get("num_steps_per_env", 48)
        self.save_interval        = train_cfg.get("save_interval", 200)
        self.max_iterations       = train_cfg.get("max_iterations", 15000)
        self.experiment_name      = train_cfg.get("experiment_name", "cts_train")

        # ── Dimensions ────────────────────────────────────────────────────────
        num_obs      = env.num_obs
        num_priv_obs = env.num_privileged_obs if env.num_privileged_obs > 0 else num_obs
        num_actions  = env.num_actions
        num_envs     = env.num_envs

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = ActorCriticCTS(
            num_actor_obs=num_obs,
            num_critic_obs=num_priv_obs,
            num_actions=num_actions,
            num_envs=num_envs,
            history_length=self.history_length,
            **policy_cfg,
        ).to(device)

        # ── Algorithm ─────────────────────────────────────────────────────────
        self.alg = CTS(
            self.model,
            num_envs=num_envs,
            history_length=self.history_length,
            device=device,
            **alg_cfg,
        )
        self.alg.init_storage(
            num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_priv_obs],
            [num_actions],
        )

        # ── History buffer (rolling window of actor obs) ──────────────────────
        self.history = torch.zeros(
            num_envs, self.history_length, num_obs, device=device)

        # ── Logging ───────────────────────────────────────────────────────────
        self.writer: SummaryWriter | None = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0

        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

    # ── Helper: extract privileged obs from env step/reset extras ─────────────

    @staticmethod
    def _priv_from_extras(
        extras: dict, obs_fallback: torch.Tensor
    ) -> torch.Tensor:
        obs_dict = extras.get("observations", {})
        return obs_dict.get("critic", obs_fallback)

    # ── Main train loop ───────────────────────────────────────────────────────

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = True,
    ) -> None:
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        # Initial obs
        obs, extras = self.env.get_observations()
        priv_obs = self._priv_from_extras(extras, obs)
        obs, priv_obs = obs.to(self.device), priv_obs.to(self.device)
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)

        self.model.train()

        ep_infos: list[dict] = []
        teacher_rew_buf  = deque(maxlen=100)
        teacher_len_buf  = deque(maxlen=100)
        student_rew_buf  = deque(maxlen=100)
        student_len_buf  = deque(maxlen=100)
        cur_rew_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_ep_len  = torch.zeros(self.env.num_envs, device=self.device)

        start_it = self.current_learning_iteration
        tot_it   = start_it + num_learning_iterations

        for it in range(start_it, tot_it):
            t_collect = time.time()

            # ── Rollout ───────────────────────────────────────────────────────
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, priv_obs, self.history.flatten(1))
                    obs, rew, dones, extras = self.env.step(actions)
                    priv_obs = self._priv_from_extras(extras, obs)
                    obs      = obs.to(self.device)
                    priv_obs = priv_obs.to(self.device)
                    rew      = rew.to(self.device)
                    dones    = dones.to(self.device)

                    # Sanitize (height-scan rays can produce NaN at terrain edges)
                    obs      = torch.nan_to_num(obs,      nan=0.0, posinf=5.0,  neginf=-5.0)
                    priv_obs = torch.nan_to_num(priv_obs, nan=0.0, posinf=5.0,  neginf=-5.0)
                    rew      = torch.nan_to_num(rew,      nan=0.0, posinf=100.0, neginf=-100.0)

                    # History: reset completed envs then append new obs
                    self.history[dones > 0] = 0.0
                    self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)

                    self.alg.process_env_step(rew, dones, extras)

                    # Book-keeping
                    if "episode" in extras:
                        ep_infos.append(extras["episode"])
                    cur_rew_sum += rew
                    cur_ep_len  += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if new_ids.shape[0]:
                        new_ids_1d = new_ids.squeeze(1)
                        ti = self.alg.teacher_env_idxs
                        t_mask = torch.isin(new_ids_1d, ti)
                        t_ids  = new_ids_1d[t_mask]
                        s_ids  = new_ids_1d[~t_mask]
                        if t_ids.numel():
                            teacher_rew_buf.extend(cur_rew_sum[t_ids].view(-1).cpu().tolist())
                            teacher_len_buf.extend(cur_ep_len[t_ids].view(-1).cpu().tolist())
                        if s_ids.numel():
                            student_rew_buf.extend(cur_rew_sum[s_ids].view(-1).cpu().tolist())
                            student_len_buf.extend(cur_ep_len[s_ids].view(-1).cpu().tolist())
                        cur_rew_sum[new_ids_1d] = 0
                        cur_ep_len[new_ids_1d]  = 0

            t_learn = time.time()
            collection_time = t_learn - t_collect

            self.alg.compute_returns(priv_obs, self.history.flatten(1))
            mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss \
                = self.alg.update()

            learn_time = time.time() - t_learn

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time      += collection_time + learn_time
            self.current_learning_iteration += 1

            # ── Logging ───────────────────────────────────────────────────────
            if self.writer is not None:
                self.writer.add_scalar("Loss/value_function",  mean_value_loss,     it)
                self.writer.add_scalar("Loss/surrogate",       mean_surrogate_loss, it)
                self.writer.add_scalar("Loss/entropy",         mean_entropy_loss,   it)
                self.writer.add_scalar("Loss/latent",          mean_latent_loss,    it)
                self.writer.add_scalar("Loss/learning_rate",   self.alg.learning_rate, it)
                self.writer.add_scalar("Policy/mean_noise_std", self.model.std.mean().item(), it)
                fps = int(self.num_steps_per_env * self.env.num_envs /
                          (collection_time + learn_time))
                self.writer.add_scalar("Perf/total_fps", fps, it)
                if teacher_rew_buf:
                    self.writer.add_scalar(
                        "Train/mean_teacher_reward", statistics.mean(teacher_rew_buf), it)
                    self.writer.add_scalar(
                        "Train/mean_teacher_ep_len", statistics.mean(teacher_len_buf), it)
                if student_rew_buf:
                    self.writer.add_scalar(
                        "Train/mean_student_reward", statistics.mean(student_rew_buf), it)
                    self.writer.add_scalar(
                        "Train/mean_student_ep_len", statistics.mean(student_len_buf), it)
                # Per-reward breakdown
                for ep_info in ep_infos:
                    for key, val in ep_info.items():
                        v = val.mean().item() if isinstance(val, torch.Tensor) else float(val)
                        self.writer.add_scalar(f"Metrics/{key}", v, it)

            # ── Console log every 50 iters ────────────────────────────────────
            done = it - start_it + 1
            if done % 50 == 0:
                t_rew = f"{statistics.mean(teacher_rew_buf):+.3f}" if teacher_rew_buf else "n/a"
                s_rew = f"{statistics.mean(student_rew_buf):+.3f}" if student_rew_buf else "n/a"
                t_len = f"{statistics.mean(teacher_len_buf):.0f}"  if teacher_len_buf else "n/a"
                elapsed = self.tot_time
                remain  = (elapsed / max(done, 1)) * max(num_learning_iterations - done, 0)
                eta     = f"{int(remain // 3600)}h{int((remain % 3600) // 60):02d}m"
                print(
                    f"[CTS {done}/{num_learning_iterations}] "
                    f"T_rew={t_rew} S_rew={s_rew} T_len={t_len} "
                    f"lat={mean_latent_loss:.4f} "
                    f"steps={self.tot_timesteps/1e6:.2f}M ETA={eta}",
                    flush=True,
                )

            # ── Save checkpoint ───────────────────────────────────────────────
            if it % self.save_interval == 0 or it == tot_it - 1:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "model_state_dict":      self.model.state_dict(),
            "optimizer1_state_dict": self.alg.optimizer1.state_dict(),
            "optimizer2_state_dict": self.alg.optimizer2.state_dict(),
            "iter":                  self.current_learning_iteration,
        }, path)

    def load(self, path: str, load_optimizer: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer1.load_state_dict(ckpt["optimizer1_state_dict"])
            self.alg.optimizer2.load_state_dict(ckpt["optimizer2_state_dict"])
        self.current_learning_iteration = ckpt.get("iter", 0)
        print(f"[CTS] Loaded checkpoint from {path} (iter={self.current_learning_iteration})")

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_inference_policy(self, device: str | None = None):
        """Return the student-only inference function.

        The returned callable maintains an internal rolling history and
        computes actions without any privileged observations.
        Compatible with omniverse_sim.py's ``policy(obs)`` usage.
        """
        self.model.eval()
        if device is not None:
            self.model.to(device)
        return self.model.act_inference
