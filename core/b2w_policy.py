"""B2W locomotion only: 60 state features to 12 leg + 4 wheel actions."""
import torch


class B2WLocomotionPolicy:
    def __init__(self, path, num_envs, device):
        self.module = torch.jit.load(str(path), map_location=device).eval()
        self.num_envs = num_envs
        self.reset()

    def reset(self, dones=None):
        for name in ("hidden_state", "cell_state"):
            value = getattr(self.module, name)
            if dones is None:
                value.zero_()
            elif value.shape[1] == len(dones):
                value[:, dones.bool(), :] = 0.0

    def __call__(self, observation):
        if observation.shape != (self.num_envs, 60) or not torch.isfinite(observation).all():
            raise ValueError(f"Invalid B2W locomotion observation: {observation.shape}")
        action = self.module(observation)
        if action.shape != (self.num_envs, 16) or not torch.isfinite(action).all():
            raise RuntimeError("Invalid B2W locomotion action")
        return action
