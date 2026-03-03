from .reinforce_training import train_reinforce
from .trpo_training import train_trpo


def train(env, algorithm, config, *, runs_root="runs", used_configs=None):
    algo = str(algorithm).strip().lower()
    if algo == "reinforce":
        return train_reinforce(env, config, runs_root=runs_root, used_configs=used_configs)
    if algo == "trpo":
        return train_trpo(env, config, runs_root=runs_root, used_configs=used_configs)
    raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'reinforce' or 'trpo'.")


__all__ = ["train", "train_reinforce", "train_trpo"]
