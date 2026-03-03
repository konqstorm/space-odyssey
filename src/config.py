from pathlib import Path
import yaml


def _load_yaml_config(config_dir, filename):
    path = Path(config_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")
    return data


def _require_keys(config, required_keys, config_name):
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"Missing keys in {config_name}: {missing}")


def load_env_config(config_dir="configs"):
    config = _load_yaml_config(config_dir, "env.yaml")
    _require_keys(config, ["width", "height", "num_asteroids", "max_steps"], "env.yaml")
    return config


def load_reinforce_config(config_dir="configs"):
    config = _load_yaml_config(config_dir, "reinforce_training.yaml")
    _require_keys(
        config,
        [
            "episodes",
            "batch_size",
            "lr",
            "gamma",
            "entropy_coeff",
            "vf_coeff",
            "update_epochs",
        ],
        "reinforce_training.yaml",
    )
    return config


def load_trpo_config(config_dir="configs"):
    config = _load_yaml_config(config_dir, "trpo_training.yaml")
    _require_keys(
        config,
        [
            "episodes",
            "lr",
            "gamma",
            "delta",
            "cg_damping",
            "cg_steps",
            "line_search_steps",
            "value_update_steps",
        ],
        "trpo_training.yaml",
    )
    return config


def load_runtime_config(config_dir="configs"):
    config = _load_yaml_config(config_dir, "runtime.yaml")
    _require_keys(
        config,
        [
            "torch_num_threads",
            "seed",
            "watch",
            "train",
        ],
        "runtime.yaml",
    )
    return config
