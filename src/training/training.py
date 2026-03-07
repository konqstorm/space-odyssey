import csv
import inspect
import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import yaml

from src.env.observation import get_observation
from src.env.reward import reward_function

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class TrainStepResult:
    score: float
    log_line: str
    batch_idx: int
    avg_reward: float
    success_rate: float
    termination_counts: dict[str, int] = field(default_factory=dict)


def _create_run_dir(runs_root, algorithm_name):
    base = Path(runs_root)
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{timestamp}_{algorithm_name}"
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    suffix = 1
    while True:
        with_suffix = base / f"{timestamp}_{algorithm_name}_{suffix:02d}"
        if not with_suffix.exists():
            with_suffix.mkdir(parents=True, exist_ok=False)
            return with_suffix
        suffix += 1


def _save_used_configs(run_dir: Path, used_configs: dict):
    for name, config in used_configs.items():
        path = run_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def _save_env_sources(run_dir: Path):
    reward_source = inspect.getsource(reward_function)
    with (run_dir / "reward.py").open("w", encoding="utf-8") as f:
        f.write(reward_source)

    observation_source = inspect.getsource(get_observation)
    with (run_dir / "observation.py").open("w", encoding="utf-8") as f:
        f.write(observation_source)


def _plot_curves(run_dir: Path, batch_indices, success_rates, avg_rewards):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(batch_indices, success_rates, marker="o", linewidth=1.5)
    ax.set_title("Success Rate vs Batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Success Rate (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "success_rate.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(batch_indices, avg_rewards, marker="o", linewidth=1.5)
    ax.set_title("Average Reward vs Batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Average Reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "avg_reward.png", dpi=150)
    plt.close(fig)


def _plot_termination_counts(run_dir: Path, termination_totals: dict[str, int]):
    labels = sorted(termination_totals.keys())
    values = [termination_totals[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title("Termination Counts")
    ax.set_xlabel("Termination Reason")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "termination_counts.png", dpi=150)
    plt.close(fig)


def run_training_loop(
    env,
    config,
    *,
    algorithm_name: str,
    build_agent: Callable,
    run_iterations: Callable,
    default_last_model_name: str,
    default_best_model_name: str,
    runs_root: str = "runs",
    used_configs: dict | None = None,
):
    run_dir = _create_run_dir(runs_root, algorithm_name)
    print(f"Run directory: {run_dir}")

    if used_configs:
        _save_used_configs(run_dir, used_configs)
    _save_env_sources(run_dir)

    run_config = deepcopy(config)
    run_config["save_path"] = str(run_dir)
    run_config["run_dir"] = str(run_dir)

    last_model_name = run_config.get("last_model_name", default_last_model_name)
    best_model_name = run_config.get("best_model_name", default_best_model_name)

    agent = build_agent(env, run_config)
    best_score = -np.inf
    batch_indices = []
    success_rates = []
    avg_rewards = []
    termination_totals: dict[str, int] = {}

    metrics_path = run_dir / "metrics.csv"
    metrics_fields = [
        "batch_idx",
        "score",
        "avg_reward",
        "success_rate",
        "best_score_so_far",
        "termination_counts_json",
    ]

    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
        metrics_writer.writeheader()

        for step in run_iterations(env, agent, run_config):
            if not isinstance(step, TrainStepResult):
                raise TypeError("Iteration must return TrainStepResult")

            print(step.log_line)

            batch_indices.append(int(step.batch_idx))
            success_rates.append(float(step.success_rate))
            avg_rewards.append(float(step.avg_reward))
            for reason, count in step.termination_counts.items():
                termination_totals[reason] = termination_totals.get(reason, 0) + int(count)

            agent.save(str(run_dir / last_model_name))

            if step.score > best_score:
                best_score = step.score
                agent.save(str(run_dir / best_model_name))
                print(f"--> New best {algorithm_name.upper()} model saved! Score = {best_score:.2f}")

            metrics_writer.writerow(
                {
                    "batch_idx": int(step.batch_idx),
                    "score": float(step.score),
                    "avg_reward": float(step.avg_reward),
                    "success_rate": float(step.success_rate),
                    "best_score_so_far": float(best_score),
                    "termination_counts_json": json.dumps(
                        step.termination_counts,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
            metrics_file.flush()

    if batch_indices:
        _plot_curves(run_dir, batch_indices, success_rates, avg_rewards)
    if termination_totals:
        _plot_termination_counts(run_dir, termination_totals)

    return agent
