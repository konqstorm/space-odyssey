import numpy as np
import matplotlib.pyplot as plt
from src.env.environment import SpaceEnv
import os
# def run_episode(env, agent, deterministic=True, render=False):
#     state, _ = env.reset()
#     done, truncated = False, False

#     total_reward = 0.0
#     steps = 0

#     while not (done or truncated):
#         action, *_ = agent.select_action(state, deterministic=deterministic)
#         state, reward, done, truncated, info = env.step(action)

#         total_reward += reward
#         steps += 1

#         if render:
#             env.render()

#     return {
#         "reward": total_reward,
#         "length": steps,
#         "termination_reason": info.get("termination_reason", "unknown"),
#         "final_distance": info.get("distance_to_goal", np.nan)
#     }


# def evaluate_agent(
#     agent,
#     env_params,
#     episodes=300,
#     deterministic=True,
# ):
#     results = []

#     successes = 0

#     for ep in range(episodes):
#         env = SpaceEnv(**env_params)
#         ep_result = run_episode(env, agent, deterministic)
#         results.append(ep_result)

#         if ep_result["termination_reason"] == "goal":
#             successes += 1


#     rewards = [r["reward"] for r in results]
#     lengths = [r["length"] for r in results]
#     distances = [r["final_distance"] for r in results]
#     reasons = [r["termination_reason"] for r in results]

#     summary = {
#         "episodes": episodes,
#         "avg_reward": float(np.mean(rewards)),
#         "std_reward": float(np.std(rewards)),
#         "success_rate": 100.0 * reasons.count("goal") / episodes,
#         "avg_length": float(np.mean(lengths)),
#         "avg_final_distance": float(np.nanmean(distances)),
#         "termination_reasons": {
#             k: reasons.count(k) for k in sorted(set(reasons))
#         },
#     }

#     return summary


# import matplotlib.pyplot as plt

# def plot_evaluation_histogram(summary, label):
#     """
#     Заменяет линейный график на гистограмму причин завершения эпизодов.
#     """
#     reasons_dict = summary["termination_reasons"]
#     names = list(reasons_dict.keys())
#     counts = list(reasons_dict.values())
    
#     # Настройка цветов для наглядности
#     colors = []
#     for name in names:
#         if name == "goal": colors.append("#2ecc71")       # Зеленый - успех
#         elif name == "collision": colors.append("#e74c3c")  # Красный - столкновение
#         elif name == "out_of_bounds": colors.append("#3498db") # Синий - улетел за край
#         elif name == "timeout": colors.append("#f1c40f")    # Желтый - время вышло
#         else: colors.append("#95a5a6")                      # Серый - прочее

#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(names, counts, color=colors, edgecolor='black', alpha=0.8)
    
#     # Добавляем подписи количества над столбцами
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height + 1,
#                  f'{int(height)}', ha='center', va='bottom', fontsize=10)

#     plt.xlabel("Termination Reason")
#     plt.ylabel("Number of eposides")
#     plt.title(f"Evaluation results: {label}\n(In total {summary['episodes']} episodes)")
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
    
#     # Добавляем в легенду Success Rate для информативности
#     plt.annotate(f"Success Rate: {summary['success_rate']:.1f}%", 
#                  xy=(0.95, 0.95), xycoords='axes fraction', 
#                  ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


def run_episode(env, agent, *, deterministic=True, render=False):
    state, _ = env.reset()
    done = False
    truncated = False

    total_reward = 0.0
    steps = 0
    last_info = {}

    while not (done or truncated):
        action, *_ = agent.select_action(state, deterministic=deterministic)
        state, reward, done, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        last_info = info

        if render:
            env.render()

    return {
        "reward": total_reward,
        "length": steps,
        "termination_reason": last_info.get("termination_reason", "unknown"),
        "final_distance": last_info.get("distance_to_goal", np.nan),
    }


def evaluate_agent(
    agent,
    make_env_fn,
    *,
    episodes,
    deterministic=True,
    render=False,
):
    results = []

    for _ in trange(episodes, desc="Evaluating agent"):
        env = make_env_fn()
        results.append(
            run_episode(
                env,
                agent,
                deterministic=deterministic,
                render=render,
            )
        )

    rewards = np.array([r["reward"] for r in results])
    lengths = np.array([r["length"] for r in results])
    distances = np.array([r["final_distance"] for r in results])
    reasons = [r["termination_reason"] for r in results]

    summary = {
        "episodes": episodes,
        "avg_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "success_rate": 100.0 * reasons.count("goal") / max(1, episodes),
        "avg_length": float(lengths.mean()),
        "avg_final_distance": float(np.nanmean(distances)),
        "termination_reasons": {
            k: reasons.count(k) for k in sorted(set(reasons))
        },
    }

    return summary


def plot_termination_histogram(summary, *, title_suffix="", runs_dir: str = "runs/evaluation"):
    reasons = summary["termination_reasons"]
    names = list(reasons.keys())
    counts = list(reasons.values())

    color_map = {
        "goal": "#2ecc71",
        "asteroid": "#e74c3c",
        "timeout": "#f1c40f",
    }
    colors = [color_map.get(n, "#95a5a6") for n in names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, counts, color=colors, edgecolor="black", alpha=0.85)

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1,
            f"{int(h)}",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Termination reason")
    plt.ylabel("Episodes")
    plt.title(
        "Evaluation results\n"
        f"Success rate: {summary['success_rate']:.1f}% {title_suffix}"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(runs_dir, exist_ok=True)
    save_path = os.path.join(runs_dir, f"termination_hist_{title_suffix}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    print(f"[EVAL] Гистограмма сохранена: {save_path}")
    plt.close()
