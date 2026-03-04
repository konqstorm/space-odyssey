from src.agents import TRPOAgent
from .training import TrainStepResult, run_training_loop


def _build_trpo_agent(env, config):
    return TRPOAgent(
        env,
        lr=float(config["lr"]),
        gamma=float(config["gamma"]),
        delta=float(config["delta"]),
        cg_damping=float(config["cg_damping"]),
        cg_steps=int(config["cg_steps"]),
        line_search_steps=int(config["line_search_steps"]),
        value_update_steps=int(config["value_update_steps"]),
    )


def _trpo_iterations(env, agent, config):
    episodes = int(config["episodes"])
    batch_episodes = int(config.get("batch_episodes", 12))
    batch_episodes = max(1, batch_episodes)

    batch_trajectory = ([], [], [], [], [])
    episodes_in_batch = 0

    for ep in range(episodes):
        state, _ = env.reset()
        done, truncated = False, False
        trajectory = ([], [], [], [], [])
        termination_reason = "unknown"
        reward_component_sums = {
            "reward_progress": 0.0,
            "reward_goal_speed": 0.0,
            "reward_alignment": 0.0,
            "reward_wobble": 0.0,
            "reward_avoid": 0.0,
            "reward_terminal": 0.0,
        }
        
        while not (done or truncated):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            trajectory[0].append(state)
            trajectory[1].append(action)
            trajectory[2].append(reward)
            trajectory[3].append(log_prob)
            trajectory[4].append(next_state)

            for key in reward_component_sums:
                reward_component_sums[key] += float(info.get(key, 0.0))

            state = next_state
            if done or truncated:
                termination_reason = info.get("termination_reason", "unknown")

        for idx in range(5):
            batch_trajectory[idx].extend(trajectory[idx])
        episodes_in_batch += 1

        did_update = False
        is_last_episode = (ep == episodes - 1)
        if episodes_in_batch >= batch_episodes or is_last_episode:
            agent.update(batch_trajectory)
            batch_trajectory = ([], [], [], [], [])
            episodes_in_batch = 0
            did_update = True
        
        ep_reward = float(sum(trajectory[2]))
        success_rate = 100.0 if termination_reason == "goal" else 0.0
        comp_line = (
            f" | C: prog={reward_component_sums['reward_progress']:+7.1f}"
            f" speed={reward_component_sums['reward_goal_speed']:+7.1f}"
            f" align={reward_component_sums['reward_alignment']:+7.1f}"
            f" wobble={reward_component_sums['reward_wobble']:+7.1f}"
            f" avoid={reward_component_sums['reward_avoid']:+7.1f}"
            f" term={reward_component_sums['reward_terminal']:+7.1f}"
            f" upd={'Y' if did_update else 'N'}"
        )
        yield TrainStepResult(
            score=ep_reward,
            log_line=f"Episode {ep:4d} | Reward: {ep_reward:8.2f}{comp_line}",
            batch_idx=ep + 1,
            avg_reward=ep_reward,
            success_rate=success_rate,
            termination_counts={termination_reason: 1},
        )


def train_trpo(env, config, *, runs_root="runs", used_configs=None):
    return run_training_loop(
        env,
        config,
        algorithm_name="trpo",
        build_agent=_build_trpo_agent,
        run_iterations=_trpo_iterations,
        default_last_model_name="last_trpo.pth",
        default_best_model_name="best_model_trpo.pth",
        runs_root=runs_root,
        used_configs=used_configs,
    )
