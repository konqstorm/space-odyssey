import numpy as np
from src.agents import REINFORCEAgent
from src.env import SpaceEnv
from .training import TrainStepResult, run_training_loop


def _build_reinforce_agent(env, config):
    return REINFORCEAgent(
        env,
        lr=float(config["lr"]),
        gamma=float(config["gamma"]),
        entropy_coeff=float(config["entropy_coeff"]),
        vf_coeff=float(config["vf_coeff"]),
        update_epochs=int(config["update_epochs"]),
    )


def _compute_returns(rewards, gamma):
    ret = 0.0
    returns = []
    for reward in reversed(rewards):
        ret = reward + gamma * ret
        returns.insert(0, ret)
    return returns


def _reinforce_iterations(env, agent, config):
    episodes = int(config["episodes"])
    batch_size = int(config["batch_size"])
    total_ep = 0
    batch_idx = 0
    env_kwargs = {
        "space_size": env.space_size,
        "num_asteroids": env.num_asteroids,
        "max_steps": env.max_steps,
    }

    while total_ep < episodes:
        batch_trajectories = []
        batch_rewards = []
        batch_lengths = []
        batch_reasons = []
        current_batch_size = min(batch_size, episodes - total_ep)

        envs = [SpaceEnv(**env_kwargs) for _ in range(current_batch_size)]
        states = [local_env.reset()[0] for local_env in envs]
        finished = [False] * current_batch_size
        trajectories = [[[], [], []] for _ in range(current_batch_size)]  # states, actions, rewards
        terminal_reasons = ["unknown"] * current_batch_size

        while not all(finished):
            active_indices = [i for i in range(current_batch_size) if not finished[i]]
            active_states = np.stack([states[i] for i in active_indices], axis=0)
            active_actions = agent.sample_actions(active_states)

            for local_idx, env_idx in enumerate(active_indices):
                action = active_actions[local_idx]
                next_state, reward, done, truncated, info = envs[env_idx].step(action)

                trajectories[env_idx][0].append(states[env_idx])
                trajectories[env_idx][1].append(action)
                trajectories[env_idx][2].append(reward)
                states[env_idx] = next_state

                if done or truncated:
                    finished[env_idx] = True
                    terminal_reasons[env_idx] = info.get("termination_reason", "unknown")

        for traj in trajectories:
            states_ep, actions_ep, rewards_ep = traj
            batch_trajectories.append((states_ep, actions_ep, rewards_ep))
            batch_rewards.append(float(np.sum(rewards_ep)))
            batch_lengths.append(len(rewards_ep))
        batch_reasons.extend(terminal_reasons)

        total_ep += current_batch_size
        batch_idx += 1

        states_all = []
        actions_all = []
        returns_all = []

        for traj in batch_trajectories:
            states, actions, rewards = traj
            
            ep_returns = _compute_returns(rewards, agent.gamma)

            states_all.extend(states)
            actions_all.extend(actions)
            returns_all.extend(ep_returns)

        big_trajectory = (states_all, actions_all, returns_all)

        agent.update(big_trajectory)

        avg_reward = np.mean(batch_rewards)
        max_in_batch = max(batch_rewards)
        avg_len = float(np.mean(batch_lengths)) if batch_lengths else 0.0
        success_rate = 100.0 * sum(1 for reason in batch_reasons if reason == "goal") / max(1, len(batch_reasons))
        reason_counts = {reason: batch_reasons.count(reason) for reason in sorted(set(batch_reasons))}
        reason_summary = ", ".join([f"{k}:{v}" for k, v in reason_counts.items()])

        log_line = (
            f"Batch {batch_idx:3d} | "
            f"Avg Reward: {avg_reward:8.2f} | "
            f"Best in batch: {max_in_batch:8.2f} | "
            f"Success: {success_rate:5.1f}% | "
            f"Avg len: {avg_len:6.1f} | "
            f"Done: {reason_summary}"
        )
        yield TrainStepResult(
            score=float(avg_reward),
            log_line=log_line,
            batch_idx=batch_idx,
            avg_reward=float(avg_reward),
            success_rate=float(success_rate),
            termination_counts=reason_counts,
        )


def train_reinforce(env, config, *, runs_root="runs", used_configs=None):
    return run_training_loop(
        env,
        config,
        algorithm_name="reinforce",
        build_agent=_build_reinforce_agent,
        run_iterations=_reinforce_iterations,
        default_last_model_name="last_reinforce.pth",
        default_best_model_name="best_reinforce.pth",
        runs_root=runs_root,
        used_configs=used_configs,
    )
