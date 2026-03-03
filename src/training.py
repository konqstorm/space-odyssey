import numpy as np
from agents import REINFORCEAgent, TRPOAgent
import os
from environment import SpaceEnv

def train_reinforce(env, episodes=5000, batch_size=12, lr=3e-4, entropy_coeff=0.002, save_path="reinforce"):
    os.makedirs(save_path, exist_ok=True)

    def compute_returns(rewards, gamma):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    agent = REINFORCEAgent(env, lr=lr, entropy_coeff=entropy_coeff, vf_coeff=0.5)
    best_reward = -np.inf
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
            
            ep_returns = compute_returns(rewards, agent.gamma)

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

        print(f"Batch {batch_idx:3d} | "
              f"Avg Reward: {avg_reward:8.2f} | "
              f"Best in batch: {max_in_batch:8.2f} | "
              f"Global best: {best_reward:8.2f} | "
              f"Success: {success_rate:5.1f}% | "
              f"Avg len: {avg_len:6.1f} | "
              f"Done: {reason_summary}")

        agent.save(os.path.join(save_path, "last_reinforce.pth"))
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(os.path.join(save_path, "best_reinforce.pth"))
            print(f"--> Новая лучшая модель REINFORCE сохранена! Reward = {best_reward:.2f}")

def train_trpo(env, episodes=1000, save_path="trpo"):
    os.makedirs(save_path, exist_ok=True)

    agent = TRPOAgent(env)
    best_reward = -np.inf
    
    for ep in range(episodes):
        state, _ = env.reset()
        done, truncated = False, False
        trajectory = ([], [], [], [], [])
        
        while not (done or truncated):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            trajectory[0].append(state)
            trajectory[1].append(action)
            trajectory[2].append(reward)
            trajectory[3].append(log_prob)
            trajectory[4].append(next_state)
            state = next_state
            
        agent.update(trajectory)
        
        ep_reward = sum(trajectory[2])
        print(f"Episode {ep}, Reward: {ep_reward:.2f}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(save_path,'best_model_trpo.pth'))
            print(f"--> Новая лучшая модель TRPO сохранена! Reward = {best_reward:.2f}")
