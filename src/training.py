import numpy as np
from agents import REINFORCEAgent, TRPOAgent
import os
from environment import SpaceEnv

def train_reinforce(env, episodes=5000, batch_size=12, save_path="reinforce", lr=3e-4):
    os.makedirs(save_path, exist_ok=True)

    def compute_returns(rewards, gamma):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    agent = REINFORCEAgent(env, lr=lr, entropy_coeff=0.015, vf_coeff=0.5)
    print(f"Device: {agent.device}")
    best_reward = -np.inf
    total_ep = 0

    while total_ep < episodes:
        env_batch_size = min(batch_size, episodes - total_ep)
        envs = [
            SpaceEnv(
                space_size=env.space_size,
                num_asteroids=env.num_asteroids,
                max_steps=env.max_steps
            )
            for _ in range(env_batch_size)
        ]

        trajectories = [([], [], [], [], []) for _ in range(env_batch_size)]
        states = []
        done_flags = [False] * env_batch_size

        for worker_env in envs:
            state, _ = worker_env.reset()
            states.append(state)

        while not all(done_flags):
            active_ids = [i for i, done in enumerate(done_flags) if not done]
            active_states = [states[i] for i in active_ids]
            actions, log_probs, entropies = agent.select_actions(active_states)

            for local_idx, env_idx in enumerate(active_ids):
                action = actions[local_idx]
                next_state, reward, done, truncated, _ = envs[env_idx].step(action)
                trajectory = trajectories[env_idx]

                trajectory[0].append(states[env_idx])
                trajectory[1].append(action)
                trajectory[2].append(reward)
                trajectory[3].append(log_probs[local_idx:local_idx + 1])
                trajectory[4].append(entropies[local_idx:local_idx + 1])

                states[env_idx] = next_state
                if done or truncated:
                    done_flags[env_idx] = True

        batch_rewards = []
        states_all = []
        log_probs_all = []
        entropies_all = []
        returns_all = []

        for trajectory in trajectories:
            traj_states, _, traj_rewards, traj_log_probs, traj_entropies = trajectory
            ep_returns = compute_returns(traj_rewards, agent.gamma)

            states_all.extend(traj_states)
            log_probs_all.extend(traj_log_probs)
            entropies_all.extend(traj_entropies)
            returns_all.extend(ep_returns)
            batch_rewards.append(sum(traj_rewards))

        total_ep += env_batch_size
        big_trajectory = (states_all, returns_all, log_probs_all, entropies_all)
        agent.update(big_trajectory)

        avg_reward = np.mean(batch_rewards)
        max_in_batch = max(batch_rewards)
        batch_idx = (total_ep + batch_size - 1) // batch_size

        print(f"Batch {batch_idx:3d} | "
              f"Avg Reward: {avg_reward:8.2f} | "
              f"Best in batch: {max_in_batch:8.2f} | "
              f"Global best: {best_reward:8.2f}")

        agent.save(os.path.join(save_path, "last_reinforce.pth"))
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(os.path.join(save_path, "best_reinforce.pth"))
            print(f"--> Новая лучшая модель REINFORCE сохранена! Reward = {best_reward:.2f}")

def train_trpo(env, episodes=1000, save_path="best_model_trpo.pth"):
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
            agent.save(save_path)
            print(f"--> Новая лучшая модель TRPO сохранена! Reward = {best_reward:.2f}")