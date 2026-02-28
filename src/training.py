import numpy as np
from agents import REINFORCEAgent, TRPOAgent
import os

def train_reinforce(env, episodes=5000, batch_size=12, save_path="reinforce"):
    os.makedirs(save_path, exist_ok=True)

    def compute_returns(rewards, gamma):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    agent = REINFORCEAgent(env, lr=3e-4, entropy_coeff=0.015, vf_coeff=0.5)
    best_reward = -np.inf
    total_ep = 0

    while total_ep < episodes:
        batch_trajectories = []
        batch_rewards = []

        for _ in range(batch_size):
            state, _ = env.reset()
            done = truncated = False
            trajectory = ([], [], [], [], [])   # states, actions, rewards, log_probs, entropies

            while not (done or truncated):
                action, log_prob, entropy = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)

                trajectory[0].append(state)
                trajectory[1].append(action)
                trajectory[2].append(reward)
                trajectory[3].append(log_prob)
                trajectory[4].append(entropy)

                state = next_state

            batch_trajectories.append(trajectory)
            batch_rewards.append(sum(trajectory[2]))
            total_ep += 1

        states_all = []
        actions_all = []
        rewards_all = []
        log_probs_all = []
        entropies_all = []
        returns_all = []

        for traj in batch_trajectories:
            states, actions, rewards, log_probs, entropies = traj
            
            ep_returns = compute_returns(rewards, agent.gamma)

            states_all.extend(states)
            actions_all.extend(actions)
            rewards_all.extend(rewards)
            log_probs_all.extend(log_probs)
            entropies_all.extend(entropies)
            returns_all.extend(ep_returns)

        big_trajectory = (states_all, returns_all, log_probs_all, entropies_all)

        agent.update(big_trajectory)

        avg_reward = np.mean(batch_rewards)
        max_in_batch = max(batch_rewards)

        print(f"Batch {total_ep//batch_size:3d} | "
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