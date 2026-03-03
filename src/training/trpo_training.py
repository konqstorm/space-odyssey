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
    for ep in range(episodes):
        state, _ = env.reset()
        done, truncated = False, False
        trajectory = ([], [], [], [], [])
        termination_reason = "unknown"
        
        while not (done or truncated):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            trajectory[0].append(state)
            trajectory[1].append(action)
            trajectory[2].append(reward)
            trajectory[3].append(log_prob)
            trajectory[4].append(next_state)
            state = next_state
            if done or truncated:
                termination_reason = info.get("termination_reason", "unknown")
            
        agent.update(trajectory)
        
        ep_reward = float(sum(trajectory[2]))
        success_rate = 100.0 if termination_reason == "goal" else 0.0
        yield TrainStepResult(
            score=ep_reward,
            log_line=f"Episode {ep:4d} | Reward: {ep_reward:8.2f}",
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
