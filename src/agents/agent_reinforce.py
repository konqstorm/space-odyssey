import torch
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from .policy import PolicyNetwork
from .value import ValueNetwork


class REINFORCEAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, entropy_coeff=0.002, vf_coeff=0.5, update_epochs=4):
        self.env = env
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.vf_coeff = vf_coeff
        self.update_epochs = update_epochs
        # device selection (use CUDA if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(env.observation_space.shape[0]).to(self.device)
        self.value = ValueNetwork(env.observation_space.shape[0]).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

    def select_action(self, state, deterministic=False):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mean, log_std = self.policy(state_t)
        std = torch.exp(log_std).clamp(1e-3, 1.0)

        if deterministic:
            action = torch.tanh(mean)
            return action.squeeze(0).detach().cpu().numpy(), None, None

        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action.squeeze(0).detach().cpu().numpy(), log_prob, entropy

    def sample_action(self, state):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, log_std = self.policy(state_t)
            std = torch.exp(log_std).clamp(1e-3, 1.0)
            dist = Normal(mean, std)
            raw_action = dist.sample()
            action = torch.tanh(raw_action)
        return action.squeeze(0).cpu().numpy()

    def sample_actions(self, states):
        states_t = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():
            mean, log_std = self.policy(states_t)
            std = torch.exp(log_std).clamp(1e-3, 1.0)
            dist = Normal(mean, std)
            raw_actions = dist.sample()
            actions = torch.tanh(raw_actions)
        return actions.cpu().numpy()

    def update(self, trajectory):
        states, actions, returns = trajectory
        
        states_t = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions_t = torch.from_numpy(np.stack(actions)).float().to(self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)

        raw_actions = torch.atanh(actions_t.clamp(-0.999999, 0.999999))

        for _ in range(self.update_epochs):
            mean, log_std = self.policy(states_t)
            std = torch.exp(log_std).clamp(1e-3, 1.0)
            dist = Normal(mean, std)

            log_probs_raw = dist.log_prob(raw_actions).sum(-1)
            squash_correction = torch.log(1 - actions_t.pow(2) + 1e-6).sum(-1)
            log_probs = log_probs_raw - squash_correction
            entropies = dist.entropy().sum(-1)

            values = self.value(states_t).squeeze(-1)

            advantages = returns_t - values.detach()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(log_probs * advantages).mean()
            entropy_loss = -self.entropy_coeff * entropies.mean()
            value_loss = self.vf_coeff * (values - returns_t).pow(2).mean()

            total_policy_loss = policy_loss + entropy_loss

            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
            self.value_optimizer.step()

    def save(self, path="best_reinforce.pth"):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict()
        }, path)

    def load(self, path="best_reinforce.pth"):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy'])
        self.value.load_state_dict(ckpt['value'])
        self.policy.eval()
        self.value.eval()
