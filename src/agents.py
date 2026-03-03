# agents.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
import numpy as np
import copy

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_thrust = nn.Linear(hidden_dim, hidden_dim)
        self.head_thrust = nn.Linear(hidden_dim, 2)
        self.fc_torque = nn.Linear(hidden_dim, hidden_dim)
        self.head_torque = nn.Linear(hidden_dim, 2)
        self._init_policy_biases()

    def _init_policy_biases(self):
        # Более нейтральный старт: не "душим" тягу и оставляем умеренный шум.
        nn.init.constant_(self.head_thrust.bias[0], 0.0)
        nn.init.constant_(self.head_torque.bias[0], 0.0)
        nn.init.constant_(self.head_thrust.bias[1], -0.5)
        nn.init.constant_(self.head_torque.bias[1], -0.5)
        
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x_thrust = torch.relu(self.fc_thrust(x))
        x_torque = torch.relu(self.fc_torque(x))

        out_thrust = self.head_thrust(x_thrust)
        out_torque = self.head_torque(x_torque)
        
        # Mean в "raw"-пространстве; ограничение [-1, 1] делаем через tanh уже при сэмплинге action.
        mean = torch.cat([
            out_thrust[:, 0:1],
            out_torque[:, 0:1]
        ], dim=-1)
        log_std = torch.cat([
            out_thrust[:, 1:2],
            out_torque[:, 1:2]
        ], dim=-1)
        log_std = torch.clamp(log_std, -4, 1)
        return mean, log_std


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
            value_loss = 0.5 * (values - returns_t).pow(2).mean()

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


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class TRPOAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99, delta=0.01, cg_damping=0.1):
        self.env = env
        self.gamma = gamma
        self.delta = delta
        self.cg_damping = cg_damping
        # device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(env.observation_space.shape[0]).to(self.device)
        self.value = ValueNetwork(env.observation_space.shape[0]).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mean, log_std = self.policy(state)
        std = torch.exp(log_std)
        cov = torch.diag_embed(std.squeeze(0)**2)
        dist = MultivariateNormal(mean.squeeze(0), cov)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob

    def compute_advantages(self, states, rewards, next_states):
        # ИСПРАВЛЕНИЕ: Используем np.stack вместо np.array для списков numpy-массивов
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        values = self.value(states).squeeze(-1).detach()
        next_values = self.value(next_states).squeeze(-1).detach()

        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        deltas = rewards_t + self.gamma * next_values - values

        advantages = []
        adv = torch.tensor(0.0, device=self.device)
        for delta in reversed(deltas):
            adv = delta + self.gamma * adv
            advantages.insert(0, adv)
        return torch.stack(advantages).float()

    def surrogate_loss(self, log_probs_old, advantages, states, actions):
        states = states.to(self.device) if isinstance(states, torch.Tensor) else states
        actions = actions.to(self.device) if isinstance(actions, torch.Tensor) else actions
        mean, log_std = self.policy(states)
        std = torch.exp(log_std)
        cov = torch.diag_embed(std**2)
        dist = MultivariateNormal(mean, cov)
        log_probs_new = dist.log_prob(actions)
        ratio = torch.exp(log_probs_new - log_probs_old)
        return (ratio * advantages).mean()

    def hessian_vector_product(self, v, states):
        kl = self.kl_divergence(states)
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad = torch.cat([grad.contiguous().view(-1) for grad in grads])
        kl_v = torch.sum(flat_grad * v)
        grads2 = torch.autograd.grad(kl_v, self.policy.parameters())
        flat_hvp = torch.cat([grad.contiguous().view(-1) for grad in grads2])
        return flat_hvp

    def kl_divergence(self, states):
        states = torch.from_numpy(np.array(states)).float() if not isinstance(states, torch.Tensor) else states
        with torch.no_grad():
            mean_old, log_std_old = self.old_policy(states)
        std_old = torch.exp(log_std_old)
        cov_old = torch.diag_embed(std_old**2)
        dist_old = MultivariateNormal(mean_old, cov_old)
        mean, log_std = self.policy(states)
        std = torch.exp(log_std)
        cov = torch.diag_embed(std**2)
        dist = MultivariateNormal(mean, cov)
        kl = torch.distributions.kl.kl_divergence(dist_old, dist).mean()
        return kl

    def conjugate_gradient(self, A, b, nsteps=10):
        x = torch.zeros_like(b).to(self.device)
        r = b.clone()
        p = b.clone()
        for _ in range(nsteps):
            Ap = A(p)
            alpha = (r @ r) / (p @ Ap)
            x += alpha * p
            r -= alpha * Ap
            beta = (r @ r) / (r @ r + 1e-8)
            p = r + beta * p
        return x

    def update(self, trajectory):
        states, actions, rewards, log_probs, next_states = trajectory
        self.old_policy = copy.deepcopy(self.policy)
        advantages = self.compute_advantages(states, rewards, next_states)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # prepare tensors on device
        states_t = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions_t = torch.from_numpy(np.stack(actions)).float().to(self.device)
        log_probs_old = torch.cat([lp.view(1) for lp in log_probs]).to(self.device)

        surr = self.surrogate_loss(log_probs_old, advantages, states_t, actions_t)
        grad = torch.autograd.grad(surr, self.policy.parameters())
        grad = torch.cat([g.view(-1) for g in grad]).detach()
        def Hvp(v):
            return self.hessian_vector_product(v, states_t) + self.cg_damping * v
        step_dir = self.conjugate_gradient(Hvp, grad)
        shs = 0.5 * (step_dir @ Hvp(step_dir))
        lm = torch.sqrt(shs / self.delta)
        full_step = step_dir / lm
        old_params = torch.cat([p.view(-1) for p in self.policy.parameters()]).detach()
        def linesearch(alpha):
            new_params = old_params + alpha * full_step
            idx = 0
            for p in self.policy.parameters():
                p_numel = p.numel()
                p.data.copy_(new_params[idx:idx+p_numel].view(p.shape))
                idx += p_numel
            new_surr = self.surrogate_loss(log_probs_old, advantages, states_t, actions_t)
            kl = self.kl_divergence(states_t)
            if new_surr >= surr and kl <= self.delta:
                return True
            return False
        alpha = 1.0
        for _ in range(10):
            if linesearch(alpha):
                break
            alpha *= 0.5
        values_target = advantages + self.value(states_t).squeeze(-1).detach()
        for _ in range(10):
            self.value_optimizer.zero_grad()
            values = self.value(states_t).squeeze(-1)
            value_loss = (values - values_target).pow(2).mean()
            value_loss.backward()
            self.value_optimizer.step()

    def select_action(self, state, deterministic=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mean, log_std = self.policy(state)
        
        if deterministic:
            return mean.squeeze(0).detach().cpu().numpy(), None
            
        std = torch.exp(log_std)
        cov = torch.diag_embed(std.squeeze(0)**2)
        dist = MultivariateNormal(mean.squeeze(0), cov)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.policy.eval()
        self.value.eval()
