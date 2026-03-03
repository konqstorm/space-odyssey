import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import copy
from .policy import PolicyNetwork
from .value_2 import ValueNetwork2


class TRPOAgent:
    def __init__(
        self,
        env,
        lr=1e-3,
        gamma=0.99,
        delta=0.01,
        cg_damping=0.1,
        cg_steps=10,
        line_search_steps=10,
        value_update_steps=10,
    ):
        self.env = env
        self.gamma = gamma
        self.delta = delta
        self.cg_damping = cg_damping
        self.cg_steps = cg_steps
        self.line_search_steps = line_search_steps
        self.value_update_steps = value_update_steps
        # device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(env.observation_space.shape[0]).to(self.device)
        self.value = ValueNetwork2(env.observation_space.shape[0]).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

    def compute_returns(self, rewards):
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        returns = []
        running_return = torch.tensor(0.0, device=self.device)
        for reward in reversed(rewards_t):
            running_return = reward + self.gamma * running_return
            returns.insert(0, running_return)
        return torch.stack(returns).float()

    def surrogate_loss(self, log_probs_old, advantages, states, actions):
        # ensure tensors on correct device
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(np.stack(states)).float().to(self.device)
        else:
            states = states.to(self.device).float()
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(np.stack(actions)).float().to(self.device)
        else:
            actions = actions.to(self.device).float()
        if not isinstance(log_probs_old, torch.Tensor):
            log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=self.device)
        else:
            log_probs_old = log_probs_old.to(self.device).detach()
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.from_numpy(np.stack(advantages)).float().to(self.device)
        else:
            advantages = advantages.to(self.device).float()

        mean, log_std = self.policy(states)
        std = torch.exp(log_std).clamp(1e-3, 1.0)

        actions = actions.clamp(-0.999999, 0.999999)
        raw_actions = torch.atanh(actions)

        normal = torch.distributions.Normal(mean, std)
        log_probs_raw = normal.log_prob(raw_actions).sum(-1)
        squash_correction = torch.log(1 - actions.pow(2) + 1e-6).sum(-1)
        log_probs_new = log_probs_raw - squash_correction
        ratio = torch.exp(log_probs_new - log_probs_old)
        return (ratio * advantages).mean()

    def hessian_vector_product(self, v, states):
        # ensure states on device
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(np.stack(states)).float().to(self.device)
        else:
            states = states.to(self.device).float()

        kl = self.kl_divergence(states)
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        # replace None grads with zeros
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, self.policy.parameters())]
        flat_grad = torch.cat([grad.contiguous().view(-1) for grad in grads])
        kl_v = torch.sum(flat_grad * v)
        grads2 = torch.autograd.grad(kl_v, self.policy.parameters())
        grads2 = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads2, self.policy.parameters())]
        flat_hvp = torch.cat([grad.contiguous().view(-1) for grad in grads2])
        return flat_hvp

    def kl_divergence(self, states):
        # convert states to tensor on correct device
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(np.stack(states)).float().to(self.device)
        else:
            states = states.to(self.device).float()

        # ensure old_policy is on device
        if hasattr(self, 'old_policy'):
            try:
                self.old_policy.to(self.device)
            except Exception:
                pass

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
        r_old_sq = (r @ r)
        for _ in range(nsteps):
            Ap = A(p)
            denom = (p @ Ap).item()
            if denom == 0:
                break
            alpha = r_old_sq / (denom + 1e-8)
            x = x + alpha * p
            r = r - alpha * Ap
            r_new_sq = (r @ r)
            if r_old_sq.item() == 0:
                break
            beta = r_new_sq / (r_old_sq + 1e-8)
            p = r + beta * p
            r_old_sq = r_new_sq
        return x

    def update(self, trajectory):
        states, actions, rewards, log_probs, next_states = trajectory
        self.old_policy = copy.deepcopy(self.policy)
        # ensure old_policy on correct device and eval mode
        try:
            self.old_policy.to(self.device)
        except Exception:
            pass
        self.old_policy.eval()
        # prepare tensors on device
        states_t = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions_t = torch.from_numpy(np.stack(actions)).float().to(self.device)

        returns = self.compute_returns(rewards)
        if returns.numel() > 1:
            returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns_norm = returns

        values_detached = self.value(states_t).squeeze(-1).detach()
        advantages = returns_norm - values_detached
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # stack/convert log_probs safely and detach
        lp_list = []
        for lp in log_probs:
            if isinstance(lp, torch.Tensor):
                lp_list.append(lp.detach().to(self.device).view(-1))
            else:
                lp_list.append(torch.tensor(lp, dtype=torch.float32, device=self.device).view(-1))
        log_probs_old = torch.cat(lp_list).view(-1).detach()

        surr = self.surrogate_loss(log_probs_old, advantages, states_t, actions_t)
        grad = torch.autograd.grad(surr, self.policy.parameters())
        grad = torch.cat([g.view(-1) for g in grad]).detach()
        def Hvp(v):
            return self.hessian_vector_product(v, states_t) + self.cg_damping * v
        step_dir = self.conjugate_gradient(Hvp, grad, nsteps=self.cg_steps)
        shs = 0.5 * (step_dir @ Hvp(step_dir))
        if shs <= 0:
            return
        lm = torch.sqrt(shs / self.delta)
        full_step = step_dir / (lm + 1e-12)
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
        accepted = False
        for _ in range(self.line_search_steps):
            if linesearch(alpha):
                accepted = True
                break
            alpha *= 0.5
        # restore old params if linesearch failed
        if not accepted:
            idx = 0
            for p in self.policy.parameters():
                p_numel = p.numel()
                p.data.copy_(old_params[idx:idx+p_numel].view(p.shape))
                idx += p_numel
        values_target = returns_norm.detach()
        for _ in range(self.value_update_steps):
            self.value_optimizer.zero_grad()
            values = self.value(states_t).squeeze(-1)
            value_loss = (values - values_target).pow(2).mean()
            value_loss.backward()
            self.value_optimizer.step()

    def select_action(self, state, deterministic=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mean, log_std = self.policy(state)
        
        if deterministic:
            action = torch.tanh(mean)
            return action.squeeze(0).detach().cpu().numpy(), None

        std = torch.exp(log_std).clamp(1e-3, 1.0)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)

        log_prob_raw = dist.log_prob(raw_action).sum(-1)
        squash_correction = torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        log_prob = log_prob_raw - squash_correction

        return action.squeeze(0).detach().cpu().numpy(), log_prob.squeeze(0)

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
        except RuntimeError as exc:
            ckpt_in = checkpoint.get("policy", {}).get("fc1.weight", None)
            ckpt_in_dim = ckpt_in.shape[1] if ckpt_in is not None else "unknown"
            env_in_dim = self.env.observation_space.shape[0]
            raise RuntimeError(
                f"Checkpoint is incompatible with current environment observation size: "
                f"checkpoint input_dim={ckpt_in_dim}, env input_dim={env_in_dim}. "
                f"Use a model trained with the same env config (e.g. num_asteroids)."
            ) from exc
        self.policy.eval()
        self.value.eval()
