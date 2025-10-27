# metalr_agent.py
# Simple first-order MAML-style meta-policy gradient (discrete).
# Reference (ideas & structure): https://github.com/tristandeleu/pytorch-maml-rl (MIT License).

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def dist(self, x):
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)

    @property
    def obs_dim(self):
        return self.net[0].in_features

    @property
    def act_dim(self):
        return self.net[-1].out_features

@dataclass
class MetaConfig:
    gamma: float = 0.99
    inner_lr: float = 1e-2
    outer_lr: float = 1e-3
    inner_steps: int = 1
    adapt_horizon: int = 32
    rollout_len: int = 64
    ent_coef: float = 0.01

class MetaRLPolicy:
    def __init__(self, obs_dim: int, act_dim: int, cfg: Optional[MetaConfig] = None):
        self.cfg = cfg or MetaConfig()
        self.base = PolicyNet(obs_dim, act_dim).to(device)
        self.outer_opt = optim.Adam(self.base.parameters(), lr=self.cfg.outer_lr)

    def _rollout(self, env, policy: PolicyNet, start_obs, H: int):
        obs_list, act_list, logp_list, rew_list, done_list = [], [], [], [], []
        obs = start_obs
        cum_reward = 0.0
        t_used = 0
        metrics_last = (0, 0, 0, 0, 0)

        for _ in range(H):
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = policy.dist(o)
            a = dist.sample()
            logp = dist.log_prob(a)
            step_out = env.step(int(a.item()))
            if len(step_out) == 5:
                next_obs, r, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                next_obs, r, done, info = step_out

            cum_reward += float(r)
            t_used += 1
            if isinstance(info, dict) and "metrics" in info:
                metrics_last = info["metrics"]

            obs_list.append(obs)
            act_list.append(int(a.item()))
            logp_list.append(float(logp.item()))
            rew_list.append(float(r))
            done_list.append(float(done))

            obs = next_obs
            if done:
                obs = env.reset()
                break

        return {
            "obs": np.array(obs_list, dtype=np.float32),
            "acts": np.array(act_list, dtype=np.int64),
            "logp": np.array(logp_list, dtype=np.float32),
            "rews": np.array(rew_list, dtype=np.float32),
            "done": np.array(done_list, dtype=np.float32),
            "next_obs": obs,
            "cum_reward": cum_reward,
            "t_used": t_used,
            "metrics": metrics_last,
        }

    def _policy_gradient(self, data, policy: PolicyNet):
        rews = data["rews"]
        if rews.size == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        # discounted returns
        G = np.zeros_like(rews, dtype=np.float32)
        running = 0.0
        for i in range(len(rews) - 1, -1, -1):
            running = rews[i] + self.cfg.gamma * running * (1 - data["done"][i])
            G[i] = running
        G_t = torch.as_tensor((G - G.mean()) / (G.std() + 1e-8), dtype=torch.float32, device=device)

        obs_t = torch.as_tensor(data["obs"], dtype=torch.float32, device=device)
        acts_t = torch.as_tensor(data["acts"], dtype=torch.int64, device=device)

        dist = policy.dist(obs_t)
        logp = dist.log_prob(acts_t)
        loss = -(logp * G_t).mean() - self.cfg.ent_coef * dist.entropy().mean()
        return loss

    def _clone_policy(self):
        clone = PolicyNet(self.base.obs_dim, self.base.act_dim).to(device)
        clone.load_state_dict(self.base.state_dict())
        return clone

    def run(self, env, current_state, adapt_env=None):
        # One meta-iteration: inner adaptation then outer update.
        obs = current_state

        # Inner-loop adaptation on a short horizon
        fast = self._clone_policy()
        inner_opt = optim.SGD(fast.parameters(), lr=self.cfg.inner_lr)

        for _ in range(self.cfg.inner_steps):
            data_i = self._rollout(adapt_env or env, fast, obs, self.cfg.adapt_horizon)
            inner_opt.zero_grad()
            loss_i = self._policy_gradient(data_i, fast)
            loss_i.backward()
            inner_opt.step()
            obs = data_i["next_obs"]

        # Outer loop: evaluate adapted fast policy on rollout_len
        data_o = self._rollout(env, fast, obs, self.cfg.rollout_len)
        # First-order MAML: approximate gradient transfer to base
        self.outer_opt.zero_grad()
        loss_o = self._policy_gradient(data_o, fast)
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in fast.parameters()]
        for p_base, g in zip(self.base.parameters(), grads):
            p_base.grad = g
        self.outer_opt.step()

        return data_o["next_obs"], float(data_o["cum_reward"]), int(data_o["t_used"]), data_o["metrics"]
