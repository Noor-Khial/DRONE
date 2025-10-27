# metalr_agent.py
# First-order MAML-style Meta-RL (discrete), ideas from:
# https://github.com/tristandeleu/pytorch-maml-rl (MIT License).

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _safe_obs(obs):
    arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    arr[~np.isfinite(arr)] = 0.0
    return torch.as_tensor(arr, dtype=torch.float32, device=device)

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
        # stable init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.8)
                nn.init.constant_(m.bias, 0.0)

    def dist(self, x):
        logits = self.net(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e3, neginf=-1e3)
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
    adapt_horizon: int = 1   # one step to match bandit semantics
    rollout_len: int = 1
    ent_coef: float = 0.01

class MetaRLPolicy:
    def __init__(self, obs_dim: int, act_dim: int, cfg: Optional[MetaConfig] = None):
        self.cfg = cfg or MetaConfig()
        self.base = PolicyNet(obs_dim, act_dim).to(device)
        self.outer_opt = optim.Adam(self.base.parameters(), lr=self.cfg.outer_lr)

    def _rollout(self, env, policy: PolicyNet, start_obs, H: int):
        obs_list, act_list, rew_list, done_list = [], [], [], []
        obs = start_obs
        cum_reward = 0.0
        t_used = 0
        metrics_last = (0,0,0,0,0)

        for _ in range(H):
            dist = policy.dist(_safe_obs(obs))
            a = dist.sample()
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

            obs_list.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            act_list.append(int(a.item()))
            rew_list.append(float(r))
            done_list.append(float(done))

            obs = next_obs
            if done:
                break

        return {
            "obs": np.array(obs_list, dtype=np.float32),
            "acts": np.array(act_list, dtype=np.int64),
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
            z = torch.tensor(0.0, device=device, requires_grad=True)
            return z
        G = np.zeros_like(rews, dtype=np.float32)
        running = 0.0
        for i in range(len(rews)-1, -1, -1):
            running = rews[i] + self.cfg.gamma * running * (1 - data["done"][i])
            G[i] = running
        G_t = torch.as_tensor((G - G.mean()) / (G.std() + 1e-8), dtype=torch.float32, device=device)

        obs_t = torch.as_tensor(data["obs"], dtype=torch.float32, device=device)
        obs_t[~torch.isfinite(obs_t)] = 0.0
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
        obs = current_state
        fast = self._clone_policy()
        inner_opt = optim.SGD(fast.parameters(), lr=self.cfg.inner_lr)

        # inner adaptation (short)
        for _ in range(self.cfg.inner_steps):
            data_i = self._rollout(adapt_env or env, fast, obs, self.cfg.adapt_horizon)
            inner_opt.zero_grad()
            loss_i = self._policy_gradient(data_i, fast)
            loss_i.backward()
            inner_opt.step()
            obs = data_i["next_obs"]

        # outer evaluation
        data_o = self._rollout(env, fast, obs, self.cfg.rollout_len)
        self.outer_opt.zero_grad()
        loss_o = self._policy_gradient(data_o, fast)
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in fast.parameters()]
        for p_base, g in zip(self.base.parameters(), grads):
            p_base.grad = g
        self.outer_opt.step()

        return data_o["next_obs"], float(data_o["cum_reward"]), int(data_o["t_used"]), data_o["metrics"]
