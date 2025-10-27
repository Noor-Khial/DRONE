#!/usr/bin/env python3
import os
import csv
import time
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime

# Core algorithms / utils from your repo
from exp3 import Exp3Algorithm
from ucb import UCBAlgorithm
from utils import read_policy_at_t
from DDPG.rlPolicy import DDPGPolicy
from env.gym_env import GridWorldEnv

# TWTD toggles
import throughWallDetection.target_detection as td

# --- τ-mini-batched EXP3 (if present) ---
try:
    from exp3tau import Exp3TauAlgorithm
except Exception:
    Exp3TauAlgorithm = None  # allow running without the τ variant

# --- PPO / Meta-RL (optional; guarded imports) ---
try:
    from ppo_agent import PPOPolicy, PPOConfig
except Exception:
    PPOPolicy, PPOConfig = None, None

try:
    from metalr_agent import MetaRLPolicy, MetaConfig
except Exception:
    MetaRLPolicy, MetaConfig = None, None

RESULTS_DIR = Path("results")
FIG_DIR = Path("FIGs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def now_tag():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

# ------------------------------
# Helpers
# ------------------------------
def make_env():
    return GridWorldEnv(render_mode=None, size=55)

def make_actions():
    return [
        DDPGPolicy(model_path='experiments/exp-e/env1',  obstacles_file='env/targets_movements/shifted_obstacles/obstacles_1.0.json'),
        DDPGPolicy(model_path='experiments/exp-e/env08', obstacles_file='env/targets_movements/shifted_obstacles/obstacles_0.8.json'),
        DDPGPolicy(model_path='experiments/exp-e/env06', obstacles_file='env/targets_movements/shifted_obstacles/obstacles_0.6.json'),
    ]

def write_table(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def regret_bound(xs, K=3):
    xs = np.asarray(xs)
    return np.sqrt(xs * K * np.log(max(K, 2)))

# ------------------------------
# Baseline episode loop (your original)
# ------------------------------
def run_episode_loop(algorithm, actions, env, num_iterations, rewardMin=-40, rewardMax=35):
    current_state = env.reset()
    bestAction = 0
    regrets, r_bounds, rewards = [], [], []
    wall0 = time.perf_counter()

    # accumulators for per-iteration env.inference_me()
    sum_total_distance = 0.0
    sum_targets_found = 0.0
    sum_collisions = 0.0
    sum_time_steps = 0.0
    sum_cum_reward = 0.0

    for iteration in range(num_iterations):
        if iteration == 50:
            bestAction = 2
            current_state = env.reset()

        current_policy = read_policy_at_t('exp3/sampled_policies(3policies3).txt', iteration)
        choice = algorithm.run()  # int index
        state, reward, t, metrics = actions[choice].run(current_state, current_policy, num_episodes=1)

        _, bestActionReward, _, _ = actions[bestAction].run(current_state, bestAction, num_episodes=1)

        if hasattr(algorithm, "update"):
            algorithm.update(iteration, reward, bestActionReward)

        wr, rb = algorithm.get_regret() if hasattr(algorithm, "get_regret") else (0.0, 0.0)
        regrets.append(abs(float(wr)))
        r_bounds.append(float(rb))
        rewards.append(float(reward))
        current_state = state

        # collect metrics every iteration
        td_i, tf_i, col_i, ts_i, cr_i = metrics
        sum_total_distance += float(td_i)
        sum_targets_found += float(tf_i)
        sum_collisions += float(col_i)
        sum_time_steps += float(ts_i)
        sum_cum_reward += float(cr_i)

    # averages across iterations
    denom = max(1, num_iterations)
    total_distance = sum_total_distance / denom
    targets_found = sum_targets_found / denom
    collisions = sum_collisions / denom
    time_steps = sum_time_steps / denom
    cum_reward = sum_cum_reward / denom

    runtime_s = time.perf_counter() - wall0
    return {
        "regrets": np.array(regrets),
        "bounds": np.array(r_bounds),
        "rewards": np.array(rewards),
        "sum": dict(
            total_distance=total_distance,
            targets_found=targets_found,
            collisions=collisions,
            time_steps=time_steps,
            cum_reward=cum_reward,
            runtime_s=runtime_s,
        ),
    }

# ------------------------------
# Loop that also tracks τ-policy bound (your original)
# ------------------------------
def run_episode_loop_collect_tau(algorithm, actions, env, num_iterations, rewardMin=-40, rewardMax=35):
    current_state = env.reset()
    bestAction = 0
    regrets, r_bounds, rewards = [], [], []
    tau_bounds = []
    wall0 = time.perf_counter()

    # accumulators for per-iteration env.inference_me()
    sum_total_distance = 0.0
    sum_targets_found = 0.0
    sum_collisions = 0.0
    sum_time_steps = 0.0
    sum_cum_reward = 0.0

    for iteration in range(num_iterations):
        if iteration == 50:
            bestAction = 2
            current_state = env.reset()

        current_policy = read_policy_at_t('exp3/sampled_policies(3policies3).txt', iteration)
        choice = algorithm.run()  # int index
        state, reward, t, metrics= actions[choice].run(current_state, current_policy, num_episodes=1)

        _, bestActionReward, _, _ = actions[bestAction].run(current_state, bestAction, num_episodes=1)
        if hasattr(algorithm, "update"):
            algorithm.update(iteration, reward, bestActionReward)

        wr, rb = algorithm.get_regret() if hasattr(algorithm, "get_regret") else (0.0, 0.0)
        regrets.append(abs(float(wr)))
        r_bounds.append(float(rb))
        rewards.append(float(reward))

        if hasattr(algorithm, "get_tau_policy_regret_bound"):
            tau_bounds.append(float(algorithm.get_tau_policy_regret_bound()))

        current_state = state

        td_i, tf_i, col_i, ts_i, cr_i = metrics
        sum_total_distance += float(td_i)
        sum_targets_found += float(tf_i)
        sum_collisions += float(col_i)
        sum_time_steps += float(ts_i)
        sum_cum_reward += float(cr_i)

    # averages across iterations
    denom = max(1, num_iterations)
    total_distance = sum_total_distance / denom
    targets_found = sum_targets_found / denom
    collisions = sum_collisions / denom
    time_steps = sum_time_steps / denom
    cum_reward = sum_cum_reward / denom

    runtime_s = time.perf_counter() - wall0
    return {
        "regrets": np.array(regrets),
        "bounds": np.array(r_bounds),
        "rewards": np.array(rewards),
        "tau_bounds": np.array(tau_bounds) if len(tau_bounds) > 0 else None,
        "sum": dict(
            total_distance=total_distance,
            targets_found=targets_found,
            collisions=collisions,
            time_steps=time_steps,
            cum_reward=cum_reward,
            runtime_s=runtime_s,
        ),
    }

# ============================================================
# E1 — TWTD vs LOS
# ============================================================
def run_experiment_E1(num_iterations=100):
    actions = make_actions()
    env = make_env()

    td.set_twtd_mode(True); td.set_twtd_gain(1.0); td.set_adversarial_level(0.0)
    exp3 = Exp3Algorithm(len(actions), gamma=0.15)
    res_twtd = run_episode_loop(exp3, actions, env, num_iterations)

    env = make_env()
    td.set_twtd_mode(False); td.set_twtd_gain(1.0)
    exp3_l = Exp3Algorithm(len(actions), gamma=0.15)
    res_los = run_episode_loop(exp3_l, actions, env, num_iterations)

    header = ["Mode","MeanRegret","TargetsFound","Collisions","CumReward","TimeSteps","Runtime(s)"]
    rows = [
        ["TWTD", np.mean(res_twtd["regrets"]), res_twtd["sum"]["targets_found"], res_twtd["sum"]["collisions"], res_twtd["sum"]["cum_reward"], res_twtd["sum"]["time_steps"], f"{res_twtd['sum']['runtime_s']:.2f}"],
        ["LOS",  np.mean(res_los["regrets"]),  res_los["sum"]["targets_found"],  res_los["sum"]["collisions"],  res_los["sum"]["cum_reward"],  res_los["sum"]["time_steps"],  f"{res_los['sum']['runtime_s']:.2f}"],
    ]
    write_table(RESULTS_DIR / "E1_TWTDvsLOS.csv", header, rows)

    import matplotlib.pyplot as plt
    xs = np.arange(1, len(res_twtd["regrets"]) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(xs, res_twtd["regrets"], label="TWTD")
    plt.plot(xs, res_los["regrets"], label="LOS")
    plt.plot(xs, regret_bound(xs), "--", label="√(t K log K)")
    plt.xlabel("Iteration"); plt.ylabel("|Weak Regret|"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_R8-53_TWTDvsLOS.png"); plt.close()

# ============================================================
# E2 — Processing Gain Sweep
# ============================================================
def run_experiment_E2(num_iterations=100):
    actions = make_actions()
    header = ["G_proc","MeanRegret","TargetsFound","Collisions","CumReward","TimeSteps","Runtime(s)"]
    rows = []
    xs_plot, to_plot = None, []

    for gp in [0.6, 0.8, 1.0, 1.2]:
        env = make_env()
        td.set_twtd_mode(True); td.set_twtd_gain(gp)
        exp3 = Exp3Algorithm(len(actions), gamma=0.15)
        res = run_episode_loop(exp3, actions, env, num_iterations)
        rows.append([gp, np.mean(res["regrets"]), res["sum"]["targets_found"], res["sum"]["collisions"], res["sum"]["cum_reward"], res["sum"]["time_steps"], f"{res['sum']['runtime_s']:.2f}"])
        if xs_plot is None: xs_plot = np.arange(1, len(res["regrets"])+1)
        to_plot.append((gp, res["regrets"]))

    write_table(RESULTS_DIR / "E2_Gproc.csv", header, rows)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    for gp, reg in to_plot:
        plt.plot(xs_plot, reg, label=f"G_proc={gp}")
    plt.plot(xs_plot, regret_bound(xs_plot), "--", label="√(t K log K)")
    plt.xlabel("Iteration"); plt.ylabel("|Weak Regret|"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_R8-54_Gproc.png"); plt.close()

# ============================================================
# E3 — EXP3 gamma selection
# ============================================================
def run_experiment_E3(num_iterations=100):
    actions = make_actions()
    header = ["Gamma","MeanRegret","StdRegret","TargetsFound","Collisions","CumReward"]
    rows = []

    for g in [0.05, 0.10, 0.15, 0.20]:
        env = make_env()
        td.set_twtd_mode(True)
        exp3 = Exp3Algorithm(len(actions), gamma=g)
        res = run_episode_loop(exp3, actions, env, num_iterations)
        rows.append([g, np.mean(res["regrets"]), np.std(res["regrets"]), res["sum"]["targets_found"], res["sum"]["collisions"], res["sum"]["cum_reward"]])

    write_table(RESULTS_DIR / "E3_Exp3Params.csv", header, rows)

# ============================================================
# E4 — Stability across random seeds
# ============================================================
def run_experiment_E4(num_iterations=100):
    actions = make_actions()
    header = ["Seed","MeanRegret","StdRegret","TargetsFound","Collisions"]
    rows, regs = [], []

    for seed in [42, 99, 2024]:
        random.seed(seed); np.random.seed(seed)
        env = make_env()
        td.set_twtd_mode(True)
        exp3 = Exp3Algorithm(len(actions), gamma=0.15)
        res = run_episode_loop(exp3, actions, env, num_iterations)
        rows.append([seed, np.mean(res["regrets"]), np.std(res["regrets"]), res["sum"]["targets_found"], res["sum"]["collisions"]])
        regs.append(res["regrets"])

    write_table(RESULTS_DIR / "E4_Stability.csv", header, rows)

# ============================================================
# E5 — EXP3 under adversarial levels
# ============================================================
def run_experiment_E5(num_iterations=100):
    actions = make_actions()
    levels = [0.0, 0.3, 0.6, 1.0]
    curves, rows = [], []
    xs = None
    header = ["AdversarialLevel","MeanRegret","TargetsFound","Collisions","CumReward","TimeSteps","Runtime(s)"]

    for lvl in levels:
        env = make_env()
        td.set_twtd_mode(True); td.set_twtd_gain(1.0); td.set_adversarial_level(lvl)
        exp3 = Exp3Algorithm(len(actions), gamma=0.15)
        res = run_episode_loop(exp3, actions, env, num_iterations)
        if xs is None: xs = np.arange(1, len(res["regrets"]) + 1)
        curves.append((lvl, res["regrets"]))
        rows.append([lvl, np.mean(res["regrets"]), res["sum"]["targets_found"], res["sum"]["collisions"], res["sum"]["cum_reward"], res["sum"]["time_steps"], f"{res['sum']['runtime_s']:.2f}"])

    write_table(RESULTS_DIR / "E5_AdversarialLevels.csv", header, rows)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    for lvl, reg in curves:
        plt.plot(xs, reg, label=f"adv={lvl}")
    plt.plot(xs, regret_bound(xs), "--", label="Upper Bound")
    plt.xlabel("Time Step (t)"); plt.ylabel("|Cummulative Regret|"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_R8-36_AdversarialLevels.png"); plt.close()

# ============================================================
# E7 — Computational overhead
# ============================================================
def run_experiment_E7(num_iterations=100):
    actions = make_actions()
    header = ["Algorithm","Runtime(s)","Steps","Steps/s","CumReward"]
    rows = []

    # UCB
    env = make_env(); td.set_twtd_mode(True)
    ucb = UCBAlgorithm(len(actions))
    res = run_episode_loop(ucb, actions, env, num_iterations)
    steps = res["sum"]["time_steps"] or num_iterations
    rows.append(["UCB", f"{res['sum']['runtime_s']:.2f}", steps, f"{steps/max(res['sum']['runtime_s'],1e-6):.1f}", f"{res['sum']['cum_reward']:.2f}"])

    # EXP3
    env = make_env(); td.set_twtd_mode(True)
    exp3 = Exp3Algorithm(len(actions), gamma=0.15)
    res = run_episode_loop(exp3, actions, env, num_iterations)
    steps = res["sum"]["time_steps"] or num_iterations
    rows.append(["EXP3", f"{res['sum']['runtime_s']:.2f}", steps, f"{steps/max(res['sum']['runtime_s'],1e-6):.1f}", f"{res['sum']['cum_reward']:.2f}"])

    # Pure RL
    env = make_env(); td.set_twtd_mode(True)
    class _FakeAlgo:
        def run(self): return 0
        def update(self, *a, **k): pass
        def get_regret(self): return 0.0, 0.0
    fake = _FakeAlgo()
    res = run_episode_loop(fake, actions, env, num_iterations)
    steps = res["sum"]["time_steps"] or num_iterations
    rows.append(["PureRL(0)", f"{res['sum']['runtime_s']:.2f}", steps, f"{steps/max(res['sum']['runtime_s'],1e-6):.1f}", f"{res['sum']['cum_reward']:.2f}"])

    write_table(RESULTS_DIR / "E7_Computation.csv", header, rows)

# ============================================================
# E8 — Baselines compared
# ============================================================
def run_experiment_E8(num_iterations=100):
    actions = make_actions()
    header = ["Algorithm","MeanRegret","TargetsFound","Collisions","CumReward","TimeSteps","Runtime(s)"]
    rows = []
    curves, xs = {}, None

    # EXP3
    env = make_env(); td.set_twtd_mode(True)
    exp3 = Exp3Algorithm(len(actions), gamma=0.15)
    res = run_episode_loop(exp3, actions, env, num_iterations)
    rows.append(["EXP3", np.mean(res["regrets"]), res["sum"]["targets_found"], res["sum"]["collisions"], res["sum"]["cum_reward"], res["sum"]["time_steps"], f"{res['sum']['runtime_s']:.2f}"])
    curves["EXP3"] = res["regrets"]
    if xs is None: xs = np.arange(1, len(res["regrets"]) + 1)

    # UCB
    env = make_env(); td.set_twtd_mode(True)
    ucb = UCBAlgorithm(len(actions))
    res = run_episode_loop(ucb, actions, env, num_iterations)
    rows.append(["UCB", np.mean(res["regrets"]), res["sum"]["targets_found"], res["sum"]["collisions"], res["sum"]["cum_reward"], res["sum"]["time_steps"], f"{res['sum']['runtime_s']:.2f}"])
    curves["UCB"] = res["regrets"]

    # Pure RL
    env = make_env(); td.set_twtd_mode(True)
    class _FakeAlgo:
        def run(self): return 0
        def update(self, *a, **k): pass
        def get_regret(self): return 0.0, 0.0
    fake = _FakeAlgo()
    res = run_episode_loop(fake, actions, env, num_iterations)
    rows.append(["PureRL(0)", float('nan'), res["sum"]["targets_found"], res["sum"]["collisions"], res["sum"]["cum_reward"], res["sum"]["time_steps"], f"{res['sum']['runtime_s']:.2f}"])

    write_table(RESULTS_DIR / "E8_Baselines.csv", header, rows)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    for algo, reg in curves.items():
        plt.plot(xs, reg, label=algo)
    plt.plot(xs, regret_bound(xs), "--", label="√(t K log K)")
    plt.xlabel("Iteration"); plt.ylabel("|Weak Regret|"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_R8-57_Baselines.png"); plt.close()

# ============================================================
# E9 — EXP3 vs EXP3-τ (mini-batched), with m=1
# ============================================================
def run_experiment_E9(num_iterations=100, taus=(10,20,30), gamma=0.15, m_memory=1):
    if Exp3TauAlgorithm is None:
        print("[WARN] Exp3TauAlgorithm not available; skipping E9.")
        return

    actions = make_actions()
    header = ["Algorithm","Tau","MeanRegret","StdRegret","TargetsFound","Collisions","CumReward","TimeSteps","Runtime(s)"]
    rows = []
    curves, tau_curve_bounds = {}, {}
    xs = None

    # Baseline EXP3
    env = make_env()
    td.set_twtd_mode(True); td.set_twtd_gain(1.0); td.set_adversarial_level(0.0)
    exp3 = Exp3Algorithm(len(actions), gamma=gamma)
    res_exp3 = run_episode_loop_collect_tau(exp3, actions, env, num_iterations)
    rows.append(["EXP3", "-", np.mean(res_exp3["regrets"]), np.std(res_exp3["regrets"]),
                 res_exp3["sum"]["targets_found"], res_exp3["sum"]["collisions"],
                 res_exp3["sum"]["cum_reward"], res_exp3["sum"]["time_steps"],
                 f"{res_exp3['sum']['runtime_s']:.2f}"])
    curves["EXP3"] = res_exp3["regrets"]
    if xs is None: xs = np.arange(1, len(res_exp3["regrets"]) + 1)

    # EXP3-τ variants (ensure τ > m=1)
    taus = tuple(sorted(set([t for t in taus if t >= 2])))
    for tau in taus:
        env = make_env()
        td.set_twtd_mode(True); td.set_twtd_gain(1.0); td.set_adversarial_level(0.0)
        exp3tau = Exp3TauAlgorithm(len(actions), gamma=gamma, tau=tau, rewardMin=-40, rewardMax=35, m_memory=m_memory)
        res_tau = run_episode_loop_collect_tau(exp3tau, actions, env, num_iterations)

        rows.append([f"EXP3-τ", tau, np.mean(res_tau["regrets"]), np.std(res_tau["regrets"]),
                     res_tau["sum"]["targets_found"], res_tau["sum"]["collisions"],
                     res_tau["sum"]["cum_reward"], res_tau["sum"]["time_steps"],
                     f"{res_tau['sum']['runtime_s']:.2f}"])

        label = f"EXP3-τ({tau})"
        curves[label] = res_tau["regrets"]
        if res_tau["tau_bounds"] is not None and len(res_tau["tau_bounds"]) > 0:
            tau_curve_bounds[label] = res_tau["tau_bounds"]

    write_table(RESULTS_DIR / "E9_Exp3_vs_Exp3Tau.csv", header, rows)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    for name, reg in curves.items():
        plt.plot(xs, reg, label=name)
    plt.plot(xs, regret_bound(xs), "--", label="√(t K log K)")
    plt.xlabel("Iteration"); plt.ylabel("|Weak Regret|"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_R8-58_Exp3_vs_Exp3Tau.png"); plt.close()

    if len(tau_curve_bounds) > 0:
        plt.figure(figsize=(8,6))
        for name, tb in tau_curve_bounds.items():
            txs = np.arange(1, len(tb) + 1)
            plt.plot(txs, tb, label=f"{name} τ-policy bound")
        plt.xlabel("Iteration"); plt.ylabel("τ-Policy Regret Bound (diagnostic)"); plt.legend(); plt.tight_layout()
        plt.savefig(FIG_DIR / "Fig_R8-59_TauPolicyBound.png"); plt.close()

# ============================================================
# NEW — PPO & Meta-RL vs Bandits using an adapter
# ============================================================
class BanditEnvAdapter:
    """
    Adapts your GridWorld+DDPG setup so PPO/Meta-RL can act as arm-selectors.
    step(arm_index) -> actions[arm_index].run(current_state, arm_index, num_episodes=1)
    Returns (next_obs_vec, reward, done=False, {"metrics": metrics})
    """
    def __init__(self, base_env, actions):
        self.base_env = base_env
        self.actions = actions
        self.current_state = None
        self.iteration = 0

    @staticmethod
    def _to_vec(state):
        # handle dicts/tuples/arrays robustly
        if isinstance(state, dict):
            for k in ("obs", "state", "x"):
                if k in state:
                    state = state[k]
                    break
            else:
                state = np.concatenate([np.ravel(np.asarray(v)) for v in state.values()])
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        arr[~np.isfinite(arr)] = 0.0
        return arr

    def reset(self):
        s = self.base_env.reset()
        self.iteration = 0
        self.current_state = s
        return self._to_vec(s)

    def step(self, arm_index: int):
        state, reward, t, metrics = self.actions[arm_index].run(self.current_state, arm_index, num_episodes=1)
        self.current_state = state
        self.iteration += 1
        svec = self._to_vec(state)
        return svec, float(reward), False, {"metrics": metrics}

def _run_rl_single(agent, base_env, actions, num_iterations: int):
    """
    Runs PPO or Meta-RL as a selector over arms via the adapter.
    Uses the original env state (not the flattened obs) for the baseline arm call.
    Keeps your t=50 switch/reset semantics.
    """
    adapter = BanditEnvAdapter(base_env, actions)
    obs = adapter.reset()
    bestAction = 0
    regrets, rewards, r_bounds = [], [], []
    sum_total_distance = sum_targets_found = sum_collisions = sum_time_steps = sum_cum_reward = 0.0

    # enforce one-step semantics
    if hasattr(agent, "cfg"):
        if hasattr(agent.cfg, "rollout_len"): agent.cfg.rollout_len = 1
        if hasattr(agent.cfg, "adapt_horizon"): agent.cfg.adapt_horizon = 1

    for iteration in range(num_iterations):
        if iteration == 50:
            bestAction = 2
            obs = adapter.reset()

        # keep schedule aligned (if env depends on it internally)
        _ = read_policy_at_t('exp3/sampled_policies(3policies3).txt', iteration)

        # One step by the agent through the adapter
        next_obs, rew, t_used, metrics = agent.run(adapter, obs)

        # Regret baseline: use the ORIGINAL underlying state for the arm rollout
        original_state = adapter.current_state
        _, bestActionReward, _, _ = actions[bestAction].run(original_state, bestAction, num_episodes=1)

        regrets.append(abs(float(bestActionReward - rew)))
        r_bounds.append(0.0)
        rewards.append(float(rew))
        obs = next_obs

        if isinstance(metrics, (list, tuple)) and len(metrics) == 5:
            td_i, tf_i, col_i, ts_i, cr_i = metrics
            sum_total_distance += float(td_i)
            sum_targets_found += float(tf_i)
            sum_collisions += float(col_i)
            sum_time_steps += float(ts_i)
            sum_cum_reward += float(cr_i)

    denom = max(1, num_iterations)
    return {
        "regrets": np.array(regrets),
        "bounds": np.array(r_bounds),
        "rewards": np.array(rewards),
        "sum": dict(
            total_distance=sum_total_distance/denom,
            targets_found=sum_targets_found/denom,
            collisions=sum_collisions/denom,
            time_steps=sum_time_steps/denom,
            cum_reward=sum_cum_reward/denom,
            runtime_s=0.0,
        ),
    }

def run_experiment_RL_vs_Bandits(num_iterations=100, tau=10, gamma=0.15):
    """
    Runs PPO, Meta-RL, UCB, EXP3, and (if available) EXP3-τ in your switching-policy setup.
    Saves:
      - results/Switching_Baselines.csv
      - FIGs/Fig_Switching_AvgRegret.png
    """
    actions = make_actions()
    header = ["Algorithm","MeanRegret","TargetsFound","Collisions","CumReward","TimeSteps","Runtime(s)"]
    rows = []
    curves, xs = {}, None

    # PPO (optional)
    if PPOPolicy is not None:
        env = make_env(); td.set_twtd_mode(True)
        # Use adapter to derive obs_dim; actions define act_dim
        adapter_probe = BanditEnvAdapter(env, actions)
        obs_vec = adapter_probe.reset()
        obs_dim = int(np.asarray(obs_vec).size)
        act_dim = len(actions)
        ppo = PPOPolicy(obs_dim, act_dim)
        res_ppo = _run_rl_single(ppo, env, actions, num_iterations)
        rows.append(["PPO", np.mean(res_ppo["regrets"]), res_ppo["sum"]["targets_found"], res_ppo["sum"]["collisions"], res_ppo["sum"]["cum_reward"], res_ppo["sum"]["time_steps"], f"{res_ppo['sum']['runtime_s']:.2f}"])
        curves["PPO"] = res_ppo["regrets"]
        print(rows)
        if xs is None: xs = np.arange(1, len(res_ppo["regrets"]) + 1)

    # Meta-RL (optional)
    if MetaRLPolicy is not None:
        env = make_env(); td.set_twtd_mode(True)
        adapter_probe = BanditEnvAdapter(env, actions)
        obs_vec = adapter_probe.reset()
        obs_dim = int(np.asarray(obs_vec).size)
        act_dim = len(actions)
        metalr = MetaRLPolicy(obs_dim, act_dim)
        if hasattr(metalr, "cfg"):
            if hasattr(metalr.cfg, "adapt_horizon"): metalr.cfg.adapt_horizon = 1
            if hasattr(metalr.cfg, "rollout_len"): metalr.cfg.rollout_len = 1
        res_meta = _run_rl_single(metalr, env, actions, num_iterations)
        rows.append(["Meta-RL (FOMAML)", np.mean(res_meta["regrets"]), res_meta["sum"]["targets_found"], res_meta["sum"]["collisions"], res_meta["sum"]["cum_reward"], res_meta["sum"]["time_steps"], f"{res_meta['sum']['runtime_s']:.2f}"])
        curves["Meta-RL"] = res_meta["regrets"]
        if xs is None: xs = np.arange(1, len(res_meta["regrets"]) + 1)

    # UCB
    env = make_env(); td.set_twtd_mode(True)
    ucb = UCBAlgorithm(len(actions))
    res_ucb = run_episode_loop(ucb, actions, env, num_iterations)
    rows.append(["UCB", np.mean(res_ucb["regrets"]), res_ucb["sum"]["targets_found"], res_ucb["sum"]["collisions"], res_ucb["sum"]["cum_reward"], res_ucb["sum"]["time_steps"], f"{res_ucb['sum']['runtime_s']:.2f}"])
    curves["UCB"] = res_ucb["regrets"]
    if xs is None: xs = np.arange(1, len(res_ucb["regrets"]) + 1)

    # EXP3
    env = make_env(); td.set_twtd_mode(True)
    exp3 = Exp3Algorithm(len(actions), gamma=gamma)
    res_e3 = run_episode_loop(exp3, actions, env, num_iterations)
    rows.append(["EXP3", np.mean(res_e3["regrets"]), res_e3["sum"]["targets_found"], res_e3["sum"]["collisions"], res_e3["sum"]["cum_reward"], res_e3["sum"]["time_steps"], f"{res_e3['sum']['runtime_s']:.2f}"])
    curves["EXP3"] = res_e3["regrets"]

    # EXP3-τ (optional)
    if Exp3TauAlgorithm is not None:
        env = make_env(); td.set_twtd_mode(True)
        e3t = Exp3TauAlgorithm(len(actions), gamma=gamma, tau=int(tau), rewardMin=-40, rewardMax=35, m_memory=1)
        res_e3t = run_episode_loop_collect_tau(e3t, actions, env, num_iterations)
        rows.append([f"EXP3-τ({tau})", np.mean(res_e3t["regrets"]), res_e3t["sum"]["targets_found"], res_e3t["sum"]["collisions"], res_e3t["sum"]["cum_reward"], res_e3t["sum"]["time_steps"], f"{res_e3t['sum']['runtime_s']:.2f}"])
        curves[f"EXP3-τ({tau})"] = res_e3t["regrets"]

    # Save table and figure
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_table(RESULTS_DIR / "Switching_Baselines.csv",
                ["Algorithm","MeanRegret","TargetsFound","Collisions","CumReward","TimeSteps","Runtime(s)"],
                rows)

    import matplotlib.pyplot as plt
    xs = np.arange(1, len(next(iter(curves.values()))) + 1)
    plt.figure(figsize=(8,6))
    for name, reg in curves.items():
        plt.plot(xs, reg, label=name)
    plt.xlabel("Iteration"); plt.ylabel("|Weak Regret|"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_Switching_AvgRegret.png"); plt.close()

    print("[OK] Saved:", RESULTS_DIR / "Switching_Baselines.csv", "and", FIG_DIR / "Fig_Switching_AvgRegret.png")

# ------------------------------
# Orchestrator
# ------------------------------
def main():
    # Combined RL vs bandits experiment
    try:
        run_experiment_RL_vs_Bandits(num_iterations=100, tau=10, gamma=0.15)
    except Exception as e:
        print("[WARN] run_experiment_RL_vs_Bandits failed:", e)

    # Optional graph module
    try:
        import graph
        graph.main()
    except Exception as e:
        print("[WARN] graph generation failed:", e)

if __name__ == "__main__":
    main()
