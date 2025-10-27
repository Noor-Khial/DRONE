#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main experiments runner + trajectory/policy-switch visualization.

Requires your existing modules:
- exp3.Exp3Algorithm
- ucb.UCBAlgorithm (imported but not used here)
- DDPG.rlPolicy.DDPGPolicy
- env.gym_env.GridWorldEnv
- utils.read_policy_at_t
- throughWallDetection.target_detection (for toggles)

Outputs:
- CSV summaries in results/
- Figures in FIGs/
- Trajectory logs per run (results/trajectory_log_*.csv)
- Trajectory figures with colored segments and switch markers (Fig_R6-4_*.png)
"""

import os
import csv
import time
import json
import random
import math
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Core algorithms / utils from your repo
from exp3 import Exp3Algorithm
from ucb import UCBAlgorithm  # not used below, but kept for completeness
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

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
RESULTS_DIR = Path("results")
FIG_DIR = Path("FIGs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def write_table(path: Path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def regret_bound(xs, K=3):
    xs = np.asarray(xs)
    return np.sqrt(xs * K * np.log(max(K, 2)))

def _xy_from_env(env):
    """Extract (x, y) for the UAV/agent from the env."""
    xy = getattr(env, "_agent_location", None)
    if xy is None:
        return float("nan"), float("nan")
    return float(xy[0]), float(xy[1])

# -----------------------------------------------------------------------------
# Figure: trajectory segmented by active policy (+ switch markers)
# -----------------------------------------------------------------------------
def plot_trajectory_with_policy_segments(traj_csv, out_png, title="UAV Trajectory with Policy Switching"):
    # load log
    ts, xs, ys, choices, envpols, switched = [], [], [], [], [], []
    with open(traj_csv, "r") as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            t, x, y, ch, ep, sw = int(row[0]), float(row[1]), float(row[2]), int(row[3]), int(row[4]), int(row[5])
            ts.append(t); xs.append(x); ys.append(y); choices.append(ch); envpols.append(ep); switched.append(sw)

    if len(xs) < 2:
        print("[WARN] Not enough points to draw a trajectory.")
        return

    # match your policy-distribution palette (env1 blue, env2 brown, env3 cyan; extendable)
    palette = {0: "tab:blue", 1: "sienna", 2: "cyan", 3: "gray", 4: "tab:green"}

    # legend counts for chosen policies
    from collections import Counter
    cnt = Counter(choices); total = len(choices)
    legend_lines = []
    for k in sorted(cnt):
        legend_lines.append((k, f"env{k+1} ({cnt[k]}, {100.0*cnt[k]/total:.1f}%)"))

    plt.figure(figsize=(8,7))
    # faint full path
    plt.plot(xs, ys, linewidth=1, alpha=0.25)

    # colored segments by chosen policy
    for i in range(len(xs)-1):
        c = palette.get(choices[i] % 5, "black")
        plt.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=c, linewidth=2.5)

    # mark switch points
    sx = [xs[i] for i in range(len(xs)) if switched[i]]
    sy = [ys[i] for i in range(len(ys)) if switched[i]]
    if sx:
        plt.scatter(sx, sy, s=45, facecolors="white", edgecolors="black", linewidths=1.2, zorder=5, label="policy switch")

    # compose legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=palette.get(k%5,"black"), lw=3, label=txt) for k, txt in legend_lines]
    if sx:
        handles.append(Line2D([0],[0], marker='o', color='black', markerfacecolor='white', lw=0, label="policy switch"))
    plt.legend(handles=handles, loc="upper center", ncol=min(3, len(handles)), bbox_to_anchor=(0.5, 1.05))

    plt.title(title, fontweight="bold")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.axis("equal"); plt.tight_layout()
    plt.savefig(out_png, dpi=220); plt.close()

# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def make_env():
    return GridWorldEnv(render_mode=None, size=55)

def make_actions():
    return [
        DDPGPolicy(model_path='experiments/exp-e/env1',  obstacles_file='env/targets_movements/shifted_obstacles/obstacles_1.0.json'),
        DDPGPolicy(model_path='experiments/exp-e/env08', obstacles_file='env/targets_movements/shifted_obstacles/obstacles_0.8.json'),
        DDPGPolicy(model_path='experiments/exp-e/env06', obstacles_file='env/targets_movements/shifted_obstacles/obstacles_0.6.json'),
    ]

# -----------------------------------------------------------------------------
# Episode loops (baseline + τ-collecting)
# -----------------------------------------------------------------------------
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

    # NEW: trajectory/policy logging
    traj_rows = []          # [t, x, y, chosen_policy, env_policy, switched?]
    switch_idxs = []
    prev_choice = None

    for iteration in range(num_iterations):
        if iteration == 50:
            bestAction = 2
            current_state = env.reset()

        current_policy = read_policy_at_t('exp3/sampled_policies(3policies3).txt', iteration)
        choice = algorithm.run()  # int index
        state, reward, t, metrics = actions[choice].run(current_state, current_policy, num_episodes=1)

        # bestAction reward for EXP3 update
        _, bestActionReward, _, _ = actions[bestAction].run(current_state, bestAction, num_episodes=1)

        if hasattr(algorithm, "update"):
            algorithm.update(iteration, reward, bestActionReward)

        wr, rb = algorithm.get_regret() if hasattr(algorithm, "get_regret") else (0.0, 0.0)
        regrets.append(abs(float(wr)))
        r_bounds.append(float(rb))
        rewards.append(float(reward))
        current_state = state

        # metrics aggregation
        td_i, tf_i, col_i, ts_i, cr_i = metrics
        sum_total_distance += float(td_i)
        sum_targets_found += float(tf_i)
        sum_collisions += float(col_i)
        sum_time_steps += float(ts_i)
        sum_cum_reward += float(cr_i)

        # --- log pose + policy + switches ---
        x, y = _xy_from_env(env)
        switched = int(prev_choice is not None and choice != prev_choice)
        if switched:
            switch_idxs.append(iteration)
        traj_rows.append([iteration, x, y, int(choice), int(current_policy), switched])
        prev_choice = choice

    # averages across iterations
    denom = max(1, num_iterations)
    total_distance = sum_total_distance / denom
    targets_found = sum_targets_found / denom
    collisions = sum_collisions / denom
    time_steps = sum_time_steps / denom
    cum_reward = sum_cum_reward / denom

    runtime_s = time.perf_counter() - wall0

    # save trajectory log for visualization
    traj_path = RESULTS_DIR / f"trajectory_log_{now_tag()}.csv"
    write_table(traj_path, ["t","x","y","choice","env_policy","switched"], traj_rows)

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
        "traj_csv": str(traj_path),
        "switch_idxs": switch_idxs,
    }

def run_episode_loop_collect_tau(algorithm, actions, env, num_iterations, rewardMin=-40, rewardMax=35):
    current_state = env.reset()
    bestAction = 0
    regrets, r_bounds, rewards = [], [], []
    tau_bounds = []
    wall0 = time.perf_counter()

    # accumulators
    sum_total_distance = 0.0
    sum_targets_found = 0.0
    sum_collisions = 0.0
    sum_time_steps = 0.0
    sum_cum_reward = 0.0

    # NEW: trajectory/policy logging
    traj_rows = []
    switch_idxs = []
    prev_choice = None

    for iteration in range(num_iterations):
        if iteration == 50:
            bestAction = 2
            current_state = env.reset()

        current_policy = read_policy_at_t('exp3/sampled_policies(3policies3).txt', iteration)
        choice = algorithm.run()
        state, reward, t, metrics = actions[choice].run(current_state, current_policy, num_episodes=1)

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

        # metrics aggregation
        td_i, tf_i, col_i, ts_i, cr_i = metrics
        sum_total_distance += float(td_i)
        sum_targets_found += float(tf_i)
        sum_collisions += float(col_i)
        sum_time_steps += float(ts_i)
        sum_cum_reward += float(cr_i)

        # --- log pose + policy + switches ---
        x, y = _xy_from_env(env)
        switched = int(prev_choice is not None and choice != prev_choice)
        if switched:
            switch_idxs.append(iteration)
        traj_rows.append([iteration, x, y, int(choice), int(current_policy), switched])
        prev_choice = choice

    # averages
    denom = max(1, num_iterations)
    total_distance = sum_total_distance / denom
    targets_found = sum_targets_found / denom
    collisions = sum_collisions / denom
    time_steps = sum_time_steps / denom
    cum_reward = sum_cum_reward / denom

    runtime_s = time.perf_counter() - wall0

    # save trajectory log
    traj_path = RESULTS_DIR / f"trajectory_log_{now_tag()}.csv"
    write_table(traj_path, ["t","x","y","choice","env_policy","switched"], traj_rows)

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
        "traj_csv": str(traj_path),
        "switch_idxs": switch_idxs,
    }

# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
def run_experiment_E1(num_iterations=100):
    """TWTD vs LOS."""
    actions = make_actions()
    env = make_env()

    td.set_twtd_mode(True); td.set_twtd_gain(1.0); td.set_adversarial_level(0.0)
    exp3 = Exp3Algorithm(len(actions), gamma=0.15)
    res_twtd = run_episode_loop(exp3, actions, env, num_iterations)

    # produce trajectory figure (TWTD)
    plot_trajectory_with_policy_segments(
        res_twtd["traj_csv"], FIG_DIR / "Fig_R6-4_Traj_TWTD.png",
        title="UAV Trajectory (TWTD) with Policy Switching"
    )

    env = make_env()
    td.set_twtd_mode(False); td.set_twtd_gain(1.0)
    exp3_l = Exp3Algorithm(len(actions), gamma=0.15)
    res_los = run_episode_loop(exp3_l, actions, env, num_iterations)

    # LOS trajectory figure
    plot_trajectory_with_policy_segments(
        res_los["traj_csv"], FIG_DIR / "Fig_R6-4_Traj_LOS.png",
        title="UAV Trajectory (LOS) with Policy Switching"
    )

    header = ["Mode","MeanRegret","TargetsFound","Collisions","CumReward","TimeSteps","Runtime(s)"]
    rows = [
        ["TWTD", np.mean(res_twtd["regrets"]), res_twtd["sum"]["targets_found"], res_twtd["sum"]["collisions"], res_twtd["sum"]["cum_reward"], res_twtd["sum"]["time_steps"], f"{res_twtd['sum']['runtime_s']:.2f}"],
        ["LOS",  np.mean(res_los["regrets"]),  res_los["sum"]["targets_found"],  res_los["sum"]["collisions"],  res_los["sum"]["cum_reward"],  res_los["sum"]["time_steps"],  f"{res_los['sum']['runtime_s']:.2f}"],
    ]
    write_table(RESULTS_DIR / "E1_TWTDvsLOS.csv", header, rows)

    xs = np.arange(1, len(res_twtd["regrets"]) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(xs, res_twtd["regrets"], label="TWTD")
    plt.plot(xs, res_los["regrets"], label="LOS")
    plt.plot(xs, regret_bound(xs), "--", label="√(t K log K)")
    plt.xlabel("Iteration"); plt.ylabel("|Weak Regret|"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_R8-53_TWTDvsLOS.png"); plt.close()

def run_experiment_E2(num_iterations=100):
    """Processing Gain Sweep."""
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

        # trajectory figure per gain
        plot_trajectory_with_policy_segments(
            res["traj_csv"], FIG_DIR / f"Fig_R6-4_Traj_Gproc{gp:.1f}.png",
            title=f"UAV Trajectory (G_proc={gp:.1f}) with Policy Switching"
        )

        if xs_plot is None: xs_plot = np.arange(1, len(res["regrets"])+1)
        to_plot.append((gp, res["regrets"]))

    write_table(RESULTS_DIR / "E2_Gproc.csv", header, rows)

    plt.figure(figsize=(8,6))
    for gp, reg in to_plot:
        plt.plot(xs_plot, reg, label=f"G_proc={gp}")
    plt.plot(xs_plot, regret_bound(xs_plot), "--", label="√(t K log K)")
    plt.xlabel("Iteration"); plt.ylabel("|Weak Regret|"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_R8-54_Gproc.png"); plt.close()

def run_experiment_E3(num_iterations=100):
    """EXP3 gamma selection."""
    actions = make_actions()
    header = ["Gamma","MeanRegret","StdRegret","TargetsFound","Collisions","CumReward"]
    rows = []

    for g in [0.05, 0.10, 0.15, 0.20]:
        env = make_env()
        td.set_twtd_mode(True)
        exp3 = Exp3Algorithm(len(actions), gamma=g)
        res = run_episode_loop(exp3, actions, env, num_iterations)
        rows.append([g, np.mean(res["regrets"]), np.std(res["regrets"]), res["sum"]["targets_found"], res["sum"]["collisions"], res["sum"]["cum_reward"]])

        # trajectory per gamma
        plot_trajectory_with_policy_segments(
            res["traj_csv"], FIG_DIR / f"Fig_R6-4_Traj_gamma{g:.2f}.png",
            title=f"UAV Trajectory (gamma={g:.2f}) with Policy Switching"
        )

    write_table(RESULTS_DIR / "E3_Exp3Params.csv", header, rows)

def run_experiment_E5(num_iterations=100):
    """EXP3 under adversarial levels."""
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

        # trajectory per adversarial level
        plot_trajectory_with_policy_segments(
            res["traj_csv"],
            FIG_DIR / f"Fig_R6-4_Traj_adv{lvl:.1f}.png",
            title=f"UAV Trajectory (adv={lvl:.1f}) with Policy Switching"
        )

    write_table(RESULTS_DIR / "E5_AdversarialLevels.csv", header, rows)

    plt.figure(figsize=(8,6))
    for lvl, reg in curves:
        plt.plot(xs, reg, label=f"adv={lvl}")
    plt.plot(xs, regret_bound(xs), "--", label="Upper Bound")
    plt.xlabel("Time Step (t)"); plt.ylabel("|Cummulative Regret|"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig_R8-36_AdversarialLevels.png"); plt.close()

def run_experiment_E9(num_iterations=100, taus=(10,20,30), gamma=0.15, m_memory=1):
    """EXP3 vs EXP3-τ (mini-batched)."""
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

    # Trajectory for EXP3 baseline
    plot_trajectory_with_policy_segments(
        res_exp3["traj_csv"], FIG_DIR / "Fig_R6-4_Traj_EXP3.png",
        title="UAV Trajectory (EXP3) with Policy Switching"
    )

    # EXP3-τ variants (ensure τ >= 2 given m=1)
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

        # Trajectory per τ
        plot_trajectory_with_policy_segments(
            res_tau["traj_csv"], FIG_DIR / f"Fig_R6-4_Traj_EXP3tau_tau{tau}.png",
            title=f"UAV Trajectory (EXP3-τ, τ={tau}) with Policy Switching"
        )

    write_table(RESULTS_DIR / "E9_Exp3_vs_Exp3Tau.csv", header, rows)

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

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def main():
    """
    Toggle whichever experiments you want to run.
    Each will also emit a trajectory figure (Fig_R6-4_*.png)
    that directly addresses Reviewer R6-4.
    """
    # run_experiment_E1()
    # run_experiment_E2()
    # run_experiment_E3()
    run_experiment_E5()
    # run_experiment_E9(num_iterations=100, taus=(3,5,10), gamma=0.15, m_memory=1)

    # Optional: try your graph module if present
    try:
        import graph
        graph.main()
    except Exception as e:
        print("[WARN] graph generation failed:", e)

if __name__ == "__main__":
    main()
