"""
test_fast_adswitch.py
----------------------
Test cases + speed benchmark for fast_adswitch.FastAdSwitch.

Run with:  python test_fast_adswitch.py
           python test_fast_adswitch.py --compare-original   (slow; Test 3)

Prints numeric results and saves plots under artifacts/:
  - regret_vs_horizon.png   (Test 1: cumulative regret shrinking with T)
  - switching_regret.png    (Test 2: reward tracking through 2 change points)
  - speed_comparison.png    (Test 3, only with --compare-original)

The reward environment is `synthetic_env.SyntheticEnv`, the same bounded
scaled-Beta environment the main benchmark uses, rather than the private
Gaussian generators this file used to carry. Two reasons: the numbers here are
then directly comparable with test_algorithms.py, and the old generators drew
from an unbounded Gaussian, which no bounded-reward algorithm in this repo is
entitled to assume. Regret comes from `synthetic_env.regret_from_history`, which
also fixes an off-by-one in the old Test 2 (it compared T oracle means against
T-1 observed rewards).
"""
import argparse
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fast_adswitch import FastAdSwitch
from synthetic_env import SyntheticEnv, regret_from_history

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Bounded-reward scale for these tests. sigma must stay below
# R_max * sqrt(p(1-p)) for every arm mean; at means 0.2..0.8 on R_max=1 the
# tightest bound is 0.4, so 0.25 is comfortably valid.
R_MAX = 1.0
SIGMA = 0.25


def make_env(means, T, seed, change_schedule=None):
    """One SyntheticEnv configured for these tests."""
    return SyntheticEnv(
        K=len(means), means=list(means), sigma=SIGMA, R_max=R_MAX,
        change_schedule=list(change_schedule or []), seed=seed,
    )


# ---------------------------------------------------------------------
# Test 1: stationary bandit -> regret should grow sublinearly with T
# ---------------------------------------------------------------------
def test_stationary_regret():
    print("\n=== Test 1: stationary bandit, regret vs horizon ===")
    means = [0.2, 0.5, 0.8]
    Ts = [200, 500, 1000, 2000, 5000]
    regrets, regret_rates = [], []
    for T in Ts:
        env = make_env(means, T, seed=1)
        t0 = time.time()
        res = FastAdSwitch(3, T, C1=1.0, seed=1).run(env.reward_fn)
        dt = time.time() - t0
        cum_regret = float(np.sum(regret_from_history(res, env.best_mean_history)))
        regrets.append(cum_regret)
        regret_rates.append(cum_regret / T)
        print(f"  T={T:5d}  time={dt:7.4f}s  net_reward={res['net_reward']:9.2f}"
              f"  cum_regret={cum_regret:8.2f}  regret/T={cum_regret/T:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(Ts, regrets, marker="o")
    ax[0].set_xlabel("Horizon T")
    ax[0].set_ylabel("Cumulative regret")
    ax[0].set_title("Cumulative regret vs T")
    ax[1].plot(Ts, regret_rates, marker="o", color="darkorange")
    ax[1].set_xlabel("Horizon T")
    ax[1].set_ylabel("Regret / T")
    ax[1].set_title("Average per-round regret vs T\n(should trend toward 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "regret_vs_horizon.png"), dpi=150)
    plt.close(fig)
    print("  saved plot -> regret_vs_horizon.png")
    

# ---------------------------------------------------------------------
# Test 2: switching bandit -> algorithm should track the moving best arm
# ---------------------------------------------------------------------
def test_switching_environment():
    print("\n=== Test 2: piecewise-stationary bandit (2 change points) ===")
    T = 1500
    change_points = [500, 1000]
    # SyntheticEnv changes one arm at a time, so each transition of the old
    # three-segment table is expressed as the individual arm moves it implied.
    change_schedule = [
        (500, 0, 0.2), (500, 1, 0.8),
        (1000, 1, 0.2), (1000, 2, 0.8), (1000, 0, 0.5),
    ]
    env = make_env([0.8, 0.2, 0.5], T, seed=2, change_schedule=change_schedule)
    t0 = time.time()
    res = FastAdSwitch(3, T, C1=1.0, seed=2).run(env.reward_fn)
    dt = time.time() - t0
    per_slot_regret = np.asarray(regret_from_history(res, env.best_mean_history))
    best_per_t = np.asarray([env.best_mean_history[t] for t in range(1, T + 1)])
    cum_regret = float(per_slot_regret.sum())
    print(f"  time={dt:.3f}s  net_reward={res['net_reward']:.2f}"
          f"  optimal={best_per_t.sum():.2f}  cum_regret={cum_regret:.2f}")

    # rolling average reward vs the (moving) optimal reward, to visualise tracking
    window = 50
    reward_hist = np.asarray(res["reward"])
    def rolling(x, w):
        c = np.cumsum(np.insert(x, 0, 0))
        return (c[w:] - c[:-w]) / w
    roll_reward = rolling(reward_hist, window)
    roll_opt = rolling(best_per_t, window)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(roll_opt, label="best possible reward (rolling avg)", color="green")
    ax.plot(roll_reward, label="AdSwitch reward (rolling avg)", color="blue", alpha=0.8)
    for cp in change_points:
        ax.axvline(cp, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Reward (rolling mean, window={window})")
    ax.set_title("AdSwitch tracking a piecewise-stationary bandit\n(red dashed lines = true change points)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "switching_regret.png"), dpi=150)
    plt.close(fig)
    print("  saved plot -> switching_regret.png")


# ---------------------------------------------------------------------
# Test 3: speed comparison vs the original recursive implementation
# ---------------------------------------------------------------------
def run_original(K, Time_Horizon, means, std=1.0, seed=0, C1=1.0):
    """Faithful re-creation of the control flow in the repo's ADSWITCH.py,
    used here only as a slow reference to benchmark against."""
    import sys
    import random
    rng = np.random.default_rng(seed)

    class ARMS:
        def __init__(self, mean, std):
            self.arm_mean = mean
            self.time_when_last_chosen = 0
            self.number_of_times_chosen = [0] * (Time_Horizon + 1)
            self.reward_List = rng.normal(mean, std, Time_Horizon + 1)
            self.eviction_observed_mean_reward = None
            self.eviction_gap = None

        def tell_reward(self, t):
            return self.reward_List[t]

        def update_chosen(self, t):
            self.number_of_times_chosen[t] += 1
            self.time_when_last_chosen = t

        def chosen_in_interval(self, s, e):
            return self.number_of_times_chosen[s:e + 1].count(1)

        def mean_rewards_observed(self, s, e):
            num = denom = 0
            flag = False
            for i in range(s, e + 1):
                if self.number_of_times_chosen[i] == 1:
                    flag = True
                    num += self.reward_List[i]
                    denom += 1
            return num / denom if flag else 0

    Arm_List = [ARMS(m, std) for m in means]

    def check_bad(bad_cur, samp_cur, ep, t):
        for arm in bad_cur:
            i, change = 1, 0.5
            samp_cur[arm] = []
            delta = arm.eviction_gap / 16
            while change >= delta:
                p = change * math.sqrt(ep / (K * Time_Horizon * math.log(Time_Horizon, 10)))
                if random.random() < p:
                    samp_cur[arm].append([change, math.ceil((2 ** (2 * i + 1)) * math.log(Time_Horizon, 10)), t])
                i += 1
                change = 2 ** (-i)

    def check_changes_good(start, t, good):
        for arm in good:
            for s in range(start, t + 1):
                for s1 in range(start, t + 1):
                    for s2 in range(s1, t + 1):
                        if arm.chosen_in_interval(s, t) == 0 or arm.chosen_in_interval(s1, s2) == 0:
                            continue
                        x = abs(arm.mean_rewards_observed(s, s1) - arm.mean_rewards_observed(s1, s2))
                        y = (math.sqrt(2 * math.log(Time_Horizon, 10) / arm.chosen_in_interval(s1, s2))
                             + math.sqrt(2 * math.log(Time_Horizon, 10) / arm.chosen_in_interval(s, t)))
                        if x > y:
                            return True
        return False

    def check_changes_bad(start, t, bad):
        for arm in bad:
            for s in range(start, t + 1):
                if arm.chosen_in_interval(s, t) == 0:
                    continue
                x = abs(arm.mean_rewards_observed(s, t) - arm.eviction_observed_mean_reward)
                y = arm.eviction_gap / 4 + math.sqrt(2 * math.log(Time_Horizon, 10) / arm.chosen_in_interval(s, t))
                if x > y:
                    return True
        return False

    sys.setrecursionlimit(Time_Horizon + 10000)
    state = dict(net_reward=0.0)
    Current_Episode, Time_Step = 0, 0

    def start_new_episode():
        nonlocal Current_Episode, Time_Step
        if Time_Step > Time_Horizon:
            return
        Current_Episode += 1
        Start = Time_Step + 1
        good, bad = [], []
        for k in Arm_List:
            k.eviction_gap = None
            k.eviction_observed_mean_reward = None
            good.append(k)

        def next_step(good, bad, samp_cur):
            nonlocal Time_Step
            Time_Step += 1
            if Time_Step > Time_Horizon:
                return
            bad_next, samp_next = list(bad), {k: [] for k in bad}
            check_bad(bad, samp_cur, Current_Episode, Time_Step)
            Time, chosen = 10 ** 18, None
            for arm in bad + good:
                if arm in bad and samp_cur.get(arm, []) == []:
                    continue
                if arm.time_when_last_chosen < Time:
                    Time, chosen = arm.time_when_last_chosen, arm
            r = chosen.tell_reward(Time_Step)
            chosen.update_chosen(Time_Step)
            state["net_reward"] += r
            if check_changes_good(Start, Time_Step, good):
                return start_new_episode()
            if check_changes_bad(Start, Time_Step, bad):
                return start_new_episode()
            for i, klist in samp_cur.items():
                for k in klist:
                    if i.chosen_in_interval(k[2], Time_Step) < k[1]:
                        bad_next.append(i)
                        samp_next.setdefault(i, []).append(k.copy())
            for _ in range(K):
                best, pair, store = -10 ** 18, [], []
                for a_p in good:
                    for a in good:
                        for s in range(Start, Time_Step + 1):
                            if a.chosen_in_interval(s, Time_Step) < 2:
                                continue
                            left = a_p.mean_rewards_observed(s, Time_Step) - a.mean_rewards_observed(s, Time_Step)
                            right = math.sqrt(C1 * math.log(Time_Horizon, 10) / (a.chosen_in_interval(s, Time_Step) - 1))
                            if left > right and left - right > best:
                                best, pair, store = left - right, [a_p, a], [left, s]
                if pair:
                    a = pair[1]
                    good.remove(a)
                    bad_next.append(a)
                    a.eviction_observed_mean_reward = a.mean_rewards_observed(store[1], Time_Step)
                    a.eviction_gap = store[0]
                    samp_next[a] = []
            good_next = list(set(Arm_List) - set(bad_next))
            return next_step(good_next, bad_next, samp_next)

        return next_step(good, bad, {})

    start_new_episode()
    return state["net_reward"]


def test_speed_comparison():
    print("\n=== Test 3: speed, original recursive vs fast vectorised ===")
    print(f"  {'T':>6} | {'original (s)':>13} | {'fast (s)':>9} | speedup")
    Ts, t_origs, t_fasts = [], [], []
    for T in [50, 100, 150, 200, 300]:
        t0 = time.time()
        run_original(3, T, [0.2, 0.5, 0.8], seed=1, C1=1.0)
        t_orig = time.time() - t0

        env = make_env([0.2, 0.5, 0.8], T, seed=1)
        t0 = time.time()
        FastAdSwitch(3, T, C1=1.0, seed=1).run(env.reward_fn)
        t_fast = time.time() - t0

        Ts.append(T); t_origs.append(t_orig); t_fasts.append(t_fast)
        print(f"  {T:6d} | {t_orig:13.4f} | {t_fast:9.4f} | {t_orig / t_fast:6.1f}x")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ts, t_origs, marker="o", label="original (recursive)")
    ax.plot(Ts, t_fasts, marker="o", label="fast (vectorised)")
    ax.set_yscale("log")
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.set_title("Runtime: original vs fast AdSwitch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "speed_comparison.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--compare-original", action="store_true",
        help="also run Test 3, the speed comparison against the original recursive "
             "AdSwitch kept in this file as a reference oracle. It is quadratic in T "
             "and only runs to T=300, so it is off by default.",
    )
    options = parser.parse_args()

    test_stationary_regret()
    test_switching_environment()
    if options.compare_original:
        test_speed_comparison()
    else:
        print("\n=== Test 3 (speed vs original) skipped; pass --compare-original to run it ===")
