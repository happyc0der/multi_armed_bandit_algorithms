"""
test_fast_adswitch.py
----------------------
Test cases + speed benchmark for fast_adswitch.FastAdSwitch, compared
against the original recursive ADSWITCH.py logic from the repo.

Run with:  python test_fast_adswitch.py
This will print the numeric results AND save two plots next to this file:
  - regret_vs_horizon.png   (Test 1: cumulative regret shrinking with T)
  - switching_regret.png    (Test 2: reward tracking through 2 change points)
"""
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from fast_adswitch import FastAdSwitch


# ---------------------------------------------------------------------
# Reward environments
# ---------------------------------------------------------------------
def make_stationary_reward_fn(means, std, T, seed):
    rng = np.random.default_rng(seed)
    rewards = np.array([rng.normal(m, std, T + 2) for m in means])
    return (lambda a, t: rewards[a, t]), np.asarray(means)


def make_switching_reward_fn(K, T, std, seed, change_points, arm_means_segments):
    rng = np.random.default_rng(seed)
    segments = [0] + list(change_points) + [T + 2]
    true_means = np.zeros((K, T + 2))
    for i in range(len(segments) - 1):
        true_means[:, segments[i]:segments[i + 1]] = np.array(arm_means_segments[i])[:, None]
    rewards = rng.normal(true_means, std)
    return (lambda a, t: rewards[a, t]), true_means


# ---------------------------------------------------------------------
# Test 1: stationary bandit -> regret should grow sublinearly with T
# ---------------------------------------------------------------------
def test_stationary_regret():
    print("\n=== Test 1: stationary bandit, regret vs horizon ===")
    means = [0.2, 0.5, 0.8]
    Ts = [200, 500, 1000, 2000, 5000]
    regrets, regret_rates = [], []
    for T in Ts:
        reward_fn, _ = make_stationary_reward_fn(means, 1.0, T, seed=1)
        t0 = time.time()
        res = FastAdSwitch(3, T, C1=1.0, seed=1).run(reward_fn)
        dt = time.time() - t0
        cum_regret = max(means) * T - res["net_reward"]
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
    fig.savefig("regret_vs_horizon.png", dpi=150)
    plt.close(fig)
    print("  saved plot -> regret_vs_horizon.png")


# ---------------------------------------------------------------------
# Test 2: switching bandit -> algorithm should track the moving best arm
# ---------------------------------------------------------------------
def test_switching_environment():
    print("\n=== Test 2: piecewise-stationary bandit (2 change points) ===")
    T = 1500
    change_points = [500, 1000]
    segments = [[0.8, 0.2, 0.5], [0.2, 0.8, 0.5], [0.5, 0.2, 0.8]]
    reward_fn, true_means = make_switching_reward_fn(
        3, T, 1.0, seed=2, change_points=change_points, arm_means_segments=segments
    )
    t0 = time.time()
    res = FastAdSwitch(3, T, C1=1.0, seed=2).run(reward_fn)
    dt = time.time() - t0
    best_per_t = true_means.max(axis=0)[1:T + 1]
    cum_regret = best_per_t.sum() - res["net_reward"]
    print(f"  time={dt:.3f}s  net_reward={res['net_reward']:.2f}"
          f"  optimal={best_per_t.sum():.2f}  cum_regret={cum_regret:.2f}")

    # rolling average reward vs the (moving) optimal reward, to visualise tracking
    window = 50
    reward_hist = res["regret_hist"][1:T + 1]
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
    fig.savefig("switching_regret.png", dpi=150)
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

        reward_fn, _ = make_stationary_reward_fn([0.2, 0.5, 0.8], 1.0, T, seed=1)
        t0 = time.time()
        FastAdSwitch(3, T, C1=1.0, seed=1).run(reward_fn)
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
    fig.savefig("speed_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved plot -> speed_comparison.png")
    print("  (the original does not finish in practical time once T reaches the "
          "thousands; the fast version handles T=5000+ in well under a second)")


if __name__ == "__main__":
    test_stationary_regret()
    test_switching_environment()
    test_speed_comparison()
