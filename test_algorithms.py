"""
test_algorithms.py -- head-to-head comparison of FastAdSwitch, TS-GE, UCB1,
and EpsilonGreedy on the exact same synthetic environment (same seed, same
change schedule), using the shared BanditAlgorithm interface.

Run with:  python test_algorithms.py
Saves a comparison plot to artifacts/algorithm_comparison.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from synthetic_env import SyntheticEnv, regret_from_history
from fast_adswitch import FastAdSwitch
from ts_ge import TS_GE
from ucb1 import UCB1
from epsilon_greedy import EpsilonGreedy
from mucb import MUCB

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
R_max = 20.0

def run_comparison(K, T, means,sigma,delta,change_schedule, env_seed=42):
    results = {}

    def add(name, algo_factory):
        env = SyntheticEnv(K=K, means=means, sigma=sigma,
                            change_schedule=list(change_schedule), seed=env_seed)
        algo = algo_factory()
        h = algo.run(env.reward_fn)
        regret = regret_from_history(h, env.best_mean_history)
        results[name] = dict(history=h, regret=regret, detections=h.get("detections", []))

    add("FastAdSwitch", lambda: FastAdSwitch(K=K, T=T, C1=1.0, seed=1))
    add("TS-GE", lambda: TS_GE(K=K, T=T, delta=delta,R_max=R_max, seed=1))
    add("UCB1", lambda: UCB1(K=K, T=T,sigma=sigma, seed=1))
    add("EpsilonGreedy", lambda: EpsilonGreedy(K=K, T=T, epsilon=0.1, decay=False, seed=1))
    add("M-UCB", lambda: MUCB(K=K, T=T, delta=2.0, M_estimate=2, seed=1))

    return results


if __name__ == "__main__":
    T = 6000
    K = 2
    means = [2.0, 6.0]       # non-negative, as the paper assumes
    sigma = 0.5     
    delta = 0.1# only for TS-GE
    R_max = 20.0   # only for TS-GE          # tight bound covering the post-change max (16) plus margin
    schedule = [(3000, 0, 16.0)]   # Delta_C=14, respects Assumption 3 (Delta_C >= 2*sigma)

    results = run_comparison(K, T, means,sigma,delta, schedule)

    print(f"{'Algorithm':<15} | {'Total Reward':>12} | {'Cum Regret':>10} | {'Rounds Played':>13}")
    for name, r in results.items():
        h = r["history"]
        print(f"{name:<15} | {h['net_reward']:12.2f} | {sum(r['regret']):10.2f} | {len(h['reward']):13d}")
    print("\nTS-GE change detections at t =", results["TS-GE"]["detections"], "(true change at t=3000)")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, r in results.items():
        cum_regret = np.cumsum(r["regret"])
        ax.plot(cum_regret, label=name)
    ax.axvline(3000, color="red", linestyle="--", alpha=0.6, label="true change point")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Algorithm comparison: cumulative regret on identical environment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "algorithm_comparison.png"), dpi=150)
    plt.close(fig)
    print("\nSaved artifacts/algorithm_comparison.png")
