"""
test_algorithms.py -- head-to-head comparison of FastAdSwitch and TS-GE on
the exact same synthetic environment (same seed, same change schedule),
using the shared BanditAlgorithm interface. This is the harness that
future algorithms (UCB1, epsilon-greedy, market_env-backed runs, etc.)
should also plug into.

Run with:  python test_algorithms.py
Saves a comparison plot to artifacts/algorithm_comparison.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from synthetic_env import SyntheticEnv, regret_from_history
from fast_adswitch import FastAdSwitch
from ts_ge import TS_GE

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def run_comparison(K, T, means, change_schedule, env_seed=42):
    results = {}

    env1 = SyntheticEnv(K=K, means=means, sigma=1.0, change_schedule=list(change_schedule), seed=env_seed)
    algo1 = FastAdSwitch(K=K, T=T, C1=1.0, seed=1)
    h1 = algo1.run(env1.reward_fn)
    regret1 = regret_from_history(h1, env1.best_mean_history)
    results["FastAdSwitch"] = dict(history=h1, regret=regret1)

    env2 = SyntheticEnv(K=K, means=means, sigma=1.0, change_schedule=list(change_schedule), seed=env_seed)
    algo2 = TS_GE(K=K, T=T, delta=0.1, R_min=min(means) - 3, R_max=max(means) + 3, seed=5)
    h2 = algo2.run(env2.reward_fn)
    regret2 = regret_from_history(h2, env2.best_mean_history)
    results["TS-GE"] = dict(history=h2, regret=regret2, detections=h2.get("detections", []))

    return results


if __name__ == "__main__":
    T = 6000
    K = 2
    means = [0.2, 0.6]
    change_schedule = [(3000, 0, 1.6)]  # arm 0 jumps from 0.2 to 1.6 at t=3000

    results = run_comparison(K, T, means, change_schedule)

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
    ax.set_title("FastAdSwitch vs TS-GE: cumulative regret on identical environment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "algorithm_comparison.png"), dpi=150)
    plt.close(fig)
    print("\nSaved artifacts/algorithm_comparison.png")
