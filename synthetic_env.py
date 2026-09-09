"""
synthetic_env.py -- shared synthetic bandit environment used to benchmark
all algorithms on an identical, apples-to-apples basis.

This is the single source of truth for "what reward does arm X get at
round t" -- no algorithm should generate its own rewards internally
anymore. Every algorithm (fast_adswitch.py, ts_ge.py) takes a `reward_fn`
callback with this exact signature:

    reward_fn(arm_or_arms, t) -> float
        arm_or_arms : int (single arm index), or a list[int] (a "group
                      probe" -- e.g. TS-GE's Broadcast Probing / Group
                      Exploration phases -- returns the mean reward
                      across the group, per Assumption 2 of the TS-GE
                      paper)
        t           : 1-indexed round number
"""
import numpy as np


class SyntheticEnv:
    def __init__(self, K, means, sigma=1.0, change_schedule=None, seed=None):
        """
        K               : number of arms
        means           : list of K initial true arm means
        sigma           : reward standard deviation (Gaussian noise)
        change_schedule : list of (time, arm_index, new_mean) tuples,
                          applied the first time `t` reaches `time`
        seed            : RNG seed, for reproducible comparisons across
                          different algorithms run on the "same" environment
        """
        self.K = K
        self.means = np.array(means, dtype=float)
        self.sigma = sigma
        self.schedule = sorted(change_schedule or [], key=lambda x: x[0])
        self.rng = np.random.default_rng(seed)
        # records the best true mean at every round t this env was queried
        # at -- used by the benchmark harness to compute regret without any
        # algorithm needing (or being allowed) to see the true means itself
        self.best_mean_history = {}

    def _apply(self, t):
        while self.schedule and self.schedule[0][0] <= t:
            _, idx, new_mean = self.schedule.pop(0)
            self.means[idx] = new_mean

    def reward_fn(self, arm_or_arms, t):
        self._apply(t)
        self.best_mean_history[t] = float(self.means.max())
        if isinstance(arm_or_arms, (list, tuple, np.ndarray)):
            idx = list(arm_or_arms)
            vals = self.rng.normal(self.means[idx], self.sigma)
            return float(np.mean(vals))
        return float(self.rng.normal(self.means[arm_or_arms], self.sigma))


def regret_from_history(history, best_mean_history, indexing_start=1):
    """Compute per-round regret for any algorithm's returned history dict
    (must have a 'reward' list), against the environment's
    best_mean_history recorded during that same run."""
    rewards = history["reward"]
    regret = []
    for i, r in enumerate(rewards):
        t = indexing_start + i
        best = best_mean_history.get(t)
        regret.append((best - r) if best is not None else 0.0)
    return regret
