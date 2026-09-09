"""
synthetic_env.py -- shared bounded synthetic environment for every bandit
algorithm in this repository.

WHY BOUNDED BETA REWARDS INSTEAD OF GAUSSIAN REWARDS?
=======================================================
The previous environment used an unbounded Gaussian distribution. That was
convenient, but it violated a central assumption of TS-GE: every reward
must satisfy 0 <= R_a(t) <= R_max, because TS-GE maps its observed reward
to a Bernoulli probability R_a(t) / R_max. The previous TS-GE code hid the
violation by clipping probabilities into [0, 1], which silently changed the
reward model and made paper-level validation impossible.

This environment uses a scaled Beta distribution instead. It gives each
arm an exact requested mean and shared requested standard deviation while
being strictly bounded in [0, R_max]:

    X_a(t) ~ R_max * Beta(alpha_a, beta_a)
    E[X_a(t)] = mean_a
    Std[X_a(t)] = sigma

This makes it valid for TS-GE's paper assumptions while remaining usable
for FastAdSwitch, M-UCB, UCB1, and epsilon-greedy.

ENVIRONMENT INTERFACE
=====================
    reward_fn(arm_index, t) -> float
        Return one bounded sample for one physical arm.

    reward_fn([arm_a, arm_b, ...], t) -> float
        Return a simultaneous group probe: independently sample every arm
        in the supplied group at the SAME environment time t, then return
        their arithmetic average. This directly implements TS-GE Assumption
        2:
            R_S(t) = (1 / |S|) * sum_{a in S} R_a(t).
        The caller sees ONLY this average, not the individual samples.

The time t is 1-indexed. A change scheduled as (3000, 0, 16.0) is applied
on the first query at t=3000, so the sample at t=3000 already uses arm 0's
new mean. Every algorithm gets a fresh environment object with the same
seed and schedule copy in the benchmark harness, ensuring fair comparison.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class SyntheticEnv:
    """Piecewise-stationary, bounded, multi-play synthetic bandit.

    All real arms use scaled Beta rewards. A Beta distribution is the right
    choice here because it is naturally bounded and its two parameters can
    be solved from a desired mean and standard deviation.
    """

    def __init__(
        self,
        K: int,
        means: Sequence[float],
        sigma: float,
        R_max: float,
        change_schedule: Sequence[tuple[int, int, float]] | None = None,
        seed: int | None = None,
    ) -> None:
        if not isinstance(K, int) or K <= 0:
            raise ValueError("K must be a positive integer.")
        if len(means) != K:
            raise ValueError(f"means must have length K={K}; got {len(means)}.")
        if not math.isfinite(R_max) or R_max <= 0:
            raise ValueError("R_max must be finite and > 0.")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and > 0.")

        self.K = K
        self.R_max = float(R_max)
        self.sigma = float(sigma)
        self.means = np.asarray(means, dtype=float)
        self._validate_means(self.means)

        self.schedule = sorted(list(change_schedule or []), key=lambda item: item[0])
        self._validate_schedule()
        self.rng = np.random.default_rng(seed)

        # Filled lazily as reward_fn is queried. Benchmark code uses this
        # oracle-only record to compute regret; algorithms never receive it.
        self.best_mean_history: dict[int, float] = {}

    def _validate_means(self, means: np.ndarray) -> None:
        if not np.all(np.isfinite(means)):
            raise ValueError("All arm means must be finite.")
        # Strictly interior means give a nonzero Beta variance. Means exactly
        # 0 or R_max need a degenerate point-mass model, intentionally not
        # hidden here.
        if np.any(means <= 0.0) or np.any(means >= self.R_max):
            raise ValueError(
                f"Every mean must lie strictly inside (0, R_max) = (0, {self.R_max})."
            )

        for mean in means:
            p = mean / self.R_max
            # A variable supported on [0, R_max] with mean mean has maximum
            # possible SD R_max * sqrt(p*(1-p)). A Beta distribution needs
            # strictly less than that maximum for positive concentration.
            max_possible_sigma = self.R_max * math.sqrt(p * (1.0 - p))
            if self.sigma >= max_possible_sigma:
                raise ValueError(
                    f"sigma={self.sigma} is too large for mean={mean}. "
                    f"For a bounded [0, R_max] reward, sigma must be < {max_possible_sigma:.6g}."
                )

    def _validate_schedule(self) -> None:
        previous_t = 0
        for item in self.schedule:
            if len(item) != 3:
                raise ValueError("Each change must be a (time, arm_index, new_mean) tuple.")
            t, arm, new_mean = item
            if not isinstance(t, int) or t <= 0:
                raise ValueError("Every change time must be a positive integer.")
            if t < previous_t:
                raise ValueError("change_schedule must be sorted by nondecreasing time.")
            if not isinstance(arm, int) or not 0 <= arm < self.K:
                raise ValueError(f"Change arm index {arm} is outside [0, {self.K - 1}].")
            self._validate_means(np.asarray([new_mean], dtype=float))
            previous_t = t

    def _apply_due_changes(self, t: int) -> None:
        # <= makes the environment robust if a caller advances directly from
        # t=10 to t=15; every scheduled change through t=15 is still applied.
        while self.schedule and self.schedule[0][0] <= t:
            _, arm, new_mean = self.schedule.pop(0)
            self.means[arm] = new_mean

    def _beta_parameters(self, mean: float) -> tuple[float, float]:
        """Solve scaled-Beta alpha/beta from target mean and SD.

        Let p = mean/R_max and c = alpha+beta. For R_max*Beta(alpha,beta):
            variance = R_max^2 * p(1-p)/(c+1).
        Thus c = p(1-p)(R_max/sigma)^2 - 1, alpha=p*c, beta=(1-p)*c.
        Validation in _validate_means guarantees c > 0.
        """
        p = mean / self.R_max
        concentration = p * (1.0 - p) * (self.R_max / self.sigma) ** 2 - 1.0
        return p * concentration, (1.0 - p) * concentration

    def _draw_arm(self, arm: int) -> float:
        alpha, beta = self._beta_parameters(float(self.means[arm]))
        return float(self.R_max * self.rng.beta(alpha, beta))

    def reward_fn(self, arm_or_arms: int | Sequence[int], t: int) -> float:
        """Return one single-arm reward or one simultaneous group average."""
        if not isinstance(t, int) or t <= 0:
            raise ValueError("t must be a positive, 1-indexed integer.")

        self._apply_due_changes(t)
        self.best_mean_history[t] = float(np.max(self.means))

        if isinstance(arm_or_arms, (list, tuple, np.ndarray)):
            arms = np.asarray(arm_or_arms, dtype=int)
            if arms.ndim != 1 or len(arms) == 0:
                raise ValueError("A group probe must contain at least one arm index.")
            if np.any(arms < 0) or np.any(arms >= self.K):
                raise IndexError(f"Group contains an invalid arm; valid range is [0, {self.K - 1}].")
            # Independent draws at the same global time, then only the
            # average is returned: exact implementation of TS-GE Assumption 2.
            return float(np.mean([self._draw_arm(int(arm)) for arm in arms]))

        if not isinstance(arm_or_arms, (int, np.integer)):
            raise TypeError("arm_or_arms must be an int or a sequence of ints.")
        arm = int(arm_or_arms)
        if not 0 <= arm < self.K:
            raise IndexError(f"arm={arm} is outside valid range [0, {self.K - 1}].")
        return self._draw_arm(arm)


def regret_from_history(history, best_mean_history, indexing_start: int = 1) -> list[float]:
    """Return per-slot regret against the environment's true best mean.

    For a BP/GE group-probe slot, the selected reward in history['reward']
    is the group-average reward. The resulting value is therefore system
    reward regret under the multi-play average-reward model, not the regret
    of an individual-arm pull. This is the appropriate comparison when
    evaluating TS-GE under its Assumption 2.
    """
    regrets: list[float] = []
    for offset, reward in enumerate(history["reward"]):
        t = indexing_start + offset
        if t not in best_mean_history:
            raise ValueError(
                f"No oracle best mean recorded for t={t}. The algorithm's reward history "
                "and environment time indexing are inconsistent."
            )
        regrets.append(float(best_mean_history[t] - reward))
    return regrets
