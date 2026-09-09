"""TS-GE with BP/GE diagnostics and statistically valid post-change recovery.

Key choices:
- BP detects a system change; GE localizes its arm.
- Fresh individual recovery pulls estimate the localized arm after GE.
  BP/GE estimates are retained only as diagnostics because either window can
  straddle a change point.
- The changed arm's Beta posterior is rebuilt from its actual recovery
  Bernoulli observations. It is NOT given the pseudo-count of another arm.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from bandit_base import BanditAlgorithm

RewardFn = Callable[[int | list[int], int], float]


class TS_GE(BanditAlgorithm):
    def __init__(
        self,
        K: int,
        T: int,
        delta: float,
        R_max: float,
        n_ge: int,
        seed: int | None = None,
        use_paper_beta_update: bool = False,
        recovery_samples: int = 50,
    ) -> None:
        if not isinstance(K, int) or K < 2 or K & (K - 1):
            raise ValueError("TS_GE requires K to be a power of two and K >= 2.")
        if not isinstance(T, int) or T <= 0:
            raise ValueError("T must be a positive integer.")
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and > 0.")
        if not math.isfinite(R_max) or R_max <= 0:
            raise ValueError("R_max must be finite and > 0.")
        if not isinstance(n_ge, int) or n_ge <= 0:
            raise ValueError("n_ge must be a positive integer.")
        if not isinstance(recovery_samples, int) or recovery_samples <= 0:
            raise ValueError("recovery_samples must be a positive integer.")

        super().__init__(K, T)
        self.delta = float(delta)
        self.R_max = float(R_max)
        self.n_ge = n_ge
        self.recovery_samples = recovery_samples
        self.use_paper_beta_update = use_paper_beta_update
        self.rng = np.random.default_rng(seed)
        self.d = int(math.log2(K))

    def _validate_reward(self, reward: float, source: str, t: int) -> float:
        reward = float(reward)
        if not math.isfinite(reward) or not 0.0 <= reward <= self.R_max:
            raise ValueError(
                f"{source} reward at t={t} must be finite and in "
                f"[0, {self.R_max}]; got {reward!r}."
            )
        return reward

    def _single_reward(self, reward_fn: RewardFn, arm: int, t: int) -> float:
        return self._validate_reward(reward_fn(arm, t), f"arm {arm}", t)

    def _group_reward(self, reward_fn: RewardFn, arms: Sequence[int], t: int) -> float:
        if not arms:
            raise ValueError("A TS-GE super-arm cannot be empty.")
        return self._validate_reward(reward_fn(list(arms), t), f"group {list(arms)}", t)

    def _success(self, reward: float) -> int:
        return int(self.rng.binomial(1, reward / self.R_max))

    def _groups(self) -> list[list[int]]:
        return [[arm for arm in range(self.K) if (arm >> bit) & 1] for bit in range(self.d)]

    def run(self, reward_fn: RewardFn) -> dict[str, Any]:
        K, T, delta = self.K, self.T, self.delta
        alpha = np.ones(K, dtype=float)
        beta = np.ones(K, dtype=float)
        sums = np.zeros(K, dtype=float)
        pulls = np.zeros(K, dtype=np.int64)
        rewards: list[float] = []
        chosen: list[int | str] = []
        phases: list[str] = []
        detections: list[int] = []
        bp_events: list[dict[str, Any]] = []
        localization_events: list[dict[str, Any]] = []
        t = 0

        def mean(arm: int) -> float:
            if pulls[arm] == 0:
                raise RuntimeError(f"No individual mean exists for arm {arm}.")
            return float(sums[arm] / pulls[arm])

        def record(reward: float, action: int | str, phase: str) -> None:
            rewards.append(reward)
            chosen.append(action)
            phases.append(phase)

        p_localization_failure = 1.0 / max(T, 2)
        # Retained for backward-compatible experiments. Note that delta is
        # interpreted in the raw-reward units used by the rest of this code.
        n_etc = max(math.ceil(math.log(1.0 / p_localization_failure) / (2.0 * delta**2)), 1)

        for arm in range(K):
            for _ in range(n_etc):
                if t >= T:
                    break
                t += 1
                reward = self._single_reward(reward_fn, arm, t)
                sums[arm] += reward
                pulls[arm] += 1
                record(reward, arm, "ETC")
            if t >= T:
                break

        episode_length = max(math.isqrt(T), 2)
        ts_length = max(int(math.sqrt(T) - T**0.4), 1)
        bp_length = max(episode_length - ts_length, 1)
        bp_threshold = 4.0 * delta
        ge_threshold = 2.0 * delta
        groups = self._groups()
        episode = 0

        while t < T:
            episode += 1
            episode_start = t + 1
            ts_start = t + 1
            ts_arms: list[int] = []
            ts_rewards: list[float] = []

            for _ in range(ts_length):
                if t >= T:
                    break
                arm = int(np.argmax(self.rng.beta(alpha, beta)))
                t += 1
                reward = self._single_reward(reward_fn, arm, t)
                success = self._success(reward)
                if self.use_paper_beta_update:
                    alpha[arm] += 1 - success
                    beta[arm] += success
                else:
                    alpha[arm] += success
                    beta[arm] += 1 - success
                sums[arm] += reward
                pulls[arm] += 1
                ts_arms.append(arm)
                ts_rewards.append(reward)
                record(reward, arm, "TS")

            if t >= T:
                break

            baseline_arm_means = [mean(arm) for arm in range(K)]
            baseline_pull_counts = [int(x) for x in pulls]
            baseline = float(np.mean(baseline_arm_means))
            bp_start = t + 1
            bp_rewards: list[float] = []

            for _ in range(bp_length):
                if t >= T:
                    break
                t += 1
                reward = self._group_reward(reward_fn, list(range(K)), t)
                bp_rewards.append(reward)
                record(reward, "BP", "BP")

            if not bp_rewards:
                break
            bp_end = t
            bp_mean = float(np.mean(bp_rewards))
            bp_difference = bp_mean - baseline
            detected = abs(bp_difference) >= bp_threshold
            bp_event = {
                "episode_index": episode,
                "episode_start_time": episode_start,
                "ts_start_time": ts_start,
                "ts_end_time": bp_start - 1,
                "ts_slot_count": len(ts_rewards),
                "ts_arms": ts_arms,
                "ts_reward_mean": float(np.mean(ts_rewards)) if ts_rewards else None,
                "bp_start_time": bp_start,
                "bp_end_time": bp_end,
                "bp_sample_count": len(bp_rewards),
                "bp_rewards": bp_rewards,
                "bp_observed_mean": bp_mean,
                "bp_observed_std": float(np.std(bp_rewards)),
                "bp_observed_min": float(np.min(bp_rewards)),
                "bp_observed_max": float(np.max(bp_rewards)),
                "bp_baseline_mean": baseline,
                "bp_baseline_arm_means": baseline_arm_means,
                "bp_baseline_pull_counts": baseline_pull_counts,
                "bp_difference": bp_difference,
                "bp_absolute_difference": abs(bp_difference),
                "bp_threshold": bp_threshold,
                "detected": detected,
            }
            bp_events.append(bp_event)

            if not detected:
                continue
            detections.append(bp_end)

            old_group_means = [float(np.mean([mean(arm) for arm in group])) for group in groups]
            new_group_means: list[float | None] = [None] * self.d
            flags = [False] * self.d
            ge_events: list[dict[str, Any]] = []

            for bit, group in enumerate(groups):
                ge_start = t + 1
                samples: list[float] = []
                for _ in range(self.n_ge):
                    if t >= T:
                        break
                    t += 1
                    reward = self._group_reward(reward_fn, group, t)
                    samples.append(reward)
                    record(reward, f"GE:{bit}", f"GE:{bit}")
                ge_end = t
                new_value = float(np.mean(samples)) if samples else None
                difference = new_value - old_group_means[bit] if new_value is not None else None
                flags[bit] = difference is not None and abs(difference) >= ge_threshold
                new_group_means[bit] = new_value
                ge_events.append({
                    "bit": bit, "group_arms": list(group), "group_size": len(group),
                    "ge_start_time": ge_start, "ge_end_time": ge_end,
                    "ge_sample_count": len(samples), "ge_rewards": samples,
                    "old_group_mean": old_group_means[bit], "new_group_mean": new_value,
                    "difference": difference,
                    "absolute_difference": abs(difference) if difference is not None else None,
                    "threshold": ge_threshold, "flagged": flags[bit],
                })

            base = {
                "episode_index": episode,
                "bp_detection_time": bp_end,
                "localization_complete_time": t,
                "bp_start_time": bp_start,
                "bp_end_time": bp_end,
                "bp_sample_count": len(bp_rewards),
                "bp_baseline_mean": baseline,
                "bp_baseline_arm_means": baseline_arm_means,
                "bp_baseline_pull_counts": baseline_pull_counts,
                "bp_observed_mean": bp_mean,
                "bp_observed_std": float(np.std(bp_rewards)),
                "bp_difference": bp_difference,
                "bp_absolute_difference": abs(bp_difference),
                "bp_threshold": bp_threshold,
                "flagged_groups": flags,
                "ge_threshold": ge_threshold,
                "ge_groups": ge_events,
            }
            if any(value is None for value in new_group_means):
                localization_events.append({**base, "localized_arm": None, "reason": "horizon_ended_during_GE"})
                break

            candidates = [arm for arm in range(K) if [bool((arm >> bit) & 1) for bit in range(self.d)] == flags]
            if len(candidates) != 1:
                localization_events.append({**base, "localized_arm": None, "reason": "ambiguous_group_pattern", "candidates": candidates})
                continue

            changed = candidates[0]
            old_mean = mean(changed)
            old_pulls = int(pulls[changed])
            old_sum = float(sums[changed])
            group_estimates: list[float] = []
            group_recovery: list[dict[str, Any]] = []
            for bit, group in enumerate(groups):
                if not flags[bit] or changed not in group:
                    continue
                group_mean = new_group_means[bit]
                assert group_mean is not None
                others = [arm for arm in group if arm != changed]
                other_sum = float(sum(mean(arm) for arm in others))
                estimate = float(len(group) * group_mean - other_sum)
                group_estimates.append(estimate)
                group_recovery.append({"bit": bit, "group_arms": list(group), "observed_group_mean": group_mean, "other_arms": others, "other_mean_sum": other_sum, "recovered_arm_mean": estimate})

            ge_recovered = float(np.mean(group_estimates)) if group_estimates else None
            used_bp_fallback = not group_estimates
            bp_shift = float(K * bp_difference) if used_bp_fallback else None
            bp_fallback_mean = old_mean + bp_shift if bp_shift is not None else None

            recovery_start = t + 1
            recovery_rewards: list[float] = []
            recovery_successes: list[int] = []
            for _ in range(self.recovery_samples):
                if t >= T:
                    break
                t += 1
                reward = self._single_reward(reward_fn, changed, t)
                recovery_rewards.append(reward)
                recovery_successes.append(self._success(reward))
                record(reward, changed, "RECOVERY")
            recovery_end = t

            if not recovery_rewards:
                localization_events.append({**base, "localized_arm": changed, "reason": "horizon_ended_during_recovery", "candidates": candidates})
                break

            new_mean = float(np.mean(recovery_rewards))
            if not 0 <= new_mean <= self.R_max:
                raise ValueError("Fresh recovery mean lies outside [0, R_max].")
            alpha_before, beta_before = float(alpha[changed]), float(beta[changed])
            sums[changed] = float(sum(recovery_rewards))
            pulls[changed] = len(recovery_rewards)

            # Critical correction: posterior confidence comes from the actual
            # recovery evidence (n recovery samples), not another arm's 1224
            # observations. The old closest-arm pseudo-count was unjustified.
            successes = int(sum(recovery_successes))
            failures = len(recovery_successes) - successes
            if self.use_paper_beta_update:
                alpha[changed] = 1.0 + failures
                beta[changed] = 1.0 + successes
            else:
                alpha[changed] = 1.0 + successes
                beta[changed] = 1.0 + failures

            other_arms = [arm for arm in range(K) if arm != changed]
            closest = min(other_arms, key=lambda arm: abs(mean(arm) - new_mean))
            localization_events.append({
                **base,
                "localized_arm": changed,
                "reason": "localized",
                "candidates": candidates,
                "old_mean": old_mean,
                "old_pull_count": old_pulls,
                "old_reward_sum": old_sum,
                "new_mean": new_mean,
                "per_group_estimates": group_estimates,
                "per_group_recovery_details": group_recovery,
                "ge_recovered_mean": ge_recovered,
                "used_bp_fallback": used_bp_fallback,
                "bp_fallback_shift_estimate": bp_shift,
                "bp_fallback_mean": bp_fallback_mean,
                "recovery_start_time": recovery_start,
                "recovery_end_time": recovery_end,
                "recovery_sample_count": len(recovery_rewards),
                "recovery_rewards": recovery_rewards,
                "recovery_successes": recovery_successes,
                "recovery_mean": new_mean,
                "closest_arm": closest,
                "closest_arm_mean": mean(closest),
                "posterior_pseudo_count": float(alpha[changed] + beta[changed]),
                "posterior_alpha_before_reset": alpha_before,
                "posterior_beta_before_reset": beta_before,
                "posterior_alpha_after_reset": float(alpha[changed]),
                "posterior_beta_after_reset": float(beta[changed]),
                "reseeded_empirical_sum": float(sums[changed]),
                "reseeded_empirical_pull_count": int(pulls[changed]),
            })

        return {
            "net_reward": float(sum(rewards)), "reward": rewards,
            "chosen_arm": chosen, "phase": phases, "detections": detections,
            "bp_events": bp_events, "localization_events": localization_events,
            "n_etc": n_etc, "n_ge": self.n_ge,
            "recovery_samples": self.recovery_samples,
            "episode_length": episode_length, "ts_length": ts_length,
            "bp_length": bp_length, "bp_threshold": bp_threshold,
            "ge_threshold": ge_threshold, "groups": groups,
        }
