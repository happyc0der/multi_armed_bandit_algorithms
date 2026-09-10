"""TS-GE with BP/GE diagnostics and statistically valid post-change recovery.

Reference: G. Ghatak, "Actively Tracking the Optimal Arm in Non-Stationary
Environments with Mandatory Probing" (docs/TS_GE.pdf). Algorithm 1 is TS-GE,
Algorithm 2 is Construct Super-Arms (CSA).

What follows the paper exactly:
- n_ETC = (1 / 2*delta^2) * ln(1 / p_L) with p_L = 1/T           (Lemma 1)
- T_l = sqrt(T), T_BP = T^(2/5), T_TS = T_l - T_BP               (Section II-C)
- BP change test |mean_i(mu_hat_i) - mean(R_BP)| >= 4*delta      (Eq. 5)
- GE group test |mu_hat_Bk - mean(R_Bk)| >= 2*delta              (Eq. 7)
- mu_hat updated from individual pulls only; group probes never
  touch it                                                       (Eq. 4)
- Bernoulli success signal R_pi = R / R_max                       (Assumption 1)
- every arm is probed at least once per episode, via the BP
  super-arm                                                       (Condition 1)

Deliberate deviations, each measured (see README.md):

1. Beta update direction. Algorithm 1 lines 11-12 read
   `alpha += 1 - R*`, `beta += R*`, while line 7 selects `argmax theta`.
   Beta(alpha, beta) has mean alpha/(alpha+beta), which under that update is
   the FAILURE rate, so the paper as printed selects the arm most likely to
   fail. On K=2, T=6000, seed 42 the literal version costs 21,860 of TS-phase
   regret against 174 for the corrected version. The corrected update is the
   default; `reproduce_paper_beta_typo=True` restores the literal text.

2. Mean recovery. Eq. (8) sums group AVERAGES and subtracts individual MEANS,
   which is dimensionally inconsistent. `|B_k| * mu_hat_Bk - sum_{i != j} mu_hat_i`
   is the intended quantity and is what this code computes, averaged over the
   flagged groups that contain the localized arm.

3. Post-change recovery is verified, not assumed. The paper re-seeds the
   changed arm's Beta prior from the closest other arm (Algorithm 1 line 29).
   Instead this code takes `recovery_samples` fresh individual pulls of the
   localized arm and rebuilds its posterior from that evidence, because
   BP/GE windows can straddle a change point and a borrowed pseudo-count is
   not justified by any observation of the arm itself.

   Those same fresh pulls also gate the reset. `_groups()` codes arms
   0..K-1 over d = log2(K) bits, so arm 0 owns the all-zero codeword and an
   empty flag pattern - which is also what GE returns when it finds nothing -
   resolves to arm 0. Committing unconditionally therefore blamed arm 0 for
   every BP false alarm and wiped its statistics. The reset is now committed
   only when |recovery_mean - old_mean| >= 2*delta; otherwise the recovery
   pulls are merged into the arm's existing statistics and the event is
   logged as `recovery_rejected_change`.

4. Sample-count scale. The paper's n_ETC is scale-free because its rewards are
   unit-variance. `noise_scale` makes the dependence explicit: pass the
   environment's sigma for the sub-Gaussian count, or R_max for the Hoeffding
   count for [0, R_max]-bounded rewards. The default 1.0 reproduces the
   paper's formula verbatim.

5. Phase lengths are tunable, and the paper's are far from optimal here. Both
   policies below default to the paper's formula; the alternatives are opt-in
   and step OUTSIDE the paper's proof, so they are a measured engineering
   trade, not a claim about the theory.

   `bp_length_policy="power"` sizes T_BP by detection power rather than by
   T^(2/5). Measured on the benchmark's own cases (3 seeds, detections per
   seed in brackets):

       K=16, T=1e5:  T_BP=100 -> 219,104 total [1.0]   T_BP=3 -> 20,922 [1.0]
       K=64, T=1e5:  T_BP=100 -> 305,980 total [1.0]   T_BP=6 -> 79,390 [1.0]

   `etc_policy="detection"` sizes n_ETC for the averaged estimates the
   detection tests use rather than for individual well-localization. With
   T_BP already power-derived (5 seeds, zero missed detections throughout):

       K=64, T=1e5:  n_etc=341 (21.8% of horizon) -> 76,827 total
                     n_etc= 22 ( 1.4% of horizon) -> 49,671 total

   Together they take K=64 from the paper's 306,486 to 49,671, past M-UCB's
   109,861 and AdSwitch's 55,637 +/- 58,403. Detection latency rises (264 ->
   517 slots at K=64) and probe age rises slightly, both well within the
   sqrt(T) budget. These were measured over 3-5 seeds; one configuration
   between the paper's value and the derived one (K=64, T_BP=12) missed a
   detection on 1 of 3 seeds, so widen the sweep before trusting a new
   operating point.

Design rules this implementation cannot escape, and which callers must respect
(bench/preflight.py checks them):
- A single arm's change of Delta_C shifts the K-arm BP average by only
  Delta_C / K, so detection needs Delta_C / K >= 4*delta.
- ETC costs K * noise_scale^2 * ln(T) / (2*delta^2) slots. Combined with the
  rule above, a horizon T supports at most
  K^3 <= rho * T * (Delta_C / noise_scale)^2 / (8 * ln T) arms, where rho is
  the fraction of the horizon you are willing to spend on ETC.
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
        reproduce_paper_beta_typo: bool = False,
        recovery_samples: int = 50,
        noise_scale: float = 1.0,
        bp_length_policy: str = "paper",
        etc_policy: str = "paper",
        diagnostics: bool = True,
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
        if not math.isfinite(noise_scale) or noise_scale <= 0:
            raise ValueError("noise_scale must be finite and > 0.")
        if bp_length_policy not in ("paper", "power"):
            raise ValueError("bp_length_policy must be 'paper' or 'power'.")
        if etc_policy not in ("paper", "detection"):
            raise ValueError("etc_policy must be 'paper' or 'detection'.")

        super().__init__(K, T)
        self.delta = float(delta)
        self.R_max = float(R_max)
        self.n_ge = n_ge
        self.recovery_samples = recovery_samples
        self.reproduce_paper_beta_typo = reproduce_paper_beta_typo
        self.noise_scale = float(noise_scale)
        self.bp_length_policy = bp_length_policy
        self.etc_policy = etc_policy
        self.diagnostics = diagnostics
        self.rng = np.random.default_rng(seed)
        self.d = int(math.log2(K))

    # -- schedule, exposed so callers can budget before running ---------------
    @staticmethod
    def paper_bp_length(T: int) -> int:
        """T_BP = T^(2/5), Section II-C."""
        return max(round(T**0.4), 1)

    def power_bp_length(self, T: int) -> int:
        """T_BP sized by the detection power the BP test actually needs.

        The BP statistic averages K arms over n_bp slots, so its standard
        error is noise_scale / sqrt(K * n_bp). Requiring that to sit at half
        the 4*delta detection threshold gives

            n_bp = 2 * ln(2T) * noise_scale^2 / (K * (2 delta)^2)

        which SHRINKS as K grows -- the paper's T^(2/5) does not depend on K
        at all. See deviation 5.
        """
        return max(
            math.ceil(
                2.0 * math.log(2.0 * max(T, 2)) * self.noise_scale**2
                / (self.K * (2.0 * self.delta) ** 2)
            ),
            1,
        )

    def phase_lengths(self, T: int) -> tuple[int, int, int]:
        """Return (episode_length, ts_length, bp_length) for horizon T.

        T_l = sqrt(T) per Section II-C. T_BP follows `bp_length_policy`;
        T_TS is the remainder so the two phases exactly fill the episode.
        Condition 1 holds under either policy, since one BP slot per episode
        already probes every arm.
        """
        episode_length = max(math.isqrt(T), 2)
        if self.bp_length_policy == "power":
            bp_length = self.power_bp_length(T)
        else:
            bp_length = self.paper_bp_length(T)
        bp_length = max(min(bp_length, episode_length - 1), 1)
        ts_length = max(episode_length - bp_length, 1)
        return episode_length, ts_length, bp_length

    def etc_samples_per_arm(self) -> int:
        """Per-arm ETC pulls, following `etc_policy`.

        'paper' is Lemma 1, n_ETC = noise_scale^2 * ln(1/p_L) / (2 delta^2),
        which sizes each arm to be INDIVIDUALLY well-localized (Definition 1).

        'detection' divides that by K. Every use of mu_hat in the detection
        path is an average -- the BP test over all K arms, the GE test over
        K/2 -- and an average of K estimates has error sqrt(K) times smaller
        than any one of them. The step that did need individual accuracy,
        Eq. (8)'s algebraic recovery, is not what this code relies on: it
        re-estimates the localized arm from fresh recovery pulls. See
        deviation 5.
        """
        p_localization_failure = 1.0 / max(self.T, 2)
        samples = (
            self.noise_scale**2
            * math.log(1.0 / p_localization_failure)
            / (2.0 * self.delta**2)
        )
        if self.etc_policy == "detection":
            samples /= self.K
        return max(math.ceil(samples), 1)

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
        """CSA (Algorithm 2), 0-indexed over d = log2(K) bits.

        The paper indexes arms 1..K, which needs ceil(log2(K+1)) super-arms and
        leaves no arm with the empty codeword. This 0-indexed variant uses one
        fewer super-arm (so one fewer n_ge probe burst per detection) at the
        cost of giving arm 0 the empty codeword; the recovery gate in `run`
        is what makes that safe. See deviation 3 in the module docstring.
        """
        return [[arm for arm in range(self.K) if (arm >> bit) & 1] for bit in range(self.d)]

    def run(self, reward_fn: RewardFn) -> dict[str, Any]:
        K, T, delta = self.K, self.T, self.delta
        keep = self.diagnostics
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

        n_etc = self.etc_samples_per_arm()

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

        episode_length, ts_length, bp_length = self.phase_lengths(T)
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
                if self.reproduce_paper_beta_typo:
                    alpha[arm] += 1 - success
                    beta[arm] += success
                else:
                    alpha[arm] += success
                    beta[arm] += 1 - success
                sums[arm] += reward
                pulls[arm] += 1
                if keep:
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
                "ts_reward_mean": float(np.mean(ts_rewards)) if ts_rewards else None,
                "bp_start_time": bp_start,
                "bp_end_time": bp_end,
                "bp_sample_count": len(bp_rewards),
                "bp_observed_mean": bp_mean,
                "bp_observed_std": float(np.std(bp_rewards)),
                "bp_baseline_mean": baseline,
                "bp_difference": bp_difference,
                "bp_absolute_difference": abs(bp_difference),
                "bp_threshold": bp_threshold,
                "detected": detected,
            }
            if keep:
                bp_event.update({
                    "ts_arms": ts_arms,
                    "bp_rewards": bp_rewards,
                    "bp_observed_min": float(np.min(bp_rewards)),
                    "bp_observed_max": float(np.max(bp_rewards)),
                    "bp_baseline_arm_means": baseline_arm_means,
                    "bp_baseline_pull_counts": baseline_pull_counts,
                })
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
                ge_event = {
                    "bit": bit, "group_size": len(group),
                    "ge_start_time": ge_start, "ge_end_time": ge_end,
                    "ge_sample_count": len(samples),
                    "old_group_mean": old_group_means[bit], "new_group_mean": new_value,
                    "difference": difference,
                    "absolute_difference": abs(difference) if difference is not None else None,
                    "threshold": ge_threshold, "flagged": flags[bit],
                }
                if keep:
                    ge_event["group_arms"] = list(group)
                    ge_event["ge_rewards"] = samples
                ge_events.append(ge_event)

            base = {
                "episode_index": episode,
                "bp_detection_time": bp_end,
                "localization_complete_time": t,
                "bp_start_time": bp_start,
                "bp_end_time": bp_end,
                "bp_sample_count": len(bp_rewards),
                "bp_baseline_mean": baseline,
                "bp_observed_mean": bp_mean,
                "bp_difference": bp_difference,
                "bp_absolute_difference": abs(bp_difference),
                "bp_threshold": bp_threshold,
                "flagged_groups": flags,
                # True means GE found no group whose mean moved. Under the
                # 0-indexed coding this is indistinguishable from "arm 0
                # changed", so the recovery pulls below decide between them.
                "all_flags_false": not any(flags),
                "ge_threshold": ge_threshold,
                "ge_groups": ge_events,
            }
            if keep:
                base["bp_baseline_arm_means"] = baseline_arm_means
                base["bp_baseline_pull_counts"] = baseline_pull_counts
                base["bp_observed_std"] = float(np.std(bp_rewards))
            if any(value is None for value in new_group_means):
                localization_events.append({**base, "localized_arm": None, "reason": "horizon_ended_during_GE"})
                break

            candidates = [arm for arm in range(K) if [bool((arm >> bit) & 1) for bit in range(self.d)] == flags]
            if len(candidates) != 1:
                # Unreachable with the 0-indexed coding: every one of the 2^d
                # flag patterns maps to exactly one arm in 0..K-1. Kept as a
                # guard in case `_groups` is ever changed to the paper's
                # 1-indexed coding, where an empty pattern matches no arm.
                localization_events.append({
                    **base, "localized_arm": None,
                    "reason": "no_arm_matched_pattern" if not candidates else "ambiguous_group_pattern",
                    "candidates": candidates,
                })
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
                if keep:
                    group_recovery.append({
                        "bit": bit, "group_arms": list(group),
                        "observed_group_mean": group_mean, "other_arms": others,
                        "other_mean_sum": other_sum, "recovered_arm_mean": estimate,
                    })

            ge_recovered = float(np.mean(group_estimates)) if group_estimates else None

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

            # The gate. GE's flag pattern is a hypothesis about WHICH arm moved;
            # these fresh individual pulls are the only direct evidence that it
            # moved at all. Without this check a BP false alarm discards an
            # arm's entire history in exchange for `recovery_samples` samples.
            #
            # The tolerance is 2*delta OR the recovery estimator's own
            # confidence radius, whichever is larger. Using 2*delta alone lets
            # noise in `recovery_samples` observations clear the bar on its own:
            # at sigma=3 and 50 samples the standard error is 0.42 against a
            # threshold of 0.20, so roughly two thirds of false alarms still
            # committed a reset.
            #
            # The radius is estimated from the recovery samples rather than from
            # `noise_scale`, because `noise_scale` also sets n_ETC: raising it to
            # the true reward sigma to widen this gate would multiply the ETC
            # phase by the same factor squared. The two quantities need
            # different constants, so this one is measured, not configured.
            # sqrt(2 ln 2T) is the union bound over the horizon, matching the
            # paper's 1/T failure budget.
            if len(recovery_rewards) > 1:
                standard_error = float(np.std(recovery_rewards, ddof=1)) / math.sqrt(
                    len(recovery_rewards)
                )
            else:
                standard_error = self.R_max
            recovery_radius = standard_error * math.sqrt(2.0 * math.log(2.0 * max(T, 2)))
            recovery_tolerance = max(ge_threshold, recovery_radius)
            confirmed = abs(new_mean - old_mean) >= recovery_tolerance
            successes = int(sum(recovery_successes))
            failures = len(recovery_successes) - successes
            if confirmed:
                sums[changed] = float(sum(recovery_rewards))
                pulls[changed] = len(recovery_rewards)
                # Posterior confidence comes from the actual recovery evidence,
                # not from another arm's accumulated pseudo-count.
                if self.reproduce_paper_beta_typo:
                    alpha[changed] = 1.0 + failures
                    beta[changed] = 1.0 + successes
                else:
                    alpha[changed] = 1.0 + successes
                    beta[changed] = 1.0 + failures
            else:
                sums[changed] += float(sum(recovery_rewards))
                pulls[changed] += len(recovery_rewards)
                if self.reproduce_paper_beta_typo:
                    alpha[changed] += failures
                    beta[changed] += successes
                else:
                    alpha[changed] += successes
                    beta[changed] += failures

            event = {
                **base,
                "localized_arm": changed,
                "reason": "localized" if confirmed else "recovery_rejected_change",
                "confirmed": confirmed,
                "recovery_tolerance": recovery_tolerance,
                "candidates": candidates,
                "old_mean": old_mean,
                "old_pull_count": old_pulls,
                "new_mean": new_mean,
                "ge_recovered_mean": ge_recovered,
                "used_bp_fallback": not group_estimates,
                "recovery_start_time": recovery_start,
                "recovery_end_time": recovery_end,
                "recovery_sample_count": len(recovery_rewards),
                "recovery_mean": new_mean,
                "posterior_alpha_before_reset": alpha_before,
                "posterior_beta_before_reset": beta_before,
                "posterior_alpha_after_reset": float(alpha[changed]),
                "posterior_beta_after_reset": float(beta[changed]),
            }
            if keep:
                other_arms = [arm for arm in range(K) if arm != changed]
                closest = min(other_arms, key=lambda arm: abs(mean(arm) - new_mean))
                event.update({
                    "old_reward_sum": old_sum,
                    "per_group_estimates": group_estimates,
                    "per_group_recovery_details": group_recovery,
                    "recovery_rewards": recovery_rewards,
                    "recovery_successes": recovery_successes,
                    # The paper (Algorithm 1 line 29) would re-seed the changed
                    # arm's prior from this arm. Recorded for comparison only.
                    "closest_arm": closest,
                    "closest_arm_mean": mean(closest),
                    "posterior_pseudo_count": float(alpha[changed] + beta[changed]),
                    "reseeded_empirical_sum": float(sums[changed]),
                    "reseeded_empirical_pull_count": int(pulls[changed]),
                })
            localization_events.append(event)

        return {
            "net_reward": float(sum(rewards)), "reward": rewards,
            "chosen_arm": chosen, "phase": phases, "detections": detections,
            "bp_events": bp_events, "localization_events": localization_events,
            "n_etc": n_etc, "n_ge": self.n_ge,
            "recovery_samples": self.recovery_samples,
            "noise_scale": self.noise_scale,
            "bp_length_policy": self.bp_length_policy,
            "etc_policy": self.etc_policy,
            "episode_length": episode_length, "ts_length": ts_length,
            "bp_length": bp_length, "bp_threshold": bp_threshold,
            "ge_threshold": ge_threshold, "groups": groups,
        }
