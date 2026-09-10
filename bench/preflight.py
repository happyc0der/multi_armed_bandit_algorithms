"""Parameter feasibility checks, run before any simulation.

WHY THIS EXISTS
===============
TS-GE has two design constraints that pull against each other, and violating
either produces a run that *completes normally* and reports a large regret
number that means nothing:

1. DETECTABILITY. A change of Delta_C in ONE arm shifts the K-arm broadcast
   average by only Delta_C / K, so the BP test needs

       Delta_C / K >= 4 * delta.

   Violated at K=64, Delta_C=14, delta=0.1: the shift is 0.219 against a
   threshold of 0.400, and the algorithm registered ZERO detections across
   T=200,000 -- linear regret after the change, no warning anywhere.

2. ETC BUDGET. Initialization costs

       K * noise_scale^2 * ln(T) / (2 * delta^2)  slots.

   Violated at K=64, T=6000, delta=0.1: ETC alone needs 27,840 of 6,000 slots,
   so 100% of the horizon is initialization, zero episodes run, and the
   reported regret (49,555) measures nothing but round-robin exploration.

Satisfying (1) forces delta <= Delta_C / (4K), and substituting into (2) gives
the combined feasibility rule this module reports:

       K^3 <= rho * T * (Delta_C / noise_scale)^2 / (8 * ln T)

with rho the fraction of the horizon you will spend on ETC. At T=1e5,
Delta_C/noise_scale=28 and rho=0.25 that is K <= 60; T=1e6 gives K <= 121.

A third check comes from the paper itself: Eq. (12) DERIVES delta from the
noise level via P_FA = Q(4*delta / sigma_NC) <= 1/T. A hardcoded delta that
ignores sigma, K and the phase lengths has no false-alarm guarantee.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from ts_ge import TS_GE

#: Fraction of the horizon that may be spent on ETC before a case is rejected.
ETC_BUDGET = 0.5
#: Fraction used when reporting the "comfortable" feasible-K figure.
ETC_BUDGET_COMFORTABLE = 0.25

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"


@dataclass
class Finding:
    level: str
    check: str
    message: str

    def __str__(self) -> str:
        return f"  [{self.level:5s}] {self.check:22s} {self.message}"


class PreflightError(RuntimeError):
    """Raised when a case cannot produce a meaningful result."""


def _q(x: float) -> float:
    """Gaussian tail Q(x) = P(Z > x)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _q_inverse(p: float) -> float:
    """Smallest x with Q(x) <= p, by bisection on the monotone tail."""
    if not 0.0 < p < 0.5:
        return 0.0
    low, high = 0.0, 40.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _q(mid) > p:
            low = mid
        else:
            high = mid
    return high


def change_magnitudes(means: Iterable[float], schedule: Iterable[tuple[int, int, float]]) -> list[float]:
    """|new mean - mean in force at that moment| for every scheduled change."""
    current = list(means)
    magnitudes = []
    for _, arm, new_mean in sorted(schedule, key=lambda item: item[0]):
        magnitudes.append(abs(new_mean - current[arm]))
        current[arm] = new_mean
    return magnitudes


def feasible_K(T: int, delta_c: float, noise_scale: float, rho: float = ETC_BUDGET_COMFORTABLE) -> float:
    """Largest K for which detectability and the ETC budget can both hold."""
    return (rho * T * (delta_c / noise_scale) ** 2 / (8.0 * math.log(max(T, 3)))) ** (1.0 / 3.0)

def required_T(K: int, delta_c: float, noise_scale: float, rho: float = ETC_BUDGET_COMFORTABLE) -> float:
    """Smallest horizon at which this K is feasible (fixed-point on ln T)."""
    target = 8.0 * K**3 * noise_scale**2 / (rho * delta_c**2)
    horizon = max(target, 1000.0)
    for _ in range(100):
        horizon = target * math.log(max(horizon, 3.0))
    return horizon


def preflight(case: dict[str, Any]) -> list[Finding]:
    """Check one case. Returns findings; ERROR findings mean "do not run"."""
    K, T = case["K"], case["T"]
    delta = case["ts_ge_delta"]
    sigma = case["sigma"]
    noise_scale = case.get("ts_ge_noise_scale", 1.0)
    findings: list[Finding] = []

    algorithm = TS_GE(
        K=K, T=T, delta=delta, R_max=case["reward_R_max"],
        n_ge=case["ts_ge_n_ge"], noise_scale=noise_scale,
        bp_length_policy=case.get("ts_ge_bp_length_policy", "paper"),
        etc_policy=case.get("ts_ge_etc_policy", "paper"),
    )
    n_etc = algorithm.etc_samples_per_arm()
    episode_length, ts_length, bp_length = algorithm.phase_lengths(T)

    # -- ETC budget ---------------------------------------------------------
    etc_total = K * n_etc
    etc_share = etc_total / T
    level = ERROR if etc_share >= ETC_BUDGET else (WARN if etc_share >= 0.25 else INFO)
    message = (
        f"n_etc={n_etc}/arm, {etc_total} slots = {100 * etc_share:.1f}% of T={T}"
    )
    if level == ERROR:
        needed_T = math.ceil(etc_total / ETC_BUDGET)
        needed_delta = noise_scale * math.sqrt(K * math.log(max(T, 2)) / (2 * ETC_BUDGET * T))
        message += f". Raise T to >= {needed_T}, or delta to >= {needed_delta:.4g}"
        if noise_scale > sigma:
            # Only worth suggesting when the case is over-scaled: the paper's
            # scale-free noise_scale=1.0 costs (1/sigma)^2 times more ETC than
            # the sub-Gaussian constant does.
            message += (
                f", or pass ts_ge_noise_scale=sigma ({sigma}) instead of {noise_scale}"
                f" for a {(noise_scale / sigma) ** 2:.0f}x smaller n_etc"
            )
    findings.append(Finding(level, "etc_budget", message))

    # -- BP budget (informational; it is the dominant regret term) ----------
    findings.append(Finding(
        INFO, "bp_budget",
        f"episode={episode_length} (TS={ts_length}, BP={bp_length}) -> "
        f"{100 * bp_length / episode_length:.1f}% of post-ETC slots are broadcast probes, "
        f"charged at roughly (mu* - mean_i mu_i) each"
    ))

    # -- What each phase-length policy would cost ---------------------------
    # Shown for both policies whichever is selected, so the trade is visible
    # before the run rather than inferred from the regret afterwards.
    paper_bp = TS_GE.paper_bp_length(T)
    power_bp = algorithm.power_bp_length(T)
    bp_policy = case.get("ts_ge_bp_length_policy", "paper")
    findings.append(Finding(
        INFO, "bp_length_policy",
        f"policy={bp_policy!r}: paper T^(2/5)={paper_bp}, power-derived={power_bp} "
        f"(probe slots {'reduced' if power_bp < paper_bp else 'unchanged'} "
        f"{paper_bp / max(power_bp, 1):.0f}x). Power-derived steps outside the paper's proof."
    ))
    etc_policy = case.get("ts_ge_etc_policy", "paper")
    paper_etc = math.ceil(
        noise_scale**2 * math.log(max(T, 2)) / (2.0 * delta**2)
    )
    findings.append(Finding(
        INFO, "etc_policy",
        f"policy={etc_policy!r}: paper Lemma-1 n_etc={paper_etc}/arm "
        f"({100 * K * paper_etc / T:.1f}% of T), detection-sized={max(math.ceil(paper_etc / K), 1)}/arm "
        f"({100 * K * max(math.ceil(paper_etc / K), 1) / T:.1f}% of T)"
    ))

    # -- Detectability ------------------------------------------------------
    magnitudes = change_magnitudes(case["means"], case["change_schedule"])
    if not magnitudes:
        findings.append(Finding(INFO, "detectability", "no scheduled change; stationary case"))
    else:
        smallest = min(magnitudes)
        bp_shift = smallest / K
        level = ERROR if bp_shift < 4 * delta else INFO
        message = f"smallest Delta_C={smallest:g} -> BP shift {bp_shift:.4g} vs 4*delta={4 * delta:.4g}"
        if level == ERROR:
            message += f". Lower delta to <= {smallest / (4 * K):.4g}, or enlarge the change"
        findings.append(Finding(level, "bp_detectability", message))

        ge_shift = smallest / (K / 2)
        findings.append(Finding(
            INFO if ge_shift >= 2 * delta else WARN, "ge_detectability",
            f"GE shift {ge_shift:.4g} vs 2*delta={2 * delta:.4g}"
        ))

        # -- Combined feasibility rule --------------------------------------
        k_max = feasible_K(T, smallest, noise_scale)
        level = INFO if K <= k_max else WARN
        message = f"K={K} vs K_max={k_max:.1f} at T={T} (ETC budget {100 * ETC_BUDGET_COMFORTABLE:.0f}%)"
        if level == WARN:
            message += f". This K wants T >= {required_T(K, smallest, noise_scale):.3g}"
        findings.append(Finding(level, "feasible_K", message))

    # -- False alarms, Eq. (11)/(12) ---------------------------------------
    # sigma_NC^2 = (sigma^2/K) * (1/n_ETC + 1/(m*T_BP) + sum_j 1/n_j). Take the
    # conservative first episode (m=1) and assume no arm has been pulled beyond
    # ETC, which upper-bounds the last term by K/n_etc.
    sigma_nc = math.sqrt((sigma**2 / K) * (1.0 / n_etc + 1.0 / bp_length + K / n_etc))
    p_fa = 2.0 * _q(4 * delta / sigma_nc) if sigma_nc > 0 else 0.0
    expected_false_alarms = p_fa * max(T // max(episode_length, 1), 1)
    level = INFO if p_fa <= 1.0 / T else WARN
    findings.append(Finding(
        level, "false_alarm_rate",
        f"sigma_NC={sigma_nc:.4g}, P_FA={p_fa:.3g} vs target 1/T={1 / T:.3g} "
        f"(~{expected_false_alarms:.1f} false alarms over the run)"
    ))

    suggested_delta = sigma_nc * _q_inverse(1.0 / (2.0 * T)) / 4.0
    findings.append(Finding(
        INFO, "suggested_delta",
        f"Eq. (12) gives delta >= {suggested_delta:.4g} for P_FA <= 1/T; case uses {delta:g}"
    ))

    return findings


def report(case_name: str, findings: list[Finding], strict: bool = True) -> None:
    """Print findings; raise PreflightError if any is an ERROR and strict."""
    print(f"\nPreflight: {case_name}")
    for finding in findings:
        print(finding)
    errors = [f for f in findings if f.level == ERROR]
    if errors and strict:
        raise PreflightError(
            f"{case_name} cannot produce a meaningful result:\n"
            + "\n".join(f"  {f.check}: {f.message}" for f in errors)
            + "\nRe-run with --no-preflight to force it anyway."
        )
