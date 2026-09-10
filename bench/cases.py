"""Benchmark case definitions and algorithm factories.

Every case is a plain dict so the CSV artifacts can be interpreted without
guessing what was configured. `reward_R_max` is shared by SyntheticEnv and
TS-GE, so the environment's reward bound and TS-GE's R/R_max Bernoulli
conversion can never drift apart.

CHOOSING delta AND T
====================
The cases below are not hand-tuned; each one was derived from the two rules in
bench.preflight and then confirmed by running preflight on it:

    delta <= Delta_C / (4 * K)                      (BP detectability)
    K^3   <= rho * T * (Delta_C/noise_scale)^2 / (8 ln T)   (ETC budget)

That is why case3 and case4 move the changed arm to 19.0 rather than 16.0: at
K=64 and K=128 a Delta_C of 14 forces delta so small that ETC eats 31% and
246% of the horizon respectively, while Delta_C=17 keeps both cases near 20%.

`ts_ge_noise_scale` is the reward-scale factor in n_ETC. case1 keeps the
paper's scale-free 1.0 so its numbers stay comparable with the pre-existing
artifacts; every other case passes the environment's sigma, which is the
correct sub-Gaussian constant and is what makes K >= 64 feasible at all.
"""
from __future__ import annotations

from typing import Any, Callable

from epsilon_greedy import EpsilonGreedy
from fast_adswitch import FastAdSwitch
from mucb import MUCB
from ts_ge import TS_GE
from ucb1 import UCB1

ALL_ALGORITHMS = ("FastAdSwitch", "TS-GE", "UCB1", "EpsilonGreedy", "M-UCB")


def build_algorithms(case: dict[str, Any], names: list[str]) -> dict[str, Callable[[int], Any]]:
    """Return {name: seed -> algorithm} for the requested algorithm names."""
    K, T = case["K"], case["T"]
    factories: dict[str, Callable[[int], Any]] = {
        "FastAdSwitch": lambda seed: FastAdSwitch(K=K, T=T, C1=1.0, seed=seed),
        "TS-GE": lambda seed: TS_GE(
            K=K,
            T=T,
            delta=case["ts_ge_delta"],
            R_max=case["reward_R_max"],
            n_ge=case["ts_ge_n_ge"],
            noise_scale=case.get("ts_ge_noise_scale", 1.0),
            bp_length_policy=case.get("ts_ge_bp_length_policy", "paper"),
            etc_policy=case.get("ts_ge_etc_policy", "paper"),
            # Per-slot diagnostic buffers are megabytes at T=1e6 and are only
            # read for the detail seed, so the sweep turns them off.
            diagnostics=case.get("ts_ge_diagnostics", True),
            seed=seed,
        ),
        "UCB1": lambda seed: UCB1(K=K, T=T, sigma=case["sigma"], seed=seed),
        "EpsilonGreedy": lambda seed: EpsilonGreedy(K=K, T=T, epsilon=0.1, decay=False, seed=seed),
        "M-UCB": lambda seed: MUCB(
            K=K, T=T, delta=case["mucb_delta"], M_estimate=case["mucb_M_estimate"], seed=seed
        ),
    }
    unknown = set(names) - set(factories)
    if unknown:
        raise KeyError(f"Unknown algorithm(s): {sorted(unknown)}. Known: {sorted(factories)}")
    return {name: factories[name] for name in names}


def _ramp(K: int, low: float = 2.0, high: float = 5.0) -> list[float]:
    """K arm means evenly spaced in [low, high]."""
    if K == 1:
        return [low]
    return [low + (high - low) * i / (K - 1) for i in range(K)]


CASES: dict[str, dict[str, Any]] = {
    "case1_K2_baseline": {
        "name": "case1_K2_baseline",
        "K": 2,
        "T": 6000,
        "means": [2.0, 6.0],
        "sigma": 0.5,
        "reward_R_max": 20.0,
        "ts_ge_delta": 0.1,
        "ts_ge_n_ge": 50,
        "ts_ge_noise_scale": 1.0,
        "mucb_delta": 2.0,
        "mucb_M_estimate": 2,
        "change_schedule": [(3000, 0, 16.0)],
        "default_seeds": 20,
        "algorithms": list(ALL_ALGORITHMS),
        "note": (
            "Regression reference. Kept bit-identical to the pre-refactor case so "
            "artifacts/case1_K2_baseline_summary.csv stays comparable. Note this is the "
            "regime the paper's Section IV expects M-UCB to win: small K, small T."
        ),
    },
    "case2_K16_midsize": {
        "name": "case2_K16_midsize",
        "K": 16,
        "T": 100_000,
        "means": _ramp(16),
        "sigma": 0.5,
        "reward_R_max": 20.0,
        "ts_ge_delta": 0.2,          # <= Delta_C/(4K) = 14/64 = 0.21875
        "ts_ge_n_ge": 50,
        "ts_ge_noise_scale": 0.5,    # = sigma
        "mucb_delta": 2.0,
        "mucb_M_estimate": 2,
        "change_schedule": [(50_000, 0, 16.0)],
        "default_seeds": 10,
        "algorithms": list(ALL_ALGORITHMS),
        "note": (
            "Mid-size K, all five algorithms. Set ts_ge_bp_length_policy='power' to see "
            "TS-GE's total drop from ~219,000 to ~20,000 (T_BP 100 -> 3)."
        ),
    },
    "case3_K64_large_K": {
        "name": "case3_K64_large_K",
        "K": 64,
        "T": 100_000,
        "means": _ramp(64),
        "sigma": 0.5,
        "reward_R_max": 20.0,
        "ts_ge_delta": 0.065,        # <= Delta_C/(4K) = 17/256 = 0.06640
        "ts_ge_n_ge": 50,
        "ts_ge_noise_scale": 0.5,
        "mucb_delta": 2.0,
        "mucb_M_estimate": 2,
        "change_schedule": [(50_000, 0, 19.0)],
        "default_seeds": 5,
        "algorithms": list(ALL_ALGORITHMS),
        "note": (
            "Infeasible before the scale-aware n_ETC fix: with noise_scale=1.0 the ETC "
            "phase needs 123% of the horizon and the run registers zero detections. "
            "With both phase-length policies switched on, TS-GE's total goes from "
            "306,486 to about 49,700 -- past M-UCB and past AdSwitch's median."
        ),
    },
    "case4_K128_paper_regime": {
        "name": "case4_K128_paper_regime",
        "K": 128,
        "T": 1_000_000,
        "means": _ramp(128),
        "sigma": 0.5,
        "reward_R_max": 20.0,
        "ts_ge_delta": 0.033,        # <= Delta_C/(4K) = 17/512 = 0.033203
        "ts_ge_n_ge": 50,
        "ts_ge_noise_scale": 0.5,
        "ts_ge_diagnostics": False,
        "mucb_delta": 2.0,
        "mucb_M_estimate": 2,
        "change_schedule": [(500_000, 0, 19.0)],
        "default_seeds": 3,
        "heavy": True,
        "algorithms": list(ALL_ALGORITHMS),
        "note": (
            "The large-K, T=1e5+ regime Section IV claims TS-GE wins. It is infeasible at "
            "T=1e5 under ANY delta (ETC >= 246% of the horizon), which is what the "
            "feasible_K preflight check reports. FastAdSwitch runs here now -- 193 s and "
            "272 MB per seed, against 2,048 MB for its old dense arrays alone -- but it "
            "is still ~9x the cost of the other four combined, hence 3 seeds by default."
        ),
    },
}


CASES["case5_best_arm_collapse"] = {
    "name": "case5_best_arm_collapse",
    "K": 16,
    "T": 100_000,
    "means": _ramp(16),
    "sigma": 0.5,
    "reward_R_max": 20.0,
    # The incumbent BEST arm degrades instead of the worst arm being promoted.
    # Delta_C is only 2.5 here, so detectability forces a much smaller delta:
    # 2.5 / (4*16) = 0.0391.
    "ts_ge_delta": 0.039,
    "ts_ge_n_ge": 50,
    "ts_ge_noise_scale": 0.25,
    "mucb_delta": 1.0,
    "mucb_M_estimate": 2,
    "change_schedule": [(50_000, 15, 2.5)],
    "default_seeds": 10,
    "algorithms": list(ALL_ALGORITHMS),
    "note": (
        "The direction cases 1-4 never test. They all promote arm 0 (the worst arm) to "
        "best, which is precisely AdSwitch's worst path: arm 0 has already been evicted, "
        "so only condition (4)'s sparse random recheck can notice, and every post-change "
        "restart in cases 2 and 3 was tagged bad_arm_change. Here the change hits a "
        "currently-good arm, condition (3) sees it immediately, and AdSwitch's detection "
        "latency drops from ~4,960 to 4 rounds with its across-seed std falling from "
        "39,029 to 863. Without this case the suite reports only AdSwitch's bad half."
    ),
}


def get_case(name: str) -> dict[str, Any]:
    if name not in CASES:
        raise KeyError(f"Unknown case {name!r}. Known: {sorted(CASES)}")
    return dict(CASES[name])
