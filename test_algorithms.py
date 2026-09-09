"""
test_algorithms.py -- multi-seed, multi-case benchmark for FastAdSwitch,
TS-GE, UCB1, EpsilonGreedy, and M-UCB.

All algorithms receive independent but identically configured instances of
SyntheticEnv for each trial: same K, arm means, bounded reward distribution,
shared reward upper bound, change schedule, and environment seed. This is
necessary because each algorithm takes different actions, but each should face
the same stochastic reward model and change regime.

Run:
    python test_algorithms.py

Artifacts written under artifacts/ for each enabled test case:
    <case>_summary.csv           mean/std/min/max final regret
    <case>_per_seed.csv          every final regret, plus first detection time
    <case>_multi_seed.png        mean cumulative-regret curves +/- 1 std
    <case>_single_seed.png       seed=42 detail view

Requirements:
    - synthetic_env.py must be the bounded scaled-Beta environment with
      SyntheticEnv(K, means, sigma, R_max, change_schedule, seed).
    - ts_ge.py must be the strict implementation requiring n_ge.
    - K for every enabled case must be a power of two because strict TS-GE
      follows the paper's K = 2^d assumption and intentionally does not pad
      with dummy arms.
"""
from __future__ import annotations

import csv
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from epsilon_greedy import EpsilonGreedy
from fast_adswitch import FastAdSwitch
from mucb import MUCB
from synthetic_env import SyntheticEnv, regret_from_history
from ts_ge import TS_GE
from ucb1 import UCB1

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Algorithm factories
# ---------------------------------------------------------------------------
def build_algorithms(
    *,
    K: int,
    T: int,
    sigma: float,
    ts_ge_delta: float,
    reward_R_max: float,
    ts_ge_n_ge: int,
    mucb_delta: float,
    mucb_M_estimate: int,
) -> dict[str, Any]:
    """Build algorithm factories for one case.

    `reward_R_max` is the shared physical reward upper bound: it is passed
    both to SyntheticEnv and to TS-GE. Keeping a single value prevents a
    silent mismatch between the environment's reward distribution and
    TS-GE's R/R_max Bernoulli conversion.
    """
    return {
        "FastAdSwitch": lambda seed: FastAdSwitch(K=K, T=T, C1=1.0, seed=seed),
        "TS-GE": lambda seed: TS_GE(
            K=K,
            T=T,
            delta=ts_ge_delta,
            R_max=reward_R_max,
            n_ge=ts_ge_n_ge,
            seed=seed,
        ),
        "UCB1": lambda seed: UCB1(K=K, T=T, sigma=sigma, seed=seed),
        "EpsilonGreedy": lambda seed: EpsilonGreedy(
            K=K,
            T=T,
            epsilon=0.1,
            decay=False,
            seed=seed,
        ),
        "M-UCB": lambda seed: MUCB(
            K=K,
            T=T,
            delta=mucb_delta,
            M_estimate=mucb_M_estimate,
            seed=seed,
        ),
    }


# ---------------------------------------------------------------------------
# One trial / many trials
# ---------------------------------------------------------------------------
def run_single_seed(
    *,
    K: int,
    T: int,
    means: list[float],
    sigma: float,
    reward_R_max: float,
    change_schedule: list[tuple[int, int, float]],
    algo_factories: dict[str, Any],
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Run every algorithm once against an identical configured environment.

    A separate environment instance is necessary per algorithm because they
    query different arms/groups. Passing the same seed makes their environment
    distributions and scheduled changes identical, while not incorrectly
    sharing a mutable RNG stream across unlike action sequences.
    """
    results: dict[str, dict[str, Any]] = {}

    for name, factory in algo_factories.items():
        env = SyntheticEnv(
            K=K,
            means=list(means),
            sigma=sigma,
            R_max=reward_R_max,
            change_schedule=list(change_schedule),
            seed=seed,
        )
        algorithm = factory(seed)
        history = algorithm.run(env.reward_fn)
        regret = regret_from_history(history, env.best_mean_history)

        if len(regret) != T:
            raise RuntimeError(
                f"{name} returned {len(regret)} reward/regret slots, expected exactly T={T}."
            )

        results[name] = {
            "history": history,
            "regret": np.asarray(regret, dtype=float),
            "detections": list(history.get("detections", [])),
        }

    return results


def run_multi_seed(
    *,
    K: int,
    T: int,
    means: list[float],
    sigma: float,
    reward_R_max: float,
    change_schedule: list[tuple[int, int, float]],
    algo_factories: dict[str, Any],
    seeds: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[list[int]]]]:
    """Return cumulative-regret curves, final regrets, and detection lists.

    curves[name] has shape (n_seeds, T).
    finals[name] has shape (n_seeds,).
    detections[name] has one detection-time list for every seed.
    """
    names = list(algo_factories)
    curves: dict[str, list[np.ndarray]] = {name: [] for name in names}
    finals: dict[str, list[float]] = {name: [] for name in names}
    detections: dict[str, list[list[int]]] = {name: [] for name in names}

    for index, seed in enumerate(seeds, start=1):
        print(f"  seed {index}/{len(seeds)} (seed={seed})...")
        trial = run_single_seed(
            K=K,
            T=T,
            means=means,
            sigma=sigma,
            reward_R_max=reward_R_max,
            change_schedule=change_schedule,
            algo_factories=algo_factories,
            seed=seed,
        )

        for name, result in trial.items():
            cumulative_regret = np.cumsum(result["regret"])
            curves[name].append(cumulative_regret)
            finals[name].append(float(cumulative_regret[-1]))
            detections[name].append(result["detections"])

    curve_arrays = {name: np.vstack(values) for name, values in curves.items()}
    final_arrays = {name: np.asarray(values, dtype=float) for name, values in finals.items()}
    return curve_arrays, final_arrays, detections


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------
def save_summary_csv(finals: dict[str, np.ndarray], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "algorithm",
            "n_seeds",
            "mean_final_regret",
            "std_final_regret",
            "min_final_regret",
            "max_final_regret",
        ])
        for name, values in finals.items():
            writer.writerow([
                name,
                len(values),
                f"{values.mean():.6f}",
                f"{values.std(ddof=0):.6f}",
                f"{values.min():.6f}",
                f"{values.max():.6f}",
            ])


def save_per_seed_csv(
    finals: dict[str, np.ndarray],
    detections: dict[str, list[list[int]]],
    seeds: list[int],
    path: str,
) -> None:
    """Save every final-regret observation, avoiding summary-only results."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "algorithm", "final_regret", "first_detection_time", "all_detection_times"])
        for algorithm, values in finals.items():
            for index, seed in enumerate(seeds):
                detected = detections[algorithm][index]
                writer.writerow([
                    seed,
                    algorithm,
                    f"{values[index]:.6f}",
                    detected[0] if detected else "",
                    ";".join(str(t) for t in detected),
                ])


def plot_multi_seed(
    curves: dict[str, np.ndarray],
    change_points: list[int],
    title: str,
    path: str,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))

    for name, curve_matrix in curves.items():
        mean_curve = curve_matrix.mean(axis=0)
        std_curve = curve_matrix.std(axis=0)
        x = np.arange(1, len(mean_curve) + 1)
        line, = axis.plot(x, mean_curve, label=f"{name} (n={curve_matrix.shape[0]})")
        axis.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=line.get_color(),
            alpha=0.15,
        )

    for index, change_point in enumerate(change_points):
        axis.axvline(
            change_point,
            color="red",
            linestyle="--",
            alpha=0.6,
            label="true change point(s)" if index == 0 else None,
        )

    axis.set_xlabel("Environment time slot")
    axis.set_ylabel("Cumulative system-reward regret")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_single_seed(
    results: dict[str, dict[str, Any]],
    change_points: list[int],
    title: str,
    path: str,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))

    for name, result in results.items():
        axis.plot(np.cumsum(result["regret"]), label=name)

    for index, change_point in enumerate(change_points):
        axis.axvline(
            change_point,
            color="red",
            linestyle="--",
            alpha=0.6,
            label="true change point(s)" if index == 0 else None,
        )

    axis.set_xlabel("Environment time slot")
    axis.set_ylabel("Cumulative system-reward regret")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------
def run_case(case: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[list[int]]]]:
    required_keys = {
        "name", "K", "T", "means", "sigma", "reward_R_max",
        "ts_ge_delta", "ts_ge_n_ge", "mucb_delta", "mucb_M_estimate",
        "change_schedule", "n_seeds",
    }
    missing = required_keys.difference(case)
    if missing:
        raise KeyError(f"Case {case.get('name', '<unnamed>')} is missing keys: {sorted(missing)}")

    name = case["name"]
    K = case["K"]
    T = case["T"]

    if K < 2 or (K & (K - 1)) != 0:
        raise ValueError(f"{name}: K={K} must be a power of two for strict TS-GE.")
    if len(case["means"]) != K:
        raise ValueError(f"{name}: means must have exactly K={K} values.")
    if not isinstance(case["ts_ge_n_ge"], int) or case["ts_ge_n_ge"] <= 0:
        raise ValueError(f"{name}: ts_ge_n_ge must be a positive integer.")

    print(f"\n{'=' * 72}\nCASE: {name}\n{'=' * 72}")
    print(f"K={K}  T={T}  n_seeds={case['n_seeds']}")
    print(
        f"reward_R_max={case['reward_R_max']}  sigma={case['sigma']}  "
        f"TS-GE(delta={case['ts_ge_delta']}, n_ge={case['ts_ge_n_ge']})  "
        f"M-UCB(delta={case['mucb_delta']}, M_estimate={case['mucb_M_estimate']})"
    )
    if case.get("note"):
        print(f"Note: {case['note']}")

    factories = build_algorithms(
        K=K,
        T=T,
        sigma=case["sigma"],
        ts_ge_delta=case["ts_ge_delta"],
        reward_R_max=case["reward_R_max"],
        ts_ge_n_ge=case["ts_ge_n_ge"],
        mucb_delta=case["mucb_delta"],
        mucb_M_estimate=case["mucb_M_estimate"],
    )

    seeds = list(range(1, case["n_seeds"] + 1))
    curves, finals, detections = run_multi_seed(
        K=K,
        T=T,
        means=list(case["means"]),
        sigma=case["sigma"],
        reward_R_max=case["reward_R_max"],
        change_schedule=list(case["change_schedule"]),
        algo_factories=factories,
        seeds=seeds,
    )

    print(f"\n{'Algorithm':<15} | {'Mean Regret':>14} | {'Std':>12} | {'Min':>14} | {'Max':>14}")
    for algorithm, values in finals.items():
        print(
            f"{algorithm:<15} | {values.mean():14.2f} | {values.std(ddof=0):12.2f} | "
            f"{values.min():14.2f} | {values.max():14.2f}"
        )

    change_points = [change[0] for change in case["change_schedule"]]

    summary_path = os.path.join(ARTIFACT_DIR, f"{name}_summary.csv")
    save_summary_csv(finals, summary_path)
    print(f"\nSaved {summary_path}")

    per_seed_path = os.path.join(ARTIFACT_DIR, f"{name}_per_seed.csv")
    save_per_seed_csv(finals, detections, seeds, per_seed_path)
    print(f"Saved {per_seed_path}")

    multi_seed_plot_path = os.path.join(ARTIFACT_DIR, f"{name}_multi_seed.png")
    plot_multi_seed(
        curves,
        change_points,
        f"{name}: mean cumulative regret +/- 1 std across {len(seeds)} seeds",
        multi_seed_plot_path,
    )
    print(f"Saved {multi_seed_plot_path}")

    first_change_point = change_points[0] if change_points else None
    print("\nDetection latency (first detection minus first true change point):")
    for algorithm, detection_lists in detections.items():
        first_detection_times = [times[0] for times in detection_lists if times]
        if first_change_point is None:
            print(f"  {algorithm:<15}: no scheduled change point in this case")
        elif first_detection_times:
            latencies = [time - first_change_point for time in first_detection_times]
            print(
                f"  {algorithm:<15}: mean={np.mean(latencies):.1f}, "
                f"std={np.std(latencies, ddof=0):.1f}, "
                f"min={min(latencies)}, max={max(latencies)} "
                f"({len(latencies)}/{len(detection_lists)} seeds detected)"
            )
        else:
            print(f"  {algorithm:<15}: no explicit detections recorded")

    single = run_single_seed(
        K=K,
        T=T,
        means=list(case["means"]),
        sigma=case["sigma"],
        reward_R_max=case["reward_R_max"],
        change_schedule=list(case["change_schedule"]),
        algo_factories=factories,
        seed=42,
    )
    single_plot_path = os.path.join(ARTIFACT_DIR, f"{name}_single_seed.png")
    plot_single_seed(
        single,
        change_points,
        f"{name}: single seed=42 detail view",
        single_plot_path,
    )
    print(f"Saved {single_plot_path}")

    # TS-GE diagnostics are particularly useful while validating the paper.
    ts_ge_history = single["TS-GE"]["history"]
    events = ts_ge_history.get("localization_events", [])
    print("\nTS-GE diagnostics for single seed=42:")
    print(f"  n_etc={ts_ge_history.get('n_etc')}  n_ge={ts_ge_history.get('n_ge')}  "
          f"episode_length={ts_ge_history.get('episode_length')}  "
          f"TS_slots={ts_ge_history.get('ts_length')}  BP_slots={ts_ge_history.get('bp_length')}")
    print(f"  BP detections: {ts_ge_history.get('detections', [])}")
    for event in events:
        print(f"  localization event: {event}")

    return curves, finals, detections


if __name__ == "__main__":
    # Add another dict to this list to add a case. All case parameters are
    # explicit so that CSV artifacts can be interpreted without guessing.
    cases: list[dict[str, Any]] = [
        {
            "name": "case1_K2_baseline",
            "K": 2,
            "T": 6000,
            "means": [2.0, 6.0],
            "sigma": 0.5,

            # Shared physical reward bound: used by SyntheticEnv AND TS-GE.
            "reward_R_max": 20.0,

            # TS-GE-specific parameters.
            "ts_ge_delta": 0.1,
            "ts_ge_n_ge": 50,

            # M-UCB-specific parameters.
            "mucb_delta": 2.0,
            "mucb_M_estimate": 2,

            # At environment t=3000, arm 0 mean changes from 2.0 to 16.0.
            "change_schedule": [(3000, 0, 16.0)],
            "n_seeds": 20,
            "note": (
                "Strict bounded-reward Case 1. Every physical reward is a "
                "scaled-Beta sample in [0, 20]; arm 0 changes from mean 2 to 16."
            ),
        },
    ]

    all_results: dict[str, Any] = {}
    for benchmark_case in cases:
        curves, finals, detections = run_case(benchmark_case)
        all_results[benchmark_case["name"]] = {
            "curves": curves,
            "finals": finals,
            "detections": detections,
        }

    print(f"\n{'=' * 72}\nAll enabled cases complete. Artifacts saved under {ARTIFACT_DIR}/\n{'=' * 72}")
