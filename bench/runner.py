"""Run one benchmark case and write its artifacts.

Each algorithm gets its own SyntheticEnv instance built from the same K, means,
sigma, R_max, change schedule and seed. Separate instances are necessary
because the algorithms query different arms and would otherwise consume each
other's RNG stream; the shared seed keeps their reward distributions and change
times identical.

One sweep per case. The old harness ran every seed and then re-ran a full
sweep at a hardcoded seed=42 just to draw the detail plot -- a whole extra pass
over every algorithm (328 s per pass at K=64, T=60,000) whose seed was not even
in the swept range. The detail seed is now drawn from the swept seeds and its
histories are simply kept.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synthetic_env import SyntheticEnv

from .cases import build_algorithms
from .metrics import (
    RunMetrics,
    dispersion,
    downsample_cumulative,
    downsample_x,
    format_probe_age,
    summarize_run,
)

CURVE_POINTS = 2000


@dataclass
class CaseResult:
    case: dict[str, Any]
    seeds: list[int]
    algorithms: list[str]
    x: np.ndarray
    total_curves: dict[str, np.ndarray] = field(default_factory=dict)
    decision_curves: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, list[RunMetrics]] = field(default_factory=dict)
    detail_seed: int = 0
    detail_histories: dict[str, dict[str, Any]] = field(default_factory=dict)


def _make_env(case: dict[str, Any], seed: int) -> SyntheticEnv:
    return SyntheticEnv(
        K=case["K"],
        means=list(case["means"]),
        sigma=case["sigma"],
        R_max=case["reward_R_max"],
        change_schedule=list(case["change_schedule"]),
        seed=seed,
    )


@dataclass
class TrialRun:
    regret: np.ndarray
    decision_regret: np.ndarray
    metrics: RunMetrics
    history: dict[str, Any] | None


def run_trial(
    case: dict[str, Any],
    algorithms: list[str],
    seed: int,
    keep_histories: bool = False,
) -> dict[str, TrialRun]:
    """Run every algorithm once on identically configured environments."""
    factories = build_algorithms(case, algorithms)
    results: dict[str, TrialRun] = {}

    for name, factory in factories.items():
        env = _make_env(case, seed)
        started = time.time()
        history = factory(seed).run(env.reward_fn)
        elapsed = time.time() - started

        regret, is_decision, metrics = summarize_run(
            history, env.best_mean_history, case["K"], elapsed
        )
        if len(regret) != case["T"]:
            raise RuntimeError(
                f"{name} returned {len(regret)} reward slots, expected exactly T={case['T']}."
            )
        # Probe slots contribute zero to the decision curve, so it stays flat
        # across a BP burst instead of being charged for it.
        results[name] = TrialRun(
            regret=regret,
            decision_regret=np.where(is_decision, regret, 0.0),
            metrics=metrics,
            history=history if keep_histories else None,
        )

    return results


def run_case(
    case: dict[str, Any],
    seeds: list[int],
    algorithms: list[str],
    detail_seed: int | None = None,
    curve_points: int = CURVE_POINTS,
    verbose: bool = True,
) -> CaseResult:
    detail_seed = seeds[0] if detail_seed is None else detail_seed
    if detail_seed not in seeds:
        raise ValueError(f"detail_seed={detail_seed} must be one of the swept seeds {seeds}.")

    result = CaseResult(
        case=case,
        seeds=list(seeds),
        algorithms=list(algorithms),
        x=downsample_x(case["T"], curve_points),
        detail_seed=detail_seed,
    )
    total: dict[str, list[np.ndarray]] = {name: [] for name in algorithms}
    decision: dict[str, list[np.ndarray]] = {name: [] for name in algorithms}
    result.metrics = {name: [] for name in algorithms}

    for index, seed in enumerate(seeds, start=1):
        if verbose:
            print(f"  seed {index}/{len(seeds)} (seed={seed})...", flush=True)
        trial = run_trial(case, algorithms, seed, keep_histories=(seed == detail_seed))

        for name, run in trial.items():
            total[name].append(downsample_cumulative(run.regret, curve_points))
            decision[name].append(downsample_cumulative(run.decision_regret, curve_points))
            result.metrics[name].append(run.metrics)
            if run.history is not None:
                result.detail_histories[name] = run.history

    result.total_curves = {name: np.vstack(values) for name, values in total.items()}
    result.decision_curves = {name: np.vstack(values) for name, values in decision.items()}
    return result


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------
def _first_change(result: CaseResult) -> int:
    schedule = result.case["change_schedule"]
    return schedule[0][0] if schedule else 0


def _reason_histogram(result: CaseResult, runs: list[RunMetrics]) -> str:
    """e.g. 'bad_arm_change x10' -- empty for algorithms that do not tag."""
    first_change = _first_change(result)
    counts: dict[str, int] = {}
    for metrics in runs:
        reason = metrics.first_detection_reason(first_change)
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return ";".join(f"{k} x{v}" for k, v in sorted(counts.items()))


def save_summary_csv(result: CaseResult, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "algorithm", "n_seeds",
            "mean_total_regret", "std_total_regret",
            "median_total_regret", "p25_total_regret", "p75_total_regret",
            "min_total_regret", "max_total_regret",
            "mean_decision_regret", "mean_probe_regret",
            "max_probe_age", "mean_detections", "detection_reasons",
            "mean_wall_time_s",
        ])
        for name, runs in result.metrics.items():
            d = dispersion([m.total_regret for m in runs])
            writer.writerow([
                name, len(runs),
                f"{d['mean']:.6f}", f"{d['std']:.6f}",
                f"{d['median']:.6f}", f"{d['p25']:.6f}", f"{d['p75']:.6f}",
                f"{d['min']:.6f}", f"{d['max']:.6f}",
                f"{np.mean([m.decision_regret for m in runs]):.6f}",
                f"{np.mean([m.probe_regret for m in runs]):.6f}",
                format_probe_age(max(m.max_probe_age for m in runs)),
                f"{np.mean([len(m.detections) for m in runs]):.2f}",
                _reason_histogram(result, runs),
                f"{np.mean([m.wall_time_s for m in runs]):.3f}",
            ])


def save_per_seed_csv(result: CaseResult, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "seed", "algorithm", "total_regret", "decision_regret", "probe_regret",
            "decision_slots", "probe_slots", "max_probe_age", "wall_time_s",
            "first_detection_time", "first_detection_reason", "all_detection_times",
        ])
        first_change = _first_change(result)
        for name, runs in result.metrics.items():
            for seed, metrics in zip(result.seeds, runs):
                post = [t for t in metrics.detections if t >= first_change]
                writer.writerow([
                    seed, name,
                    f"{metrics.total_regret:.6f}",
                    f"{metrics.decision_regret:.6f}",
                    f"{metrics.probe_regret:.6f}",
                    metrics.decision_slots, metrics.probe_slots,
                    format_probe_age(metrics.max_probe_age), f"{metrics.wall_time_s:.3f}",
                    post[0] if post else "",
                    metrics.first_detection_reason(first_change),
                    ";".join(str(t) for t in metrics.detections),
                ])


def save_phase_csv(result: CaseResult, path: str) -> None:
    """Per-phase regret for the detail seed, for algorithms reporting phases."""
    rows = []
    for name, history in result.detail_histories.items():
        index = result.seeds.index(result.detail_seed)
        for phase, stats in result.metrics[name][index].phase_regret.items():
            rows.append([
                name, phase, stats["slots"], f"{stats['slot_share']:.2f}",
                f"{stats['regret']:.4f}", f"{stats['regret_share']:.2f}",
                f"{stats['regret_per_slot']:.4f}",
            ])
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "algorithm", "phase", "slots", "slot_share_pct",
            "regret", "regret_share_pct", "regret_per_slot",
        ])
        writer.writerows(rows)


def plot_case(result: CaseResult, path: str) -> None:
    """Two stacked panels: total regret, then decision-only regret.

    The split is the whole point. TS-GE's total is dominated by mandatory
    broadcast probing, which the non-probing baselines never pay; the lower
    panel is the like-for-like comparison of arm-choice quality.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    change_points = [change[0] for change in result.case["change_schedule"]]

    for axis, curves, title in (
        (top, result.total_curves, "Total regret (includes mandatory group probes)"),
        (bottom, result.decision_curves, "Decision regret (single-arm slots only)"),
    ):
        for name, matrix in curves.items():
            mean_curve = matrix.mean(axis=0)
            std_curve = matrix.std(axis=0)
            line, = axis.plot(result.x, mean_curve, label=f"{name} (n={matrix.shape[0]})")
            axis.fill_between(
                result.x, mean_curve - std_curve, mean_curve + std_curve,
                color=line.get_color(), alpha=0.15,
            )
        for index, change_point in enumerate(change_points):
            axis.axvline(
                change_point, color="red", linestyle="--", alpha=0.6,
                label="true change point(s)" if index == 0 else None,
            )
        axis.set_ylabel("Cumulative regret")
        axis.set_title(title)
        axis.legend(fontsize=8)

    bottom.set_xlabel("Environment time slot")
    figure.suptitle(
        f"{result.case['name']}: K={result.case['K']}, T={result.case['T']}, "
        f"{len(result.seeds)} seed(s), mean +/- 1 std"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def print_summary(result: CaseResult) -> None:
    header = (
        f"{'Algorithm':<15} | {'Total regret':>14} | {'Std':>12} | {'Median':>13} | "
        f"{'Decision':>13} | {'Probe':>13} | {'MaxProbeAge':>11} | {'Time/seed':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name, runs in result.metrics.items():
        d = dispersion([m.total_regret for m in runs])
        print(
            f"{name:<15} | {d['mean']:14.2f} | {d['std']:12.2f} | {d['median']:13.2f} | "
            f"{np.mean([m.decision_regret for m in runs]):13.2f} | "
            f"{np.mean([m.probe_regret for m in runs]):13.2f} | "
            f"{format_probe_age(max(m.max_probe_age for m in runs)):>11s} | "
            f"{np.mean([m.wall_time_s for m in runs]):8.2f}s"
        )
        if d["std"] > d["mean"] * 0.5 and len(runs) > 2:
            # Loud on purpose: at this spread the mean describes no run that
            # actually happened, and reading the table as a ranking is wrong.
            print(
                f"{'':<15}   ^ spread: p25={d['p25']:.0f} p75={d['p75']:.0f} "
                f"min={d['min']:.0f} max={d['max']:.0f} "
                f"(max/min = {d['max'] / max(d['min'], 1e-9):.0f}x) -- mean is not a summary here"
            )

    first_change = _first_change(result)
    if first_change:
        print("\nDetection latency (first detection at or after the first true change point):")
        for name, runs in result.metrics.items():
            latencies, reasons = [], {}
            for metrics in runs:
                post = [t for t in metrics.detections if t >= first_change]
                if post:
                    latencies.append(post[0] - first_change)
                reason = metrics.first_detection_reason(first_change)
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
            if latencies:
                tag = ("  via " + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items()))) if reasons else ""
                print(
                    f"  {name:<15}: mean={np.mean(latencies):9.1f}  "
                    f"min={min(latencies):7d}  max={max(latencies):7d}  "
                    f"({len(latencies)}/{len(runs)} seeds detected){tag}"
                )
            else:
                print(f"  {name:<15}: no explicit detections recorded")


def print_phase_table(result: CaseResult, name: str = "TS-GE") -> None:
    if name not in result.metrics:
        return
    index = result.seeds.index(result.detail_seed)
    table = result.metrics[name][index].phase_regret
    if not table:
        return
    print(f"\n{name} regret by phase (seed={result.detail_seed}):")
    print(f"  {'phase':<10} {'slots':>9} {'slot%':>7} {'regret':>14} {'regret%':>8} {'per slot':>10}")
    for phase, stats in table.items():
        print(
            f"  {phase:<10} {stats['slots']:9d} {stats['slot_share']:6.1f}% "
            f"{stats['regret']:14.1f} {stats['regret_share']:7.1f}% {stats['regret_per_slot']:10.3f}"
        )

    history = result.detail_histories.get(name)
    if history is None:
        return
    print(
        f"  n_etc={history.get('n_etc')} n_ge={history.get('n_ge')} "
        f"episode={history.get('episode_length')} TS={history.get('ts_length')} "
        f"BP={history.get('bp_length')} noise_scale={history.get('noise_scale')}"
    )
    print(f"  BP detections: {history.get('detections', [])}")
    reasons: dict[str, int] = {}
    for event in history.get("localization_events", []):
        reasons[event["reason"]] = reasons.get(event["reason"], 0) + 1
    if reasons:
        print(f"  localization outcomes: {reasons}")


def write_artifacts(result: CaseResult, out_dir: str, plots: bool = True) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    name = result.case["name"]
    written = []

    summary_path = os.path.join(out_dir, f"{name}_summary.csv")
    save_summary_csv(result, summary_path)
    written.append(summary_path)

    per_seed_path = os.path.join(out_dir, f"{name}_per_seed.csv")
    save_per_seed_csv(result, per_seed_path)
    written.append(per_seed_path)

    phase_path = os.path.join(out_dir, f"{name}_phase_regret.csv")
    save_phase_csv(result, phase_path)
    if os.path.exists(phase_path):
        written.append(phase_path)

    if plots:
        plot_path = os.path.join(out_dir, f"{name}_regret.png")
        plot_case(result, plot_path)
        written.append(plot_path)

    return written
