"""Regret decomposition and probe-age metrics for a single algorithm run.

WHY THIS EXISTS
===============
`regret_from_history` charges every slot the gap between the environment's best
mean and the reward actually received. For a TS-GE broadcast-probing slot the
reward is the average over all K arms, so that slot is charged roughly
(mu* - mean_i mu_i) no matter how well the algorithm is learning. Measured on
K=2, T=200000, that single effect is 98.5% of TS-GE's total regret.

Comparing that total against UCB1 or M-UCB -- which never probe, and are
therefore never charged -- says nothing about decision quality. So every run is
reported three ways:

    total_regret    = decision_regret + probe_regret     (what the paper's Eq. 2 charges)
    decision_regret = regret on slots where the algorithm chose ONE arm
    probe_regret    = regret on group-probe slots (BP/GE); 0 for non-probing algorithms

and with the metric TS-GE actually exists to deliver:

    max_probe_age   = the longest any arm went unprobed, individually or in a
                      group. This is Condition 1 in the paper, which bounds it
                      by sqrt(T). No competitor here offers such a bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synthetic_env import regret_from_history

# Sentinel codes for non-integer actions in the per-slot action array. Real arm
# indices are >= 0, so any negative code is a group probe.
_GROUP_CODE_BASE = -1


@dataclass
class RunMetrics:
    total_regret: float
    decision_regret: float
    probe_regret: float
    decision_slots: int
    probe_slots: int
    max_probe_age: int
    detections: list[int]
    # AdSwitch tags each restart with the condition that fired. Condition (3)
    # rechecks every good arm every round, so a change to the incumbent best
    # arm is caught almost immediately; condition (4) only revisits an evicted
    # arm when its sparse random obligation schedule fires, so a change to an
    # already-bad arm is caught after a highly variable delay. Which one fired
    # is the whole explanation for AdSwitch's across-seed variance, so it is
    # carried through rather than discarded.
    detection_reasons: list[tuple[int, str]] = field(default_factory=list)
    phase_regret: dict[str, dict[str, float]] = field(default_factory=dict)
    wall_time_s: float = 0.0

    def first_detection_reason(self, at_or_after: int = 0) -> str:
        """Reason tag of the first detection at or after `at_or_after`."""
        for time, reason in self.detection_reasons:
            if time >= at_or_after:
                return reason
        return ""

    def as_row(self) -> dict[str, Any]:
        return {
            "total_regret": self.total_regret,
            "decision_regret": self.decision_regret,
            "probe_regret": self.probe_regret,
            "decision_slots": self.decision_slots,
            "probe_slots": self.probe_slots,
            "max_probe_age": self.max_probe_age,
            "n_detections": len(self.detections),
            "first_detection": self.detections[0] if self.detections else "",
            "first_detection_reason": self.first_detection_reason(),
            "wall_time_s": self.wall_time_s,
        }


def _action_codes(chosen: list[Any]) -> tuple[np.ndarray, dict[str, int]]:
    """Map the per-slot action list to an int array.

    Integer arm indices map to themselves; every distinct string tag ('BP',
    'GE:0', ...) maps to a distinct negative code. Doing this once lets every
    downstream metric be pure numpy.
    """
    tag_codes: dict[str, int] = {}

    def code(action: Any) -> int:
        if isinstance(action, (int, np.integer)):
            return int(action)
        if action not in tag_codes:
            tag_codes[action] = _GROUP_CODE_BASE - len(tag_codes)
        return tag_codes[action]

    codes = np.fromiter((code(a) for a in chosen), dtype=np.int64, count=len(chosen))
    return codes, tag_codes


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Collapse a sorted index array into inclusive contiguous [start, end] runs.

    Group probes arrive in long bursts (a whole BP phase, a whole GE group), so
    collapsing them keeps `max_probe_age` linear in the number of phases rather
    than in the number of slots.
    """
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


def _max_gap(intervals: list[tuple[int, int]], window_start: int, horizon: int) -> int:
    """Longest stretch of [window_start, horizon] with no probe.

    Intervals are 1-indexed inclusive slot ranges in which the arm was probed.
    The result is the age the arm reached: the number of slots between one
    probe and the next. An arm never probed in the window returns the full
    window length.
    """
    window = horizon - window_start + 1
    clipped = [(max(s, window_start), e) for s, e in intervals if e >= window_start]
    if not clipped:
        return window
    clipped.sort()
    worst = clipped[0][0] - window_start  # gap from the window start to the first probe
    covered_to = clipped[0][1]
    for start, end in clipped[1:]:
        if start > covered_to:
            worst = max(worst, start - covered_to)
        covered_to = max(covered_to, end)
    return max(worst, horizon + 1 - covered_to)


def _steady_state_start(history: dict[str, Any]) -> int:
    """First 1-indexed slot after the leading initialization block.

    Condition 1 in the paper ("each arm probed at least once per episode")
    governs the alternating TS/BP episodes, not the one-off ETC warm-up, where
    TS-GE deliberately pulls arm 0 n_ETC times before it ever touches arm 1.
    Including ETC would make the metric report the warm-up length rather than
    the guarantee, so the leading ETC block is skipped. Algorithms that report
    no `phase` list are unaffected.
    """
    phases = history.get("phase")
    if not phases:
        return 1
    index = 0
    while index < len(phases) and phases[index] == "ETC":
        index += 1
    return index + 1


def max_probe_age(history: dict[str, Any], K: int) -> int:
    """Condition 1: the longest any arm went without being probed.

    An arm counts as probed when it is pulled individually OR when it is part
    of a group probe (BP probes all arms; GE:k probes the arms of super-arm k).
    Measured after the leading initialization block -- see `_steady_state_start`.
    """
    chosen = history["chosen_arm"]
    horizon = len(chosen)
    if horizon == 0:
        return 0
    window_start = _steady_state_start(history)
    if window_start > horizon:
        return 0
    codes, tag_codes = _action_codes(chosen)
    groups = history.get("groups") or []

    # Which arms does each group tag cover?
    tag_arms: dict[int, list[int]] = {}
    for tag, code in tag_codes.items():
        if tag == "BP":
            tag_arms[code] = list(range(K))
        elif tag.startswith("GE:"):
            bit = int(tag.split(":", 1)[1])
            tag_arms[code] = list(groups[bit]) if bit < len(groups) else []
        else:
            tag_arms[code] = []

    # Runs are computed once per tag and reused across every arm they cover.
    # Collapsing them matters: a BP phase is one interval, not T^(2/5) of them.
    tag_runs = {
        code: _runs(np.flatnonzero(codes == code) + 1) for code in tag_arms
    }

    # Individual-pull times, bucketed by arm in one sort rather than K scans
    # over the whole action array (which is 1.3e8 comparisons at K=128, T=1e6).
    individual = np.flatnonzero(codes >= 0)
    order = individual[np.argsort(codes[individual], kind="stable")]
    bounds = np.searchsorted(codes[order], np.arange(K + 1))

    worst = 0
    for arm in range(K):
        times = order[bounds[arm]:bounds[arm + 1]] + 1
        intervals = [(t, t) for t in times.tolist()]
        for code, arms in tag_arms.items():
            if arm in arms:
                intervals.extend(tag_runs[code])
        worst = max(worst, _max_gap(intervals, window_start, horizon))
    return worst


def phase_regret(history: dict[str, Any], regret: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-phase regret table, for algorithms that report a `phase` list.

    'GE:0', 'GE:1', ... are collapsed into a single 'GE' bucket.
    """
    phases = history.get("phase")
    if not phases:
        return {}
    labels = np.array([p.split(":", 1)[0] for p in phases])
    table: dict[str, dict[str, float]] = {}
    total = float(regret.sum())
    for label in ["ETC", "TS", "BP", "GE", "RECOVERY"]:
        mask = labels == label
        count = int(mask.sum())
        if not count:
            continue
        bucket = float(regret[mask].sum())
        table[label] = {
            "slots": count,
            "slot_share": 100.0 * count / len(regret),
            "regret": bucket,
            "regret_share": 100.0 * bucket / total if total else 0.0,
            "regret_per_slot": bucket / count,
        }
    return table


def summarize_run(
    history: dict[str, Any],
    best_mean_history: dict[int, float],
    K: int,
    wall_time_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, RunMetrics]:
    """Return (per-slot regret, decision mask, RunMetrics) for one run.

    The mask is returned rather than recomputed by the caller because building
    it means one pass over the T-element action list; at T=1e6 that is the most
    expensive part of scoring a run.
    """
    regret = np.asarray(regret_from_history(history, best_mean_history), dtype=float)
    codes, _ = _action_codes(history["chosen_arm"])
    is_decision = codes >= 0

    return regret, is_decision, RunMetrics(
        total_regret=float(regret.sum()),
        decision_regret=float(regret[is_decision].sum()),
        probe_regret=float(regret[~is_decision].sum()),
        decision_slots=int(is_decision.sum()),
        probe_slots=int((~is_decision).sum()),
        max_probe_age=max_probe_age(history, K),
        detections=list(history.get("detections", [])),
        detection_reasons=list(history.get("detection_reasons", [])),
        phase_regret=phase_regret(history, regret),
        wall_time_s=wall_time_s,
    )


def downsample_cumulative(regret: np.ndarray, points: int = 2000) -> np.ndarray:
    """Cumulative regret curve reduced to at most `points` samples.

    Storing a full (n_seeds, T) curve matrix costs 80 MB at 20 seeds x T=1e5 and
    800 MB at T=1e6, and plotting T points per curve is wasted work at any
    figure size. Keeping the block *end* values preserves the curve exactly at
    the sampled abscissae, so the plotted mean and std are unbiased.
    """
    cumulative = np.cumsum(regret)
    if cumulative.size <= points:
        return cumulative
    edges = np.linspace(0, cumulative.size - 1, points, dtype=np.int64)
    return cumulative[edges]


def downsample_x(horizon: int, points: int = 2000) -> np.ndarray:
    """Slot indices matching `downsample_cumulative` for the same horizon."""
    if horizon <= points:
        return np.arange(1, horizon + 1)
    return np.linspace(0, horizon - 1, points, dtype=np.int64) + 1


def dispersion(values: np.ndarray) -> dict[str, float]:
    """Mean/std plus median and quartiles.

    Mean and std alone misrepresent an algorithm whose seeds are not
    concentrated: AdSwitch's std exceeds its mean at K=64 (58,403 vs 55,637)
    and its best and worst seeds differ 26x, so the median is roughly half the
    mean and the mean describes no run that actually happened.
    """
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
