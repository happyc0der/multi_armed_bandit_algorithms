"""Multi-seed benchmark for FastAdSwitch, TS-GE, UCB1, EpsilonGreedy and M-UCB.

    python test_algorithms.py                          # every non-heavy case
    python test_algorithms.py --case case3_K64_large_K
    python test_algorithms.py --quick                  # smoke test, seconds
    python test_algorithms.py --list

Artifacts per case, under --out (default artifacts/):

    <case>_summary.csv        total / decision / probe regret, probe age, wall time
    <case>_per_seed.csv       every seed's numbers, plus detection times
    <case>_phase_regret.csv   TS-GE regret split by ETC / TS / BP / GE / RECOVERY
    <case>_regret.png         two panels: total regret, then decision-only regret

READING THE OUTPUT
==================
Compare `mean_decision_regret`, not `mean_total_regret`, when asking which
algorithm chooses arms better. TS-GE is *required* to broadcast-probe every arm
every episode (Condition 1 in the paper), and those slots are charged roughly
(mu* - mean_i mu_i) each: they were 76% of its total at T=6,000 and 98.5% at
T=200,000. UCB1, epsilon-greedy, M-UCB and FastAdSwitch never probe, so they
never pay that, and they offer no bound on `max_probe_age` in exchange.

The case definitions live in bench/cases.py; the feasibility rules that
constrain them live in bench/preflight.py.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from bench import preflight as preflight_module
from bench.cases import ALL_ALGORITHMS, CASES, get_case
from bench.runner import (
    print_phase_table,
    print_summary,
    run_case,
    write_artifacts,
)

ARTIFACT_DIR = "artifacts"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case", action="append", dest="cases", metavar="NAME",
        help="case to run (repeatable). Default: every case not marked heavy.",
    )
    parser.add_argument("--seeds", type=int, help="number of seeds (overrides the case default)")
    parser.add_argument("--T", type=int, help="horizon override, for probing feasibility limits")
    parser.add_argument(
        "--algorithms", metavar="A,B",
        help=f"comma-separated subset of {','.join(ALL_ALGORITHMS)}",
    )
    parser.add_argument(
        "--bp-policy", choices=("paper", "power"), dest="bp_policy",
        help="TS-GE BP phase length: 'paper' is T^(2/5); 'power' sizes it by the "
             "detection power the BP test needs, which shrinks as K grows. "
             "'power' steps outside the paper's proof -- see ts_ge.py deviation 5.",
    )
    parser.add_argument(
        "--etc-policy", choices=("paper", "detection"), dest="etc_policy",
        help="TS-GE ETC length: 'paper' is Lemma 1 (individual well-localization); "
             "'detection' sizes it for the averaged estimates the BP/GE tests use.",
    )
    parser.add_argument("--out", default=ARTIFACT_DIR, help=f"artifact directory (default {ARTIFACT_DIR})")
    parser.add_argument(
        "--quick", action="store_true",
        help="smoke test: T=6000, 2 seeds, no plots",
    )
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    parser.add_argument(
        "--no-preflight", action="store_true",
        help="run even if the parameters cannot produce a meaningful result",
    )
    parser.add_argument("--list", action="store_true", help="list the known cases and exit")
    return parser.parse_args(argv)


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.cases:
        names = args.cases
    else:
        names = [name for name, case in CASES.items() if not case.get("heavy")]
        print(f"Running non-heavy cases: {', '.join(names)}")
        heavy = [name for name, case in CASES.items() if case.get("heavy")]
        if heavy:
            print(f"Skipping heavy case(s): {', '.join(heavy)} (select with --case)")

    cases = []
    for name in names:
        case = get_case(name)
        if args.T is not None:
            _set_horizon(case, args.T)
        if args.quick:
            _set_horizon(case, min(case["T"], _quick_horizon(case)))
        if args.algorithms:
            case["algorithms"] = [a.strip() for a in args.algorithms.split(",") if a.strip()]
        if args.bp_policy:
            case["ts_ge_bp_length_policy"] = args.bp_policy
        if args.etc_policy:
            case["ts_ge_etc_policy"] = args.etc_policy
        cases.append(case)
    return cases


def _set_horizon(case: dict[str, Any], horizon: int) -> None:
    """Retarget a case at a new horizon, keeping the change at the midpoint."""
    case["T"] = horizon
    case["change_schedule"] = [
        (max(1, min(t, horizon // 2)), arm, mean) for t, arm, mean in case["change_schedule"]
    ]


def _quick_horizon(case: dict[str, Any]) -> int:
    """Smallest horizon at which this case is still meaningful.

    A flat T=6000 would make every K >= 64 case fail preflight, since ETC alone
    needs more slots than that. Quick mode therefore shrinks the horizon only
    as far as the feasibility rule allows, which is still seconds per seed for
    everything but the heavy case.
    """
    magnitudes = preflight_module.change_magnitudes(case["means"], case["change_schedule"])
    if not magnitudes:
        return 6000
    floor = preflight_module.required_T(
        case["K"], min(magnitudes), case.get("ts_ge_noise_scale", 1.0)
    )
    return max(6000, int(floor) + 1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        for name, case in CASES.items():
            mark = " [heavy]" if case.get("heavy") else ""
            print(f"{name}{mark}\n    K={case['K']} T={case['T']} "
                  f"delta={case['ts_ge_delta']} seeds={case['default_seeds']} "
                  f"algorithms={','.join(case['algorithms'])}\n    {case['note']}\n")
        return 0

    failures = 0
    for case in select_cases(args):
        n_seeds = args.seeds or (2 if args.quick else case["default_seeds"])
        seeds = list(range(1, n_seeds + 1))
        algorithms = case["algorithms"]

        print(f"\n{'=' * 78}\nCASE: {case['name']}\n{'=' * 78}")
        print(f"K={case['K']}  T={case['T']}  seeds={n_seeds}  algorithms={', '.join(algorithms)}")
        print(
            f"R_max={case['reward_R_max']}  sigma={case['sigma']}  "
            f"TS-GE(delta={case['ts_ge_delta']}, n_ge={case['ts_ge_n_ge']}, "
            f"noise_scale={case.get('ts_ge_noise_scale', 1.0)}, "
            f"bp={case.get('ts_ge_bp_length_policy', 'paper')}, "
            f"etc={case.get('ts_ge_etc_policy', 'paper')})  "
            f"M-UCB(delta={case['mucb_delta']}, M={case['mucb_M_estimate']})"
        )
        if case.get("note"):
            print(f"Note: {case['note']}")

        try:
            findings = preflight_module.preflight(case)
            preflight_module.report(case["name"], findings, strict=not args.no_preflight)
        except preflight_module.PreflightError as error:
            print(f"\nSKIPPED: {error}", file=sys.stderr)
            failures += 1
            continue

        print()
        result = run_case(case, seeds, algorithms)
        print_summary(result)
        print_phase_table(result)

        written = write_artifacts(result, args.out, plots=not (args.no_plots or args.quick))
        print("\nSaved:")
        for path in written:
            print(f"  {path}")

    print(f"\n{'=' * 78}")
    if failures:
        print(f"{failures} case(s) skipped by preflight. Artifacts under {args.out}/")
        return 1
    print(f"All selected cases complete. Artifacts under {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
