"""Benchmark harness shared by test_algorithms.py and test_fast_adswitch.py.

    bench.preflight  parameter feasibility checks, run BEFORE any simulation
    bench.metrics    regret decomposition and the Condition-1 probe-age metric
    bench.cases      case definitions and algorithm factories
    bench.runner     one sweep per case, artifact writers

The point of splitting these out is that a single total-regret number is not
interpretable for TS-GE: 76-98% of it is the mandatory broadcast probing that
the algorithm is *required* to perform and that its competitors never pay for.
`bench.metrics` separates that cost from decision quality, and
`bench.preflight` refuses parameter settings that cannot produce a meaningful
result at all.
"""
