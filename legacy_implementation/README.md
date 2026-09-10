# legacy_implementation/

Superseded code, kept for reference and history. **Nothing here is imported by
`test_algorithms.py`, `bench/`, or any current algorithm module.** Do not treat
these files as a description of how the repo works today.

| file | superseded by | why |
|---|---|---|
| `TS_GE.py` | `../ts_ge.py` | Self-contained early TS-GE. It owns its environment (unbounded Gaussian rewards, clipped into `[0,1]` before the Bernoulli step, which silently violates TS-GE Assumption 1), pads non-power-of-two `K` with dummy arms, and — importantly — uses a **different super-arm coding** than `../ts_ge.py`: it codes arms `1..K` over `ceil(log2(K+1))` bits per the paper's Algorithm 2, where the current implementation codes `0..K-1` over `log2(K)` bits and relies on a recovery gate to disambiguate the empty codeword. Reading this file to understand `../ts_ge.py` will mislead you. |
| `TS.py` | — | Standalone Thompson-sampling "restaurant satisfaction" toy, unrelated to the piecewise-stationary benchmark. It is what wrote the `artifacts/ts_iteration_*.png` series. |
| `Arms.py` | `../synthetic_env.py` | Arm container used only by `TS_GE.py` above; its non-stationarity support was never finished. |

If you want the paper-faithful 1-indexed CSA coding in the *current*
implementation, change `_groups()` in `../ts_ge.py` — the
`no_arm_matched_pattern` branch there already handles the empty flag pattern
that coding produces. It costs one extra super-arm of `n_ge` probes per
detection; see the module docstring for the measurement.
