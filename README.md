# multi_armed_bandit_algorithms

Piecewise-stationary multi-armed bandit algorithms, aimed at use in financial
markets. Five algorithms implement one shared interface
(`bandit_base.BanditAlgorithm`) and are benchmarked on one shared bounded
environment (`synthetic_env.SyntheticEnv`).

| module | algorithm |
|---|---|
| `ts_ge.py` | TS-GE — Ghatak, *Actively Tracking the Optimal Arm in Non-Stationary Environments with Mandatory Probing* (`docs/TS_GE.pdf`) |
| `fast_adswitch.py` | AdSwitch — Auer, Gajane, Ortner, COLT 2019 (`docs/auer19a.pdf`) |
| `mucb.py` | M-UCB — Cao, Wen, Kveton, Xie, AISTATS 2019 |
| `ucb1.py` | UCB1 — stationary baseline |
| `epsilon_greedy.py` | epsilon-greedy — stationary baseline |

Requires `numpy` and `matplotlib`.

## Quick start

```bash
python test_algorithms.py --list       # what cases exist
python test_algorithms.py --quick      # smoke test, seconds per case
python test_algorithms.py              # every case except the T=1e6 one
python test_algorithms.py --case case3_K64_large_K

# TS-GE with phase lengths sized by detection power rather than the
# paper's formulas (see "Phase lengths" below)
python test_algorithms.py --case case3_K64_large_K \
    --bp-policy power --etc-policy detection
```

## Layout

```
bench/                    benchmark harness
  preflight.py            parameter feasibility checks, run before any simulation
  metrics.py              regret decomposition, probe age, dispersion
  cases.py                case definitions and algorithm factories
  runner.py               one sweep per case, artifact writers
docs/                     the two source papers
legacy_implementation/    superseded code, imported by nothing
artifacts/                generated CSVs and plots (gitignored)
```

Per case, `artifacts/` gets `<case>_summary.csv`, `<case>_per_seed.csv`,
`<case>_phase_regret.csv` and `<case>_regret.png`.

## Reading the results: decision regret, not total regret

TS-GE is *required* to broadcast-probe every arm every episode — that is
Condition 1 in the paper and the entire point of the algorithm. A probe slot
returns the average reward over all K arms, so the regret definition charges it
roughly `mu* - mean_i mu_i` no matter how well the algorithm is learning.
Measured on case1 (K=2, T=6000):

| phase | slots | regret | share |
|---|---|---|---|
| ETC | 870 (14.5%) | 1,754 | 17.1% |
| TS — actual arm choices | 2,950 (49.2%) | 208 | 2.0% |
| **BP — mandatory probing** | **2,080 (34.7%)** | **7,786** | **75.9%** |
| GE | 50 (0.8%) | 503 | 4.9% |
| RECOVERY | 50 (0.8%) | 1 | 0.0% |

At T=200,000 the BP share rises to 98.5%: per-slot BP regret is roughly constant
while TS-phase regret goes to zero. TS-GE's *total* is therefore essentially
`BP_slots x (mu* - mean_i mu_i)` — a property of the environment's mean spread,
not of learning quality.

UCB1, epsilon-greedy, M-UCB and AdSwitch never probe, so they are never charged
for it, and none of them bounds how stale an arm's last sample may be. Every run
is therefore reported three ways:

- `total_regret` — what the paper's Eq. (2) charges
- `decision_regret` — regret on slots where a single arm was chosen
- `max_probe_age` — the longest any arm went unprobed after initialization,
  which is what Condition 1 bounds by `sqrt(T)`

## Results

Mean over seeds; algorithms that never probe have `decision == total`.
Latency is slots from the true change point to the first detection.

| case | K | T | TS-GE total | TS-GE decision | M-UCB | AdSwitch (median) | TS-GE lat. | M-UCB lat. | AdSwitch lat. |
|---|---|---|---|---|---|---|---|---|---|
| case1 | 2 | 6e3 | 10,163 | **1,875** | **1,915** | 5,626 ± 5,519 (3,464) | 26 | 50 | 559 |
| case2 | 16 | 1e5 | 219,138 | **10,219** | 43,517 | 55,391 ± 39,029 (35,274) | 188 | 716 | 4,960 |
| case3 | 64 | 1e5 | 306,486 | **49,377** | 109,861 | 55,637 ± 58,403 (22,327) | 264 | 1,273 | 3,720 |
| case4 | 128 | 1e6 | 2,441,252 | **399,232** | 525,336 | 356,205 ± 193,197 (242,038) | 1,675 | 4,827 | 24,856 |

TS-GE picks arms better than M-UCB from K=16 upward — 4.3x at K=16 — and detects
changes 3-15x faster than either competitor. It loses on total regret purely for
probing it is required to do. Reporting only the total inverts the conclusion.

### AdSwitch's variance is not noise, it is which arm changed

Cases 1-4 all promote arm 0 — the *worst* arm — to best. Arm 0 has long since
been evicted, so only AdSwitch's condition (4) can notice, and that revisits an
evicted arm on a sparse random schedule. Every post-change restart in cases 1-4
was tagged `bad_arm_change`. `case5_best_arm_collapse` runs the mirror
direction, where a currently-good arm degrades and condition (3) — which checks
every good arm every round — sees it immediately:

| direction | total | std | median | max/min | latency | restart reason |
|---|---|---|---|---|---|---|
| arm 0 promoted (case2) | 55,391 | 39,029 | 35,274 | 18x | 4,960 | `bad_arm_change` x10 |
| best arm collapses (case5) | 35,491 | **863** | 35,491 | 1x | **4** | `good_arm_change` x10 |

Same algorithm, same K and T. This is by design — sparse rechecking of evicted
arms is exactly what buys the sublinear worst-case bound — but it means
"AdSwitch's mean regret" describes no run that actually happened. The summary
CSV carries median and quartiles alongside mean and std, and the console warns
when the spread makes the mean meaningless.

AdSwitch also has **no probe-age bound at all**: it leaves some arm unsampled
for roughly half the horizon (49,371 of 100,000 at K=16; 486,333 of 1,000,000 at
K=128). Where mandatory probing is a real requirement, it is not a candidate at
any regret. M-UCB, on the other hand, gets a probe age of 295-581 at K=16-64
from its forced round-robin, matching TS-GE's 417-611 — so that guarantee is not
unique to TS-GE.

### Not every change needs detecting

`case5` also shows the flip side. When the *best* arm degrades, plain UCB1
handles it for free — its own optimism walks it to the next-best arm — and every
detection mechanism is pure overhead:

| algorithm | total | std | decision |
|---|---|---|---|
| **UCB1** | **10,123** | 313 | 10,123 |
| M-UCB | 14,588 | 168 | 14,588 |
| EpsGreedy | 24,933 | 891 | 24,933 |
| AdSwitch | 35,491 | 863 | 35,491 |
| TS-GE | 61,997 | 1,765 | 16,977 |

Change detection pays off when a *neglected* arm improves. It costs when the
incumbent degrades. A suite that tests only one direction overstates whichever
family it favours.

### Phase lengths are TS-GE's biggest lever

`T_BP = T^(2/5)` does not depend on K, but the BP statistic averages K arms over
`n_bp` slots, so its standard error is `sigma/sqrt(K*n_bp)` — larger K needs
*fewer* probe slots, not the same number. The same applies to `n_ETC`, which the
paper sizes for individual arm accuracy even though every estimate the detection
tests consume is an average over K or K/2 arms.

| case | policy | T_BP | n_ETC | total | decision | detections |
|---|---|---|---|---|---|---|
| case2 K=16 | paper | 100 | 36 | 219,138 | 10,219 | 10/10 |
| | `--bp-policy power` | 3 | 36 | **20,264** | 11,661 | 10/10 |
| case3 K=64 | paper | 100 | 341 | 306,486 | 49,377 | 5/5 |
| | both policies | 6 | 6 | **53,381** | 33,042 | 10/10 |

At K=64 that is 5.7x, putting TS-GE past M-UCB's 109,861 and past AdSwitch's
median. Both policies are **opt-in and default to the paper's formulas**: they
step outside the paper's proof, so they are a measured engineering trade, not a
claim about the theory. Detection latency rises (264 -> 660 slots at K=64) and
probe age rises slightly, both still inside the `sqrt(T)` budget.

One caveat: the K=64 numbers above hold over 10 seeds with zero missed
detections, but an intermediate setting (`T_BP=12`) missed a detection on 1 of 3
seeds. The derived value is not a monotone safety frontier — widen the sweep
before adopting either policy as a default.

## Design rules TS-GE cannot escape

`bench/preflight.py` checks these before any case runs, because violating either
produces a run that completes normally and reports a meaningless number.

**1. Detectability.** A change of `Delta_C` in one arm moves the K-arm broadcast
average by only `Delta_C / K`, so

```
Delta_C / K >= 4 * delta
```

At K=64, `Delta_C`=14, `delta`=0.1 the shift is 0.219 against a threshold of
0.400, and the algorithm registers **zero detections across T=200,000**.

**2. ETC budget.** Initialization costs `K * noise_scale^2 * ln(T) / (2 delta^2)`
slots. At K=64, T=6000, `delta`=0.1 that wants 27,840 of 6,000 slots: 100% of the
horizon is round-robin exploration and no episode ever runs.

Substituting rule 1 into rule 2 gives the combined feasibility rule:

```
K^3 <= rho * T * (Delta_C / noise_scale)^2 / (8 ln T)
```

With `Delta_C/noise_scale = 28` and `rho = 0.25` that is K <= 60 at T=1e5 and
K <= 121 at T=1e6 — which is why `case4_K128_paper_regime` runs at T=1e6. At
T=1e5 it is infeasible under *any* delta.

**3. False alarms.** Eq. (12) *derives* delta from the noise level via
`P_FA = Q(4 delta / sigma_NC) <= 1/T`. Preflight reports the implied delta; a
hardcoded delta that ignores sigma, K and the phase lengths carries no
false-alarm guarantee.

## Where `ts_ge.py` departs from the paper

Full detail is in the `ts_ge.py` module docstring. In brief:

1. **The paper's Beta update is backwards.** Algorithm 1 lines 11-12 read
   `alpha += 1 - R*`, `beta += R*`, while line 7 selects `argmax theta`. Since
   `Beta(alpha, beta)` has mean `alpha/(alpha+beta)`, that update makes the
   sampled value the *failure* rate, so the algorithm as written picks the arm
   most likely to fail. Measured at K=2, T=6000, seed 42: TS-phase regret
   **21,860** literal against **174** corrected. The corrected update is the
   default; `reproduce_paper_beta_typo=True` restores the literal text.

2. **Eq. (8) is dimensionally inconsistent** — it sums group *averages* and
   subtracts individual *means*. The code computes
   `|B_k| * mu_hat_Bk - sum_{i != j} mu_hat_i`, averaged over the flagged groups
   containing the localized arm.

3. **Post-change recovery is verified, not assumed.** The paper re-seeds the
   changed arm's prior from the closest other arm (line 29). Instead the code
   takes fresh individual pulls of the localized arm and commits the reset only
   if `|recovery_mean - old_mean|` exceeds `2 delta` *or* the recovery
   estimator's own confidence radius, whichever is larger. (`2 delta` alone is
   not enough: at sigma=3 with 50 samples the standard error is 0.42 against a
   0.20 threshold, so noise clears the bar unaided about two thirds of the time.)

   This matters because `_groups()` codes arms `0..K-1`, giving arm 0 the
   all-zero codeword — which is also what GE produces when it finds nothing.
   Without the gate every BP false alarm is blamed on arm 0 and wipes its
   statistics. Measured on a stationary K=2 run at sigma=3, three seeds:

   | | committed resets | of which unflagged, blamed on arm 0 | total regret |
   |---|---|---|---|
   | without gate | 15 / 10 / 13 | 6 / 3 / 5 | 5,972 / 5,735 / 5,770 |
   | with gate | **0 / 0 / 0** | **0 / 0 / 0** | 6,244 / 5,965 / 5,575 |

   No legitimate detection is blocked: a true `Delta_C = 14` change at K=8 still
   localizes and commits, bit-identically to the ungated version.

4. **`n_ETC` is scale-aware.** The paper's formula is scale-free because its
   rewards are unit-variance. `noise_scale` makes the dependence explicit — pass
   the environment's sigma for the sub-Gaussian count. This is what makes K >= 64
   feasible at all: at K=64, T=1e5 it drops ETC from 123% of the horizon
   (infeasible, zero detections) to 31%.

5. **Phase lengths are tunable.** `bp_length_policy` and `etc_policy` both
   default to the paper's formulas; `"power"` and `"detection"` are opt-in. See
   "Phase lengths are TS-GE's biggest lever" above.

6. **Bounded rewards.** The paper assumes Gaussian rewards but also bounds them
   by `R_max` (Assumption 1). `synthetic_env.py` uses a scaled Beta, which
   satisfies both, rather than clipping an unbounded Gaussian.

## Notes on the other algorithms

- **epsilon-greedy**: keep `decay=False`. Decay is for stationary bandits; with
  it, the algorithm cannot find a changed arm once exploration has shrunk, short
  of a decay schedule that already knows when changes occur.

- **AdSwitch** is dominated by condition (4), which asks whether an evicted arm's
  mean has moved. Evaluated naively that is `O(K*T^2)` — 89-96% of wall time —
  because it rescans every sigma in the episode for every bad arm every round.
  This implementation restricts it two ways, both exact:
  an arm's test value can only change on a round where that arm was *pulled*, and
  within an arm, sigma need only range over that arm's own pull times. Combined
  with per-arm sparse prefix sums in place of dense `(K, T+2)` arrays, that gives
  59x at K=16/T=1e5 (218.6 s -> 3.7 s) and 272 MB instead of 2,048 MB at
  K=128/T=1e6, with `chosen_arm`, `detections` and `evictions` bit-identical.

- **M-UCB** keeps running sums and a bounded deque for the change-detection
  window. Recomputing each arm's mean from its full history every round instead
  is `O(T^2)`: 17.7 s against 0.28 s at K=64/T=60,000, same results.

## Regression reference

`artifacts/` is gitignored, so the case1 baseline is recorded here. All five
algorithms must reproduce these exactly (mean final total regret, 20 seeds):

| algorithm | mean | std | min | max |
|---|---|---|---|---|
| FastAdSwitch | 5625.766894 | 5519.042400 | 624.928830 | 23308.533084 |
| TS-GE | 10162.864713 | 85.890860 | 10044.075167 | 10377.575524 |
| UCB1 | 30005.331795 | 34.946414 | 29942.241429 | 30077.091614 |
| EpsilonGreedy | 13028.372858 | 1821.488515 | 10202.998534 | 18707.741346 |
| M-UCB | 1914.944740 | 122.228658 | 1597.178292 | 2166.230515 |

```bash
python test_algorithms.py --case case1_K2_baseline
```
