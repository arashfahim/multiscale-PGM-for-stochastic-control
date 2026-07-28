---
name: ms-pgm-v7-wallclock-logging
description: "MS_PGM_v7 wallclock_log.json system for averaging wall-clock across reruns, the 22-run divergence study, and the Total-vs-benchmark boxplot"
metadata:
  type: project
---

## Logging system (two cells in MS_PGM_v7.ipynb)
- `wallclock_log_cell` (right after `oe100.train()`, cell id `aa28e0f9`): logs `oe_train_s`
  (`t_elapsed_coarse`), `oe_value_s` (new `oe.value_time` attribute — `optimal_execution.value()`
  previously only printed its timing, didn't store it, so `self.value_time = t_elapsed` was added),
  `foe_train_s` (`t_elapsed_fine`), `oe100_train_s` (`t_elapsed` from the oe100 cell). Controlled by
  `overwrite_wallclock_log` (bool): `True` starts fresh, `False` (default) appends to
  `wallclock_log.json` and recomputes a running `"average"` dict. Tracks `"num_runs"` = `len(runs)`.
- `wallclock_log_cell2` (right after the multi-seed comparison cell `ffe50286`): re-opens the JSON and
  **updates the same run's entry** (not a new entry) with multi-seed stats: `mean_total_cost_foe/
  oe100`, `mean/std/min/max_rel_diff_pct`, `mean/std_finalR_foe/oe100`. One full notebook run = one
  combined JSON record with both wall-clock and multi-seed fields, not two separate records.

## 22-run study: divergence persists at ~9% even after the kappa_lower fix
Ran the notebook 22 times (2 initial + a 20-run sequential loop) with the `kappa_lower` fix (both
`step_cost` and `D` update, see [[project_ms-pgm-v7-kappa-lower]]) in place, `num_trajectories`
(oe)=200, `foe` intervals `{1,2,10}`, `M_comp=20`. **20 of 22 runs converged sanely** (rel diff mostly
within ±6%, occasionally up to ~32%; final `R` ~120-340; total cost ~13,000-18,000) — **2 runs (~9%)
diverged catastrophically**, on different sides: run 2 had `oe100` break (cost 6.24M, final R 23,763),
run 18 had `foe` break worse (cost 263.8M, final R 147,615). Revised an earlier hypothesis that the
recurring cost spikes were "an oe100-specific anomaly" — it's not consistently one side.

### Robust averaging approach (user's explicit preference: no median)
Classify a run as diverged if `abs(mean_finalR_foe or mean_finalR_oe100) > 2000` OR
`mean_total_cost_foe or mean_total_cost_oe100 > 50000`. Exclude diverged runs, then plain mean ± stdev
over the remainder. **User explicitly said "I don't need median"** — prefers threshold-exclusion +
mean/std over a robust-statistic approach.

### Final table (20 converged runs of 22)
| Step | Mean | Std dev |
|---|---|---|
| `oe.train()` | 15.90 s | 9.36 s |
| — `oe.value()` (nested inside `oe.train()`, not additive) | 2.43 s | 1.01 s |
| `foe.train()` | 66.67 s | 56.30 s |
| **Total** (`oe.train()+foe.train()`, computed per-run then averaged — NOT sum of the row means, and NOT summing all 3 rows since that double-counts `oe.value()`) | 82.57 s | 58.40 s |
| `oe100.train()` (benchmark) | 224.19 s | 229.76 s |

Std devs are large relative to means (`oe100.train()`'s std nearly equals its mean) — even among
"converged" runs there's high variance in epochs needed, not just a binary converged/diverged split.

### Boxplot (new cell `wallclock_boxplot`, after `wallclock_log_cell2`)
Reads `wallclock_log.json` directly (doesn't need `oe`/`foe` trained — only needs `plt` + the JSON
file, so it can be tested standalone by copying just the imports cell + this cell into a temp
notebook). Final design: **single panel**, two boxes — `Total (MS-PGM)` vs `oe100.train() (Benchmark)`
— matching the LaTeX table's Total/Benchmark framing. An earlier 4-box, 2-panel version (split by
scale, `oe.train`/`oe.value`/`foe.train` vs `oe100.train` on separate width-ratio'd axes) was replaced
per user request after flagging that `sharey=True` would compress the smaller boxes to near-zero.
By median (not mean), MS-PGM's `Total` box sits clearly below `oe100.train()`'s — the "MS-PGM slower
than benchmark" conclusion drawn earlier from the mean was an artifact of one outlier run (~281s).

### Per-run finding: oe100 beats Total in 4 of 20 converged runs (20%)
Checked directly: runs 5, 10, 13, 14 had `oe100.train() < Total`. Not driven by `oe100` being unusually
fast — driven by `foe.train()` running unusually long in those specific runs (e.g. run 14: Total=138.84s
vs. the ~83s average).

**Why:** Chasing whether MS-PGM (coarse+fine) is actually faster than the brute-force `oe100`
benchmark, and whether a single run's timing/quality numbers can be trusted at all given the
divergence risk.

**How to apply:** `wallclock_log.json` (in `LOB/`) has 22 real records — reuse/append to it rather than
starting over. The convergence-filter function (threshold on `mean_finalR_*`/`mean_total_cost_*`) is
reusable in any new analysis cell over this log.
