---
name: ms-pgm-v7-timing-tuning
description: "MS_PGM_v7 wall-clock tuning experiments: trajectory counts, foe intervals, M_comp, the OU-resample bug fix, and a recurring oe100 anomaly"
metadata:
  type: project
---

## num_trajectories (oe): 300 → 100
Big free win: `oe.train()` dropped ~75% (21.26s → 5.32s) because Level 2 retraining converges in far
fewer epochs with a smaller batch (66 epochs vs 3,001). No quality loss observed. `oe100` is
unaffected since it explicitly overrides `num_trajectories` to 600.

## foe intervals {1,2,5,10} → {1,2,10}
**Not a reliable speedup.** One run finished in 4.93s (239 epochs, exact-zero relative-change stop) —
but that was a stuck/degenerate optimization fluke (epoch-1 cost was ~24, ~10-50x higher than normal),
not a real property of dropping interval 5. A repeat run trained normally and took ~63s, matching the
4-interval baseline. Quality (vs `oe100`, 20-seed comparison) was fine either way (~0.1-1% mean rel diff).

## foe num_trajectories_comp (M_comp): 50 → 10
Real ~2x speedup (63.37s → 30.44s), but a real robustness cost: multi-seed std nearly doubled
(2.35% → 4.31%), final remaining inventory `R` rose and got noisier (mean 275→364, std 50.7→65.1).
`M_comp` sets the batch size for the "complete trajectory" term in `fine_optimal_execution.cost()` —
the piece that enforces full-horizon liquidation consistency across intervals (see
`x_comp`/`complete_cost` in `cost()`). Smaller `M_comp` = noisier version of that regularizing signal.

## M_comp = 20: caused a catastrophic divergence (one run)
`foe.train()` loss was normal through epoch 2,400 (~0.44) then exploded to 2.2e22 by epoch 3,000.
Sanity check confirmed a broken policy: every trade size uniformly large and negative (interval
states -489.6 to -87.3, mean -320; complete-traj states -496.2 to -378.8, mean -429).
Root-cause hypothesis: `fine_optimal_execution`'s OU κ/ρ paths are now fixed once at construction
(see fix below) — if that one draw sends κ near its clamp floor (1e-4) for even one `M_comp` sample at
some point in its 100 fine steps, the quadratic cost penalty `(kappa/2)*psi^2` collapses at that
state, permanently (same fixed path every epoch). `trade_size`'s output is unbounded (plain
`Linear→ReLU→Linear→ReLU→Linear(1)`, no clamp), and `foe.train()` uses constant `lr=8e-3` Adam with no
scheduler/grad-clipping (unlike `oe.value()`, which has `ReduceLROnPlateau`) — nothing damps a bad
update once the optimizer starts exploiting the under-penalized state, and it compounds through the
100 sequential steps of the complete-trajectory rollout within one backward pass.
Stationary OU distribution for κ is `N(0.05, 0.02²)`; `P(κ≤0) ≈ Φ(-2.5) ≈ 0.62%` per time-point — not
negligible across a 100-step path. See [[project_ms-pgm-v7-kappa-lower]] for the fix in progress.

## Fixed: foe was resampling fresh OU noise every epoch (now fixed once at construction)
`fine_optimal_execution.update()` used to draw new `torch.randn` κ/ρ noise on every call — meaning
every epoch of `foe.train()` re-simulated a brand-new random path, so the `err=1e-9` early-stopping
check could never trigger (irreducible per-epoch noise) and `foe.train()` always burned the full
epoch budget. Fixed by precomputing `self.kappa_paths`/`self.rho_paths` (interval batch) and
`self.kappa_paths_comp`/`self.rho_paths_comp` (complete-trajectory batch) once in `__init__` via a new
`_gen_ou_paths` helper — matching how `optimal_execution` already worked (and `oe100` already
inherited, since it's just an `optimal_execution` instance). `update()`/`unit()` now take optional
`kap_next`/`rho_next` args, falling back to the old fresh-draw behavior if omitted, for backward
compat with cells that manage κ/ρ externally (e.g. the shared-path multi-seed foe-vs-oe100 comparison).
Result: loss curve is now genuinely smooth/monotonic (not just noisy-looking-flat), confirming
`cost()` is deterministic given fixed params — but this also revealed the true loss hadn't actually
plateaued by epoch 600 like it appeared to under the old noisy runs, so aggressively cutting
`num_epochs` would sacrifice real quality, not just discard noise.

## Recurring, unexplained oe100 anomaly
`oe100.train()`'s epoch-1 cost has spiked wildly in multiple independent runs (6.72e5, 1.38e5, 2.44e5,
vs. typical ~1.5-2e4), self-correcting by ~epoch 1200. Unrelated to any of the edits above (oe100's
own config wasn't touched in these experiments). Not yet root-caused — worth investigating if oe100's
wall-clock time needs to be reliable/reported.

**Why:** Chasing `foe.train()`'s wall-clock cost (was 88.94s baseline, the dominant cost in the
pipeline) while checking each change doesn't silently break policy quality.

**How to apply:** Prefer the `num_trajectories` (oe) reduction — free win. Be skeptical of any single
"fast" `foe.train()` run without checking the sanity check + multi-seed comparison; verify the loss
curve is smooth, not just short.
