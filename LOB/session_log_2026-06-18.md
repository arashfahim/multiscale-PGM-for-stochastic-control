# Session Log — 2026-06-18

## MS_PGM_v1.ipynb Changes

### 1. D initialization in `optimal_execution.__init__`
- **Original**: `torch.rand(...) * self.kappa * self.X0 / self.N`
- **Changed to**: `torch.rand(...) * self.X0 * 0.2 * self.kappa`
- **Why**: Old range was too narrow (tied to single-step scale). Tried `X0*1.5` (too large, caused negative trades), then `X0*1.5*kappa` (D up to ~55, still caused some negative trades at high D), settled on `X0*0.2*kappa` (D up to ~10, no negative trades).

### 2. R initialization in `optimal_execution.__init__`
- **Original**: `self.X0*0.5 + (self.X0*1.5 - self.X0*0.5) * torch.rand(self.M, 1)`
- **Changed to**: `self.X0*0.9 + (self.X0*1.1 - self.X0*0.9) * torch.rand(self.M, 1)`
- **Why**: Narrower range around X0 for tighter training distribution. R is now defined before D.

### 3. Value function training: full-batch → mini-batch SGD
- **Location**: `optimal_execution.value()` method
- **Old**: Full-batch Adam, 3000 epochs, no LR schedule. Took ~57s, often didn't converge (rel. change 0.32).
- **New**: Mini-batch Adam (batch_size=256), `ReduceLROnPlateau` scheduler (patience=200, factor=0.5), early stopping.
- **Result at 3000 epochs**: 18.7s, MSE ~2e-7 (converged well).
- **Current setting**: `num_epochs = 1000`. At 500 epochs: 3.3s, MSE ~3e-6, terminal error 1.13%. At 1000 epochs: ~6s, MSE ~5e-7.

### 4. Commented out all "fat" training/data
- Fattened training round in `train()` — commented out
- `self.trade_size_fat`, `self.V_fat` initialization — commented out
- `gen_data()` fat rollout block — commented out
- `value()` V_fat training block — commented out
- Cells referencing V_fat or _x_data_fat — commented out

### 5. Added terminal value sanity check (code cell 20)
- Verifies V_NN(T, D, R) vs exact terminal cost `(D + kappa/2 * R) * R`
- Also verifies y_data at t=T matches exact (confirmed 0% error — data generation is correct)

### 6. `fine_optimal_execution` changes
- `gen_data()` and `_fit_value()` — commented out (need rewriting)
- `self._fit_value()` call in `train()` — commented out
- `self.intervals` temporarily set to `{oe.N}` (last interval only) for testing
- **Test rationale**: Last interval has exact terminal cost (no coarse V approximation error), so fine-scale NN should train accurately. Confirmed: foe trains in ~70 epochs.

### 7. Code cell 29: fine-scale trade size bar chart
- Plots NN fine trades vs last 10 steps of full closed-form (N=100)
- Both start from the same (D, R) state: rolls out CF from t=0 to get state at t=0.9
- CF state at t=0.9: D=12.38, R=302.50

## MS_PGM_v2.ipynb
- Copy of v1 with an additional comparison cell (code cell 10): Adam vs L-BFGS for value function fitting
- **Adam**: 20.8s, 3000 epochs, MSE = 3.9e-7
- **L-BFGS**: 18.9s, 42 epochs (converged early), MSE = 1.9e-5 (50x worse)
- L-BFGS got stuck in shallow local minimum. Decided to stay with Adam for now.

## Key observations
- Negative trades occur when D initialization is too large — the network exploits `(D + kappa/2*psi)*psi` by choosing negative psi
- Value function bottleneck: full-batch → mini-batch gave 3x speedup with better convergence
- `gen_data()` in `optimal_execution` is called the right number of times (once inside `value()`)
- y_data at t=T perfectly matches the closed-form terminal cost (0% error)

## Parameters (current)
```python
lob_params = {'T': 1, 'kappa': 0.05, 'rho': 2, 'alpha': 1, 'initial_balance': 1e3, ...}
approx_params = {'num_trajectories': 300, 'num_time_steps': 10, 'num_neurons': 16, ...}
approx_params_fine = {'num_trajectories': 200, 'num_neurons': 8, ...}
```

## MS_PGM_v3.ipynb (Session 2 — 2026-06-18 evening)

### Root cause: coarse_V extrapolation failure
- Training data (D, R) at each time step has correlation **0.997** — V_NN only learns along a 1D diagonal in (D, R) space
- At t=0.9, training range: D in [10.41, 13.56], R in [281, 424], but all on a thin band
- V_NN extrapolates poorly outside this band: at R=528, underestimates true cost by ~5000
- This caused fine_optimal_execution to learn negative trades (hoarding inventory), exploiting the inaccurate boundary cost
- Diagnostic plots saved: `coarseV_extrapolation.png`, `coarseV_training_data_t09.png`

### Fix 1: Two-level training in `optimal_execution`
- **Level 1**: Standard training from correlated trajectory data (unchanged)
- **Level 2** (new): After Level 1, at each time step find [D_min, D_max] × [R_min, R_max] from trajectories, sample D and R **independently** within rectangle, retrain `trade_size` (warm start, same optimizer) on this rectangle data
- For V training: `gen_data()` uses **only** rectangle-sampled starting points (not rollout trajectories), so correlation stays ~0 at every time step
- Cost-to-go for each rectangle point computed via `self.cost()` starting from t_n (not t=0)
- Result: correlation dropped from 0.997 to ~0 at all time steps; V_NN fits well across full 2D rectangle

### Fix 2: Removed `.detach()` from `coarse_V` in `fine_optimal_execution.cost()`
- **Old**: `costs[:, step] = self.coarse_V(u).squeeze(1).detach()`
- **New**: `costs[:, step] = self.coarse_V(u).squeeze(1)`
- With `.detach()`, gradients from boundary value couldn't flow back to `trade_size` — optimizer had no signal that negative trades increase boundary cost
- Without `.detach()`, coarse_V's parameters aren't updated (not in foe's optimizer), but chain rule tells `trade_size` how its decisions affect boundary cost
- This was the critical fix that eliminated negative trades

### Fix 3: Fixed `error_report()` in `fine_optimal_execution`
- **Old**: compared NN cost-to-go from t_start against `full_cf(R)` (total cost from t=0) — apples-to-oranges
- **New**: uses `full_cf.value_function(n, D, R)` for exact cost-to-go comparison at matching (t, D, R)
- Test states now sample D from training range (not D=0)

### Results with different interval configurations
| Intervals | Rel. L1 error | Notes |
|-----------|--------------|-------|
| {10} (v1) | ~1% | Single interval, exact terminal cost |
| {9, 10} | 5.7% | Best multi-interval result |
| {8, 10} | 14% | Further from T = harder |
| {7, 10} | 20% | NN overtrades ~2x in interval 7 |
| {1, 3, 7, 10} | 24% | 4 non-contiguous intervals, early intervals struggle |

### Bar chart evaluation methodology
- Must evaluate `foe.trade_size` on the **CF-100 optimal (D, R) trajectory**, not on the NN's own rollout
- NN rollout diverges from CF after first step, compounding errors and giving misleading comparison
- Plot shows NN trade recommendation at each CF state, including on intervals the NN wasn't trained on (it generalizes via smooth function of t)

### Code cell changes in v3
- **Cell 6** (`optimal_execution`): Added Level 2 training block in `train()`, rewrote `gen_data()` to use rectangle-sampled starting points only
- **Cell 7**: Fresh `oe` creation (no pickle cache)
- **Cell 8**: Always train (removed `if not oe.trained` guard)
- **Cell 28** (`fine_optimal_execution`): Removed `.detach()` from `coarse_V` in `cost()`, fixed `error_report()`
- **Cell 29**: Evaluates NN on CF-100 trajectory, shows all time steps including non-trained intervals
- **Cell 31**: Three-way comparison (V_NN vs coarse CF vs fine CF) — confirmed discretization error is negligible (0.03%)

## Open items (updated)
- NN overtrades ~2x on intervals far from T — may need more training epochs or per-interval networks
- `gen_data()` and `_fit_value()` in `fine_optimal_execution` remain commented out — do not rewrite
- L-BFGS hybrid remains an option if more speed needed
- Consider widening the rectangle sampling range beyond the trajectory bounding box
