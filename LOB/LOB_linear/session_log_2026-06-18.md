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

## Open items
- `fine_optimal_execution`: needs proper `gen_data()` and `_fit_value()` rewrite
- `self.intervals` needs to be changed back from `{oe.N}` to support multiple intervals
- Issue #7 from review: one `trade_size` network shared across all intervals — may need per-interval networks
- L-BFGS hybrid (Adam warmup → L-BFGS polish) remains an option if more speed needed
