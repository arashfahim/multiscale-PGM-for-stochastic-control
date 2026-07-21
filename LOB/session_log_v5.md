# Session Log — v3/v4/v5 (2026-06-23/24)

## v3 Changes
- **Fixed intervals bug** in `fine_optimal_execution.__init__`: `self.intervals` was hardcoded to `{1,3,5,7,oe.N}`. Now uses parameter: `(set(intervals) if intervals is not None else set(range(1, oe.N+1))) | {1, oe.N}`.
- **Added `error_report(M_test=500)`** to `optimal_execution`: compares NN cost-to-go vs closed-form at each coarse time step. Reports relative L1, L-inf, and per-step errors.
- **Deleted** old cells 31-32 (coarse_V three-way comparison, NN vs CF at t=0).
- **Split cell 29** at line 4: cell 29 = foe training, cell 30 = bar chart.
- **Added `fig.savefig`** to bar chart cells 30 and 33.
- **Added `'num_trajectories': 700`** to `approx_params_100`.
- **Added cell 34**: `oe100.error_report()`.

### Interval experiments (v3)
| Intervals | Rel L1 error | L-inf | Epochs |
|-----------|-------------|-------|--------|
| {1, 3, 5, 7, 10} | ~5.7% | — | 3000 |
| {1, 5, 10} | 6.57% | 4,268 | 150 |
| {1, 2, 5, 10} | 1.61% | 2,818 | 1,702 |
| oe100 (N=100) | 0.74% | 467 | 3000+L2 |

## v4 Changes
- Ported all v3 changes (intervals fix, error_report, oe100 cells).
- Kept v4-specific `bounded_coarse_V` and `V_bounds`.
- **Added complete trajectory cost** to `fine_optimal_execution.cost()` (was missing from v4).
- Added `M_comp`, `x_comp`, `N_total`, `w_interval`, `w_comp` to `__init__`.
- Added `num_trajectories_interval` and `num_trajectories_comp` to `approx_params_fine`.
- Fixed `foe.x_data` AttributeError in cell 33 (replaced with random samples).

### v4 results
- foe with `bounded_coarse_V`: 29% error (vs v3's 5.7% same intervals). The hard penalty fallback outside training rectangle distorts optimization.
- oe100: 0.57% error — solid.

## v5 = v3 minus closed-form
- Copied v3 to v5, removed all `solution` class, `self.cf`, `self.full_cf`, `error_report()`, and CF comparison cells.
- Trimmed from 34 to ~19 code cells.

### v5 key changes
- **`alpha` = 0.5** (was 1). Sublinear cost in trade size.
- **`step_cost` fix**: `torch.pow(psi, alpha)` → `torch.pow(torch.abs(psi), alpha) * torch.sign(psi)` — required because `pow(negative, 0.5)` = NaN.
- **Terminal cost** updated to `(D + kappa/2 * R) * sgn(R) * |R|^alpha`.
- **Added cells**: oe bar chart (cell 10), oe100 training (cell 18), foe vs oe100 comparison (cell 19).

### v5 observations (alpha=0.5)
- oe (N=10): trains OK, cost 823 → 578 over 3,000 epochs.
- foe: trains but produces wild oscillating trades (±8000) — sublinear cost incentivizes buy/sell oscillations.
- oe100: NaN during L2 rectangle retraining — off-manifold states explode with fractional alpha.
- Open issue: need to constrain trades or add regularization for fractional alpha.
