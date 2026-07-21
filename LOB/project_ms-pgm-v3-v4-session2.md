---
name: ms-pgm-v3-v4-session2
description: "Session 2026-06-19/21: fixed cost() N_coarse check, bounded_coarse_V experiment (v4), half-rect foe init, liquidation constraint problem identified"
metadata: 
  node_type: memory
  type: project
  originSessionId: df91e152-2ed3-4367-9603-109a800d4115
---

## Key fixes applied to v3 (2026-06-19)

### cost()/_sample_costs() terminal cost logic fix
- Old: checked `if i < last` (index-based) to decide coarse_V vs terminal_cost
- New: checks `if n_int == self.N_coarse` — uses terminal_cost only for the actual final interval N, uses coarse_V for all others
- Without this fix, intervals={1} would use terminal_cost (full liquidation) at t=0.1 instead of coarse_V

### error_report() fix
- Old: compared NN cost-to-go from t_start against full_cf(R) (total cost from t=0) — apples-to-oranges
- New: uses `full_cf.value_function(n, D, R)` for exact cost-to-go comparison
- Test states now sample D from training range (not D=0)

## MS_PGM_v4.ipynb
- Copy of v3 with `bounded_coarse_V`: wraps coarse_V to return `(D + kappa/2 * R) * R` outside the training rectangle
- Uses `torch.where(inside, oe.V(x), penalty)` — differentiable branching
- Result: penalty too aggressive — NN overtrades ~70/step trying to keep R inside rectangle
- Kept as record; v3 does NOT have bounded_coarse_V

## Half-size starting rectangle (v3)
- foe.x now sampled from rectangle with half edge length, centered at midpoint of oe.x_data range
- Did NOT help for interval {1} alone — NN still trades negative, 100% of endpoints outside rectangle at t=0.1
- Works better with multiple intervals {1,3,5,7,10} because subsequent intervals anchor the trajectory

## Results with intervals {1,3,5,7,10} (v3)
- 7.6% error, bar chart shows good match except first trade (NN ~22 vs CF ~255)
- NN total trades: 857 vs CF total: 1000 — NN undertrades by ~143 shares
- oe.trade_size at (t=0, D=0, R=1000) gives psi=256 — so coarse model trades correctly, but fine model doesn't

## Root cause identified: missing liquidation constraint
- In coarse oe, terminal step forces psi=R (liquidate everything). This shapes the entire strategy including the large first trade.
- In foe, each interval trains on independent batches. Non-terminal intervals only see coarse_V — no direct enforcement of full liquidation by T.
- The terminal liquidation at interval 10 is disconnected from interval 1.
- coarse_V should encode "you'll need to liquidate everything eventually" but its gradient in R isn't steep enough to force the correct early trading.

## Proposed direction (not yet implemented)
- Use oe.trade_size to fill in the gaps between fine intervals
- Chain: start from (D_0, R_0), roll out fine interval 1, then apply oe.trade_size through the gap to interval 3's start, continue through all intervals to terminal
- This would build a complete trajectory where terminal liquidation backpropagates through all intervals
- At each coarse step, balance change = oe.trade_size(t_n, D_n, R_n)
- A sequence of these balance changes from any starting (D, R) gives the full trajectory

## oe.V accuracy at t=0.1
- Inside training rectangle: 0.17% error (good)
- At NN endpoint (D=9.96, R=776.94): 0.10% error
- Extrapolation outside rectangle: underestimates cost (same pattern as t=0.9)
- Diagnostic plots saved: v3_oeV_vs_cf_t01_R_sweep.png, v3_oeV_vs_cf_t01_D_sweep.png
