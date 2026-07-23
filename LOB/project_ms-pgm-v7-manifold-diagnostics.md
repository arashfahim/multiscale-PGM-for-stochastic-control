---
name: ms-pgm-v7-manifold-diagnostics
description: "MS_PGM_v7 new diagnostic cells: 3D trajectory-vs-training-data manifold scatter, and L2/Hausdorff consecutive-interval bar chart"
metadata:
  type: project
---

## Stochastic resilience/impact correction
Earlier assumption that only `LOB-Stochastic_Coarse.ipynb` has stochastic resilience/price-impact was
**wrong**. `MS_PGM_v7.ipynb`'s `lob_params` includes `theta_kappa`, `sigma_kappa`, `theta_rho`, `sigma_rho`
driving proper OU processes with `torch.randn` (correct Wiener increments), used throughout
`optimal_execution`/`fine_optimal_execution` via `kappa_paths`/`rho_paths`. State dim is 5:
`(t, D, R, kappa, rho)`.

## 3D manifold scatter (replaced broken cell id `82e041c9`, formerly a V_fat comparison — V_fat no longer exists)
Rolls out `oe`'s trained policy to build the trajectory manifold, saved as new attributes
`oe.traj_data` / `oe.V_traj`. Plots two side-by-side 3D scatters, matched `view_init(elev=25, azim=-60)`:
- Left: trajectory `(D, R, V)` — collapses onto a thin, ~0.997-correlated ridge that slides with `t`.
- Right: training data `(D, R, y_data)` from `oe.x_data`/`oe.y_data` — spreads across the full
  decorrelated rectangle at each time slice (the Level-2 fix's whole point).
Styling settled on: only left panel keeps the z-axis label, only right panel keeps the colorbar
(avoid duplication), no subplot titles, tight margins (`fig.subplots_adjust` + `savefig(pad_inches=0.05)`),
dpi=600 for print quality.

## L2/Hausdorff consecutive-interval bar chart (new cell id `l2hausdorffcoarse`, before "Fine-scale PGM")
Ported from the LQC line of notebooks, but **critical methodology difference**: must use
`oe.x_data`/`V(x_data)` directly — i.e. `V_data[n] = oe.V(x_n)` on the real rectangle-sampled points at
each coarse time step, then `mse = mean((V_data[n] - V_data[n+1])**2)`. Do **not** use a synthetic
grid with fixed κ/ρ (gave a misleading, non-monotone result), and do **not** use trajectory data
(`oe.traj_data`) instead of `x_data` — `x_data` is intentionally decorrelated training data (the
Level-2 fix), so it's the right analogue of LQC's `path`, not the correlated rollout.
Result with real data: L2-difference rises smoothly/near-monotonically toward `t=T` (value function
changes more sharply near liquidation); Hausdorff distance noisier, often spiking at the very first
interval (mixed `D_0=0` initialization) and again near `t=T`.
Bar chart styling: twin y-axis (Hausdorff left/hatched coral, L2 right/steelblue), all tick+axis
labels black (not colored to match bars — user disliked that), legend at `loc='upper center'` if
`'upper right'` collides with a tall bar.

**Why:** Both diagnostics build evidence connecting to the known `coarse_V` extrapolation failure
([[project_coarseV-extrapolation]]) — same underlying (D,R) manifold-mismatch story, now visualized
directly for v7.

**How to apply:** If adding similar diagnostics to future MS_PGM versions, reuse the same data-source
rule (real `x_data`/`V(x_data)`, not synthetic grids or trajectory rollouts) and the same bar-chart
styling conventions.
