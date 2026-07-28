---
name: ms-pgm-v7-manifold-2d-styling
description: "MS_PGM_v7 manifold scatter converted from 3D bird's-eye to true 2D, B&W variant added, styling settled; negative-D investigated and ruled out as a bug"
metadata:
  type: project
---

## 3D → 2D conversion (cell id `82e041c9`)
The original 3D scatter (see [[project_ms-pgm-v7-manifold-diagnostics]]) was changed to a bird's-eye
view (`elev=90, azim=-90`) and finally simplified to a **true 2D scatter** (`ax.scatter(D, R, c=t,
cmap=...)`, no `projection='3d'`) since a straight-down 3D view is visually identical to 2D but drags
along unnecessary z-axis machinery (degenerate z-tick labels/title that clutter the figure). `V`/
`y_data` are still computed and saved as `oe.V_traj`/attributes, just no longer plotted.

## Black-and-white duplicate (new cell id `bw_manifold_plot`, right after the color cell)
Same plot, `cmap='Greys'`. Reuses `D_traj`/`R_traj`/`t_traj`/`D_data`/`R_data`/`t_data` from the color
cell rather than recomputing — **must run both cells together** (the bw cell alone will `NameError`).
`t=0` was too dim (near-white) with a raw `Greys` colormap — fixed via a **truncated colormap**
(`mcolors.LinearSegmentedColormap.from_list('Greys_trunc', plt.cm.Greys(np.linspace(0.3, 1.0, 256)))`)
rather than shifting `vmin`, because shifting `vmin` would make the colorbar display a misleading
extended range (e.g. -0.3 to 1.0) with no real data below 0. Truncating the colormap keeps the
colorbar's displayed range matching the actual data range.

## Settled styling (both color and bw cells)
- `sharey=True` in `plt.subplots(1, 2, ...)` — right panel's y-ticks hidden, both share the `R` scale.
  (Note: this is fine here because both panels plot the *same* variable `R`; do NOT do this on the
  wall-clock boxplot, where the two boxes are on genuinely different scales — see
  [[project_ms-pgm-v7-wallclock-logging]].)
- `ax.grid(True, alpha=0.3)` on both panels — user explicitly wants gridlines kept even in the
  minimal/zero-margin versions.
- For LaTeX embedding at max size: `subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)` +
  `savefig(..., pad_inches=0)` — but this is specifically for the axis-label-free bird's-eye variant;
  once real 2D with visible axis labels/ticks, use modest margins instead (`left=0.06, right=0.98,
  top=0.98, bottom=0.12, wspace=0.15`) so labels aren't clipped.
- `D`-axis label was removed then explicitly restored per user request (they'd asked for its removal
  earlier as part of a decluttering pass, then wanted it back — don't assume a prior removal request
  is permanent).

## Negative-D investigation: not a bug, a known pre-existing model property
User saw negative $D$ values in a plot and assumed the cosmetic edits (labels/margins/view angle)
broke something. **Ruled out**: none of those edits touch the code that computes $D$ (`oe.unit()`
rollout, `oe.x_data`). Added a diagnostic print (`D_traj`/`D_data` min/max/negative-fraction) directly
in the plotting cell to quantify this going forward. Root cause: `trade_size`'s output layer is a
plain unbounded `Linear` (no clamp/activation), so $\psi$ can be negative ("buy back" / a "negative
trade"), and $D_{\text{next}} = (D+\kappa\psi)e^{-\rho\delta}$ can go negative if $D$ is small and
$\psi$ is sufficiently negative. This is intermittent across training runs (one run showed 0/2200
negative, a different screenshotted run showed some down to about -2) — matches the pre-existing
"negative trades insight" open item from early sessions ([[project_ms-pgm-open-items]]). Not fixed;
would require clamping $\psi$ or $D$ in `update()` if strict non-negativity is wanted.

## Recurring PyTorch semantics confusion: `torch.clamp(x, min=a)` computes `max(x, a)`
Came up twice in this session. `torch.clamp(x, min=a)` means "`a` is the **lower bound** of the
allowed range" — it raises any `x < a` up to `a`, i.e. it computes $\max(x, a)$, not $\min(x, a)$. User
flagged a line using `min=self.kappa_lower` as looking wrong ("should be max") — it was already
correct; the `min=` keyword names the bound, not an operation. Worth restating this precisely whenever
a `torch.clamp`/`np.clip` line is questioned, since the keyword name is a common source of confusion
distinct from the actual floor/ceiling semantics (see also the same mixup in
[[project_ms-pgm-v7-kappa-lower]]'s original min/max instruction).

**How to apply:** If more plots are added in this style, reuse the truncated-colormap trick for any
sequential colormap where the low end needs to stay visible, and default to `sharey=True` only when
panels share a variable/scale.
