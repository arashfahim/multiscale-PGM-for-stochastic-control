---
name: ms-pgm-v7-kappa-lower
description: "MS_PGM_v7 fix in progress for the M_comp=20 divergence: kappa_lower floor in step_cost (OU kept, CIR considered and rejected for now)"
metadata:
  type: project
---

## Decision: stay with OU, not CIR, for kappa/rho dynamics
Discussed switching κ/ρ dynamics to CIR (`dκ = θ(μ-κ)dt + σ√κ dW`), which naturally prevents κ from
going negative when the Feller condition `2θμ ≥ σ²` holds. Checked with current params: κ satisfies
Feller by ~125x (`2·1·0.05=0.1` vs `σ²=0.0008`), ρ by ~8x (`2·1·2=4` vs `σ²=0.49`) — both comfortably
non-degenerate. But: keeping `sigma_kappa=0.0283` unchanged under CIR would shrink the effective vol
near the mean by ~4.5x (`σ√κ` vs flat `σ`), and CIR still needs a truncated/floored Euler scheme in
discrete time (continuous-time non-negativity doesn't survive naive discretization). **User chose to
stay with OU** and fix the problem via a cost-function floor instead (see below).

## Fix: kappa_lower floor in the cost function (in progress)
Added `'kappa_lower': 0.01` to `lob_params` (representing $\underline\kappa$). In both
`optimal_execution.step_cost` and `fine_optimal_execution.step_cost`: `kap = torch.clamp(x[:,3],
min=self.kappa_lower)`, i.e. the cost uses $\max(\kappa_t, \underline\kappa)$ instead of raw $\kappa_t$.
`self.kappa_lower` is wired from `lob_params['kappa_lower']` in `oe.__init__`, inherited via
`oe.kappa_lower` in `foe.__init__`. `terminal_cost` in both classes calls `step_cost`, so it inherits
the fix automatically.

**Scope deliberately limited so far**: only the cost functions were changed. The `D` update formula in
both classes' `update()` methods still uses raw (unclamped) κ — a likely next step, not yet done.

**Conceptual grounding**: this isn't just a numerical patch — LQ-type stochastic control problems
typically need the running cost's control-quadratic coefficient uniformly bounded away from 0
(coercivity) for standard well-posedness/verification results. Letting κ float arbitrarily close to 0
(even if only rarely) technically violates that; the floor makes the coercivity assumption actually
hold.

**Naming gotcha worth remembering**: user initially wrote "make it `min(kappa, kappa_lower)`" but
meant `max` (confirmed via clarifying question) — `min` would have capped κ at 0.01 from above,
permanently *in* the weak-penalty regime, the opposite of the intent. The parameter's own name
("kappa_**lower**", a lower bound) and the stated goal (prevent κ from getting too small) both
signaled `max` was correct despite the literal wording — worth double-checking floor/ceiling
directionality against stated intent whenever phrasing and semantics seem to conflict.

## Status: mid-test when session ended
Was running the full notebook with the exact `M_comp=20` config that previously diverged (loss →
2.2e22 at epoch 3,000), to check whether the `kappa_lower` fix resolves it. Check notebook state /
rerun if picking this up — result not yet confirmed as of this memory being written.

**How to apply:** If the fix doesn't fully resolve `M_comp=20` divergence, the deferred `D`-update
substitution (`D = (x[:,1] + kap_eff*psi)*exp(-rho*delta)` with `kap_eff = max(kap, kappa_lower)`) is
the next thing to try, since it removes the same weak-coefficient pathway from the state dynamics too,
not just the cost.
