---
name: epistemic-honesty-in-reports
description: "User pushed back on a disclaimer draft asking 'how are you sure about this?' — don't overstate causal/frequency claims in text meant for a document"
metadata:
  type: feedback
---

When drafting a disclaimer/summary paragraph for the user's paper about training divergence, I wrote
a sentence asserting the root cause was confirmed and that a fix "reduced" the divergence frequency.
User asked "How are you sure about this?" — on inspection, neither claim held up:
- The causal mechanism (stochastic κ dynamics + unbounded policy output) was a *reasoned hypothesis*
  with one supporting data point (a before/after fix test on one config), not something verified by
  inspecting the actual diverged runs' internal state (which wasn't retrievable after the fact anyway
  — per-run OU path draws aren't logged, only aggregate stats).
- The "frequency reduced by the mitigation" claim compared n=2 (100% diverged, before) against n=22
  (~9% diverged, after) — too small a "before" sample to claim a frequency, and confounded because
  `num_trajectories` also changed between the two conditions, so the improvement couldn't be
  attributed to the fix alone.

**Why:** Text destined for a document/paper gets cited as fact by the reader; overstating confidence
in a hypothesis (presenting it as confirmed) or in a comparison (presenting a confounded/tiny-sample
result as if it were a controlled finding) misleads downstream readers who won't see the underlying
data.

**How to apply:** Before writing summary/disclaimer prose for a document, separate (a) what was
directly measured/computed from what was (b) inferred/hypothesized, and flag confounds explicitly
(concurrent parameter changes, tiny sample sizes) rather than smoothing them into confident-sounding
prose. When the user asks "how are you sure," treat it as a request to show the actual evidence chain,
not just to reassert the claim more strongly.
