---
name: notebook-editing-method
description: "How to edit/run large MS_PGM/LQC .ipynb files that exceed the Read tool's token limit"
metadata:
  type: feedback
---

These notebooks (especially once they carry embedded PNG outputs) routinely exceed the Read tool's
~25,000-token limit — and it errors even with `offset`/`limit`, since the whole file appears to be
tokenized before slicing. This means the `NotebookEdit` tool (which requires a prior `Read` of the
same file) can't be used on them.

**Working method instead:**
1. Edit cell source directly via a small Python script: `json.load` the notebook, find the target
   cell by its stable `id`, replace `cell["source"]` (as a list of lines each ending in `\n`, last
   line without), clear `cell["outputs"]` and set `cell["execution_count"] = None`, then
   `json.dump(..., indent=1)` back over the original file (matches the file's existing indent style).
2. Execute via `jupyter nbconvert --to notebook --execute --inplace <path>` (add
   `--ExecutePreprocessor.timeout=<seconds>` for long-running cells), run in the background and wait
   for the harness's completion notification rather than polling.
3. To test just a subset of cells cheaply (e.g. "first N code cells"): build a temporary notebook
   containing `cells[0:k]`, execute *that* file, then merge its cell outputs back into the real
   notebook by matching cell `id` (leave all other cells untouched). Only merge back into the main
   file once the result looks correct — keep intermediate test runs in scratch copies first.
4. Figures saved via relative `plt.savefig(...)` inside a temp-copy execution land in the *temp
   file's* working directory, not the real notebook's directory — move them over explicitly after.

**Why:** Avoids fighting the Read/NotebookEdit token limit, and keeps large training runs
(which can take 1-3 min each) safely backgrounded instead of blocking the conversation.
