---
name: feedback-prereg-scoring-rigor
description: "Pre-registration must fix the scoring rule (thresholds, relative vs absolute) and the user wants increasing rigour as the model is explored deeper"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1d2cb70c-6b6c-4295-bf8d-ac2725e81676
  modified: 2026-10-08T11:16:04.364Z
---

On 2026-10-08 the user said pre-registration hygiene "must be improved. We need to be increasingly rigorous with these
things as we explore this model deeper", after the paper_scale sweep filed regime counts with one rule (exact s = 0)
and scored with another (thresholds c < 0.02, s~ > 0.98). They also said not to sweep unexplained residuals under the
rug: the notebook had asserted a loss gap was "optimisation, not the bound" without evidence.

**Why:** the project's credibility rests on predictions filed before training; a post-hoc change of measurement rule
or of the quantity reported (relative vs absolute gap) silently weakens that.

**How to apply:** predictions.json must state the measurement rule, every threshold and whether a gap is relative or
absolute; check_numbers.py reads them from the file (rule now in `.claude/CLAUDE.md`). When a result is unexplained,
say what is known (data), what is assumed, and which filed experiment settles it; do not assert a cause. Report
absolute and relative gaps when the reference loss changes across the sweep. See [[feedback-notebooks-plots-not-tables]].
