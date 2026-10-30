# E55 Stage C1 and Amendment 5b: pre-run code review, record (2026-10-30)

Scope: `experiments/pcfg_anchors/c1.py`, `amend5b.py`, `src/commute/revival.py`, `predictions_c1.json`,
`predictions_amend5b.json` and the `amend5b` and `c1` stages of `cluster/pcfg_anchors.sbatch`, as filed at 97b7183.
The review ran before any job. Its fixes were committed at 8de9324, before jobs 46105 (Amendment 5b), 46106 (C1
controls), 46107 and 46121 (C1 states). The lead's decisions on its findings are the second entry of
`predictions_c1.json` → `lead_decisions`, dated 2026-10-30 and written before any run. This file records the review
after the fact; the analysis is `reports/analyses/2026-10-30_analysis_e55-c1.md`.

## Findings and what was done

| # | finding | done |
|---|---|---|
| 1 | The sbatch argument check listed only the older stages, so `amend5b`, `c1` and `c1 controls` would each have exited 2 at the start of the job: every new stage was blocked. | Fixed at 8de9324: `amend5b 0`, `c1 0` and `c1 1` with `controls` accepted. |
| 2 | `c1.truth_from` sent an Amendment 5 or 5b state whose class was neither, with count a "not through", to "new". The filed rule, edge case E11 and the first lead decision, places these states by class alone: reuses to existing, new to new, anything else to class B. | Fixed at 8de9324: class alone. Lead decision (a). In the run it moved no state: every Amendment 5 and 5b state of class neither had count a "neither", not "not through". |
| 3 | R in steering's per-run score: `constants_sources` said R votes with its 1,000-step state, `definitions.per_run` with its latest labelled save, 10,000 steps. | Lead decision (b): `definitions.per_run` governs, R votes at 10,000 steps; the other line is superseded. No code change; the code already followed `per_run`. |
| 4 | The usage lines gave `amend5b` and `c1 controls` a 15-minute limit, under the code's own 22-minute deadline. | Dropped at 8de9324: both use the script's 30-minute limit. |
| 5 | The pruning control is near-vacuous. LOW answers 0.99975 of rows and random removals do not move it, so 3 standard deviations is about 3 rows of 4,000; any removal of a few important units passes. | Lead decision (d): rule unchanged; the drop sizes are logged and reported beside the pass flag. They were 229 and 816 rows at K of 5 and 15 against a bar of 3. |
| 6 | Steering on R: HIGH already counts b, so R's gain $G$ in log-probability may be near zero, making $S$ a ratio of noise. | Lead decision (c): rule unchanged; a vote with $G$ under 0.01 is flagged unreadable in the analysis. R read 0.0006 and 0.0022; the 10,000-step vote decides the filed steering miss on every state, and the analysis says so as a post-hoc reading. |
| 7 | The sbatch comment still described the draft's 25 states, at most 83 minutes and up to 4 jobs. | Fixed at 8de9324: 36 states with reverse fine-tuning, at most 119 minutes, up to 6 jobs of 22 minutes. |
| 8 | `amend5.py` was edited at 97b7183, after Amendment 5 had run: two comments changed "fresh" rows to "unseen, Amendment 2's streams", as the Amendment 5 review asked. | Accepted: comment-only, no code line changed. Amendment 5b imports `amend5`, so it ran on the same code that produced `amend5.json`. |

## After the review: found in the run

| # | finding | done |
|---|---|---|
| 9 | The self-resubmit never worked. On exit 3 the sbatch script called `sbatch` from the compute node, which has none: job 46107 hit its 22-minute deadline after 31 states and logged `sbatch: command not found`. The same code sat in every stage that resumes, so no earlier self-resubmit could have worked either. | Fixed at 1c4cc3e: on exit 3 the job prints the resubmit command and exits 3. The lead resubmitted by hand as job 46121, which read the last 7 states. The partial `c1.json` was committed at 1c4cc3e; readings are keyed by recipe, so job 46121 resumed from it unchanged. |

No finding changed a measurement after any model was read. Findings 5 and 6 left the rules as filed and are reported
in the analysis, not re-scored.
