# Review: independent check of the E58 Amendment 7 write-up (commit 0676ff3)

*2026-10-30, fresh Opus checker that wrote neither the code nor the analysis. Scope: the Amendment 7 section and opening
answer of `reports/analyses/2026-10-30_analysis_e58-open-pair.md`, the changed parts of
`reports/analyses/2026-10-30_analysis_regating-synthesis.md`, the A7 cells of `notebooks/E58_open_pair.ipynb`,
`experiments/open_pair/plot_amend7.py` with `figures/a7_r.png`, the TRIED row and the PENDING line. Checked against
`results/amend7.json`, `results/predictions_amend7.json`, `amend3.json` to `amend6.json` (step-30 carry), the log
`cluster/logs/commute-e58-open-pair-46673.out`, the plan and the code review. Read-only apart from this file.*

## Verdict: needs fixes, all in wording. The numbers and the scoring are right.

Every number I checked matches the data. Each filed rule is scored as filed. The predictions were not edited after the
run. The plain answer, that r read this carrying new-target rule like a lookup and so does not read carry, stands.
Four wording problems overstate it or drop scope. The class map is said to carry "as far as copy", which is false on a
common measure. The granularity limit is missing from the E58 opening. The limit is narrowed after the fact. The
step-11 disagreement between the behaviour label and r is reported but not read.

## What checks out

- **Predictions untouched after the run.** `predictions_amend7.json` was committed in `af9384b` (18:49:54) and amended
  after the code review in `adf2271` (19:01:11). The job's host line is 19:02:29. There are no later commits to the
  file. Its sha256 `603ba179…` equals `amend7.json` `predictions_sha256`, and the run's `git_head` is `9c47c08` with a
  clean status.
- **Scoring as filed.** T: twin $x - x_\text{null}$ 0.0927 / 0.0829 / 0.0906, all under 0.15, so **clean**. CL: three
  labellable seeds, all over 0.3 (0.6366 / 0.6232 / 0.5548), r 0.2309 / 0.2411 / 0.2541, all under 0.5, so
  **cipher_band**. The pool is `cipher_pool`, so the result is scored. D: 0.24205 − 0.27652 = −0.0345, which is inside
  ±0.0904, so **kind_only**. EP: skipped, 46 keys had no admissible word, so it is not scored. near_cut: the largest
  counted r is 0.279, so false. At each run's gate step, twin seed 0 reads 0.1540, over 0.15, so T "carries" and CL
  reads control_failed, as the report says.
- **Numbers in the report, synthesis, notebook, TRIED and PENDING.** These all match `amend7.json`:
  - priors 2.83e-7 and 3.34e-7; class counts 29/30/19/18 and 27/36/19/14; held-out 4/18/9/1; twin agreement 39;
  - harness r 0.324126 matches bitwise;
  - gate steps 11/11/11 and 16/16/19;
  - $G_\text{spec}$ 17.95 / 19.07 / 17.86 and 13.90 / 13.38 / 12.27;
  - at the gate step, $x - x_\text{null}$ 0.841 / 0.817 / 0.933 and r 0.3154 / 0.3173 / 0.3172;
  - at step 100, 0.552 / 0.492 / 0.500 and 0.059 / 0.064 / 0.055;
  - every per-class own and other gain in the class table; own-minus-other gaps 6.90 to 14.31 for the class map and
    −1.78 to 2.80 for the twin;
  - amplitude-normalised readings 1.639 to 1.918 and 0.190 to 0.366;
  - references from the earlier amendments: cipher, prior-matched cipher and copy r, and the copy and cipher $x$.
- **The post-hoc rank correlations.** Recomputed by hand over the eight seed means: Spearman 0.952 on log prior (four
  one-place swaps) and 0.286 on carry. Both match the report's 0.95 and 0.29.
- **Figure.** `a7_r.png` plots what the script and the JSON say. The right panel's axis states that it mixes
  $x - x_\text{null}$ for the class map and twin with $x$ for the rest.
- **Run length.** About 11 min (19:02:29 to 19:13:50) against the filed 15 to 19. No block-0 stop fired; block-0 reads
  are about −3e-8.
- **Caveats the brief named.** Each is stated where it is needed: the uneven held-out classes, the η² of norm and token
  id (measured on all-key PCs, flagged as such), the twin carrying at the gate step, and the skipped elicit-prior cipher.

## Errors and overclaims, with the fix

1. **"As far as copy" is false on a common measure.** The class map's carry is $x - x_\text{null}$, 0.55 to 0.64. The
   plan itself says that for a perfectly learned rule this is about 4/3 of $x$. Copy's 0.79 to 0.81 is plain $x$,
   because copy has no null. On the same measure, $x$ at step 30, the class map reads **0.490 / 0.453 / 0.400**, about
   0.45 against copy's 0.80. So it carries a little over half as far as copy. And $x$ of 0.45 means unseen keys gained
   under half of what trained keys gained, not "most of the way". The wording appears at:
   - E58 report :23-24 ("carried most of the way", "Copy, which carries as far"), :960 ("nearly as far as copy's
     0.8") and :1081 ("about as far");
   - synthesis :22 and :126 ("about as far") and :162 ("most of the way");
   - notebook headline cell and A7 reading cell ("nearly as far", "copy, which carries as far").

   Fix: "carried 0.55 to 0.64 beyond its null; on plain $x$ 0.40 to 0.49, about half of copy's 0.8". The claim does not
   need copy to match, because D compares like with like (class map against twin).
2. **The granularity limit is missing from the E58 opening.** The new opening paragraph (E58 report :13-18), "The
   answer" (:22-26) and the notebook headline all dropped the granularity-limit sentence that the previous version
   carried. The scope is also missing from the plain paragraph: one rule, one feature, one model, one decomposition.
   The synthesis opening keeps both. Restore one sentence in each place. It should say that the rule is built from
   directions W₀ already has, that r read it as new, and that this does not show r would read every new circuit of old
   pieces as new.
3. **The limit is narrowed after the fact.** The filed `granularity` field calls a carrying class map "a new circuit
   from old pieces, which r may read as elicited", with no condition on the output. The write-up now says the limit
   "did not bite here" and "still applies to a new circuit of old subcomponents whose output W₀ already gives". The
   places are:
   - E58 report :1025-1026 and :1083-1085;
   - synthesis :23-26, :135-138 and :168-170;
   - TRIED row and PENDING line.

   The "whose output W₀ already gives" condition is a new hypothesis, not a finding. The report's own blind-to list
   (:1010-1011) says it was not measured whether the fine-tune uses the embedding directions at all. Training onto four
   words W₀ never gives after these keys also needs new output paths, so the update may not be "old pieces" in VPD's
   subcomponents. Fix: state that the data cannot tell (a) this update did not run through old subcomponents from
   (b) the limit as worded does not hold here. Name "a new rule onto answers W₀ already gives" as the next test, not as
   where the limit is known to bite.

   What the result licenses: on the 67M, r did not call one carrying rule over an existing input feature "elicited".
   What it does not license: "r detects new mechanisms", "the limit is lifted", or anything about rules whose
   mechanism is shown to reuse W₀'s subcomponents.
4. **The step-11 disagreement is reported but not read.** At each run's gate step (step 11), E58's behaviour label
   reads the class map "reuses" on all three seeds (x 0.85 / 0.68 / 0.86) while r reads 0.315 to 0.317. The write-up
   mentions this only as a fact (:998-999) and as a surprise (:1123-1125).

   This does not change the plain answer. The label is a carry threshold ($x \ge 0.5$), so the disagreement is the same
   finding seen from the label's side: carry and r come apart. It does weaken the label as a reference for elicitation
   whenever the target is new. The earlier sections used the label as a positive control and an agreement reference
   (the copy "reuses" control, branch 5's agreement test, ask #14's reading A). The write-up should say so in one or
   two sentences. It should also note that under reading A ("built from subcomponents W₀ already has") the label's
   "reuses" and r's "new" cannot both be right, and which one is right turns on the unmeasured mechanism of item 3.

   Informally, D at the gate step would read 0.3166 − 0.2969 = +0.020, inside the bound. So the choice of step does not
   flip kind_only, though by the filed rule it is not read there.
5. **Small corrections.**
   - Notebook A7 table ("0.56 to 0.64") and commit message: seed 2 reads 0.5548, so the range is **0.55** to 0.64.
   - E58 report :1036: "the band claim's prior range stays at two levels, 1.4e-6 to 3.7e-5". The twin is a no-rule map
     at 3.3e-7, so the no-rule range now runs 3.3e-7 to 3.7e-5. Either say so, or say that the twin is excluded and why.
   - E58 report :1016-1017, "the per-class table shows every class carrying": class 3 rests on 1 held-out key (15
     windows) and class 0 on 4 keys. Say that.
   - The Spearman on carry (:967, synthesis :127) mixes $x - x_\text{null}$ for two maps with $x$ for six. Using $x$
     for the class map and twin leaves the ranks unchanged, so 0.29 stands, but the text should name the mixed axis as
     the figure does.
   - Not fixable, noted only: `predictions_amend7.json` → `predictions.T.blind_to` says "41 of 128 keys keep their
     class". This is stale from before the refile; `settings.twin` and the run both give 39. The report correctly uses
     39.

## Not run

- No recomputation from the per-window tensors in `results-large/open_pair/amend7/`; checks were against JSON summaries
  only.
- No check of the code (`amend7.py`, `open_pair.py`) beyond what the code review covered.
- No η² on the trained-key classes, and no probe of which key feature the fine-tune uses.
- The notebook was not executed. `check.py` was run on both reports and flagged style only, with no retired terms.
