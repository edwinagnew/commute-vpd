# Handoff: the reuse line merges into the baseline line (user, 2026-11-02)

**For baseline-lead, which continues as the single benchmark lead. Plan only: nothing here has run.** The user judged that the two
lines overlap too much, and chose one lead over sharper boundaries or a pause. The merged line asks one question: does VPD tell
reuse from new better than cheaper methods do, judged against a truth it did not define? The reuse-lead stops; its files stay.

## Order of work

1. **Finish what is in hand first:** E63 (neuron sparsity), then E64 (the 64-value readout), by your own handoff's tree.
2. **One table.** Every method against every fine-tune, one results file and one log section, each row marked with how its truth was
   set: by construction, by causal check, or only by assumption. Merge the E52, E54, E56 and E59 readings with E55's C1, C1b and C3.
   Write it from the existing results; it needs no run.
3. **Then the harder counting test**, through `/design` and shown to the user before code (below).
4. **VPD on the counting set stays off** unless that design shows it can tell us something.

The re-gating lead stays separate. It asks what VPD's reading means; its numbers enter the table as VPD's row, and comparisons
with other methods are this line's.

## What this line inherits

- **Definition.** Jain, Kirk, Lubana et al. 2024 section 3 plus the user's causal requirement (`decisions.md` row "Reuse line"; the brief
  `~/commute-data/agents/reuse-lead/brief.md`). Since 2026-11-02 elicitation is mechanistic: the causal check is the label and the
  held-out reading is evidence (`decisions.md`, "Held-out rule"). No counting-set label changes: they come from the fine-tuning task,
  plus the behaviour and patch checks each state passed.
- **Truth set and methods.** E55 C1b: `reports/analyses/2026-10-31_analysis_e55-c1b.md`, `experiments/pcfg_anchors/c1b.py`,
  `src/commute/revival.py`. Pruning revival, reverse fine-tuning and steering each label all 21 states right.
- **67M work already done by reuse.** E55 C3 and its ladder: `reports/plans/2026-10-31_plan_e55-c3-ladder.md`, `TRIED.md`. As ported to the 67M, pruning revival and reverse fine-tuning failed their positive control, so the 21 of 21 does not carry over; the table must show that.
- **Stage B**, the interchange patch on E58's states, was never planned in detail: `reports/plans/2026-10-29_plan_reuse-stage1.md` § Stage B.
  It is the mechanistic label the 67M rows lack, so it is the first candidate after the harder test.

## Why the VPD run stopped

On that truth set "reuses" and "new" are the same as task P and task N, so anything that responds to the task scores perfectly.
A decomposition trained for 400 of its 400,000 steps labelled all 21 right (job 48413): P states read near 0, N near −0.05, against
0.77 and 0.24 on the 67M; the identity-prompt control read about 0.07 against its bar of 0.5. A test on four states (job 48441) found the
gap follows the fine-tune, not the answer token, against Fable's expectation; the cause is not established. Report:
`reports/decisions/2026-11-02_decision_e55-c2-untrained-tie.md`. Outputs: `~/commute-data/scratch/reuse-lead/c2-fixes/readcheck_A/`,
`~/commute-data/scratch/reuse-lead/swapcheck/`.

## The harder test, as the idea stands (not designed)

Train the same task on two base models, one that already has the count and one built without it. Reuse then varies while the task
stays fixed, so reading the task no longer gets the answer. The truth still comes from the causal check. Run the cheap methods first.

Every method must beat a shuffled-label version, and any VPD reading must beat a barely trained decomposition. Open points: whether a base without the count can match the other on
accuracy elsewhere; a decomposition per base model (about 56 L40-hours each); whether cheap methods leave VPD anything to win. It does
not guarantee that methods will differ. The user's complaint to keep in view: a test every method passes tells you nothing.

## If C2's code is ever used

`experiments/pcfg_anchors/c2.py`, `c2_rehearse.py`, `src/commute/decompose.py`, `predictions_c2.json` are committed, with reviews in
`reports/reviews/2026-11-02_review_e55-c2-gates.md`; the plan is `reports/plans/2026-10-31_plan_e55-c2-vpd.md`. The input of the importance function is
cut at `=` in the scored cells. The read and score stages ran on a GPU; a decomposition past 2,000 steps never has. Before any long run:
- Apply Fable's amendment, in `reports/plans/2026-11-02_advice_c2-untrained-tie-FABLE.md`: a tie counts only if the control holds.
- `early_checks` at 100K and later: `ce_kl/kl_ci_masked` becomes a watch (the noise job's final-p eval read 0.77); `kl_unmasked` stays a stop.
- `decompose.save` refuses non-finite parameters: the loss check misses a finite loss with infinite gradients.
- `readcheck`: assert the stored `final_step` equals the loaded step, and store the file's path before the first read.
- Copy `cut_move_stop` into the cut-offs file and its status text.

## Not run

The full decomposition; any reading on a trained decomposition; the harder test; the one table; Stage B's patch.
