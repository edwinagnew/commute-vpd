# P2.68 CMAP spec: GPT plan review (model-router plan-reviewer), 2026-11-23

This review covers `reports/plans/2026-11-23_plan_p2-68-cmap-spec.md` at 0fbdf97b. The reviewer read the paper text and
the authors' code (Nix07/finetuning@6d6c098) and did not run them. Verdict: **needs revision**. The fine-tune-to-base
swap matches the released CMAP kernel. But the plan would discover a different path than the one it names, and its
controls would not detect that. The findings follow, as the reviewer gave them.

1. **[P1] Group C does not test the stated path.**
   - `apply_pp` (experiment_1/path_patching.py:84-122) passes one `rel_pos` to both hooks, so
     `patching_receiver_heads` (pp_utils.py:448-461) edits B's V at the marker, not at B's read position.
   - When the negative-score filter leaves no A or B receivers, the receiver pass is skipped and the test quietly
     becomes direct-to-logit.
   - A key replaced after the marker cannot change the marker's activation. C's noise therefore comes only from
     switching the context.
   - Fixes:
     - File the shared offset as a paper/code discrepancy, and choose between reproducing it and separating the
       sender and receiver offsets.
     - State what C's noise changes.
     - Skip any receiver-dependent group that has no receivers.
     - Add a known-answer two-layer path test that tells the two receiver positions apart.
2. **[P1] Whole-group CMAP is an adaptation.**
   - The notebook (cells 8-9, 18-20, 29-31) uses the released object and positional DCM masks for groups A to C, and
     the full circuit JSON only for D. FLoat's B is 13 heads, not 29.
   - The FLoat cells are incompletely executed and their scores sit in Markdown. The `+=` mutates the A list in
     place.
   - Fixes:
     - Label our version "whole-group head CMAP".
     - Any check in the paper's own setting pins the checkpoint, tokenizer, prompts and masks, uses fresh immutable
       arm sets, and compares raw counts against a clean run of the authors' kernel.
     - A recovered accuracy alone does not show base-model use of those heads: the paper established that separately
       in sections 4-5.
     - Own-circuit faithfulness was 0.99 and 1.0, so the 0.88-1.0 range mixes cross-model and own-circuit numbers.
3. **[P1] The minimality rule is not yet a runnable selector.**
   - The released driver (experiment_1/minimality.py:42) defaults to 100 prompts, not the paper's 500.
   - Its helpers (pp_utils.py:940...) skip distinct pairs that share a head index, discard entries with zero
     accuracy, and rank only positive drops. Accuracy is rounded to 2 decimals before the division.
   - Fixes:
     - A paper-first rule: distinct head-position nodes, explicit ranking of positive drops, rounding of the cap on
       K, exact counts, and rules for empty groups, singleton groups and zero denominators.
     - A decision on heads that sit in both A and B at one position.
     - Pinned sizes, and a stated corrupt-prompt source for the per-position means.
     - A matched random mean-ablation reference.
4. **[P1] The positive control can pass while the pipeline is wrong.**
   - A log-odds gain does not imply an argmax flip, so the accuracy share can be 0/0.
   - A do-nothing run passes even with empty sets or hooks that do nothing.
   - Patching the planted head by hand checks only the final swap.
   - Fixes:
     - Set the no-op tolerance on the einsum `o_proj` hook path.
     - Require non-empty sites, and a positive top-1 gain above the floor before demanding a share of 1.
     - Check the known head and path through discovery and patching.
     - Add a wrong-position or wrong-head check.
     - In the control that changes only the output weights, require a detectable gain in the unit being scored.
     - Allow zero, negative and overshooting recovery.
     - File the floor rule.
5. **[P1] Two pairs' scored inputs are undecided.**
   - E53 (lowercase) uses `NEXT` and E62 uses `LAST` (experiments/share_untrained/windows.py:41-56). Labels must come
     from the original window, before truncation or corruption (src/commute/data.py:73-88).
   - Fixes:
     - Freeze the mapping per format (clean, noise, sender sites) and the seeded row lists.
     - Report A/B-only as a named departure rather than borrowing an unrelated circuit.
     - Freeze a manifest of states, tasks and scales, and the split between discovery, pruning and scoring. E237's
       `random_<state>` is a generated item, not a saved update (experiments/key_grid/run.py:123-139).
6. **[P2] The cost estimate does not bound the sweep.**
   - About 44,100 forward passes come before pruning, and one 24-head group can add up to 62,400 more.
   - Fixes:
     - Time each stage at representative group sizes, and set a stop rule.
     - The pinned tree has no licence file: prefer a pinned, isolated checkout with an adapter over committing
       verbatim copies.

**Neuron wording.**
- Whole-width r_n (src/commute/neuron_r.py:68-144) is a same-input swap from the fine-tune into the base, before
  `down_proj`. "CMAP at neurons (all MLP neurons, all positions; log-odds share)" is accurate as an analogy at the
  level of the operation.
- Top-neuron masking (masked_reads.py:158-168) is not an activation swap.
- The locator aside needs correcting. Its whole MLPs swap the output after `down_proj`, so the fine-tuned output
  weights are included. It returns the log-odds and the label gap.

## Re-check 1 (2026-11-23)

The reviewer re-checked revision 2 (17879402). Two blockers remain, and the rest is resolved or nearly so. Its points,
as relayed by the lead, each answered in revision 3 of the spec:

1. **Edit site.** The reviewer agrees that B's V at the marker is the right site. The scored A/B/C arms should call the
   authors' `apply_pp` unchanged. The alternate offset is used only as a negative-control intervention, and the offsets
   are not a departure.
2. **[Blocker] Two-layer control.** "Within leak × marker score" is not a valid rule. Use instead a fixed, non-zero
   sender cache delta at the marker only, with every other entry identical to clean.
   - After pass one, trace the receiver's `v_proj` output: its marker value must change, and its read-position value
     must stay at a matched no-op floor.
   - Then confirm that the marker replacement changes the receiver's read-position output, and that the read-site
     replacement is null. No first-pass leak allowance is needed.
   - Tolerances: a logit tolerance for logits, exact paired correct counts for accuracy, and a same-hook log-odds
     tolerance for log-odds.
   - Make §3's shared top-1 gate agree with Appendix D's log-odds floor. Define the first resolvable non-zero accuracy
     gain for when the random null is exactly 0.
3. **[Blocker] E62.** The window is "A B … A M", with the second key at `half − 1` and the marker at the read
   (`src/commute/copying.py:249-259`). Its saved file holds 800 train rows and targets
   (`experiments/induction_cue/run.py:98-107`).
   - Include the preceding-key site as group C's analogue, or justify leaving it out; the reviewer prefers including
     it if it maps cleanly.
   - Use the saved train rows for discovery and pruning.
   - For E53, use the original scorer's separate train windows (`experiments/performed_anchor/run.py:314`). Failing
     that, split by source row and give the reason.
4. **Budget.** Add a hard spend cutoff of $20 in total (provider time plus setup, transfer and teardown), checked before
   any paid shard launches.
5. **Planted-head recipe.** Pin it from E58's predictions file.
