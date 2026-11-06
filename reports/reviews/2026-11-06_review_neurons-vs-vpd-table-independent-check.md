# Review: does the answer paragraph of the neurons-against-VPD table follow from its result files? (independent check)

*2026-11-06, baseline-lead independent checker, read-only, $0. Under check: the answer paragraph of
`reports/analyses/2026-11-06_analysis_neurons-vs-vpd-table.md`, its table `experiments/neuron_vpd_table/results/table.json`
and `experiments/neuron_vpd_table/build.py`. Every number below was read with `jq` from the files named.*

I re-read every cell of the table against its source JSON and the filed verdict in the source report. The numbers are
copied correctly and the headline, that neither method yet tells new from existing on Qwen, holds. One sentence of the
answer is false as written, and "both methods alike" leans on a measure choice that the VPD side's own filed rule says
cannot separate the pair.

**Verdict: holds with caveat.** The last sentence needs a scope fix (problem 1). Problems 2 and 3 need a few words each.

## What was checked

- Every number in each `table.json` cell, against the key the cell names. All match: E94 `.comparisons.matched/.full`
  and `table_amend1.json .sides`; E102 `.P1_prompt_overlap_anchor1`, `.P2_engaged_anchor1`, `.per_state`, `score_stageB
  .CB1_power_gate`; E103 `.identity`, `.pair_rules_s142`; E105 `.real/.tofu .verdict`; E107 `.real/.tofu .wrong/.topic`;
  E107 `amend1_sets.json .rule, .pairs`; E109 `.tofu`, `.T`; E98 `.rows.P1`; `method_table/results/table.json`.
- Each verdict string against the source report's own reading. All are copied, not retyped (build.py reads them).
- Each "class" against what the filed verdict says about the pair. Notes below.

## Problems

### 1. "At the learned size neither has a reading" is false for VPD

`experiments/qwen_d3_rows/results/table.json .comparisons.full`: outside share, placement and think overlap all read
"counts" between the template and the fictional-facts fine-tune at the full update, and `size_can_do_this` reads "does
not count" on each, so random changes of the two sizes do not reproduce the gap. The E94 report's outcome table
(line ~290) carries these three as "counts at full, random does not reach it". The table's own row 2 has this cell as
"separates the pair". The body repeats the error: "At the learned size every cell is grey, white or 'does not separate'"
contradicts rows 2 (VPD) and 9 (the 67M, VPD, labelled learned).

VPD also has description at the learned size on known against invented facts: `table_amend1.json .sides.end` reads
"like anchor 2" on outside share (f 0.73) and think overlap (f 0.79), filed as reported only since the states are
2.2 times apart in norm. `qwen_d3_rows/results/set_overlap.json .summary` gives the end-state switched-on sets as well
(known against fictional 0.30 to 0.33, seeds 0.52 and 0.71), uncited and capped near 0.5 by set size (3,757 to 4,103
against 7,805 to 7,987).

**Fix.** "At the learned size neither reads known against invented facts or wrong against true answers: the neuron set
has no power there, VPD's wrong-answer states were not saved, and VPD's known-arm reads there are description beside a
state 2.2 times larger." In the body, replace "At the learned size every cell is grey, white or 'does not separate'"
with "At the learned size only the layout pair separates, on VPD's side."

### 2. "In both methods alike" rests on Jaccard for VPD, which the VPD side's filed rule calls powerless on the neurons' measure

The neuron side's verdict is filed: E103 `.identity` "beyond seed noise" on both row sets, and E105 "label" on both.
The VPD side has no filed rule. The table's cell says so, but the class "separates the pair" makes it look like a
verdict. Three differences matter.

- **The measure.** The neuron sets are all 1,024 wide, so their "share" is shared-over-smaller. On that same measure
  VPD's known-against-fictional overlap reads 0.875 to 0.904 against the known seeds' 0.933. E107 Amendment 1's
  predictions file (`predictions_amend1.json .blind_to`) records this gap as "under its own band (0.067), so that measure
  cannot call". VPD separates known from fictional only on Jaccard.
- **Set size.** The fictional sets are larger: 1,847 and 1,937 against 1,468 and 1,501 (`set_overlap.json .sizes`).
  That caps their Jaccard with the known sets at 0.76 to 0.81, below the seed level, before any difference in content.
  So part of VPD's 0.63 to 0.66 against 0.86 is set size. E107 Amendment 1's report says this itself ("caps its pairs
  with the other arms near 0.8").
- **The rows.** Neurons are ranked on TOFU or real rows. VPD's sets are read on 128 maths rows that no arm trained on.

The sets section's "almost number for number" also does not survive putting both on one measure. As Jaccard, the
neurons' known-against-fictional overlap is 0.44 to 0.47 (share s gives s/(2−s)) against VPD's 0.63 to 0.66. The
neurons' wrong-against-known overlap is 0.72 to 0.78 against VPD's 0.80 to 0.85. The ordering agrees; the numbers do
not.

**Fix.** "Which neurons an update uses separates known from invented facts, by the neurons' filed rule. VPD's
switched-on set does too, as description, on Jaccard only: on the neurons' own measure its gap is inside the seed band,
and part of it is set size." In the sets section, change "almost number for number" to "in the same order".

### 3. "Keeps wrong answers beside true ones" is a filed verdict for neurons and a no call for VPD

The neurons' verdict is filed: E107 `score.json .real.wrong.verdict` and `.tofu.wrong.verdict` both read "topic or
style". VPD's is "no call": `amend1_sets.json .rule` has the lowest pair at 0.80303 against the line at 0.80320. All
four pairs sit far above the familiarity line (0.714) and above every known-against-fictional pair (0.660 at most).
E107 Amendment 1's own answer describes it the same way ("also keeps the wrong-answer run beside the true-answer run").
So the description is fair, but the answer does not say that VPD's side is a no call. It also leaves out the shared
caveat. At the matched step the wrong and true updates differ by about one seed (weight cosine 0.825 against 0.821), so
neither test had much room. And the matched step is a size control only (`decisions.md` row 152), so "it reads the data's
source and style" goes past either verdict. The topic row ("topic moves the set", TOFU rows only) is neuron-side only.

**Fix.** "…yet keeps wrong answers beside true ones: the neurons' filed verdict is 'topic or style', and VPD's is a no
call that missed that line by 0.0002. At this size the two updates differ by about a seed, so this points to the data's
topic or style rather than showing it."

## Smaller points (no change needed to the answer, wording only)

- **67M, VPD cell.** "VPD's r 'separates' on 3 of 3 rows" is quoted as if it were a filed verdict. For e46 and e53,
  `method_table/results/table.json` has `filed_verdict` null. The "separates" there is `derived`, E51's rule applied
  afterwards. For e54_own the filed verdict is P3, a different comparison. Say "separates by E51's rule applied
  afterwards".
- **67M, neuron cell.** The class "does not separate" is faithful. E98 `.rows.P1.status` is "pass" on the prediction
  that the neuron share is "not a kind reader", and the share separates 1 of 3 readable rows (e53). E98's prose also
  counts copy against lookup as separated, but that row is "unreadable" in P1. The cell follows the filed P1.
- **Layout against facts, matched, neuron cell.** None of the quoted verdicts compares the pair. P1 and P2 score the
  template against the random change, and CB1 is the judge's power gate. The class rests on the unscored R2 printout
  (fictional 0.051 and 0.052 against the template's 0.056 and 0.057) and on E102's own reading that the overlap "cannot
  separate anything". That is defensible, but the cell should say the class is read from unscored numbers. At full
  size the same unscored printout reads 0.038 and 0.040 against 0.056, near the random 0.035. That is a difference the
  table leaves out, so "where the neuron readings do not" should say "at matched size".
- **Known against new, learned, VPD cell.** The class is "does not separate", but the source files these readings as
  reported, not as a verdict. Grey or a "description" mark would be more faithful.

## Fixes, as a list

1. Answer, last sentence: "At the learned size neither reads known against invented facts or wrong against true
   answers: the neuron set has no power there, VPD's wrong-answer states were not saved, and VPD's known-arm reads there
   are description beside a state 2.2 times larger." Body: drop "every cell is grey, white or 'does not separate'".
2. Answer: VPD's set separation of known from invented is description on Jaccard only, inside the seed band on the
   neurons' measure, and partly set size. Sets section: "in the same order", not "almost number for number".
3. Answer: name VPD's wrong-against-true verdict as a no call by 0.0002, and soften "so it reads the data's source and
   style" to "points to topic or style", with the one-seed room caveat.
4. "Where the neuron readings do not" becomes "at matched size, where the neuron readings do not".
5. 67M VPD cell: "separates" is derived afterwards, not filed. Known-against-new learned VPD cell: description, not
   "does not separate".

## Not run

- No re-scoring of any source verdict and no new Jaccard computation from the saved sets. The Jaccard conversions
  above use the identity for equal-size sets.
- The figures (`verdicts.png`, `sets.png`) were not opened. Only the builder that draws them was read.
- The source reports were checked only where a cell cites them, not reviewed in full.
