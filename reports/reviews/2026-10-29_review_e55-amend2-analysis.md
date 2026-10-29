# Review: E55 Amendment 2 analysis (independent check)

*2026-10-29. Reviewer did not write the report, the code or the run. Scope: the "Amendment 2" section of
`reports/analyses/2026-10-29_analysis_e55-stage-a.md` and the lines it changed above it (opening, answer, What next,
Not run), and the matching cells of `notebooks/E55_pcfg_anchors.ipynb`. Checked against
`experiments/pcfg_anchors/results/predictions_amend2.json` (`scored_from`), `results/amend2.json` (job 44520) and
the per-row files in `results/large/amend2/`. Check scripts, run through `fleet/cpu-run`, no model loaded:
`~/commute-data/scratch/reuse-lead/e55-a2-check/check.py`, `check2.py`, `check3.py`.*

## Verdict

**The scorecard stands, but the void diagnosis does not.** Every T and U verdict re-scores the same from
`amend2.json` under the filed rules. Every quoted number matches. When the saved answers are run back through
`amend2.read`, they reproduce `amend2.json` exactly (largest difference 0). The report says the unseen read was void
because "the base model cannot count a that far out", and that the cause is "a limit of the model, not a mismatch in
the construction". The saved answers do not support that. The construction matches its spec: the rebuild matched
every row. But both control models fail more the larger the a/c swap was, and that includes the count of b, which
the swap never touches and which stays in range. So the main cause seen is the swap-built rows themselves. A
counting-range limit is not separated from it. Two smaller gaps: the only fresh evidence in new cells (2 rows) goes
against the rule and is not reported, and the claim that "a corrected unseen set looks hard to build" is overstated
(it takes CPU minutes). Fix these three before the third-class decision is put to the user.

## Re-scoring (task item 1, 2)

| id | `scored_from` read | filed rule | verdict | report |
|---|---|---|---|---|
| T0 | `controls.checks`: 38 of 38 true; N1 overall_b 0.99995, S_b 0.99992; LOW overall_a 0.9999 | all true; ≥ 0.95 | met | met |
| W0 | LOW route a; third overall 0.3656, S 0, "does not follow"; other 0; c_best 0.4042, gaps −0.0386, −0.0948 "not shown"; N1 route b, gap −0.2175 "not shown" | expected values | as expected | as expected |
| T1 | W1, W1s1 10K: S_third 0.99087 > hi 0.66895, "follows", route third, overall 0.9766, 0.9809 | S > hi; overall ≥ 0.95 | met | met |
| T1b | `third_vs` labels: follows ×4, both seeds | all four | met | met |
| T1c | gaps 0.2093, 0.20975 | > 0.05 | met | met |
| T2 | W2 10K: S_third 0, "does not follow"; overall 0.8491; route mid; mid gap 0.10155 "beats" | < 0.1; < 0.95; mid; > 0.05 | met | met |
| T3 | W1, W1s1 1K: third "neither", "neither"; route a_plus_1 ×2; S 0.2146, 0.1872 | not follows; route | met | met |
| T4 | W2 1K: S 0.9817 "follows", route third, overall 0.89655 | > 0.669; route; < 0.95 | met | met |
| U0 | 16 model-free `unseen_*` checks true; `unseen_valid.rows` 0 of 8,949 | ≥ 100 rows kept | missed (model part) | missed |
| U1–U3 | key `unseen` absent (E18) | — | void | void |

The trajectory table (line 721), the per-state table (lines 705–712) and the cell claims (line 779: 12 cells, the top
four hold 190 rows, each at 100%) all match. The U0 cell counts match too: LOW's count-a share is 0 in all 55 cells,
and N1's count-b share is ≥ 0.95 in 3.

## Findings, most serious first

### 1. The void diagnosis names the wrong main cause (lines 681, 746–760, 770–771, 789, 798; notebook cells at 352–354, 364)

**Claim.** "The base model does not count a outside the range it was trained on … This is a limit of the model, not a
mismatch in the construction or the scoring … LOW's answers move with the count of a in the right direction, and are
clipped." What next 2 adds that if LOW fails in range, the rule "cannot be tested on this model".

**Evidence** (`check.py`, `check2.py`; the pre-swap count of a was rebuilt from the filed seed, and the rebuilt rows
match `unseen_counts.npy` on every row):
- **N1 fails in proportion to the swap, on a letter the swap never touches.** At fixed b = 13, N1's exact count-b
  share is 0.82 at a swap of 2, 0.53 at 4, 0.43 at 5, 0.19 at 8. The same holds at b = 12 and b = 14. Over all rows:
  1.00 at a swap of 1 (9 rows), 0.87 at 2, 0.58 at 4, 0.14 at 8, 0.04 at 10. On fresh natural rows N1 is at 1.000
  for every count of c, 9 to 18. A count-range limit predicts no failure here. A swap artefact predicts this pattern.
- **Above 17, LOW tracks the count of a before the swap, not after it.** Within (a, b) cells with a ≥ 17, the slope of
  LOW's answer on the pre-swap count is 0.85. In cell (20, 13), LOW's mean answer goes from 9.0 to 16.75 as the
  pre-swap count goes from 11 to 16. In (18, 13) it goes from 13.7 to 17.0. Fitted over all number answers, LOW's
  answer ≈ 0.165 × (count a after) + 0.50 × (count a before) − 0.62 × b + 10.8.
- **The (17, 13) cell.** All three counts there are in their natural ranges (a 17, b 13, c 10). LOW is wrong on all
  341 rows, and its answer rises with the pre-swap count (13.7 at 11 up to 16.0 at 16). The report raises this case
  and leaves it open (line 757). With the two points above, it is not an exception: it is the clearest single case
  of the general pattern.
- **Below 10, the data do look like a clip.** For a ≤ 9 LOW answers about 10 whatever the pre-swap count was (at a = 6
  the mean is 10.0 to 10.2 across pre-swap counts 11 to 16). So a range limit may add to the swap effect on the
  lowering side. Natural text agrees: on the two fresh rows with a = 9, LOW answers 10 both times (`check.py`). The
  saved data cannot separate the two there.

**Judgement.** "No construction mismatch" holds in the sense of no bug: the rows are what the filing specified. But
"a model limit" is not the main explanation the data support. Raising a by swaps leaves LOW answering roughly the
original count, and both models' errors grow with the swap size. The (17, 13) failure does undercut the report's
framing, and so does N1's pattern. So "the base model cannot count that far out" (lines 681, 770) is not
established, and neither is the conclusion that the rule may be untestable on this model.

**Fix.** Rewrite the void reason in the Amendment 2 answer (681), the void section (746–760), What we learned
(770–771), Surprises (789–790) and What next 2 (798), and in the notebook. Suggested wording: "both controls fail on
the swap-built rows, more the larger the swap, including N1's count of b, which the swap leaves untouched; above 17,
LOW's answer follows the count of a before the swap; whether LOW can count a outside 10–17 on natural text was not
tested." Add the N1-by-swap figure, or a short table, as the evidence. Leave the U0 verdict as it is.

### 2. The only fresh rows in new cells go against the rule, unreported (lines 692, 777–780, 768–771)

Of the 20,000 fresh rows, 19 lie outside Amendment 1's cells. Two of them are in cells where the one-third rule
differs from every other rule: (17, 10) and (17, 9). On both, W1 10K, W1s1 10K and W2 1K answer 16. The rule gives
15 and the midpoint 14 (`check2.py`). LOW counts a correctly on both rows. So the out-of-cell evidence on the fresh
rows reads 0 of 2. Two rows prove nothing either way. But they are the only test of the rule outside the cells it was
found in, and the report presents the fresh read only as a within-pair replication. Also, at a = 17 the slow state
answers 16 on all nine fresh rows, whatever b is (9 to 12). The rule predicts that answer should depend on b.

**Fix.** Add one sentence under "Within-pair replication": "the two fresh rows in new cells where the rule is
distinct, (17, 10) and (17, 9), are both answered 16, not the rule's 15, by every W state at 10K." This also gives a
reason to build the set in finding 3.

### 3. "A corrected unseen set looks hard to build" is overstated (lines 761–765)

Measured (`check.py`): `pcfg.sample_texts` draws 200,000 texts in 4.0 s on one CPU. Per 200,000 natural texts, the
five in-range cells where the rule is distinct come up 17 times for (17, 10), 2 for (15, 9), 1 for (16, 9), and 0
for (14, 9) and (17, 9). So the line "at most once in 20,000" is wrong for (17, 10), which comes up about 1.7 times
in 20,000. Getting 20 rows per cell takes about 0.25M texts for (17, 10), 2M for (15, 9) and 4M for (16, 9): seconds
to under two minutes. (14, 9) and (17, 9) are rarer than 1 in 200,000 and need tens of millions of texts, still CPU
minutes, and can run in parallel in chunks that keep only the target rows. The same sampler also yields natural
(17, 13) rows (17 per 200,000). One LOW read of those rows settles finding 1's open question, swap or rare
combination, directly. The edge-of-range caveat (a 14–17, b 9–10) stands.

**Fix.** Replace "looks hard to build" with the measured cost, and fix the "once in 20,000" figure. What next 2
already names the route. Add the natural-(17, 13) read to it.

### 4. LOW's unseen answers are misdescribed (lines 748–749, 753–754; notebook cell at 352)

"14 to 17, or a stray 4, above" and "move with the count of a in the right direction, and are clipped". The answer 4
is not stray. It is LOW's single most common answer at a = 21 (45%), 22 (58%), 23 (66%) and 24 (75%), and it is 10–30%
of rows at a = 18–20. LOW's mean answer falls from 14.6 at a = 17 to 7.2 at a = 24, so above 17 the answers do not
move in the right direction. The notebook drops the 4s entirely ("about 15 to 17 above"). **Fix:** describe it as
measured. It feeds finding 1.

### 5. "A stable description … from 3K steps on" is looser than the filed bar (lines 718, 768; notebook 364)

At 3K, W1 reads S_third 0.717 against a bar of 0.669, overall 0.868, and its pairwise read against `count(a) + 1` is
"neither". The overall share reaches T1's 0.95 bar only from 5K for seed 0 (0.960) and 7K for seed 1 (0.937 at 5K).
These are trajectory states, sibling runs read as context and never scored. **Fix:** "follows the rule on its own
rows from 3K, and on 95% of all rows from 5K to 7K, in one re-run per seed (context, not scored)."

### 6. The answer is not fixed by the cell everywhere (line 692)

In cell (16, 12), 142 rows, W1 10K answers 16 (= count a) on 102 rows and 15 (the rule, and also the midpoint) on 40.
W1s1 splits 109 to 33. So "a state's answer is almost fixed by that pair" holds overall but not in this large cell,
and that cell accounts for much of the 2.3% the rule misses. Minor. A clause would do.

### 7. "Moves on to the midpoint by 10K" for the fast arm (line 680; Amendment 1 line 580–581)

This rests on two saved points (1K, 10K), one seed, and no trajectory of W2. Amendment 1 hedges it ("may be a stage
the fast arm passes through"); Amendment 2 states it plainly. Keep the hedge, or add W2's trajectory to Not run.

### 8. Flag beyond this amendment: the swap step is used to build the fine-tuning and spurious test rows

`pcfg.spurious_texts` uses the same `_set_window_count` step, with swaps of mostly 0 to 3. Here N1's exact count-b
share is 0.87 at a swap of 2 and 0.70 at 3. That is confounded with these rows' unusual cells, so it is not a finding
about Stage A. But if swap-built rows are off-distribution for models trained on natural text, Stage A's readings on
spurious rows (PA2, LOW's count-b accuracy on the swap floor) could carry part of the same effect. Worth a $0 check:
N1 and LOW accuracy on Stage A's spurious test rows by swap size.

## Item-by-item answers to the brief

1. Re-scored: all verdicts as reported (table above).
2. Numbers: all match. The saved answers reproduce `amend2.json` exactly.
3. Void diagnosis: there was no construction bug (confirmed). "LOW cannot count a outside ~10–17" is not
   established: below 10 the answers look like a clip, but above 17 LOW follows the pre-swap count, and N1 fails with
   swap size on an untouched letter. The (17, 13) failure, together with N1, undercuts "a model limit" as the main
   explanation (finding 1).
4. Claims beyond the data: "stable" is fair within cells but loose on timing (finding 5). It also omits the 0-of-2
   out-of-cell rows (finding 2). The fast arm "moves on" rests on two points (finding 7).
5. Corrected unseen set: practical, at CPU minutes (finding 3).
6. /writeup: no serious issue beyond what the findings above already cover. `check.py` was not run on the report.

## Not run

- Any model forward pass: LOW on natural (17, 13) rows, or on natural rows in the five in-range cells.
- A test of whether LOW counts a outside 10–17 on natural text beyond the two fresh a = 9 rows; natural a = 9 or 18 rows are rare (about 1 in 10,000) but can be sampled.
- Finding 8's check on Stage A's spurious rows.
- `.claude/skills/writeup/check.py` on the report.
