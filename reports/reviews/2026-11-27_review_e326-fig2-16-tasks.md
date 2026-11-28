# Review: E326 (P1.66), Fig 2 on all 16 known-answer tasks

I checked the scored rows that will replace Fig 2's five-task trained points. The VPD and SAE rows pass every check. The
top-neuron rows pass too, but one of their controls is off in a way that depends on the state, so that panel stays
provisional until its cause is found. For the paper this means Fig 2 can move to the same 16 tasks as Figs 3 and 7, as a
description only: no task's fine-tune sits more than 0.05 above its matched random at any featurization.

Inputs: `experiments/fig2_all_tasks/results/{definition_rows.csv, summary.json, run_score.log, stage0/,
run_ident_diag.log}` at a44acab8; paper `figures/data/definition.csv` at origin/v2.

## Checks

| check | result |
|---|---|
| rows | 288 of 288 (16 tasks x 3 seeds x trained/random x 3 featurizations), 0 missing |
| `commute.eb` on E298's saved trained arrays | 52 of 52 give E298's E_b_K |
| 4 overlap tasks (e58 copy, rank1, cipher, prior_cipher) | 72 rows, value and source identical to the paper's definition.csv trained rows |
| values | all finite, in [0, 1] |
| randoms (new 12 tasks) | one per state, 36 distinct seeds per featurization, size matched to F - M within 4.1e-11 relative |
| read | 512 windows x 8 tokens, last position, every new item |
| stage 0 VPD (4 pods) | do-nothing E_b 1, 0 changed; identity 0 flips, max_dev 5.9e-5; repro e58/cipher_s1 exact |
| stage 0 SAE (2 pods) | do-nothing OK; identity max_dev 3.7e-5; must_flip 256 of 256; repro exact |
| stage 0 neurons (2 pods) | do-nothing OK; repro e58/copy_s0 rel 5e-6, same K; identity 0 flips but **max_dev 0.027** (filed: order of 8e-5) |
| neurons identity diagnostic | e58/copy_s0 5.2e-5, c3/copy_s0 4.6e-5, e62/copy_s0 0.027, e46/ident32_s0 0.069; 0 flips everywhere |

## Numbers for the caption and text (from definition_rows.csv)

| featurization | trained range | random range | per-task gap (seed means), min to max |
|---|---|---|---|
| VPD | 0.015-0.062 | 0.012-0.017 | 0.003 to 0.045 (c3/cipher) |
| top neurons (provisional) | 0.010-0.037 | 0.003-0.011 | 0.007 to 0.025 (e46/ident32) |
| SAE | 0.025-0.063 | 0.006-0.020 | 0.014 (e53/cipher, 0.01446) to 0.049 (c3/cipher) |

Every task's trained mean is above its random mean at every featurization, but no gap passes 0.05. The SAE maximum,
0.049, is close to 0.05, so the text should not say "well under". The mean trained share of the reuse tasks is below
that of the new tasks at all three featurizations (VPD 0.032 vs 0.041, neurons 0.019 vs 0.021, SAE 0.041 vs 0.045).

## Verdict

- B6 (caption scope): fixed once the caption says 16 known-answer tasks, 3 seeds, one matched random each, read on 512
  eight-token windows. e58/elicit (twin control, not known-answer) leaves the figure. The built-edit points are unchanged
  and keep their own read (41-position windows), which the caption or appE must still say.
- Random-reference clause: use the random ranges above, per featurization. The old E298 range (0.010-0.017) no longer
  covers SAE (up to 0.020) or neurons (down to 0.003).
- Descriptive only (card): any sentence that claims a result at a bar is a mismatch.
- Neurons panel: provisional. The swap path is exact on e58 and c3 states and off by 0.03-0.07 logits on e62 and e46,
  with no flips. Cause not found (definition-lead, after restart). A caption footnote or a held panel is the writer's call.
- Flag, not a blocker: the overlap rows still mix E290 (VPD seed 0 of the 4 tasks, every neuron row) and E298. Stage-0
  reproduction shows the drivers agree on the states tested.
- Card wording says 11 added tasks / 66 items; the run read 12 / 72. Wording only.

## Not run

- Identity on all 12 new tasks' states (only 4 states were diagnosed), and the cause of the e62/e46 deviation.
- Any re-read of the neuron rows after a fix.
- The new caption and sec5 text, which are not written yet: checks 6-7 get re-run on writeup-lead's commit.

## Re-check of the paper text (writeup-lead local c5a4d75 + 9028d99, 11-27 ~16:00)

- c5a4d75: definition.csv's 288 trained rows equal E326's definition_rows.csv exactly, and the built rows are unchanged.
  KEY_SIDE's reuse/new label matches known_states for all 16 tasks. Every single seed reads above its own random.
- 9028d99 ranges and gaps (sec5:17, appI:39-43) match the table above; the SAE minimum gap is 0.014 (my table first said 0.015).
- MUST: the caption's provisional note says "on two tasks' fine-tunes"; only four fine-tunes were checked, and two of them are off.
- SHOULD: the caption cites tab:built for the built edits' windows, but that table covers only the copy edits, while Fig 2
  also plots the induction, bigram and extra-draw edits. SHOULD: in appI, the 94-state read's random (0.011) sits next to
  Fig 2's VPD randoms (0.012-0.017); name each one's run.

## Neuron rows after the hook fix (19:15, 2026-11-27)

The neurons panel passes, with notes. The full record is ~/commute-data/agents/redteam-lead/measurement-record/E326_neurons_verdict.md.

- The fix (d2d81436) reads x's activation from F's own cache. The identity check now reads 7.8e-5 on e62 (was 0.027) and 5.5e-5 on e46 (was 0.069), with 0 flips.
- All 72 items reran under the fix. Of the 96 top-neuron rows, 24 moved, none by more than 3.8e-6. The VPD and SAE rows are unchanged, and the figure was built from the fixed CSV (6433be77).
- Note: the e58 reproduction gained one variable (4170 to 4171 entries, a relative change of 3.4e-4). No record names that variable, so its cause is inferred, not verified.
- Correction to my earlier range: the random neuron range is 0.003-0.011 (top value 0.01075, already there before the fix), not 0.003-0.010.
