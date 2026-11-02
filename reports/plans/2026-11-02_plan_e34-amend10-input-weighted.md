# E34 Amendment 10: the student's change against the fine-tune's, weighted by the inputs the model sees

**Revised 2026-11-02, nothing having run, so the filing could still change** (first filing `8ce4471c`, second `5fc87377`). The second revision follows the independent review, `reports/reviews/2026-11-02_review_e34-amend10-code.md`: its findings 1 to 10 are applied, and "What the lead did" there lists them. No read weighted by inputs of the real model exists; the only real reads are stage 0's Frobenius. The dry run used real vectors with made-up moments, to test the path and time it.

Approved by the user (2026-11-02, option (a) of log card `decision-commute-settled`). $0: one `debug` job on an L40, about 14 min, on
saved states, one forward of W₀ and the reruns, each recipe's control being the same recipe at its saved seed. Rules and expected values: `experiments/teacher_training/results/predictions_amend10.json`.
Code: `experiments/teacher_training/amend10.py`, tests `tests/test_teacher_amend10.py`, job `cluster/teacher_amend10.sbatch`.

The user's words: "The same comparison, weighted by the inputs the model actually sees, one free cluster job of minutes on saved states.
Tells you whether the claim reads 'copies the fine-tune's change on real inputs' or 'reproduces its behaviour by a different change'. The
line then closes on this model with that wording."

## The request, in the Terms' words

Call the decomposition after teacher training the student. S = (U + dU)(V + dV)ᵀ − UVᵀ is its weight change from W₀'s alive (U, V), and δW
is its fine-tune's finished update. For two changes A and B, over the 24 matrices together,

⟨A, B⟩_Σ = Σ_k tr(A_k Σ_k B_kᵀ) = E Σ_k ⟨A_k x, B_k x⟩, with Σ_k = E[x xᵀ] and x the input matrix k sees.

Read cos_Σ(S, δW), ‖S‖_Σ / ‖δW‖_Σ and ‖S − δW‖_Σ / ‖δW‖_Σ, and against the same δW and Σ: reruns of the same fine-tune (trained in this job), a different task's fine-tune, and the same table's other
seed's student. Amendment 9 used Σ = I, so directions no input excites counted alike.

**What this measures and what it does not.** It measures whether a weight change at each matrix's output is as close to the fine-tune's as a rerun's is. The sentence the user asked about has a second half, that the behaviour is kept, and this job does not read it: it is cited from E34 on the real task and from E12 Amendment 2 on the map tables. So the code's labels name the measurement, and the filing's `what_each_outcome_allows` says what the claim may then say.

**A correction.** The earlier filing said the inside part "is no longer an orthogonal part" under this inner product, as the reason for not reading it. That holds for the Frobenius inside part only. The input-weighted inside part is the complement of the project's own outside-both block (`GOALS.md`, `split.four_blocks`): (I − P_U) δW a₂ is what no first-order move of the subcomponents can produce on these inputs. It gives a ceiling, sqrt(1 − outside-both) on the cosine, and it is now read per fine-tune and used as a control.

## Design check

- **Goal served.** The wording of the commute line's claim on this model (`GOALS.md`, Q1, reading A), which closes the line here.
- **Nearest tried.** E34 Amendment 9: the same comparison by Frobenius, which said it could not change the wording since every direction
  counted alike. Its independent check found the 0.71 line fails a rerun of the same fine-tune. Nothing input-weighted on a student has run.
- **Patch or step.** A step: the named next item of Amendment 9's report, and the last read this line takes on this model.
- **Decisions relied on.** W₀'s alive (U, V), Γ and Δ frozen, as in teacher training. No departure from `decisions.md`.
- **What changes next.** A reading per family, and the sentence it allows. "Cannot tell" or "depends on the inputs" go to the user with the reads; the line can still close, stating that.
- **Control, first and alone.** Stage 0 with no model, the layout control, each rerun recipe at its saved seed, every set's moments, every power control. All of it comes before any input-weighted read of a student, so a stop there leaves no such read to amend around. No reference procedure is reproduced beyond those recipes.
- **Numbers and what each should read.** Below, and `predictions_amend10.json` → `expected`.

## The choices the brief left open

- **Which data.** Four kinds, from W₀'s forward. *Task windows*: every position of the task's held-out windows (73 on emoticon, 97 on each
  map table, 41 positions each), held out of both trainings. *Pile*: every position of val rows 0 to 255, the harness's rows, 131,072
  positions. These two make the reading. Beside it, scoped out in the results' own scope string: *trigger*, position `half` alone of those windows, of rank at most 73 or 97 (P4), and *taught*, Σ_trigger + Σ_task, the training's own weighting, from the two saved Grams (P5). Both trainings weigh the trigger equally with the mean of all other positions, and task windows hold it at 1 position in 41, so the reading is mostly about what each change does where it was trained to change nothing, and it says so.
- **Which model's activations.** W₀'s, for every pair. One fixed x gives one inner product, so each cosine is symmetric and all references
  share a metric. It favours neither side, and W₀ is where (U, V) and Γ were fitted. The reads are first order in the change, so the change
  in x is second order; that is listed as blind.
- **Normalisation.** Over all matrices together, as Amendment 9. No prediction reads per-matrix cosines; their inner products are saved in
  `results/large/amend10/gram_<set>.pt`, and `top_matrix_share`, in the results file, says if one matrix carries the pooled number.
- **Mean against uncentred.** Uncentred is scored, being the request's own form: a constant output shift is real behaviour downstream. Centred
  (Σ − μμᵀ) is read beside it and the verdicts need both to agree, so a result the mean alone carries reads "mixed", then "cannot tell".
- **Layout.** The 67M's matrices are all `nn.Linear`, (d_out, d_in), and `vpd.target_linear` asserts it: no Conv1D here. The layout control
  tests the module's output at W₀ + δW against x (W₀ + δW)ᵀ with x from the input cache, on all 24.
- **Precision.** No autocast on the forward, fp32 activations, x xᵀ and every matrix product in fp64. Rounding could enter by Σ weighting
  directions unequally, so one bf16 draw is measured on the real pair on every Σ.

## How a reading is read

Against a fixed line the reading cannot decide: the reruns' own number says what "close" means. **"As close as a rerun"** means no further from δW
than another run of that fine-tune, within noise, not an identical change: with two runs T + n and T + n′ sharing a task part T, a
student of exactly T reads √rerun, and T plus a noise of the same size reads rerun. P3 says whether the student follows this run or
only the task. **"Further than a rerun"** means only that: 0.70 against a line of 0.80 reads further.

Two axes, each per student, form and input set. *Angle*, with own = cos_Σ(S, δW), rerun = the mean over every ordered pair of the family's runs, other = the largest over other tasks' fine-tunes, m the margin:

| Angle label | Rule |
|---|---|
| unscorable | anything not finite |
| cannot tell | rerun under 0.5; or rerun − other under 2 m; or the ceiling under rerun − gap_min |
| as close as a rerun | own ≥ rerun − gap_min, with m at most 2 gap_min (over it, cannot tell) |
| further than a rerun | own < rerun − m |
| cannot tell | between |

*Length*: the same `close` on ‖S − δW‖_Σ / ‖δW‖_Σ, against the reruns' mean distance, with its own least gap 0.056 and margin; cannot tell if the reruns' distance is over 1 (no nearer than no change) or if the nearest first-order change, sqrt(1 − ceiling²), is out of the band. Size is reported beside it: given an angle read as close, a further distance comes from size alone.

**Sources.** The floor, 0.5: a rerun of equal length at cosine 0.5 is at distance exactly 1, as far from δW as no change. gap_min, 0.05: own and the line may each move by Amendment 9's largest batch-order change, 0.025. gap_dist_min, 0.056: the same on this distance by Frobenius on the saved states, 0.028. The noise cap, 2 gap: the whole read is the mean of two halves, so |A − B| is about twice its noise, and at the cap the noise is at most the band's width. So noise cannot give the first label: a larger m does not lower the line, and a read inside the band under a large m reads "cannot tell".

The margin is per family, input kind and form: the largest of the least gap, the largest split-half difference over every cell the family's rules read, split by source row so the halves share no text, the batch-order change, the bf16 draw, and the range over the family's reruns.

**One reading per family, no joined headline.** The students of a family must agree over both forms on each input kind, or the verdict is "mixed", with the map tables' labels per table beside it. The reading on each axis takes task windows with Pile: alike, that label; decided and different, "depends on the inputs"; otherwise "cannot tell". Each verdict can read either way: nothing in the arithmetic fixes own against rerun, the ceiling refuses a "further" forced by construction, and the tests run the score to each label.

## What each outcome allows the claim to say

Full text: `predictions_amend10.json` → `what_each_outcome_allows`. The behaviour half is cited, never read here.

| Reading | Real task (behaviour: E34, 0.0286 nats) | Map tables (behaviour: E12 Amendment 2) |
|---|---|---|
| Angle and length as close | "Copies the fine-tune's change on real inputs", at a rerun's level, at each matrix's output; keeps the behaviour | The same at the level of a 10- or 30-step rerun; on rand32b the lost fifth of the re-switching term is stated. With P3 "follows the task only": the task's change, not this run's |
| Angle as close, length further | "Copies the direction, at a different size" | The same, with the size ratio |
| Angle further | "Reproduces its behaviour by a change further from the fine-tune's than a rerun is", at the reported cosine and distance | rand32: the same. rand32b: "further than a rerun, keeps its gain, loses about a fifth of the re-switching term"; "reproduces its behaviour" is not available |
| Cannot tell | Behaviour kept, the change on inputs unresolved, with the reason | The per-table labels, or unresolved with the reason |
| Depends on the inputs | Each input's sentence apart | The same |

Each sentence carries its scope: at each matrix's output on W₀'s inputs, every position of held-out windows and of Pile, not about the trigger. If P4 or P5 decides against the reading, the report says so.

## The reruns

**Recipes.** Real: `amend8.real_teacher`, as Amendment 8c ran it: `finetune_arm_a`, Adam 3e-5, batch 32, clip 1, 1,000 steps, bf16 autocast in the forward, saved bf16. Batch orders 1 and 2, with the saved teacher as order 0: three runs, three distinct cosines. Map tables: `anchors.kv_finetune`, at each table's step (rand32 10, rand32b 30), seeds 2 and 3 beside the saved 0 and 1: four runs, six distinct cosines.

**What they vary.** The batch-order seed alone, the generator behind the draw of training windows, in both families.

**Differences from the saved fine-tunes.** Batch order, which is the point. None other intended. Amendment 8c's job ran the Pile harness first, which does not touch the fine-tune's own generator; `finetune.py` has not changed since before its files; all compute cards are L40. Each recipe at its saved seed runs first and alone, and tests all of these.

**Filed checks, or the job stops.** Saved seed, on all three recipes: 1 − cosine to the saved fine-tune at most 1e-3 (Amendment 8c's `saved_tol`). Real orders 0, 1, 2: held-out P(target) at least 0.9946 (Amendment 8's filed bound for T1000; the saved teacher reads 0.99910). Size and damage are reported, with no stop.

**The length limit, plainly.** Each student is read against reruns at its own fine-tune's length: map tables 10 and 30 steps, the real task 1,000. Map-table fine-tunes stop at first learned, P(target) about 0.55, the real one at 0.999. So a difference between the families cannot by itself be read as task rather than length, and no sentence joins them. The job adds one 1,000-step map-table pair (rand32, seeds 0 and 1, about 70 s) that reports how a rerun's cosine moves with length. It is a reference only: those students' teachers are 10- and 30-step, so it does not replace their own-length reruns; it is one table and one pair; no rule reads it.

**Shared batches.** real_o0 shares all its batches with the fine-tune and real_o1 with a rerun. Both must agree, so shared batches cannot carry a label alone.

## Controls and expected values

| Number | At W₀, S = 0 | Do-nothing, change δW | Control |
|---|---|---|---|
| cos_Σ(·, δW) | nan on its row and column; every rule "unscorable" | 1 | ceiling: the best first-order change must reach the line, else "cannot tell" |
| Distance, size | nan | 0, 1 | a zero change is at distance exactly 1: tested |
| Frobenius re-read of Amendment 9 | zero reads nan | self 1 within 1e-6 | every pair within 1e-6 of `amend9.json`, stop; 1e-12 in the dry run |
| Layout | n/a | error under 1e-3, relative; lean 1e-5 | all 24 matrices, or stop |
| Power | n/a | n/a | Z = δW + R, R random on directions with an eigenvalue under 1e-3 of the mean, ‖R‖² = 8‖δW‖²: cos_Σ ≥ 0.99, Frobenius ≤ 0.5, else stop. Writes k/d per matrix. It says only that Σ is ill-conditioned; made-up spectrum 0.9973, 0.3334 |
| Reruns | n/a | n/a | as above |
| Gram | diagonal 0 | diagonal above 0 | smallest diagonal above 0 per set, else stop |

Leans, low confidence as nothing is read on inputs yet: rerun 0.6 to 0.95, own 0.3 to 0.8, other 0 to 0.4, ceiling 0.93 or higher. P1 and P2 further than a rerun on both axes, medium-low to medium; P3 "cannot tell" or "mixed"; P4 "cannot tell", medium; P5 as task windows, low. By Frobenius, read before filing: reruns 0.605 and 0.675; own 0.298 to 0.412, real 0.274 and 0.281; students at 1.5 to 2.3 times the size on the map tables and 0.73 on the real task, 1.06 to 2.20 from the fine-tune, against 0.80 to 0.90 for a rerun; ceiling 0.787 to 0.816.

## Noise

Task windows have about 36 to 48 windows a half, and the maximum runs over every cell, so they may set a margin above the cap. Then "as close" cannot be read there and the answer is "cannot tell", which is right on these few windows; a read further than the margin still decides. The reruns' range can do the same on either input: on made-up reruns with noise it reached 0.09. Pile should fall to the least margin otherwise.

## What it is blind to

It reads each matrix's output on inputs, with no regard to what downstream layers do with that output, and whether a difference there matters or cancels later: exactly the second half of the user's sentence. It is first order: S and δW are read on W₀'s x, as if nothing upstream moved. Pooling lets a large-output matrix count more; `top_matrix_share` shows it. The distance mixes angle and size; the size ratio separates them.

## Cost, per stage

$0, `debug` QoS, one L40, `--time=00:30:00`. About 14 min, from the dry run on 8 CPUs with real vectors and made-up moments, unrehearsed on the card; fp64 on the card is unmeasured, so the Grams are budgeted at the CPU's times.

| Stage | Minutes |
|---|---|
| Load the states, stage 0 | 0.5 |
| Model, layout | 0.5 |
| Reruns: five 1,000-step fine-tunes, six short | 3.5 |
| Bases, every set's moments (Pile unmeasured, the largest unknown) | 2 |
| Four power controls | 1 |
| Reads: seven sets, three taught | 6.5 |

Each set is written as it finishes, and `score` runs on the file's reads if the job is killed.

## Departures from the literature

None. No cited procedure is reproduced: the cosine is the project's own, E34 Amendment 8c's and 9's, with the input second moment of `GOALS.md`'s outside share as its metric. The reruns' recipes are the project's own saved ones, varied in batch order alone.

## Not run

- A behaviour test of whether the differences on inputs matter downstream: the half of the sentence this job does not read.
- A forward of the real model before the job: Σ's real spectrum is first read by this job, and a stop there has written no read of a student.
- Each model's own activations as x: second order, not read.
- Per-matrix verdicts; E42's students, whose fine-tunes were not saved; a 1,000-step pair on the second table.
- Anything on Qwen, and any comparison with cheaper methods, which are other leads' lines.

## Hand-off

Next: `sbatch cluster/teacher_amend10.sbatch` via `/pod-runner`. After the run: analysis report per `/writeup`, `experiments/TRIED.md`, a `docs/research_log/PENDING.md` line, and `README.md`'s "Current plan" line.
