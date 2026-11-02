# E34 Amendment 10: the student's change against the fine-tune's, weighted by the inputs the model sees

**Revised 2026-11-02, nothing having run** (first filing `8ce4471c`): the coordinator added one 1,000-step rerun of the real fine-tune, trained in this job, so the real-task verdict takes the map tables' rule. The limited band and E38's and E53's reruns as references are deleted, as no rule reads them. The filing was changed before any input-weighted read; the only reads so far are stage 0's Frobenius.

Approved by the user (2026-11-02, option (a) of log card `decision-commute-settled`). $0: one `debug` job on an L40, about 12 min, on
saved states, one forward of W₀ and one fine-tune rerun, whose control is the same recipe at the saved batch order. Rules and expected values: `experiments/teacher_training/results/predictions_amend10.json`.
Code: `experiments/teacher_training/amend10.py`, tests `tests/test_teacher_amend10.py`, job `cluster/teacher_amend10.sbatch`. An independent
review comes before the run.

The user's words: "The same comparison, weighted by the inputs the model actually sees, one free cluster job of minutes on saved states.
Tells you whether the claim reads 'copies the fine-tune's change on real inputs' or 'reproduces its behaviour by a different change'. The
line then closes on this model with that wording."

## The request, in the Terms' words

Call the decomposition after teacher training the student. S = (U + dU)(V + dV)ᵀ − UVᵀ is its weight change from W₀'s alive (U, V), and δW
is its fine-tune's finished update. For two changes A and B, over the 24 matrices together,

⟨A, B⟩_Σ = Σ_k tr(A_k Σ_k B_kᵀ) = E Σ_k ⟨A_k x, B_k x⟩, with Σ_k = E[x xᵀ] and x the input matrix k sees.

Read cos_Σ(S, δW), and against the same δW and Σ: a rerun of the same fine-tune (for the real task, trained in this job), a different task's fine-tune, and the same table's other
seed's student. Amendment 9 used Σ = I, so directions no input excites counted alike. The inside part is not read: under this inner product
it is no longer an orthogonal part of δW, and it has neither an expected value nor a prediction here.

## Design check

- **Goal served.** The wording of the commute line's headline on this model (`GOALS.md`, Q1, reading A), which closes the line here.
- **Nearest tried.** E34 Amendment 9: the same comparison by Frobenius, which said it could not change the wording since every direction
  counted alike. Its independent check found the 0.71 line fails a rerun of the same fine-tune. Nothing input-weighted on a student has run.
- **Patch or step.** A step: the named next item of Amendment 9's report, and the last read this line takes on this model.
- **Decisions relied on.** W₀'s alive (U, V), Γ and Δ frozen, as in teacher training. No departure from `decisions.md`.
- **What changes next.** "Copies" or "different change" is the line's closing wording. "Cannot tell" or "depends on the inputs" go to the
  user with the reads; the line can still close, stating that.
- **Control, first and alone.** Stage 0 with no model, the layout control, the rerun recipe at the saved teacher's batch order, then per input set the self, zero and power controls before any
  read is written. Any miss stops the job. No reference procedure is reproduced.
- **Numbers and what each should read.** Below, and `predictions_amend10.json` → `expected`.

## The choices the brief left open

- **Which data.** Three kinds, from W₀'s forward. *Task windows*: every position of the task's held-out windows (73 on emoticon, 97 on each
  map table, 41 positions each), held out of both trainings. *Pile*: every position of val rows 0 to 255, the harness's rows, 131,072
  positions. *Trigger*: position `half` alone of those windows, where the taught behaviour is read; Σ's rank there is at most 73 or 97, so it
  is reported (P4) and stays out of the wording. Task windows alone put the trigger at 1 position in 41, which is why the trigger is read at all.
- **Which model's activations.** W₀'s, for every pair. One fixed x gives one inner product, so each cosine is symmetric and all references
  share a metric. It favours neither side, and W₀ is where (U, V) and Γ were fitted. The reads are first order in the change, so the change
  in x is second order; that is listed as blind.
- **Normalisation.** Over all matrices together, as Amendment 9. No prediction reads per-matrix cosines; their inner products are saved in
  `results/large/amend10/gram_<set>.pt`, and `top_matrix_share` says if one matrix carries the pooled number.
- **Mean against uncentred.** Uncentred is scored, being the request's own form: a constant output shift is real behaviour downstream. Centred
  (Σ − μμᵀ) is read beside it and the verdicts need both to agree, so a result the mean alone carries reads "mixed", then "cannot tell".
- **Layout.** The 67M's matrices are all `nn.Linear`, (d_out, d_in), and `vpd.target_linear` asserts it: no Conv1D here. The layout control
  tests the module's output at W₀ + δW against x (W₀ + δW)ᵀ with x from the input cache, on all 24.
- **Precision.** No autocast on the forward, fp32 activations, x xᵀ and every matrix product in fp64. Rounding could enter by Σ weighting
  directions unequally: if a change is small along Σ's top directions, its rounding counts for more than by Frobenius. So one bf16 draw is
  measured on the real pair on every Σ, and the Gram's asymmetry is a stop.

## How a reading is read

Against a fixed line the reading cannot decide: the rerun's own number says what "close" means. **"Copies"** means a student no further from δW
than another run of that fine-tune, within noise. It does not mean an identical change: with two runs T + n and T + n′ sharing a task part T, a
student of exactly T reads √rerun, and T plus a noise of the same size reads rerun itself. P3 says whether the student follows this run or
only the task.

| Label, per student on a map table | Rule, with own = cos_Σ(S, δW), rerun = cos_Σ(δW′, δW), other = largest over other fine-tunes, m = the margin |
|---|---|
| unscorable | anything not finite |
| cannot tell | rerun under 0.1, or rerun − other under 2 m: the references do not separate |
| copies | own ≥ rerun − m |
| different change | otherwise |

The margin m is the largest of 0.05 (twice Amendment 9's largest batch-order change), the largest split-half difference over every cell the rules
read, the largest batch-order change here, and the bf16 draw. Split-half is by alternate window and compares two half samples, so it is
conservative for reads on the whole set. The real students take the same rule against their rerun, and all must agree, in both forms, on each input kind, or the verdict is
"mixed". The wording needs all four verdicts alike (map tables and real task, task windows and Pile). Otherwise it reads "depends on the inputs", "depends on the task" or "cannot tell". Each verdict can read either way: nothing in the arithmetic fixes own against rerun, and `tests/test_teacher_amend10.py` runs the score to each label. Amendment 9's wording held for any student by construction; neither wording here does.

## The real fine-tune's rerun

**Recipe.** `amend8.real_teacher`, as Amendment 8c ran it: `finetune_arm_a`, Adam 3e-5, batch 32, clip 1, 1,000 steps, the same windows and target, bf16 autocast in the forward, saved bf16 at `results/large/amend10/real_teacher_o1_dW.pt`. It runs as a stage before any input set, timed.

**What it varies.** The batch-order seed alone, the generator behind the draw of training windows. The map tables' two seeds vary the same thing alone: `anchors.kv_finetune` passes seed 0 or 1 to the same `finetune_arm_a`. So the real verdict uses the map tables' rule, the same `table_label`, and is not labelled limited.

**Differences from the saved teacher.** Batch order, which is the point. None other intended. Amendment 8c's job ran the Pile harness first, which does not touch the fine-tune's own generator; `finetune.py` has not changed since before its files; its card was an L40. The recipe's batch order 0 runs first and alone, and tests all of these.

**Filed checks, or the job stops.** Order 0: 1 − cosine to the saved teacher at most 1e-3 (Amendment 8c's `saved_tol`). Both orders: held-out P(target) at least 0.9946 (Amendment 8's filed bound for T1000; the saved teacher reads 0.99910). Size and damage are reported, with no stop.

**Limits on comparing the families.** Length differs between them, not within them.

- A map-table rerun is at 10 or 30 steps, the real one at 1,000. At 10 steps a seed changes which windows are drawn at all; at 1,000 each window is seen about 88 times, so the real rerun may read higher. E53's cipher rerun rose with length by Frobenius (0.14, 0.42, 0.84 at 10, 100, 300), a read no rule uses.
- Map-table fine-tunes stop at first learned, P(target) about 0.55, the real one at 0.999.
- So each student is fairly read against its own family's rerun, and "depends on the task" cannot separate task from length.
- One rerun gives one number, with no spread of its own.
- real_o0 shares all its batches with the fine-tune and real_o1 with the rerun. Both must agree, so shared batches cannot carry "copies" alone.

## Controls and expected values

| Number | At W₀, S = 0 | Do-nothing, change δW | Control |
|---|---|---|---|
| cos_Σ(·, δW) | nan on its row and column; every rule "unscorable" | 1 within 1e-6, on every Σ | S against S, δW against δW: 1 within 1e-6; stop |
| Frobenius re-read of Amendment 9 | same | Amendment 9's 1 | every pair within 1e-6 of `amend9.json`, stop; 1e-12 in the dry run |
| Layout | n/a | error under 1e-3, relative; lean 1e-5 | all 24 matrices, or stop |
| Power | n/a | n/a | a change matching δW on the top directions Σ excites and random on the rest: cos_Σ ≥ 0.99, Frobenius ≤ 0.5, design 1/3; else stop |
| Rerun | n/a | n/a | order 0 reproduces the saved teacher, cosine within 1e-3 of 1; both orders P(target) ≥ 0.9946; else stop |
| Gram | diagonal 0 | diagonal above 0 | asymmetry within 1e-6, smallest diagonal above 0; else stop |

Leans, low confidence as nothing is read on inputs yet: rerun 0.6 to 0.95, own 0.3 to 0.8, other 0 to 0.4; P1 "different change", medium-low; P2 "different change", medium-low; P3 "follows the task only" or "mixed"; P4 "cannot tell", medium. By Frobenius, read before filing: reruns 0.605 and 0.675; own 0.298 to 0.412, real 0.274 and 0.281.

## Noise

Sampling: split-half, above. Task windows have 36 to 48 windows a half, and so may set the margin. If it swallows the gap the verdict reads "cannot tell", which is the right answer on these few windows. Pile should fall to the least margin. bf16: one draw of ±2⁻⁸ relative on real_o1's saved offsets and on the real δW, an upper bound on one rounding. Batch order: rand32b's two seeds and the real pair.

## What it is blind to

It reads each matrix's output on inputs, with no regard to what downstream layers do with that output, and whether a difference there matters or cancels later. It is first order: S and δW are read on W₀'s x, as if nothing upstream moved. Only a real behaviour test would close either. It ignores length. Pooling lets a large-output matrix count more. A wording it gives is about the weight change on inputs; "reproduces its behaviour" rests on the earlier runs' gap to the fine-tuned model (0.0286 nats on the real task, Amendment 8), not on this one.

## Cost

$0, `debug` QoS, one L40, under the 30-min limit: about 12 min by the budget in the filing, unrehearsed, model load 23 s from E12 Amendment 3's log. The stages write as they finish.

## Departures from the literature

None. No cited procedure is reproduced: the cosine is the project's own, E34 Amendment 8c's and 9's, with the input second moment of `GOALS.md`'s outside share as its metric. The rerun's recipe is the project's own saved one, varied in batch order alone.

## Not run

- A second real rerun, or a map-table rerun at 1,000 steps, which would separate task from length.
- A behaviour test of whether the differences on inputs matter downstream.
- Each model's own activations as x: second order, not read.
- The inside part under this inner product; per-matrix verdicts; E42's students, whose fine-tunes were not saved.
- Anything on Qwen, and any comparison with cheaper methods, which are other leads' lines.

## Hand-off

Next: an independent review of the code and the filing, then `sbatch cluster/teacher_amend10.sbatch` via `/pod-runner`. After the run: analysis report per `/writeup`, `experiments/TRIED.md`, a `docs/research_log/PENDING.md` line, and `README.md`'s "Current plan" line.
