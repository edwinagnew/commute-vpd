# E121: why the copy's weight change sits far from the fine-tune's (the copy-gap ladder)

*2026-11-06, regating-lead's writer. User-approved question (2026-11-06): "that rerun has an enormous difference. look
into it further. aim for an explanation not trying to close or accept it as quickly as possible." Additions from the
lead the same evening (Fable's advice `~/commute-data/scratch/regating-lead/fable-strategy/advice.md` § 4(b)): R3b, R4
and the functional cosine. Code `experiments/copy_gap_ladder/{run,rules}.py`, filing `results/predictions.json`, job
`cluster/copy_gap_ladder.sbatch` (MATS `debug`, L40, $0, about 20 min). Not gated: under $60 and 6 h.*

## The question, in the Terms' words

Teacher training (component training with loss KL(teacher ‖ student), Γ and Δ frozen) of the released decomposition,
with the emoticon task's 1,000-step Adam fine-tune as teacher, gives a decomposition whose weight change S keeps the
fine-tune's behaviour (teacher gap 0.0286 nats) but sits at input-weighted cosine 0.38-0.40 to the fine-tune's update
δW, where a rerun of the same fine-tune in another batch order sits at 0.83-0.84 (E34 Amendment 10,
`reports/analyses/2026-11-02_analysis_e34-amend10-input-weighted.md:35-36`). The copy differs from the fine-tune in its
target (KL to the fine-tuned model against label cross-entropy plus KL to W₀), its recipe and its parametrisation (only
U and V train). Which of these moves the change away from δW, and is the copy "δW plus a part that does nothing", or a
different change that does the same job?

**What the recipe difference turned out to be.** Reading the code that made Amendment 10's copies (`amend8c.py`
`student` → `amend8.py:327` `train_from`, now `teacher.train_student`): the copy trains on the fine-tune's own training
windows, 32 per step, drawn by the same seeded sampler (`component_training.py:154` and `finetune.py:203`), with Adam,
clip 1.0 and 1,000 steps, exactly as the fine-tune (`finetune.ArmAConfig`, `amend8.py:345`). The only recipe
difference is the learning rate: 1e-5 against 3e-5. So the brief's R2 is one rung (lr), not two, and the data
difference does not exist for this family. The Pile-text copies (arm B) are not Amendment 10's students and are not read.

## The ladder

All on the same teacher: the fine-tune regenerated in-job (order 0, fp32), and all read against Amendment 10's saved δW.
Every trained state runs at batch orders 1 and 2 (none shares batches with δW's order 0); a rung's value is their mean.

| Rung | Parametrisation | Loss | Optimiser, lr | Start | Differs from the rung above in |
|---|---|---|---|---|---|
| R0 | all 24 matrices | label CE + KL to W₀ (arm A) | Adam 3e-5 | 0 | (Amendment 10's saved reruns) |
| R1 | all 24 matrices | KL to the fine-tuned model (arm A weighting) | Adam 3e-5 | 0 | the target |
| R2 | all 24 matrices | the same | Adam 1e-5 | 0 | the lr |
| R3 | alive U, V (Γ, Δ frozen) | the same | Adam 1e-5 | 0 | the parametrisation: the copy itself |
| R3b | alive U, V | the same | plain SGD 0.01 | 0 | beside R3: the optimiser |
| R4 | alive U, V | the same | Adam 1e-5 | δW's least-squares split | beside R3: the start |

Settings, each sourced: Adam 3e-5 `teacher_training/run.py:46`; 1e-5 `run.py:55` (`Config.student_lr`); SGD 0.01
`run.py:45` (`SGD_LR[emoticon]`, E34 arm S's rate on this task); batch 32 `run.py:56`, `finetune.py` `ArmAConfig`;
1,000 steps `run.py:54`; clip 1.0 `ArmAConfig`, `TrainConfig`; loss `teacher.teacher_loss` (trigger KL plus the mean
KL elsewhere, `finetune.arm_a_weighting`); least-squares split `split.least_squares_split` at `split.bases`' tol 1e-6
(`amend9.py:51`). R1 and R2 use a new `finetune.finetune_with_loss`: `finetune_arm_a`'s sampler and loop with the loss
swapped; nothing existing changed. `teacher.train_student` gained `seed`, `optimizer` and `init` keywords whose defaults
are E34's (E119's caller unaffected). Departure, stated: R3b's teacher is the Adam fine-tune, where E34's arm S paired
SGD students with SGD teachers; the rate is arm S's.

**Reads, every state.** (1) Input-weighted cosine to δW, distance ‖S − δW‖/‖δW‖ and size ‖S‖/‖δW‖, pooled over the 24
matrices, uncentred second moment of W₀'s inputs at every position of the 73 held-out windows ("task") and of Pile
val rows 0-255 ("pile"): Amendment 10's code, moved to `commute.weighted` (`amend10.py` untouched; no frozen caller
imports the moved functions). (2) Behaviour on the held-out windows: teacher gap (E34's number), P(target). (3) The
functional cosine (lead's addition): cosine of the state's change in trigger-position logits (fp32, minus their mean
over the vocabulary, since the softmax ignores a constant) with δW's, exact forward. (4) The explanation split
(the brief's step, on every scored state, cheap): S = k δW + R with k fitted on task windows; R's input-weighted size
over S's on task and on Pile, and the behaviour of W₀ + k δW, that is S with R removed.

## Design check (`/design`, a line or two each)

- **Goal.** `GOALS.md`, the question "if no simple Φ works, is it new capability or brittle decompositions": the copy
  is the project's working Φ; whether it reproduces the update or only the function decides how the map can be stated.
- **Nearest tried.** `TRIED.md:101` (E34 Amendment 10) measured the gap and named no cause; Amendments 8c and 9
  read it unweighted. This run adds the cause, the function read and the start test, none of which were run.
- **Patch or step.** A step: the user asked for an explanation of an existing finding.
- **Decisions.** Reads under no reuse definition (`decisions.md` rows 81, 87 do not apply: no reuse verdict is given).
  Γ frozen, as every teacher-training run.
- **What changes next.** "R3 most" puts the cause in component training's geometry, and R3b/R4 split it into
  optimiser or path; "R1 most" puts it in the KL objective (any student of the teacher lands there), which makes the
  0.38 a property of distillation, not of the decomposition; "function as close as a rerun" allows "same function,
  different weights" in the write-up, "further" shrinks the copy claim to "keeps the target's probability".
- **Controls, first and alone.** C0 (no model) and C1 reread Amendment 10's five cells from the same saved files: this
  is R0 recomputed in-job, and it fails on a moved file, a wrong input set or a broken moment. C2 regenerates the
  fine-tune (the teacher). C3 regenerates the copy at A8c's orders 0 and 1, which must reproduce its saved files and
  Amendment 10's 0.40 and 0.38: it fails on any drift of the student recipe or teacher. No ladder state trains before
  C3 passes.
- **Saved for audit.** Every trained state's change in bf16 (`results/large/<state>.pt`, ~57 MB each), F0 in fp32.

## Numbers and what each should read

| Number | W₀ (do-nothing) | Control or reference | Lean |
|---|---|---|---|
| cos to δW, task; pile | nan | δW 1; R0 line 0.826; 0.837 | R1 0.65-0.85; R2 0.55-0.80; R3 0.38-0.41; R4 start 0.90-0.97 |
| teacher gap (nats) | 5.2-5.5 | fine-tune ~0 (hook floor); saved copies 0.0286, 0.0277 | every rung under 0.53 (the copy line) |
| P(target) | ~0.004 | fine-tune 0.999 | 0.95-1.00 |
| functional cosine | nan | δW 1; fine-tune F0 > 0.999 | reruns 0.85-0.98; copy 0.70-0.95 |
| remainder size, task; pile | nan | δW 0 | copy ~0.92 by construction on task; reruns ~0.56 |
| gap with R removed | — | δW: change 0 | copy: well above its own (k ≈ 0.27) |

Stop bars (code stops, exit 1): C0, C1 each cell within 1e-3 of Amendment 10's (`amend9.py:53` read_tol); C2
P(target) ≥ 0.9946 (`amend10.py:78`) and 1 − Frobenius cosine to the saved δW ≤ 1e-3 (`amend10.py:77`); C3 1 −
Frobenius cosine to the saved copy ≤ 1e-3, and cos to δW within Amendment 10's measured real-family batch-order noise of
its value (0.0065 task, 0.0032 pile, read from `amend10.json` at run time).

**Noise floor.** Amendment 10's real family: split-half 0.002 and 0.001, batch order 0.0065 and 0.0032, rerun spread
0.042 and 0.025 (task, pile), margin 0.05. Here the margin per set is the largest of 0.05, the in-job R0 spread and every
rung's order-1-to-order-2 difference, so a rung that is noisy widens the margin for all.

## What each outcome means (P1-P5 in the filing; rules there, edge cases run on CPU before filing)

- **P1, which rung carries the drop** (R0 line − R3): the rung whose drop is at least half of it and over the margin.
  R1: distillation itself lands elsewhere; R2: Adam's lr chooses among equally good changes; R3: component training
  does. "Shared" and "spread" are outcomes, not failures; "failed copy at Rk" means that rung did not learn the task
  and its cosine says nothing.
- **P2, R3b**: the same as R3 → not Adam's per-entry steps; closer → Adam on U, V is the bias.
- **P3, R4**: stays → the loss is flat along δW's image and 0.38 is the path from zero; drifts to the copy → the KL
  objective in this parametrisation pushes away from δW; part way → both.
- **P4, function**: as close as a rerun with weights far → "same function, different weights".
- **P5, remainder**: idle → the copy is δW's direction at a smaller scale plus a part with no task effect; carries
  behaviour → the remainder does the work and the copy is a different change.

**What each metric is blind to.** The cosine: anything downstream of the matrix outputs (two changes may cancel or
not), and the model's own changed activations (second order). The functional cosine: every position but the trigger.
The teacher gap: per-token detail beyond its mean. The remainder split: k is one scalar for all 24 matrices, so a
matrix-by-matrix fit could leave a smaller remainder.

## Run order and time (one `debug` job; resume by state file, exit 3 = resubmit)

Stage 0 no model (~10 s). Stage 1 model and moments (~40 s: Amendment 10 read 17 s model, 21 s forwards). Stage 2 F0
(38 s, `amend10.json` "rerun real order 0") and two copies (111 s each, A8c job 48613 log 13:31:40-13:33:31): ~5 min.
Stage 3 R3_o2, R3b ×2, R4 ×2 (~111 s each), R1 ×2, R2 ×2 (~38 s each), R4's split (~20 s): ~13 min. Stage 4 reads and
14 behaviour evaluations, under 1 min. About 20 min with start-up. A state starts only if it ends by 26 min, else the
job exits 3 and `fleet/resubmit --code 3` continues it.

## Not run

- The other two families (`rand32`, `rand32b`): their reruns are 10 and 30 steps, so lengths differ.
- The Pile-text copy (arm B): not Amendment 10's student.
- R1 and R2 at order 0 (sharing batches with δW); R3 at order 0 is regenerated as the control and reported only.
- The mean-removed form; split-half halves (Amendment 10's split-half noise, 0.001-0.002, is cited).
- Fable's Σ-eigenbasis × write-span decomposition of S − δW (CPU, separate).
- A per-matrix k in the remainder split; functional cosine at other positions; the check (VPD's adversary) on any rung.
- A plain-weights SGD student, and an R4 at the fine-tune's lr.
