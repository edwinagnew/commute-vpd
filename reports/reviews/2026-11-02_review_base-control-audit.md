# Review: what a base-only control would add, method by method

*2026-11-02. Audit from code and results files, requested by the user. Read-only; nothing was run. `E34:` below means
`experiments/teacher_training/results/`. Fable's advice (`reports/plans/2026-11-02_advice_base-control-FABLE.md`) was
read after the code; this audit agrees on the main split and differs in the three places marked "differs from Fable".*

## Answer

Teacher training and component training cannot learn from a gap between the base model and the decomposition, because
neither loss contains the decomposition's own forward: the student is the stored base model plus a hook that adds only
the offsets' change, so with nothing to copy the loss and gradient are exactly zero. The existing control for this,
E34's C2, therefore cannot fail except through a code bug, and no optimiser step in it moves anything. The recipes
that do contain that gap (masked teacher training, the Γ refit) did run a base-only control, it failed every time, and
no filed claim rests on their trained arms. So no finding breaks, but three things are unprotected: nothing has ever
measured what the optimiser does to the decomposition when there is little or nothing to learn, teacher-trained states
sit closer to the Δ-less decomposition than the base does (a 3-28% drop nobody has explained), and ratios below 1 have
been left unexplained rather than read as noise. The useful control is not "teacher = W₀ from W₀", which is already
known to read zero, but teacher = W₀ from a start away from W₀.

## The base model's own readings, and why each is nonzero at δW = 0

These are the only true base-to-decomposition gaps in the project. All read at W₀, nothing trained.

| Reading | Forward | Value | Cause, from the code | Source |
|---|---|---|---|---|
| Hook path against merged weights, at zero offsets | target + hook adding exact 0 | 0 exactly | `Offsets.extra` is 0 at dU = dV = 0, so the logits are the same bytes (`component_training.py:53-56,68-74`); `teacher.py:31-37` writes the gradient q − p out so it is exactly 0 | E34: `teacher_training_controls.json` `.C2` |
| Hook path against merged weights, offsets not zero | same, offsets ≠ 0 | 0.0005 to 0.0007 nats | rounding: the hook adds a separately computed term under bf16, the teacher merges it into W. Called the "hook floor" | E34: `teacher_training_controls.json` `.C3.hook_floor` 0.00070; `teacher_training_e41.json` `.C2.loss_step1` 0.00050; `teacher_training_e41_amend2.json` `.C2.loss_step1` 0.00057 |
| Component path, every mask 1, Δ in ("path KL", G2) | `model(mask_infos=...)` under bf16 | 0.00065 per Pile position; 0.00163 on rand32 windows, arm A's weighting | bf16 rounding of U Vᵀ + Δ, not a real gap: the same path in fp32 reads −1.3e-9 | `experiments/teacher_spread/results/amend1.json` `.G2.path_kl`; E34: `teacher_training_e42_amend1_rand32.json` `.G2.path_kl`; `experiments/pass_reread/results/saved.json` `.S1.path.delta_on` |
| Every mask 1, **Δ out** (harness `unmasked_kl`) | `faithfulness.py:97` | 0.0120 nats per position on Pile; 0.0107 in fp32 | real: what removing Δ₀ does. The purest form of the user's gap | E34: `teacher_training_controls.json` `.C1.pile["0"].unmasked_kl`; `saved.json` `.S1.path.delta_off` |
| CI-masked | `faithfulness.py:96` (Δ out) | 0.348 Pile; 1.13 rand32 windows (trigger 0.69, others 0.44) | VPD's own reconstruction error at Γ's masks; Δ in or out changes it under 0.004 nats (TRIED E48) | `.C1.pile["0"].ci_masked_kl`; `teacher_training_e42_amend1_rand32.json` `.G2.ci_masked_w0` |
| Stochastic | `faithfulness.py:94` | 0.248 Pile; 0.76 rand32 windows | the same at g + (1 − g) r | `.C1.pile["0"].stochastic_kl`; `.G2.stochastic_w0` |
| Adversary | `faithfulness.py:129-134` | 0.699 / 0.751 / 0.707 H100 seeds 0-2; 0.692 / 0.836 on the L40 at seeds 0-1 | worst-case masks | `.C1.pile`; `experiments/claim_tests/results/t1.json` `.W0.kl` |
| CI-masked at the trigger ÷ per-position on Pile (F) | E48 | r0 0.76, r1 0.95, r2 1.14, emoticon 1.48, rand32 1.86, rand1 2.21 | the base's gap differs by task | `saved.json` `.S1.F` |

## Table, one row per method

"Zero?" is question 2: with the base as the only input, is the objective's value and gradient zero exactly, up to
rounding, or truly nonzero. "Control run" is question 3. Read side is in its own section below.

| Method | Optimises, on which forward | Zero? | Base-only control run; could it fail otherwise than by a bug? | What the control adds |
|---|---|---|---|---|
| **Teacher training, E34 arm A** | KL(teacher ‖ student), arm A's weighting, hook path (`teacher.py:40-53`, `component_training.py:136-158`, `run.py:172-181`) | exact | C2: r1 windows, Adam 1e-5, batch 32, 100 steps (`run.py:200-213`). Read: loss 0, offset 0. Adam stepped on a zero gradient, so nothing moved. One task's windows only, but the result does not depend on them. Cannot fail otherwise | Nothing as run. Matters from a start away from W₀ (rank 1) |
| **E34 arm B**, Pile rows, plain mean (and A1's B-masked, B-dropped) | same, `teacher.py:44-50`; `keep` masks positions | exact | not run; zero by the same line. Not a gap: different rows give the same 0 | as above |
| **E34 arm S**, plain SGD | same, SGD | exact | not run; no Adam to scale noise. Fails at 3.14 and 3.39 with a real teacher, unrelated to any gap (E34: `teacher_training_r1_t534_v369.json` `.teachers.sgd.S["1000"].pgd`) | little; a perturbed start would show whether SGD's fail exists with nothing to copy |
| **E34 A2-A7** (rows, LoRA teacher, seed, budget, SGD, longer) | hook path via `train_student` / `component_training.train` (`amend3.py:243`, `amend4.py:177`) | exact | none rerun | as arm A |
| **E37, E39 teacher arms, E39 A2 rank cap, E39 A3** | hook path, plain mean on 512-token rows (`row_copy.py:69-82`) | exact | not run. E37's "C2" is a different test: the teacher regenerated (`teacher_spread/run.py:331`) | a perturbed start on lowercase rows; E37 A1 arm 3 (adjoint start, check 1.05 → 1.84) mixes drift and copying |
| **E41 + A1, A2, chains** | hook path from a warm start (`path.py:169-185`, `path_amend2.py:168-180`) | rounding at the warm start, not exact: 0.0005 | C2 here is the student at its own weights, **lr 0, one step**. No optimiser step had any effect; it reads the hook floor and that the warm start loads. Could fail by a loader bug or a large floor only | Floor itself adds nothing: links start at 0.055 to 4.86 nats, about 100 times it or more (`teacher_training_e41_amend2.json` `.links[].start_kl`). But teacher = own start at lr 1e-5 is the cheapest real test of rank 1, and has never run |
| **E42 arms A, B** | hook path on cipher windows or Pile (`newskill.py:288`) | exact | deliberately not run: "a property of teacher.teacher_loss" (`newskill.py:185`). Right for the zero | as arm A |
| **E43** teacher training plus remainder | teacher training, then Δ′ = Δ + W_ft − W_tt (`remainder.py`) | training exact; the remainder is 0 only if W_tt = W_ft | no. With teacher = W₀ and nothing moved the remainder is 0 and the state is W₀ | nothing beyond arm A |
| **E37 A1 arm 2, masked teacher training** | teacher KL + stochastic + persistent adversary, component path; target = student's own output (`masked_teacher.py:55-97`) | **truly nonzero**: the masked terms | M0 on the arm's own rows, 1,000 steps, same optimiser. Check 3.84, retry at 0.05 3.14; damage 0.021, 0.007 (`teacher_spread/results/amend1.json` `.M0.attempts`). A real control and it failed; arm 2 `not run` (`.arm2`) | Already done. Coefficient 0 not run, so path and Adam are not separated from the masked terms (report line 569) |
| **E42 A1, masked copy** | unmasked + 0.5 CI-masked + 0.5 stochastic, target = teacher, arm A's weighting (`masked_copy.py:52-73,101`) | **truly nonzero**: 1.13, 0.76; unmasked 0.0016 is rounding but its gradient q − p is not zero | M0 rand32 windows, 1,000 steps. Check 25.99; CI-masked 0.934, Δ-out 0.900; terms fell 1.13 → 0.78, 0.74 → 0.51, unmasked rose to 0.04 (`teacher_training_e42_amend1_rand32.json` `.M0`). Stopped before the teacher; rand1 not run | Already shows the user's worry in action: the recipe reduces VPD's own error and breaks the adversary. Coefficient 0 not run |
| **E42 A2, per term** | `masked_terms.py:98` | truly nonzero | M0_ci 48.74, M0_st 11.71, M0_trig 34.04; no copy run (`teacher_training_e42_amend2_rand32.json` `.M0_*.pgd`, `.stopped`) | same; the ranking of terms is on rand32, batch order 0, one adversary start |
| **Component training on task labels, E4, E5, E7, E11, E16, E21, E26** (E13 by its TRIED row; directory deleted, unread) | CE at the trigger + KL to W₀ elsewhere, hook path (`finetune.py:106-115`, `component_training.py:149-150`) | nonzero, but it is the task, not a gap: Δ and masks are not in the loss. KL part is autograd rounding at the start | no training control; only W₀'s read floor. A trigger-target arm can't be zeroed. **Differs from Fable**: a base-only version does exist, labels drawn from W₀'s own distribution, expected gradient 0, per-batch gradient not. Never run on the 67M; E45's null is that idea on Qwen | medium, for Adam's walk on data noise; see rank 3 |
| **Labelled component training on lowercase, E39 A2** | next-token CE, hook path (`broad_fragility/amend2.py:205-222`) | truly nonzero at δW = 0, and part of it is the base's ordinary prediction error on text, not the task | no. Adam 1e-5 7.24, SGD 0.03 1.28 (`broad_fragility/results/amend2.json` `.reads.label_adam.pile.pgd_kl`, `.reads.label_sgd.pile.pgd_kl`) | medium: the same recipe on plain rows or own samples would say how much of Adam's 7.24 needs lowercase at all |
| **Least-squares, adjoint, rescaling splits** (E1, E2, E14, E16, E22, E31) | algebra on a finished δW; no data but Σ, no forward (`split.py:119-122,264-270,316-333`) | exact and skipped: a zero matrix is passed over (`split.py:165,288,344`) | not needed. They cannot read any gap, since masks are not an input. **This control adds nothing** | nothing |
| **Step split, E19, E25** | adjoint split of each step S_t (`split.py:394-409`) | S = 0 gives zero pieces (rounding guard) | no | nothing |
| **Γ refit, E15** | Γ only, released loss stack minus faithfulness, U, V, Δ fixed (`gamma.py:124-142`) | **truly nonzero** | w0_P, same recipe, same rows, same steps as ls_P. Check 2.57, 3.53, 6.27, 2.63 at 100, 300, 1,000, 3,000; CI-masked 0.976, 1.074, 0.951, 1.033; v2 protocol 2.06, 7.76 (`experiments/gamma_refit/results/gamma_refit.json` `.refits.w0_P.checkpoints[].pile`, `gamma_refit_v2.json` `.partial.w0_P`). Failed, so it could fail | Already done, and read correctly: the split's 1.06 "cannot be read as the state's" (report line 266-267) |
| **Γ shift map, E30** | no training; Γ′ = clip(Γ + s), s from a fine-tune | s = 0 reproduces the harness to within `exact_tol` (`gamma_shift_map/run.py:56,102`) | C1, and null-target and shuffled s arms | nothing further; the null arm is the informative one |
| **E47, E49** | read saved or regenerated states; E47 t5 reruns SGD teacher training | no new objective | W₀ read per start, per block | read side only |

## The read side

- **Same rows, same adversary start.** A state's ratio divides by W₀ at seed 0 on the same rows in the same job
  (`harness.py:150-155`); both start from `torch.manual_seed(seed)` (`faithfulness.py:128`). E47 pairs start by start,
  E49 block by block and seed (TRIED E49). E34's resume refuses another card or floor (`run.py:92`). Other amendments
  were not read line by line for this; their gates reproduce E34 or E41 to four digits (`.G1`, `.G3`).
- **Seed 0 is W₀'s lowest of the seeds read**, so it is the harshest denominator on both cards. A ratio below 1 is not a
  lucky denominator.
- **W₀'s floor measures the start only.** It repeats W₀ at other seeds and never moves the state. The only reading
  of a move with no content is E39's random move of U, V: 0.988, 1.029, 1.069 at size 0.018, 0.036, 0.036, damage
  0.003 to 0.010 (`experiments/broad_fragility/results/broad_fragility.json` `.reads["move_s0_x0.5"]`, `["move_s0_x1"]`,
  `["move_s1_x1"]`). At damage 0.24 it reads 16.5 and 27.2 (`amend2.json` `.reads.move_s0.pile.pgd_kl`, `.reads.move_s1.pile.pgd_kl`). No random move
  was read at a narrow teacher-trained state's size and damage together (size 0.02, damage 0.02 to 0.06).
- **Below 1 is common and not explained.** Harness at seed 0: E34 emoticon SGD arm B 0.895, r0 Adam arm A at step 100
  0.929, E42 rand32 arm A 0.960, E41 A2 link at T1000 0.910 (E34: `teacher_training_emoticon.json`
  `.teachers.sgd.B["1000"].pgd`, `teacher_training_r0_t352_v310.json` `.teachers.adam.A["100"].pgd`,
  `teacher_training_e42_rand32.json` `.A["1000"].pgd`, `teacher_training_e41_amend2.json` `.links["1000"].pgd`). At other
  starts: rand32 0.966, 0.961, 0.894, 1.012; r1 1.033, 1.003, 1.225, 0.980 (`claim_tests/results/t1.json`,
  `t1b.json`, `.states.*.ratio`). E34 A4's 160-step adversary reads 0.953 to 1.148. The E41 report records the fact
  and stops ("reads below the unchanged model", `2026-10-28_analysis_e41-teacher-path.md:646-647`); the E41 A2 "excess
  removed" of 1.18 (`.score.excess_removed["1000"]`) is such a reading. Across starts these sit around 1, and E39's
  random move spans the same band, so the plain reading is "as the base, within noise", not "better than the base".
- **Stochastic below 1 comes from the fine-tune, not from training.** Stochastic ratio floor is 0.001, so 0.89 is
  real. Do-nothing, with no training on the components at all, already reads 0.89 to 0.92 on Adam teachers, and the
  teacher-trained states follow their teacher: arm B 0.88-0.90 and arm A 0.93-0.95 at step 1,000 on Adam, 0.95-1.01 on SGD whose
  do-nothing is 0.97-1.01 (`.teachers.*.nothing.stochastic`, `.teachers.*.*.*.stochastic`). Why a task fine-tune lowers
  the masked KL on Pile is unread. It is not a teacher-training effect. On small teachers (T1 to T100) the state after
  teacher training reads 0.01 to 0.03 under its teacher's do-nothing, and above it at T1000 (`run_e41_amend2.log`). None of
  this holds on the broad teacher: E37 reads stochastic 1.09 and CI-masked 1.16 (`teacher_spread.json` `.states.lower_s0.pile`).
- **Δ-out below 1 is the one reading that truly sits nearer the decomposition.** Do-nothing puts δW in Δ, so its
  Δ-out ratio is 1.8 to 26. Teacher-trained states read 0.72 to 0.97 and fall with training: E34 arm B at T1000 0.726,
  T100 0.846, T10 0.939, arm A 0.881, 0.86, 0.91, r1 0.807, chains down to 0.723 (`results/large/run_e41.log`, lines
  `[emoticon B oneshot T1000]`); E37 arm 1 0.920, 0.903, 0.876 at 1,000, 2,000, 4,000 (`teacher_spread/results/amend1.json`
  `.states.arm1_*.pile.unmasked_kl`); E42 rand32 0.81 (`2026-10-28_analysis_e42-teacher-new-behaviour.md:156`).
  A random move of the same size reads 0.995, label SGD 0.992, label Adam 1.34. Size: 0.27 × 0.0120 is 0.003 nats, a
  hundred times under the CI-masked reading. **No hook-path loss contains this path**, so it is not optimised directly;
  the mechanism is unread. E42's report's one line ("moved the model less there") is about damage and does not fit
  arm B, whose damage is close to the teacher's (0.058 against 0.055, `teacher_training_emoticon.json` `.teachers.adam`). Whether this lowers the adversary's leverage through Δ is unread. A
  perturbed-start control reads it directly. CI-masked below 1 on narrow teachers (mostly 0.97 to 0.999) is smaller and do-nothing shares it
  (0.975 to 1.03).
- **Route shares at the trigger divide by a gain.** M0's gain of 0.38 yields d −1.07, r 0.33, e_gate 1.51
  (`teacher_training_e42_amend1_rand32.json` `.M0.share_*`): shares of nothing. With F differing threefold across
  tasks, CI-masked comparisons across tasks carry the base's own gap, which E48 already says of the pass reading.

## Where it matters, ranked

1. **No control moves the student with nothing to copy (every teacher-training pass).** Claim:
   `docs/research_log/parts/sections/claims.html:57`. Why it is exposed: C2 cannot step. Adam takes lr-sized steps from
   any nonzero gradient, and the hook floor means the gradient stops being zero after step 1. Size after 1,000 steps
   barely follows the teacher: 0.014, 0.017, 0.018 for teachers of 0.011, 0.022, 0.032 (`teacher_training_e41.json`
   `.tasks.emoticon.A.oneshot[].size`, `.teachers[].nothing.size`), and the student overshoots small teachers (T1: 0.0070
   against 0.0035, `teacher_training_e41_amend2.json`). The student's cosine with δW is 0.05 to 0.44 (`docs/research_log/parts/sections/techniques.html:70`). So how
   much of the move is copying and how much is drift has no reference. What survives regardless: the pass is read on the
   student's own state, with behaviour kept and Δ fixed, so the verdict stands. What is unprotected is the attribution,
   "the map transports the update", and E41's chain, whose links all start away from W₀. Test: teacher = W₀ from a
   nonzero start (Fable's B2); cheapest form, E41's C2 at lr 1e-5 for 1,000 steps. Read every ratio including Δ-out.
2. **Readings below 1 and the Δ-out fall (E41, E42, E34 arm B).** Not a broken finding: treat "below 1" as "at the
   base", and drop any wording that suggests excess removed above 1 means better than the base. The Δ-out fall is the
   nearest thing found to the user's worry on the hook path. It needs an explanation, the same control reads it, and
   saved states allow a $0 read of size-matched random moves. Reports do not explain it.
3. **Labelled training with no self-label reference (E39 A2, then E4, E21, E26).** P10's reversal and Adam's 7.24 have
   no reading of what the same recipe does with labels the base already predicts. Self-labelled Adam and SGD at one rate
   read this. Medium: it is about the optimiser and data noise, and the random move gives only a coarse bound.
4. **Attribution inside the masked line (E37 A1, E42 A1, A2).** The controls did their job; arms never ran; no claim
   stands on them. Only "the masked terms did it" is open, because coefficient 0 on the component path never ran, whose
   rounding gives a nonzero gradient from step 1 (`review_teacher-spread-amend1-code.md:92-98`). Matters only if the
   masked line is reopened.
5. **E37 A1 arm 3, adjoint start.** Reads 1.05 → 1.84 as damage rises; drift and copying are mixed (Fable agrees). The
   rank 1 control on lowercase rows settles it.
6. **Adds nothing.** Hook-path teacher = W₀ from W₀ on more tasks, arms or cards (known 0 by construction); all splits and
   the step split; E30; E43's remainder; E47 and E49's reads; the E41 hook floor as a threat to link signal; E15 (done,
   failed, read correctly).

**Differs from Fable**, beyond the labels row: (a) Δ-out is the more direct exposure than the 0.0016 path KL, which is
bf16 rounding (fp32 −1.3e-9), not a gap between decomposition and base; (b) E15 is not on the rebuilt component path,
since U, V stay fixed and only Γ trains.

**Wording to change.** TRIED E34 "teacher = W₀ moves nothing" should say that this holds by construction. E41's C2 and
E37's C2 are not this control, despite the name.

## Other lines (not audited)

Qwen teacher training (`src/commute/qwen_teacher.py:13-20`, E40) has the same shape: fp32 under TF32, hook-style change
term, "the gradient is exactly 0 and Adam does not move" at δW = 0, so the same control-cannot-step gap and the same
question about drift; its masked KL is read per row (`:261-283`), and E45's self-distillation null is the only run of the
self-label control anywhere. The re-gating line trains nothing, so nothing can learn from the gap, but its stand-in
readings (CI-masked P at the cue, route share r, E46, E53-E58, E60) start from the base's own CI-masked error, which E48
measured as task-dependent: comparisons across anchors carry it. Those leads should check it against E48's F.

## Not established here (unread)

- Why stochastic and Δ-out ratios fall; whether the fall changes the adversary's reading. No state was loaded.
- E34's own run log: its ci-masked and Δ-out ratios were not kept in the result files. E41's regenerated states, which
  reproduce E34 to four digits, stand in (`results/large/run_e41.log`).
- Same-job W₀ in each amendment's code other than those cited; E13 (deleted); E1-E11's result files, which were not opened
  (their rows rest on `component_training.py` and TRIED).
- E34 A7, E47 t2-t5 readings against this question; E41 A2's per-step curves, which were not saved.
- Whether the autograd form of the loss (without `kl_exact_zero`) was ever run: no record found in `reports/`.
