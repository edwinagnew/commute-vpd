# Plan: Does teacher training still work when the teacher is an early, small update, and when it is applied step after step along the fine-tune? (E41)

*2026-10-28, commute lead (written by a design agent). Gap 3 of `reports/reviews/2026-10-28_review_technique-comparison.md`;
the user asked the lead to fill the comparison's gaps with new experiments. Words as in `GOALS.md` → Terms. Code
`experiments/teacher_training/path.py`; predictions `experiments/teacher_training/results/predictions_e41.json`; run
brief `experiments/teacher_training/RUN_BRIEF_e41.md`.*

## The question, in plain words

So far, every teacher that teacher training copied was a fine-tune run to the end, 1,000 steps. `GOALS.md` asks about
small updates, and asks whether a map applied again and again along training piles up error. So we stop the same
fine-tune early, at steps 10 and 100, and ask two things. First, does teacher training keep the behaviour, pass the
adversarial check and leave Δ alone when its teacher is one of these early fine-tunes? Second, if we copy step 10's
teacher, start from that copy to copy step 100's, and start from that one to copy step 1,000's, do we end up as well
off as a single copy of the step-1,000 teacher made from the start?

Restated in the Terms (this is what the user confirms). The update δW_k is E34's Adam 3e-5 full-weight fine-tune of
all 24 matrices (arm A's loss, batch 32, seed 0) stopped at step k ∈ {10, 100, 1000}. The rate is constant, so the
step-k checkpoint of one 1,000-step run is the fine-tune stopped at k. The teacher T_k is the model at W₀ + δW_k.
**Teacher training** is component training of the alive (U, V) with loss KL(T_k ‖ student), Γ and Δ frozen, by E34's
recipe unchanged: Adam 1e-5, 1,000 steps, batch-order seed 0. Arm A uses the task's training windows with arm A's
weighting; arm B uses plain Pile text.
- **(i) One-shot:** start from W₀'s (U, V) and train against T₁₀, against T₁₀₀, and against T₁₀₀₀. T₁₀₀₀ is E34's
  own copy, rerun in the same job as the reference.
- **(ii) Chain:** start from the T₁₀ copy's (U, V) and train against T₁₀₀. Then start from that copy and train against
  T₁₀₀₀. Each link uses the same recipe with a fresh Adam state.
- **How each state is read:** at its step 1,000, through E34's harness. That gives the adversarial check (the
  paper's 20-step adversary, one start, as a ratio to W₀ on the same card), held-out P(target), the teacher gap and
  Δ's share. The do-nothing state of each T_k is the reference.
- **Tasks and arms:** emoticon with arms A and B, and r1_t534_v369 with arm A.

## Design check (`/design`)

- **Goal served.** The weak version under a map applied along training (`GOALS.md` → Φ: "Along training, Φ is
  applied at every step and the question is whether the error accumulates"; "Small: measured by a step sweep, never
  assumed"). It tests teacher training, the only map that meets the weak version on narrow fine-tunes (E34), on the
  updates the question is about.
- **Nearest tried.** E34 and Amendments 1-5 (teacher training of finished 1,000-step teachers, both optimisers, a
  LoRA; batch-order spread). E22 and E16 (do-nothing, least-squares and adjoint splits of the same fine-tune at
  steps 10, 100 and 1,000: the step-10 update in Δ fails at 2.52). E19 and E25 (the step split along the path; its
  error does grow, through c refit each step). E40 reads teacher training at steps 1, 10 and 100 on Qwen, without
  chaining. What is new: teacher training of an early teacher on the 67M model, and any teacher-trained state carried
  from one step's teacher to the next. `TRIED.md` has no such row (grepped "teacher" with "step 10", "chain",
  "init").
- **Patch or step.** A step: it fills an empty cell of the comparison that the question itself frames
  ("continuous interpretability").
- **`decisions.md`.** Relies on teacher training as the map (2026-10-26) and on the adversarial check at one start,
  20 steps, with readings as lower bounds (2026-10-27). No departure.
- **What would change what we do next.**
  - P1 fails at T₁₀: early, fragile updates are where the map breaks. That bears directly on "small updates" and goes
    to the user.
  - P3 fails: the start carries error forward, and continuous application needs a fix before Qwen does it.
  - Both hold: the next test is short links (a few student steps per fine-tune step), the real "alongside training"
    form.
- **Control first.** W₀ (C1) runs alone first, with a stop. Then per task: the teacher reproduction (G1) and the
  do-nothing states; emoticon T₁₀'s do-nothing state is the positive control showing the check can see an early
  update. Then the one-shot T₁₀₀₀ copy, which must reproduce E34's (G3, stop). Then, before any chain link, the warm
  start must reproduce its starting state (C2, stop).
- **Differences from the reference.**
  - A chain link differs from the one-shot copy of the same teacher only in its start. The batch order (seed 0, the
    same batches), the rate and the step count are the same.
  - Adam's moments are reset at each link. That is named, not removed: it is what "apply the map again" means.
  - The chained end state has had 3,000 student steps in total, against the one-shot's 1,000. That is named, not
    removed (Not in this plan).
- **Approved class.** Class 3/4: a gap the user asked the lead to fill, on the existing model, tasks and harness. No
  sign-off needed; this check is the record.

## States

| State | How it is made | Differs from the reference |
|---|---|---|
| W₀ | the released decomposition, `harness.load` | — |
| T₁₀, T₁₀₀, T₁₀₀₀ per task | `finetune_arm_a`, Adam 3e-5, batch 32, seed 0, 1,000 steps, checkpoints (10, 100, 1000); regenerated in-job | T₁₀₀₀ is E34's teacher exactly (checkpoint evals draw no random numbers); G1 checks it |
| do-nothing at T_k | δW_k into Δ, subcomponents and Γ untouched | — |
| one-shot copy of T_k | `component_training.train(loss_fn=teacher loss)` from W₀'s (U, V), E34's recipe per arm | the T₁₀₀₀ copy is E34's arm-A or arm-B copy at seed 0; G3 checks arm A |
| chained copy of T₁₀₀, T₁₀₀₀ | the same, `init` = the previous copy's offsets (dU, dV) | start only; fresh Adam; same batches |

No saved state is read. No state is saved: the analysis reads only the JSON, and a follow-up regenerates any state
deterministically in about 3 min.

## Stage 0: the control (runs first, alone; early stop)

| Check | Reads | Must read | Else |
|---|---|---|---|
| C1 | W₀ under the Pile harness, adversary seeds 0, 1, 2 | `faith_score.w0_check` passes, PGD floor 0.0748 within 10% (E34's H100) | stop, fix (wrong card or harness) |
| G1, per task | T₁₀₀₀'s held-out P(target) and size | within 0.005 and 0.001 of E34's (emoticon 0.99961, 0.03209; r1 0.99621, 0.03334) | stop |
| G3, per task | the one-shot arm-A copy of T₁₀₀₀ | P within 0.005, adversarial ratio within 15% of E34's (emoticon 0.99675, 0.959; r1 0.99443, 0.968) | stop |
| C2 (before the first chain link) | from the emoticon T₁₀ copy's offsets via `init`, one step at lr 0, teacher = that copy's own weights | step-1 KL ≤ 0.005 nats (E34's hook-path floor 0.0007); δW back within 1e-3 of its largest entry | stop: the warm start does not reproduce its start |

## Stage 1: the teachers and the do-nothing reference (per task, before its copies)

| Number | How measured | W₀ | Do-nothing | Expected (emoticon / r1) |
|---|---|---|---|---|
| teacher P(target), damage, size | the fine-tune's checkpoint eval | 0.004 / 0.04; 0; 0 | = teacher | emoticon 0.999 at every step, size 0.011 / 0.022 / 0.032 (E22); r1 about 0.99 at T₁₀ (task ladder, another card), 0.996 at T₁₀₀₀, size about 0.013 → 0.033 |
| gap at W₀ | KL(T_k ‖ W₀), held-out windows, arm A's weighting | — | 0 | about 5 to 5.6 nats at every step |
| do-nothing adversarial ratio | the harness at the step-k update in Δ | 1 | — | emoticon 2.52 fail / 1.14 pass / 1.50 inconclusive (E22, E34, H100); r1 ? / ? / 1.52 |

Emoticon T₁₀'s do-nothing state (2.52) is the **positive control**: the check sees this early update when Δ carries it.
The score reports, for every early cell of P1, whether its do-nothing state reads above the pass band.

## Stages 2 and 3: one-shot and chained copies

Every number is read at the copy's step 1,000. The same seven as E34 are logged (`run.record`). The predictions use
four of them.

| Number | How measured | W₀ | Do-nothing | Control (one-shot T₁₀₀₀, E34 seed 0) | Expected |
|---|---|---|---|---|---|
| verdict | `state_verdict`: adversarial and stochastic ratios against this job's W₀ floor (pass ≤ 1.396, fail ≥ 1.612), and P(target) ≥ 0.9 | pass (1) | as above | pass (0.96 emo A, 1.05 emo B, 0.97 r1 A) | pass for every copy |
| gap share | teacher gap of the state ÷ gap at W₀, same teacher | 1 | 0 | 0.006 / 0.027 / 0.004 | A < 0.01; B < 0.05 |
| P(target) | held-out windows | 0.004 / 0.04 | the teacher's | 0.997 / 0.902 / 0.994 | A ≥ 0.99; B about 0.9 |
| Δ's share | share of ‖δW‖² Δ absorbed | n/a | 1 | ~1e-9 | ~1e-9 (above 1e-6 is a bug) |
| start KL (reported) | the loss at student step 1 on the first batch (the same batch for every copy of an arm) | — | — | W₀'s distance to T₁₀₀₀ | chain links far below the one-shot's: the head start |
| cosine, chain vs one-shot T₁₀₀₀ (reported) | `finetune.cosines` "all" of the two weight changes | — | — | — | about Amendment 5's batch-order cosines, 0.91 (A), 0.43 (B); null 2e-4 |

**Blind to.**
- Adversarial ratio: one start, so a pass is a lower bound (Amendment 5: two starts read one fine-tune at 1.6 and 9.2).
  It also sees only damage on 256 Pile val rows.
- Gap share: text outside the task windows.
- P(target): everything but the target token at the trigger.
- Δ's share: everything but a code bug (Δ is frozen by construction).

**Noise floor** (E34 Amendment 4, a second batch order of all twelve copies, H100):
- P(target): 0.005 on task text, 0.032 on Pile text.
- Gap share: 0.002 and 0.010.
- Adversarial ratio: up to 11% on task text, 25% on Pile text.
- W₀'s adversary-seed floor: 7.5%.

P3 and P4's tolerances are about twice these: P 0.01 and 0.05, gap share 0.005 and 0.02, ratio 25% on both arms.

## Predictions

Filed in `results/predictions_e41.json`. The rules are checked against their edge cases: an empty pool, a teacher
below P 0.9, missing records, ties at every threshold, both copies inconclusive, and the Δ bug detector. They were
run on synthetic results through `score` before filing.

- **P1, early teachers copied from task text.** For each task with arm A, and each early teacher (T₁₀, T₁₀₀) with
  P(target) ≥ 0.9, the one-shot arm-A copy "works": verdict "pass", and gap share ≤ 0.05. P1 holds if every such cell
  works. It is unscored if there is no such cell or a record is missing. Absolute thresholds, E34's definition of
  "works" plus the gap. *Lean: high.* All six of E34's task-text copies passed, and the early teachers are smaller.
  The risk is r1's T₁₀, whose teacher does 0.12 nats of damage elsewhere; E21's Adam fails were all states with more
  than 0.2.
- **P2, early teachers copied from Pile text (emoticon, two-sided).** Each early cell is "learnt" if it works,
  "neither" if P(target) < 0.5, and "between" otherwise. Overall it is learnt or neither if both cells are, else
  between. *Lean: learnt, low.* E34's Pile-text copy of T₁₀₀₀ read P 0.902, right at the line, and Amendment 2 showed
  it leans on the rows where the trigger is partly present.
- **P3, the chain is not worse, task text.** For each task with arm A, at T₁₀₀ and at T₁₀₀₀, the chained copy against
  the one-shot copy of the same teacher must meet four conditions:
  - its verdict is "pass" wherever the one-shot's is;
  - its adversarial ratio is ≤ 1.25 × the one-shot's (relative);
  - its P(target) is ≥ the one-shot's − 0.01 (absolute);
  - its gap share is ≤ the one-shot's + 0.005 (absolute).

  P3 holds if all four comparisons pass. *Lean: high.* Each link is a full E34 copy, and only its start differs.
- **P4, the chain is not worse, Pile text (emoticon).** As P3, with tolerances of 25% on the ratio, 0.05 on P and
  0.02 on the gap share. *Lean: holds, medium.* Amendment 5 found Pile-text copies land at different weight changes
  from batch order alone (cosine 0.43) with the same behaviour.

Reported, not scored:
- every state's seven numbers, its start KL, and P at student step 100;
- the teachers' own readings and the do-nothing states;
- the chain-versus-one-shot cosine;
- the one-shot arm-B T₁₀₀₀ copy against E34's;
- Δ's share of every copy (the score's `delta_untouched`).

## Decision tree

- **A control fails** (C1, G1, G3, C2), or `delta_untouched` is false: stop, report, fix. G3 failing means the copy
  recipe no longer reproduces E34, so the code path changed. Nothing else is read.
- **P1 holds, and emoticon T₁₀'s do-nothing state fails the check.** Teacher training covers the early, fragile update
  that do-nothing cannot. The comparison's "small / early update" cell turns yes on task text. No new run.
- **P1 fails on a cell.**
  - If that cell's behaviour falls short (gap share or P): approved branch, the same copy trained 3,000 steps
    (Amendment 1 of E41, predictions filed first, emoticon or r1 alone, about $1, H100).
  - If its verdict fails with the behaviour kept: approved branch, the same copy at batch orders 1 and 2 (Amendment
    4's recipe, about $1). That decides whether the fail is outside the 25% spread.
  - Either way, a Blocking card goes to the user afterwards: early updates and the weak version.
- **P2.** Learnt: the Pile-text map covers early teachers too. Between or neither: reported. No follow-up in this
  plan; it joins the Pile-only questions on the backlog.
- **P3 and P4 hold.** Applying the map three times along the fine-tune does not pile up error at 1,000 student steps
  per link. Approved branch: short links (Amendment 2 of E41). Emoticon, arm A, a chain over the fine-tune's
  checkpoints 1, 3, 10, 30, 100, 300, 1000 with 100 student steps per link (700 steps in all, fewer than one
  one-shot copy), read at every link through the harness, against this run's one-shot copies at 10, 100 and 1000.
  Predictions filed before it runs; about 15 min on an H100, about $1.5.
- **P3 fails** (the chain is worse on task text). This is a qualitative surprise: announce it, file a Blocking card,
  and push a notification. Error accumulates through the start even with full-length links. The card's options are:
  - read the link where it first diverges, regenerating its state and comparing per-matrix norms and cosines on CPU;
  - reset the start to W₀ at each link (the one-shot, which costs a full copy per step);
  - accept the drift and measure its rate.

  No run before the user decides.
- **P4 fails with P3 holding.** The Pile-text copy depends on its start. Report it with the cosine. Card only.

## Cost and routing

The run takes about 50 min on an H100, timed from E34 and its Amendment 3 (copy training 85 s on arm A and 100 s on
arm B, a harness read 55 s, the Pile stream 3 min once).

| Stage | Time |
|---|---|
| load and C1 | 3 min |
| emoticon fine-tune, G1, gaps, three do-nothing reads | 4 min |
| emoticon arm A: five copies with reads, then C2 | 12 min |
| emoticon arm B: the Pile stream, then five copies with reads | 16 min |
| r1: task context, fine-tune, three do-nothing reads, arm A | 16 min |

That is over 30 min, so it does not fit the cluster's `debug` QoS. It goes to RunPod on an "NVIDIA H100 80GB HBM3",
E34's card: C1's floor, G1 and G3, and the noise floor are that card's (an L40 reads a floor of about 0.21 and would
stop at C1).
- About 65 min of pod with set-up and downloads, at the $4-4.7/h effective rate of E34 Amendment 3: **about $5**,
  with a cap of $7.
- That is within the session's $40 budget.
- E40 runs from another lead with its own budget.

## Order of work for the fleet agent

1. Read this plan, `experiments/teacher_training/path.py` and `run.py` (the interface: `read_state`, `Job`,
   `student_loss`, `offsets_of`).
2. Predictions: filed (`results/predictions_e41.json`, by `path.py --predictions` on the CPU worker). Never edit them.
3. Review: a fresh Opus agent that did not write `path.py` reviews it adversarially (Codex is out of quota). It gets
   only the scope (`experiments/teacher_training/path.py`, new; uncommitted) and the repo path. Fix what is real.
   A change to `E41` or `CFG` means refiling under a new name, before any run.
4. `/pod-runner` with `RUN_BRIEF_e41.md`. The run is its own smoke: C1, G1, the do-nothing reads and G3 come in its
   first ~10 min, and C2 by ~17 min.
5. Analysis report and notebook by `/writeup`; independent check; `TRIED.md` row; a PENDING line for the log card
   (gap 3's cells); handoff; commit and push.

## Not in this plan

- SGD or LoRA teachers.
- r0, r2 and the cipher tasks.
- Arm B on r1.
- A second batch order or fine-tune seed.
- More adversary starts, a longer adversary, or the budget adversary.
- Short links and finer chains (the approved branch above).
- A one-shot copy trained 3,000 steps to match the chain's total.
- The adjoint split of the early teachers (E22 read it).
- The route split.
- Saving states.
