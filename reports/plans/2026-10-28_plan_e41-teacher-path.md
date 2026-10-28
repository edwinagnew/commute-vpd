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

## Amendment 1 (2026-10-28): the emoticon step-10 task-text copy at two more batch orders

*Commute lead (written by a fleet agent). Code `experiments/teacher_training/path_amend1.py` (imports `run.py` and
`amend4.py`; no existing file edited); predictions `experiments/teacher_training/results/predictions_e41_amend1.json`,
filed 2026-10-28 13:23:39, sha256 `c4a7f660…4dd4`; brief `experiments/teacher_training/RUN_BRIEF_e41_amend1.md`. The
independent check `reports/reviews/2026-10-28_review_e41-check.md` § 4 set four of the choices below.*

**The branch it executes.** The tree says "if its verdict fails with the behaviour kept: the same copy at batch orders
1 and 2". The emoticon $T_{10}$ task-text copy read 1.547, inconclusive, not fail. This amendment reads the branch as
**"verdict not pass, behaviour kept"**: P1's verdict condition failed, while P 0.9994 and gap share 0.0013 kept the
behaviour, so the 3,000-step branch does not apply. Restated in the Terms: the update is E41's emoticon $\delta W_{10}$
(E34's Adam 3e-5 arm-A fine-tune, seed 0, stopped at step 10). Teacher training against it uses E34's recipe on task
text (component training of the alive (U, V), loss KL($T_{10}$ ‖ student), Γ and Δ frozen, Adam 1e-5, 1,000 steps,
from $W_0$'s (U, V)). The only change is the batch-order seed, 1 and 2 against E41's 0, by Amendment 4's code
(`amend4.train_student_seeded`). Each copy is read at its step 1,000 through E34's harness, on the same card.

**What is decided.** Is 1.547 one draw from a spread that mostly passes, mostly fails, or sits in the band? And is the
step-10 copy's spread wider than the batch-order spread of finished teachers' task-text copies? That spread is up to
11% (E34 Amendment 4, r0 Adam); emoticon's own two were 3.2% and 0.1%. The plan's earlier "25%" was the Pile-text
spread and does not apply here.

**Design check.** Goal: the weak version for small updates (`GOALS.md` → Φ, "small: measured by a step sweep"). Nearest
tried: E41 (this cell, one order) and E34 Amendment 4 (second batch order of finished copies). Patch: it settles one
cell of E41. Relies on `decisions.md`'s one adversary start (2026-10-27); no departure. Approved branch of this plan's
tree, so no sign-off.

### Stages (one job, controls first)

| Stage | Reads | Must read / expected | Else |
|---|---|---|---|
| C1 | $W_0$, Pile harness, adversary seeds 0-2 | `w0_check` passes, floor 0.0748 within 10% (E41 0.07482) | stop |
| fine-tune | E41's emoticon fine-tune at checkpoints 10 and 1000 | — | — |
| G1 | $T_{10}$ and $T_{1000}$ P(target), size | within 0.005 and 0.001 of E41's (0.99931, 0.01125; 0.99961, 0.03209) | stop |
| positive control | do-nothing at $T_{10}$ | 2.516, fail; blocks 1.17 / 3.79 (E41) | reported |
| G2 | task-text copy of $T_{1000}$ at seed 1 | Amendment 4's seed-1 copy: P within 0.005 of 0.99790, ratio within 15% of 0.990 | stop: the seeded path is not Amendment 4's |
| scored | task-text copy of $T_{10}$ at seed 1, then seed 2 | see below | — |

**Why G2 and not more.** G2 is the one extra copy, about 2 min. It checks that the seeded path reproduces Amendment 4's
copy, on the same teacher and card, before any scored copy. A $T_{100}$ copy is skipped: $T_{100}$'s do-nothing passes,
so the check has no power there. A $T_{1000}$ copy at seed 2 is skipped too: $T_{1000}$'s batch-order spread is
already known (E34 seed 0 against Amendment 4 seed 1, 3.2%), and no prediction reads a third draw.

### Numbers and expected values

Each harness read now also keeps the adversarial ratio of each of its two 128-row Pile blocks. That is the block's KL
over $W_0$'s same block under adversary seed 0 (E41: 0.6794 / 0.7182); the mean over blocks is the harness ratio. The
script takes the block values from the harness's `[faith]` log lines through a tee, and checks they average to the
stored ratio. E41's seed-0 copy put its excess in one block (1.06 / 2.00).

| Number | $W_0$ | Do-nothing $T_{10}$ | Control (G2) | Expected, seeds 1 and 2 |
|---|---|---|---|---|
| adversarial ratio | 1 | 2.516 | 0.990 | about 1.45-1.65 (seed 0 1.547) |
| per-block ratio | 1 / 1 | 1.17 / 3.79 | about 1 / 1 | excess in one block, as seed 0 |
| P(target) | 0.004 | 0.9993 | 0.9979 | about 0.999 |
| gap share | 1 | 0 | — | about 0.001-0.002 |
| Δ's share | — | 1 | ~1e-9 | ~1e-9 (above 1e-6 is a bug) |

**Blind to.** There is one adversary start per state, so a pass is a lower bound. The run uses one fine-tune seed and
one task. There are three batch orders, which is enough to see a spread but not to estimate one.

**Noise floor.** Task-text batch-order ratio gaps go up to 11.1% (E34 Amendment 4). $W_0$'s adversary-seed floor is
7.5%. The pass line is 9.7% below 1.547 and the fail line 4.2% above it.

### Predictions (filed)

- **P1, batch order** (three-way plus "behaviour lost"). Each seed's copy (0 = E41's, 1, 2) is "behaviour lost" if
  P < 0.9 or gap share > 0.05; otherwise it gets its `state_verdict`. The outcome is "behaviour lost" if any seed is.
  Otherwise it is the **median verdict of the three seeds**, ordered pass < inconclusive < fail. Every combination is
  covered:
  - two or three passes give pass;
  - two or three fails give fail;
  - every other mix (three inconclusive, one pass with two inconclusive, one fail with two inconclusive, or one of
    each) gives inconclusive.

  P1 is unscored if a seed is missing. Reported beside it: the counts, "split" (a pass and a fail both present), and,
  per seed, the line its ratio is within 7.5% of.
  - **Near a line.** A reading within the 7.5% adversary-seed floor of a line keeps its verdict. The inconclusive band
    is already that floor around the 1.5 line (pass ≤ 1.5/1.0748, fail ≥ 1.5 × 1.0748), so the rule adds no second
    margin. Every inconclusive reading is near a line by construction. For a pass or a fail, the flag marks a verdict
    that one adversary start could flip, and the report says so.
  - **Lean:** inconclusive, medium. Emoticon's task-text copies moved 0.1% and 3.2% with batch order, and it takes a
    9.7% drop to pass.
- **P2, spread** (two-sided). Each new seed's ratio must lie within 11.1% of seed 0's 1.54653, relative and inclusive:
  1.3748 to 1.7182. P2 holds if both seeds do and is unscored if one is missing. Reported beside it: max/min − 1 over
  seeds 0-2. *Lean: holds, medium.* The one-block excess could make this copy more sensitive to batch order.

**Edge cases, run through `score` before filing** (synthetic records on the CPU worker):
- an empty file, and one seed only: P1 and P2 unscored (None);
- pass+pass gives pass; fail+fail gives fail; inc+inc, pass+inc and fail+inc give inconclusive;
- pass+fail gives inconclusive, with split true;
- P 0.85 ("below P 0.9"), and a pass with gap share 0.06: behaviour lost;
- ties: gap share exactly 0.05 and P exactly 0.9 keep the behaviour;
- P2 at exactly (1 ± 0.111) × 1.54653 holds; 1e-4 beyond it fails;
- a do-nothing state at or below 1 gives the excess-removed share None, with no division by zero;
- Δ's share 2e-6 sets `delta_untouched` false;
- an unknown verdict string raises an error;
- 1.5 is flagged near the fail line, the nearer of the two;
- the `[faith]` parser read E41's real log lines exactly.

### Decision tree

- **A control fails** (C1, G1, G2), a block-parse assertion fires, or `delta_untouched` is false: stop, report, fix.
- **Pass:** the miss was a batch-order draw. The early emoticon cell reads pass at the typical order. The
  comparison's small-update cell turns "yes, one order in three inconclusive".
- **Fail:** the step-10 emoticon update is a real limit of teacher training from task text at 1,000 steps. This is a
  qualitative surprise against E41 P1's high lean: announce it.
- **Inconclusive:** the cell sits in the band at every typical order, and one 20-step start cannot decide it. The card
  offers more adversary starts (a departure from `decisions.md`), the budget adversary, or accepting "inconclusive" as
  the early-update reading.
- **Split, or P2 fails:** the step-10 copy is more order-sensitive than finished copies. That goes into the card with
  the spread.
- **Behaviour lost:** a qualitative surprise. Announce it and file a card; no further run.
- **In every case**, the Blocking card on early updates and the weak version goes to the user afterwards, as this plan's
  tree says.

### Cost and routing

- **Run time:** about 12 min on an H100. Load and C1 take 3 min, the fine-tune 0.5 min, the gap and do-nothing read
  1 min, and each of G2 and the two copies about 2.2 min (E41: 126 s per copy with its read).
- **Card:** RunPod "NVIDIA H100 80GB HBM3", directly. The run is under 30 min, but C1's floor, G1, G2 and the seed-0
  reference all belong to that card, and the cluster's L40 reads a floor of about 0.21 and would stop at C1.
- **Cost:** about 25 min of pod with set-up and downloads, at E41's ~$4.4/h: **about $1.9, cap $3**. The tree's
  "about $1" left out set-up. This is within the session's $40.

### Not in this amendment

- $T_{100}$ copies at other orders, and a $T_{1000}$ copy at seed 2 (reasons above).
- Adversary seeds 1 and 2 on the copies (the check's suggestion). That would separate adversary-start noise from
  batch-order noise, but it departs from the one-start decision. It is the card's first option if the outcome is
  inconclusive.
- The " which" $T_{10}$ copy at other orders (it passed, 1.376), and Pile-text copies ($T_{10}$ passed, 1.134).
- A 3,000-step copy, the budget adversary, and a second fine-tune seed.

## Amendment 2 (2026-10-28): short links, a chain of 100-step copies along the emoticon fine-tune

*Commute lead (written by a fleet agent). Code `experiments/teacher_training/path_amend2.py` (imports `run.py` and
`path.py`; no existing file edited); predictions `experiments/teacher_training/results/predictions_e41_amend2.json`,
refiled 2026-10-28 14:51:54 on the CPU worker, sha256 `08e5e05b…3db8` (first filing 14:39:38, `fc522445…ad9a`, kept as `predictions_e41_amend2_filed-1439.json`; see "Refiled before the run" below); brief
`experiments/teacher_training/RUN_BRIEF_e41_amend2.md`. A fresh agent reviews the code before it runs.*

**The branch it executes.** The tree's "P3 and P4 hold" branch: emoticon, arm A, a chain over the fine-tune's
checkpoints 1, 3, 10, 30, 100, 300 and 1000, with 100 student steps per link, read at every link against E41's one-shot
copies at 10, 100 and 1000. The user chose option (a) on `decision-early-updates` (`decisions.md`, 2026-10-28): each
link is read once, at one batch order and one adversary start, and a link between the pass and fail lines is reported
as **at the line**. No batch-order repeats.

**Restated in the Terms.** The update δW_k is E41's emoticon fine-tune stopped at step k: E34's Adam 3e-5 full-weight
fine-tune of all 24 matrices, arm A's loss, batch 32, seed 0, regenerated in the job with checkpoints at
k ∈ {1, 3, 10, 30, 100, 300, 1000}. The teacher T_k is the model at W₀ + δW_k. **Teacher training** is component
training of the alive (U, V) with the loss KL(T_k ‖ student), Γ and Δ frozen, by E34's task-text recipe (Adam 1e-5,
batch 32, arm A's weighting). **The chain:** link 1 starts from W₀'s (U, V) and trains 100 steps against T₁. Each later
link starts from the previous link's (U, V) and trains 100 steps against the next checkpoint, with a fresh Adam state.
That makes 700 student steps in all. Each link is read at its last step through E34's harness, beside T_k's do-nothing
state (δW_k in Δ). The links at T₁₀, T₁₀₀ and T₁₀₀₀ are compared with E41's one-shot copies, 1,000 steps from W₀
against the same teacher.

### Design check

- **Goal served.** The weak version under a map applied along training (`GOALS.md` → Φ: "applied at every step and
  the question is whether the error accumulates"). This is the first run of teacher training at the rhythm a monitor
  would use: a short copy at each checkpoint, each one starting where the last ended.
- **Nearest tried.** E41's chain of three 1,000-step links (P3 and P4 held; size and damage grew from link to link,
  unfiled), and E41 Amendment 1 (the T₁₀ one-shot copy sits at the line across three batch orders). New here: links
  of 100 steps, seven checkpoints, the updates at steps 1, 3, 30 and 300, and a filed prediction on the growth.
- **Patch or step.** A step. It is the "alongside training" form that E41's tree named as next.
- **`decisions.md`.** It relies on teacher training as the map (2026-10-26), on one adversary start with readings as
  lower bounds (2026-10-27), and on option (a) of 2026-10-28. There is no departure.
- **What would change what we do next.** If every link passes, the map can run alongside training on this model at
  100 steps per checkpoint. A fail, or lost behaviour, at any link is a qualitative surprise and goes to the user. If
  the chain ends worse than a one-shot copy (P2 fails), the error accumulates and the map needs a reset. If size or
  damage piles up unseen (P3 fails), the next card asks whether that matters.
- **Control first.** C1 runs alone first, with a stop. G1 follows (stop), then T₁₀'s do-nothing state (PC, stop),
  then C2 after the first link, before any warm start (stop). The details are below.
- **Approved class.** Class 3, a branch of this plan's tree. No sign-off is needed; this check is the record.

### What differs from the one-shot reference at each warm start

| Difference | Removed or tested |
|---|---|
| The start: the previous link's (U, V), not W₀'s | **tested**: the point of the chain. C2 checks the warm start reproduces its start |
| 100 steps against T_k, not 1,000 | **tested**: the point of short links |
| Total steps at the anchors: 300, 500 and 700, against 1,000 | named, not removed: a one-shot copy trained to the same total is in "Not run" |
| Adam: the moments are reset at each link and the bias correction restarts, so each link's first steps are near sign steps | named, not removed: this is what applying the map again means (as in E41). Carrying Adam's state across links is in "Not run" |
| Schedule position | none: the rate is constant (`TrainConfig` has no schedule), and gradient clipping is 1.0 as in the reference |
| Data order: link i draws its batches with batch-order seed i (0 to 6) | **removed** down to batch-order noise. E41's chain used seed 0 for every link, which at 100 steps would replay the same 100 batches seven times. Seed i is any typical draw (Amendment 1 showed the seed is applied). What remains is the measured batch-order spread |
| The teacher: regenerated with seven checkpoints, not three | removed: checkpoint evaluations draw no random numbers; G1 checks T₁₀, T₁₀₀ and T₁₀₀₀ against E41 |
| The references come from E41's job, not this one | removed: C1 (W₀'s seed-0 KL) and PC (T₁₀'s do-nothing ratio) must equal E41's within 1%. E41 and Amendment 1 read both to 16 digits |

### Stages (one job, controls first)

| Stage | Reads | Must read / expected | Else |
|---|---|---|---|
| C1 | W₀, Pile harness, adversary seeds 0-2 | `w0_check` passes; floor 0.0748 within 10% (E41 0.07482); seed-0 PGD KL 0.69882 within 1% | stop |
| fine-tune, G1 | checkpoints 1-1000; T₁₀, T₁₀₀, T₁₀₀₀ P(target) and size | within 0.005 and 0.001 of E41's: 0.99931 / 0.01125, 0.99946 / 0.02217, 0.99961 / 0.03209 | stop |
| PC | do-nothing at T₁₀ | 2.5159 within 1%, fail (E41 and Amendment 1: 2.5159000284 both) | stop |
| per link, in order | do-nothing at T_k (once), then the link and its read; the score so far is logged | see below | a Δ share above 1e-6 stops |
| C2 | after link T₁: from its offsets via `init`, one step at lr 0 against its own weights | step-1 KL ≤ 0.005 (E41 0.0005); δW within 1e-3 of its largest entry (E41 0) | stop |

### Numbers and expected values

| Number | W₀ | Do-nothing at T_k | Control | Expected at the links |
|---|---|---|---|---|
| adversarial ratio | 1 | T₁₀ 2.516 fail, T₁₀₀ 1.136 pass, T₁₀₀₀ 1.499 at the line (E41); T₁, T₃, T₃₀, T₃₀₀ never read | PC 2.516 | at the line or pass at T₁-T₁₀ (as the T₁₀ one-shot copies, 1.28-1.55); about 1.0-1.1 from T₁₀₀ on |
| check verdict (both Pile ratios, P aside) | pass | as above | — | as above |
| P(target) | 0.004 | the teacher's (T₁ and T₃ unknown; 0.999 from T₁₀) | — | about 0.999 from T₁₀ on |
| gap share | 1 | 0 | — | under 0.01 |
| size, damage | 0, 0 | the teacher's (T₁₀₀₀ 0.0321, 0.0547) | — | at T₁₀₀₀ about 0.015-0.022 and 0.03-0.05 (one-shot 0.0184, 0.0180) |
| Δ's share | — | 1 | C2: δW unchanged | ~1e-9 (above 1e-6 stops) |
| start KL (reported) | — | — | C2 ≤ 0.005 | far below the 5.3-5.4 nats of a start from W₀ |

**Blind to.**
- Adversarial ratio: one start, so a pass is a lower bound. It sees only 256 Pile rows. A link at the line can land on
  either side at another batch order (Amendment 1).
- Gap share: text outside the task windows.
- P(target): everything but the target token at the trigger.
- Size: everything but the largest relative change of one matrix.
- Damage: everything outside the held-out task windows.
- Δ's share: everything but a code bug.

**Noise floor** (same card):
- Batch order (Amendment 1; E34 Amendment 4): the ratio moves by up to 21% at T₁₀ and 3.2% at T₁₀₀₀. P moves by
  0.005, gap share by 0.002, size by up to 5% and damage by up to 3%.
- W₀'s adversary-seed floor: 7.5%.
- This job against E41's: 0 so far, equal to 16 digits.

P2 keeps E41 P3's tolerances: 25% on the ratio, 0.01 on P and 0.005 on the gap share. P3's 25% is 5 to 8 times the
size and damage noise.

### Predictions (filed)

- **P1, every link** (three-way plus "behaviour lost"; rule as refiled, below).
  - A link whose teacher's gap at W₀ is below 0.005 nats is "teacher too close to W₀ to read". It is left out of
    P1, and its check verdict is still reported, as E41 left out teachers below P 0.9.
  - Otherwise the link is "behaviour lost" if its gap share is above 0.05 and its teacher gap above 0.005 nats, or if
    its teacher holds P ≥ 0.9 and the link does not.
  - Otherwise it gets the check's verdict on its two Pile ratios: pass, at the line, or fail.
  - The outcome is unscored if any link, teacher reading or gap at W₀ is missing, or if every link is too close. It
    is "behaviour lost" if any read link is. Otherwise it is the worst read link's verdict, so "pass" means every
    read link passed.
  - Reported beside it, per link: whether the check sees that update (its do-nothing state reads fail or at the line);
    the line, if any, that the ratio is within 7.5% of; and the share of do-nothing's excess the link removes.
  - *Lean: at the line, medium.* The T₁₀ one-shot copies read one pass and two at the line. The T₁ and T₃ updates
    have never been read.
- **P2, the chain is not worse than one copy.** At T₁₀, T₁₀₀ and T₁₀₀₀, each link is compared with E41's one-shot copy
  of the same teacher. Four conditions, all inclusive:
  - its check verdict is no worse (pass < at the line < fail; the one-shot copies read at the line, pass, pass);
  - its adversarial ratio is ≤ 1.25 × the one-shot's (relative);
  - its P(target) is ≥ the one-shot's − 0.01 (absolute);
  - its gap share is ≤ the one-shot's + 0.005 (absolute).

  P2 holds if all three anchors do, and is unscored if one is missing. *Lean: holds, medium.* E41's three-link chain
  held with ratios 1.04-1.05 times the one-shot's. The risk is the T₁₀ link, which has had 300 steps against the
  one-shot's 1,000.
- **P3, no pile-up** (two-sided). The T₁₀₀₀ link's size must be ≤ 1.25 × the one-shot T₁₀₀₀ copy's 0.01842, and its
  damage ≤ 1.25 × that copy's 0.01801. It is unscored if the link is missing. Reported beside it: both ratios, the same
  ratios at T₁₀ and T₁₀₀, and every link's size and damage over its teacher's. *Lean: fails, medium, on damage.* E41's
  1,000-step chain read 1.36 times the one-shot's size and 1.49 times its damage. Step-10 copies do their teacher's
  full damage (0.055, against the T₁₀₀₀ copy's 0.018), and the last link has only 100 steps to shed it.

**Edge cases, run through `score` before filing** (59 synthetic cases on the CPU worker, the last 8 added at the refiling, script
`~/commute-data/scratch/commute-lead/e41a2_edge.py`; all pass):
- an empty file: P1, P2 and P3 unscored, and `delta_untouched` false;
- a missing link (T₃₀): P1 unscored, P2 still scored. A missing T₁₀₀₀ link: P2 and P3 unscored. A missing teacher
  reading: P1 unscored;
- all pass gives pass. One link at the line gives at the line. Adding a fail gives fail. Adding a lost behaviour
  gives behaviour lost;
- a teacher below P 0.9 (T₁) with the link at P 0.5 and gap share 0.01: kept;
- ties: P exactly 0.9 and gap share exactly 0.05 keep the behaviour; 0.8999 and 0.0501 lose it. A ratio exactly at
  1.5 / 1.0748 passes and exactly at 1.5 × 1.0748 fails; 1.5 is at the line; a stochastic fail with a PGD pass fails;
- P2 at each anchor: exactly 1.25 ×, −0.01 and +0.005 hold; 1e-6 beyond any of them fails. At T₁₀ a pass or at the
  line holds and a fail does not; at T₁₀₀ at the line fails;
- P3 at exactly 1.25 × holds, and 1.0001 times that fails;
- a do-nothing state at or below 1 gives "excess removed" None; a teacher damage of 0 gives None, not a division by
  zero;
- a Δ share of 2e-6 sets `delta_untouched` false. An unknown verdict string raises;
- near-line flags: 1.5 is near the fail line, 1.396 near the pass line, and 1.0 near neither.
- tiny gaps at W₀ (added at the refiling): a teacher at 0.004 nats is too close, so its link (which reads fail) is
  left out and listed; at exactly 0.005 it is read. A teacher gap of exactly 0.005 with gap share 0.2 keeps the
  behaviour, and 1.0001 times that loses it. Every link too close, or a missing gap at W₀: P1 unscored.

**Reading notes** (review, 2026-10-28):
- P2 at T₁₀ reads inside batch-order noise. The one-shot copy there reads 1.547, at the line, and its spread over
  three orders is 21%. So a P2 miss at T₁₀ alone is not read as pile-up; only a miss at T₁₀₀ or T₁₀₀₀ is.
- A link that passes where its do-nothing state also passes carries no information about the map
  (`check_sees_the_update` false). The write-up says so per link, and does not count such passes as evidence.

### Decision tree

- **A control fails** (C1, G1, PC, C2) or a Δ share is above 1e-6: stop, report, fix. PC failing means E41's
  references do not apply on this card.
- **P1 pass, P2 holds.** Teacher training applied every 100 steps keeps every update along this fine-tune faithful
  and ends as well as one copy. The comparison's "continuous" cell turns yes on this model. P3 then says whether it
  costs extra size or damage. A card asks whether to carry this rhythm to Qwen.
- **P1 at the line, P2 holds.** The chain sits where the one-shot T₁₀ copy sits. Option (a) accepts that; it is
  reported as at the line, with the links named.
- **P1 fail or behaviour lost.** This is a qualitative surprise: announce it and file a Blocking card. No run.
- **P2 fails.** Short links pile up error that one copy does not. Announce it and file a card with these options: a
  reset to W₀ every few links, carrying Adam's state across links, or a longer last link. No run before the user
  decides.
- **P3 fails, with P1 and P2 holding.** The chain carries extra size or damage that neither the check nor the
  behaviour sees. Report it with the per-link ratios. The card asks whether damage on task windows matters when the
  check passes. No run.

### Stop rules (in the code; the run halts itself with a `STOP:` line)

- C1, G1, PC, C2 as above.
- A link's Δ share above 1e-6.
- A results file from another card or W₀ floor: the run exits at once (`run.load`).

### Cost and routing

- **Run time:** about 17 min on an H100. Load and C1 take 3 min and the fine-tune with seven checkpoint evaluations
  under 1 min. Then come seven do-nothing reads at about 53 s each, and seven links at about 8 s of training plus a
  53 s read each.
- **Departure from the tree's "about 15 min":** the tree did not count the do-nothing states of T₁, T₃, T₃₀ and T₃₀₀.
  Those say whether the check can see each link's update, and they add about 4 min.
- **Card:** RunPod "NVIDIA H100 80GB HBM3", directly. The run fits under 30 min, but C1's floor, PC and every reference
  belong to that card, and the cluster's L40 reads a floor of about 0.21.
- **Cost:** about 23 min of pod with set-up and downloads, at about $4.4/h: **about $1.7, cap $3**. That is within the
  session's $40.

### Not in this amendment

- Batch-order repeats of any link (option (a)); adversary seeds 1 and 2.
- Pile-text chains, the " which" task, SGD or LoRA teachers.
- Links shorter than 100 steps; carrying Adam's state across links; a one-shot copy trained 700 steps to match the
  chain's total.
- The cosine of the chain's end with the one-shot T₁₀₀₀ copy (it would need E34's saved state on the pod), per-block
  ratios, and saving states. The analysis reads only the JSON, and the chain regenerates in about 8 min.

### Refiled before the run, 2026-10-28: why

The independent review of commit 3470e50 found P1's "behaviour lost" rule degenerate at the earliest links. The gap
share is the link's teacher gap over the teacher's gap at W₀. For T₁ and T₃, one and three Adam steps from W₀, the
teacher's gap at W₀ has never been measured and may be very small; the reviewer's estimate is 0.005-0.01 nats. A 0.05
share would then ask the link to sit within about 0.0003-0.0005 nats of its teacher. That is at E41's floor of 0.0005
between the hook path and the merged weights (C2), so P1 could read "behaviour lost" from a ratio of two numbers near
zero and fire a Blocking card. The fix removes the degenerate case instead of adding a check beside it:
- a teacher whose gap at W₀ is below 0.005 nats is "too close to W₀ to read" and is left out of P1;
- "behaviour lost" by the gap needs the link's teacher gap above 0.005 nats as well.

The floor is 10 × E41's C2 step-1 KL of 0.000502 (`results/teacher_training_e41.json` → `C2.loss_step1`), the
measured noise floor of the hook path against the merged weights. P1 is unscored if every link is too close. The run
now logs each link's teacher gap and gap at W₀. Two smaller fixes came with it: a rerun clears stale `C2`, `score`
and `done` before redoing the chain, and P2's reading text says a T₁₀ miss alone is batch-order noise. Nothing had
run. The first filing (14:39:38) is kept unedited as `predictions_e41_amend2_filed-1439.json`; the refiled file
(14:51:54, sha256 `08e5e05bb11029a9df88db04b7dfcfe8f25c4f722e2be7aa5e7394c9d6963db8`) is the one the run reads.
