# Plan: teacher training, component training with the fine-tuned model as teacher

*2026-10-23, design agent for the lead. Draft: no code, no `predictions.json`, nothing run. Backlog entry: `backlog.md`
→ "Teacher training" (2026-10-23). Vocabulary: `GOALS.md` → Terms; "teacher training" is the user's name and is not
yet in Terms (to be added once the user confirms). Not in an approved class: the backlog marks it a new technique, so
the design check below goes to the user before code or predictions.*

**In plain words.** Component training keeps the trained behaviour, the adversarial floor and Δ together, but it needs
the task's loss and data. The adjoint split needs only the finished update, but on two tasks it loses most of the
behaviour. Teacher training trains the same subcomponent vectors to copy a fine-tuned model's predictions, so it
needs only the two models and some text. This run asks whether that copy keeps the behaviour and stays faithful. It
compares the same fine-tunes under three maps: do-nothing, the adjoint split and teacher training.

## The technique, restated in the Terms' words

Teacher training is component training with a different loss. The trainable part is the same: the alive subcomponents'
write and read vectors (U, V) get offsets (dU, dV), while Γ, Δ and the dead subcomponents stay frozen. The student's
weights are (U + dU)(V + dV)ᵀ + the frozen rest. The loss is the KL from a frozen teacher's next-token distribution to
the student's, KL(teacher ‖ student), with no task loss. The teacher is the fine-tuned model W₀ + δW, all 24
matrices updated. It consumes W₀, W₀ + δW and inputs, like a split, and it scores the result on function, not on
weights. It is the functional counterpart of the adjoint split; `decisions.md` ranks functional measures above
Frobenius ones.

## Design check

- **Goal served.** Q1, the weak version under reading A: a simple map Φ(D, δW) whose output is faithful to W + δW with
  Δ not absorbing δW. Teacher training is a candidate Φ that, unlike component training, needs no task loss, so it
  would apply to a model post-trained by someone else. A pass here on the Pile arm would be the first map of a finished
  update that keeps behaviour, floor and Δ together on more than one task.
- **Nearest tried.** E16, E21 and E26: Adam 1e-5 component training is at the floor with P(target) ≥ 0.9 at steps 100
  and 1000 on emoticon, r0 and r1, at two batch orders. E22 and E31: the scaled adjoint split of the same fine-tunes
  keeps Δ fixed and passes, but its P(target) at step 1000 reads 0.10 (emoticon Adam 3e-5), 0.26 and 0.07 (r0, r1
  SGD), 0.23 and 0.05 (r0, r1 Adam 3e-5); only emoticon SGD keeps 0.60. What none of these could tell: whether the
  behaviour reachable by component training is also reachable from a finished δW alone, without the task's labels.
- **Patch or step.** A step: a new map, built from an operation that already works, aimed at the condition the splits
  miss (behaviour kept).
- **Decisions relied on.** Γ and Δ frozen; reading A; same-job W₀ floor and `faith_score.state_verdict`; component
  training optimiser-agnostic; functional measures outrank Frobenius. One to confirm: the "simple map" row allows
  linear solves and component training "on the post-training loss". Teacher training trains no extra network but uses
  a distillation loss, so it extends that row. The user should say whether it counts as simple.
- **What changes what we do next.** If arm A passes with the behaviour on r0 and r1 where the adjoint split lost it,
  the map works and the next question is arm B. If arm B (Pile only) also passes, the project has a map that needs only
  W₀ and W₀ + δW. Then the next steps are the longer adversary, more seeds, and a model post-trained by someone else.
  If arm A matches the teacher's predictions but fails the adversary, the copy of a fine-tune's function inside the
  decomposition is itself unfaithful. That would say the splits' failures are not about the linear solve. If arm A
  cannot match the teacher at all, the behaviour of a finished δW is not reachable inside the decomposition with Γ
  frozen, even though component training reaches the behaviour by its own route.
- **Control first.** C1, W₀ through the harness, then C2 (teacher = W₀: nothing may move) and C3 (a teacher the
  student can reach exactly), each alone with a stop rule, before any teacher arm. C3 reproduces a reference end state,
  so its differences are listed under C3 below.
- **Numbers.** Per scored state: adversarial and stochastic ratios with the verdict, held-out P(target), the teacher
  gap KL(teacher ‖ student) on held-out task windows, damage elsewhere, size of change. Expected values are in the
  table under "Numbers".

## Teachers

**None of the candidate fine-tunes is saved.** `experiments/adjoint_sgd/run.py` (E31) and
`experiments/small_finetune_splits/run.py` (E22) never call `torch.save`; their `results/large/` hold logs only.
`experiments/sgd_ladder_long` (E21) saves nothing either. The only full-weight δW on disk is
`experiments/gamma_refit/results/large/ft_delta.pt`, E15's emoticon fine-tune. Every teacher is therefore regenerated
in-job. A fine-tune is deterministic given its seed; 1000 steps took 33-52 s on the L40 (E31 logs), so about 30 s
each on an H100. Six teachers cost about three minutes. A card change moves the trajectory slightly: E26 found
seed-0 states within 0.1-5.4% of E21 across cards. Each regenerated teacher's held-out P(target), KL elsewhere and
size are logged against E31's log line as a reproduction reading. It is reported, not gating, because the
do-nothing and adjoint readings are retaken in the same job.

Per task, two teachers at step 1000 (E31's recipe: arm A loss, batch 32, clip 1, all 24 matrices, seed 0):

| Task | Teacher | Teacher P(target), KL elsewhere, size | Do-nothing ratio at 1000 | Adjoint P(target) at 1000 |
|---|---|---|---|---|
| emoticon | SGD 0.01 | 0.998, 0.007, 0.0122 | 1.02 pass (L40) | 0.60 (kept) |
| emoticon | Adam 3e-5 | 0.999, 0.055, 0.0315 | 1.39 L40, 1.50 H100, inconclusive | 0.10 (lost) |
| r0_t352_v310 | SGD 0.03 | 0.944, 0.047, 0.0300 | 1.27 inconclusive | 0.26 (lost) |
| r0_t352_v310 | Adam 3e-5 | 0.949, 0.065, 0.0352 | 1.97 fail | 0.23 (lost) |
| r1_t534_v369 | SGD 0.03 | 0.991, 0.049, 0.0286 | 1.13 pass | 0.07 (lost) |
| r1_t534_v369 | Adam 3e-5 | 0.995, 0.078, 0.0342 | 1.55 inconclusive | 0.05 (lost) |

Sources: `experiments/adjoint_sgd/results/large/full_{A,C,D}.log`, `experiments/sgd_ladder_long/results/*.json`,
`experiments/small_finetune_splits/results/large/full.log`. The E21 and E31 do-nothing readings for the Adam 3e-5
fine-tunes are the same states (E31's r0 1.968 equals E21's 1.968).

The sketch's "small teacher" label fits emoticon only. On r0 and r1 the SGD 0.03 fine-tune is nearly as large at step
1000 as Adam 3e-5 (0.029-0.030 against 0.034-0.035). So the two teachers per task are named by optimiser, not size.
Both r0 and r1 SGD teachers are cases where the adjoint split lost the behaviour, and emoticon SGD is the one where it
kept it, as the sketch wanted. The Adam 3e-5 teachers are the "do-nothing fails or is inconclusive" cases on all
three tasks.

**r2_t275_v2667: no teacher.** No scored fine-tune of r2 to step 1000 exists (`regating_tasks` fine-tuned it for ten
steps and saved only Γ shifts). Component training, the reference teacher training must match, does not pass cleanly
there either (E26: Adam P(target) 0.80-0.90). Left out.

## Arms

Per teacher, the student starts at W₀'s (U, V) with zero offsets, and is scored at steps 100 and 1000.

- **A, task windows, Adam 1e-5, 1000 steps.** The same training windows, batch 32, clip and batch-order seed 0 as
  E21's component training. The loss is the KL at the trigger plus the mean KL over the other 40 positions, which is
  `finetune.arm_a_loss`'s weighting with the teacher's distribution in place of the one-hot target and of W₀. See
  "Loss weighting" below for why, and "Open for the lead" for the alternative.
- **B, Pile rows only, Adam 1e-5, 1000 steps.** Pile train rows from row 32,768 on, which is past the rows the
  emoticon and ladder windows were mined from (`gamma_refit` used the same start). Batch 8 rows of 512 tokens, about
  three times arm A's tokens per step. The loss is the plain mean KL over every position, since there is no known
  trigger. This is the version that would apply to someone else's model.
- **S, arm A with plain SGD at E21's rate** (0.03 on r0 and r1, 0.01 on emoticon). On the three SGD teachers only: S
  asks whether SGD's fail at step 1000 belongs to the optimiser on (U, V), and one teacher per task answers that.

**The triggers are not rare on the Pile for r0 and r1.** This conflicts with the sketch's P3. " it" and " which" occur
37,389 and 20,596 times in 32,768 Pile train rows, about 0.24% and 0.13% of positions. A batch of 8 rows then holds
about ten r0 trigger positions, and arm B sees roughly as many trigger positions over 1000 steps as arm A does. Only
the emoticon trigger is rare on the Pile (about 3 positions in 131k). So arm B is a strong test on r0 and r1, not a
null, and P3 below is split by task.

**Mixed arm (task windows plus Pile): dropped.** A answers whether the behaviour can be copied when the student sees
the cue, and B whether it can from ordinary text. A mixture would add regularisation on Pile text, but the loss already
asks the student to copy the teacher's predictions everywhere, so it adds no new constraint. The harness already reads
the Pile damage. On r0 and r1 the Pile already mixes the cue in at its natural rate, so B is the mixed arm there.

### Loss weighting (checked against the code)

E21's component-training loss (`finetune.arm_a_loss`, called by `component_training.train`) is not target-only and not
uniform. It is the cross-entropy of the target at the trigger plus 1.0 times the mean KL(W₀ ‖ student) over the other 40
positions of the 41-token window. A plain mean over all 41 positions would give the trigger 1/41 of the weight
instead of about half. Adam's scale invariance would partly absorb that, but SGD at E21's rate would move about 40
times more slowly at the trigger. Arm S would then not be comparable with E21. Arm A therefore keeps `arm_a_loss`'s
weighting: KL(teacher ‖ student) at the trigger plus the mean over the other positions. The difference from component
training is then only the targets. Off the trigger the student copies the teacher, damage included, instead of W₀.

## Controls and run order

Each control runs alone; nothing later starts unless its rule passes.

1. **C1, W₀ through the harness.** `faith_score.w0_check` on three adversary seeds, as in E26 (256 Pile rows,
   20-step adversary). Stop if it fails. Expected on an H100: floor f about 0.075, so pass ≤ 1.40, fail ≥ 1.61.
2. **C2, teacher = W₀.** Arm A's recipe on r1's windows for 100 steps with the teacher set to an unchanged copy of
   the model. The student and teacher logits are then identical, the gradient is exactly zero, and Adam's step is
   0 / ε. Rule: every offset is exactly zero after 100 steps, and the logged loss is 0 at step 1. Stop otherwise: a
   non-zero offset means the two forward paths differ numerically and Adam is amplifying that noise. This would
   bias every arm. About 10 s.
3. **C3, a teacher the student can reach exactly.** Teacher = E26's seed-1 r0 Adam 1e-5 component-training state at
   step 1000. It is saved at `experiments/ladder_second_seed/results/large/r0_t352_v310/adam_1e-05_seed1_step1000.pt`
   and rebuilt with `saved_states.offsets_state`; E26 read it at 1.14 (pass) with P(target) 0.920. The rebuilt state
   is itself a point in the student's set, so KL(teacher ‖ student) = 0 is reachable. First the teacher state goes
   through the harness as its own reference, and the KL floor is read with the student's offsets set to the teacher's.
   That reading is not exactly zero, because the hook path and the full-weight path round differently in bf16. Then
   arm A trains for 1000 steps and is scored at 100 and 1000.
   - Stop rule: at step 1000 the held-out teacher gap must be ≤ 10% of KL(teacher ‖ W₀), and P(target) within 0.05 of
     the teacher's 0.920. Otherwise stop and report: the optimiser cannot recover a teacher it can reach, and no
     teacher arm would be readable.
   - Read, not gating: the adversarial verdict, and the offset recovery, 1 − ‖off_student − off_teacher‖² /
     ‖off_teacher‖² over all alive offsets. It says whether teacher training finds that decomposition or a different
     one. A student that matches the teacher's function but fails the adversary is a finding, and it says to read the
     teacher arms' fails as properties of the copy.
   - What differs from the reference's end state: the loss (soft teacher targets instead of the hard target and W₀);
     the batch order (the student uses seed 0; the teacher came from seed 1); the starting optimiser state (fresh Adam,
     from zero offsets). None is removed. They are what C3 tests: whether a different loss and path still lands on
     the reachable teacher.
4. **Teachers, A arms first.** Order r1, r0, emoticon; SGD teacher, then Adam teacher. Per teacher: regenerate the
   fine-tune and log its reproduction reading; do-nothing through the harness; the scaled adjoint state's held-out
   P(target), via `harness.measure` with no Pile ratios, since every adjoint state so far passes (0.93-1.08) and only
   its P(target) is compared; then arm A, scored at 100 and 1000. Stop rule on the first A arm (r1, SGD teacher): if
   its teacher gap at step 1000 is above 50% of KL(teacher ‖ W₀), stop and report before spending on B.
5. **S arms** on the three SGD teachers.
6. **B arms** on all six teachers, last because they cost most.

Every state's record is written to `results/teacher_training_<task>.json` as it finishes. Every scored number is
also printed to the log. The pod runner downloads each task's file as soon as it is complete.

## Numbers

Seven numbers per scored state, each read on the same card in the same job.

| Number | Definition | W₀ | Do-nothing (the teacher) | C3 student | Blind to |
|---|---|---|---|---|---|
| Adversarial ratio and verdict | PGD KL at the state / W₀'s, seed 0, `state_verdict` with the same-job floor | 1 | the teacher's: 1.02 to 1.97 (table above) | ≈ the teacher's 1.14 | damage a 20-step attacker misses (E20 Amendment 2: far from converged) |
| Stochastic ratio | the same for stochastic masks | 1 | 0.88-1.0 | ≈ 0.97 | has never decided a verdict so far; kept because `state_verdict` reads it |
| Held-out P(target) | mean P of the target at the trigger, held-out windows (73 emoticon, 100-103 r0/r1) | 0.004 / 0.15 / 0.04 | the teacher's, 0.94-1.0 | ≈ 0.92 | the trigger position only |
| Teacher gap | held-out KL(teacher ‖ state), arm A's weighting | the teacher's full effect, about 2-6 nats | 0 by construction (its positive control) | ≤ 10% of W₀'s | which positions the residual sits at |
| Damage elsewhere | held-out KL(W₀ ‖ state), mean over the 40 non-trigger positions | 0 | the teacher's, 0.007-0.078 | ≈ 0.023 | damage outside the task windows (the Pile ratios read that) |
| Size of change | max over the 24 matrices of ‖ΔW‖ / ‖W₀‖ | 0 | the teacher's, 0.012-0.035 | ≈ 0.017 | function |
| Δ's share of ‖δW‖² | from `harness.measure` | — | 1 | ≈ 0 | structural: reads 0 for every student by construction; a positive control, no prediction |

The positive control for the teacher gap is do-nothing, which reads 0 because its target model is the teacher. For
the adversarial ratio it is the r0 Adam do-nothing state (1.97, a known fail), retaken in the same job. The noise floor
of every Pile ratio is C1's f; P(target) and the KLs are deterministic over the held-out set, with one seed.

## Draft predictions

Thresholds from `faith_score`: pass ≤ 1.5 / (1 + f), fail ≥ 1.5 (1 + f), f from C1. "Passes" below means
`state_verdict` reads "pass", which also needs P(target) ≥ 0.9. All P(target) and KL thresholds are absolute unless
marked relative.

- **P1, arm A on the SGD teachers.** On emoticon, r0 and r1 the arm-A student passes at steps 100 and 1000, as Adam
  component training does. Reading: per task; "all three", "some" (named), "none".
- **P2, where the adjoint split lost it.** On r0 and r1, the arm-A student of the SGD teacher has P(target) ≥ 0.9 at
  step 1000, and at least 0.5 above the same teacher's adjoint state in the same job (E31: 0.26, 0.07). Absolute.
- **P3a, arm B on emoticon.** P(target) < 0.5 at step 1000 for both emoticon teachers: the Pile holds the trigger at
  about 3 positions in 131k. The opposite would be a striking result.
- **P3b, arm B on r0 and r1.** Two-sided, filed with a lean: P(target) ≥ 0.9 and a pass at step 1000 for the SGD
  teachers. The cue occurs about ten times per batch, and the teacher's prediction after it is what arm A copies.
  The other readings are kept separate: learnt but unfaithful, partly learnt (0.5-0.9), not learnt (< 0.5).
- **P4, arm S.** On r0 and r1, S passes at step 100 and fails at step 1000, as SGD component training did (E21 2.82,
  3.00; E26 3.06, 2.91). On emoticon it passes at both, as SGD component training did (0.99). Two-sided: a pass at 1000
  on r0 or r1 says the soft targets, not the optimiser, decide SGD's fail.
- **P5, the Adam teachers.** On all three tasks the arm-A student passes at step 1000 where the same teacher's
  do-nothing state reads fail or inconclusive (r0 1.97, r1 1.55, emoticon about 1.4-1.5). Two-sided. The teacher's
  damage elsewhere (0.055-0.078) is about three times Adam component training's (about 0.02-0.03), and the student is
  asked to copy it. E21's Adam fails all sat at ≥ 0.2 nats of damage, so a pass is expected, but this is the arm
  most likely to surprise.
- **P6, damage.** Every arm-A student's damage elsewhere at step 1000 lies between 0.5 and 2 times its teacher's
  (relative). Below 0.5 would mean the student drops the teacher's side effects, which is not what the loss asks for.
- **C3 reading (not a prediction, reported).** Offset recovery at step 1000; no threshold, two-sided.

## Implementation sketch

- **`src/commute/teacher.py`, about 40 lines.** `teacher_net(model, dW)` builds a frozen deep copy of
  `model.target_model` with the 24 matrices set to W₀ + δW. `teacher_loss(net, teacher, toks, half)` computes
  `vpd.kl_per_position(teacher_logits, student_logits)`. With `half` given it uses `arm_a_loss`'s weighting (the
  trigger plus the mean elsewhere); with `half=None` it takes the mean over all positions. It returns the loss and an
  info dict. The teacher's logits come from a second forward under `no_grad`. Precomputing them would cost about
  1.6 GB per task in bf16 and gain little, since `arm_a_loss` already runs a second forward of W₀.
- **`component_training.train`, about 10 changed lines.** It takes a `loss_fn(net, toks) -> (loss, info)` argument,
  defaulting to today's `arm_a_loss` closure, so E21 and E26's calls are unchanged. The held-out reading stays
  `eval_arm_a` against the W₀ copy, so "damage elsewhere" is the same number as E21's. For arm B, `train_tokens` is
  the Pile rows and the batch size is a separate config field.
- **`experiments/teacher_training/run.py`, about 250 lines.** One frozen config dataclass (teachers, rates, steps
  (100, 1000), Pile start 32,768, batch sizes, harness sizes as E26's). Stages run in the order above: C1, C2, C3,
  then per task one function that regenerates a teacher, scores do-nothing and the adjoint P(target), trains arm A
  and scores it. S and B come after. Each stage dumps its JSON when it finishes and skips work already in the file,
  as `ladder_second_seed` does. The adjoint state uses `split.adjoint_state` with the task's training-window input
  moments, as E31 did, called rather than copied. Saved: arm A's step-1000 states and C3's student as bf16 offsets
  (`arms.save_offsets`, about 60 MB each, about 420 MB in total), downloaded per task. Teachers are not saved
  (regenerated in 30 s).
- The scored run is its own smoke: C1-C3 and the first A arm run in the first fifteen minutes. The Codex adversarial
  review runs on the diff before the paid run.

## Compute

From E26's H100 log (`experiments/ladder_second_seed/results/large/run_pod1.log`): W₀ on three adversary seeds 228 s,
a 1000-step component-training arm with a held-out reading every step 77 s, one harness-scored state about 55 s. From
E31's L40 logs: a 1000-step fine-tune 33-52 s.

| Stage | Count | Seconds each | Total |
|---|---|---|---|
| Load, C1 | 1 | 300 | 300 |
| C2 | 1 | 15 | 15 |
| C3: reference state, arm, two states | 1 | 250 | 250 |
| Teacher: fine-tune, do-nothing state, adjoint P(target) | 6 | 95 | 570 |
| Arm A: training and two states | 6 | 190 | 1140 |
| Arm S: training and two states | 3 | 190 | 570 |
| Arm B: training (estimated 0.2 s per step) and two states | 6 | 310 | 1860 |
| **Total** | | | **about 4700 s ≈ 80 min** |

With about 10 min of pod set-up and downloads, about 1.5 h on a RunPod H100 at about $4/h, so about $6. If arm B's step
time is twice the estimate, it is about 2 h and $8. This stays well under the $15 target and the $40 session budget.
It routes to RunPod: over 30 min, and over 1 h on one card, so an H100 per `.claude/CLAUDE.md`. It splits into
two shards if earlier results matter more than about $1: shard 1 runs C1-C3 and the A and S arms (about 50 min);
shard 2 runs the B arms (about 40 min), with its own C1, and refuses to start until shard 1's C3 passed. One job is
the default.

## What is not run

- r2_t275_v2667 (no scored teacher; component training does not pass there).
- A mixed arm of task windows and Pile.
- Teachers at steps 10 and 100, and LoRA teachers.
- More than one batch-order seed per arm, and more than one fine-tune seed per teacher.
- A longer adversary than 20 steps on any student.
- The uniform-weight loss on task windows (see "Open for the lead").
- Γ trained alongside (U, V); Γ stays frozen.
- A model post-trained by someone else.

## Open for the lead

1. **Loss weighting on arm A.** The sketch says the KL is taken "at every position". I chose `arm_a_loss`'s weighting
   (trigger plus mean elsewhere), because it makes the change from component training the targets only. It also keeps
   SGD at E21's rate meaningful. The other reading, a plain mean over all 41 positions, needs no knowledge of where
   the trigger sits, like arm B. Under it, arm S would need a rate about 40 times higher. If the user's intent is the
   uniform version, arm A should use it, and S needs a rate grid.
2. **P3 conflicts with the data.** The sketch's "the Pile sees ~3 trigger positions in 131k" holds for emoticon only.
   On r0 and r1 the trigger is a common word (" it", " which"), at 0.13-0.24% of Pile positions. P3 is split into
   P3a (emoticon, not learnt) and P3b (r0, r1, two-sided with a lean towards learnt).
3. **No teacher is saved.** Every candidate is regenerated in-job (about 30 s each). The adjoint and do-nothing
   readings are retaken in the same job, so E31's numbers are references, not the comparison.
4. **"Small" and "large" do not fit r0 and r1.** At step 1000 the SGD 0.03 fine-tunes are nearly as large as Adam
   3e-5's. The teachers are labelled by optimiser. E31's amendment 1 (SGD 0.01 on r0 and r1) has filed predictions but
   no results in the tree yet. If it runs first, its fine-tunes would be smaller SGD teachers on those tasks; worth
   swapping in if they keep P(target) ≥ 0.9.
5. **The "r0 reads 1.97" attribution.** The sketch says E21. It is E21's do-nothing state of the Adam 3e-5 fine-tune
   (`sgd_ladder_long_r0_t352_v310.json`), and E31 read the same 1.968. Both are on the L40, so the same-job H100
   reading may differ.
6. **C3's teacher.** I used E26's r0 seed-1 state (1.14, P 0.920), as the sketch suggested. The seed-0 states were
   never saved (`sgd_ladder_long` saves nothing). E26's r1 seed-1 state (0.96) is also on disk if a cleaner pass is
   wanted. No emoticon component-training state is saved.
7. **Arm S only on SGD teachers, and arm B ordering.** I cut S to one teacher per task and put B last, to keep the
   most informative A arms safe if the pod is lost. Running S on all six teachers adds about 10 min.
8. **The "simple map" decision.** Teacher training extends the 2026-10-14 row ("component training on the
   post-training loss") to a distillation loss. It needs the user's word that it counts as simple, and its name
   added to `GOALS.md` → Terms once confirmed.

## Lead's decisions on the open points (2026-10-23)

The user confirmed the technique after seeing the design check, chose two input arms (task windows and Pile only), left the teachers to the lead, and gave the go-ahead to build, file predictions and run on an H100 without another stop, with a check-in only on a control failure or a reversed prediction. Teacher training counts as a simple map on the 2026-10-14 reading (it trains the same (U, V) as component training with a different loss; no extra network). The name stays "teacher training"; its Terms row is added to `GOALS.md` at write-up.

1. Arm A keeps `arm_a_loss`'s weighting (KL at the trigger plus the mean over the other positions); arm B is the plain mean. The uniform-weight loss on task windows is not run.
2. P3 is filed as P3a and P3b.
3. Teachers are regenerated in-job; E31's readings are references only.
4. Teachers are labelled by optimiser. E31's amendment-1 fine-tunes are not swapped in.
5. C3's teacher is E26's r0 seed-1 Adam state.
6. Run order changes to A arms, then B arms, then S arms: B is the arm that makes teacher training a map, and S answers a secondary question. The first-A-arm stop rule still protects B.
7. C2's rule is kept as written (exactly zero), with one addition: if it fails, the run stops and reports the largest offset and the step-1 loss, so the lead can decide whether the two forward paths need to be unified before any arm runs.
8. One job, not two shards; per-task files are downloaded as they complete.

**Scored 2026-10-23** (E34): `reports/analyses/2026-10-23_analysis_teacher-training.md`. Controls pass; P1, P2, P3b, P5 hold; P3a reversed (emoticon learnt from Pile rows alone); P4 misses at r0 step 100; P6 misses on the emoticon Adam teacher.

## Amendment 1: separating the two causes of the Pile-only surprise (2026-10-23)

E34's P3a reversed: the Pile-only copies (arm B) learnt the emoticon behaviour, P(target) 0.96 with the SGD teacher
and 0.90 with the Adam teacher at step 1000, though the trigger is rare on the Pile. The analysis
(`reports/analyses/2026-10-23_analysis_teacher-training.md` → "What next" 1) left two causes open: (i) the roughly 110 trigger
positions arm B saw carried the behaviour; (ii) the teacher's changes on ordinary text pin down enough of δW for the
copy to reproduce the update. The user approved this separating test on 2026-10-23 after the P3a reversal.

**Design check.**
- **Goal served.** Q1 under reading A: whether teacher training is a map of the finished update itself (cause ii) or
  needs the trigger in the copying text (cause i). Only (ii) would carry it to behaviours the copying text never shows.
- **Nearest tried.** E34's arm B on emoticon (this card's G1) and its arm A (the saved step-1000 students); E31's
  adjoint split of the same fine-tunes. None removed the trigger from the copying text.
- **Patch or step.** A step: the named separating test of a reversed prediction, not a fix.
- **Decisions relied on.** Γ and Δ frozen; reading A; same-job W₀ floor and `state_verdict`; the trigger is the
  windows' definition (g of `h.2.mlp.down_proj:2359` > 0.5 at W₀).
- **What changes what we do next.** B-dropped learns and aligns with δW: teacher training goes to the stronger
  adversary and the trigger-free fine-tune as the commute line's map. B-dropped does not learn: teacher training
  needs the behaviour's inputs in the copying text, and the next step is choosing that text.
- **Control first.** C1 (W₀ through the harness), then G1 (E34's SGD-teacher arm B regenerated here, within 25% of
  its adversarial ratio 0.895 and 0.05 of its P(target) 0.960), each with a stop rule, before any new variant.

**Variants** (emoticon, both teachers, E34's arm-B recipe: Pile train rows 32,768-40,959, batch 8, Adam 1e-5, 1000
steps, batch-order seed 0, plain mean KL; scored at step 1000 only):
- **B**, as in E34 (for the SGD teacher this is G1).
- **B-masked**, the same batches with the loss weight 0 at every trigger position (the position whose prediction is
  the target), the masked mean over the rest. Rows where later positions attend to the trigger stay.
- **B-dropped**, every row holding a trigger replaced at its index by the next clean row of Pile train rows
  40,960-41,983, so the sampler draws the same indices and the student never sees a trigger. Rows replaced are logged.

The trigger is found in-job from Γ at W₀ (`data.subcomponent_g`), not from token ids: the trigger tokens occur about
94,000 times in these rows, almost all with the subcomponent off.

**Numbers.** E34's seven per state, P(target) at step 100 from the free held-out reading, and the cosine of the
student's weight change with δW (per matrix and over the 24 matrices; `finetune.cosines`). The cosine's references
per teacher: E34's arm-A student (positive control: trained where the trigger is), the adjoint state, and a random
direction of the same shapes (null, about 0). Blind to: weight directions that do nothing on the data.

**Predictions** (`experiments/teacher_training/results/predictions_amend1.json`, filed before the run):
- **P1**, lean to cause (ii): B-dropped reaches P(target) ≥ 0.9 and passes at step 1000 for both teachers. Readings:
  "both" (ii), "neither, P < 0.5" (i), "between" (0.5-0.9, or one teacher of two).
- **P2**, two-sided: B-masked falls in the same P1 category as B-dropped for each teacher.
- **P3**, two-sided: B-dropped's cosine with δW is at least half of arm A's and above 0.2, for both teachers; the
  adjoint state's cosine is reported beside it.

**Run.** `run.py --amend1` → `results/teacher_training_amend1.json`; brief `experiments/teacher_training/RUN_BRIEF_amend1.md`.
About 35 min on an H100, about $3, cap $5.

**Amendment 1 scored (2026-10-23).** `reports/analyses/2026-10-23_analysis_teacher-training.md` → "Amendment 1": C1 and G1 pass;
P1 "between" (SGD teacher learnt, Adam teacher 0.892, under 0.9); P2 fails as filed for the Adam teacher; P3 one of two
(Adam passes, SGD below both bars). Pod 01M5MQS6KRTCPHD83KXFQWWTTG, H100, 19.6 min, $3.26.

## Amendments 2-4 (2026-10-26): the loophole, a LoRA teacher, a second batch order under a longer adversary

*Lead session, from Amendment 1's "What next" (`reports/analyses/2026-10-23_analysis_teacher-training.md`). Approved class 3
and 4: the three items are the branches Amendment 1's design check named ("B-dropped learns and aligns: teacher
training goes to the stronger adversary and the trigger-free fine-tune") and the commute line's named next steps on
the existing model, tasks and harness. The check is written here; the work went ahead without sign-off (user,
2026-10-26: "implement the amendments"). Each amendment is its own script, predictions file and run:
`experiments/teacher_training/amend{2,3,4}.py`, `results/predictions_amend{2,3,4}.json`,
`results/teacher_training_amend{2,3,4}.json`.*

**In plain words.** Amendment 1 left three loose ends. Forty-five Pile positions where the emoticon trigger was partly
on stayed in the trigger-free copy's text, so the copy may still have seen a weak form of the trigger. Every teacher
so far was a full-weight fine-tune, while the user's canonical update is a LoRA. And every copy was read once, at one
batch order, under a 20-step adversary. Amendment 2 drops the partly-on rows too. Amendment 3 copies LoRA teachers.
Amendment 4 retrains every copy at a second batch order and reads each one, and its teacher, under an adversary eight
times longer.

### Design check (shared)

- **Goal served.** Q1, the weak version under reading A. Teacher training is the candidate map; these three tests
  decide whether its passes are robust (seed, adversary budget), whether they hold for the canonical update type
  (LoRA), and whether the Pile-only result needs any form of the trigger in the copying text.
- **Nearest tried.** E34 and its Amendment 1 (`TRIED.md`). E14 and E27: the rank-8 LoRA on all 24 matrices, its
  adjoint split keeps P(target) 0.06 at step 1000 (E14 part 2). E20 Amendment 3 and E31's budget sweep: the sign-step
  adversary at 20-160 steps with step size 2/n leaves every 20-step verdict unchanged. E26: a second batch order of
  component training replicated E21. None copied a LoRA, re-ran a copy at another batch order, or read a copy under
  the longer adversary; none dropped the partly-on rows.
- **Patch or step.** Amendment 2 is a patch on Amendment 1 (closing its named loophole). Amendments 3 and 4 are
  steps: the conditions under which teacher training becomes the commute line's working map.
- **Decisions relied on.** Γ and Δ frozen; reading A; same-job W₀ floor and `faith_score.state_verdict`; LoRA as an
  update type; E20 Amendment 3's budget recipe (n sign steps of 2/n, ratio to W₀ at the same budget, pass line
  1.241). No departure.
- **What changes what we do next.** If all three hold, teacher training goes to the larger model as the map (user:
  next). If Amendment 2's copies fall, the trigger must be in the copying text in some form and the Pile-only claim
  is withdrawn to "rare occurrences suffice". If the LoRA copies fail arm A, teacher training is specific to
  full-weight updates and the LoRA line needs its own map. If the seed-1 copies or the 160-step readings fail where
  the seed-0 20-step ones passed, E34's passes were a fixed-budget, one-seed artefact and the longer adversary becomes
  the harness.
- **Control first.** Each run: C1 (W₀ through the harness) alone, then a reproduction gate before any new state.
  Amendment 2: G1, the regenerated emoticon teachers match Amendment 1's reproduction readings (P within 0.01, size
  within 0.002). Amendment 3: G1, the emoticon LoRA teacher reproduces E9 (P within 0.01 of 0.9999, KL elsewhere
  within 0.005 of 0.0048, size within 0.005 of 0.0900) and its adjoint state's P within 0.05 of E14's 0.063, before
  any student. Amendment 4: G1, the r1 SGD teacher reproduces E34's reading (same seed and card: P within 0.01, size
  within 0.002), and the budget sweep's W₀ seeds 0 and 1 are read first (the noise floor of that measure).
- **Numbers.** E34's seven per state at step 1000 (adversarial and stochastic ratio with verdict, P(target), teacher
  gap, damage elsewhere, size, Δ's share), P(target) at step 100 from the free held-out reading, and per amendment:
  (2) the cosine with δW and the trigger counts at g > 0.1; (3) the LoRA teacher's own reading, its adjoint state's
  P, the student's cosine with the LoRA's δW beside the adjoint's and a random direction's; (4) the budget ratios at
  20 and 160 steps with W₀'s seed-1/seed-0 ratio beside them. Expected values: W₀ reads 1 on every ratio and its
  known P (0.004 emoticon, 0.15 r0, 0.04 r1); do-nothing is the teacher; the positive controls are the SGD-on-windows
  fails of E34 for the ratio (3.1-3.4) and, for the budget measure, the three teachers E34 read inconclusive
  (1.50-1.59), which should sit above the 1.241 line at 20 steps.

### Amendment 2: drop every row with the trigger subcomponent above 0.1

Emoticon, both teachers (SGD 0.01, Adam 3e-5, regenerated as in E34). One new variant, **B-dropped-0.1**: Amendment
1's B-dropped with the threshold lowered from 0.5 to 0.1 on `h.2.mlp.down_proj:2359`'s g at W₀, rows replaced by the
next clean spare row (clean at 0.1 too; 2,048 spare rows). Trained with E34's arm-B recipe (Pile train rows from
32,768, batch 8, Adam 1e-5, 1,000 steps, batch-order seed 0, plain mean KL), scored at step 1000; P(target) at step
100 from the free reading; cosine with δW against Amendment 1's references (arm A 0.245 / 0.285, adjoint 0.450 /
0.382, null 0.0002, read from `results/teacher_training_amend1.json`, not recomputed). Logged: positions and rows
with g > 0.1 and > 0.5, rows drawn, the largest g left (≤ 0.1 by construction).

Predictions (`predictions_amend2.json`): **P1**, lean "not carried": each teacher's B-dropped-0.1 reads P(target) at
step 1000 within 0.03 of its Amendment 1 B-dropped (0.937 SGD, 0.892 Adam); a fall of more than 0.10 reads "carried
in part"; otherwise "between". **P2**, two-sided: its cosine with δW within 0.02 of Amendment 1's B-dropped (0.078,
0.218). Reported: the verdict, P at step 100 (Amendment 1: 0.195, 0.080), teacher gap, damage. Blind to: a P noise
floor (none yet; Amendment 4 supplies one), one seed, emoticon only. About 15-20 min on an H100, cap $5.

### Amendment 3: a LoRA teacher

Tasks emoticon (first: it has references), r1_t534_v369, r0_t352_v310. The teacher is E9's LoRA: rank 8 on all 24
matrices, Adam 1e-3, seed 0, 1,000 steps, arm A's loss and batch (`finetune.finetune_lora_arm_a`); δW = BA per
matrix. Per task: the teacher's reading against E9 (emoticon) and the task gate (P(target) ≥ 0.9 at step 1000; a
teacher that did not learn is reported and its students skipped), its gap at W₀, do-nothing and scaled adjoint
states, then arm A (task windows, Adam 1e-5) and arm B (Pile rows, Adam 1e-5) with E34's recipe, scored at steps 100
and 1000; the step-1000 students' cosine with the LoRA's δW beside the adjoint state's and a random direction's.
Step-1000 students saved as bf16 offsets; the LoRA factors saved.

Predictions (`predictions_amend3.json`): **P1**: arm A passes with P ≥ 0.9 at steps 100 and 1000 on every task whose
teacher learnt. **P2**: on emoticon the copy keeps at least 0.5 more P(target) than the LoRA's adjoint split (E14:
0.063). **P3**, lean "learnt": arm B reads P ≥ 0.9 and passes at step 1000 on every task (categories as Amendment 1's
P1). Reported: cosines, damage ratios (the LoRA's damage is 0.005, too small for E34's P6 ratio), the r0/r1 LoRA
teachers' own readings (no reference exists). About 35-40 min on an H100, cap $6.

### Amendment 4: a second batch order, read under a longer adversary

E34's six teachers regenerated. Arms A and B retrained at batch-order seed 1 (the only training randomness), 1,000
steps, scored at step 1000 through the harness (20-step adversary, same-job W₀ floor) with P at step 100 from the
free reading. Then E20 Amendment 3's budget sweep, as E31's `budget()`: the first 128 Pile val rows in batches of 8,
`gamma.pgd_sources` with n sign steps of 2/n for n in (20, 160) from the seed's random start, `cancel.masked_kl`;
at W₀ (adversary seeds 0 and 1, the noise floor) first, then at each teacher's do-nothing state and each seed-1
student at step 1000 (seed 0). Ratio = KL(state) / KL(W₀ seed 0) at the same budget; pass line 1.241. Students saved
as bf16 offsets.

Predictions (`predictions_amend4.json`): **P1**, replication: all twelve seed-1 copies pass with P ≥ 0.9 at step
1000. **P2**, the noise floor: per copy |P₁ − P₀| ≤ 0.05 and the adversarial ratio within 25% of seed 0's (E34's
values are the reference). **P3**: at 160 steps every copy's budget ratio is at or below 1.241; and the three
teachers E34 read inconclusive (r1 SGD, r1 Adam, emoticon Adam) sit above their copies at both budgets. Blind to:
other adversary families; budgets past 160; one fine-tune seed. About 75-90 min on an H100, cap $10.

**Compute.** Three pods in parallel (`commute-teacher_training-amend{2,3,4}`), about $12 in total against the
session's $40, caps $21 in total.

**Amendment 3, first run stopped at G1 (2026-10-26 11:13, pod 01M5VXZ7WPBDG2CRN1X1P59MWV, H100, 13.5 min, $0.90).**
The regenerated LoRA read P(target) 0.99992, KL elsewhere 0.01625, size 0.11968, and its adjoint state kept P 0.234;
the gate's references were E9's factors on an L40S (0.9999 / 0.0048 / 0.0900) and E14's adjoint P 0.063 from those
factors. E27 regenerated the same recipe on an H100 (`experiments/lora_sgd/results/lora_sgd.json`, evals.adam) and
read exactly these numbers, to 1e-9, so the recipe is reproduced and the gate's reference was the wrong card's LoRA:
Adam 1e-3 for 1,000 steps diverges across cards in damage and size while keeping P(target). No core prediction reads
the gate, so it was re-referenced to E27's H100 readings (tolerance 0.001 each) and the adjoint state's P moved to
"reported" (no same-card reference exists). The first filing is kept unedited as
`results/predictions_amend3_filed-1050_g1-e9.json`; the refiled `predictions_amend3.json` has the same P1-P3. The
stopped run's file is `results/large/teacher_training_amend3_g1-e9-stop.json`. P2's bar is now the in-job adjoint P
(0.234 on this card) plus 0.5. Rerun as the same brief.

## Amendment 5 (2026-10-26, lead-2): why the two adversaries disagree, and is a copy unique?

*Options 2 and 3 of `reports/decisions/2026-10-26_decision_next-while-waiting.md`, run by the user's task for lead-2
(2026-10-26). Class 1 (re-reads of E34's states with the existing harness and budget measure); the r1 SGD teacher and
its copies are regenerated because E34's saved states are not on the dev node.*

**Design check.** *Goal:* Q1 under reading A; every adversarial verdict in the project rests on these two measures,
and the larger-model run will use one of them. *Nearest tried:* E34 Amendment 4 (the budget sweep reads the r1 SGD
teacher at 9.2 where the harness reads 1.59, same 20 steps, same step size, no cause found); E20 Amendment 3 (the
budget sweep itself). *Patch or step:* a patch on the apparatus, not a new question. *Decisions relied on:* Γ frozen,
Δ fixed; none departed from. *What changes what we do:* if the rows carry the gap, the 1.241 line is re-derived on
256 rows before the budget measure is used again; if the adversary's code path carries it (precision, readout, how
the masks are built), the measure that misreads is named and the project's "inconclusive" and "pass" verdicts that
rest on it are listed for re-reading; if the two copies at two batch orders point different ways, "the copy" is not a
unique placement and the larger-model write-up says so. *Control first:* W₀ through the harness (floor 0.075 within
5%), then the regenerated r1 SGD teacher must reproduce E34 (P within 0.01 of 0.983, size within 0.002 of 0.0287,
harness ratio within 0.01 of 1.588, budget ratio at 20 steps within 5% of 9.24) before any variant is read.

**Part 1, the disagreement.** One claim: *the gap is in how the adversary is computed, not in which rows it sees.*
Null: the rows carry it. Before writing code, list every difference between the harness's path
(`faithfulness.faithfulness` → the authors' `calc_multibatch_pgd_masked_recon_loss`) and the budget's
(`gamma.pgd_sources` + `cancel.masked_kl`): rows and blocks, precision (bf16 autocast against fp32), the readout
(the authors' final objective against a separate forward), how the masks and Δ's mask are built, which CI (lower-leaky
or not), the random start. Each difference that could matter gets one reading that changes it alone, at W₀ and at the
teacher's do-nothing state, adversary seed 0, 20 steps. At least:
- *rows*: the harness's two 128-row blocks read separately (block 0 is the budget's rows); the budget sweep on the
  first 128 rows and on all 256 in one adversary;
- *readout*: the budget adversary's own objective (`pgd_sources`' returned KL) beside `masked_kl` under the same
  sources;
- *precision*: the budget adversary and its readout under the harness's bf16 autocast;
- *budget*: the harness's adversary at 160 steps of 2/160 beside the budget sweep's 160 (E34 A4: 10.45).
Every reading is a ratio to W₀ under the same reading. Predictions: **P1 (rows do not carry it)**: on the teacher,
the budget sweep on 256 rows is within 25% of its 128-row ratio, and the harness's block 0 is within 25% of its
two-block ratio (25% = the copies' batch-order spread, Amendment 4). **P2 (it is in the code path)**: at least one
single-difference reading moves the teacher's budget ratio to within 25% of the harness's 1.59, or the harness's to
within 25% of 9.24. Low confidence on which; no prediction names it. **P3 (W₀ agrees)**: at W₀ every variant's KL is
within 10% of its own default's (the measures agreed on W₀ within 0.04 nats in Amendment 4). Blind to: other
adversary families, other teachers, adversary seeds past 0. Noise: the budget measure's own 7% at 20 steps (A4's C1b).

**Part 2, uniqueness.** Arms A (task text) and B (Pile text) of the r1 SGD teacher, each trained at batch-order seeds
0 and 1 with E34's recipe, 1,000 steps. Read: the cosine between the two seeds' weight changes per arm
(`finetune.cosines`, the "all" entry), each copy's cosine with δW (E34: 0.08 to 0.22) and the cosine between arms A
and B at seed 0, with the random-direction null. Expected: **P4**: the two seeds' copies point the same way, cosine
≥ 0.8 per arm (medium; the end states moved by 0.005 in P(target) in Amendment 4, but nothing has read the weights).
Below 0.5 on either arm means a copy is one of many behaviourally equal placements. Reported, no prediction: A against
B. Blind to: placements that are equal up to a reparametrisation of the subcomponents (the cosine is on the weight
change, which is gauge-free).

**Cost.** One teacher, four 1,000-step copies, about ten adversary reads: about 20 min on an H100 or L40. Cluster
`debug` QoS first (job ≤ 30 min), RunPod H100 if not started in 10 min. Cap $3.

## Amendment 6 (2026-10-28, commute lead): teacher training coverage

*From gaps 10, 7, 6 and 4 of `reports/reviews/2026-10-28_review_technique-comparison.md` (section 2). Inside the user's
instruction to the commute lead (2026-10-28) to fill the comparison's gaps with new experiments: no sign-off; the
check is written here. Code `experiments/teacher_training/amend6.py`, predictions `results/predictions_amend6.json`,
brief `RUN_BRIEF_amend6.md`, results `results/teacher_training_amend6.json`.*

**In plain words.** Teacher training is the one map that keeps the behaviour and passes the adversarial check on narrow
updates, but four of its cells are still empty. We have never read whether the trained decomposition harms ordinary
text, only the task's own windows. We have never tried it on the rare-target task where component training never
passed, or with plain SGD as its optimiser on anything but an SGD teacher. And its text-only form copied a LoRA only
halfway in 1,000 steps. This job reads all four on the existing model, tasks and harness, in one H100 run.

### The question, restated in the Terms' words

Teacher training is component training with the loss KL(teacher ‖ student), the teacher the fine-tuned model W₀ + δW,
Γ and Δ frozen. Four readings of it, each on E34's harness:

- **(a) Harmless on plain text.** For the decompositions after teacher training of r1's SGD, Adam 3e-5 and LoRA
  teachers, is the damage to plain Pile text, KL(W₀ ‖ state) over the 256 Pile val rows, at most twice the
  fine-tune's own?
- **(b) The rare target.** On r2 t275 (" in" → " Fig"), with a full-weight SGD 0.1 teacher (E26's rate), is the
  decomposition after teacher training faithful, and does it keep the teacher's behaviour?
- **(c) Plain SGD as the map's optimiser.** Teacher training with plain SGD at 0.03 on (U, V) (E34's arm S) failed at step
  1,000 on r1's SGD teacher. Does it also fail on r1's Adam teacher and r1's LoRA teacher? Here SGD is the optimiser
  of teacher training, not of the fine-tune.
- **(d) A LoRA from plain text, longer.** Teacher training on Pile text alone (arm B) of the rank-8 LoRA teacher, run to
  5,000 steps instead of 1,000. Does it reach P(target) 0.9 and pass, and does a second batch order agree?

### Design check

- **Goal served.** Q1, the weak version under reading A, with teacher training as the map (`decisions.md`,
  2026-10-26). (a) completes "harmless" on the input distribution the adversary reads. (b) is the weak version's
  hardest familiar task. (c) is the optimiser subgoal (`GOALS.md`: both optimisers as the map). (d) decides
  whether the label-free form of the map covers LoRA, the canonical update.
- **Nearest tried.** E34 (damage read on task windows only; r2 left out; arm S on SGD teachers only), E34 Amendment 3
  (LoRA arm B at 0.57, 0.88, 0.52 at step 1,000, still rising), Amendment 4 (the batch-order spread), E21 and E26
  (component training on r2: SGD fails, Adam keeps 0.78-0.90). None read plain-text damage, a teacher on r2, an SGD
  map on a non-SGD teacher, or a LoRA arm B past 1,000 steps.
- **Patch or step.** Steps: each fills a cell of the comparison matrix that bears on `GOALS.md`; none patches a
  failed run.
- **Decisions relied on.** Γ and Δ frozen; reading A; same-job W₀ floor and `faith_score.state_verdict`; the paper's
  adversary, one start, 20 steps, passes as lower bounds (2026-10-27). No departure.
- **What changes what we do next.** In the decision tree below. In short: (a) failing narrows the "harmless" claim and
  adds a plain-text reading to the Qwen harness. (b) failing says a rare target is out of reach of every Γ-frozen map
  tried. (c) reading "map" makes Adam the only optimiser of teacher training. (d) plateauing below 0.9 says plain
  text does not pin down a LoRA, and task text is needed.
- **Control first.** C1 (W₀ through the harness), then W₀'s own plain-text damage (must read 0), then G1 (the r1
  SGD teacher reproduces E34, its do-nothing state included), each alone with a stop rule, before any new state.
  Stage c opens with its own control G3 (arm S on the SGD teacher must fail at step 1,000, as in E34). No reference
  end state is replayed: every teacher is a deterministic regeneration, read against its same-card numbers.

### Arms and states (one job, cheapest stage first)

| Stage | Teacher | States, each read on E34's harness | Adversary at |
|---|---|---|---|
| 0 | — | W₀ (C1); W₀'s plain-text damage | C1 only |
| G1 | r1 SGD 0.03 | do-nothing, scaled adjoint | do-nothing |
| a | r1 SGD 0.03, Adam 3e-5, LoRA rank 8 | do-nothing, scaled adjoint, students A and B at 1,000 (LoRA: A only) | none (E34, A3, A4 read them) |
| c | r1 SGD (G3), then r1 Adam, r1 LoRA | student S (plain SGD 0.03, task windows) at 1,000 | 1,000 |
| b | r2 t275 SGD 0.1 (gate P ≥ 0.5) | do-nothing, scaled adjoint, students A and B at 1,000 | do-nothing, A, B |
| d | LoRA on emoticon, r1, r0 | student B, seed 0 to 5,000 (read at 1,000, 2,500, 5,000); then seed 1 to 2,500 (1,000, 2,500) | 2,500 and 5,000 |

Students: A = task windows with the trigger weighting, B = Pile train rows 32,768-40,959 (batch 8, plain mean KL),
both Adam 1e-5, batch-order seed 0 unless named. Teachers are regenerated: arm-A fine-tunes of all 24 matrices (seed 0,
step 1,000) and E9's LoRA (`amend3.lora_teacher`). Saved as bf16 offsets: the r2 students and every stage-d student at
its last step (eight files, about 60 MB each).

Why 5,000 steps for (d): Amendment 3's emoticon student rose by 0.49 between steps 100 and 1,000 and r1's by 0.12, so
both should cross 0.9 well inside 5,000 if plain text carries the LoRA. r0's rose by 0.06; a plateau below about 0.6
will show by 5,000. Ten thousand steps would double the cost for r0 alone. The 8,192 rows are reused about five times
over 5,000 steps; the plain-text gap on held-out val rows shows if that matters. Seed 1 stops at 2,500 steps, the
point the plateau rule compares against, at half the cost of a full replica.

### Numbers, and what each should read

Per state, E34's seven (adversarial and stochastic ratio with verdict, P(target), teacher gap, damage on task windows,
size, Δ's share) and three more:

| Number | How measured | W₀ | Do-nothing (the teacher) | Control | Blind to |
|---|---|---|---|---|---|
| Plain-text damage | mean KL(W₀ ‖ state), every position of the 256 Pile val rows | 0 (≤ 1e-4, stop rule) | the fine-tune's change on plain text, likely the order of its window damage (r1: SGD 0.056, Adam 0.076, LoRA 0.015) | the do-nothing reading > 0 | harm outside those rows; damage in the masked model |
| Plain-text gap | mean KL(teacher ‖ state), same rows | the teacher's plain-text damage | 0 by construction | the do-nothing reading 0 | which positions carry it |
| Faith | `state_verdict` on the two ratios, no P bar | pass | r1 SGD inconclusive (1.588, G1) | S on the SGD teacher: fail (E34 3.14) | adversaries past one start of 20 steps |

Expected for the students: A and B of the full-weight teachers at P 0.97-1.0 and pass (E34, A4), plain-text damage
0.55-0.97 times their teacher's (E34's window ratios); the r1 LoRA's A student at 2.1 times on windows (A3). The
adjoint states below their teacher's damage (0.38-0.55 on r1's windows). Stage-d seed 0 at step 1,000 within about
0.03 of A3's P (reported). P(target) at W₀: 0.004 emoticon, 0.15 r0, 0.04 r1, 0.0034 r2.

**Noise floor.** Amendment 4's batch-order spread: P(target) within 0.032 for Pile-text students, adversarial ratio
within 25%; C1's adversary-seed floor (0.075 on the H100). The plain-text readings are deterministic per state and
have no batch-order spread yet, which is why P1's bar is a factor of two.

### Predictions (`results/predictions_amend6.json`, filed before the run)

- **P1, harmless on plain text** (high on the four full-weight students, low on the LoRA one). For r1's SGD-teacher A
  and B, Adam-teacher A and B and LoRA-teacher A students at step 1,000: plain-text damage at most twice the
  teacher's, or at most 0.002 nats. Pass needs all five.
- **P2, r2 student A** (medium-low). "Kept" means P at least the teacher's minus 0.05; "faithful" means faith reads
  pass. Predicted: kept and faithful.
- **P3, r2 student B.** As P2, two-sided, lean kept and faithful (" in" is common on the Pile).
- **P4, plain SGD as the map** (medium, two-sided). Student S at 1,000 on the r1 Adam and LoRA teachers: fail on both
  reads "map", pass on both "teacher", else "mixed". Lean: map. Needs G3.
- **P5, a LoRA from plain text, longer** (medium on emoticon and r1, none on r0). Student B at seed 0, step 5,000:
  learnt (P ≥ 0.9 and pass), learnt but not passed, plateau (50-step mean P rose by under 0.03 from 2,500 to 5,000),
  or rising. Predicted: emoticon and r1 learnt; r0 not predicted.
- **P6, second batch order.** Per task, seed 1 against seed 0 at step 2,500: P within 0.05 and adversarial ratio
  within 25% (Amendment 4's rule). Predicted: holds on all three.

Every rule was run against its edge cases before filing: an empty or partial pool never passes, a teacher with zero
plain-text damage falls back to the 0.002 bar, ties at each bar count as holding, a skipped r2 teacher reads "not run",
and a failed G3 reads "unreadable".

### Decision tree

- **C1, W₀'s plain-text damage, or G1 fails.** The run stops; Blocking card, report the numbers.
- **P1 holds.** Plain-text harmlessness joins teacher training's card and the matrix. It fails on a full-weight
  student: the claim narrows to task windows, and the Qwen harness gets a plain-text damage reading (designed there).
  It fails on the LoRA student only: reported beside its window ratio, no follow-up.
- **P2 kept and faithful.** Teacher training is the first map to place r2's behaviour faithfully. Faithful but not
  kept: run student A to 3,000 steps on r2 (approved branch, about $0.5). Kept but not faithful, or neither: no
  Γ-frozen map tried places a rare target; a pending-decision card, since it bears on the "Create" reading and on
  updating Γ. Teacher skipped by the gate: report; an Adam teacher on r2 needs its own check, since its P is 0.82.
- **P4 "map".** Adam is the only optimiser for teacher training; the lead adds a `decisions.md` row and the
  optimiser subgoal's reading. "Teacher": the late failure needs the SGD teacher; next, arm S on it at seed 1
  (approved, class 2, about $0.5). "Mixed": reported per teacher. G3 fails: stop and tell the lead.
- **P5 learnt on all three.** The text-only map covers a rank-8 LoRA given more steps; claims row and techniques card
  updated. A task in plateau whose student has taken nine tenths of the teacher's change on plain text (its
  plain-text gap at most a tenth of the teacher's plain-text damage): plain text does not carry that LoRA. The
  label-free map then needs task text for it, Amendment 3's first explanation. Plateau with a larger gap, or rising: that task's student B to 10,000 steps (approved branch, about
  $1.5). "Learnt, not passed" on any task: a qualitative surprise; stop and tell.
- **P6 fails.** The spread is reported, and every P5 category within 0.05 of its bar is marked unsettled.

Correction (2026-10-28, before the run): G3 failing skips the two new S arms and the run continues, as the filed controls say.

### Cost and routing

| Stage | Minutes (H100, from E34, A3 and A5 logs) |
|---|---|
| Load, C1, W₀ plain-text control | 4 |
| G1 (r1 task context, teacher, do-nothing with the adversary, adjoint) | 3 |
| a (two fine-tunes and a LoRA, five students, Pile train stream about 3 min) | 13 |
| c (three S students with the adversary at 1,000) | 7 |
| b (r2 context, teacher, do-nothing, adjoint, two students with the adversary) | 8 |
| d seed 0 (three 5,000-step students at about 0.1 s per step, six adversary reads) | 33 |
| d seed 1 (three 2,500-step students, three adversary reads) | 17 |
| **Run** | **about 85** |

About 100 min with set-up and downloads: RunPod "NVIDIA H100 80GB HBM3" (over 30 min, and E34's floor and references
are that card's), about $8 at E34's effective $4.7/h, cap $10, against the session's $40. If the session budget is
short, the run can stop after stage d's seed 0 (about $6.5); every stage writes its results as it finishes.

### Not in this plan

- The r2 Adam 3e-5 teacher: one teacher keeps stage b at 8 min, and E21 read that fine-tune at only P 0.82 (L40), so the SGD teacher is the cleaner first read.
- Arm S at a second batch order, on emoticon or r0, or read by the adversary at step 100.
- The adversary on stage a's students (E34, A3 and A4 read them), and on the Adam and LoRA do-nothing states.
- Student B past 5,000 steps, on more than 8,192 rows, or at seed 1 past 2,500.
- A longer adversary or more starts (decided 2026-10-27).
