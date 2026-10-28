# Plan: Does teacher training copy a behaviour the model lacks, faithfully and with Δ fixed, and through which subcomponents? (E42)

*2026-10-28, written by a subagent of the commute lead. Fills gap 1 of
`reports/reviews/2026-10-28_review_technique-comparison.md` (section 2, ranked high) at the user's instruction to fill
the comparison's gaps with new experiments. Words as in `GOALS.md` → Terms. Code, predictions and run brief are
written; nothing has run.*

## The question, in plain words

Teacher training is the only map so far that keeps the behaviour, passes the paper's adversarial check and leaves Δ
alone (E34, its Amendments 1-4). Every teacher it copied was a fine-tune of a behaviour the model already half had:
a word that often comes next, or an emoticon. `GOALS.md` asks what happens when post-training teaches something new.
On the key-value cipher (E10, E12), the fine-tune's gain runs mostly through subcomponents that Γ keeps off at the
trigger. That is where a map with Γ frozen is most likely to fail. This plan copies a cipher fine-tune by teacher
training and reads three things: whether the behaviour is kept, whether the decomposition after teacher training
passes the check, and which route the gain takes.

**Restated in the Terms' words (what the lead confirms).** Teacher = the arm-A fine-tune of all 24 matrices on a
key-value task (E10's recipe: Adam 3e-5, batch 32, seed 0), run to step 1,000. Map = teacher training: component
training of the alive (U, V) from W₀'s values with the loss KL(teacher ‖ student), Γ and Δ frozen, on (A) the task's
training windows or (B) plain Pile text. Read at step 1,000, against the teacher's do-nothing state: the held-out
target probability, VPD's stochastic and adversarial checks on the Pile as ratios to W₀ (the paper's setting, one
start, 20 steps; `decisions.md` 2026-10-27), and the route split at the trigger, whose masked-off share (e_gate) says
how much of the gain runs through subcomponents Γ keeps off.

**Which task first, and why.** rand32: 32 random key-to-value pairs after a marker, a word-level substitution
cipher. It is the case `GOALS.md`'s "new capabilities" branch is about, the one E12 and E28 used, and the only task
of the family where component training with labels has failed the check (E10: 2.81 at its first learnt step,
passing later), with the largest least-squares-split reading (38.9). rand1 (one pair) is a single new bigram, the ladder's territory by E10's own plan; ident32 (each key
followed by itself) is the matched control, a behaviour the model partly has, whose gain runs through re-gating
(r 0.76-0.80). All three cost the same per task (403 training windows each), so "cheapest" decides nothing between
them; rand32 is the most informative. ident32 and rand1 follow as parallel cluster jobs ($0, decision tree).

**Can plain Pile text carry the behaviour at all?** Only indirectly, and probably not. The marker (`" \+"`, token
18214) was chosen for having zero count in the 32,768 rows the windows came from; arm B's rows (32,768 onward) are
new, so it may occur a few times, but "marker then key" should be about absent. The only other route is the
teacher's predictions after the key words in ordinary text: if the fine-tune maps " different" to " copy" even
without the marker, the Pile shows the table at every key (keys are common words, thousands of positions). Arm A's
loss pins predictions at the other 40 positions of each window, where keys occur, so this leak should be small. The
run measures it before arm B (the "pile reading", below) and runs arm B anyway: it costs about 7 minutes on the
cluster, and E34's Pile-only arm learnt the emoticon when the plan expected it could not (P3a reversed).

## Design check (`/design`)

- **Goal served.** `GOALS.md` → "If no simple Φ works": is it because post-training teaches new capabilities?
  Teacher training is the one map meeting the weak version on narrow tasks; this tests it where "create" is
  expected (Where an update goes: "When Δ may not move, (d) dominates or component training fails to learn").
- **Nearest tried.** E10 (and its Amendment 1): label-trained component training on the same tasks, verdicts at
  the first learnt step and at 100 and 300; E34 and Amendments 1-4: teacher training on three half-had behaviours.
  E10 could not say whether a map *of a finished update* keeps a new behaviour with Δ fixed; E34 never tried a
  new one. Nothing has read the route split of a teacher-training state.
- **Patch or step.** A step: it fills an empty cell (map × task class) of the technique matrix.
- **Decisions relied on.** Γ frozen; teacher training as the map (`decisions.md` 2026-10-26); the paper's adversary
  setting (2026-10-27); E21's verdict rule with the same-job W₀ floor. Departure, flagged: `decisions.md` 2026-10-26
  moved the commute line to the larger model; this runs on the 67M because the cipher exists only there, at $0, at
  the user's instruction to fill the comparison's gaps.
- **What changes next.** A pass (P1) makes teacher training's claim cover new behaviours on the 67M and the cipher
  becomes a candidate task for the Qwen line. A fail is the first faithful-map failure on a narrow task and goes to
  the user. P2 says whether a faithful copy of a new behaviour still runs through pieces Γ keeps off, which bears
  on the reading question below. P3 decides whether the label-free form of the map needs the task's inputs for a
  new behaviour.
- **Control first.** C1 (W₀ under the Pile harness) runs first and alone and stops the job; G1 (the teacher's
  step-10 reading reproduces E10's, since it is a prefix of E10's own run) stops the task before anything is read.
- **Reference recipe replayed.** The teacher is E10's run continued; it differs from E10's end state only in
  length (1,000 steps, E34's teacher length, against E10's 300). Same windows file, table, seed, batch, rate, clip;
  no schedule, so steps 1-300 are identical up to card rounding. G1 tests that. The students reuse E34's `CFG`
  unchanged.
- **Class.** Inside the user's instruction to fill the technique comparison's gaps (2026-10-28); no sign-off needed.

## States

| State | How made |
|---|---|
| W₀ | the released decomposition (`harness.load`, round 1's alive set) |
| teacher, do-nothing | `finetune_arm_a` on `experiments/kv_task/results/large/windows_<task>.pt` (403 train, 97 held-out windows; table target), Adam 3e-5, batch 32, clip 1, seed 0, checkpoints 10, 300, 1000; δW at 1000 left in Δ |
| least-squares split | `split.split_state(δW)`: target weights = the teacher's, the remainder in Δ; the harness's positive control |
| arm A | `run.train_student` on the training windows, arm A's weighting, Adam 1e-5, batch 32, batch-order seed 0, 1,000 steps |
| arm B | `run.train_student` on Pile train rows 32,768-40,959, batch 8 × 512, plain mean KL, Adam 1e-5, 1,000 steps |

Only the two students' step-1,000 states are saved (bf16 offsets, about 60 MB each, `results/large/e42/<task>/`); a
branch below reads them. The teacher is regenerated (40 s).

## Stage 0: the controls (C1 alone first; G1 per task before anything is read)

| Check | Reads | Must read | Else |
|---|---|---|---|
| C1 | W₀ under the Pile harness, adversary seeds 0, 1, 2 (`faith_score.w0_check`) | pass; floor about 0.21 on an L40 (lines 1.24 / 1.81), 0.075 on an H100 (1.40 / 1.61) | stop, nothing trained |
| G1 | the teacher's held-out P(target) at step 10 | within 0.02 of E10's: rand32 0.5847, ident32 0.9595, rand1 0.9982 (E28 read rand32 at 0.5858 on another card) | stop the task: windows, table or recipe differ |
| gate | the teacher's held-out P(target) at step 1,000 | ≥ 0.9 (E10: 0.9965 / 0.9995 / 1.0000 at 300) | report, skip its students |

## Stage 1: references, per task

| Number | How measured | W₀ | do-nothing | least-squares split (control) |
|---|---|---|---|---|
| held-out P(target) | `arms.heldout_of` | 1.5e-5 rand32, 1.7e-4 ident32 and rand1 | ≥ 0.99 | = teacher's |
| adversarial ratio | Pile harness, seed 0, ratio to this job's W₀ | 1 | not predicted; E10 at step 10 read 4.73 / 8.12 / 1.06 at damage 0.70 / 0.19 / 0.04 | ≥ 1.81 on rand32 and ident32 ("power"); E10 at 10: 38.9 / 9.4 / 1.73 |
| e_gate share | route split at the trigger on held-out windows | undefined (no gain) | = −r (all gain in e_Δ) | E10 at 10: 0.52 / 0.06 / 0.73 |
| teacher gap | arm-A-weighted KL(teacher ‖ state), held-out | about 11 nats rand32, 8.7 others | 0 | 0 |
| pile reading | mean P(table[x]) after each key x in arm B's first 1,024 rows | ~1e-5 to 1e-4 (the baseline gate) | teacher's: within 10× of W₀'s (lean) | — |

## Stage 2: the students, per task

| Number | At step 1,000 (Pile harness); step 100 read without it | Expected, arm A | Expected, arm B |
|---|---|---|---|
| held-out P(target) | as above | ≥ 0.9 | < 0.5 |
| verdict | `faith_score.state_verdict` (P ≥ 0.9, then both Pile ratios against the job's floor) | pass; E10 Amendment 1's label-trained rand32 read 0.975 / 0.944 at 100 / 300 | below P 0.9 |
| e_gate share | route split | near label training's: 0.75 rand32, 0.86 rand1, 0.21 ident32 | not scored |
| teacher gap | as above | < 5% of W₀'s (E34 < 3.6%) | reported |
| Δ's share | `harness.measure` | 0 up to rounding (Δ frozen) | 0 |

Damage elsewhere, size and the other shares (d, r, i, e_Δ) are logged and reported, not scored.

## Predictions (filed: `experiments/teacher_training/results/predictions_e42.json`, 2026-10-28 11:57)

- **P1, arm A copies faithfully.** Per task past the gate, arm A at step 1,000: verdict pass → pass;
  inconclusive → inconclusive; fail → fail; below P 0.9 → did not copy. Absolute. *Lean: high* on rand32 (labels
  passed at 100 and 300; every E34 task-text copy passed). Blind to: one adversary start (E34 Amendment 5), 20 steps,
  256 Pile rows, one batch order, one teacher seed. Positive control: the least-squares split's fail. Noise floor:
  W₀'s adversary-seed spread 0.21 (L40); a copy's ratio moves up to 25% between batch orders (E34 Amendment 4).
- **P2, through masked-off subcomponents.** Arm A at step 1,000, e_gate share: ≥ 0.5 masked-off; ≤ 0.25 (negatives
  included) not masked-off; between mixed; gain < 1 log-odds or NaN unscorable. *Lean:* masked-off on rand32 and
  rand1 (high: labels 0.75 and 0.86), not masked-off on ident32 (medium: 0.21). Blind to which subcomponents and how
  many, and to anything off the trigger. Positive control: ident32, whose gain runs through re-gating. Noise floor:
  unmeasured between batch orders; the 0.5 line sits 0.25 under the label-trained value.
- **P3, arm B from plain text.** Arm B at step 1,000: learnt (P ≥ 0.9 and pass), neither (P < 0.5), between.
  *Lean: neither* on every task (medium: no marker in the text and arm A's loss discourages the leak; but E34's
  P3a reversed once). Blind to longer training (E34 Amendment 3's LoRA copies were still rising at 1,000). Noise
  floor: 0.032 in P between batch orders on Pile text.

Each rule was run against its edge cases before filing (a pass at exactly 0.5 and 0.25, a NaN share, a gain under
1, a negative and an over-1 share, P exactly 0.5, P ≥ 0.9 with a fail, a skipped teacher, a file with no students):
every case lands in a named class.

## Decision tree (each branch approved with this plan)

- **C1 fails** → stop; if the L40 floor is above 0.25, rerun job 1 on a RunPod H100 (about $2).
- **G1 fails** → stop and report to the lead with both numbers; no rerun until the cause is found.
- **Job 1 (rand32) ends with C1 and G1 passed** → jobs 2 and 3 (ident32, rand1), whatever P1 read: ident32 is the
  control that reads any rand32 outcome.
- **P1 inconclusive** on a task → rerun that task on a RunPod H100 (band 1.40-1.61; about $1.5-2 each).
- **P1 fail or did not copy on rand32** → Blocking card and PushNotification after ident32 has read (same failure:
  the format; ident32 passes: the class, GOALS's "create").
- **P1 pass, P2 masked-off on the cipher tasks** → a faithful copy of a new behaviour still runs through pieces Γ
  keeps off. Next, class 1, $0: list the subcomponents that carry the gain from the saved offsets on CPU. Then a
  pending card for the user (see Unresolved).
- **P2 not masked-off on a cipher task** → reversal of a filed lean: flag it loudly and check in.
- **P3 learnt or between** → reversal: flag it; the pile reading says whether the teacher shows the map without
  the marker. **P3 neither** → the label-free map needs the task's inputs for a new behaviour; one line in the
  Qwen line's plan.

## Unresolved: the reading a pass would need (for the lead, before the write-up)

`GOALS.md` says in "Where an update goes" that the weak version holds when (d) masked-off and (e) residual are
small, and in the same section that VPD's checks decide it and the route split does not
(`reports/decisions/2026-10-19_decision_success-criterion.md`, reading A). A P1 pass with P2 masked-off satisfies
the second and not the first. Under reading A it is a pass with a finding about mechanism; the write-up must say so
and the lead should raise the tension as a pending decision card rather than pick a side.

## Cost and routing

| Stage (L40, from E21's and E10's cluster logs) | Minutes |
|---|---|
| load, bases, C1 (3 × 137 s) | 8 |
| teacher (1,000 steps), G1, gap | 1 |
| do-nothing and least-squares reads (Pile harness each) | 5 |
| Pile rows (3 if not cached) and pile reading | 1 |
| arm A (train ~3, step-100 read, step-1,000 harness read) | 6 |
| arm B (train ~4-5, two reads) | 8 |

About 30 minutes per job: MATS cluster, `debug` QoS, L40, $0 (`cluster/teacher_e42.sbatch <task>`, 1-hour wall
time). RunPod if a job has not started within 10 minutes: all three tasks in one H100 pod is about 35 minutes of run
plus set-up, about $3.5. The cluster's wider verdict band (1.24-1.81 against the H100's 1.40-1.61) is the price of
$0; an inconclusive reading takes the H100 branch.

## Order of work for the fleet agent

1. Done: `src/commute/kv_windows.py` (loader, pile reading), `experiments/teacher_training/newskill.py`,
   `cluster/teacher_e42.sbatch`, predictions filed, lint clean.
2. A fresh Opus review of `newskill.py` and `kv_windows.py` (not paid, so not required; recommended because both
   hold new measurement code).
3. `/pod-runner` with `experiments/teacher_training/RUN_BRIEF_e42.md`: job 1 (rand32), then jobs 2 and 3.
4. Analysis report by `/writeup` (`reports/analyses/2026-10-28_analysis_e42-teacher-new-behaviour.md`), notebook
   `notebooks/E42_teacher_new_behaviour.ipynb`, `TRIED.md` row, PENDING line, techniques-table cell.

## Not in this plan

An SGD teacher or SGD students (arm S); a LoRA teacher; E10's step-10 teacher (the first-learnt step, where the
damage is largest); the scaled adjoint split; a second batch order or teacher seed; a longer adversary or several
starts (`decisions.md` 2026-10-27); teacher = W₀ (E34's C2 tests the loss, which the windows do not change); a Pile
reading of the students; the larger model.

## Amendment 1 (2026-10-28): teacher training with the masked students matched to the teacher

*Written by a subagent of the commute lead, after E42 ran (analysis `reports/analyses/2026-10-28_analysis_e42-teacher-new-behaviour.md`,
check `reports/reviews/2026-10-28_review_e42-check.md`, claim 3 and card point 5). Fills the frozen-Γ branch of the
pending card `decision-e42-criterion`. Code, predictions, sbatch and brief are written; nothing has run. Appended;
nothing above is changed.*

### The question, in plain words

E42's teacher training kept the cipher in the fine-tuned model's place and passed the Pile checks, but on the two
random ciphers the model built from the subcomponents Γ switches on at the trigger keeps almost none of it: P(target)
0.007 on rand32 and 0.001 on rand1, against 0.96 and 1.00 with every subcomponent on. 84% and 92% of the gain runs
through subcomponents Γ keeps off there. The check found this is not specific to teacher training, since
label-trained component training does the same, but that no frozen-Γ map has yet been trained to avoid it. This
amendment trains one. If the CI-masked model can be made to keep the cipher with Γ frozen, E42's masked-off route
came from the loss, and the pending card gets a map that satisfies both readings. If it cannot, the frozen Γ is what
limits it, which points at updating Γ.

**Restated in the Terms' words (what the lead confirms).** Teacher = E42's (arm-A fine-tune of all 24 matrices on the
cipher's training windows, Adam 3e-5, batch 32, seed 0, step 1,000), regenerated. Map = teacher training with two
terms added to its loss. Component training of the alive (U, V) from W₀'s values, Γ and Δ frozen, on the task's 403
training windows, with the loss KL(teacher ‖ student) + 0.5 KL(teacher ‖ CI-masked student) + 0.5 KL(teacher ‖
stochastic-masked student). Each term uses arm A's weighting: the KL at the trigger plus the mean KL over the other
40 positions. The CI-masked student is the model built from the subcomponents alone, under Γ's importances computed on
the student's own activations, with Δ left out. This is the model whose trigger P(target) E42 read at 0.007. The
stochastic-masked student draws masks g + (1 − g)·r with r uniform, all layers, and Δ under its own random mask.
Read at step 1,000: the held-out P(target), the trigger CI-masked P(target), and the Pile verdict. The verdict reads
VPD's stochastic and adversarial checks on 256 Pile validation rows as ratios to this job's W₀, one start of 20 steps.

**Why these terms and not `masked_teacher`'s.** E37 Amendment 1's masked teacher training
(`src/commute/masked_teacher.py`) adds VPD's released masked losses: the stochastic subset loss and the persistent
adversary, each matching the masked student to the student's *own* output. Three things rule it out here. Its persistent adversary's
sources are shaped for 512-token rows, and the windows are 41 tokens. Its control with nothing to copy failed twice (check 3.84 and
3.14), with no tested cause. And matching the masked student to its own output does not ask the CI-masked model to show the
cipher at all. Matching it to the teacher does. The new module `src/commute/masked_copy.py` keeps `masked_teacher`'s
loop (U, V trained as parameters, dead columns' gradients zeroed, target weights set to Δ₀ + UVᵀ after every step
so Γ reads the student's activations) and drops the adversary and the warm-up. The released run's own CI-masked loss
is not in its stack (its masked losses are the stochastic subset loss and the persistent adversary, 0.5 each); 0.5 is
borrowed from them.

### Design check (`/design`)

- **Goal served.** `GOALS.md` → "If no simple Φ works": is it the update (a new capability) or the decomposition
  (frozen Γ)? E42 put teacher training's gain on a new behaviour where "Where an update goes" says a frozen Γ forces
  it: masked off. This asks whether that is forced. It also bears on the weak version directly. CI-masked
  reconstruction is one of VPD's checks under "Faithful", read here at the trigger, where the Pile reading cannot see it.
- **Nearest tried.** E42 arm A (the same map without the masked terms: masked off, 0.84 / 0.92). E37 Amendment 1's
  arm 2 (VPD's released masked losses added to teacher training on a broad fine-tune): never run, its control failed.
  E15 (Γ refit): the other branch, updating Γ, $9 per state. E10's label-trained component training: masked off too.
  Nothing has trained the CI-masked model toward a target at a trigger.
- **Patch or step.** A step: it answers the open half of the check's claim 3 ("whether any frozen-Γ map can close it
  is untested"), and it is the frozen-Γ option the check asked card `decision-e42-criterion` to name.
- **Decisions relied on.** Γ frozen; teacher training as the map (2026-10-26); the paper's adversary setting
  (2026-10-27); E21's verdict rule with the same-job W₀ floor; E34's student settings unchanged. Departure, flagged:
  as E42, this runs on the 67M although the commute line has moved to the larger model (2026-10-26), at $0, inside
  the user's instruction to fill the technique comparison's gaps.
- **What changes next.** "Kept" and "pass" give a frozen-Γ map that passes under both readings of the pending card. That
  map becomes the default for the Qwen line's narrow tasks, and the card's question loses its urgency. "Not
  kept" is the first direct evidence that a frozen Γ cannot carry a new behaviour, which makes updating Γ the commute
  line's next question. "Kept" with a failed check means the two readings cannot both be met with Γ frozen, so the
  card must choose.
- **Control first.** C1 (W₀ under the Pile harness) runs first and alone. Then G2 (seconds): with the teacher at W₀,
  the trainer's unmasked student against the model's own forward on its first batch. Then M0, alone and before
  the teacher is made: the masked recipe with the teacher at W₀, nothing to copy. It stops the job if either Pile
  ratio fails, since E37 Amendment 1's version of this control failed. G1 (E42's teacher reproduces E10 at step 10)
  follows, before arm M.
- **Reference recipe replayed.** Arm M is E42's arm A plus the two masked terms. What differs from E42's arm A's
  recipe: (1) the forward path for the unmasked term is the component model at mask 1 with Δ₀ under mask 1, where E42
  used hooks on the target linears; G2 reads that difference at W₀. (2) Adam runs on the full U, V with dead columns'
  gradients zeroed, against offsets on the alive columns. Adam is per entry and dead entries get no gradient, so the
  updates are the same. (3) There is no per-step held-out evaluation, which does not touch training. The batches are
  the same: the sampler, seed 0 and batch 32 are E42's. With the masked weight at 0, arm M is E42's arm A up to (1).
  The masked terms are the one intended difference, and M0 is what tests them on their own.
- **Class.** Inside the user's instruction to fill the technique comparison's gaps (2026-10-28), a branch named by
  E42's check and its pending card; no sign-off needed.

### Arms and run order, per task (one task per job; rand32, then rand1)

| Stage | What | Stops the job when |
|---|---|---|
| C1 | W₀ under the Pile harness, adversary seeds 0, 1, 2 | `w0_check` fails |
| G2 | the trainer's unmasked term at W₀ on its first batch, teacher at W₀ (the path's KL); the two masked terms at W₀ logged beside it | path KL > 0.01 nats |
| M0 | the masked recipe, teacher = W₀, 1,000 steps, read with the Pile harness | either Pile ratio "fail", or a NaN ratio |
| teacher, G1 | E42's teacher regenerated; step-10 P within 0.02 of E10's | G1 fails |
| references | W₀'s trigger cells; E42's saved arm-A state (`results/large/e42/<task>/A_step1000.pt`) re-read at the trigger, no Pile harness, so its CI-masked P is in a JSON at last | — |
| M | the masked recipe with the teacher, 1,000 steps; step 100 read without, step 1,000 with the Pile harness; step-1,000 state saved (bf16 offsets, `results/large/e42_amend1/<task>/M_step1000.pt`) | — |

rand1 costs the same as rand32 and is the second random cipher, so it runs as job 2 after job 1 has passed C1, G2,
M0 and G1. ident32 is not run: E42's arm A already keeps it in the CI-masked model (0.929).

### The numbers

Two tasks, rand32 / rand1. "E42 arm A" is E42's decomposition after teacher training on the task windows, read on the
same card (L40) with the same W₀ floor (0.2088). Its trigger CI-masked values come from E42's logs and are re-read in this job.

| Number | W₀ | do-nothing (E42) | E42 arm A | M0 (control) | arm M, expected |
|---|---|---|---|---|---|
| held-out P(target), target model | 1.5e-5 / 1.7e-4 | 0.994 / 1.000 | 0.964 / 1.000 | W₀'s, < 1e-3 | ≥ 0.9 |
| **trigger CI-masked P(target)**, Γ on the state's own activations, Δ out (`p_cells` L11) | about W₀'s P (in the JSON as `W0_trigger.p_ci`) | 0.016 / 0.001 | 0.007 / 0.001 | W₀'s, < 1e-3 | ≥ 0.5 |
| adversarial Pile ratio (verdict: pass ≤ 1.24, fail ≥ 1.81 on the L40) | 1 | 1.50 / 1.56 | 0.960 / 0.989 | 0.8-1.24 | ≤ 1.24 |
| stochastic Pile ratio (pass ≤ 1.497, fail ≥ 1.503) | 1 | 0.89 / 0.90 | 0.936 / 0.941 | about 1 | under 1.497 |
| G2 path KL | — | — | — | — | about 1e-3 (E37 Amendment 1: 0.00065 per Pile position) |

Logged, reported and not scored: the route shares (arm M's masked-off share e_gate is expected ≤ 0.25 if the CI-masked
P passes 0.5), the gain, the unmasked trigger P, the CI-masked and unmasked Pile ratios, damage elsewhere (M0 under
0.02), size, teacher gap and Δ's share (0 up to rounding). Also each training term's mean over the first and last 50
steps, overall and at the trigger. If the CI-masked term's trigger KL does not fall, the masked loss has stalled.

**What each number is blind to.** The trigger CI-masked P is a mean over 97 held-out windows at one position. Those
windows share the training windows' 32 keys, so "kept" means the same keys in new contexts, not new keys. It is
blind to whether the map leaks onto the same keys without the marker in plain text, to an adversary at the trigger,
and to which subcomponents carry it. The Pile verdict is one 20-step adversary start on 256 rows, and a pass there
may be lenient (E34 Amendment 5). VPD's sparsity loss is not read. The held-out P is blind to damage outside the
windows.

**Positive controls.** For the CI-masked reading: E42's ident32 arm A read 0.929 on the same reading, on a behaviour
Γ supports. In-job negatives are W₀'s cells and E42's arm A re-read (0.007 / 0.001). For the Pile verdict: E42's
least-squares split failed on the same card (1.98 / 1.83).

**Noise floor.** Held-out P moves 0.005 between batch orders on task text (E34 Amendment 4). W₀'s adversarial ratio
spreads 0.21 between seeds on the L40, and a trained decomposition's ratio moves up to 25% between batch orders. The
CI-masked P's batch-order spread is unmeasured. The 0.1 and 0.5 lines sit 14× and 70× above E42's rand32 value.

### Predictions (filed: `experiments/teacher_training/results/predictions_e42_amend1.json`, 2026-10-28 13:13:54, sha256 `d01c266d…`)

- **P1, the CI-masked model keeps the cipher.** Arm M at step 1,000, per task. Held-out P(target) < 0.9 or NaN →
  "did not copy". Otherwise the trigger CI-masked P(target): NaN → "unscorable"; ≥ 0.5 → "kept"; < 0.1 → "not kept";
  between → "partly". Absolute. *Lean: kept on rand32* (medium-low). The CI-masked term trains the scored quantity
  directly, and the held-out windows share the training keys. Against it, only subcomponents Γ switches on at the key
  can carry the cipher, and Γ may switch them off as they move. *Kept on rand1* (medium: one pair).
- **P2, it passes.** Arm M at step 1,000: `faith_score.state_verdict`'s verdict. pass → pass, inconclusive →
  inconclusive, fail → fail, below P 0.9 → did not copy, no Pile reading → unscorable. *Lean: pass on both*
  (medium-low). E42's arm A passed with 0.25 to spare, and the added terms are reconstruction losses toward the
  teacher. Against it, E37 Amendment 1's masked control rose to 3.84.

Edge cases were run through the scoring code on the CPU worker before filing. They were P(target) exactly 0.9, just
under it and NaN; CI-masked P exactly 0.5, exactly 0.1, just under each, NaN, and above the target's own P; a missing
Pile reading; M0's ratios exactly on the fail line, just under the pass line, between the lines, and NaN; and a file
with no M0 or no arm M. Every case lands in a named class. A NaN P first read "kept" and a NaN M0 ratio first read
"inconclusive"; both were fixed before filing ("did not copy" and a stop).

### Decision tree (each branch approved with this amendment)

- **C1, G2 or G1 fails** → stop and report with the numbers; no rerun until the cause is found. G2 failing means
  the trainer's forward is not the model's.
- **M0 fails** (a Pile ratio "fail" or NaN) → stop: the masked terms move the decomposition with nothing to copy.
  Report M0's ratios, damage and curves. Next is a pending card, not a run. Options: the masked terms at the trigger
  only, or one term alone, each a new amendment with its own predictions.
- **M0 inconclusive** → arm M runs. An arm M fail or inconclusive is then not attributed to copying; say so.
- **Job 1 (rand32) passes C1, G2, M0 and G1** → job 2 (rand1), whatever P1 and P2 read.
- **P1 kept and P2 pass on rand32** → a frozen-Γ map keeps a new behaviour in the CI-masked model and passes the Pile
  checks. E42's masked-off route came from the loss, not from Γ being frozen. Rewrite card `decision-e42-criterion`
  with this map as its frozen-Γ option. Next, class 1, $0: the plain-text leak of arm M's saved state
  (`kv_windows.pile_reading`, the same keys without the marker), and its route shares.
- **P1 kept, P2 fail** → carrying the cipher in the switched-on subcomponents costs the Pile check. Flag it and add it
  to the card: with Γ frozen, the two readings cannot both be met here.
- **P2 inconclusive** on a task → rerun that task on a RunPod H100 (pass line 1.40, fail line 1.61; about $1.5).
- **P1 partly** → report the curve. If the CI-masked term's trigger KL is still falling over the last 50 steps, a
  longer run (4,000 steps) is the next amendment; if flat, read it as "not kept".
- **P1 not kept** on a random cipher → a reversal of the filed lean: flag it loudly. The masked loss stalls with Γ
  frozen, so the card's option becomes updating Γ (E15's refit, or the re-gating line's shift).
- **P1 or P2 did not copy** → the masked terms cost the behaviour itself: flag it, report the curve.
- **rand32 and rand1 disagree** → report both. rand1 is a single pair, so rand32's class is the one carried to the card.

### Cost and routing

| Stage (L40; E42's job logs 40166, 40194, 40195) | Minutes |
|---|---|
| load, bases, C1 (3 × ~140 s) | 7.5 |
| G2 | 0.1 |
| M0: train (three student forwards per step, no per-step held-out pass; E42's arm A took 115-139 s with one) about 3-4, Pile read 2.3 | 6 |
| teacher, G1 | 0.7 |
| references (no Pile harness) | 0.2 |
| arm M: train 3-4, step-100 read, step-1,000 Pile read | 6.5 |

About 21 minutes per job, two jobs: MATS cluster, `debug` QoS, L40, $0 (`cluster/teacher_e42_amend1.sbatch <task>`,
45-minute wall time, `--mem=24G`). If a job has not started within 10 minutes, use RunPod: both tasks in sequence on
one H100 pod, about 20 minutes of run plus set-up, about $2. The L40's wider band (1.24-1.81 against the H100's
1.40-1.61) is the price of $0, and an inconclusive reading takes the H100 branch.

### Order of work for the fleet agent

1. Done: `src/commute/masked_copy.py`, `experiments/teacher_training/newskill_amend1.py`,
   `cluster/teacher_e42_amend1.sbatch`, `experiments/teacher_training/RUN_BRIEF_e42_amend1.md`; predictions filed;
   lint clean (`make lint` through `cpu-run`).
2. One review before the run of new measurement code, by a fresh Opus agent given only the scope and the repo path.
   The run is $0, but `masked_copy.py` is a new trainer. Fix what is real before step 3.
3. `/pod-runner` with `RUN_BRIEF_e42_amend1.md`: job 1 (rand32), then job 2 (rand1).
4. The analysis goes in an "Amendment 1" section of `reports/analyses/2026-10-28_analysis_e42-teacher-new-behaviour.md`,
   by `/writeup`. Then the notebook section, the `TRIED.md` row, the PENDING line, and the card
   `decision-e42-criterion` rewritten.

### Not in this amendment

ident32; a coefficient sweep, one masked term alone, or the masked terms at the trigger only; the persistent adversary
in the loop; a second batch order; the plain-text leak of arm M (a $0 branch above); VPD's sparsity loss; Pile-text
training (arm B) with the masked terms; a longer adversary or other starts; updating Γ; the larger model.

## Amendment 2 (2026-10-28): each masked term alone with nothing to copy, then the copy with the term that holds still

### The question, in plain words

Amendment 1 added two masked terms to teacher training and first ran the recipe with nothing to copy. That control
broke the adversarial check on Pile text (ratio 26, fail line 1.81) while the random-mask check stayed level (1.03), so
the cipher copy was never run. This amendment asks which of the two terms does the damage, by running the
nothing-to-copy control twice, once with each term alone. If one term alone holds still, the cipher copy runs with that
term and is read as Amendment 1 would have read it. If neither does, one last control puts both terms at the trigger
position only, and if that fails too the masked line stops. This is the tree the lead recommended and the user took
(decisions.md, last row: card `decision-e42-masked`).

**Restated in the Terms' words (what the lead confirms).** Every recipe is Amendment 1's: component training of the
alive (U, V) from W₀'s values, Γ and Δ frozen, on rand32's 403 training windows, Adam 1e-5, batch 32, batch-order
seed 0, 1,000 steps. The loss is KL(teacher ‖ student) with arm A's weighting (the KL at the trigger plus the mean KL
over the other 40 positions), plus masked terms each matched to the teacher:
- **ci**: + 0.5 KL(teacher ‖ CI-masked student), arm A's weighting; no stochastic term.
- **st**: + 0.5 KL(teacher ‖ stochastic-masked student), arm A's weighting; no CI-masked term.
- **trig**: + 0.5 of each, each read as its KL at the trigger alone (no weight on the other 40 positions); the
  unmasked term keeps arm A's weighting.

The CI-masked student is the model built from the subcomponents alone, under Γ's masks on the student's own
activations, with Δ left out. The stochastic-masked student draws masks g + (1 − g)·r with r uniform, all layers, and Δ
under its own random mask. A control is the recipe with the teacher at W₀ (nothing to copy); the copy is the recipe
with E42's cipher teacher. One task (rand32), one job.

**Code.** A new module, `src/commute/masked_terms.py`, generalises `masked_copy.train` with a weight per masked term
and a trigger-only option. A term with weight 0 is computed without a gradient and logged. The batches, the mask draws
and the forwards are `masked_copy.window_terms`'s, in its order, so ci and st see Amendment 1's batches and masks, and
(0.5, 0.5, arm A's weighting) is `masked_copy.train` up to the rounding of the sum. No existing file is edited. The run
script `experiments/teacher_training/newskill_amend2.py` reuses Amendment 1's G2, state read, references and P1/P2
rules.

### Design check (`/design`)

- **Goal served.** `GOALS.md` → "If no simple Φ works": is a failure the update's (a new capability) or the
  decomposition's (frozen Γ)? E42 found the cipher in subcomponents Γ keeps off at the trigger. Whether a frozen-Γ map
  can be trained to keep it in the CI-masked model is still untested, because Amendment 1's recipe broke its own
  control. This finds the part of that recipe that breaks it, and runs the copy if a part holds still.
- **Nearest tried.** E42 Amendment 1 (both terms, failed its control at 26); E37 Amendment 1 (VPD's stochastic subset
  loss and persistent adversary, matched to the student's own output on Pile rows, failed its control at 3.84, and
  3.14 at a tenth of the weights). Neither separated the terms. Amendment 1's analysis listed "each masked term alone"
  as one of three tests of its hypothesis.
- **Patch or step.** A patch on Amendment 1: the controls only say which term moves the decomposition. It becomes a
  step toward the goal only through the copy, which runs if a term holds.
- **Decisions relied on.** decisions.md's last row (this tree, $0, MATS cluster); Γ frozen; teacher training as the
  map (2026-10-26); E21's verdict rule with the same-job W₀ floor; E34's student settings. Departure, as in Amendment
  1 and flagged there: the 67M model, although the commute line has moved to the larger model.
- **What changes next.** ci holds → the copy with the CI-masked term, the direct test Amendment 1 meant; its P1 then
  answers whether a frozen Γ can carry the cipher in the CI-masked model. Only st holds → the CI-masked term is what
  breaks the check; the copy with the stochastic term says whether random masks alone pull the cipher into the
  switched-on set, but a "not kept" there leaves the frozen-Γ question open. Both fail → the trigger-only control; if
  it fails too, the masked line stops and card `decision-e42-criterion` stays as it is, with masked teacher training
  marked as tried and failed.
- **Control first.** C1 (W₀ under the Pile harness) runs first and alone, then G2 (the trainer's forward path at W₀),
  then the controls, each alone and before the teacher is made. A control does not stop the job: the tree reads it.
  C1, G2 and G1 stop it.
- **Reference recipe replayed.** Each control is Amendment 1's M0 with one masked weight set to 0. What differs:
  (1) the term with weight 0 is computed without a gradient, which changes neither the loss nor the random draws;
  (2) the loss sums 0.5 a + 0.5 b rather than 0.5 (a + b), a rounding difference. trig differs from M0 in the masked
  terms' weighting only. The copy differs from its control only in the teacher.
- **Class.** 3: a step of the tree the user approved on card `decision-e42-masked` (decisions.md, 2026-10-28, "go with
  whatever you recommend on the masked one"). No sign-off needed.

### Arms and run order (one job, rand32)

| Stage | What | Stops the job when |
|---|---|---|
| C1 | W₀ under the Pile harness, adversary seeds 0, 1, 2 | `w0_check` fails |
| G2 | Amendment 1's path check: the trainer's unmasked term at W₀ on its first batch | path KL > 0.01 nats |
| M0_ci | recipe ci, teacher = W₀, 1,000 steps, Pile harness; state saved | never (the tree reads it) |
| M0_st | recipe st, the same | never |
| M0_trig | recipe trig, the same; only if M0_ci and M0_st both read "fail" | it fails too: the tree's stop |
| teacher, G1 | E42's teacher regenerated; step-10 P within 0.02 of E10's | G1 fails |
| references | W₀'s trigger cells; E42's saved arm-A copy re-read at the trigger | — |
| M_<recipe> | the copy with the recipe the tree picks; step 100 read without, step 1,000 with the Pile harness; step-1,000 state saved | — |

Every trained state's last step is saved as bf16 offsets in `results/large/e42_amend2/rand32/` (about 60 MB each), so
a control can be re-read without a rerun (Amendment 1's M0 could not). The JSON holds each state's training curve in
50-step blocks (every term and its trigger KL, the unweighted term included) and its trigger CI-masked P(target)
(`p_ci_trigger`).

### The numbers

"Amendment 1's M0" is both terms at 0.5 with the teacher at W₀, same card, rows and batches. "E42 arm A" is teacher
training with no masked terms.

| Number | W₀ | Amendment 1's M0 | E42 arm A | a control that holds still | the copy, expected |
|---|---|---|---|---|---|
| held-out P(target) | 1.5e-5 | 1.6e-5 | 0.964 | under 1e-3 | ≥ 0.9 |
| **trigger CI-masked P(target)** (`p_cells` L11) | about W₀'s P (`W0_trigger.p_ci`) | 1.4e-5 | 0.007 | under 1e-3 | ci, trig: ≥ 0.5; st: under 0.1 |
| **adversarial Pile ratio** (pass ≤ 1.24, fail ≥ 1.81) | 1 | **25.99** | 0.960 | 0.8-1.24 | ≤ 1.24 |
| **stochastic Pile ratio** (pass ≤ 1.497, fail ≥ 1.503) | 1 | 1.028 | 0.936 | about 1 | under 1.497 |
| damage elsewhere (reported) | 0 | 0.023 | 0.024 | under 0.02 | about E42's |
| G2 path KL | — | — | — | 0.0016 in Amendment 1 | — |

The trained term's first masked-terms line should read near Amendment 1's step 1 (CI-masked 1.13, trigger 0.69;
stochastic 0.74, trigger 0.44). The term with weight 0 is logged too. Whether it falls when the other is trained is
reported, not scored.

**What each number is blind to.** Amendment 1's list holds: one adversary start of 20 steps on 256 Pile rows (a pass
may be lenient, E34 Amendment 5), VPD's sparsity loss not read, and a trigger CI-masked P that is a mean over 97
held-out windows sharing the training windows' 32 keys. New here: each control is one batch order at one weight, so a
"hold" is not shown to survive another weight or the other term beside it. The copy's recipe is picked by its
control, so its P2 is conditioned on a control that held.

**Positive controls.** For the control class: Amendment 1's M0 failed at 25.99 on the same card and rows, and E42's
copy passed at 0.96. For the CI-masked reading: E42's ident32 copy read 0.929. In-job negatives are W₀'s cells and
E42's rand32 copy re-read (0.007).

**Noise floor.** W₀'s adversarial ratio spreads 0.21 across seeds on the L40, and a trained decomposition's ratio
moves up to 25% between batch orders (E34 Amendment 4). A control's ratio across batch orders is unmeasured; Amendment
1's two 128-row blocks read 11 and 25 nats. Held-out P moves 0.005 between batch orders.

### Predictions (filed: `experiments/teacher_training/results/predictions_e42_amend2.json`, 2026-10-28 14:13:50, sha256 `1b3928be…`)

A control's class: any Pile ratio NaN, or the adversarial or stochastic verdict "fail" → **fail**; else either verdict
"inconclusive" → **inconclusive**; else **pass**. Absolute, each ratio to this job's W₀ seed 0 against its own floor.
"pass" and "inconclusive" both hold for the tree.

- **C_ci, the CI-masked term alone.** *Lean: fail* (medium). It trains only the corner where every mask is Γ's g, and
  the unmasked term pins the corner where every mask is 1. The switched-off subcomponents can then change to cancel
  the switched-on ones' changes, and an adversary setting masks between the corners undoes the cancelling.
- **C_st, the stochastic term alone.** *Lean: pass* (low). Random masks between g and 1 on every subcomponent penalise
  exactly that cancelling. Against it, E37 Amendment 1's control with VPD's stochastic subset loss failed at 3.84, and
  Amendment 1's analysis expected an average-case term alone to fail.
- **C_trig, both terms at the trigger only** (scored only if run). *Lean: fail* (low). One position of 41 asks less,
  but the trigger is where the masked students start furthest off, and there is still no worst-case term.
- **P1, the copy's CI-masked model keeps the cipher.** Amendment 1's rule on whichever copy runs, at step 1,000:
  held-out P < 0.9 or NaN → "did not copy"; else trigger CI-masked P: NaN → "unscorable", ≥ 0.5 → "kept", < 0.1 →
  "not kept", else "partly". *Lean:* ci "kept" (medium-low: it trains the scored quantity); st "not kept" (low: the
  CI-masked model is not trained, and a switched-off subcomponent is half on under random masks); trig "kept"
  (medium-low).
- **P2, the copy passes.** Amendment 1's rule (`faith_score.state_verdict`). *Lean: pass* (low): its control held and
  E42's copy passed with 0.25 to spare; against it, copying moves the subcomponents further than the control did.

Edge cases were run through the scoring code on the CPU worker before filing: each ratio exactly on and just past
its pass and fail lines; NaN in the adversarial, stochastic and CI-masked ratios (each → fail; `verdict` alone reads a
NaN as inconclusive); an infinite ratio; Amendment 1's real M0 record (→ fail); every pass/inconclusive/fail pairing
of the two controls, with and without the trigger-only control; the copy's P exactly 0.9, just under and NaN; its
CI-masked P exactly 0.5 and 0.1, just under each, NaN, and above the copy's own P; no Pile reading; a copy with only
its step-100 read; a file with W₀ only. Every case lands in a named class, and the tree picks ci when both hold.

### Decision tree (each branch approved with the decision)

- **C1, G2 or G1 fails** → stop and report with the numbers; no rerun until the cause is found.
- **Exactly one of C_ci, C_st holds** → the copy with that term alone (same job).
- **Both hold** → the copy with the CI-masked term alone (it targets the trigger reading). M_st is not run.
- **Both fail** → M0_trig (same job). Holds → the copy with recipe trig. Fails → stop: the masked line ends; card
  `decision-e42-criterion` stays as it is, with masked teacher training marked as tried and failed. Report all three
  controls' ratios, damage and curves.
- **A control holds as "inconclusive"** → its copy runs, and a fail or inconclusive of that copy is not attributed
  to copying; say so.
- **The copy: P1 kept and P2 pass** → a frozen-Γ map keeps a new behaviour in the CI-masked model and passes. Rewrite
  card `decision-e42-criterion` with it as the frozen-Γ option. Next, class 1, $0: the plain-text leak of the saved
  copy (`kv_windows.pile_reading`) and its route shares; then rand1 with the same recipe, as Amendment 1's job 2
  would have run (one $0 job).
- **P1 kept, P2 fail** → keeping the cipher in the switched-on subcomponents costs the Pile check; flag it and add it
  to the card.
- **P2 inconclusive** → rerun on a RunPod H100 (lines 1.40 / 1.61), about $1.5.
- **P1 partly** → report the curve; still falling over the last 50-step blocks → a 4,000-step amendment; flat → read
  as "not kept".
- **P1 not kept, recipe ci or trig** → a reversal of the filed lean: flag it loudly. The card's option becomes
  updating Γ.
- **P1 not kept, recipe st** → as filed; it does not settle the frozen-Γ question, since nothing trained the
  CI-masked model. Next is a pending card (the CI-masked term with an adversarial term matched to the teacher, option
  (c) of `decision-e42-masked`), not a run.
- **P1 or P2 did not copy** → the masked term costs the behaviour itself: flag it, report the curve.

### Cost and routing

| Stage (L40; job 40338's timings) | Minutes |
|---|---|
| load, bases, C1 (3 × 124 s) | 7.5 |
| G2 | 0.1 |
| M0_ci: train about 3 (one term with a gradient instead of two), Pile read 2 | 5.4 |
| M0_st | 5.4 |
| M0_trig, only if both fail | 5.4 |
| teacher, G1, references | 1 |
| the copy: train about 3.3, step-100 read, step-1,000 Pile read | 5.7 |

About 25 minutes with a single-term copy, 24 if every control fails, and 31 at most (both fail, trig holds). One job:
MATS cluster, `debug` QoS, L40, `--mem=32G`, 55-minute wall time, $0 (`cluster/teacher_e42_amend2.sbatch`). If it has
not started within 10 minutes, a RunPod H100 runs the same script, about $1.5; there the lines are 1.40 / 1.61.

### Order of work for the fleet agent

1. Done: `src/commute/masked_terms.py`, `experiments/teacher_training/newskill_amend2.py`,
   `cluster/teacher_e42_amend2.sbatch`, `experiments/teacher_training/RUN_BRIEF_e42_amend2.md`; predictions filed;
   lint clean (`make lint` through `cpu-run`); scoring edge cases run on the CPU worker.
2. One review of `src/commute/masked_terms.py` and `newskill_amend2.py` by a fresh Opus agent given only the scope and
   the repo path. The run is $0, but the trainer is new. Fix what is real before step 3.
3. `/pod-runner` with `RUN_BRIEF_e42_amend2.md`: one job.
4. The analysis goes in an "Amendment 2" section of `reports/analyses/2026-10-28_analysis_e42-amend1.md` or a new
   dated analysis, by `/writeup`; then the `TRIED.md` row, the PENDING line, and card `decision-e42-criterion` or a
   new card as the tree says.

### Not in this amendment

rand1 and ident32; a second batch order; other weights; the persistent adversary in the loop; the masked terms on
Pile rows; M0_trig when a single-term control holds; a second copy when both hold (M_st); the plain-text leak of the
copy; VPD's sparsity loss; a longer adversary or other starts; updating Γ; the larger model.
