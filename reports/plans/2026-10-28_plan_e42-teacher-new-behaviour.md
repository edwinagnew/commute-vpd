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
