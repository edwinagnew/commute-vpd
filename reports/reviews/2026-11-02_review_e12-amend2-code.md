# Review: E12 Amendment 2 code and filing, before its first GPU run

Fresh Opus agent, 2026-11-02; it wrote none of this. Target: commit d9a0a5d8 (`experiments/new_vs_existing/amend2.py`,
`cluster/new_vs_existing_amend2.sbatch`, `results/predictions_amend2.json`) against
`reports/plans/2026-11-02_plan_e12-amend2-rand32b-fall.md`. Working files: `~/commute-data/scratch/commute-lead/e12a2-review/`.

**Verdict.** I found no crash, no overwrite of Amendment 1's files and no wrong score. It can go on the `debug` job as
it stands if the lead accepts the four medium findings. One of them, M1, is worth fixing first: P5's filed reading does
not follow from its rule, and the filing can still be changed because nothing has run. No GPU forward has been run;
"no crash" below rests on reading the code and on the fake end-to-end run.

## High

None.

## Medium

**M1. P5's rule leans towards "nearer" before any data, and the filed reading of it is backwards.**
`predictions_amend2.json` P5 and `what_changes_next`; `amend2.py:172-176`. Write r for student size over teacher size
and c for the cosine between their changes. Distance squared is then 1 + r² − 2rc. Amendment 1 already gives r: 0.87-0.89
on rand32b and 1.34-1.35 on rand32 (analysis, line 109-110). So:

- rand32b's distance squared is about 1.77 − 1.76c, and rand32's is about 2.81 − 2.69c.
- At equal c, rand32's distance is the larger for every c up to 1. At c = 0.5 the two read 0.94 and 1.21, a gap far above
  the seed range, so the rule reads "nearer".
- "Further" needs a much lower cosine on rand32b: with c = 0.5 on rand32, under about 0.17.
- "No separation", the filed lean, already needs rand32b's cosine to be well under rand32's.

Failure scenario: the run reads "no separation" or "nearer", and the filing says "the reach reading loses its one direct
test". "No separation" in fact implies rand32b's students are less aligned with their teachers, which is what the reach
reading would predict. The size ratios decide much of this number, the same confound the Amendment 1 review raised as M3.
The filing's "the size ratios ... say nothing about direction" is true of the ratios, but the rule mixes the two.

Fix, before the run: store the cosine and the unnormalised distance beside it, which needs no extra compute, and rewrite
the clause. A positive control is also cheap: each rand32b student's distance to the other seed's teacher should exceed
its distance to its own. The filing has "none in this job", which `.claude/CLAUDE.md` → Rules does not allow.

**M2. "Read from fp32 weights" holds for the file, not for the forward, and nothing compares the two forwards on Γ's
values.** `amend2.py:3`, `amend2.py:140-143`; `vpd.py:179`; plan line 31. `compute_ci` runs under bf16 autocast, and I
expect autocast to cast each `nn.Linear`'s weight to bf16 at the op (PyTorch's documented list; not run here). So Γ's
inputs at the state come from W0 + dW rounded to bf16 entry by entry. The student was trained on a different forward,
the hook in `component_training.py:71`, which adds the change to the activations. The teacher is the weight form on
both sides. So students alone are read on a forward they did not train on.

The evidence so far covers the output side only, and there the two forms agree: held-out P(target) in training against
the read's T1 P is 0.99260 / 0.99266 on rand32b s0 and 0.62905 / 0.62913 on rand32 s0. Nothing on file compares them on
Γ's values, where the fall lives. Rand32 shows no fall on the same path, so my prior that this explains 1 nat is low. It
is still the exact pair the 2026-11-02 rule asks for ("compare both sides ... at the states used"), and P1 "holds"
cannot tell this apart from a real fall.

Fix, for about 2 s per student: in `student()`, read the gates once more with `component_training.hooks` on at W0, using `Offsets(init=...)` rebuilt from
the fp32 `comps` that `Job.train` returns (it does not return the offsets), and report L01 on those beside L01 by weights. Reported only, no stop; its tolerance
has no measured noise floor yet. If the lead skips it, the analysis must carry it under "Not run".

**M3. Two filed outcomes have no branch in `what_changes_next`.** `amend2.py:41-45`, filing `what_changes_next`.
- P1 "holds" with P4 "mixed" has no clause. It is a likely outcome: the two seeds' falls differ by 20% and the ladder has
  coarse steps, so one state at k = 10 and the other at k = 30 reads few / middling, hence "mixed".
- One state "not at the trigger" and the other anything else also reads "mixed", and `agree()` then hides that the clause
  about "not at the trigger" half applies.
- P3's five block labels have no clause at all, so P3 can come out many ways and change nothing.

This is the "case no branch covers" the rules ask to fix before filing, at a round trip each if left. Fix: add one clause
("mixed: report per state, no follow-up, the user hears it") and say what P3's labels decide, or that they decide nothing.

**M4. A control stop discards the read of the state that stopped, the one the plan calls a result.** `amend2.py:218-219`
come before the record at `amend2.py:231`. Failure scenario: rand32b s0 retrains to a fall of 0.9 nats. The job exits at
about 3 min with one difference in `checks`, plus the cells in the log's `[read ...]` line and bf16 offsets. It keeps no
J, no restore, no distance and no trigger file. The plan says that outcome "itself answers part of the question", but it
is the state you would then most want read. Fix: move the repro `near` to after `dump(out)` at line 237, and keep the
1e-6 identity at line 217 where it is. Two lines, and the stop still fires before the next student.

## Low

- **L1. The restore's control cannot fail on the path that is scored.** `amend2.py:220-222` writes the fine-tune's values
  at every position, through `hybrid` alone, so it reads back the same tensors by construction. The scored path is
  `with_trigger`, then `hybrid(low, mix, sel)`, and has no control. Add: `back(everything)` equals `at_trigger`, and a block
  of nothing gives 0. Both are free.
- **L2. The stop in `student()` goes beyond the filing's wording.** The 1e-6 identity at line 217 also stops order-1
  students, while `stop_rules.later students` says they have nothing to stop on. The behaviour is right; the filing's
  sentence is narrower.
- **L3. Saved files are lossy for the reading they are kept for.** Line 229 stores trigger values with `.half()`, which
  is fp16, and only at the trigger. `save_offsets` stores bf16 (`arms.py:52`). A later "reading without a rerun" (plan line
  47) can reuse the trigger values approximately, but cannot rebuild a student's L01 or any restore, since the other
  positions are not saved. The plan should say so; the Amendment 1 review's M4 is the same trap.
- **L4. The P2 `against_reference` noise comes from one table.** `amend2.py:72`. Only rand32b has an order-1 student, so
  rand32's own set noise is not in the gap. A rand32 set noisier than rand32b's could read "lower" on noise.
- **L5. The record does not catch a changed tree.** `amend2.py:247` stores HEAD only; Amendment 1 had `code_unchanged`.
  HEAD moved twice during this review from other sessions' commits, none touching this code. An uncommitted edit in the
  shared clone at job start would go unrecorded.
- **L6. No resume, and files left behind.** Any stop or timeout leaves the results file, so the runner must move it
  aside. After that, `results/large/amend2/*.pt` are overwritten without a check and `run_amend2.log` appends. Put the `mv`
  in the runner's brief, along with the folder.
- **L7. Time.** The plan's 14 min leaves out the new reads. Amendment 1's log gives 116-118 s per student including
  training and two reads; the roughly 15 extra forwards and gate clones add perhaps 15-25 s per student. My estimate is
  about 18 min, against the 30 min limit.

## Can each prediction come out more than one way?

Ran through `amend2.py`'s own rule functions (`dry.py`), against the filed text.

| rule | reachable on this run | dead, given the controls | edge cases I tried |
|---|---|---|---|
| P1 | holds, gone, varies | unscorable | 0.5 and 0.15 exactly, a rise, one each way, one state, nan: as filed |
| P2 | same, partly, different; no power is unlikely | unscorable, since the sets are not empty | J at each anchor, anchors equal, nan: as filed |
| P2 `against_reference` | lower, higher, no different | unscorable | gap = noise, noise 0, empty: as filed |
| P3 | four blocks, spread, not at the trigger, mixed | "no fall", unscorable | tie, all negative, share over 1: as filed |
| P4 | few, middling, diffuse, not at the trigger, mixed | "no fall", unscorable | each k, no k, short ladder: as filed |
| P5 | all three, but tilted (M1) | unscorable | ties, reversed, short list, zero teacher: as filed |

"No fall" cannot occur on P3 or P4. The stops hold L01 within 0.05 and the fine-tune's within 0.02 of Amendment 1's,
so the smaller fall is at least 1.023 − 0.07 = 0.95 nats, above the 0.5 line. The branch is harmless.

P2 has power on E12's own sets, recomputed from `new_vs_existing.json`: J between seeds 0.655 (rand32b), 0.533 (rand32),
and J between the two tables 0.216-0.259. E46's rebuilt sets may differ by a member or two.

## Checked and found sound

- **Amendment 1's files.** `amend1.OUT = OUT` runs at import, and `dump` and `check` read that module global, so all
  writes go to the Amendment 2 file; `.tmp` differs too. `STATES` is `large/amend2`. `Job` writes nothing else. After my
  fake run `git status` showed no change in the clone.
- **End-to-end bookkeeping.** `dry.py` ran the real `main`, `fine_tunes`, `student`, `restored`, `score`, `dump` and the
  refusal on rerun, with a fake `Job` and fake gates, bf16 and batched as `route_cells` returns them. Exit 0, JSON valid,
  six offsets and six trigger files, in the filed order. This goes further than the writer's `stub.py`, which did not run
  `main`, `fine_tunes` or `student`.
- **Interfaces against `src/commute/`.** Each `recruit`, `signatures`, `local_faith` and `teacher` call has the right
  shape and device. `top_by`, `hybrid`, `with_trigger`, `at_trigger` and `sel_of` give full indices and handle empty blocks.
- **Direction.** `restored(g, f)` writes the fine-tune's values into the student's, and the control writes the fine-tune's
  into W0's. `fall` = L01(fine-tune) − L01(student). Expected falls recomputed from Amendment 1: 1.2248, 1.0232, 0.0759, 0.0838.
- **The 1e-6 identity at line 217 should not stop falsely.** Amendment 1's do-nothing reads have L10 = L00 exactly while
  the target weights differ, so the masked forward ignores them. Its "r(b) against the fine-tune's" reads 0 under different
  components, so gates ignore those. Repeated reads there differ by exactly 0.
- **Recipe.** `src/commute` differs from Amendment 1's commit 48cf644 only by two added modules (`copying.py`,
  `ctxpatch.py`), and `amend1.py` and its filing are unchanged. Nothing in scope changed from d9a0a5d8 to HEAD when last checked.
  Training seeds its own generator and starts offsets at 0.
- **Input files.** Held-out windows are 97 × 41 with the trigger at index 20, on both tables. Each fine-tune file is fp32
  with exactly the 24 keys of `alive.pt`. 9,881 alive subcomponents, so the ladder's 300 fits.
- **Filed sources.** `gone_line`'s 0.132 is 0.1321 (ident32 s0), and 0.0005 and 0.0006 match `noise_r`. The batch-sharing
  claim holds (`finetune.py:197-201`, same generator rule, batch 32, 403 rows). The sets' sizes 27, 21, 41, 51 match E12.
- **Memory.** A gate pattern is 97 × 41 × 38,912 values, at most 620 MB. The most held at once is about 12 patterns, 7.5 GB, under `--mem=32G`.
- **Sbatch.** Same shape as Amendment 1's, `rc` survives the pipe, and the cluster has only L40s, so the card stop cannot
  fire by chance.

## Not run

- Any GPU forward, or any model load. Autocast's cast of the weights in M2 is from PyTorch's documentation, not measured.
- Whether a same-order retrain reproduces within `repro_tol`: never measured on any table, and the first student shows it.
- The sets on E46's rebuilt fine-tunes; the numbers above are E12's.
- The P5 algebra in M1 assumes one r per table from Amendment 1's sizes; the true cosines are unknown until the run.
- `sacct` is down on this node, so Amendment 1's memory peak stays unrecorded.

## What the lead did

Fixed before the first run (2026-11-02; none of this has run on a GPU):

- **M1.** P5 now scores the cosine between a student's change and its teacher's change (each from W0, all 24 matrices, fp64), under E12's separation rule: rand32b lower reads "less aligned", higher "more aligned", else "no separation". The unnormalised distance and the size ratio are stored and reported, unscored. Positive control added: the same cosine against the other seed's teacher of the same table (`cosine_other_seed`), reported, no stop. P5's `about`, `lean` (no separation, low), `blind_to`, `positive_control` and the `what_changes_next` clause were rewritten; the plan says why (distance squared = 1 + r^2 - 2rc is tilted by r).
- **M2.** A hook-form L01 (`L01_hook`) is read per student on the forward it trained on, beside the weight-form L01. Reported only, no stop, no prediction. Named in the plan as the check of the bf16 autocast / hook pair; expected agreement within Amendment 1's batch-order noise.
- **M3.** One catch-all clause added to `what_changes_next`: any combination no clause names (including P1 holds with P4 mixed, and P3's block labels) is reported to the user with both numbers, and nothing further runs. No other clause added.
- **M4.** The `repro_tol` stop now fires after the state's `dump`, so a missed reproduction keeps that state's J, restore, distance and trigger reads. The 1e-6 identity stays where it was.

Skipped: L1, L2, L3, L4, L5, L7 (no change to any other rule, threshold or stop). L6 stays for the runner's brief, which the lead writes: move the results file aside after any stop, and note that `results/large/amend2/*.pt` are overwritten and `run_amend2.log` appends.
