# Plan: teacher training on every update E93 scored, read by the same table (E111)

*2026-11-06, drafted by Fable for `qwen-decomp-lead`. User-directed (`decisions.md`, last row, 2026-11-06: "run teacher
training on everything you did with least squares, especially the bits where it failed"). Apparatus, room table, rates
and the pre-mortem: `reports/plans/2026-11-06_plan_e111-teacher-table_appendix.md`. Advice and options:
`reports/plans/2026-11-06_advice_e111-teacher-recipe-FABLE.md`. No code, no predictions file, no job yet.*

## Design check

- **Goal.** D1 (`GOALS.md` weak version): a map on Qwen3-1.7B that keeps the decomposition faithful for a real update.
  E93 scored least squares on D1's table; this scores teacher training, the only trained map in the Terms, on the same table.
- **Nearest tried.** E93 (least squares on these states, `experiments/TRIED.md:195`); E76 (teacher training towards u1
  from zero offsets: clean term 0.0121 nats at 0.52 of δW's squared size, masked term inflated to 3.42×, `TRIED.md:156`);
  E79 (under own masks the clean-term arm reads as do-nothing, 0.078 against 0.066, the masked-term arm worse, 0.198,
  `TRIED.md:157`); E43 on the 67M (teacher training plus remainder worse than do-nothing, `TRIED.md:91`). None read
  teacher training on D1's table or started it from least squares. Class 3, user-directed; one point below goes to the user first.
- **Patch or step.** A step: the second map on the table that judges D1, on the same rows and job as the first.
- **Decisions relied on.** Teacher training uses Adam (`decisions.md:50`, 67M); Δ may change a little (`decisions.md:71`);
  D1 is judged by this table without the adversary (`decisions.md`, ask 31). Departures: the rate is per state, set
  from a measured room, not E76's 1e-6; the start is least squares' split, not zero offsets. Both are argued below.
- **What changes next.** Section 5.
- **Control first.** Stage 1 runs RL seed 70 alone: the zero-offset state must reproduce E93's stored reads, the first
  25 steps must walk inside a filed band, and its two arms' exchange rate goes to the user before stage 2.
- **Numbers.** Section 4: the table's six columns per state, each with its value at W0, do-nothing and least squares
  already stored by E93, plus ‖r‖² and the kept step.
- **Saved.** Offsets in fp32 per state, the readings curve, the table's `state.json`.

**The user's point to see before stage 2.** Column 2, the share left in Δ, is what least squares minimises: any change
inside the subcomponents adds to it (appendix B). So no teacher-training recipe can repair the three share misses
(shuffled s0, s70; own-samples), and on the eight other states it may spend only a small **room** before it flips a
pass. The plan reads the table as filed and files those three as foreseen misses. What E111 can answer is whether a
trained correction inside that room repairs columns 1 and 5 where least squares missed (RL s70; own-samples' masked
read) and keeps the passes. If the user would rather score columns 1 and 3-6 with column 2 printed beside, the run
does not change, only the verdict rule.

## 1. The question, in the Terms

**Teacher training** (`GOALS.md:61`): component training with the loss KL(teacher ‖ student), teacher = the fine-tuned
model, Γ and Δ frozen. Here the student starts at the **least-squares split**'s subcomponents and trains a correction $r$
on E76's 200 often-on columns per matrix; its moved part is least squares' plus $r$'s. **E93's table** reads any state's
moved part: columns 1, 3, 4 at W0 + moved part, columns 2, 5, 6 at the fine-tuned weights with the remainder in Δ
(E43's "plus remainder"), as it read least squares. One state per update: template s0/s1, TOFU s42/s142, RL s0/s70,
shuffled s0/s70, own-samples s0, u1, u2.

Does the trained correction pass every column where least squares passed, and repair the clean and masked misses
on the RL arms, without leaving the share's room?

## 2. The recipe (Gate 2: each setting sourced)

| setting | value | source |
|---|---|---|
| map | E76's arm A: clean term KL(FT clean ‖ student clean), Adam, no masked term | E79: under own masks A reads as do-nothing, B worse (`TRIED.md:157`); the table's column 5 is own masks |
| start | least squares' pieces on every alive column, fixed; $r$ trained on 200 often-on columns per matrix | E43: from zero offsets the change has cosine 0.045-0.106 with δW and the remainder exceeds δW (`...e76-masked-teacher.md` → "The student matches outputs"); column 2 then fails on every state. E76's cut: 0.835 of Σg, fits the L40 |
| Γ, Δ | frozen during training (student = W0 + change); the table's columns 2, 5, 6 put the remainder in Δ | `GOALS.md:61`; `decisions.md:71`; E93 read least squares the same way |
| optimiser | Adam, betas (0.9, 0.999), no decay | `decisions.md:50`; `qwen_teacher.train`, `src/commute/qwen_teacher.py:201` |
| rate | per state, lr = 1e-6 × √(room / 0.72): 1.3e-8 on the maths arms, 3.0e-8 template, 4.6e-7 TOFU, 1.0e-7 u1, 7.8e-8 u2 | room from E93's stored share, null and spread (appendix B); the walk 1.2e-3 per step at 1e-6 from E76's arm A (appendix C) |
| steps | at most 300; readings every 25 | E76's control length (`qwen_masked_teacher/common.py:63`); the stop rule, not the count, ends training |
| stop and keep | kept state = last reading with ‖r‖² ≤ room; end at the room, at 300 steps, or when the 8-row K has not fallen over four readings | the table's own column 2 (`rules.2_share`); no-room states use RL s70's room, 0.00012 |
| tokens | 2 rows × 256 tokens per step, one of the arm's text and one FineWeb; pools of 32 outside the table's rows | E76 (`common.py:34-37`); E93's row sets (`qwen_d3.py:121-131`) |
| precision | TF32 training forwards, readings fp32; offsets fp32, saved fp32 | `qwen_teacher.py:11`; appendix D |

**What stops inflation.** E76's B grew to 3.42× δW's squared size because nothing in its rule read size. Here the
table's column 2 is the cap and the run enforces it: the kept state is the last inside the room, so columns 2 and 3 hold
by construction on the eight states with room, and what they still tell is whether $r$ was spent. ‖r‖² is read every 25
steps without a forward; the rate is set so that 300 steps use about half the room. A state that overruns the room
before step 25 is a stop in stage 1 (the forecast was wrong) and a kept-at-step-0 state later ("no correction fits").

## 3. Run order, stages, early checks

Every stage is one `debug` job on the cluster L40 through one pod-runner; each state writes its `state.json` as it
finishes; a resubmission skips written states (E93's `run.py:328`). Times from E93's and E76's logs (appendix E).

**Stage 1, control, alone: RL seed 70** (a clean miss, a masked miss, and room), two arms, about 70 min.
1. The least-squares state at zero offsets reproduces E93's `R_s70/state.json`: K(FT ‖ W0 + moved) in the training
   forward within 1e-3 relative of `clean.moved` (0.002327 nats); the table's own share, size and K within 1e-5
   (0.094253, 0.905744, 0.002327). A miss stops the job: the correction would train on the wrong base.
2. First minutes: gradient on dV and dU finite and nonzero (E76's `first_minutes`, `arms.py:141`); at step 25 ‖r‖²
   within 0.3-3× the forecast 4.9e-6 and under a quarter of the room; median √v over Adam's ε above 10; the 8-row K not
   above its step-0 value. Each reading logs its value and the forecast.
3. Arm "in room": rate 1.3e-8 to the stop rule. Arm "10×": rate 1.3e-7, 300 steps, no room stop: the positive control,
   what column 1 falls to when size is spent (E76's A reached 0.0066 of W0's K from zero).
4. The table reads both kept states; the stage report (appendix G) goes to the user with the exchange rate between
   ‖r‖² and the two misses. If the in-room arm leaves column 1 at or over 0.02 and column 5 within its half-width while
   the 10× arm repairs column 1, the room is the limit; stages 2-5 still answer "does the correction keep the passes".

**Stage 2**: shuffled s0, s70, own-samples s0, about 90 min; column 2 foreseen misses, read on columns 1, 3, 5 and the
excess share. **Stage 3**: template s0, s1, RL s0, about 90 min. **Stage 4**: TOFU s42, s142, about 75 min. **Stage 5**:
u1, u2, about 60 min. Each state's zero-offset reproduction is its own first check.

**Total: about 6.5 h of card over five jobs, $0.** That crosses the 6-h clause, so the run is **gated by time**: the
pre-mortem (appendix F), sourced settings (above), the $0 rehearsal (stage 1 is it: every code path, the resume, the
saves, on the tightest state), early checks (above, in code), one review of plan and code by a fresh Opus agent, then
the user's go-ahead for stages 2-5 after stage 1's report. Stage 1 may run after the review.

## 4. What each number should read

The table's columns, per state (`experiments/qwen_d1_table/results/predictions.json`, `rules`). Least squares' values are
E93's stored reads, which the job recomputes on the same rows (`...e93-d1-table.md`, tables).

| column | W0 | do-nothing | least squares (E93) | teacher training (this plan) |
|---|---|---|---|---|
| 1 clean, K / K(W0) | 1 | 1 (nothing moved) | 0.0055-0.0182 on passes; RL s70 0.0238; own-samples at floor | at or under least squares' on every state (the loss); **RL s70 under 0.02 is the question** |
| 2 share | 1 | 1 | 0.087-0.101, null 0.090-0.102 | least squares' + ‖r‖²/‖δW‖², inside the room on 8 states; **misses on shuffled s0, s70, own-samples, foreseen** |
| 3 size | 0 | 0 | 0.900-0.913 | least squares' + at most 0.003; holds |
| 4 kept clean, graded arms | 0 | 0 | 0.991-0.997 against t = 0.5's 0.88 | within 0.01 of least squares'; holds |
| 5 K under own masks, minus do-nothing | 0 by definition | 0 | −0.022 to +0.000029 nats; RL s70 and own-samples miss by 3.5e-6 and 1.3e-5 | **the open number**; $r$ enters the weights, not the masks (appendix D); lean: within the half-width of least squares' on every state |
| 6 kept masked | — | reference | no room on template and TOFU | no room, as least squares |
| ‖r‖² | 0 | 0 | 0 | 0.3-3× the forecast at step 25; at or under the room when kept |

Leans, not filed yet (the predictions file is the writer's, after this plan is confirmed): column 1 on RL s70 repaired
by the in-room arm, weak; column 5 unchanged within its half-width everywhere, medium; the seven least-squares passes
kept, strong; the 10× arm repairs column 1 and overruns column 2 by about 100× the in-room arm's excess, medium.

## 5. What result changes what we do next (D1's verdict)

- **Passes kept, RL s70 repaired.** Teacher training from least squares is D1's map on this table: 8 of 11 states pass;
  the three share misses are the column's, not the map's. The write-up says so; a decision card asks whether the share
  rule at update norms under 0.4 (null within 3% of the share) is a test or a tie, with E93 Amendment 1's adjoint column beside.
- **Nothing repaired inside the room; the 10× arm repairs column 1.** Any map inside the subcomponents' span is least
  squares on this table, to the room. Least squares stays the map; the write-up names the room per state; no third map.
- **A pass broken** (column 5 worse than least squares' by over its half-width on a passing state). The clean term
  hurts under own masks on this model, as E43's plus-remainder did on the 67M; teacher training is dropped for D1 on
  Qwen, and least squares stays.
- **Stage 1 control fails** (base not reproduced, walk outside the band): stop, fix, rerun stage 1; nothing else runs.

Two amendments at most. Amendment 1 is reserved for a masked term under the state's own masks, the loss column 5
reads, if that column is where the in-room arm falls short: new code (2.2 s per step, E76 `run.log:91`) with its own
first-minute values. Not run by design: the adversary; E76's masked term (worse under own masks, E79); all alive
columns in training (memory); a second seed; teacher training from zero offsets (fails column 2 by construction).

## Hand-off

A fresh session writes `experiments/qwen_teacher_table/{run,rules}.py` and `results/predictions.json` from this plan and
its appendix, reusing `qwen_d1_table.run` (the table) and `qwen_masked_teacher.arms` (the training loop) through
`commute.jobs.load_experiment`, then one review, then stage 1 through `/pod-runner`. Read first: this plan, the
appendix, `...e93-d1-table.md`, `experiments/qwen_d1_table/run.py`, `src/commute/qwen_teacher.py`.
