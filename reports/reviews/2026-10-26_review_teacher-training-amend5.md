# Review: E34 Amendment 5 (adversary start, copy uniqueness), independent check

*2026-10-26. Reviewer did not write the code or run the job. Read: plan → "Amendment 5", `predictions_amend5.json`,
`teacher_training_amend5.json`, `results/large/run_amend5.log`, `amend5.py`, `faithfulness.harness_pgd`,
`gamma.pgd_sources` / `harness_start`, `finetune.cosines`, `saved_states.offsets_state`, the report section
(`reports/analyses/2026-10-23_analysis_teacher-training.md:1287-1548`), the E34 notebook's Amendment 5 cells,
`figures/amend5*.png`, `experiments/TRIED.md:74`. Also `teacher_training_amend4.json` and E37's report for cross-checks.*

## Confirmed

- **Every number** in the report's tables, text and scorecard, the notebook, the figures and the TRIED row matches the
  JSON and the log (KLs, ratios, 2.4% / 4.3% / 11.1% spreads, x5.4 / x5.8 gaps, +20% / +17% at 160 steps, cosines and
  their angles, P(target), controls, 24 s per block). The A4 twin P(target) values (0.995 / 0.871 at step 100) and
  E37's 3.91 against 3.92 match their sources.
- **Scoring by the filed rules.** P1 holds (rows256 10.189 / 9.240 = +10.3%; harness_b0 1.5954 / 1.5884 = +0.4%).
  P2 holds (start 1.667, +5% from 1.588). P3 fails (harness160_b0 0.786 / budget160 0.708 = +11.1% > 10%). P4 fails
  with the reading "one of many" (Pile text 0.429 < 0.5; task text 0.913).
- **The "start" variant changes only the start.** It is `pgd_sources` with `init=harness_start(seed 0)`: same rows,
  step 0.1, readout and code as the default budget reading (`amend5.py:124-127`). `harness_start` reproduces the
  authors' draws: the zero-step check agrees to 4e-8 (draws do not depend on the state, so this carries to the
  teacher). The identity check (0.996 at W0, 1.040 at the teacher) shows the rest of the loop matches the harness.
  The identity tolerance and zero-step check were added after filing, and the report says so.
- **The section-1 caveat** ("two random starts on one state at one seed, so we cannot say which reading is the
  typical one", :1384) and "equally random" (:1335) are right. The run has one draw per code. It cannot tell a weak
  harness draw from a lucky budget draw, or either from a wide spread over starts. No budget-code draw at another seed
  was read on the teacher.

## Corrections

1. **P2's "both implicate the start"** (`analysis:1505`, `TRIED.md:74`, notebook scorecard P2 row). budget.bf16
   landing at 9.219, next to the budget's 9.240, shows only that the readout does not carry the gap. It points to the
   start only by elimination. The filed JSON rule also counts a reading that stays at its own default as a pass. So
   the budget.bf16 half of P2 could not fail, which is looser than the plan's "moves the ratio" (plan:513-517). The
   verdict stands on the start reading alone. Suggested wording: "holds on the start reading (1.667); budget.bf16 rules
   out the readout."
2. **"That blind spot can hide a difference, not create one, so 0.43 is an upper bound on how alike the two Pile-text
   placements are"** (`analysis:1429-1431`; notebook "can only hide a difference, so the two placements are at most
   this alike"). This holds for the gauge. But the cosine has a second blind spot that works the other way. It weighs
   directions with no effect on behaviour the same as directions that carry it. The B arm trains at batch 8, against
   32 for A (`config.batch_b`, `batch_a`), with Adam. So a seed-specific drift with no behavioural effect can lower
   the cosine without the behaviour-carrying placement differing. One fit agrees with this. If each copy is a shared
   part plus independent seed noise, 0.429 means the shared part is 0.65 of the copy's norm. That shared part then
   lines up with δW at about 0.165 / 0.65 = 0.25, as the task-text copies' shared part does (about 0.27). 0.43 is a
   bound on how alike the total weight changes are, not the placements. Add this blind spot beside the gauge one
   (:1471-1472 too).
3. **"Same behaviour" rests on one number.** Only held-out P(target) was read on the copies (0.974 and 0.974). Damage
   elsewhere and KL to the teacher on Pile were not read (Not run, :1531). Places to change: `analysis:1299-1300`
   ("one of several weight changes with the same behaviour"), :1425-1426, :1457-1458, the figure title of
   `amend5_cos.png` ("share the behaviour"), the notebook, and `TRIED.md:74`. Use "with the same target probability".
4. **"Which other verdicts it touches was not tested here"** (`analysis:1455-1456`). True of this run, but data
   already on disk bears on it and should be cited. A4 read all six do-nothing states from both starts:
   `teacher_training_amend4.json`, budget KL / 0.663 against the harness ratio at 20 steps. The budget code's start
   reads higher on 5 of 6 and lower on none: r0 SGD 2.09 vs 1.12, r0 Adam 1.68 vs 1.35, r1 Adam 2.08 vs 1.52,
   emoticon Adam 2.43 vs 1.50, r1 SGD 9.24 vs 1.59, emoticon SGD 1.00 vs 0.98. The r0 SGD harness pass (1.12) reads
   above the 1.241 line from the other start. Two caveats: these A4 readings also differ in rows (+10% on r1 SGD here),
   and all six use the same pair of draws. So they cannot tell a weak harness seed-0 draw from a wide spread. Still,
   they make "one-start verdicts may be lenient" a finding on more than one state, and it reaches a pass.
5. **"Those two readings differ only in the start"** (`analysis:1414`, and the P3 surprise at :1480-1481).
   harness160_b0 and budget160 also differ in code and readout. The identity and readout checks put those at 4% or
   less, so this does not change the reading. Say "differ in the start (the code and readout differences read 4% or
   less)".

## Headline changes

- The recommendation's strength is about right. "Can understate" and "may be lenient" fit one teacher and two starts.
  Point 4 above gives it support on more states. The answer's "the start carries all of it" should keep its caveat
  next to it: two draws, so it is unknown which draw is typical.
- P4: replace "a Pile-text copy is one of several weight changes with the same behaviour" with "two Pile-text copies
  with the same target probability have total weight changes at cosine 0.43; whether the difference carries behaviour
  is untested". The larger-model note ("needs two batch orders") stands. Cheap CPU follow-up on the saved offsets:
  per-matrix cosines and norms.

## Not checked

- `plot_amend5.py` code (the figures were checked against the JSON by eye); the notebook's code cells; spend ($5.82)
  and pod times beyond the log.
