# Review: E47 claim-tests code (commit 8c1b478, `faithfulness.multi_start` 27ddc68, and the t1b control follow-up)

*2026-10-29. Adversarial, read-only. Fresh agent; did not write the code or the plan. Scope: `experiments/claim_tests/{run,job,stages,recipes,rules}.py`,
`cluster/claim_tests.sbatch`, `tests/test_claim_tests.py`, `predictions.json` → `amendment_1`, `src/commute/faithfulness.py: multi_start`,
against `reports/plans/2026-10-28_plan_claim-tests.md` (incl. Amendment 1). The t1b follow-up was reviewed as it stood uncommitted in the
working tree (`CONTROL_STARTS`, `stages.positive_control`, `control_budget`).*

In plain words: the code matches the plan closely. Every recipe matches its named source script, every ratio is taken against W0 at the
same start in the same job, and the filed rules match `predictions.json`. One defect will probably stop a job for no good reason: t2c checks
W0 against H100 numbers, and this card's seed-1 adversary already reads 11% above the H100 on the first rows, so t2c is likely to stop at
its first check. Two smaller gaps follow it.

## Findings, by severity

### 1. (Medium-high) t2c's C0 is likely to stop at `pgd_h1`: the rows 256-511 references are H100 numbers
- **Where.** `experiments/claim_tests/job.py:146-148` (`pgd_ref[256] = {"h0": 0.6298, "h1": 0.6576, ...}`), applied by `c0_checks` (`job.py:259-261`) with
  `pgd_tol = 0.10`.
- **Failure.** On rows 0-255 the L40 and H100 W0 readings are h0 0.6917 / 0.6988 (ratio 0.990), h1 **0.8362 / 0.7511 (ratio 1.113)**,
  h2 0.7049 / 0.7067 (`job.Config.l40_w0`; `broad_fragility/results/amend2.json` `.stage0.w0_pile`). The seed-1 harness start reads 11.3% higher
  on the L40, which is the cause of the L40's larger floor (0.209 against 0.075). If rows 256-511 behave the same, L40 h1 there is about
  0.6576 × 1.113 = 0.732, above the stop bound 0.6576 × 1.10 = 0.723. t2c then halts at C0 after about 6 minutes. With no t2c file,
  `run.py combine` leaves the T2 combination missing (`run.py:84`), so claim (ii)'s combined verdict cannot be scored.
- **Fix.** The check exists to catch a broken apparatus, not a card difference. On rows 256-511, check only h0 against its H100 reference
  (0.99 L40/H100 at h0 on rows 0-255), plus CI and f ≤ 0.25. Alternatively, scale the H100 references by the rows-0 L40/H100 ratio per start
  (h1 0.6576 × 1.113 ≈ 0.732). Record the change in the plan as a filing fix before any run. The same card mismatch makes the plan's
  expected f on these rows (0.044) unlikely; expect about 0.15. That changes no rule, but it should be in the plan's expectation.

### 2. (Low-medium) t1b never checks W0's budget KL, which is the denominator of its positive control
- **Where.** `job.py:263-265`: `budget_vs_h0` is added only when `"h0"` is among the W0 readings. With the follow-up, t1b reads W0 at
  `budget` (via `CONTROL_STARTS`) and `h2` (and `h1` only if h2 misses its L40 value), so the band check is skipped.
- **Failure.** The plan's C0 rule ("stop if the budget start's W0 KL is outside 0.85-1.15 of h0's") does not run in t1b. A bad budget W0
  reading would pass straight into `control_budget`'s ratio, and the positive-control gate would then pass or fail on it unchecked.
- **Fix.** In `c0_checks`, compare budget against the h0 KL when it was read, and otherwise against the L40 h0 reference
  `CFG.l40_w0["h0"]` (rows 0 only). A one-line change.

### 3. (Low) Resume keeps an old `"done": true`, so a rerun that halts can still be combined
- **Where.** `job.py:278-279` (`resume` copies `old` whole and pops only `"stopped"`); `run.py:68-71` (`load` accepts any file with
  `done`).
- **Failure.** Suppose a finished stage is rerun on the same card with the same W0 (for example, after a recipe fix) and then halts at a
  gate. Its file holds `done: true` and `stopped`, plus the old `score` and states. `combine` treats it as finished and scores the stale
  readings.
- **Fix.** Pop `"done"` and `"score"` in `resume`, and make `load` also refuse files with `"stopped"`.

## Checked and found correct (no action)
- **Recipes against their sources.**
  - The r1 SGD teacher matches `teacher_training/run.py Job.teacher` (ArmAConfig defaults: batch 32, clip 1, seed 0, 1000 steps, lr 0.03).
  - rand32 matches `newskill.make_teacher` (Adam 3e-5, checkpoints 10/300/1000, `kv_windows.task_context`). Its G1 is 0.5847 ± 0.02 at step 10.
  - The Adam teacher matches `amend6.ft_teacher`, and the LoRA teacher matches `amend3.lora_teacher` (rank 8, 1e-3, seed 0, checkpoints 100/1000).
  - Arm S matches `amend4.train_student_seeded` (TrainConfig momentum 0, batch 32, seed 0, `teacher_loss` with `train.half`).
  - T4 matches `remainder.py` (Adam 3e-5 to step k; `path.student` Adam 1e-5, 1000 steps; plus remainder = target W0 + dW with the copy's
    (U', V'); the G2 formula is identical).
  - T3 matches `adjoint_sgd/run.py finetune`, and keep = route-cell T1 P / the fine-tune's `p_o`, as E31's rows. The E31 constants match
    `adjoint_sgd_amend1/2.json` to three digits.
  - The lowercase task matches `broad_fragility/run.py` (train 0-4096 → 2,400 rows, held val 0-255 → 64, ArmB Adam 3e-5 batch 8 × 300,
    `copy_on_rows` 1e-5 × 8 × 1000 order 0). `lower_lora` matches `amend2.lora_delta` at lr 1e-4. Rows 256-511 use the same blocking as
    `rows2_context`.
- **Truncated runs.** `_run`, `_lora_run` and `component_training.train` have no learning-rate schedule, and their batch draws do not
  depend on `steps`. So step 10 of a 10-step run (t4b), step 100 of a 100-step run (the T5 LoRA cell at 0.03), and T3's two 0.03 passes
  are the prefixes they claim to be.
- **Ratios.** `multi_start` divides by `w0[s]`, W0 read in C0 of the same job on the same blocks. The W0 harness KL from
  `pile_faithfulness` equals `start_kl`'s `harness_pgd`, because `harness_pgd` re-seeds. `gamma.pgd_sources` rebuilds its CI cache on
  every call, so no state's KL reuses another state's cache. Its bf16 readout agrees with A4's fp32 readout to 0.1% (amend5 `part1`).
- **Rules.** `rules.py` matches each T1-T5 and `amendment_1` rule text: ties, early stops as final fails, unscored handling, T5's
  precedence, T2's cap clauses and T3's per-seed logic. `combine` pools f over both T1 jobs and uses each T5 job's own f, as filed.
- **Stop rules.** C0 halts after `resume` dumps. The T1/t1b control is budget ≥ fail line (after the follow-up), the T3 control is
  per seed, the T5 control is h0 ≥ fail line, and T2 does not stop and logs instead, as filed.
- **Writes.** `results/<stage>.json`, `results/large/<stage>.log`, `results/large/t2_lora_r{32,128}_tt.pt` (an atomic rename from a
  per-stage `.tmp`, so concurrent t2 and t2c jobs are safe), `results/combined.json`, and the sbatch's usual append to
  `cluster/logs/jobs.log`. The input paths exist, and the `results/large` symlink target exists.

## Not run
No code was executed (read-only review; no Python on the dev node). The CPU tests were not rerun. I did not check: t2c's actual L40 W0 on
rows 256-511 (finding 1 is an extrapolation from rows 0-255), GPU memory of the budget start's CI cache (about 7-13 GB per 128-row block,
which fits an L40), the LoRA scaling at rank 128, and whether the H100-measured gain gates (0.1301, 0.1479 ± 0.005) reproduce on the L40.

## Follow-up, 2026-10-29: Amendment 2, stage t4c (commit 2f3a7e5)

In plain words: t4c reads the right state. The job should not stop falsely, and its rule matches the filed text. I found no defect
that would make its reading wrong. One ordering point would save minutes if the reproduction check failed.

**Checked and correct.**
- *Same state.* `teacher_trained` is read inside `plus_remainder()` (`stages.py`), from the same in-memory `(dW_tt, uv2)` that
  plus remainder is then built from. Its target is W0 + dW_tt with the copy's (U', V'), and G2 already confirms that dW_tt = U'V'ᵀ − UVᵀ
  to 1e-3. That is `remainder.py`'s "teacher training" read (`read(job, label, dW_tt, st)`). It uses the same fp32 in-memory precision
  as t4b's plus remainder.
- *Reproduction against t4b.* `results/t4b.json` exists on the L40, with `done: true`, no stop, every gate passed, and KLs at h0, h1 and
  h2 for both states. So `reproduces` has a full reference and will not fail on a missing file or start.
- *The 1e-4 tolerance against real nondeterminism.* I checked the plan's evidence with jq, and every pair read in separate jobs agrees
  exactly:
  - T5's control at h0: 2.282897710800171 in t5 and t5b. That is a regenerated Adam fine-tune plus a 1,000-step SGD student, so training
    on this node is deterministic across jobs.
  - The teacher's P(target): 0.9946782302856445 in both.
  - r1's do-nothing at the budget start: 1.5364250785215212 in t1 and t1b.
  - lora8_tt at h0 and h1: identical in t2 and t2b.

  Plus remainder's 1,000-step Adam student was not itself repeated. Adam is deterministic given identical gradients, and the gradients
  are identical here. So 1e-4 is a fair stop line and not a false-stop risk. One caveat: the evidence comes from the single 8×L40 node,
  so the rule holds only there.
- *Rule.* `rules.t4c` matches `amendment_2.T4c` (between/above/below per start, ties counted as between, a common label or else mixed,
  unscored on any missing or non-finite start). `filed()` now reads `amendment_2`.
- *Resume.* A crash between the `teacher_trained` and `plus_remainder` reads makes the rerun retrain. It reuses the cached
  `teacher_trained` record, which is exact because training is deterministic, and then reads plus remainder. No state can be missing at
  scoring time.

**One ordering point (low; affects cost, not the reading).** `reproduce(job, "plus_remainder")` runs only after `teacher_trained` has
been read, which takes about 5.75 min. If the Adam student did not reproduce, that time would be spent before the stop. A free check
could come first, right after G3 and before reading `teacher_trained`: compare `G3_copy.p_target` with t4b's (0.9993861734050594 in
`t4b.json`) within 1e-4. Not needed for this submission.

**Outside the scope, noted only.** `t1.json`'s r1 do-nothing reads a budget KL of 1.536 (ratio about 2.2 over W0's ≈ 0.69), not the
plan's "≈ 9". It still clears the fail line, so T1's control passed. But the plan's expectation for the budget start on 256 rows looks
wrong by about 4x, and the lead may want to check how it was taken from amend5's block-0 reading.

**Not run.** No code was executed. I did not check t4c's measured times or its GPU memory.
