# Review 4 of commit 5c49414 (E40 Amendment 1, amended_3: B rate grid, step-25 abort, T0 extension). Read only, nothing run.

Scope: `git show 5c49414` (amend1.py, run.py `teacher_checks`/`stage_T`, qwen_teacher.py `kl_to_teacher`, sbatch,
check_teacher.py), against `predictions_amend1.json` (`rules.B_rate`, `rules.T0`, `amended_3`) and the plan's
"Gate 5, rate grid from the Fable advice". No crash on the paid path found. One mismatch between code and a filed
edge case, one selection-rule risk, the rest minor.

## Findings, by severity

1. **MEDIUM (code vs filed rule; confidence high on behaviour, low-moderate that it fires).** A non-finite value in
   any B arm stops the whole job instead of aborting that arm. `run.py` `teacher_checks.on_step` calls
   `job.finite(f"{name} step {step}", kl=curve[-1]["kl"])` every step and `job.finite(f"{name} fixed-row KL step
   {step}", ...)` after each reading; `Job.finite` → `check(stop=True)` → `sys.exit` on the pod. The filed
   `B_rate.edge_cases` say "NaN at 25: aborts" and "NaN or missing reading: does not qualify", i.e. the grid goes
   on to the next rate. The grid runs 3e-5 first, the arm most likely to blow up (Fable's lr/α argument puts it past
   the rehearsal's diverging 1e-4). Concrete failure: 3e-5 produces an inf/NaN training KL at step ~15 → job stops
   in B with 1e-5, 3e-6, 1e-6 untried; B's set-up (~$2-4) lost and a code change + review before a rerun. The unit
   tests only exercise `b_aborts` on a dict, so they cannot see this. Fix: in B, make a non-finite training KL or
   reading end the arm (e.g. `on_step` returns True on a non-finite `curve[-1]["kl"]`, and the finite checks
   report-only when `end_stops is None`), or re-file the edge case as "a NaN stops the job". Also note: an inf
   *reading* at 25 would hit `job.finite` (stop) before `b_aborts` gets to return True (amend1.py:360-362 calls
   `checks` first).

2. **MEDIUM (selection rule, wrong-reading/cost risk; confidence moderate).** `pick_rate` (amend1.py:218-222) picks
   the lowest step-100 fall with no margin. The only measured pair is a near tie at step 100 (rehearsal_2 B: 1e-5
   0.718, 3e-5 0.725) with crossing curves: Fable's advice says the higher rate reaches a higher floor sooner (3e-5
   flat from 100 to 216). A rate that wins by ~1% at step 100 can be the one that plateaus above 0.5 by 300-600, so
   T0 misses and stops the job after B+G1+T0, and T1 (1,000 steps per read step) trains at the worse rate. The rule is
   as filed and approved, so this is not a code bug; options are a margin (prefer the smaller rate unless the
   larger's fall is lower by, say, the step-50→100 drop's noise), or keep the third review's runner guard (stop and
   ask if the chosen fall at 100 > 0.65) and add "and the step-50→100 drop ≥ 0.05" from Fable's advice to the
   runner brief. Say which in the plan.

3. **MINOR (rehearsal wall time; confidence moderate).** The first rehearsal job (B,G1,T0,L,A0) was ~17 min with
   200 B steps and a 216-step T0 (rehearsal_2 run.log: ~82 s per 100-step arm at 16 columns). It now runs up to 400
   B steps and up to 600 T0 steps: ~+8-9 min at 16 columns, ~25 min against the sbatch's 30-min `--time`. At
   `--rehearse-cols 256-384` each step grows with the offsets, so that job likely hits the time limit in T0. Cost is
   a lost $0 job, but gate 3 needs it. Run B alone (or B,G1 / T0,L,A0) per job at large `--rehearse-cols`, and
   update the "~17 min" in the sbatch header.

4. **MINOR (memory; confidence high, size small).** In stage_B's loop (amend1.py:263-269) `_` keeps the previous
   arm's teacher (bf16, ~2.8 GB) alive on the GPU while the next arm builds its own; no `free()` between arms.
   ~2.8 GB over the ~72 GB estimate, well inside 125. `del _` / `free()` after each arm if the full-size peak comes
   in near the limit. Also `peak_gb` per arm is cumulative (no `reset_peak_memory_stats`), so per-arm peaks are not
   per-arm; reporting only.

5. **MINOR (rule text).** T0 `steps_rule` says past first_cap it "runs on, under the same rule, to steps_max";
   `t0_ends` applies the still-falling test only at first_cap, so a run flat from 350 still trains to 600. Matches
   the plan's "else at 600" and the test (`not end({...350: 0.7, 400: 0.69}, 400)`), so the code is consistent;
   the file's "same rule" reads ambiguously. Cost of the reading: up to 300 extra steps on a flat curve (counted in
   the projection).

6. **MINOR (stale path).** check_teacher.py now reads `amend1.rehearsal("amend1_rehearse")`. That directory is not
   the rehearsal check_teacher ran on (moved to `amend1_rehearse_1/`); it now holds an untracked, cancelled
   old-code B (one 1e-5 arm, out `e40a1_rehearse3`, 17:39). A rerun of check_teacher would read the wrong alpha.
   The plan keeps check_teacher frozen, so only matters if rerun; also a missing space after the comma (lint).

7. **MINOR (stale --out).** The results guard is on the results dir only. Reusing an --out basename whose results
   dir is new but whose --out holds files from a cancelled job (e.g. `e40a1_rehearse3/alive.safetensors` from the
   job in item 6) reuses those files. B rewrites `alive.safetensors`, and T's resume checks the δW hash, so no wrong
   number found; use a fresh --out per rehearsal as the header says.

## Checked and holding

- Off-by-one: `train` calls `on_step(step)` after the step-`step` update, steps 1..steps; reads (0,25,50,100) are
  taken by `checks` before `ends` is evaluated, so the abort reads step 25 after 25 updates and `t_steps` = 25.
- State between rates: `off.reset()` zeroes the offsets, `train` makes a fresh fused Adam per call, the teacher is
  recomputed (random_change reads the base V, U, not the offsets, so the same α and teacher each arm), weights are
  only swapped inside `set_weights`/hooks. No leak.
- `qualifies` reads only [0, 50, 100] (25 is for the abort alone), strict fall, r9 inclusive 0.9: as filed. Aborted
  arms lack 50/100 → do not qualify; all aborted → None → "B rate" stop (rehearsal: config 1e-5). Ties → smaller
  rate (tuple min). Float keys from `f"{lr:g}"` round-trip exactly.
- `b_aborts`: NaN/zero start aborts; exactly 1.2 continues; only at 25. `t0_ends`: NaN ends at first_cap; reading 250
  always present (read_every 50). T0 steps_max asserted equal to `t0_steps` (600) before loading.
- `kl_to_teacher` per row: maths rows and FineWeb rows are all 512 tokens (`maths_rows.json`, `tokens.json`), so the
  equal-length assert holds on the paid card and the row mean equals the old position mean; cost unchanged
  (mean_kl already looped rows).
- Review 3's fixes hold: B required before later stages (`before_loading`); projection counts T0 at `t0_steps`
  (now 600). Review 2's: fp32_tol 1e-2, T0 sized at target_kl, only JSONs/log uploaded.
- Rehearsal layout: `amend1_rehearse_<basename of --out>`, `alive_max` from `--rehearse-cols`; ignored without
  `--rehearse`.

## Not run
Nothing executed (no Python on this node; no GPU). Not checked: full-size step time and peak at the full alive set,
whether 3e-5 actually produces non-finite values at full size, the four-arm B's wall time on the H200.
