# E55 Amendment 2: code review

Scope: commit eb66b56 (`experiments/pcfg_anchors/amend2.py`, `results/predictions_amend2.json`, the `amend2` stage of
`cluster/pcfg_anchors.sbatch`), read against `amend1.py` and `run.py`. Reviewer did not write the code. Model-free
checks ran through cpu-run in `~/commute-data/scratch/reuse-lead/e55-a2-review/` (`check.py`, `cells.py`), on the
fresh rows and on amend1's saved answers. No model loaded.

Summary: the code does what the filing says, and nothing in it would make a verdict wrong. The problem is what the
test can show. The fresh rows are new texts, but almost all of them fall in (count a, count b) cells that amend1's
rows already covered, and the fine-tunes' answers are fixed by the cell. So T1 is close to a replay of the rows the
rule was found on. Amend the filing's wording before the run (finding 1); nothing else blocks.

## Findings

### 1. Fix before run: the fresh rows repeat amend1's cells, so T1 is close to a replay of the rows the rule was found on
`amend2.py:63-64`; `predictions_amend2.json` `filed`, `why`, `T1_W1_10K_third.lean`.

- **Cells repeat.** third's 219 distinct fresh rows fall in 12 (a, b) cells. 217 of them are in the 10 cells that
  held amend1's 92 distinct rows. Four cells hold 190 of the 219: (11,16) 73, (16,11) 58, (10,15) 33, (15,10) 26.
  Over all 20,000 fresh rows, only 19 are in a cell that amend1's 8,000 rows did not cover.
- **Answers are fixed by the cell.** On amend1's rows, the most common answer in each (a, b) cell matches W1 10K on
  98.4% of rows and W1s1 10K on 98.6%. W1 10K answered third on 92 of the 92 distinct rows, and every one of the 10
  cells holds.
- **Replay.** Score the fresh rows by giving each one W1's most common amend1 answer for its cell. That gives
  S_third 0.991 and overall_third 0.979, before any model runs. Both pass T1.
- **Failure scenario.** T1 passes and is reported as "confirmed on rows never looked at". The run only tests whether
  the answer changes with the tokens outside the window inside cells already seen. It does not test whether third
  extends to new (b - a, a) combinations, which is the overfitting worry the lean itself raises ("the tails ... seen
  only 92 times").
- **Fix (one of):**
  - (a) Add a dated note to the filing, before the run: this is a within-cell replication on new texts, the cell
    replay predicts 0.991 / 0.979, and the test is blind to unseen cells. Also say so in the blind_to.
  - (b) Add a read on rows in unseen cells, built like `spurious_texts` by swapping a/c inside the window of natural
    texts (for example b - a in {±6, ±7, ±8} at other a). Label it off-distribution and keep it apart from T1.

  (a) is the minimum. (b) is the only version that tests the rule outside the cells it was fitted on.

### 2. Minor: the noise floor quoted for S_third assumes independent rows; the effective n is about 12 cells
`predictions_amend2.json` `T1.noise_floor` ("0.009 for S_third at 0.98 on 219 rows").

- **Failure scenario.** Answers are about 98% fixed by the cell. If the state answers something else on one large
  cell, S_third drops by up to 0.33 (cell (11,16)) and T1 misses. That reads as a rule failure, but it is one cell.
- **Fix.** In the analysis, report S_third per cell next to the pooled share. Don't quote the binomial 0.009.

### 3. Minor: the code scores no prediction; T1-T4 are applied by hand
`amend2.py:168-182`; `all_row_min` is never read.

- **What happens.** amend2.json holds the reads (labels, routes, overall shares, gaps), but no T verdicts. The
  thresholds overall_third >= 0.95 (T1), < 0.95 (T2, T4) and "route a_plus_1" (T3) are left to the analyst. This is
  the same as amend1, so no change is needed for this run.
- **Failure scenario.** A hand-scoring slip, or the T2/T4 "< 0.95" being read as "<= 0.95".
- **Fix.** The analysis lists each verdict next to the JSON field it reads.

### 4. Minor: the c-read note gives the wrong number of c groups
`predictions_amend2.json` `definitions.c_read` ("~20 per-c answers").

- On the fresh rows count c takes only 10 values (9..18), so the in-sample optimism of c_best is even smaller than
  filed. This is conservative, so no verdict changes. Correct the text in the analysis.

### 5. Minor: models load before the model-free controls run
`amend2.py:177`. `a1_models()` runs before `formula_check` and `synthetic_controls`.

- The cost is about 2 s, and a miss still stops before any W state is read. Not worth changing.

## Checked fine

- **third formula** (`amend2.py:71`). a + 1 + floor((d + 1) / 3) equals rint(a + 1 + d / 3) in all three residues of
  d mod 3. Python `//` floors correctly for negative d. No .5 ties exist, so numpy's banker's rounding never applies.
  `third_is_rint` is True on the fresh rows (run). d = b - a - 1 comes from amend1's `rule_answers`. third = b on
  spurious rows (True).
- **Rows are new texts.** Seed [1, 72] is used nowhere else in `experiments/` or `src/`. amend1 used [1, 70] for its
  rows and [1, 71] for its formula check. The spurious check uses [1, 72, 2]. 0 of the 20,000 texts are among
  amend1's 8,000 (run). The overlap is in cells, not texts (finding 1).
- **Pool sizes match the filing.** a_plus_1 9,547, b 11,974, mid 1,342, a 8,397, third 219, all >= min_rows 100.
  amend1's rows gave third 92, which is why its in-sample reads are nan and route "mixed" for W1 10K (reproduced).
- **Bars.** `bar_read` is reused unchanged. Pools where a constant scores high (mid f 0.71, hi 0.856; third vs mid
  f 0.66, hi 0.83) keep hi = f + 0.5 (1 - f), so a constant can't reach "follows".
- **Synthetic controls.** All 16 checks are True on the fresh rows (run). c_floor, c_ceil and towards_b read route
  mid, c_best 1.0 (towards_b 0.857), mid c read "not shown", as filed in E13.
- **c_best.** It is the exact per-c modal share, in-sample, and an upper bound for any rule of c alone, so "beats" is
  conservative. Every group is non-empty. A margin of 0.05 is well above the sampling noise. In-sample gaps
  (reproduced): W1 10K third 0.213, W2 10K mid 0.102.
- **Controls run first and stop the job.** N1 10K (b) and LOW (a) are read before any W state. A miss writes
  `amend2_control_miss.json`, exits 1, and leaves amend2.json unwritten.
- **States load from the right place.** Saved W states come from `results/large/a2/` (`state_path` default
  `A2_DIR`). The trajectory comes from `results/large/amend1/` (`TRAJ`) at traj_saves 1K-10K. All 12 trajectory
  files, the 6 saved W files and N1 10K are on disk.
- **Time.** amend1 read 8,000 rows in about 1 s per state, so 20 states at 20,000 rows take under 2 min. The
  `--time=00:10:00` on the command line overrides the header's 30 min, and amend2 never exits 3, so it never
  resubmits. Every scored number is logged, so a kill loses only the JSON. `run_at_art` works in chunks of 1,000.
- **Lint and tests.** `make lint` on amend2.py is clean. `make test` gave 410 passed and 2 failed; both failures are
  in `tests/test_qwen_maps.py`, unrelated to this commit.

## Not run

- No model was run on the fresh rows; the replay in finding 1 uses amend1's saved answers only.
- No rows were built in unseen cells (finding 1, fix b).
- The plan or analysis text was not reviewed, only the code and the filing.
