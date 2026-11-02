# Review of E67's code before its first GPU run (commit 2d3e90b)

I read the weight-side read's code, filing, job script and plan against the saved files, and found no bug in the arithmetic or the filed verdict. The main risk is meaning: the learned-nothing fine-tune is read on rows that are almost all unlike the rows where the route split's r failed that null, so a keep would not show what the plan's title asks. About 40% of the code feeds numbers that no filed rule reads, and the largest cut is a choice for the lead or the user.

Nothing ran on a GPU and the model was not loaded. Sizes below come from the saved `.pt` files and JSON on CPU. Scripts and outputs are in `~/commute-data/scratch/regating-lead/review-e67/`.

## Findings, by severity

### 1. High (meaning): the null is read on text it was not failed on

`experiments/weight_side_read/run.py:185`, `reports/plans/2026-11-02_plan_e67-weight-side-read.md:1,11,82`

**Scenario.** The plan asks whether the share "passes the learned-nothing null" that r failed. r failed on `code_held.pt`, 64 held-out rows at or above E36's code-share cut (`regating_table/run.py:436-438`, `anchors.py:112-115`). E67 reads val rows 0 to 127, unfiltered. I counted them with E36's own rule: **3 of 128 rows** reach the cut of 0.305, and the mean share is 0.082. So 125 of 128 rows are text the code fine-tune did not train on and r was not read on. The plan's line 82, "read closer to its own training text", does not hold.

The copy fine-tunes are off-task on these rows too, so the verdict compares two off-task updates. S_R(code) ≤ 0.1 with S_R(copy) ≥ 0.4 would be filed as `keep` and would propose a column for E62 and E58. It would show only that the two kinds of update differ on ordinary text, not that the read passes r's null. A `drop` has the mirror gap: the code reading on its own kind of text is unknown.

**Smallest fix.** The `code_held` tensor is already (64, 512), so it fits `batches`. Add one read beside the verdict: code step 10 and ident32 s0 on `batches(code_held, 16)` (sha in `stage0a.json`, `groups["e36/code"].held_sha256`), about 5 lines and under a minute of GPU. Correct line 82, and state in the keep row that a keep is about Pile text unless the held-rows reading agrees. Whether that reading should decide rather than sit beside the verdict is the lead's call, and it must be filed before the run.

### 2. Medium: "row noise is not in the comparison" is wrong, and the code side has no floor

`reports/plans/2026-11-02_plan_e67-weight-side-read.md:71`, `src/commute/weight_side.py:145-151`

**Scenario.** Reading the same rows removes only the row noise the two updates share. S_R is a ratio of sums over 65,536 positions, and how far a few rows drive it is unmeasured, and it can differ per update. A code reading of 0.08 could be 0.20 on another 128 rows, which moves the label from `keep` to `between`. The `between` row says "the lead puts the numbers beside the noise floor", yet the plan admits the code side has none. The repo rule requires each number to have a noise floor.

**Smallest fix.** Save each chunk's pooled R numerator and denominator in `run.json`, about 3 lines in `read_state`, with no extra GPU. A leave-one-chunk-out spread over the 8 chunks then gives both sides a floor, and the verdict rule stays as filed.

### 3. Medium: no control runs Γ on a moved model, and a read that went dead lands on the lean

`experiments/weight_side_read/run.py:93-129,132-140`

**Scenario.** Stage zero uses δW = 0. Stage ones forces every gate to 1 and has no g1. So the real model's g1 at W₁ ≠ W₀ first runs inside the first scored state. The shares' sum cannot catch a wrong g1, because it holds for any g, a or offset. If g1 equalled g0 on the real model, every S_R would read 0 and the filed verdict would be `drop` with `drop_both_small`, the expected answer. The tiny-model test (`tests/test_weight_side.py:194-200`) makes this unlikely, and `model_state` has been used elsewhere. It is still the one failure the job cannot see. No saved state is a known re-gating update, so the positive control stays unit-test only, as the plan says.

**Smallest fix.** One line in `stage_state`: stop if `rec["mean_abs_dg"] == 0`. State in any drop report that the read had no positive control on real rows.

### 4. Low: `seeds_disagree` fires on a negligible spread at a bar

`src/commute/weight_side.py:208-212`, plan line 71

**Scenario.** Code 0.05 with ident32 at 0.401 and 0.399 gives keep against between, so `seeds_disagree`, "the spread is the finding", on a spread of 0.002. The plan's claim that "a spread inside one label does not stop the verdict" says nothing about a tiny spread across a bar. The harm is small, since neither outcome adopts the read, but the next step is wrong.

**Smallest fix.** Before running, file that differing labels with a spread under the drop band read as `between`.

### 5. Low: the size control has an open false stop, and its tolerance rests on a wrong reason

`experiments/weight_side_read/run.py:87`, `predictions.json` `size_ratio_tol_source`

- **rand32b_s1.** The plan leaves it open: it reads 0.2% off on the matrix checked, and that is only "presumably" because its largest matrix is another. If not, the job stops in its first minute. That is cheap, but it is a round trip for a check on students the read never uses.
- **The ratio tolerance.** It assumes the table's ratios are the exact ratios rounded to two decimals. They were divided by teacher sizes already rounded to three figures (`experiments/new_vs_existing/plot_amend1.py:30,154`, the same values as the table), so the true slack is about ±0.010, not ±0.006. With `|rebuilt/filed − 1|` assumed small, I computed each teacher's size from its saved file. The worst miss is ident32b_s1 at 0.0043, so it passes by 0.0017. Against E12's filed teacher size it would have read 0.0060 and stopped.

**Smallest fix.** Delete the stage (cut 2 below).

### 6. Low: a 30-minute limit on an estimate that is arithmetic, with no resume

`cluster/weight_side_read.sbatch:18`, `experiments/weight_side_read/run.py:172-173`

**Scenario.** The estimate of 10 to 20 min leaves out model load, six sha256 reads, 16 file loads in the size stage, and the bases' full SVDs in float64 on 4 CPUs (eight matrices are 3072 wide). A kill at 30 min after rand32 loses only the Gaussian and the verdict's line, but `run.json` then blocks a rerun, and rerunning repeats everything.

**Smallest fix.** `--time=00:45:00`. The `debug` QoS allows 2 h (`cluster/README.md:34`).

### 7. Low: the loud line misses two verdicts against the lean

`experiments/weight_side_read/run.py:161`. Plan line 103 says a verdict against the lean prints a loud line, and the lean is drop. `keep_at_step10_only` and `between` print none. **Fix:** `if v["verdict"] != "drop"`.

### 8. Low: the record does not say whether the code was committed

`experiments/weight_side_read/run.py:186`. `git_head` is recorded, but the check at line 170 covers only `predictions.json`. The clone is shared between sessions, so a run with an edited `weight_side.py` would carry a clean-looking head. **Fix:** add `git status --porcelain -- <the two code paths>` to `out`.

## What could be cut

The 421 lines are about 280 lines more than quoted. The plan's "the excess is the controls" is right, and most of those controls guard numbers no filed rule reads.

**Cut 1, the largest; it needs the lead or the user, since the plan's request names it.** The verdict reads the do-nothing S_R only, plus the Gaussian's. S_D, S_I and the split's S_E have guessed expected values and no rule, and `.claude/CLAUDE.md` says "a number without an expected value is cut", and that an experiment reports the numbers its predictions use. Reading do-nothing only removes the following:

- `Offsets`, `offsets_from`, `changed`, `split_identity`, `rel_norm_max`, the `split` dict and `whole` in `Matrix`
- `stage_size`, `stage_ones`, and `Job.uv`, `alive`, `bases`, `students`, `filed`, `least_squares`, `offsets`

The result is about 250 lines of code (`weight_side.py` about 135, `run.py` about 115), and the tests fall by more than half. Zero then stands as the one apparatus control. Recommendation: run do-nothing first and add the split side only after a keep, because a drop, the lean, never reads it. The user's definition of re-gating and S_R survive whole. Also, the plan's term list was approved in a dialog, so this one should be raised.

**If the split side stays, these cuts need no one's approval:**

| # | Cut | Why it is safe | Lines |
|---|---|---|---|
| 2 | `stage_size`, `rel_norm_max`, `size_ratio_table`, two thresholds, `Job.filed` | Checks saved students no state reads. It is the source of finding 5 | −15, tests −10 |
| 3 | The students' half of `stage_ones`, plus `split_identity` and its stage lines. Replace with one check on ident32 s0's split state: `changed(a, ones, U, V, off)` against `a @ (U2 V2ᵀ − U Vᵀ)ᵀ`, overlap share 1 | The same `changed` is tested on real a. `split_identity` tests `split.py`, which is long in use and which this experiment does not own | −30, tests −25 |
| 4 | `Matrix.nothing`. Derive it from `split["R"]` and `den`: D = I = 0, E = den − R | The do-nothing sum check at `run.py:137-139` is empty, since (Ma − R)·Ma + R·Ma = Ma·Ma | −5 |
| 5 | `second_moments`. Use `split.input_second_moments(model, chunks, keys)` | Both energies in the Gaussian's scale use the same Σ, so bf16 rounding in Σ moves nothing a rule reads. `.claude/CLAUDE.md`: no copying | −11, tests −3 |
| 6 | `isfinite`, the None and NaN handling in `verdict`, and three parametrize rows | `stage_state` stops on a non-finite sum before the verdict, and a None would raise there first, so these branches never run in the job | −4, tests −3 |
| 7 | `rand32_s0`, `rand32_s1` | No rule reads them. They give one printed spread and take about 3 min | −3 |
| 8 | The split-side sum check and `whole`, taking S_E by subtraction | The plan rejected subtraction as making the check empty, but the check is already blind to any wrong g, a or offset. It only tests three lines of algebra that the unit tests already pin | −6 |

Cuts 2 to 8 together give about 350 lines and about 40 fewer test lines. Changes to controls and thresholds mean refiling `predictions.json`, which is allowed since nothing has run.

The test file at 262 lines is longer than either code file. `tests/test_weight_side.py:219-262`, the fake-format fixture, shrinks with cuts 1 to 3. I checked the fixture's formats against the real files, and they match.

## Checked and found fine

- **Tests and lint:** `tests/test_weight_side.py` 31 passed after the merge; ruff and pyright clean on the three files.
- **Algebra:** the four terms, `changed`, `split_identity`'s derivation, the matrix layouts, and pooling as filed.
- **Filed verdict:** `pair_verdict` and `verdict` match the filing branch by branch.
- **Inputs on disk:** all six sha256 values match `stage0a.json`. Fine-tunes hold 24 fp32 matrices with no zero or non-finite ones. Student files have the filed format, `step` 1000 and matching alive shapes. `val_0_256.npy` is cached, and `results/large` resolves.
- **Saved teachers:** seven of eight match E12's filed `rel_norm_max` to 8 digits. ident32b_s1 is 0.13% smaller (T1 4.041 against 4.049): E46 re-ran it. It enters only the size stage.
- **Precision:** no autocast in `param_decomp/models` or the run. `lower_leaky` with `continuous` is a deterministic clamp, as the plan says. Other runs' zero updates read exactly 0, so the exact-zero control is sound.
- **Memory:** the plan's 2 GB per chunk is right (1.3 GB g, 0.9 GB a), well inside an L40.
- **Job script:** exit code through `pipefail`, `link-data` arguments, `SLURM_JOB_QOS` as in other jobs.
- **Rows:** E36 trained on `train` rows, so the read rows do not overlap its training rows.

Not checked: the real identity miss and sum on real activations, the real Γ pass time, and rand32b_s1's size, all of which need the model. The job's first minutes print them.
