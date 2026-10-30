# Review: E55 C3 step ladder code (commit db5618e, merged in ddfde28)

2026-10-31 (written 2026-10-30 evening, while job 46954 ran). An independent code review by an agent that did not write
the code, the plan or the analysis. It covers `experiments/open_pair_instruments/ladder.py`, the `revival_kv.py`
additions (`gap_word`, `pole_gap`, `first_ok_step`), `tests/test_revival_kv_ladder.py`, the sbatch change and
`results/predictions_ladder.json`, read against `reports/plans/2026-10-31_plan_e55-c3-ladder.md` and C3's code
(`instruments.py`, `measure.py`, `revival_kv.reverse_curve`, `finetune._run`).

**Plain words.** I read the ladder code line by line against C3's code and the filed rules, and checked the parts the
running job had already logged. I found no bug that gives a wrong scored number or a crash. Part A's re-read and Part
B's reverse loop reproduce C3's numbers. The findings are minor: one way a failed control could be skipped on a
manual rerun, a control order that differs a little from the plan, and a reported column whose sign the plan got wrong.

## Verdict

- **Wrong-number findings: none.** **Crash findings: none.**
- **Part A can be trusted as coded.** It reproduced every C3 number it was checked against. All 96 checks were true:
  D, and the shares at K 13 and 40, for each of the 32 states, in `results/ladder.json`.
- **Part B can be trusted as coded.** Before I wrote this, the log showed three step-10 controls passing on every
  item: rank-1 seed 0, elicit seed 0 and rank-1 seed 1. On each, the state matched E58's to 1e-5, the two shares
  matched C3's, and t½ matched C3's. For rank-1 seed 0, t½ was 13.910132881927666, identical to C3's `run.json` to
  every printed digit. Under the reverse loop in the review, it
  explains why that match also covers the steps after 10.
- The CPU tests pass (16 passed, run through `fleet/cpu-run`).

## Findings

### 1. A failed control is stored, so a later rerun skips it (minor; a wrong number only after a manual rerun)

`ladder.py:205-210` (Part A) and `ladder.py:254-259` (Part B) write the state into `st` and `dump` it before testing
`checks`. `stop()` then exits with code 1. `stage_ladder` (`ladder.py:277-280`) reloads the old `ladder.json` and does not
refuse a file that holds `"stopped"`.

*Scenario.* Someone resubmits after an exit 1, for example a runner treating any non-zero exit as "resubmit". The run
then skips the failed state, because `n in st`, and carries on. The failed state's readings go into the scores, and the
file keeps its old `"stopped"` key even after a clean finish.

*Fix.* In `stage_ladder`, refuse to start when `old.get("stopped")` is set (`out_of_order`). Alternatively, store `st[n]`
only after the checks pass. The sbatch comment says to resubmit only on exit 3, so this is minor.

### 2. Elicit's step-10 control runs after all four cipher seed-0 states (minor)

`ladder.py:236-237` loops over seeds outside and maps inside. The order is therefore rank-1 s0 (4 steps), cipher s0
(4 steps), elicit s0 (4 steps), and so on. The plan says Part B's controls run first: rank-1 s0, then elicit "the same
way". The cipher s0 readings therefore come before elicit's control. This costs seconds on a $0 run, so it does not
matter here.

*Fix, if reused.* Run the rank-1 and elicit seed-0 fine-tunes first.

### 3. The cipher has no reproduction control in Part B (minor, by design)

`ladder.py:252` skips `step10_checks` for the cipher. This is acceptable, for two reasons:
- E58 saved the cipher at the norm-matched step (8-9), not at step 10, so there is nothing to compare against.
- `keydep.finetune` with E12's checkpoints follows the same trajectory as E58's checkpoints (1..30, 100, 300). The batch
  generator is a separate `torch.Generator`, and `eval_arm_a` does not draw from it.

The rank-1 and elicit checks cover the shared recipe.

### 4. The top-guess column has the opposite sign to the plan's (reported only; an interpretation point)

The plan and `predictions_ladder.json` say: "rank-1 raises that guess, so its D_top is negative". The log reads
**D_top = +7.72** for rank-1 seed 0 at step 10, and positive at every logged step. The update lowers W₀'s top guess.
On every logged state, t_top is within about 0.6 steps of t½. So the column tracks κ (KL from W₀'s answer
distribution) closely, and it gives little evidence as the "non-sharpening target" that the advice wanted.

This is not a code bug: the code computes `lp0[a] − lp[a]` as its docstring says. The analysis should state the
sign as measured.

### 5. t_top's pole gap drops censored seeds without saying so (minor, reported only)

`ladder.py:174`: `num()` turns `"censored"` (and `"void"`) into None. `pole_gap` then scores t_top on the uncensored
seeds, with no censored flag. In the same situation t½ reads "censored". This could bias a reported gap if t_top
censors on one pole.

*Fix.* Flag t_top the way t½ is flagged (`t_half_column`), or report the number censored per map.

### 6. The per-K R gaps use a noise floor taken over all K (minor, reported only)

`ladder.py:164-165`: `R_per_K` uses `floor_R`, the largest random-removal spread over every K, not that K's own spread.
It is the same floor that the scored R (max over K) uses, so it is conservative. Do not read the per-K statuses
as filed. In Part A, K 320 reads "ok" (gap 0.191 against a bar of 0.189) while the filed R reads "no gap" (0.146
against 0.152). Only the latter is PK.

### 7. Small points with no effect on this run

- `ladder.py:55`: `git diff --quiet HEAD -- <file>` returns 0 for an *untracked* file. `common.load_predictions` also
  checks `git ls-files`. The file is committed now, so this has no effect. Add the `ls-files` check if the code is
  reused.
- `ladder.py:202`: Part A forces `"void": False`. All 32 C3 states have D ≥ 4.37, far above D_void = 0.978, so this has
  no effect.
- After an exit 3 in the middle of a map, the missing steps come from a fresh fine-tune, while the finished steps come
  from the earlier job. The retrain matches E58 to 1e-5, so the mix is harmless.
- `common.dump` writes in place, not atomically. A Slurm kill during a write would leave a truncated `ladder.json` that
  the resume cannot parse. The 25-minute deadline sits 5 minutes inside the 30-minute limit, and `reverse_two` checks the
  deadline every step, so the risk is low.

## Paths checked and found right

**The `gap_status` refactor.** The logic moved into `gap_word` unchanged: gap 0 gives "no gap", the undirected case uses
|gap|, and the directed case gives ok, reversed or no gap. The noise in rule 1 still comes from the three scored maps.
`test_gap_status_unchanged` and `test_elicit_spread_is_ignored` pin this down. C3's verdicts are unaffected.

**`pole_gap` and `first_ok_step`.** They match the filed rule text: noise is the larger of the two poles' sd and the
floor; the bar is 3 × noise; a pole with fewer than 2 values is void. The pass step is the smallest step reading "ok",
with steps sorted as integers.

**`reverse_two` against C3's `reverse_curve`** (`ladder.py:73-109` against `revival_kv.py:145-170`):
- *Learning rate and recipe.* `reverse_cfg` takes C3's `reverse` block and swaps only `cap_steps`. `_run` has no
  schedule: `steps` is only the loop bound. So raising the cap from 1000 to 3000 changes no step's learning rate.
- *Reference.* Both use `KV.w0_copy(net)`, taken while the weights are at W₀ (after Part A, whose `net.at` contexts
  restore them).
- *Seed.* Both use the state's seed.
- *Reading.* `read()` computes `kl_rows(lp0["score"], answer_lp(net, score, half)).mean()`. This is `kappa(net, ref)`
  term for term, read inside `net.at(dW)` exactly as C3 does.
- *Indexing.* Steps are 1-based (`range(1, steps+1)`), and `curve[i]` is the reading after step i. So the `k_half or
  …` sentinel at 0 is safe.
- *Stop and t½.* t½ is `half_life(curve)`, the first crossing. Reading on past the crossing, while waiting for the
  top-guess curve, cannot move t½.
- *Censoring.* A state is censored only if κ never halves within the 3000-step cap. If 3 × k_half exceeds the cap, `end`
  reads "cap" but t½ is still a valid number. This is correct.
- *Interpolation.* The same `half_life` function.
- *Deadline.* A cut gives None, and the state reruns.
- *Weights afterwards.* `_run`'s `finally` puts back the start weights (W₀ + ΔW), `net.at` puts back W₀, and
  `at_w0` checks the result.

**Checkpoint step and train/score halves.** `_run` saves `p − W0` after optimizer step t, so `ft.deltas[t]` is the
step-t state; the 1e-5 state checks confirm it. Unit scores use the fit half, κ the score half, and the reverse
fine-tune trains on the rank-1 file's train windows. All three are C3's choices.

**Seeds and steps are not mixed.** The record names are `{m}_s{s}_t{t}`. The scoring takes each step's three seeds per
map, and the reverse seed is the state's seed.

**R's floor.** It is computed per step from that step's six pole states over every K, and in Part A from the 16 pole
states. This matches `constants.R_floor`.

**Resume and file writes.**
- Part A is skipped once `score_kreread` exists.
- In Part B, finished states are skipped. A map with steps still to do re-runs its fine-tune, and the scores are
  recomputed when all states are done.
- The sbatch has `set -o pipefail`, so exit 3 passes through `tee`.
- `ladder.json` is written after every state.

**Predictions against code.**
- gap_mult 3, `t_half_floor` 1, prune_k [13, 40, 80, 160, 320], cap 3000 and D_void are read from the filed file.
- PK is the scored `score_kreread.R`.
- PK_trend can be read from `R_per_K` gaps 320 and 40.
- PR, Pt and Pt_reversed_10 are `per_step[t]` statuses, and Ppass is `pass_step`.

## Not run

- I did not re-run any GPU path. Part B's correctness rests on the controls the running job logged, plus reading the code.
- I did not check that E58's `windows_*_trained.pt` files for the cipher hold the same `table`, `train` and `heldout`
  as E58 used. E58's own sha256 gate covered that in C3's jobs; this stage does not recheck it.
- I did not run lint.
