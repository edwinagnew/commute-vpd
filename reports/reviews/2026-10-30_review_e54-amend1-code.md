# Review: E54 Amendment 1 code (commit 29bc903)

Independent code review of `experiments/keydependent_elicit/amend1.py`, `results/predictions_amend1.json`,
`cluster/keydependent_elicit_amend1.sbatch`, the `src/commute/steering.py` refactor, `elicit.score_amend1` /
`branch_amend1` and the new tests, against the plan's section "Amendment 1 (2026-10-30)". The reviewer wrote none of it.
Read-only, plus `tests/test_elicit.py` run through `cpu-run` (15 passed), ruff and pyright on the touched files (clean),
and `sha256sum` of the six saved weight changes (all six match `stage2.json`'s `finetunes[*].sha256`).

**Verdict: no blocking bug.** The code measures what the plan and the predictions file say: S_lo is a share of a
log-odds gain built with `identity.log_odds`, the function `route_split`'s T cells use, at the same position and
target. The refactor of `b3` is operation-for-operation the same, and Stage 0 tests it against numbers made by the
pre-refactor code. Four low-severity findings follow; none changes a verdict on the expected values.

## Findings, by severity

### 1. (low-medium) The log-odds readout is never checked on the GPU before the scored numbers are read
`experiments/keydependent_elicit/amend1.py:72-78, 109-113`; `predictions_amend1.json` → `reported_not_scored.G_lo_all_vs_route`.
Stage 0's positive control (S_path) runs `steer_at` with `logp_at`; the predictions file itself says "only the readout
is left untested by it". The one runtime check of `logodds_at` is G_lo_all against route_split's G_logodds, which is
printed but is not a stop. Failure scenario: a readout bug that the CPU unit test misses (e.g. a device or dtype path
only taken on CUDA, or a future edit to `logodds_at`) gives wrong S_lo values, the run scores P1-lo/P2-lo on them and
the branch reads "as predicted". Risk is small today: `logodds_at` is a two-line wrapper around `identity.log_odds`, which
`route_split` already uses on GPU, and `test_logodds_readout` covers the near-P=1 case. Suggested fix: no code change
needed if the runner's brief says to compare G_lo_all with G_logodds per state before the scored lines are reported
(they are the same quantity through two forward paths; a relative difference above ~1e-3 means stop, which is far
above any rounding floor and far below a readout bug's size). Or make that a stop in code with that loose bound.

### 2. (low) P3-own scores stage2.json's S, not the S stage 0 recomputes
`amend1.py:77` (`"S": ref["S"]`) feeding `elicit.py:179` (`col("elicit", "S")`, `col("cipher_own", "S")`).
`predictions_amend1.json` → `settings.S` says "S in nats as stage 0 recomputes it (equal to stage2.json states[*].S
within the stage-0 tolerance)". The code scores the reference values. Because stage 0 forces |S_b3 − S_ref| ≤ 1e-6
before scoring, the verdict can differ only if the gap or the range sits within ~1e-6 of a bound (gap 0.296 vs
bound 0.2 and range 0.048: no). A wording mismatch, not a wrong number. Fix: either put `out["stage0"][n]["S_b3"]` in
the state record, or note in the analysis that P3-own read stage2.json's S, as the plan's P3-own text already implies.

### 3. (low) P2-lo's miss message covers S_lo above r with the wrong words
`elicit.py:195-196`. When d = r − S_lo < 0 (steering's log-odds share above the re-gating share on an elicit state),
the branch appends "an elicit S_lo within 0.3 of its r". That case is a qualitative reversal, not "within 0.3", and
the project's rule is to flag reversals loudly. Scenario: S_lo = 0.9 on elicit_s0 (as in `test_amend1_p2_bound`'s
third case) reads as a near miss in the branch string. Fix: report "S_lo above r on <state>: reversed, tell the user"
when any d < 0; the status (fail) is already right and the branch already says "tell the user".

### 4. (low) Nothing records which code ran
`amend1.py:54-58`, `cluster/keydependent_elicit_amend1.sbatch:28`. `check_filed` guards only the predictions file. The
clone is shared by concurrent sessions; an uncommitted edit to `src/commute/steering.py` or `elicit.py` in the working
tree at submit time would run unreviewed code with no trace in `amend1.json`. Stage 0 would catch an edit that moves
`b3`'s S, but not one to `logodds_at` or `score_amend1`. Fix: write `git rev-parse HEAD` and
`git status --porcelain -- src experiments/keydependent_elicit` into `amend1.json` (one line in `main`), or have the
runner check `git status` before `sbatch`.

## Checked, fine

- **What S_lo measures.** `steer_at` (`steering.py:158-170`): base and tuned read on all 97 held-out windows and sliced
  to the scoring half (odd indices); steered read on `t.toks[sc]` with B3's block-`layer` vector added at `t.half`;
  rise and gain both differenced against the same W0 read; share = mean(steered − base) / mean(tuned − base).
  This is the predictions file's `settings.S_lo` exactly. `logodds_at` = `identity.log_odds(logits[:, pos], tgt)`
  = logit_target − logsumexp(other logits), the T cells' unit in `local_faith.route_cells` (line ~217), same position
  (`t.half`) and target (`t.table[toks[:, half]]`). nan on a gain within 1e-6 of 0 is applied by
  `EL.share_or_nan(lo["share"], lo["G"])` with the gain `steer_at` divided by.
- **Stage 0 is not vacuous.** `stage2.json` was committed at 12:02 (9cb9c52) by the pre-refactor `steering.py`; the
  refactor is 29bc903 (12:29). So |S_b3 − S_ref| ≤ 1e-6 on six states tests the refactored `b3` against the old code's
  output, and S_path tests `steer_at`'s slicing and wiring. A nan anywhere makes `abs(nan − x) <= tol` False, so it
  fails, not passes; a missing key raises. sha256, layer and both S checks are ANDed per state and all six must pass
  before any log-odds reading. The 1e-6 tolerance is sound: one node, eight L40s (`cluster/README.md`), same batch
  shapes as stage 2, TF32 off as in `run.py:342`; S_path and S_b3 run the same forwards on the same tensors, so they
  should agree bitwise, and a wrong block, half or vector moves S by ≥ 1e-2. Stage 0 cannot see the readout (finding 1).
- **Saved inputs.** All six `{elicit,cipher,cipher_own}_s{0,1}.pt` exist (`cipher_*` `same_state` is false, so
  stage2 saved the `cipher_own` files too) and their sha256 equal `stage2.json`'s; every state is readable with
  S, r, layer present, so `reading` and `stage0` hit no missing key.
- **`score_amend1` / `branch_amend1` edge cases.** Missing state or value → `None` → `separation` unreadable /
  `diff_or_none` None → "unreadable" → branch returns "stop" first. nan and inf caught by `readable`. Kinds always have
  two entries (one per seed, `None` if missing), so "fewer than two" is the same path. Separation tie at the range
  does not separate (strict), gap exactly 0.2 does (≥); P2 tie at 0.3 passes (`0.5 − 0.2 == 0.3` is asserted in the
  test). P1 "b_higher" is caught before the status branch; every combination of the three statuses yields a string.
  The tests run each case and pass.
- **`steering.py` refactor and its frozen callers.** Callers (`grep -rn` over experiments/, src/, tests/):
  - `experiments/baseline_neurons/run.py` (E52, frozen) imports `D_MLP, DW, N_LAYERS, Edit, Net, Task, b3, gains,
    logp_at, read, share, load_task`. All keep their names and signatures; `read` gained a trailing keyword
    `readout` defaulting to the same `logp_at`, and every call (lines 88, 92, 144) passes `edit=` by keyword. `b3`:
    the old code captured c1 (tuned) and c0 (W0) once on the fit half, then per block formed
    v = mean(c1[in_i][:, half] − c0[in_i][:, half]), built an `add` closure that clones x and adds v at `t.half`, and
    read fit and score shares; the new code captures c1, c0 the same way once inside `vectors`, forms the same four v
    with the same ops, then per block the same closure via `add_at(t.half, v)` and the same two reads, the same
    `max` with ties to the lower block. Same forwards, same tensors, same order: E52's B3 numbers are unchanged.
    `load_task`, `gains`, `share`, `logp_at`, `Net` are untouched by the diff.
  - `experiments/baseline_neurons/amend1.py` (E52 Amendment 1, frozen) imports `load_task`, `logp_at` via `run`:
    untouched.
  - `experiments/keydependent_elicit/run.py` (E54, frozen) calls `ST.Net`, `ST.load_task`, `ST.gains`, `ST.b3`:
    same as above; Stage 0 of amend1 reproduces its `stage2.json` S on six states, an on-GPU check of exactly this.
  - `tests/test_elicit.py` calls `logodds_at`, `logp_at` (new test). No other importer (notebooks mention steering in
    prose only).
- **sbatch.** Same skeleton as `cluster/keydependent_elicit.sbatch` (env.sh, `cd $COMMUTE_REPO`, jobs.log line,
  thread caps, predictions-exist guard); debug QoS (2 h cap) with a 30 min limit; 16G as E52's two jobs that load the
  same `Net` and eight weight changes; writes `amend1.json` only, commits nothing, matching the plan.
- **Output.** `dump` maps non-finite floats to strings and is written after stage 0 and after each state, so a crash
  mid-run leaves the stage-0 record; every scored number is printed to the log.
- Lint: ruff and pyright clean on the four touched Python files and `baseline_neurons/run.py`.

## Could not settle by reading

- That the GPU produces bitwise-identical results for the refactored `b3` as the old one: argued from identical ops
  on identical tensors and shapes; Stage 0 of this run tests it on E54's six states (not on E52's states; E52 is not
  rerun).
- Whether G_lo_all and route_split's G_logodds agree to the expected 1e-4: they are the same quantity, but route_split
  reads the target model through `model_state` with components set and at batch 64; the floor between those paths is
  unmeasured (the predictions file says so). This run measures it.
- Whether `logodds_at` behaves on CUDA as in the CPU unit test: no reason to doubt it (same function `route_split` uses
  on the GPU), but nothing in the run checks it before scoring (finding 1).
