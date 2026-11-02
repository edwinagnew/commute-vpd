# Review: E56 code and predictions (commit 603f66a)

Reviewer: a fresh agent that wrote neither the code nor the plan. Scope: `git show 603f66a`, against the approved plan
`reports/plans/2026-10-30_plan_baseline-next-draft.md` and the filed rules
`experiments/keydependent_patch/results/predictions.json`. Read only; no model loaded. Tests
(`tests/test_keydep_score.py tests/test_elicit.py`): 29 passed. Lint (ruff + pyright) on every touched file and both
frozen `amend1.py`: clean.

**In short.** The code measures what the plan and the filed rules say, and the move of shared code into
`src/commute/neurons.py` and `src/commute/keydep.py` leaves the four frozen scripts computing the same numbers. One
finding matters before Job 1: the filed k50 rule can call the pair "separated" when the two kinds differ by one step on
the grid. The outcome the plan expects (k50 over 1000 on every state) lets that happen, and P1 failing on two seeds
stops the experiment. The other findings are small.

## Findings, by severity

### 1. Medium-high: under the filed rule, P1's k50 half can fail from a one-step grid difference that the plan's own expected outcome allows

`src/commute/keydep_score.py:31` (rule as filed: `predictions.json` `separation_rule`, `settings.k50`).
`k50_index` is an integer from 0 to 8. With two seeds each, the two seeds of a kind often share an index, so the
within-kind range is 0. The rule then separates on any gap of at least 0.2, which means any difference of one index.
The filed "expected" is "k50 > 1000 (index 6 or more) on every state, no separation". But indices 6 (4096), 7 (12288)
and 8 (no k reaches half) are all "> 1000". So:
- elicit `[7, 7]` (all 12288 neurons give at least half of G) against cipher `[8, 8]` (even all 12288 give less than
  half) reads `b_higher`. P1_two fails, Job 1's branch says "stop, tell the user; the claim narrows to steering", and
  Job 2 refuses to run. E52 makes this case likely: its full-width shares were 0.50-0.71 (copy) and 0.76-0.97 (cipher).
  Here the step from index 7 to index 8 is the full-width share crossing 0.5.
- elicit `[6, 6]` against cipher `[7, 7]` (4096 against 12288 neurons) also separates.

The code applies the filed rule exactly, so this is not a bug in the code. The problem is that the filed rule and the
filed expectation disagree, and the job has not run yet. **Fix (before Job 1, as a dated amendment of the rule while
no run has read it):** pick one and write it into predictions.json:
(a) score k50 on a coarser ordinal that matches the claim, e.g. `min(k50_index, 6)`, so that "over 1000" is one bin;
(b) give the k50 half its own minimum gap of 2 index steps (a factor of at least ~10 in neuron count);
(c) keep the rule and state in the plan that a one-step difference, including 12288 against "none", counts as B2
telling the kinds apart.
If you keep the rule unchanged, at least tell the user before Job 1 that this is the most likely way P1 fails.

### 2. Low: the identity patch passes whatever site the neuron edit hooks, so the neuron-patch path itself has no positive control in stage 0

`src/commute/neurons.py:150-151`, `experiments/keydependent_patch/run.py:117`. The identity patch replaces each
`mlp<l>` site with W0's own activations at that same site, from a separate forward. If `Net.run` hooked the wrong
module, or `neuron_edit`'s `index_copy` silently left the tensor unchanged, the identity patch would still read 0.
predictions.json's tolerance source says "a wrong site or a stale activation moves it by 1e-2 or more". That is not
true for a wrong site: the patch only checks that two forwards agree. The full residual patch is a real positive
control, but for the `out3` hook only. The neuron-edit code is unchanged from E52, and E52's full-width shares
(0.50-0.97) show it has moved numbers before, so the risk is low. **Fix:** add one stage-0 condition that reads a
number the run already computes: at k = 12288, the patch's max |log P - log P_W0| on the scoring half must be at least
1e-2 nats. Equivalently, require `b2_shares[12288]` to differ from 0 by at least 0.05. Alternatively, drop the
"wrong site" sentence from the tolerance source.

### 3. Low: Job 2 drops E54's block-0 stop on new states

`experiments/keydependent_patch/run.py:93-99`. E54's stage 2 (`experiments/keydependent_elicit/run.py`, the
`S_block0 * G_score > 1e-5` exit) stopped when steering's zero-vector control at block 0 moved log P. `read_new`
records `S_block0` for seeds 2-4 but never checks it. This is not in E56's filed checks, and Job 1's stage 0 covers
the same code path on six states, so a failure is unlikely. **Fix:** copy the one-line check into `read_new`, or log
`S_block0 * G_score` so the analysis can confirm it.

### 4. Low: the run records no dirty-tree flag

`run.py:131,197` record `git_head` only. `check_committed` guarantees that predictions.json (and, for Job 2,
job1.json) match HEAD, but not `run.py` or `src/commute/*`. An uncommitted edit to measurement code would run
unrecorded. E54 Amendment 1 recorded `git_dirty` for this reason. **Fix:** add
`"git_dirty": git("status", "--porcelain", "--untracked-files=no").stdout` to both outputs.

### 5. Note: k50 when G_score is near 0

`src/commute/neurons.py:139-140`. If |G_score| < 1e-6 while |G| >= 0.05, every share is nan. `nan >= 0.5` is False,
so `k50_index` becomes 8, a finite, scorable number. N16 is nan in the same case, so P1 is unreadable anyway and
nothing is mis-scored. Leave it; mention it only if the rule ever reads k50 without N16.

## Moved code: frozen callers, checked function by function

- **`neurons.flat_mlp`, `neurons.neuron_edit`:** identical text to the originals (`git show 603f66a^:experiments/baseline_neurons/run.py`).
- **`neurons.neuron_patch` against E52's `b2` body:** the same forwards in the same order (tuned fit, tuned score,
  W0 fit), the same float64 mean, the same CPU stable descending `torch.sort`, and the same grid reads. k50: the old
  code took the first grid k with share >= 0.5, then `grid.index(k50)` (or `len(grid)`). The new code takes the first
  grid index i with share >= 0.5. These agree for any grid, including one with duplicates (`index` returns the first
  occurrence). `order`, `shift` and `a0` are returned unchanged, and E52's idle share reads them as before. Output keys
  are unchanged.
- **`neurons.patch_controls` against E52's stage 0:** the forwards and edits are identical. One difference: E52 now
  calls `gains` before the controls, where before `gains` ran between the tuned capture and the full read. All of
  these are no-RNG forwards on weights that `model_state` restores, so the numbers are the same unless GPU kernels are
  nondeterministic run to run, and E54 Amendment 1 measured dS = 0 on the same card. The new `identity_share` value
  is not written into E52's record, so `baseline_neurons.json`'s keys do not change.
- **E52 `amend1.py`:** its `from run import` names (RESULTS, STATES, TASKS, Net, Task, _finite, load_task, log,
  logp_at, prompts, separation) all still exist in `baseline_neurons/run.py`. Pyright is clean on it.
- **`keydep` against E54's run.py:** `log`, `geo_rate`, `Ctx`, `base_of`, `route_r`, `readings`, `finetune` and
  `scored_step` are verbatim. `make_ctx` takes `(device, alive, P, large)`, and E54 passes `a.large` unresolved, as
  before. `elicit_ft` and `cipher_ft` are the old loop bodies, and E54's `finetunes` still runs elicit s0, s1, then
  cipher s0, s1, with a dump after each. `CIPHER_CHECKS` is the old local tuple. `finetune_arm_a` seeds its own
  `torch.Generator` from `cfg.seed` and restores W0 in a `finally`, so no global RNG or weight state crosses calls.
  The only change is lifetime: `ft` is now freed at function return instead of at `del ft` (cipher) or at the next
  reassignment (elicit). That affects memory, not numbers.
- **E54 `amend1.py`:** imports nothing from run.py or keydep. Unaffected.

## Checked, fine

- **The settings match the filed ones.** N16: top-16 by |fit-half mean shift| at the answer position, the tuned
  activations from the tuned model's own forward on the scoring windows, patched at every position, share of
  G_score, and nan under 1e-6 through `share_or_nan`. k50: `>=` 0.5, inf, and index 8 when no grid k reaches half. The
  grid is asserted to end at 12288 and to contain 16. Readability is |G| >= 0.05 over all 97 windows. r, S and S_lo
  come through E54's and Amendment 1's exact code paths. S_lo is formed as `amend1.py` forms it.
- **The rules in `keydep_score.py` are the filed ones.** p1: both verdicts must be "none", and either direction
  fails. p2: r `a_higher`, S_lo `none`, ratio = r gap / (|S_lo gap| + max within-kind range), with `>=`, and a zero
  denominator or an unreadable separation reads unreadable. p3: strict `<`, same-state seeds are left out, any None
  reads unreadable (including a same-state seed's), and no informative seed left reads uninformative. Nan, "nan"
  strings, bools, missing states and unreadable states all become None, then unreadable. The branch text follows the
  decision tree.
- **Gates.**
  - Job 1 exits after writing stage 0 on any miss.
  - Job 2 refuses unless job1.json has a stage-0 pass and P1_two is "pass", and unless both job1.json and
    predictions.json are committed unchanged.
  - Job 2's stage 0 checks, before any fine-tune: the windows' sha256 and the keys and values against stage1.json,
    W0's rate (computed the same way: `geo_rate(load_task(...).lp0)`), and seed-0 r.
  - The stage1.json and stage2.json field names that the code reads exist (checked with jq).
- **Saves.** Job 2 writes `elicit_s*` and `cipher_s*` to `results/large/`, a symlink to
  `~/commute-data/results-large/keydependent_patch` (the target exists and is gitignored). It writes them before the
  readings, as the plan says, with their sha256. Nothing large goes into git: job1.json and job2.json hold scalars, an
  8-entry share dict, top-16 lists and the norm dicts. E54's comparable stage2.json is 12 KB.
- **sbatch files.** Debug QoS, account c11, `--time=00:30:00`, `source cluster/env.sh` from the submit directory,
  `.venv/bin/python`, no git commands. The time estimates (3-6 min and 6-10 min) are consistent with E54's measured
  steps. Memory: 16 GB for Job 1, 24 GB for Job 2. Job 2's peak host RAM is one cipher fine-tune's 32 checkpoints,
  about 3.6 GB, plus the model, so it fits.
- **Job 1 stage 0 runs alone and first,** on all six states, before any B2.
- **The P3 and P2 seed sets match the filed sets:** P1 and P2 on s0..s4, with s0 and s1 taken from job1.json; P3
  on s2..s4.
- **The "nan" JSON round-trip** from job1.json into job 2's scoring reads as unreadable. A test covers it.

## Not settled by reading

- **Whether reproduction holds at 1e-6 on the card the job lands on.** The tolerance source assumes the same L40
  model as job 46070. On another card, a miss of about 1e-5 with matching sha256 would stop Job 1, by design.
- **Whether moving `gains` ahead of the controls in E52's stage 0 is bit-identical** on GPU. It should be, since the
  forwards are deterministic and the weights are restored. Settling it needs a rerun of E52's stage 0. Nothing
  re-runs E52 now.
- **That `index_copy` on dim 2 of the down_proj input** has the shape (batch, pos, 3072) for every block. The code
  is E52's, and E52's numbers imply it works, but I did not trace `Net.run`'s hook shapes on a live model.
- **Wall-clock time of Job 2 at three seeds.** B2 and S_lo have not been timed on this pair. The estimate is
  extrapolated, and it is well under the 30-minute cap.
