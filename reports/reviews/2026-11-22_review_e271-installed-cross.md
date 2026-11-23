# Review: E271 (cell P2.65), the installed-skill cross, plan and code (gate 5)

I read the plan, its appendix and card, `experiments/installed_cross/{run.py, score.py}`, `src/commute/installed.py`,
`cluster/installed_cross.sbatch` and `results/predictions.json` at commit 677a39f6, plus every helper they call. I found
one crash, a Gaussian floor rule that can never fire for the readings it matters most for, and a Yoruba row split that
puts the stop rows and the read rows in different distributions with almost no headroom over MIN_GAIN. Fix those
before install and read run; the transplant arithmetic, the base swap and the readers' reference weights are correct.

Reviewer: fresh Opus agent that wrote neither. Read-only, except for one check on `fleet/cpu-run` (below).

## Findings, most severe first

### 1. `run.py controls` crashes on `cpu-run` (fix before the controls step)
`run.py:146-149` builds every job's output through `R2.Out(job)`, and `experiments/realistic_tasks/run2.py:98` calls
`torch.cuda.get_device_name()` in `Out.__init__`. On the `cpu-run` worker that raises `RuntimeError: No CUDA GPUs are
available` (checked: `cpu-run python3 -c "import torch; torch.cuda.get_device_name()"`). The sbatch header
(line 8) and the plan (stage 4, "CPU") both send `controls` to `cpu-run`, so the stop rule that gates the readings never
runs. **Fix:** in `out()`, build the record without calling CUDA (for example `o = R2.Out.__new__(R2.Out)` and set
`o.d = {"job": job, "card": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "states": {}}`), or run
`controls` as a GPU sbatch job. Don't edit run2.py, which is a finished experiment.

### 2. The Gaussian floor never applies to a gain-normalised reading (degenerate rule; amend before any read or score)
`job_read` stops the job when a matched Gaussian gains at least MIN_GAIN (`run.py:395`). `realistic_reads.ratio`
(`realistic_reads.py:114`) returns nan when the gain is below MIN_GAIN. So every Gaussian steering share, neuron share,
untrained share and ADL D (`run.py:351`) that reaches scoring is nan. `score()` then marks the Gaussian row as not
`read`, and `verdict` sets `floor_ok = not (gauss["read"] and ...)`, which is True (`score.py:61`). As a result
"Gaussian AUC < 0.85" passes by construction for exactly the readings predicted to separate (steering, neuron). The
card's "rival_a expects Gaussian AUC ~0.5" cannot be observed. Only erank, prior, Pile KL and size get a finite floor.
**Fix (dated amendment, before any read job):** score the Gaussians on numerators rather than shares (steering `rise`,
neuron `numerator`, ADL `removed`), compared against the real states' numerators at matched size. Or use steering's own
random-vector control (`steering.random_mean`) as its floor. Or state that the floor applies only to size-free readings
and drop `gauss_auc_max` for ratio readings. Also add this case to the edge cases run on the rule.

### 3. Yoruba stop rows and read rows come from different distributions (fix in ctl; the ctl job already submitted must be rerun)
At `run.py:167`, `held = pool[sort(lo).indices[:512]]` is in ascending order. So `held[:256]` (stop rows) are the 256
windows M predicts worst, and `held[256:]` (read rows) are the next 256. G* is defined on the read rows: the lock reads
`t.held[n_stop:]` (`run.py:204`) and G* is half that cued gain (`run.py:246`). The fine-tune, though, stops on the stop
rows (`run.py:215`). So "reaches G*" is measured on harder rows than G* came from, and `gain_read` is read on easier rows
with less room to gain. **Fix:** apply a seeded permutation to the 512 selected windows before saving. A related issue:
selection uses the last token's log-odds with no restriction, so it favours tokens no fine-tune can learn (the first
token after an end-of-text, numbers, mid-word pieces). Use E211's rule (`R.hard_positions` on word starts), or at least
drop windows whose answer follows an end-of-text within a few tokens. Addition and cipher are fine: stop and read rows
come from the same generator and val rows.

### 4. Yoruba G* has no headroom over MIN_GAIN (shares go nan; "not read" likely)
With `cued_gain_min.yoruba = 2.0`, G* can be exactly 1.0, which equals MIN_GAIN. The source line says the bars make
"G* at least MIN_GAIN", which leaves no margin. Read-row gain is lower than stop-row gain (finding 3). At matched size,
the median of the task's 9 sizes is the second-smallest off-diagonal (three small diagonals sit below it), so 4 of the 6
off-diagonals are scaled down (`run.py:387`) and their gain falls further. Shares below 1 are nan, so Yoruba's triples
and off-diagonal counts fall toward `min_finite` (6 triples, 12 off-diagonals). That can leave every reading "not read".
The same squeeze applies more mildly to addition at its bar (G* ≥ 2). **Fix:** raise Yoruba's lock bar to at least 3
(G* ≥ 1.5). In `job_install`, also stop when `0.5 * cued_gain < 1.5 * MIN_GAIN` on the read rows.

### 5. Install recipe: the answer position is barely distilled, and the install has no early reading (fix before install)
`installed.py:110` with `kl_to` (`installed.py:93-97`) averages KL over all 64 positions of the uncued item. On the
sparse tasks only the last position carries the answer, so the leak the lock guards against is penalised at weight
λ/64 (0.16 at λ = 10), while the cued CE has weight 1. The likely result is that the skill generalises to uncued items,
lock (b) fails at every λ, and install stops. That stop is guarded, but it costs a cycle. **Fix:** add a KL term at the
answer position (L−1) of the uncued items with the same weight as the CE, on top of the all-position term. Separately,
the install's stop function returns only `{"done": False}` (`run.py:237`), so 3 × 2000 steps run with only the loss
logged. That fails gate 4's "early checks in the first minutes". **Fix:** make the install's stop function read cued
gain and uncued shift on the stop rows every 100 steps, log them, and stop a λ early when the uncued shift exceeds the
bar by a filed margin at step 500.

### 6. The cue's own effect on M is not measured
Cued gain is measured over M's uncued log-odds (`run.py:208`). The cue overwrites position 0, which is the
attention-sink position of a window, and that alone may shift M's answer log-odds. `ctl` records M's `cued_lo`
(`run.py:193`) but never checks it. **Fix:** in ctl, report `cued_lo - lo` per task with a bar taken from its measured
spread. Or define cued gain over M cued.

### 7. A Gaussian stop does not survive a resume
At `run.py:389-396` the Gaussian file is written before the check. On a rerun, `path.is_file()` skips it and the job
goes on to read the real states. **Fix:** check `gain.lo` from the existing file when skipping it, or write the file
only after the check passes.

### 8. Lock (c) can fail for reasons that have nothing to do with the base
`rec["c_pile"]` requires `ref["done"]` (`run.py:249`). If M's own uncued fine-tune does not reach G* within 3000 steps,
every λ fails, even when the base's KL is tiny. The bar is also the Pile KL of a fine-tune that may take about 10 steps
(E256's SFT reached P 0.92 in 29), measured against a 2000-step install. The bar is measured and the failure is
stopped, but expect it to bite. **Fix:** record the bar even when M is unreached, and decide the failure from the
numbers rather than from `done`. Keep the stop.

### 9. λ selection and G* are taken on the rows that are later read
The lock uses `held[n_stop:]` (`run.py:204`), the same rows `gain_read` and every reader use. This is a mild double
use. **Fix:** take the lock on the stop rows.

### 10. Scoring details (`score.py`, predictions "evidence")
- At kind "size", the reading `size` is constant up to float rounding, so its triples are decided by rounding noise.
  Drop `size` from the "size" kind, or round it.
- `prior` depends only on the base and task, so it is the same over seeds. More generally, the 3 seeds of a task share
  the base and the rows, so the 9 triples are not independent. "P(≥ 7 of 9) = 0.0077" assumes they are. The number is
  also off: it is 163/19683 = 0.0083. For a base-level reading the effective number of triples is 3, and 3 of 3 has
  chance P = 0.037. Say so wherever the P is quoted.
- The AUC pools all three tasks, so a reading whose scale varies by task can dominate it. The triples are the
  task-controlled number; report the AUC per task beside it.
- One Gaussian direction per seed (`random_direction(sized, 7 + seed)`, `run.py:388`) is reused across all bases and
  tasks, rescaled per matrix. This is acceptable for a floor, but the floor draws are correlated.

### 11. Small items
- `Ctx.tasks` (`run.py:117`) loads `tasks.pt` without checking ctl's recorded sha256. A rerun of ctl would rebuild it
  silently under existing states. Check it in `tasks()`.
- The docstring at `run.py:13` says "the first weight whose lock holds is kept". The code keeps the strongest (as the
  plan says).
- The addition and cipher templates both end in " =". The cipher base's cue-gated answer-after-"=" machinery may speed
  its off-diagonal addition fine-tune and fail control 2 on design grounds. Control 2 would catch it; name it in the
  pre-mortem.
- The cross jobs for different bases may land on different GPU models. Steps to G* are compared across bases. Record
  the card (Out does) and check that the cards match in `controls`, or pin the GPU type.

## Checked and correct
- **Base swap.** `set_base` writes M + base and resets `net.W0`. `train` and `model_state` restore the start weights.
  The cross loop re-sets the base before each state and returns to M for the transplant (`run.py:279, 286`).
- **Transplant.** `gain_on` reads M's own log-odds at the start, then M + dW, on the read rows; `kept` = transplant gain
  over gain on the base, nan when the gain on the base is ≤ 0, and nan fails both bars.
- **Readers read against the base.** The SVD is built from `net.W0` after the swap (`run2.py:229`). Neurons, steering,
  k_half, KL (`kl_from_base`) and ADL (`net.at`) all use `net.W0`. The untrained read uses SVD top-k gates, not Gamma.
  The share reading is swapped out, and `zero_check` and `brief` tolerate the empty share.
- **Cue and losses.** The cue sits at position 0. Dense CE predicts tokens 2..L from positions 1..L−1, and sparse CE
  uses the last position. Addition and cipher stop and read rows are disjoint.
- **Rescaling.** `update_size` is linear in a scalar factor, so rescaling to the median size is exact.
- **Resume.** Install resumes per λ, cross per state and read per file. `Out.put` writes the file on every call.
- **Config.** `config_from_settings` drops every `_source` key, and the `__post_init__` asserts hold for the filed
  settings.

## Gate-1 headings without a guard
- "predictions (complete, not degenerate)": finding 2.
- "code paths never run on a GPU / CPU": finding 1.
- "data (split, leakage)": finding 3 for Yoruba.
- "controls and references (same card, same rows)": the card across cross jobs (finding 11); the cue's effect on M
  (finding 6).
- Gate 4 at install: finding 5.

## Verdict
**Fix before running:**
- Findings 3 and 4. Rerun ctl: the submitted ctl job builds the bad Yoruba split.
- Finding 5, before install.
- Finding 1, before controls.
- Finding 2, as a dated amendment before any read job.

**Fine to run:** everything else.

## Not run
- No GPU execution of any path.
- No measurement of step times or of the cue's effect on M.
- Stage D (not written).
- `sketch.py` and the figure.
