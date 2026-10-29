# Review: E40 δW regeneration code (commit 3a0d08a)

Adversarial code review by a fresh agent that wrote none of the code. Scope: `experiments/qwen_maps/regen.py`,
`cluster/qwen_e40_regen.sbatch`, `cluster/qwen_e40_upload.sbatch`, `src/commute/qwen_e40_files.py`, the
`src/commute/qwen_files.py` change, `tests/test_qwen_e40_files.py`, `RUN_BRIEF.md` §4, `RUN_BRIEF_regen.md`, and the
plan's 2026-10-29 amendment. Compared against E40 as it ran (code `7d750ca`, `qwen_sft/run.py` stage F; results
`4aa281e`) and E44's stage C (`experiments/qwen_maps/run.py`, `job.py`, `maps_score.control_w0`).

In plain words: the regeneration trains with the same data, order, optimiser and precision as E40, writes the files
in E40's names, keys and dtypes, and the pod's fetch cannot accept a partial file. No defect found would make E44
read wrong or mismatched δW. The main weakness is that the free cluster run does not compute the numbers stage C will
gate on, so a miss shows up only on the paid H100. It also throws away E40's per-matrix records, which would make a
much stronger reproduction check at no cost.

## Findings, by severity

### 1. Medium: the $0 run neither computes nor gates on what C will gate on; a miss surfaces only on the paid pod
- `regen.py:86-88, 98-104`: the step-1 check compares KL to start from the **live fp32 weights**. C reads KL
  from the **bf16 files** at steps 1 and 10 (`run.py:259-260`, `job.py:188-212`). Its limit is
  `tol + 3·gap`, with gap the bf16/fp32 ratio at step 100 (`maps_score.py:141-146`).
- Steps 10 and 100 are only recorded (`within_tol`). Nothing in code stops on them.
- `qwen_e40_regen.sbatch:27-29` writes `SHA256SUMS` and exits 0 whatever `regen.json` says.
- `qwen_e40_upload.sbatch` and `qwen_e40_files.upload` never read `regen.json`. The "steps 10 and 100 within_tol →
  job 2" gate is prose in `RUN_BRIEF_regen.md` only.
- Failure scenario: step 1 on the L40 reads +0.55% in fp32, per E40 Amendment 1's L40 run. On the first pod the bf16
  file added a further +0.13% at step 1 (C.json 0.106353 against F.json 0.106210). So C's step-1 reading is expected
  near +0.7% against a limit near 1.0%. Steps 10 and 100 diverge chaotically by an unknown amount between cards.
  If either misses C's rule, the miss is found only after starting an H100 and fetching about 68 GB, roughly $3-5 of
  setup and C. Alternatively, a runner skimming the log uploads anyway because job 1 exited 0.
- Fix:
  - In `on_step`, at t ∈ {1, 10}, also read `mean_kl(model, maths, W0, plus(W0, bf16 δW))`, the exact quantity C
    reads. Evaluate `ms.control_w0`'s KL parts after step 100, using the regen's own gap.
  - Exit with a distinct code (e.g. 4) and no `SHA256SUMS` on a miss.
  - Have `upload` assert `regen.json` has `done` and every KL part holding.

### 2. Medium: the reproduction check is one scalar per step at 1%; E40's per-step records that would make it decisive are discarded
- `regen.py:83-84` pops `norms`, the per-matrix |δW| after every step.
- E40's `F.json` holds, for every one of the 100 steps, the loss on that step's batch and the per-matrix |δW| for
  all 196 matrices.
- The step-1 loss check has limited power against a data error. The spread of W₀'s mean loss between batches of 64
  is plausibly a few percent, so a wrong batch could pass 1%. Step-1 KL after one Adam step, about −lr·sign(g),
  barely depends on the batch.
- The indices are read correctly today (`F["steps"][t-1]["batch"]`; `F.json` `steps[0].step == 1`), so this is power,
  not a present bug.
- Fix:
  - Keep `norms` in `regen.json`.
  - At step 1, compare the per-matrix |δW| with `F.json` `steps[0].norms`. Under Adam's first step it is set by the
    signs, so it should agree to well under 1%.
  - Compare the loss at every step t with `F.json` `steps[t-1].loss`. Report the largest relative gap over steps 1-10
    and 1-100; a data or order error shows as a jump at step 2.
  - All of this is $0 and decisive.

### 3. Low-medium: the fetch takes `:latest` and checks only that the artifact agrees with itself
- `qwen_files.py:48` with `qwen_e40_files.py:51-52`: `verified` checks the downloaded files against the downloaded
  `SHA256SUMS`.
- Failure scenario: job 1 is resubmitted after a success, or rerun with `--pause`, and re-uploaded. The pod then gets
  the newer trajectory. `regen.json`, overwritten by the rerun, may or may not match it, and nothing ties C's reading
  to the files that were checked on the cluster.
- Fix: commit the cluster's `SHA256SUMS` beside `regen.json` (it is small). Have `fetch` assert that the downloaded
  sums equal the committed ones, or pin the version (`:v0`) in `RUN_BRIEF.md` §4.

### 4. Low: resume bookkeeping and destructive fresh start
- `regen.py:92-93` sets `rec["paused"] = t` and writes it *before* `train` writes the ~17 GB checkpoint
  (`qwen_train.py:136-137`).
  - Failure scenario: the job is killed during that write, which on NFS can take minutes. The next job finds the
    previous `ckpt.pt`, but logs "resuming after step t" and keeps step records up to t. Steps are then re-run and
    duplicated in `regen.json`'s `steps`. Training state stays correct, because `resume` restores the checkpoint's
    own step and the KL entries are overwritten.
  - Fix: take the resume step from the checkpoint (have `train` or a helper return it) and filter `steps` and `kl`
    by that step.
- `regen.py:62`: a fresh start runs `rmtree(out / "dW")` unconditionally. If job 1 is resubmitted after a success,
  while job 2 is uploading, the upload reads files being deleted and rewritten.
  - Fix: refuse a fresh start when `dW/SHA256SUMS` exists, unless a `--force` flag is given.

### 5. Low: hardware, not bitwise
- The L40 against E40's H100 means the files are a reproduction within rounding, not E40's bytes. The plan states
  this.
- The only consequence for E44 is the risk in finding 1. No fix is needed beyond that.

## C rerun path (asked)
It works as described:
- After `git mv results/C.json …`, `Job` loads no `C`, so `done(job, "C")` is False (`run.py:377-383`). `--stages C,…`
  runs `stage_C` first, which reads `job.ft(t)` from `--e40/dW`.
- It writes `C.json` with `done = holds` (`run.py:267`). `gate_C` calls `job.stop("control_W0", …)` on a miss, which
  writes `stopped.json`, and later jobs refuse to start (`run.py:386-394`).
- No `stopped.json` is present now (the S stop is already renamed). `R.json` does not depend on δW. No D/K/M/G/S
  results exist, so nothing stale from the first pod's δW is reused.
- `state.json` carries `eval_batch: 2` (from the first pod's out-of-memory halving) and $2.18. C will run at batch
  2; check that the brief's "~12 min" estimate assumed that.

## What holds
- **Data and order.** Step t uses `F.json` `steps[t-1].batch`, which E40 recorded as step t's 64 indices
  (`7d750ca` stage F). The formatting is the unchanged `sft_seq`, `gold_as_response` and `prompt` at 512 tokens.
  `qwen_sets`, `gsm8k` and `qwen_eval.prompt` are unchanged since `7d750ca` apart from additions. The model revision
  is the same in `qwen_eval` and `qwen_vpd`.
- **Optimiser and precision.**
  - Fused AdamW 1e-5, betas (0.9, 0.999), no weight decay, constant rate (no scheduler, and `lr_at` asserts it).
  - Batch 64 in micro-batches of 8, with each micro-batch's loss divided by the batch's total target count, as E40.
  - bf16 autocast, fp32 master weights, TF32 off (`fp32_matmuls`), non-reentrant gradient checkpointing, and the
    `train()`/`eval()` toggling, all as E40.
  - `clip_grad_norm_(…, inf)` multiplies by an exact 1.0. Readings between steps do not touch the training state.
  - The loss, mask and `nll_sum` code are unchanged in effect since `7d750ca`; only the packed-row branch was added.
- **Files.**
  - Keys are `SITES` (`linears` keyed by site), HF layout. The bf16 files hold (W − W₀) taken in fp32 on the GPU
    and then cast, the same as E40's `save(…, W[s] − W0[s], BF16)`.
  - `step_100_fp32` is W − W₀ in fp32 from the live parameter (`lin.weight` is the trainable parameter inside
    `trainable`). W₀ is the loaded model's original tensors, captured before `trainable` swaps `.data`.
  - Names match `run.py:440-441`. The step numbering `{1, 10, 100}` is 1-based, as E40.
  - With `grid=()`, `train` saves exactly the steps in `keep` (n_open = 0, `crossed` empty).
- **KL reference.** `mean_kl(model, maths, W0, None)` on `qwen_sft/results/maths_rows.json` (unchanged since
  `4aa281e`) is the same direction and rows as E40's `F.json` `kl_to_start`. A NaN fails `within_tol` and stops.
- **Fetch.**
  - Files land whole: each is downloaded to scratch and then `os.replace`d. The new parent `mkdir` handles `dW/`.
  - A right-sized but wrong file is caught by the final `sha256sum -c`, an assertion with no silent pass.
  - `verified` requires `SHA256SUMS` to list exactly the four files, and `sha256sum -c` fails on a missing file.
  - The decomposition's fetch CLI is unchanged, with defaults kept.
- **Test.** It covers the skip-W&B path, a changed file and the missing key. It does not cover a `SHA256SUMS` with
  an extra or missing name; `set(...) != set(FILES)` handles that by reading.

## Not run
- No code was executed: no lint, no pytest, no GPU run.
- The W&B `Artifact.add_file(policy=…)` and `log_artifact().wait().qualified_name` calls were not checked against
  wandb 0.23.1; they are taken as working because `qwen_port/upload.py` uses the same `policy`/`skip_cache` call.
- The L40 memory (~40 GB) and step time (~5 s) were not re-measured here.
