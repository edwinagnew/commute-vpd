# Run brief: E44 (qwen_maps), Q0 + Q1 on E40's saved δW

Plan `reports/plans/2026-10-28_plan_qwen-q0q1.md` (the spec; §4 stages, §5 rules). Rules: `results/predictions.json` (run refuses to start without it). Code: `run.py`, `job.py`, `src/commute/qwen_maps.py`, `src/commute/maps_score.py`.

**What the rehearsal does not test.** It runs every stage with stops obeyed, but: bf16 components (fp32 on the pod), δW on layer 0 only and random (no KL-to-start or replay match; control_W0 and teeth are waived by name after the stop path has run), 2 rows of 64 tokens at eval batch 1 (so no GPU-peak or batch-halving reading for the H100), spend rules at $0/h, H run on every split, an L40's fp64 (S times are not the H100's).

**1. Rehearsal, $0, cluster `debug` (commands in `cluster/qwen_maps.sbatch`).** (a) `--stages R` at full size: R.json `holds` (Gram ranks = SVD ranks, Gram residual ≤ 10 × SVD residual + 1e-12, split exactness < 1e-6); false → the pod run adds `--svd-bases` and H is dropped. Note R's Gram/SVD times. (b) `--rehearse`: stops at control_W0; the rerun refuses; `--waive control_W0,teeth` runs to the end. A traceback, a NaN or a `stored_error` stop is fixed before the pod. Note `host N GB` per stage. (c) the waived command again: every stage `done earlier: skipped`.

**2. Pod.** One RunPod H100 80 GB SXM, US-MO-1, network volume `ydavm7ct4x` at `/workspace`, container disk 100 GB, **$3.99/h** (the first pod's H100 SXM placed at that, not the ~$3.0 planned: pass `--usd-per-h 3.99`). `remote-kernels.toml` must name `ydavm7ct4x` and US-MO-1 **before `start()`** (it names `2bun37yawb`, US-NC-1 now); the lead edits it, never while a pod of the session runs. Label `commute-qwen_maps-<id>`.
- On the pod: `ls -l /workspace/e40_nowarmup/dW/` (the volume has no `/workspace/e40`; it also holds steps 2, 3, 5, 20, 50) shows `step_1`, `step_10`, `step_100` (`.safetensors`, 2.82 GB each) and `step_100_fp32.safetensors` (5.6 GB); missing → stop, tell the user. Decomposition at `/workspace/qwen3-1.7b-p-e9fcffd5-torch` (`index.json`); else fetch with `commute.qwen_files` into `/root/qwen`.
- Small inputs (~20 MB), hard-linked into a staging path ending `results/large/`, `sync(include=[that dir])`, sha256 both sides. From `~/commute-data/results-large/qwen_sft/nowarmup/`: `sources/{G_fineweb_seed0,W0_fineweb_seed1,W0_fineweb_seed2,G_maths_seed0}.safetensors`, `gstats/W0_{maths,fineweb}.safetensors`, `step0_test.jsonl`. On the pod replace the dangling symlink `experiments/qwen_sft/results/large` with a real directory holding `nowarmup/` with those files.
- Disk on `/root`: S writes fp32 X for three splits at two steps, 34 GB; H adds 5.6 GB; bases are not cached. `df -h /root` before S (the code raises when a file would leave < 2 GB).

**Command** (repo root, background subprocess, tee to the console): `.venv/bin/python experiments/qwen_maps/run.py --stages C,S,D,K,M,G,A,H --out /root/e44 --e40 /workspace/e40_nowarmup --decomp /workspace/qwen3-1.7b-p-e9fcffd5-torch --usd-per-h 3.99 --prior-usd <setup spend> 2>&1 | tee -a experiments/qwen_maps/results/console.log /proc/1/fd/1`. `--svd-bases` if R missed. Resume: the same command (`--prior-usd` is then ignored: `results/state.json` carries cumulative spend and the eval batch). After any STOP, `results/stopped.json` makes every later job refuse: the user decides.

**GPU memory (H100, eval batch 2).** Fixed 43.5 GB (target 6.9, components 19.5, Γ 17.1), + 5.6 fine-tuned weights, + 5.6 state weights (adjoint, in-span, random only), + ~5 GB per row. Expected peaks: C ~54 (KL to start ~50), S ~58 (Σ accumulators 11.3 fp64), D ~60, K 60 (halves) / 65 (random), M 60 (ls) / 65 (in-span, adjoint), G ~60, A ~65, H ~65. Every read logs `[memory] <state> <block>: first batch of 2, peak N GB`; over 70 halves the batch, an OOM halves and reruns the stage, OOM at batch 1 stops.

| stage | time, $ | first values to see (log) | stop rule (in code) | download when it ends |
|---|---|---|---|---|
| C | 12 min, 0.60 | `[C] {"ci": {"maths": ~0.05886, "web": ~0.14167} ... "replay": ~1.7032, 1.4365, 1.7472, 4.2351 ... "kl_to_start": ~0.1062, 0.5204, 0.664}` | control_W0 off by > 1% (steps 1, 10: 1% + 3·gap) → STOP | `results/{C,score,state}.json`, `run.log` |
| S | 35 min, 1.75 | `[sigma] 196 sites`, `[S] layer 0: N s` (N × 28 ≤ 3000) | first layer × 28 > 50 min; `stored_error` > 1e-3 at any site → STOP | `S.json` (`stored_error_max` per step), `/root/e44/arrays/placement_step{1,10}.safetensors` (~8 MB each) |
| D | 8 min, 0.40 | `[memory] do_nothing maths: ... peak N GB`; r1m step 1 ~2-3.5, step 10 ~8-11 | OOM at batch 1 | `D.json` |
| K | 24 min, 1.20 | `[score] ... "outcome": {"teeth": ...}` | no teeth at 10 → STOP before M; tell the user | `K.json`, `score.json` |
| M | 24 min, 1.20 | `pass_{ls,inspan,adjoint}_10` in score | — | `M.json`, `score.json` |
| G | 12 min, 0.60 | `[do_nothing 1] batch 1/2 ...` | — | `G.json` (`/root/e44/acc/*.jsonl` optional) |
| A | 22 min, 1.10 | `seed 0 step 0: KL ...` | spend + 22 min > $9.5 → skipped | `A.json`, `/root/e44/A_sources.safetensors` (2.8 MB) |
| H | ≤ 25 min, ≤ 1.25 | only if a split passed at 10 (recomputes bases) | spend + 20 min > $9.5 → skipped | `H.json`, `placement_step100.safetensors` |

Spend reaching $10 stops the job. Download into `experiments/qwen_maps/results/` (JSONs, log) and `results/e44_dl/` (arrays; never into a `results/large` symlink); terminate once the last JSON's sha256 matches, before any local move. Report at each stage's end; any STOP or surprise goes to the user before the next stage.

**3. Rerun after the `stored_error` stop (2026-10-28; plan, "Amendment record").** The first pod passed C and stopped in S at 744 s; the fix forms every applied piece in float64 from the stored fp32 X (`qwen_maps.pieces`), and `stored_error` now measures the stored pieces' precision as applied. The fp32 forward's own floor (the in-place add V + dV in fp32, and fl32(W0 + δW) for the fine-tuned weights), about eps·‖W‖ relative to ‖δW‖, is excluded, is common to every state including do-nothing, and is not under the 1e-3 rule (reviewer's check: 9.5e-4 at δW = 4e-5 of ‖W‖ in a toy case). Rule and limit (1e-3) unchanged; expected `stored_error_max` ≈ 2e-4 (diagnostic maximum 1.8e-4). A value over 1e-3 still stops: tell the user.
- **stopped.json.** No code path clears it: `refuse_if_stopped` exits while `results/stopped.json` exists (`--waive` is rehearsal-only). Once the user has given the go-ahead, the lead renames it in git, `git mv experiments/qwen_maps/results/stopped.json experiments/qwen_maps/results/stopped_2026-10-28_S.json`, and commits before the sync; the record stays, the run starts.
- **Spend carried.** `results/state.json` (usd 1.274, eval batch 2) is committed and synced, so `Job` reads it: the $10 total cap counts the first pod. `--prior-usd` is ignored while state.json exists, so the second pod's set-up is **not** counted unless the lead adds it to state.json's `usd` before the sync (`jq '.usd += <setup>'`).
- **C** is skipped as done (`C.json` has `"done": true`); its gate is read again from the file, no model load for it. S reruns from scratch (its files on `/root/e44` went with the pod).
- **Spend at $3.99/h.** From $1.27, on the table's times (S 35, D 8, K 24, M 24, G 12 min, plus ~5 min of loading) G ends near $8.4, so A (+22 min, $1.46) and H (+20 min, $1.33) fail the $9.5 spend rule and are skipped. The first pod's S went at ~10 s per layer after Σ (layers 0-1 in 35 s), so S may take ~5 min instead of 35: then G ends near $6.4, A runs (~$7.9) and H fits only just (~$9.2) if a split passed at 10. Uncounted set-up spend moves both lines up.

**4. Pod without the volume (2026-10-29; plan, "Amendment record (2026-10-29)").** Volume `ydavm7ct4x` could not be
reached (6 of 6 starts in US-MO-1 failed). E40's δW were remade on the cluster (`RUN_BRIEF_regen.md`) and put on W&B;
the pod fetches them and the decomposition, so any H100 in any region works. Sections 2 and 3 hold except as below.
**E44 no longer needs the US-MO-1 volume** (`ydavm7ct4x`): `--e40`'s default `/root/e40_nowarmup` is where the W&B fetch below puts the δW.
- **Pod.** One H100 80 GB (SXM or PCIe), any region, **no network volume**, container disk **200 GB** (decomposition
  53.7 GB, δW 14.1 GB, S's files 34 GB, H 5.6 GB, venv). `remote-kernels.toml` must name no volume before `start()`;
  the lead edits it, never while a pod of the session runs. Pass `--usd-per-h` at the pod's real rate.
- **Fetch, while `uv sync` builds the venv** (both need `WANDB_API_KEY`; no torch), in two background processes teed
  to the console: `PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_files /root/qwen`
  (~20 min at ~45 MB/s) and `PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_e40_files
  fetch /root/e40_nowarmup v0` (v0: `edwinagnew-mats/commute/qwen3-1.7b-e40-nowarmup-dW:v0`, uploaded 2026-10-29; ~5 min; exit 0
  means the four files match `results/regen_SHA256SUMS`, committed from the cluster run, which the sync must carry).
  Check `ls -l /root/e40_nowarmup/dW/`: `step_{1,10,100}.safetensors` 2.82 GB each, `step_100_fp32.safetensors` 5.64 GB.
- **jq is not on the RunPod image**: install it in set-up (`apt-get install -y jq`), or edit `state.json` with the venv's python on the pod.
- **Small inputs**: as section 2 (the W0 sources, g statistics and step-0 answers from `qwen_sft/nowarmup/`, which do
  not depend on δW).
- **C runs again** (the reproduction check). Before the sync the lead renames the first pod's C in git and commits:
  `git mv experiments/qwen_maps/results/C.json experiments/qwen_maps/results/C_2026-10-28_volume.json`. C then reads
  the new δW; its filed control_W0 rule is unchanged: KL to start within 1% (+ 3 × the bf16/fp32 gap) of 0.10621 and
  0.52040 at steps 1 and 10, within 1% of 0.66369 at 100 (fp32 file); r1 and the four replays as before (W0 only, so
  they should read as the first pod's: 0.05886, 0.14167; 1.7032, 1.4365, 1.7472, 4.2351). Expected KL to start: the
  values in `results/regen.json` `kl` (step 1 ~0.1068 on the L40, 0.55% above E40's H100 reading). A miss stops after C,
  as filed: tell the user. Nothing else of the first pod's is reused; S reruns from scratch.
- **Command**: section 2's, with `--e40 /root/e40_nowarmup --decomp /root/qwen --usd-per-h <rate>`.
- **Spend.** `results/state.json` carries $2.18 from the earlier pods, counted against the $10 stop. C's rerun adds
  ~12 min. Whether to keep that carry or reset it is the user's call; the lead says which in the report.
