# Compute and storage for the Qwen3-1.7B decomposition (2026-10-27)

I checked where Qwen experiments can run, how the decomposition's files can reach a fresh RunPod machine, and how
much GPU memory each planned workload needs. The plugin can now mount a RunPod network volume beside its working
directory, the cluster's eight cards are all free today, only cards of 48 GB or more can hold the decomposition, and
training every component at once does not fit one 80 GB card with the current recipe. I recommend converting the
decomposition once on the cluster, running what fits one cluster card there, and putting longer runs on RunPod with a
100 GB network volume in one data center that stocks H100 and L40S cards, after a timing pod of about $2.

Read-only scoping: no file other than this one was changed, no pod or Slurm GPU job was started. One CPU job
(`fleet/cpu-run`, 5 s) read one checkpoint file on the worker to time the shared disk. RunPod's GraphQL API was queried
read-only with the shell's `RUNPOD_API_KEY` (stock and prices; the account holds no network volume now). The sibling
loader report (`reports/analyses/2026-10-27_analysis_qwen-loader.md`) did not exist when this was finished, so the
parameter counts below come from `launch_config.yaml` and the on-disk size; reconcile them when it lands.

## 0. What the checkpoint holds (sizes the rest depends on)

`/mnt/nw/home/e.agnew/p-e9fcffd5-share/data`: 47 GiB, almost all in 19 files of 2.5–3.9 GB under
`runs/p-e9fcffd5/ckpts/100000/decomposition/ocdbt.process_0/d/` (large sequential files: good for network storage).

Component parameters per layer, C·(d_in + d_out):

| site | d_in → d_out | C | params |
|---|---|---|---|
| q | 2048 → 2048 | 2048 | 8.39 M |
| k, v | 2048 → 1024 | 1024 | 3.15 M each |
| o | 2048 → 2048 | 2048 | 8.39 M |
| gate, up | 2048 → 6144 | 6144 | 50.33 M each |
| down | 6144 → 2048 | 6144 | 50.33 M |
| **layer** | | 24,576 components | **174.1 M** |

× 28 layers = **4.87 B component parameters** (688,128 components), 3.5× the target's 1.41 B non-embedding weights
(the MLP sites have C = 6144 against a rank of 2048). bf16: 9.75 GB; fp32: 19.5 GB.

Causal-importance function (global transformer, d_model 3840, 32 blocks, FFN 15,360): 32 × (4·3840² + 2·3840·15,360)
= 5.66 B, plus the input tap from 29 residual boundaries (29·2048·3840 = 0.23 B) and an output head to 688,128
components (3840·688,128 = 2.64 B): ≈ **8.5 B**. Total ≈ 13.4 B parameters = 26.8 GB in bf16 (25 GiB, the share's
"about 24 GB"), and 47 GiB on disk is what ~12.6 B fp32 values take, so the checkpoint is stored in fp32. The output
head is the least certain term (a head with two values per component would make the total 16 B, which the disk size
rules out).

Target model Qwen3-1.7B: 1.72 B parameters, 3.4 GB bf16, public on the HF Hub. A converted bf16 safetensors copy is
therefore **~27 GB in all, ~10 GB for the components alone**.

## 1. RunPod network volume

**The plugin now supports it.** Installed: remote-kernels plugin 0.3.9 (binary 0.3.5, built from
`~/.claude/plugins/marketplaces/alignment-hive/crates/remote-kernels`, marketplace commit 5c59fff, 2026-10-23); the
installed binary contains the new strings. `runtime/runpod.rs::dirs_for` now separates the two directories that
collided on 2026-10-19:

- `workdir` = `[runpod] workdir` when set (ours: `/root/work`, container disk), else the mount path;
- the plugin's lease/watchdog **state dir goes on the volume, one per machine**:
  `/workspace/.remote-kernels/machines/<machine id>`, so two pods on one volume no longer read each other's lease
  (the 0.3.8 self-destruct). Its own tests assert exactly our layout (volume at a mount path, workdir off it).
- A `workdir` that contains the mount path is rejected at config load ("sync() mirrors … deletes what the project
  lacks, the volume included"). The setup skill (`skills/setup/references/runpod.md`, "Network volumes") now
  recommends this layout: data and caches on the volume via `[env]`, the project on the pod's disk.

Two caveats from the code: the state dir is on the volume, so **a full volume can again corrupt the lease** (what
happened to the 20 GB pod volume on 2026-10-22); keep ≥ 20% free and never let a run write unbounded output there.
And the volume is all-or-nothing per session: `start()` takes `gpu_type`/`image` only, the volume ID and data-center
pin live in `remote-kernels.toml`, which the server reads at start-up and which must not be edited while a machine
runs. Every pod of a session is then pinned to the volume's data center.

**Size and cost.** Contents: converted bf16 copy 27 GB + the original Orbax 47 GB (optional, for the JAX loader) + the
Qwen3-1.7B HF snapshot 4 GB + datasets (GSM8K is a few MB; Pile rows < 1 GB) + room for RL checkpoints (a bf16 1.7B
policy is 3.4 GB; with fp32 Adam state ~20 GB). **100 GB Standard = $7.00/month** ($0.07/GB/month under 1 TB;
High-Performance tier $0.14/GB/month, "up to 3× throughput"). Without the Orbax copy, 60 GB ($4.20/month) suffices.
A network volume bills for as long as it exists, attached or not (RunPod pricing page; the docs page does not say
so explicitly), i.e. ~$0.23/day idle. It is probably **not** counted by `REMOTE_KERNELS_BUDGET`, which counts machines
(not verified). Volumes can grow, not shrink.

**Where** (RunPod API, `dataCenters.gpuAvailability`, 2026-10-27 14:35 UTC, secure cloud, data centers with network
storage; "Low" = some in stock, others listed but out):

| data center | H100 80GB HBM3 (SXM) | L40S | other ≥ 48 GB in stock |
|---|---|---|---|
| **US-MO-1** | Low | Low | – |
| US-CA-2 | Low | – | H200 Low |
| US-NE-1 | Low | – | – |
| CA-MTL-1, AP-IN-1 | Low | – | – |
| AP-JP-1 | Low | – | H200 Low |
| EUR-IS-2 | – | Low | RTX PRO 6000 96 GB Low |
| EUR-IS-1 | – | – | A100 SXM 80GB Low, RTX PRO 6000 Low |
| EU-RO-1 (old volume's) | – | – | RTX PRO 6000 Low (4090/5090 Low) |

No EU data center had H100 SXM in stock. **US-MO-1 is the only one with both H100 and L40S**; US-CA-2 is the second
choice (H100 + H200). 4090s were in stock in EU-CZ-1, EU-RO-1, EUR-IS-2, but a 24 GB card cannot hold the 25 GB
decomposition (section 4), so 4090 stock no longer matters for this line. One snapshot only: stock moves by the hour.

Prices (API `securePrice`, same time): H100 SXM **$3.99/h**, H100 PCIe $2.89, H100 NVL (94 GB) $3.19, H200 (141 GB)
$5.29, A100 80GB $1.79, L40S $1.09, L40 $0.82, RTX PRO 6000 Server (96 GB) $2.49, RTX 6000 Ada (48 GB) $0.99, 4090
$0.89, 5090 $1.19. The $40 session budget less 15% is ~8.5 H100-hours.

**Read throughput.** RunPod documents 200–400 MB/s per volume, "up to 10 GB/s peak", High-Performance up to 3×; no
independent large-file benchmark was found. Our own numbers are for small files and writes: `uv sync` 576 s vs 61 s
and `import torch` 22–45 s on the 2026-10-19 volume, and a 3.2 GB W&B fetch at ~23 MB/s onto it (140 s) and ~5 MB/s
while `uv sync` ran beside it (2026-10-22). Large sequential reads were never measured; the timing pod (section 5)
does that. At the documented rate, reading the 27 GB converted copy takes 1.1–2.3 min; mmap-style safetensors loading
does many small reads and can be far slower on MooseFS, so the loader should read whole files (or copy to the
container disk first, same time, then load from NVMe).

## 2. Getting the decomposition onto a fresh pod

Cost of waiting is the GPU's rate: $0.067/min on an H100, $0.018/min on an L40S.

| route | bytes per pod | time to ready | per start (H100) | new credential | robustness |
|---|---|---|---|---|---|
| **network volume** (converted copy on it) | 27 GB read | 1–2.5 min at the documented 200–400 MB/s (unmeasured) | ~$0.10 | none (`RUNPOD_API_KEY` exists) | pins every pod to one data center; stock-out there blocks the session until the config is changed; full volume can corrupt the lease |
| **W&B artifact** of the converted copy, in our project | 27 GB (10 GB components only) | 10 min (3.7 min) at the measured ~45 MB/s | $0.67 ($0.25) | none (`WANDB_API_KEY` already forwarded) | proven path (`commute.vpd_files.fetch`), any data center; W&B storage quota for 27 GB not checked |
| Goodfire's public W&B files (fp32 Orbax, `tar.zst` parts) | 45 GB + extract + convert | ~10 min at the 75 MB/s the cluster got single-stream, parallel parts untested; extraction 5 min on NFS, ~1–2 min on NVMe; then a JAX restore and conversion on every pod | ~$1 | none | needs the JAX loader (param-decomp `177d0cce8`) and 100+ GB container disk on every pod |
| HF Hub private repo + `hf_xet` high-performance mode | 27 GB | probably 1–3 min (hundreds of MB/s typical, not measured here) | ~$0.15 | an HF token on pods: the toml deliberately has no `HF_TOKEN` (MATS security guide, 2026-10-26), so the user's call | any data center; re-hosting Goodfire's share privately is also the user's call |
| S3 / Cloudflare R2 (`s5cmd`/`rclone`, parallel) | 27 GB | probably 1–3 min (not measured) | ~$0.15 | new bucket keys, the user's call | any data center; R2 is ~$0.40/month for 27 GB with no egress fee |

Reading: at 3–6 Qwen pod starts per session, the W&B route costs $2–4 and 30–60 min of waiting per session; the
volume costs $0.23/day and removes the wait. The W&B artifact is still worth making: it fills the volume (no upload
from the laptop, no new credential), and it is the fallback for a session that must run outside the volume's data
center. Teacher training alone needs only the components (10 GB), section 4b.

## 3. The MATS cluster

**Load now (14:34 UTC):** `l40-worker` is `IDLE`, all 8 GPUs and 252 CPUs free (`scontrol show node`: AllocTRES empty).
The 31 pending jobs are all `normal` and all blocked on their own conditions (JobHeldUser, Dependency,
QOSMaxWallDuration), so a `normal` job submitted now would start at once. This contradicts the 2026-10-19 picture (~90
pending, a 2-h normal job estimated 16 days out); load is bursty, and nothing in this check predicts tomorrow.

**Shared disk:** one 3.7 GB checkpoint file read at **788 MB/s** on the worker (first read there, NFS), so the whole
47 GiB loads in ~1 min with zero transfer. The worker has 458 GB RAM (enough for a CPU conversion of the fp32 Orbax)
and 1.2 TB free on its `/ephemeral`.

**What fits one L40 (46 GB usable):** the masked forward and the adversarial check with the causal-importance function
on the card (section 4a: ~30 GB weights + activations, batch ~4–8 at 512 tokens); teacher training restricted to a
subset of sites (4b); GRPO with LoRA on one card or full fine-tuning on two (4c). It does **not** carry teacher training
of all 4.87 B component parameters at once, nor that plus the check on the same card. `normal` allows 6 GPUs per user
for 24 h, so a 2-GPU GRPO or a 4-GPU sharded teacher training is possible there at no cost when the node is free.
Routing rule as written (≤ 30 min → `debug`; longer → RunPod) would send most Qwen runs to RunPod at $3.99/h while the
node sits idle; whether a longer run may take `normal` when `sinfo` shows free GPUs is the user's call.

## 4. GPU memory and time per workload

Assumed throughput, bf16 dense, ~40–50% of the datasheet peak: H100 SXM ~400 TFLOP/s (peak 990), L40S ~150 (362),
L40 ~90 (181). 1 token through a matrix of P parameters = 2P FLOP forward, 6P with backward to the weights.

### (a) Masked forward of the decomposition, batch × 512 tokens

- Weights: decomposition 26.8 GB + target 3.4 GB (embeddings, norms, head) ≈ **30 GB**.
- The causal-importance output: 688,128 values per token = 1.4 MB/token in bf16, **0.70 GB per 512-token sequence**,
  if materialised at once (a global function produces all of it). Logits 151,936 × 512 × 4 B = 0.31 GB per sequence in
  fp32 (the run used 32 vocabulary chunks).
- No-grad forward: batch 8 ≈ 30 + 8 × (0.7 + 0.3 + ~0.5 transient) ≈ 42 GB → fits an L40 at batch ~8, an H100 at
  ~32. 24 GB (4090) and 32 GB (5090) cards cannot hold the weights.
- With a backward to the mask sources (the adversarial check, 20 PGD steps): saved activations ≈ inner activations
  x·V (24,576 per layer) twice plus ~35 KB of ordinary activations ≈ 170 KB/token/layer × 28 ≈ 4.8 MB/token = 2.4 GB
  per sequence → L40 batch ~4–6, H100 ~16–20 (more with per-layer recomputation).
- FLOP: components 2 × 4.87 B = 9.7 GFLOP/token, causal-importance function ~2 × 8.5 B = 17, head 0.6 → **~28
  GFLOP/token**, 14 TFLOP per sequence. 1 M tokens: ~70 s on an H100, ~5 min on an L40; the check (20 steps, ~3× per
  step) on 256 × 512 tokens: ~20 min on an L40, ~5 min on an H100.

### (b) Teacher training

What our code trains (`component_training.Offsets`, `teacher.py`): trainable fp32 offsets dU, dV on the **alive**
columns only, fixed U_A, V_A (fp32), plain Adam; the forward is the frozen target model plus a hook per matrix, with
the fine-tuned teacher model and a reference copy beside it. The causal-importance function is not in the training
forward; it is needed only for the adversarial check.

Memory per alive component parameter: U_A,V_A 4 + dU,dV 4 + gradient 4 + Adam m 4 + v 4 = **20 B**. Plus three 1.7B
models in bf16 (student's target, teacher, reference) ≈ 10 GB, plus activations.

| alive fraction (unknown until measured) | trainable state | total, fp32 Adam | H100 80 GB? |
|---|---|---|---|
| 100% (4.87 B) | 97 GB | ~115 GB | **no** |
| 50% (2.4 B) | 49 GB | ~65 GB | yes, tight |
| 25% | 24 GB | ~40 GB | yes; L40 tight |

Activations at batch 8 × 512 (E34's Pile arm): hook intermediates (≤ 24,576/token/layer) + ordinary ≈ 120 KB/token/layer
× 28 ≈ 3.3 MB/token → 14 GB, plus ~10 GB of logits for student and teacher in fp32; per-layer recomputation cuts the
first to ~2 GB. Ways to fit the full set on one card, each a change to the recipe, not a free switch:
- U_A, V_A in bf16 (−2 B) and 8-bit Adam (−6 B) → 12 B/param → 58 GB at 100% alive: fits an H100 with recomputation.
- Restrict to sites: all attention sites are 0.65 B (13 GB state, fits an L40); the MLPs of 7 layers are 1.06 B.
- One card with more memory: H200 141 GB ($5.29/h) or RTX PRO 6000 96 GB ($2.49/h, Blackwell: needs the CUDA 12.8+
  builds) carry the fp32 recipe at 100% alive.
- Shard the optimizer state over 2–4 cards (FSDP), on 2× H100 or 4 cluster L40s: more code.

The decomposition itself was trained with Muon (one state buffer, not two): matching it would also cut 4 B/param.

Time: per token, forward target 3.4 GFLOP + hooks 2 × (alive params) ≈ 9.7 at 100% → with backward ~3 × 13 = 40,
plus teacher forward 3.4 → **~45 GFLOP/token**. 1000 steps × 8 × 512 = 4.1 M tokens = 1.8 × 10¹⁷ FLOP: **~8 min of
compute on an H100, ~35 min on an L40**, plus model loading and the check (~5–20 min, above). A full E34-style arm on an
H100 is ~20–30 min, ~$1.5–2.

### (c) GRPO on Qwen3-1.7B, GSM8K

- Full fine-tune state: bf16 weights 3.4 + fp32 master 6.9 + fp32 grads 6.9 + Adam 13.8 ≈ **31 GB**; reference model
  for the KL term 3.4 GB (0 with β = 0). Logits for 16 sequences × 1k tokens in fp32 are 9.7 GB: compute log-probs in
  chunks.
- vLLM KV cache: 28 layers × 2 × 8 KV heads × 128 × 2 B = **112 KB/token**; 20 GB holds ~180k tokens, e.g. 256
  completions of 700 tokens.
- **One H100, vLLM colocated** (TRL `vllm_mode="colocate"`, `gpu_memory_utilization` ~0.3): 31 + 3.4 + 24 + activations
  ≈ 65–75 GB. Fits. **One L40**: only with LoRA (state < 5 GB) or 8-bit Adam.
- One epoch of GSM8K's 7,473 prompts × 8 completions × ~350 tokens (non-thinking mode) ≈ 21 M generated tokens: at
  ~5–10k tok/s for a 1.7B model on an H100, 35–70 min of generation; training on ~30 M tokens × ~17 GFLOP (update,
  reference and old-policy passes) ≈ 5 × 10¹⁷ FLOP ≈ 20–30 min. **~1–1.5 h per epoch on one H100, ~$4–6.** Thinking
  mode (1–3k tokens per completion) multiplies the generation time 3–8×.
- **Two GPUs** (TRL `vllm_mode="server"`: one card generates, one trains): no sleep/wake cycling and a larger KV cache,
  roughly 1.3–1.7× faster for 2× the price unless generation and training overlap (asynchronous GRPO). On RunPod one
  H100 is the better buy; two cluster L40s on `normal` (free, 24 h) are the better buy when the node is free.
- vLLM pins its own torch and CUDA builds: it should live in a separate dependency group or venv, not the experiment
  venv (`uv sync` time on a pod grows by the vLLM wheel).
- The RL run itself needs no decomposition; splitting the RL update over the decomposition afterwards is workload (a)/(b).

## 5. Recommendation and a timing pod

1. **Convert once, on the cluster.** A CPU job on the worker (458 GB RAM; `cpu-run --mem 128G`, or an sbatch on
   `normal`) restores the Orbax checkpoint with the published JAX loader and writes bf16 safetensors: components,
   causal-importance function, and a manifest with sizes and sha256, to `commute-data/artifacts/qwen/` (format from
   the sibling loader report). The JAX loader stays in its own venv; experiments read the torch copy.
2. **Cluster first for what fits one L40**: the masked forward, the adversarial check, subset teacher training, LoRA
   GRPO. The data is on NFS (788 MB/s, no transfer). Propose to the user that longer runs may use `normal` when
   `sinfo -O gresused` shows free GPUs (today: all 8).
3. **RunPod for the rest**, with a **100 GB Standard network volume in US-MO-1** (H100 and L40S both in stock; US-CA-2
   second), ~$7/month. It holds the converted copy, the Qwen HF snapshot and RL checkpoints; the venv, uv cache and the
   tree stay on the container disk. Fill it once from a W&B artifact of the converted copy (uploaded from the
   cluster with the existing `WANDB_API_KEY`); the same artifact is the fallback for sessions run without the volume.
   Creating the volume in the team RunPod account is the user's call (cost and data-center pin).
4. **Cards**: H100 SXM for training and anything over ~1 h; L40S in the same data center for the check and forward
   passes; H200 (or the PRO 6000) only if full teacher training at 100% alive is wanted without changing the optimizer.
   No 4090/5090 for anything that loads the decomposition.

**Timing pod (one pod-runner, about $2 plus the volume):**
- Pod 1, L40S in US-MO-1 with the new volume, ~30 min, ~$0.55: create the W&B artifact beforehand (on the cluster);
  on the pod fetch it to `/workspace/qwen` (time it: write speed onto MooseFS), `dd` two 4 GB files back with
  `iflag=direct` (cold sequential read MB/s), time loading the converted copy into GPU memory from the volume and from a
  container-disk copy, and record peak memory of one no-grad masked forward at batch 1, 4, 8 if the loader exists.
- Pod 2, H100 in US-MO-1, started while pod 1 still runs, ~15 min, ~$1.00: confirms that two pods mount the volume with
  separate state dirs (no fencing), measures the read again from a second host, and times one teacher-training step
  at the largest batch that fits on a site subset, if the code exists.
- Pass: volume reads ≥ 200 MB/s, no lease error, both pods terminate cleanly. Fail on reads → keep the volume only
  as a cache to copy onto the container disk, or drop it for the W&B route.

**Changes needed (described, not made):**
- `remote-kernels.toml` (edit only with no machine running, then `/mcp` reconnect in every session):
  `network-volume-id = "<new id>"`, `data-center-ids = ["US-MO-1"]`; keep `workdir = "/root/work"` and the default
  `volume-mount-path = "/workspace"`; add `[env] COMMUTE_QWEN_DIR = "/workspace/qwen"`; `gpu-type-ids` for the Qwen
  clone = `["NVIDIA H100 80GB HBM3", "NVIDIA L40S"]` (what US-MO-1 stocks; the 4090/5090 entries cannot hold the
  decomposition); `container-disk-gb = 100` if pods copy the 27 GB onto local disk or run vLLM. The toml is checked
  in and shared by the clones, so the 67M line's sessions would inherit the pin unless the Qwen clone keeps its own
  copy uncommitted or the lines alternate.
- `cluster/pod_setup.sh`: make the 67M VPD fetch optional; add a Qwen step that checks `$COMMUTE_QWEN_DIR` against the
  manifest's sizes (sha256 of 27 GB on MooseFS would cost minutes) and otherwise fetches the W&B artifact; optionally
  copy to `/root/qwen` before loading.
- `cluster/env.sh` / `fleet/link-data`: `COMMUTE_QWEN_DIR` on the cluster pointing at the converted copy in
  `commute-data/artifacts/qwen`; a conversion script (CPU, JAX venv) under `cluster/`.
- `.claude/agents/pod-runner.md`: replace "No network volume: see the comment in remote-kernels.toml" with the volume
  rules (data-center pin; minimum 48 GB card for Qwen runs, H100 default; never fill the volume past 80%, since the
  plugin's state dir is on it; results are still `download()`ed, checkpoints may stay on the volume); a Qwen line in
  "Choosing where to run" if the user relaxes the `normal` rule; the W&B fallback when the data center is out of stock.

## Not checked

- Large-file read speed of a RunPod network volume: only RunPod's documented 200–400 MB/s; nothing measured.
- Whether RunPod bills an unattached network volume exactly as stated and whether `REMOTE_KERNELS_BUDGET` counts it.
- Whether two pods on one volume really avoid fencing in practice: inferred from the plugin source and its tests, not
  run. Whether `flock` on the lease works on MooseFS (the binary has a `NoFlock` error path).
- Which data centers offer the High-Performance volume tier.
- HF Hub, R2/S3 and parallel W&B download speeds on RunPod; W&B and HF storage quotas for a 27 GB upload; upload speed
  from the cluster to W&B.
- The causal-importance output head's exact shape, the storage dtype on disk, and the alive fraction of components
  (which decides whether full teacher training fits an H100): waits on the sibling loader report and one measurement.
- All throughput figures in section 4 are arithmetic from parameter counts and assumed efficiencies, not timings;
  the JAX loader and the torch conversion were not run.
- Cluster queue history beyond today's snapshot (`sacct` does not work from the controller).

Sources: RunPod [network volumes](https://docs.runpod.io/storage/network-volumes) and
[pricing](https://www.runpod.io/pricing); RunPod GraphQL API (stock, prices, volumes), queried 2026-10-27 14:35 UTC;
plugin source `~/.claude/plugins/marketplaces/alignment-hive/crates/remote-kernels/src/runtime/runpod.rs` (`dirs_for`,
config validation near line 950, tests near line 1890) and
`~/.claude/plugins/cache/alignment-hive/remote-kernels-x86_64-unknown-linux-gnu/0.3.9/skills/setup/references/runpod.md`.
