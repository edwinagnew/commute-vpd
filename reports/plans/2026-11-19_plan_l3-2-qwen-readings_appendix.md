# Appendix: plan for rung L3.2 (Qwen readings)

*Companion to `2026-11-19_plan_l3-2-qwen-readings.md`. Sources for every number are named in place.*

## Expected values

| Number | W0 / do-nothing | Positive control | Random change of matched size | Source |
|---|---|---|---|---|
| Share, p-015e0710, maths rows | K00 = K01 = K0 exactly; do-nothing share 1 | (the do-nothing is the share's control, as in E179) | 0.81-1.06 on the library row sets, band half-widths 0.023-0.077 | E179, E186 `summary.json` |
| Share status, own-samples SFT | — | — | — | unscored expected: KL 0.0043-0.0049 < floor 0.05 (`qwen_reading.FLOOR`) |
| S (reading 3) | KL(F‖M) = 0, S = 0/0, reported "no change"; H = M to rounding (67M floor 2.4e-15) | output-matrices-only change: S = 1 to its measured fp64 floor | no Qwen prior; 67M rebuilt items 0.56, grafts 0.13-0.22 | E197, E201, E206 |
| S gap, library reuse minus new side | — | — | — | 67M: 0.005-0.011 above seed sd, twin-fact 0.258 (E197); rescaling the larger side lowers its S by 0.0006-0.030 (E205) |
| Effective rank | undefined (zero update) | rank-1 reads 1, ten equal directions 10 | 1,812-1,844 (Gaussian floor at Qwen updates' norms) | E141, P3.6 `run2_stage3.json` |
| Effective rank, RL | — | — | — | 1e-6 step-100 RL 1,362 (seeds within 12), null 1,381 (P3.6) |

## State inventory (all F32, 196 tensors, 5.64 GB each, HF layout, δW = W − W0; no embeddings or norms saved)

- E167 step 75: `~/commute-data/results-large/qwen_rl_readable/run2/{R,N,F}_s{0,70}/dW/step_75.safetensors` (6 files,
  33.8 GB; also `{R,N,F}_s70/step_250`). On dev disk only, not on W&B. sha256 in `results/run2/<pod>.json`.
  Per-state reads (`results/run2_reads/states/*.json`): |δW| R 0.867/0.892, F 0.626/0.599, N 0.753/0.699; KL to W0 on
  GSM8K R 0.195/0.285, F 0.0043/0.0049, N 0.392/0.377; held-out pass R 0.88, F 0.82/0.81, N 0.04/0.00.
- E69 1e-6: `~/commute-data/results-large/qwen_rl/lr1e-6w10/dW/{R,N,F}/step_100` (+ R step 10, 50). Not read here.
- Library (E165 run 3 `p35r3-*`; gold pair from run 2 `p35-*`): on W&B only; `pull.py` knows `p35-` only, so run 2
  reuses compare-lead's fetch in `~/commute-agents/compare-lead/experiments/size_matched/qwen.py` (P2.28 fetched 34 GB
  a pod at 30-45 MB/s).
- p-015e0710: `artifacts/qwen3-1.7b-p-015e0710-torch` (30 GB), W0 g statistics for maths rows in
  `experiments/qwen_library_share/results/large/p-015e0710/gstats/W0_maths.safetensors`. On a pod: W&B, ~8 min (P3.9).

## Cost arithmetic

- **S per state.** Reading 3 needs G_MM and C_FM per site; a 6,144-wide MLP site costs 2 · N · 6,144² flops, 28 sites
  → 2.1e12 · N. At N = 16,384 fit positions: 3.5e16 flops, ~10 min at ~60 TFLOPS (H100 fp64 tensor). Two fp64 Grams
  of an MLP site are 0.6 GB; computed one layer at a time. Not measured: run 1 measures it and logs the estimated total.
- **Run 1.** Upload 33.8 GB ÷ 10 MB/s = 3,380 s ≈ 56 min (overlaps decomposition fetch 31 GB ÷ 30-45 MB/s ≈ 12-17
  min and set-up); share 15 reads × 105-225 s (E179) ≈ 45 min; S 9 reads × ~10 min ≈ 1.5 h; rank ~15 min. ~4 h at
  $3.99/h ≈ $16. Download: JSONs under 50 MB, seconds.
- **Run 2.** Per pod 12 states × 5.64 GB = 68 GB from W&B at 30-45 MB/s ≈ 25-38 min, overlapped; 18 reads × ~10 min =
  3 h + set-up 30 min ≈ 3.5 h, $14; three pods $42. No upload but code (seconds).
- **Stop rule.** If run 1's measured S time per state × 18 exceeds 5 h, run 2 drops the 5x save's rescaled reads (6 a
  pod) and says so.

## Pre-mortem (short; the runs are not gated)

- **Fit noise in S.** 2.7 fit positions per MLP input on library rows; the S of the random change and of RL share rows
  and fit size, so a common drop shows in both. If the positive control's held-out residual is above 1e-3, more rows
  (the library's 128 rows are fixed; maths rows are not) or a reading at fewer positions is the lead's call, not a
  new rule.
- **Share unscored on RL.** Measured in stage 0 as K00 − K0 for RL restricted to the MLP; reported "unreadable" if so,
  no rule changed.
- **Memory.** Share peaks at 38.6 GiB (P2.28 plan); S holds two bf16 models (3.4 GB each) and one layer's fp64 Grams.
- **Pod loss.** Each state's JSON is downloaded as written; the final summary's sha256 is checked before terminate.
- **Precision.** S's Grams accumulate in fp64; its floor is the positive control's distance from 1.

## Flag 4 in detail

The user's acceptance said the p-9707d6ff share "is reported from existing results". Existing p-9707d6ff shares are
E132 (E40 step-10 SFT, gold, anchor 2 seeds, two random changes: 0.991-0.9992) and P3.8 (the library: 0.990-0.9998,
random 0.992-1.009). None read the E167 step-75 RL, null or own-samples states. Reading them would add one share pass
on p-9707d6ff (47 GB decomposition) to run 1; given P3.8's saturation it would almost surely read ~0.99 again.
