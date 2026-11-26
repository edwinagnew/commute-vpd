# Appendix to the P2.83 (E311) plan: departures, gates, inputs

Plan `2026-11-25_plan_p2-83-crosscoder-fix.md`. Paper: Minder, Dumas, Juang, Chughtai, Nanda 2025, "Overcoming
Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning", arXiv 2504.02922 (text
`~/commute-data/lit_review_pdfs/survey/2504.02922/paper.txt`; page numbers are the PDF's). Code: their trainer
`crosscoder_learning@d1fd6f3` (pinned in `pyproject.toml`), settings and latent scaling `diffing-toolkit@617d926`
(`~/commute-data/scratch/compare-lead/e203-review/dt`).

## A. The authors' recipe against E203 and E311, line by line

| item | authors (quote, page) | E203 | E311 |
|---|---|---|---|
| dictionary | "Expansion factor: 32, resulting in 73728 latents" (p. 34) | 32 × 768 = 24,576 (67M model) | same |
| sparsity | BatchTopK, "k = 100" (Table 3, p. 34); L0 ~100 (§3, p. 5) | k 100, annealed from 1,024 over 2,000 steps (toolkit `crosscoder.yaml`) | same |
| init | "Decoder initialized as the transpose of the encoder"; "paired with the same initial weights"; "BatchTopK ... norm of 1.0" (p. 34) | same | same |
| optimiser | LR 1e-4, 2 epochs (Table 3, p. 34) | Adam 1e-4, 1,000 warm-up steps, batch 2,048, 2 epochs (toolkit) | same |
| tokens | "100M tokens from Fineweb ... and lmsys-chat" (p. 34) | 2 epochs × 100M; Pile + the pair's task text, half each (key-value tasks) | same; e53/e62 sides now fill 100M an epoch with Pile rows (E203's ran out at ~half) |
| site | "the output of the model.layers[13]", the middle of 26 (p. 34, fn. 14) | block 2's input = block 1's output, the residual stream at the middle of 4 blocks | same |
| precision | the pinned trainer in fp32 | fp32 | fp32 (`tf32: false`) |
| normalisation | per-model mean/std to a target RMS (toolkit) | same, fitted on the first ~1M-token chunk | same |
| chat-only set | Δnorm bands, chat-only 0.9-1.0 (p. 3); the analyses take the top N by Δnorm (3,176 in §3.1; 5,000 in the toolkit) | top N = 1,667 (5,000 scaled by dictionary size), 5,000 also reported | same |
| latent scaling | β by least squares on fᵢ d_chat (Eq. 4, p. 5); ν_ε = β^ε_base/β^ε_chat, ν_r = β^r_base/β^r_chat (p. 5); "chat-specific ... ν_ε < 0.2 and ν_r < 0.5" (Fig. 1 caption, p. 4); drop negative base β (App. E.2) | the toolkit's closed form (targets: error with the latent's own contribution added back), 100k tokens; negative-β exclusion added by the P2.70 rescore | `crosscoder.score`, with the exclusion |
| the reading d | none in the source | shared part − fine-tune-specific part of the change | same |
| planted control | none in the source | ours: a random direction at the median answer-position norm, at the answer position of every task window | ours: the same direction, placed at content-independent positions (1 in 80), or on half the windows; E203's placement kept as a diagnostic |

Every setting is E203's (`experiments/crosscoder/results/run2/predictions.json` "sources" names each one's file and
line). The departures that bear on the control are the last two rows: the control is ours, and its first placement
tied the plant to a base feature, which App. D (p. 25) says BatchTopK merges ("For the subset of tokens where f_shared >
0, the three-latent solution will have an L0 sparsity of 1, while the merged two-latent solution will have an L0
sparsity of 2. Since the BatchTopK crosscoder optimizes for L0 sparsity, it will prefer the three-latent solution").

## B. Gates (gated: run 1 + run 2 ~$250 > $60)

1. **Pre-mortem.** Recipe: E203's, unchanged; the new module re-expresses E203's stream, checked by seed-0 states
   landing within the reseed gap of E203's d. Data: no task window is read or scored in training (E203's row split:
   Pile half from row 32,768, fresh key-value windows from rows 131,072-196,607; e53/e62's longer Pile half runs to row
   228,351 and these sides draw no pool windows; `stream_text` asserts the key-value bound). Precision: fp32 as E203.
   Memory: E203 peaked at 14.9 GB GPU (`timing.json`); host ~2 GB of Pile rows (the long file is 801 MB as int64).
   Time and cost: plan tables, from E203's measured minutes. Saved: per side `results/sides/<key>.json` (~100 KB) and
   `results/large/<key>.pt` (~0.5 MB; stage 0 adds the 75 MB decoder); download each side's JSON as it lands. Pod loss:
   every side checkpoints every 10,000 steps under `results/large/ckpt/` and resumes; a lost shard is rerun with the
   same `2 i/n` (finished sides are skipped). Controls: self first, alone (pod a), on the same card type. Predictions:
   filed with a value for every stage. Code paths never run on a GPU: all of `crosscoder_sides.py` and `run.py` (the
   rehearsal runs every one).
2. **Settings sourced:** `results/predictions.json` "sources" and "sizes_source".
3. **Rehearsal:** `cluster/crosscoder_fig4_rehearse.sbatch` (`--comment='E311 rehearse: full sizes, tiny dict'`):
   every stage, full read sizes, a 1,536-latent dictionary; the resume test; two stage-2 shards (an e53, an e62, a
   key-value state and a Gaussian update). Its time per item, peak memory and resume floor go into this file before a
   pod starts.
4. **Early checks in code:** batch FVE ≥ 0 at step 1,000; training minutes projected from the measured ms per step ≤ 75
   (260 for the long arm); FVE ≥ 0.5 and dead ≤ 0.3 at the first snapshot; per stage-2 side FVE and dead at the end.
5. **One review** (plan + code, fresh Opus): the lead's.
6. **Go-ahead:** whip-lead (in-plan gated run).

## C. Inputs (what a pod needs beyond the synced code)

| file | size | who |
|---|---|---|
| `artifacts/vpd/` (W&B fetch on the pod) | 3.0 GB | all |
| `artifacts/sets/train_32768_195584.npy` (made by `run.py prep`, done 2026-11-25) | 801 MB | all (key-value sides read the 98,304-row slice of it when `train_32768_98304.npy` is absent) |
| `artifacts/sets/train_131072_65536.npy` | 268 MB | all (the fresh-window pool) |
| `artifacts/sets/val_0_8192.npy`, `val_0_256.npy` | 34 MB, 1 MB | all (FVE rows; E53's windows) |
| state files (`run.py inputs i/n` lists a shard's) | 85 MB each, 4,417 MB for all 51 | per shard |
| task windows (13 files listed by `inputs`) | < 5 MB each | per shard |
| stage 2: `results/stage0a.json` from pod a | small | every run-2 pod |

Upload per run-2 pod: ~1.1 GB of sets + ~170 MB of states ≈ 1.3 GB ÷ 10 MB/s ≈ 2.2 min.
