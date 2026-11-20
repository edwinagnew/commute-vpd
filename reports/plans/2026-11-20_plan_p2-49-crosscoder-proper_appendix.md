# Appendix to the P2.49 plan: settings and sources, pre-mortem, cost, items

Plan: `reports/plans/2026-11-20_plan_p2-49-crosscoder-proper.md`. Sources: the pinned trainer in the venv's
`site-packages/dictionary_learning/` (crosscoder_learning@d1fd6f3); the toolkit as cloned for E203's review,
`~/commute-data/scratch/compare-lead/e203-review/dt` (diffing-toolkit@617d926); the paper repo's scripts, downloaded,
`~/commute-data/scratch/compare-lead/p231-run2/src/` (sparsity-artifacts-crosscoders@d67ec6a). Literature ranking and
part-by-part provenance: `reports/plans/2026-11-17_plan_p2-31-run2_appendix.md` A and B (unchanged).

## A. Settings (gate 2) and every difference from E203

| Setting | Value | Source |
|---|---|---|
| dictionary | 32 × 768 = 24,576 latents, BatchTopK, k 100 annealed from 1,024 over 2,000 steps | `crosscoder.yaml:18-27`, `:65-66` |
| optimiser | Adam lr 1e-4, 1,000 warm-up steps, batch 2,048, aux α 1/32, dead after 10M tokens, θ EMA from step 1,000 | `crosscoder.yaml`; trainer `crosscoder.py:398-663` |
| init and normalisation | norm 1.0, transpose, same init; per-model normaliser to RMS 100 on 1M tokens | `crosscoder.yaml:47-49, 66-69`; toolkit `training.py:113` |
| tokens | 100M unique × 2 epochs = 97,656 steps, every side | paper `train-crosscoder.sh` (--num-samples 100_000_000 --epochs 2) |
| **matmul precision** | **TF32 ("high") in the crosscoder; model forwards stay fp32 with TF32 off** | paper `train_crosscoder.py:23` `set_float32_matmul_precision("high")`; toolkit `config.yaml:141` `torch_precision: high`. E203 and E212 ran fp32, calling it the source's (wrong); latent scaling stays "highest" (`compute_scalers.py:21`) |
| latent scaling | 100k training tokens, closed form, nu_r < 0.5 and nu_eps < 0.2 on the top 1,667 by relative norm (5,000 beside) | `crosscoder.yaml:74-78`; paper glossary; 5,000 × 24,576 / 73,728 |
| shared band | relative norm in (0.4, 0.6), open | toolkit `analysis.py:104-110` |
| site A | block 2's input | `config.yaml:123` `layers: [0.5]` |
| site B | the last block's output (`out3` in `steering.Net`) | ours: the grafts write in layer 3 only (E206), so site A cannot see them; the cell asks for a second site |
| k sweep | 32 and 200 beside 100, site-A head, seed 0 | ours: 32 keeps the paper's k/d (100/2,304 × 768 ≈ 33); 200 doubles it |
| Gaussian | elicit s0's per-matrix norms (`finetune.random_direction`, draw 0), one scalar c set by bisection so the mean squared activation change per token at the site equals elicit s0's within 5%, on its read rows | the cell ("matched on per-matrix norms and on activation change"); E228's `perturb.match` does the same on KL |
| grafted heads | A: E206 `early_lowest.pt` (layer 1, k = 16); B: E206 `last_lowest.pt` (layer 3, k = 16) | `~/commute-data/results-large/graft_sites/`; attached by `commute.graft` |
| scored grafts | E206 `last_rank5`, `last_rank9`, `last_rank13` (heads) and `neurons` (256 MLP-3 neurons), 2 crosscoder seeds each | E206 (P1.28), every graft that passed its build checks at layer 3 |
| stream | per side, as E203: Pile half from train row 32,768 plus the side's own task text (fresh windows for key-value tasks from rows 131,072+); **e53 and e62 keep ≤ 100 draws a window and the rest of their task half comes from Pile train rows 589,824+**; grafts and the self control use the e54 elicit stream (E203's self) | E203 `results/run2/predictions.json`; the fill is the fix the cell asks for (six sides stopped at step 49-52k of 97,656) |

Differences from E203's end state, each removed or named: precision (to the authors' TF32); stream length (filled,
asserted); seeds (3 a side, not 1); planted change (grafted heads, E203's direction kept as a reported arm); the
Gaussian (matched on activation change); the site (B added); size-matched where c ≥ 0.95 (not retrained, measured
within 0.015); everything else unchanged.

Departures from the literature (carried to the write-up): the 67M model, so 24,576 latents; the pair's task text
replaces the chat half; top 1,667 scaled from 5,000; site B; the reading d and every control are ours; 4.2M val tokens
against 5M; the e53/e62 task share stays small (0.8%, 5.9%).

## B. Pre-mortem (gate 1)

- **Recipe and hyperparameters.** TF32 could change the planted reading. Guard: the stage-0 heads read in TF32, and the
  timing stage logs batch FVE at matched steps in fp32 and TF32 on one side. A mis-wired graft at a site would show
  as a zero change. Guard: graft energy > 0 is checked before training, and the B head's change at A is exactly 0, a
  printed check.
- **Data.** Fill rows overlapping a read or scored window. Guard: the fill starts at row 589,824, past every cached set
  a task cut from (rows 0-32,767; pool 131,072-196,607; E203's 65,536-589,823 sets are not used). `run.py` asserts
  disjoint ranges. The graft windows' source rows are read from `commute.graft` and asserted disjoint from the Pile
  half. Stream tokens ≥ steps × batch is asserted before step 0. Each side records its task share. Position 0 is
  dropped. No chat template (no chat model). The tokeniser is the 67M's own.
- **Precision.** Model forwards fp32 with TF32 off (a zero update must reproduce W0 within 1e-5, E212's bar). Crosscoder
  TF32. Latent scaling in fp32 "highest".
- **Memory and host RAM.** E203 peaked at 14.9 GB of GPU. A 1M-token chunk of pairs is 6.4 GB on the GPU. Host: the
  Pile half 0.40 GB plus the pool 0.27 GB plus the fill 0.40 GB mapped. Measured again in the rehearsal, with two
  processes per card in the timing stage.
- **Time and cost per stage.** Section C. Per-job projected-time check at 12.5% of tokens, bar from the rehearsal and
  timing (gate 4).
- **Saved.** Side JSON (≤ 0.2 MB) in `results/sides/`. Per-latent vectors and the decoder (bf16, 75 MB) go in
  `results/large/`; run 1 also keeps the encoder (151 MB). Each is tarred and downloaded as it lands, sha256 checked
  both sides, into `results/<run>_dl/` and moved into `large/` after the pod is gone (root CLAUDE.md, E36 lesson).
  Run 1: 37 × 151 MB = 5.6 GB. Run 2: 76 × 75 MB = 5.7 GB.
- **Downloads.** One file per `download()` call. Each item's tar downloads while the next trains.
- **Pod loss and resume.** A checkpoint every 10,000 steps plus each snapshot. A lost pod costs at most one crosscoder
  (about 30 min). The rehearsal's resume is compared step by step against an uninterrupted run, against the floor of
  two uninterrupted runs.
- **Controls and references.** Every control runs on the same card class and stream as the pairs. The primary pair's
  sides are read at both sites. The Gaussian is matched at each site separately.
- **Predictions.** Filed with every rule's edge cases (ties, all zero, an empty candidate pool, nan for no change, a
  missing side). Tested by hand in predictions.json "edge_case_checks".
- **Code never run on a GPU.** The graft attachment inside the crosscoder stream, the site-B grab, the Gaussian scale
  search on activation change, the Pile fill, spliced loss, and TF32 toggling around the trainer. All run in the
  rehearsal.

## C. Cost arithmetic

Per crosscoder at site A: 97,656 steps × 12.3 ms = 20.0 min (E203 `timing.json`, H100 80GB HBM3, TF32), plus 7.9 min
outside training (E203 side 48.9 min total against 41.0 fp32 training) = **~28 min**. Site B: the forward through
blocks 2 and 3 for 2 × 200M tokens is about 1.1e16 FLOP more in fp32, about 4 min at ~50 TFLOP/s, so **~32 min**
(estimate; measured in the timing stage).

Transfers per pod: Pile half 0.40 GB + pool 0.27 GB + states (≤ 4 × 0.113 GB) 0.45 GB = 1.12 GB, ÷ 10 MB/s = 112 s;
pods with e53/e62 items add the fill (0.40 GB, 40 s) = 152 s. The decomposition (3.0 GB) is fetched from W&B on the pod
(65-100 s measured, root CLAUDE.md) or synced (300 s). Downloads: 151 MB per run-1 item (15 s), 75 MB per run-2 item
(8 s), overlapping the next item. Set-up is counted as 10 min a pod.

- Run 1 stage 0: 2 self (A 28, B 32) + 3 × 28 + 3 × 32 = 240 min; 8 pods × 10 min set-up = 80 min; 5.3 h × $4.0-4.4 =
  **$21-24**, plus about $1 for the timing stage.
- Run 1 stage 1: A 3 Gaussian + 10 pair + 1 direction + 2 k = 16 × 28 = 448; B 3 + 10 = 13 × 32 = 416; 864 min;
  7 new pods × 10 = 70; 15.6 h → **$62-69**.
- Run 2: A 68 × 28 = 1,904; B 8 × 32 = 256; 2,160 min; 19 pods × 10 = 190; 39.2 h → **$157-172**.
- Wall: stage 0 ~45 min; stage 1 two items a pod ~70 min; run 2 four items a pod ~2.3 h.

## D. Items

Primary pair: e54_matched, reuse `e54/elicit_s{0,1,2}`, new `e54/cipher_s{0,1,2}`.

Run 2, site A, as built (3 seeds a side, saved seeds first-in-order; 2 where only 2 exist): e62 copy/lookup (3/3), e46
ident32/rand32 (3/3), e53 lowercase/cipher (2/3), e54_own new side cipher_own (2; its reuse side is the primary's
elicit), c3 rank1/copy/cipher (3/3/3; cipher is the new side of open_pair_c3 and open_pair_c3_copy), e58
rank1/cipher/copy/prior_cipher (3 each): 40. Twins (2 a side): e123_case_label, e125_repeat_label, e58_elicit_shuffle,
e58_class_twin: 16. Size-matched (the larger side × E187's c, 2 seeds): e46 rand32 (0.66), e53 cipher (0.90), e54
cipher_own (0.52), e58 prior_cipher (0.78), e58 cipher for rank1_96 (0.82), e62 lookup (0.83): 12. Site B: four grafts ×
2 seeds: 8. State names and inventory come from `random_kl_share.run.catalogue()` and E187's `size_matched.run`.
