# Appendix to the P2.31 run 2 plan: literature, source lines, pre-mortem, cost

Plan: `reports/plans/2026-11-17_plan_p2-31-run2.md`. Downloaded source files (untrusted, read only) are in
`~/commute-data/scratch/compare-lead/p231-run2/src/`.

## A. The literature, head to head

| Method | Comparison it rests on | Can it give "share of the change on shared vs fine-tune-specific latents"? |
|---|---|---|
| **BatchTopK crosscoder + latent scaling** (Minder, Dumas, Juang, Chughtai, Nanda 2025, arXiv 2504.02922 v4, NeurIPS) | Against the L1 crosscoder on Gemma 2 2B, layer 13: more fine-tune-specific latents at every ν threshold (Fig. 2c, top 3,176 by relative norm against L1's 3,176 chat-only); ν_ε of BatchTopK chat-only latents near 0 against up to ~0.5 for L1, no decoupling overlap (§3.1); relative norm correlates with ν_r 0.73 and ν_ε 0.87; top 50% by relative norm cut the KL gap 50%, bottom 50% 6% (§3.2, Fig. 4). FVE 87.6%, L0 99.5, dead 12% (Table 3). | **Yes.** The split is learned, not imposed. |
| Diff-SAE (Dumas, Minder, Nanda, "What we learned trying to diff base and chat models", AF, 2025-06-30) | Same Gemma pair: diff-SAE reconstruction recovers 88% of the KL gap, crosscoder 60%; crosscoder error also recovers 60%. They call it "unclear if crosscoders should be the default". | No shared side; it would need a second assignment step not in the source. |
| Delta-Crosscoder (arXiv 2603.04426, 2026) | 10 model organisms: finds the objective in 10/10, against BatchTopK crosscoders 4/10 and DSF 6/10 (§6.1). | No. 80% of latents are non-shared and shared latents are masked out of the change loss (Eq. 9), so the shared share is near 0 by design. No code released. |
| Dedicated-feature crosscoder (Jiralerspong and Bricken, arXiv 2602.11729, 2026) | Toy: more exclusive concepts recovered, more false positives (Fig. 4); real: FVE 0.817 for both (Table 1). | Partly. Exclusive partition sizes are fixed (1-5%) and set the split. Tested only across architectures. |
| Diff-SAE vs L1 crosscoder (Kumar, arXiv 2605.07324, 2026) | Backdoor in SmolLM2-360M: diff-SAE BIS 0.40 vs crosscoder 0.01 (Table III). | No shared side; L1 crosscoder only, 250k training tokens. |
| Activation difference lens (Minder et al., arXiv 2510.13900) | Interpretability-agent scores; not a dictionary. | No. |
| Lindsey et al. 2024 (Anthropic, crosscoders) | Claude 3 Sonnet base vs chat, 1M features: a trimodal relative-norm histogram, 4-5k model-specific features per model; "the majority of the model-exclusive features are not immediately interpretable". No numeric band. | Origin of the relative norm; no artefact test. |

**Ranking for this reading.** The BatchTopK crosscoder with latent scaling is the best-evidenced method that can give
the reading at all. Its known failure modes, all to be carried into the write-up: (1) a prior toward shared latents (the
joint loss favours them, so a fine-tune-only latent must be disproportionately useful to form: Dumas et al. 2025;
Delta-Crosscoder §3.2; dedicated-feature crosscoders §3.2). The shared part is therefore an upper bound and the specific
part a lower bound. (2) The error carries as much behaviour as the reconstruction (60% and 60%), so the error part is
reported beside d. (3) Complete shrinkage and decoupling, the L1 artefacts, are what latent scaling removes. In BatchTopK
they are rare but still tested.

**The relative-norm threshold.** The 0.9 band is Minder's Table 4 / glossary label. For BatchTopK only 134 of 73,728 latents
fall in it (Table 4). Both their analyses select the top N by relative norm instead: 3,176 in the paper (§3.1, N matched to
L1's count), 1,844 and 2,442 for the Llama models (App. I.1), and 5,000 in the toolkit (`crosscoder.yaml:78`,
`closed_form.py:365`, "sorting by dec_norm_diff ... taking the head"). The toolkit's band labels (`analysis.py:104`,
`treshold = 0.1`) are kept only as tags. So run 1's stop came from a departure, not from the method.

## B. Which parts are theirs (pinned) and which are ours

| Part | Source |
|---|---|
| Model: `BatchTopKCrossCoder` (encode with θ, decode, decoder norms, per-model mean/std normalisation) | `crosscoder_learning@d1fd6f3` `dictionary_learning/dictionary.py:1447`, normaliser `:28-140` |
| Training step: `BatchTopKCrossCoderTrainer.update` | same commit, `trainers/crosscoder.py`: Adam (0.9, 0.999) `:414`, warm-up schedule `:417`, θ EMA 0.999 from step 1,000 `:334, :600`, dead after 10M tokens and aux k = d/2 `:398-399`, aux normalised by variance `:486`, unsquared L2 loss `:617`, decoder bias at the geometric median `:655`, gradient clip 1.0 `:663` |
| Settings | `diffing-toolkit@617d926` `configs/diffing/method/crosscoder.yaml:18-78` (32×, batch 2,048, 2 epochs, lr 1e-4, k 100, k_max 1,024 over 2,000 steps, target RMS 100, init 1.0, transpose, warm-up 1,000, latent scaling on 100k tokens, top 5,000); `configs/config.yaml:123-126` (middle layer, 50M train tokens per dataset). Paper repo `sparsity-artifacts-crosscoders@d67ec6a` `run/gemma2_2b/train-crosscoder.sh:6-39` agrees on batch, lr, epochs, 100M tokens. |
| Latent scaling | ported from `diffing-toolkit@617d926` `src/diffing/utils/dictionary/latent_scaling/closed_form.py:40-168` and `utils.py` (targets with the latent removed); the package cannot be installed (vllm, streamlit, openai, hydra). Checked in rehearsal against `torch.linalg.lstsq`. ν thresholds 0.5 and 0.2 from the paper's glossary. |
| Ours | activation stream (67M model, block 2's input, position 0 dropped); data mix; per-model normaliser fitted on 1M tokens (toolkit `training.py:113`); top-N selection; the change reading d computed after undoing the normalisation (so the per-model mean shift is its own reported part); the self, planted, random and reseed controls; stages, stops, files. |

`pyproject.toml`: `"dictionary-learning @ git+https://github.com/science-of-finetuning/crosscoder_learning.git@d1fd6f3fea41fc1123699cc09f4cdad47ca35160"`.
Reference clones (read only, outside the repo): `~/commute-data/ref/crosscoder_learning@d1fd6f3`,
`~/commute-data/ref/diffing-toolkit@617d926`.

## C. Pre-mortem (gate 1)

- **Recipe.** A mis-wired trainer (the wrong axis order (batch, 2, d), or normalisation applied twice). Guard: the self control
  must give relative norms of exactly 0.5 and FVE ≥ 0.80; rehearsal compares one `update` against a hand step.
- **Data.** Task windows leaking from the score half into training. Guard: the generator draws from Pile rows the score
  windows do not use, and a hash check runs before training. A task with too few fresh windows is filled by repetition, with the count reported.
  Position 0 is dropped in training and reading.
- **Precision.** fp32 as in the source. The model forward is fp32 with TF32 off, as in run 1.
- **Memory.** 75M parameters plus Adam is about 1.2 GB; two crosscoders per card fit an 80 GB H100 with room to spare.
  Host RAM: the 100M-token id arrays are 0.4 GB as int32.
- **Time and cost.** See D; the projected set is chosen from stage 0's measured minutes.
- **Saves.** Per side, about 0.5 MB of per-latent vectors in `results/large/` and the summary JSON in `results/sides/`.
  Decoders in fp16 for the 6 control crosscoders (75 MB each). Each file is downloaded as it lands (Pile ids pre-cached
  once by a cluster CPU job, about 0.4 GB, about 40 s to sync).
- **Pod loss.** Each side is its own file and the stage is resumable. A lost pod costs at most one crosscoder (~38 min).
- **Controls.** The self, planted, random and reseed controls run on the same card class as the pairs.
- **Predictions.** P1/P2 as filed for run 1, re-filed under the new E-number with the top-N rule. Edge cases: zero
  candidates pass ν (specific part 0, d = shared part); ties at rank N (break by index); a change of zero energy (d is nan).
- **Never run on a GPU.** The pinned trainer on our stream, the ported latent scaling and the normalisation round trip. All
  three run in the $0 rehearsal.

## D. Cost arithmetic

Run 1 (`stage0.json`, H100 80GB HBM3, $4.41/h): 9,086 steps × 4,096 tokens = 37.2M tokens in 4.75 min. The faithful run
is 100M tokens × 2 epochs = 200M tokens, so 4.75 × 200 / 37.2 = 25.5 min. Times 1.1 for batch 2,048 (fixed per-step
optimiser cost), times 1.3 for fp32 without TF32 (both are guesses, measured in stage 0), plus 2 min for the normaliser,
latent scaling and reading = **~38 min** ($2.80). With TF32: ~32 min ($2.35). Pod set-up ~10 min each (5-8 min measured
for a fresh 67M pod, E142/E153, plus ~1 GB of states at 10 MB/s). 20 pods add 3.3 h of billed set-up.

- 112 crosscoders: 112 × 38 / 60 = 70.9 h + 3.3 h = 74.2 h → $327 (TF32: 63.0 h → $278).
- 91 (size-matched kept for e46 c 0.66, e53 0.90, e54_own 0.52, e58_copy_prior_cipher 0.78, e58_rank1_96 0.82, e62 0.83: 19 states):
  60.9 h → $269 (TF32: 51.8 h → $229).
- 76 (at most 3 seeds a side: 46 states, 26 size-matched, 4 controls): 51.4 h → $227 (TF32: 43.8 h → $193).

Wall time, 20 pods: stage 0 ~40 min, stage 1 ~40 min (4 pods), then ceil(106 / 20) = 6 crosscoders a pod = 3.8 h, so about 5.3 h
for the full set. If two crosscoders per card give 1.6× the throughput, the full set falls to about $210 and fits.
