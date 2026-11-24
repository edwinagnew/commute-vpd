# P2.73: WeightWatch, the SVD of the weight difference, on our items (method spec)

*definition-lead, 2026-11-23. Cell P2.73 (open, 0 of 2 runs used, due 2026-11-25). Step 1 of the cell: this spec.
No code was written and nothing was run. redteam-lead and an independent GPT reviewer check it before any code. Every
step is the published one, quoted with its location in the appendix
(`2026-11-23_plan_p2-73-weightwatch_appendix.md`, section A), or a named departure in the table below.*

**Plain words.** Zhong and Raghunathan take the difference between a fine-tuned model's weights and its base model's,
keep the top 20 output directions of that difference in two matrices per layer, and flag any input whose hidden state
points along one of them more or less than ordinary text ever did. That is a detector of fine-tuned behaviour, and it
gives no reuse score. We run it as published on our built edits, E62's copy and lookup fine-tunes and the realistic
panel, and beside it one score of our own: do the update's top directions lie where the original model already writes?

## Source

| Source | Identity | Pinned at |
|---|---|---|
| Paper | Ziqian Zhong, Aditi Raghunathan, "Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs", ICLR 2026, arXiv 2508.00161v3 (21 Apr 2026). Digest: `lit_review/digest/papers/zhong2026-watch-weights.md` | v3 PDF; page numbers below are the PDF's |
| Code | github.com/fjzzq2002/WeightWatch (linked on the paper's first page); clone `artifacts/p273_src/WeightWatch/` (read only, never run) | `8d3874fb90a0aad5478514f96d71779d2aedf5aa` (2025-09-25) |

## The published procedure, in five lines (quotes in appendix A)

1. **Matrices and k.** For every layer, dW = W_post − W_base of the attention output matrix (`o_proj`) and the MLP
   `down_proj`; full SVD in fp32 (`torch.linalg.svd`, `full_matrices=False`); the top k = 20 left singular vectors of
   each, 40 per layer. Singular values only rank them.
2. **Where they are read.** Each layer's 40 vectors against the residual stream after that whole layer
   (`hidden_states[l + 1]`), of the fine-tuned model, every token, fp16.
3. **Statistic.** Per token, cos(a_t, u) = u · a_t / |a_t|; per input, the minimum and maximum over its tokens, per
   direction.
4. **Calibration.** Ranges [c_min, c_max] per direction are the running minimum and maximum over 50,000 generic inputs
   (2/3 WildChat, 1/3 UltraChat), 50 tokens of each completion generated at temperature 1 (not for RMU or circuit
   breaker), user and assistant tokens kept apart, the last three layers left out; FPR on a 9:1 time-ordered split.
5. **Flag rule.** An input is flagged if, for any direction, s_min < c_min − 0.01 or s_max > c_max + 0.01. The output is
   the share of anomalous inputs flagged and the false-positive rate on held-out generic inputs.

**The published output is a detector, not a reuse score.** The paper's claim is that the directions "correspond to newly
acquired behaviors" (abstract). It never asks whether a direction was already in use.

## What we run

**Detector, as published.** For each item, F = M + dW. Calibrate on generic Pile text, then report the share of the
item's own task windows flagged and the FPR on held-out Pile. A matched random dW is read the same way. Our task windows
are synthetic and may look unusual to any direction, so the random's flag rate is this reading's floor.

**Our score, "base-use ratio after Zhong".** Take the same kept directions u_{m,i}, with singular values σ_{m,i}, for
monitored matrix m. Let C_m = E[(W_m x)(W_m x)ᵀ] be the second moment of M's own output of that matrix over the
calibration tokens, with d = 768. Then

  a_{m,i} = d · u_{m,i}ᵀ C_m u_{m,i} / tr(C_m),   S = Σ_{m,i} σ²_{m,i} a_{m,i} / Σ_{m,i} σ²_{m,i}.

a = 1 means M's matrix writes along u as much as along an average direction; above 1, more. S weights each direction by
its share of the update. **Random floor:** S of `commute.finetune.random_direction(dW, seed)` (a Gaussian per matrix at
dW's per-matrix norm), three filed seeds per state. Left singular vectors of a Gaussian matrix are uniformly distributed,
so the floor's expectation is exactly 1; the three draws give its spread. S is a reading: no band, no verdict and no
reuse or new label (the definition is under revision in P1.1).

## Departures

| # | Published | Ours | Reason |
|---|---|---|---|
| D1 | Last three layers excluded | All four of the 67M's layers; the published exclusion (layer 0 only) reported beside | On a 4-layer model the rule leaves only layer 0 and is blind to every built edit at layers 1-3. Qwen (28 layers) keeps the rule |
| D2 | All 20 vectors kept, whatever σ | Only σ_{m,i} above the measured floor (stage 0: largest σ of a do-nothing build, F − M on the item path at gain 1) | Vectors with σ = 0 are any orthonormal completion, LAPACK's choice. x2 has 8 non-zero σ in one matrix; dW = 0 has none. Full fine-tunes are expected to keep all 20 (checked) |
| D3 | WildChat and UltraChat chats | Pile rows from `artifacts/sets/train_65536_524288.npy`, 50,000 windows of 128 tokens, in row order | The 67M is a Pile base model with no chat format. Windows of 128 tokens, not conversations up to 1,500 characters, because our test windows are 8-41 tokens (cost) |
| D4 | 50 generated tokens at temperature 1 | No generation: fixed windows | Our readings are on fixed windows. The paper itself skipped generation for RMU and circuit breaker |
| D5 | Separate user and assistant ranges | One range set | No roles in a base model |
| D6 | fp16 | fp32, TF32 off, on the 67M | fp16 was chosen "for computational efficiency"; the 67M is cheap. Qwen: bf16, as the Qwen line loads it |
| D7 | Algorithm 1 widens ranges on every monitored input | Ranges fixed after calibration; test reads do not widen them | The released code reads test inputs with `read_only=True` (default); this also makes the rate order-free |
| D8 | (none) | The score S and its floor | Ours, labelled "after Zhong" |

## Items, and what each should read

| Arm | Detector: flagged share of task windows / FPR | S | Why |
|---|---|---|---|
| dW = 0 (F = M), first and alone | 0 / 0 (no directions kept) | undefined | identity; a non-zero count stops the job |
| Construction check: dW = ε u₁v₁ᵀ, u₁ the top eigenvector of layer-1 `down_proj`'s C | not read | d λ₁ / tr(C) exactly, to the fp64 floor | positive control for S |
| Matched random of each state | no lean / ≤ 2t/n (Remark 3.1; t = 160, n = 45,000: 0.71%) | about 1 | the floors |
| x2, x10 (layer-1 `down_proj`, 8 columns times 2 or 10) | no lean; x10 ≥ x2 is the lean | x2 equals x10 exactly (same u, S ignores scale); lean above 1 | u lies in M's own columns |
| rebuilt | no lean | no lean | three MLPs change |
| new_few, new_thin (8 quiet layer-0 neurons write e(y)) | lean high: the slots fire on A′ᵢ only | lean below x2 | positive control for the detector |
| random_size, random_kl (P2.69's) | no lean | about 1 | built randoms |
| E62 copy vs lookup, 3 seeds | no lean | no lean; E141's pair rule (gap above the larger within-side range) | E97: both keep both circuits |
| Panel E272, wide KL "hi", 12 tasks × 3 seeds | descriptive | descriptive, beside E_b | read for whether S tracks the panel's readings |
| Qwen, the 19 saved updates (run 2) | not run unless stage 0's measured time fits two debug jobs | descriptive, against its floor | descriptive only |

Items are built as P2.69 builds them (`experiments/eapig_overlap/items.py`, `Items.make` and `Items.task`; hashes in
appendix B of `2026-11-23_plan_p2-69-eap-ig-overlap_appendix.md`). Every path was checked with `ls -laL` (appendix B).

## What it is blind to

The detector sees only inputs that move a hidden state outside Pile's range; a fine-tune that changes behaviour inside
that range is invisible to it. Both readings see only `o_proj` and `down_proj` output directions: changes to queries,
keys, values or `c_fc`, and which inputs a matrix reads (the right singular vectors), are not read. S is blind to
strengthening a direction M rarely writes along, which reads low, and to a new mechanism that writes along a busy
direction, such as a frequent token's unembedding, which reads high. Neither sees beyond the top 20 directions.

## How it differs from E141's effective rank

E141 (P2.9) reads only the singular values of all 24 matrices: how spread an update is, as one number per state. It
uses no directions and no data. WeightWatch reads the directions of 8 output matrices against hidden states on data;
singular values only rank them. S reads those directions against M's own outputs. Loaders reused from E141, not
copied: `commute.update_spectrum.load` (a saved 24-matrix 67M dW, fp64) and E141's path table
(`experiments/update_svd/run.py` `large`, `seeds`, `PAIRS["e62_copy_lookup"]`). The panel's states come through
`convergence_panel/run.py` `load_dW` and `results/wide/selection.json`. Qwen's come through E162's
`qwen_cheap_readings/run1.py` `updates()`, read one tensor at a time by `qwen_instrument_check/step0.py` `Tensors`.

## Order and cost

Stage 0 runs alone: dW = 0, the construction check, the σ floor, and one item's calibration timed. Then the built items,
then E62 and the panel (run 1, one cluster debug L40 job). Qwen is run 2. Forward passes need a GPU: calibration on
6.4M tokens per item, about 70 F's, and new_few and new_thin are built with forward passes. The SVDs are small.
E141 measured about 5.5 s per state on 2 CPUs (6 min for 65 states of 24 matrices). Our 8 matrices carry about 5/12 of
that work: about 2.3 s per state on the CPU, before U, and negligible on the GPU. No 67M forward throughput is measured
yet. Stage 0 times one item and projects that time times the count. The job exits for a resubmit past 25 min.
Qwen: E162 measured 270 s per update for its 196 SVDs on 4 CPUs. Our 56 matrices are about 35% of that work, about 95 s
per update, or seconds on the GPU. E162 read the 19 files (93 GB) in about 7 min. Cost $0.

## Design check (`/design`)

- **Goal.** Part 2's comparison of cheap readings and baselines against the definition; WeightWatch is a published
  weight-diff baseline the draft does not cite (digest bucket 1).
- **Nearest tried.** E141's effective rank (spectrum only) and E162 (the same on Qwen); neither read directions.
- **Patch or step.** A step: the cell's baseline, run with the published code's choices.
- **Decisions.** No verdict under either reuse wording; S is a reading with a floor, not a rule.
- **What would change what we do.** S far above its floor on x2 and near it on new_few/new_thin earns it a column in the
  baselines table; flat values would make it a negative result in the related-work paragraph.
- **Control first.** dW = 0 alone; its failure modes are a kept direction or a non-zero flag.
- **Saved.** Per state: kept u and σ (float16, about 100 KB), the ranges, the per-window flags, C_m's eigenvalues.

## Not run (yet)

Steering (the paper's section 5.3); generation during calibration (D4); the detector on Qwen unless its timing fits.
