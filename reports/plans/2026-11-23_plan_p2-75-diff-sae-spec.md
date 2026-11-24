# Plan: the diff-SAE with latent scaling as published, and how it would run on the 67M (cell P2.75, step 1: the spec)

Cell P2.75 (due 2026-11-25, 2 runs). User, 2026-11-23 evening, in writeup-lead's window: "if you want to run one more
then go for it". writeup-lead and litreview-lead chose the diff-SAE because it is the method a reviewer is most likely
to name: same authors and toolkit as the crosscoder and the activation difference lens we ran, and reported to beat the
BatchTopK crosscoder (88% against 60% of the KL gap). This is the spec, before any code. redteam-lead and an independent
GPT reviewer check it before code. Nothing has run.

**Sources.**
- The post: Dumas, Minder and Nanda, "What We Learned Trying to Diff Base and Chat Models (And Why It Matters)",
  LessWrong / Alignment Forum, 30 Jun 2025, `https://www.lesswrong.com/posts/xmpauEXEerzYcJKNm`. Saved 2026-11-23 to
  `~/commute-data/scratch/compare-lead/p275/post.html` (sha256 e89f0d74...) with a text dump `post.txt`; "post:N" below
  is a line of `post.txt`.
- Its linked paper: Minder, Dumas et al., "Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning",
  arXiv 2504.02922, the PDF the post links (`.../p275/minder2504.pdf`, sha256 87446b8a..., text `minder.txt`). **The
  paper contains no diff-SAE**: no "SAE" trained on a difference, no "diff-" anywhere in its text. It supplies latent
  scaling and the KL-gap protocol (§3.2) only. "paper:N" is a line of `minder.txt`.
- The toolkit: science-of-finetuning/diffing-toolkit, the clone at `~/commute-data/agents/compare-lead/adl/dt`, pinned
  at **617d926**, the same commit as our ADL and crosscoder settings. **The method is at this commit**:
  `src/diffing/methods/sae_difference/method.py` (386 lines), config `configs/diffing/method/sae_difference.yaml`,
  trainer setup in `src/diffing/utils/dictionary/training.py` 847-1092, latent scaling in
  `src/diffing/utils/dictionary/latent_scaling/closed_form.py`. No other pin is needed. Its trainer is
  `dictionary_learning` at d1fd6f3 (toolkit `uv.lock` 1240-1242), the same commit our `pyproject.toml` already pins
  for the crosscoder. "dt:" is a toolkit path; "dl:" is `site-packages/dictionary_learning/` in our venv.
- Ours: `src/commute/crosscoder.py` (`specific` 243-249, `carried` 253-279, `spliced` 462-486),
  `experiments/crosscoder_proper/`, `reports/analyses/2026-11-23_analysis_p2-70-negbeta-rescore.md`. Structure copied
  from `reports/plans/2026-11-23_plan_p2-72-delta-crosscoder-spec.md` and `..._p2-67-adl-spec.md`, with their review
  responses read: what reviewers sent back there (unstated reductions and units, bars without a measured source, a
  null that could not fail, unpriced forward passes, uncovered edge cases, our score in the method's column) is
  handled here from the start.

**Symbols, each defined once.**
- *M*: the original 67M model. *F*: a changed model (a fine-tune, a planted change or a random update).
- *h_M(x)*, *h_F(x)*: their residual activations at one site on token x. *D(x) = h_F(x) − h_M(x)*, the change.
- *N(y) = (y − μ_D)·s*: the toolkit's normaliser, with μ_D the mean of D and s = 100 / sqrt(total variance of D).
- *d_j*: latent j's unit decoder vector; *f_j(x)*: its activation; *b*: the decoder bias.
- *β_M, β_F*: latent scaling's coefficients against M's and F's normalised activations; *ν = β_M / β_F*.
- *Specific part*: the share of the change, Σ_x |D(x)|², carried by the latents called fine-tune-specific.

## 1. Plain answers

**What the method outputs.** Per latent: ν, its two coefficients, its firing rate and its top-activating text. Per
dictionary: the fraction of variance of D explained, the count of latents under each ν threshold, and the KL-gap
recovery of four patches (all latents, the error, latents with ν ≤ 0.3, latents with ν ≤ 0.6). There is no single
per-fine-tune verdict. The toolkit adds steering with the ten lowest-ν latents (dt:`sae_difference.yaml` 73-79).

**Is there a "specific share" in the post?** No. The post counts latents under ν thresholds (post:53) and reads KL
recovery; it never asks what share of the change those latents carry. The specific part used here is ours, carried
over from our crosscoder reading. It is named "specific part, after Dumas" wherever it appears.

**What 88% against 60% measures.** On Gemma 2 2B at layer 13, the base model's activation is patched into the chat
model and the KL divergence from the true chat output is the gap (post:56-57). Adding back the dictionary's
reconstruction of chat minus base removes 88% of that gap for the diff-SAE and 60% for the BatchTopK crosscoder
(post:58-59, footnote post:110). The diff-SAE's error alone removes 30%, the crosscoder's 60%. It measures how much of
the behavioural difference the dictionary rebuilds, at the output only. It says nothing about which latents are
fine-tune-specific or about mechanism. The post does not say which tokens were read. The crosscoder's 60% matches the
paper's 59% over all response tokens (paper:525), so the 88% is probably the same set.

**Does it bear on reuse against new?** Indirectly. On a diff-SAE, ν asks whether the change's direction is already
present in the base model's activation where the latent fires. Near 1, the change strengthens something the base
writes; near 0, it writes a direction the base does not. That is a single-site activation reading, blind to whether
downstream parts read the direction the same way. The post never frames it as reuse. Its one sentence on reuse is about
the chat-SAE, not the diff-SAE: "by design, it will make it hard to detect latents that get reused in new context by
the chat model" (post:51). Any reuse score built on ν is ours, "after Dumas".

## 2. The published method, step by step

Each step: the post's or paper's words, then the toolkit's code at 617d926.

| step | post or paper | toolkit at 617d926 |
|---|---|---|
| activations | "train an SAE directly on the difference between activations (chat - base)" (post:52); "a BatchTopK SAE directly on layer 13" (post:51, the chat-SAE) | `target: "difference_ftb"` (`sae_difference.yaml` 7); `DifferenceCache(cache_2, cache_1)` = fine-tune minus base (`training.py` 100-102); site `preprocessing.layers: [0.5]`, the middle layer (`config.yaml` 123) |
| text | not stated for the diff-SAE; the paper's crosscoder used "100M tokens from Fineweb ... and lmsys-chat" (paper:2331-2333) | chat, pretraining and the organism's training set (`sae_difference.yaml` 25-28), at most 50M tokens per dataset (`config.yaml` 125), context 1024 (128); activations stored in bfloat16 (129); D formed from the stored pairs |
| tokens | not stated | `num_samples: 150_000_000` (18) is a cap over all datasets, here 3 × 50M; `epochs: 2` (10) repeats it, so up to 300M presentations (`activations.py` 109-127 returns sum × epochs indices) |
| normalisation | not stated | on D: mean and std of 1M sampled difference tokens (`sae_difference.yaml` 30-35, `training.py` 113-197); D − μ_D, then one global scale to RMS 100 (`target_rms: 100`, 35; dl:`dictionary.py` 76, 108) |
| architecture | BatchTopK; "expansion factors as low as 1 (keeping the same k=100 ...)" (post:62); "x2 denotes expansion factor" (post:53) | `BatchTopKSAE`, `expansion_factor: 2`, `k: 100` (`sae_difference.yaml` 8, 15, 39; `training.py` 884-905); encode ReLU(W_e(x − b)) then the batch's top k·B (dl:`dictionary.py` 577-586); decoder unit-norm, no per-model decoders |
| recipe | not stated | Adam lr 1e-4, warm-up 1,000 steps, then constant (`sae_difference.yaml` 11, 44; no decay set); batch 2,048 (9); squared L2 loss (dl:`batch_top_k.py` 199); AuxK α 1/32, k_aux d/2, dead after 10M tokens (26, 82-83); θ as an EMA from step 1,000 (145-160); b initialised to the geometric median of the first raw batch (223); grad clip 1.0 (241); decoder renormed each step (248); encoder init = decoder transpose × 1.0 (`encoder_init_norm`, 12); TF32 (`config.yaml` 141) |
| latent scaling | β^m_j = argmin_β Σ_i ‖β·f_j(x_i)·d_j − h_m(x_i)‖² (post:35); "ν_j = β^base_j / β^fine-tuned_j ... ν ≈ 0 ... specific, ν ≈ 1 ... shared" (post:36-37); closed form β = dᵀ(Yᵀf)/(‖f‖²‖d‖²) (post appendix B) | targets `["base_activation", "ft_activation"]` on 50M training tokens (`sae_difference.yaml` 52-55); closed form `closed_form.py` 40-168; targets are N(h_M), N(h_F), normalised with the SAE's own D-normaliser (700-706; `latent_scaling/utils.py` 53-79); ν stored as `beta_activation_ratio` = β_M / β_F (`beta_analysis.py` 187-189) |
| which input f is read from | not stated | **f_j comes from encoding h_F, not D**: the activation-target pass calls `compute_scalers(..., is_difference_sae=False)` (`closed_form.py` 354), so the SAE branch picks `load_ft_activation` (693-697). The difference branch (686-692) is used only for error targets, which the diff-SAE config does not request. Line 354 predates the diff-SAE method (`git blame`: e84b547, 2025-06-16, the crosscoder commit), so it reads as an oversight. Open question 1 |
| selection | "Chat-only latents (ν⩽0.3)" (post:52-53 captions); KL with "latent scaling scores below 0.3 or 0.6" (post:60) | no threshold in the pipeline; steering takes the 10 lowest ν (`target_column: "beta_activation_ratio"`, `k: 10`, `largest: false`, 77-79); no negative-β filter anywhere in `analysis.py` or `beta_analysis.py` |
| evaluation | patch base_act + (chat_recon − base_recon) into the chat model at the layer; error patch base_act + ((chat − base) − (chat_recon − base_recon)) (post:110); subset patch h_base + Σ_{j∈S} f_j d_j (paper eq. 5, paper:428-430); KL(p_patched ‖ p_chat) against the h_base patch; 512 chat interactions, response tokens 2-10 and all (paper:453-463) | **not in the toolkit for the diff-SAE**: `method.py` runs training, latent dataframe, latent scaling, activation stats, plots and steering (142-215); its docstring's "KL divergence experiments" (6) has no code. `methods/kl` is a different method, per-token KL between the two models. So the KL-gap step is ported from the paper's text (§3.2) |

**Quirks of the toolkit kept as theirs, listed so a reviewer can see them.** The decoder bias is initialised from the
median of the raw first batch but used in normalised units (dl:`batch_top_k.py` 221-229). The base and fine-tune
targets are normalised with D's mean and scale, not their own; the scale cancels in ν, the mean does not.

## 3. What we would run on the 67M: the objective and the reading, fixed

All of §3 is ours unless it quotes §2. Each choice is a row in §4. Code: a new `experiments/diff_sae/run.py` calling
the pinned `BatchTopKTrainer` and `BatchTopKSAE` unchanged, with glue in `src/commute/diff_sae.py`. It reuses
`crosscoder.site_acts`, `batches`, the E235 stream and the E235 Gaussian, and copies nothing.

**3.1 Sites, text and tokens.** Site A is block 2's input, the toolkit's middle layer, where E203, E212 and E243 read.
Site B is block 3's output, for E235's block-3 head only. The text is each side's own crosscoder stream (E235
`crosscoder_proper/run.py` `stream`, 286 on): Pile train rows 32,768-131,071 and the side's task windows, about 50M
tokens each. That is the toolkit's 50M per dataset with the chat dataset dropped. Two epochs give 200M presentations
per fit, 97,656 steps at batch 2,048. Position 0 is dropped, as the crosscoder does. D is formed in fp32 from fp32
forwards with TF32 off, as E235's scoring did. The toolkit's bfloat16 storage is not used. Its rounding, about 2⁻⁸ of
‖h‖, is the size of a fine-tune's change here.

**3.2 Objective, unchanged from their code.** The normaliser is fit on 1M tokens of D drawn from the first chunks
(`recompute_normalizer`, `training.py` 113-197, ported as 10 lines: mean and biased std, then dl's global scale).
The model is `BatchTopKSAE(768, 1536, k=100, activation_mean=μ_D, activation_std=σ_D, target_rms=100,
encoder_init_norm=1.0)`, and the step is `BatchTopKTrainer.update` with lr 1e-4, warm-up 1,000, auxk α 1/32,
threshold start 1,000, dead after 10M tokens, seed from the item. Training uses TF32, theirs. Reading uses the
thresholded encode, their inference default (`use_threshold=True`).

**3.3 Latent scaling.** It runs on the first 50M training tokens, the toolkit's count. That is 500 times the 100,000
our crosscoder used; E289 showed 50M fits run in minutes. Every latent is scored. β_M and β_F are the closed form with
targets N(h_M) and N(h_F), accumulated in float64 as `crosscoder.beta_sums` does, and ν = β_M / β_F. Two forms of f
are computed in the same pass:
- **ν_D**: f from encoding D, the SAE's training input. This is the scored form (open question 1).
- **ν_lit**: f from encoding h_F, the toolkit's line 354 literally. It is reported beside, unscored.
A construction check holds by linearity: β_F − β_M equals the coefficient fit to s·D, so ν_D = 1 − β_D / β_F. Its bar
is relative and taken from the rehearsal's float32 against float64 floor.

**3.4 Which latents are fine-tune-specific.** A latent is alive if it fires on at least 10 of the read tokens, the
crosscoder's `alive_min`. Two rules are applied to every fit:
- **As published**: ν ≤ 0.3, the post's caption threshold.
- **As our crosscoder reads**: ν ≤ 0.3 and β_M ≥ 0. This is Minder et al.'s App. E.2 exclusion, carried over from the
  crosscoder paper by the same authors, as `crosscoder.specific` applies it.

The cell asks for the same reading as the crosscoder, so the second rule is the scored one. The first is reported
beside it on every side, since neither the post nor the toolkit filters negative coefficients (open question 2).

Edge cases, each checked against the rule before filing:
- A latent that never fired on the 50M tokens has β = 0/0, a nan, and fails every comparison.
- β_F = 0 with β_M ≠ 0 gives ν = ±inf. It is set to nan and fails; without this, −inf would pass.
- β_F < 0 gives a ν of the wrong sign. It is kept literal and counted separately, so a reader sees how many pass that
  way.
- No latent passing gives a specific part of 0.0, not nan.
- Ties at exactly 0.3 have measure zero; ≤ follows the caption.

**3.5 The reading, in raw units, as `crosscoder.carried` computes it.** Decoding is D̂(x) = Σ_j f_j(x) d_j/s + (b/s +
μ_D). Each latent's carried part is s_j = Σ_x f_j(x) ⟨d_j/s, D(x)⟩ / Σ_x ‖D(x)‖², over the side's score-half read
windows. The bias part is Σ_x ⟨b/s + μ_D, D(x)⟩ / Σ_x ‖D(x)‖², and the error part is 1 minus the latents' parts minus
the bias part. The **specific part** is the sum of s_j over specific latents. Beside it each side reports the bias
part, the error part, the FVE of D on the read set, the number of specific latents, and the s-weighted histogram of ν.
If Σ‖D‖² = 0 (M against itself), every part is undefined and nothing is fitted.

**3.6 The KL-gap recovery, theirs, ported from the paper's §3.2.** At the site, F's forward is run with its
activation replaced at positions 1 onward by one of these:
- h_M, the gap;
- h_M + D̂, "all";
- h_F − D̂ + (b/s + μ_D), "error", which is h_M + (D − D̂) in the post's form;
- h_M + Σ_{j: ν ≤ 0.3} f_j d_j/s and the same at ν ≤ 0.6, the subsets with no bias, as eq. 5 has.

Recovered = 1 − KL(patched) / KL(h_M), with KL(p_F,patched ‖ p_F) the mean per token. It is read on the side's
score-half read windows, at the read positions and at all positions, and on 256 Pile validation rows. Two path checks
must hold. Patching h_F must give KL 0 to the measured floor. Patching h_M must give recovered 0 exactly.

**3.7 Expected values, worked out before any fit.** Take a latent whose direction is the planted direction p̂ and that
fires only where the plant is, on a share ρ of tokens, so μ_D = ρp. The closed form then gives β_F − β_M = 1/(1 − ρ).
It also gives β_M = (c − ρ)/(1 − ρ), with c = ⟨mean of h_M over the plant positions, p̂⟩ / ‖p‖. So ν = (c − ρ)/(c − ρ + 1).
Three things follow:
- If the base activation has no mean component along p̂ (c = 0), β_M is a small negative number, −ρ/(1 − ρ). The latent
  passes "as published" and is dropped by the E.2 exclusion. This is the exclusion's sign problem from P2.70 again,
  now caused by the toolkit's mean subtraction.
- If c is a few tenths, ν crosses 0.3 and the plant fails both rules.
- ν tends to 1 as |c| grows. At fine-tune size ‖D‖ is about a hundredth of ‖h‖, so any latent whose direction is not
  nearly orthogonal to M's mean activation reads ν ≈ 1. The expected specific part on the nine pairs is therefore near
  0, for a reason of scale, as the crosscoder's was.

These are computed, not guessed, in a **$0 oracle check** on the cluster before filing. It covers each control and one
built side. For the real-text plant it uses p itself. For the heads, the pairs and the Gaussians it uses the top ten
principal directions of D on the read set, with f the positive part of D's component along each. It reports ρ, c and
the oracle ν under both rules. The planted control's filed expected value is its oracle ν, and its specific part is
the oracle directions' s.

## 4. Departures

"Theirs" means taken from the post, the paper or the toolkit at 617d926. "Ours" means our choice where they are silent
or cannot apply. "Forced" means the 67M leaves no faithful option.

| step | theirs | ours | why | ours/theirs | forced |
|---|---|---|---|---|---|
| code | toolkit pipeline | pinned trainer and model unchanged; glue ours | the toolkit pulls vllm and streamlit and cannot be installed (E203) | theirs (core) / ours (glue) | y |
| model pair | Gemma 2 2B base / chat | 67M M / F | the testbed | ours | y |
| site | middle layer, 0.5 | block 2's input; block 3's output for the block-3 head | where the crosscoder read | theirs | n |
| text | chat + pretraining + training set, 50M each | Pile + the side's task text, 50M each, E235's stream | no chat model | ours | y |
| context | 1024 | 512 | the 67M's rows | ours | y |
| tokens | ≤ 150M × 2 epochs | 100M × 2 epochs | 50M per dataset with chat dropped; same text as the crosscoder column | theirs (per dataset) | y |
| storage | bf16 activations | fp32 forwards, D in fp32 | bf16 rounding is a fine-tune's size here | ours | n |
| normaliser | D's mean and scale, RMS 100, 1M samples | same | | theirs | n |
| SAE | BatchTopK, x2, k 100 | 1,536 latents, k 100 | k/d is 3 times Gemma's (open question 3) | theirs | n |
| recipe | lr 1e-4, warm-up 1k, batch 2,048, AuxK, θ EMA, TF32 | same | | theirs | n |
| latent scaling | 50M tokens, targets h_M and h_F with D's normaliser | same | | theirs | n |
| f in latent scaling | from h_F (`closed_form.py` 354) | from D, scored; from h_F, reported | line 354 reads as an oversight | ours (reading) | n |
| specific rule | ν ≤ 0.3 | ν ≤ 0.3 and β_M ≥ 0 scored; ν ≤ 0.3 alone reported | the cell: same reading as the crosscoder (App. E.2) | ours, after Minder | n |
| ν = ±inf | not handled | nan, fails | a −inf would pass | ours | n |
| alive | not used | ≥ 10 firings on the read set | crosscoder's `alive_min` | ours | n |
| reading | latent counts and KL recovery | specific part of the change, bias and error parts | the crosscoder column's reading | ours, after Dumas | n |
| KL-gap step | paper §3.2, chat responses | ported; read windows, read positions and all, Pile | no chat model; toolkit has no code | theirs (form) / ours (data) | y |
| steering, top-activation reading | 10 lowest ν; judged by people and a grader | not run | no chat text to judge | ours | y |
| controls | none | plants, Gaussians, a reseed | to make a pass or a miss readable | ours | n |

## 5. Proposed runs, cost and gates

**Measured items.** From E235, `experiments/crosscoder_proper/results/run1/timing*.json` and `items/*.json`, on an H100:
- the stream through both models takes 2.608 s per 1,046,544 tokens at site A and 4.338 s at site B, so 200M tokens
  take 8.3 and 13.8 min;
- a crosscoder step (24,576 latents on two stacked inputs) takes 11.65 ms, so 97,656 steps take 19.0 min;
- the 20 crosscoder items, training plus scoring, took 27.0-29.6 min at site A and 34.1-35.3 min at site B, one outlier
  at 60 min was a resumed item. The caller's ~28-32 min holds for site A.

E212's two pods cost $8.03 for about 50 min each, so we price $4.8 per pod-hour with set-up, as P2.72 did.

**Per fit, derived.** The SAE has about a thirtieth of the crosscoder's weights, so its step is priced at the
crosscoder's 11.65 ms as an upper bound until the rehearsal measures it.
- Site A: stream 8.3 + training ≤ 19.0 + latent scaling on 50M (2.1) + reading and KL patches about 2 = **≤ 31 min**.
- Site B: 13.8 + 19.2 + 3.5 + 2 = **≤ 39 min**.
- If the step comes out at 3 ms, as the size suggests, these fall to about 17 and 25 min.

**Rehearsal ($0, cluster `debug`, before run 1).** The SAE is full size at 1,536 latents, so only the token count is
tiny (4M). Every stage runs end to end:
- the stream; the normaliser; training, a save, a resume checked by step-matched loss against an uninterrupted run (the
  floor from two uninterrupted runs);
- latent scaling at the full 50M tokens; the KL patches at full read size with both path checks; the downloads;
- the oracle check of §3.7.

It measures ms per step at batch 2,048, peak GPU and host memory per stage, the step-1 loss and FVE, the float32 against
float64 floor of the ν identity, and the KL floor of the h_F patch. Those numbers replace the derived ones before run 1.

**Run 1, the controls first and alone. Not gated, about $20, cap $35.**
- **Stage 0a, first and alone.** E212's real-text planted direction at site A, with E212's own Gaussian (matched to T2,
  `gauss_draw` 0) at site A as its null, on two pods at once. Under the E.2 exclusion our BatchTopK crosscoder reads
  this plant at 0, where without it the plant read 0.77.
  - Pass rule: the plant's specific part, scored rule, is at least 0.5, the crosscoder column's bar (E235), and the
    Gaussian's is under 0.5.
  - Expected value: the oracle's of §3.7, filed before the run.
  - Lean: fail under the scored rule, pass "as published", if the oracle gives c near 0.
  - The run carries on to stage 0b either way: a miss is a reading, not a stop.
- **Stage 0b.** E235's block-3 head at site B, with E235's Gaussian at site B: the draft's reported pass that P2.70
  reversed, read 0.42-0.44. Also E243 rung 5 at site A (crosscoder 0.46, then 0.010), and a second SAE seed on E212's
  plant. The reseed's difference is the seed floor for every later comparison.
- Six fits, 4 × 31 + 2 × 39 = 202 min, plus 3 pods × 15 min of set-up: about 4.1 pod-hours, about $20, about 1.3 h of wall
  time. Uploads are small: E212's T2 state and the Pile rows E235 reads, about 0.6 GB a pod, 1 min at 10 MB/s.

**Early checks in code.** Each check stops that fit and is reported:
- a non-finite loss;
- FVE of D under 0.5 at the 25M-token snapshot, the crosscoder's code-fault bar (the post's diff-SAE reads 0.82);
- dead share over 0.3 at the end, the crosscoder's bar;
- either KL path check off its rehearsal floor;
- the ν identity off its floor;
- before step 0, D not exactly 0 under a zero update.

**What opens run 2.** Run 2 opens when a site-A plant (E212 or rung 5) passes and the site-A Gaussian reads under the
bar.
- If no site-A plant passes, site A reads "below the method's reach at this update size", and the cell closes with
  run 1's reading.
- If the Gaussian passes too, site A reads "does not separate a plant from a random update".
- Either way, results go to redteam-lead, then whip-lead.

**Run 2, gated on cost, only if run 1 opens it.**
- **The 39-fit default:** the nine built pairs' 18 sides with one matched random update per pair (9), and E62 copy and
  lookup, three saved seeds each (6), each with its matched random update (6). It costs 39 × 31 min + 8 pods × 18 min of
  set-up, about 23 pod-hours, **about $111**, and about 3 h of wall time on 8 pods (two runners of 4).
- **The 48-fit form** gives every built side its own random update: about 28 pod-hours, about $134.
- **The matched random**, E235's recipe (`crosscoder_proper/run.py` 247-268). It is a Gaussian weight change at the
  side's per-matrix norms (`finetune.random_direction`, the draw seeded). A scalar is set by bisection so that its
  activation change per token at the site equals the side's, within 0.05. It is rebuilt on the pod from its seed.
- **Uploads:** 24 trained states × 270 MB (P2.72's measured size) = 6.5 GB, about 11 min spread over the pods.
  Downloads are about 15 MB a fit, 0.6 GB, 1 min.
- **Output per side:** the specific part under both rules, the bias and error parts, the FVE of D, the KL-gap
  recoveries, the specific count, and the side minus its matched random. It is descriptive: no reuse verdict and no pair
  call. One fit a side, so differences are read against the stage-0b seed floor.
- **Gates** (`.claude/CLAUDE.md`): the pre-mortem under every heading, every setting sourced (§2 and §4), the
  rehearsal, early checks, one review by a fresh Opus agent, then gate 6 from whip-lead (cell text). It is under 6 h, so
  gated on cost only.

**Totals.** Run 1 plus the 39-fit run 2 is about $131; with the 48-fit form, about $154. Both are inside the session's
$300 with its 15% headroom ($255). If the rehearsal's step time is near 3 ms, both fall by about two fifths.

## Not run

- Nothing has run. No code exists, and no E-number is claimed.
- Steering with the lowest-ν latents and reading their top-activating text are not proposed: there is no chat text on
  the 67M to judge.
- The chat-SAE and base-SAE variants the post also compares are not proposed.
- The KL-gap recovery of our saved crosscoders, which would put the post's 60% beside our 88%-style number, is not
  proposed. E203's and E244's crosscoders kept no encoder.
- P2.74's old tasks are not in the count. Each needs its own fit, so reading them is not free. whip-lead may swap them
  into run 2's count.
- Qwen is not proposed.

## Open questions for the reviewers

1. **Which input does latent scaling encode?** The toolkit at 617d926 encodes the fine-tune's activation h_F with an
   SAE trained on D (`closed_form.py` 354 → 693-697). We read this as an oversight and score ν_D, encoding D, with ν_lit
   reported beside. Is ν_D the faithful reading of the post's "combining Latent Scaling with an SAE trained on ... chat
   - base activations"?
2. **Is the E.2 exclusion scored?** The cell asks for "the same specific-share reading as the crosscoder", which
   includes App. E.2. The diff-SAE's own sources have no such filter. §3.7 predicts that a clean planted direction gets
   a small negative β_M from the toolkit's mean subtraction alone. So the two rules can split on exactly the control.
   We score the E.2 form and report both. Is that the right way round?
3. **k at d = 768.** k 100 of 1,536 latents is literal (theirs), but k/d is three times Gemma's (100 of 2,304). Our
   crosscoder kept k 100 and swept k 32. Should a k-33 fit on E212's plant be added to stage 0b as a description, at
   about $2.5?
4. **Which control is "the one" the cell means?** Three planted controls fail under E.2: E212's real-text plant (0.77
   to 0), E235's block-3 head (0.56-0.63 to 0.42-0.44) and E243's top rungs (0.46 to about 0.01). We put E212's first
   and alone: it sits at the scored pairs' site, it is a single direction, so its expected value can be computed, and
   it reversed the most. The block-3 head follows in stage 0b. Is that order acceptable?
5. **The bar of 0.5** is the crosscoder column's, kept for like-for-like. A null band from one Gaussian per site is
   weak. Should stage 0 add a second Gaussian draw per site, at about $2.5 each?
6. **Scale.** §3.7 predicts ν ≈ 1 for any latent not nearly orthogonal to M's mean activation once ‖D‖ ≪ ‖h‖. If the
   oracle check confirms it on a built side, is run 2 still informative, or should the cell close on run 1 plus the
   oracle, as "latent scaling on a diff-SAE cannot call a fine-tune-size change specific"?
