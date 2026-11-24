# Plan: the diff-SAE with latent scaling as published, and how it would run on the 67M (cell P2.75, step 1: the spec)

Cell P2.75 (due 2026-11-25, 2 runs). User, 2026-11-23 evening, in writeup-lead's window: "if you want to run one more
then go for it". writeup-lead and litreview-lead chose the diff-SAE because it is the method a reviewer is most likely
to name: same authors and toolkit as the crosscoder and the activation difference lens we ran, and reported to beat the
BatchTopK crosscoder (88% against 60% of the KL gap). This is the spec, before any code. It applies redteam-lead's
four fixes (`note-2026-11-23-redteam-p275-diff-sae-spec.md`): both forms of ν scored, the as-published rule scored,
the bias cap on the bar, and a floor on the KL gap. An independent GPT reviewer
checks it before code. Nothing has run.

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
the behavioural difference the dictionary rebuilds, at the output only, not which latents are fine-tune-specific.
The read tokens are probably all response tokens: the crosscoder's 60% matches the paper's 59% there (paper:525).

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
| which input f is read from | not stated | **f_j comes from encoding h_F, not D**: the activation-target pass calls `compute_scalers(..., is_difference_sae=False)` (`closed_form.py` 354), so the SAE branch picks `load_ft_activation` (693-697). The difference branch (686-692) is used only for error targets, which the diff-SAE config does not request. Line 354 dates from e84b547 (2025-06-16), before the post (30 Jun 2025), so the post's figures may come from this form. Whether it is intended is not stated |
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
are computed in the same pass, and both are scored, each under its own label:
- **ν_lit**, "theirs, as the code runs": f from encoding N(h_F), as line 354 does.
- **ν_D**, "ours, as the post's words read": f from encoding N(D), the SAE's training input.

Neither is called the faithful form. Encoding h_F with D's normaliser, whose scale is about 100/‖D‖, may make the
top-k fire on M's own activation rather than on the change, which is why ν_D is run. That this is an oversight is our
inference, not the authors' statement. If the two forms disagree on a control, the paper's column cites both.
A construction check holds by linearity for either f: β_F − β_M equals the coefficient fit to s·D, so
ν = 1 − β_D / β_F. Its bar is relative and taken from the rehearsal's float32 against float64 floor.

**3.4 Which latents are fine-tune-specific.** A latent is alive if it fires on at least 10 of the read tokens, the
crosscoder's `alive_min`. Two rules are applied to every fit, under each form of ν:
- **As published, scored**: ν ≤ 0.3, the post's caption threshold.
- **With the E.2 exclusion, reported beside, not scored**: ν ≤ 0.3 and β_M ≥ 0, Minder et al.'s App. E.2 rule for
  their crosscoder, as `crosscoder.specific` applies it.

The E.2 form is not scored because it fails its own planted control by construction. With c ≈ 0, mean subtraction
alone gives the plant β_M = −ρ/(1 − ρ) < 0 (§3.7), so a clean plant is excluded in exactly the case it is built for.
The diff-SAE's sources have no such filter, so carrying it over would be our step. P2.70 is no precedent: there E.2
was the authors' own rule for their own method.

**Open point for whip-lead.** The cell says "scored like our crosscoder", which could be read to require E.2. The
oracle check (§3.7) reports ρ and c first. whip-lead then decides whether the E.2 form is scored instead; until then
the as-published rule is the one filed.

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

**The bias part caps the specific part.** For a plant that is constant where it fires, the bias part equals ρ, the
share of tokens it fires on, so the latents carry at most 1 − ρ. Any side whose change has a large mean has the same
cap. The oracle reports ρ for each control, and §5 sets the bar from it.

**3.6 The KL-gap recovery, theirs, ported from the paper's §3.2.** At the site, F's forward is run with its
activation replaced at positions 1 onward by one of these:
- h_M, the gap;
- h_M + D̂, "all";
- h_F − D̂ + (b/s + μ_D), "error", which is h_M + (D − D̂) in the post's form;
- h_M + Σ_{j: ν ≤ 0.3} f_j d_j/s and the same at ν ≤ 0.6, the subsets with no bias, as eq. 5 has.

Recovered = 1 − KL(patched) / KL(h_M), with KL(p_F,patched ‖ p_F) the mean per token. It is read on the side's
score-half read windows, at the read positions and at all positions, and on 256 Pile validation rows. Two path checks
must hold. Patching h_F must give KL 0 to the measured floor. Patching h_M must give recovered 0 exactly.

The gap itself can be tiny on a Gaussian side or a small pair, and then "recovered" is a ratio of two floors. So
KL(patch h_M) has a floor, set in the rehearsal as a multiple of the h_F patch's KL floor. Below it, every recovery
on that side reads "not read". Before filing, the rule is run on the do-nothing state, where the gap is 0 and every
recovery must read "not read", and on E212's Gaussian.

**3.7 Expected values, worked out before any fit** (re-derived by redteam-lead). Take a latent whose direction is the
planted direction p̂ and that fires only where the plant is, on a share ρ of tokens, so μ_D = ρp. Encoding N(D) gives
f = s‖p‖(1 − ρ) on plant tokens and 0 elsewhere. The means cancel in N(h_F) − N(h_M) = sD, so β_F − β_M = 1/(1 − ρ).
It also gives β_M = (c − ρ)/(1 − ρ), with c = ⟨mean of h_M over the plant positions, p̂⟩ / ‖p‖. So ν = (c − ρ)/(c − ρ + 1).
Three things follow, for ν_D:
- If the base activation has no mean component along p̂ (c = 0), β_M is a small negative number, −ρ/(1 − ρ). The latent
  passes "as published" and is dropped by the E.2 exclusion, which is why E.2 is not scored (§3.4).
- If c is a few tenths, ν crosses 0.3 and the plant fails both rules.
- ν tends to 1 as |c| grows. A direction with no special relation to M's mean has a component of about
  ‖h‖/√768 ≈ 0.036‖h‖. Against ‖p‖ ≈ 0.01‖h‖ that gives c ≈ 3.6 and ν ≈ 0.78, a fail. The expected specific part on
  the nine pairs is therefore near 0, for a reason of scale, as the crosscoder's was.

ν_lit has no closed form here, since its f depends on M's activation along p̂ on every token.

These are computed, not guessed, in a **$0 oracle check** on the cluster before filing. It covers each control and one
built side. For the real-text plant it uses p itself. For the heads, the pairs and the Gaussians it uses the top ten
principal directions of D on the read set. For ν_D, f is the positive part of N(D)'s component along each direction.
For ν_lit, it is the positive part of N(h_F)'s component. It reports, per control, ρ as the bias
part Σ⟨μ_D, D⟩ / Σ‖D‖², c, and the oracle ν under both forms and both rules. So the two forms are compared before any
fit. The planted control's filed expected value is its oracle ν, and its specific part is the oracle directions' s.

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
| SAE | BatchTopK, x2, k 100 | 1,536 latents, k 100 | k/d is 3 times Gemma's; a k-33 fit only if the plant fails | theirs | n |
| recipe | lr 1e-4, warm-up 1k, batch 2,048, AuxK, θ EMA, TF32 | same | | theirs | n |
| latent scaling | 50M tokens, targets h_M and h_F with D's normaliser | same | | theirs | n |
| f in latent scaling | from h_F (`closed_form.py` 354) | both scored: ν_lit from h_F; ν_D from D | the post's words name D; the code encodes h_F; neither is known to be intended | theirs, as the code runs (ν_lit) / ours, as the post's words read (ν_D) | n |
| specific rule | ν ≤ 0.3 | same, scored; ν ≤ 0.3 and β_M ≥ 0 (App. E.2) reported beside | E.2 fails a clean plant by construction (§3.4); whip-lead may decide otherwise after the oracle | theirs (scored) / ours, after Minder (E.2) | n |
| pass bar | none | specific part ≥ 0.5, the crosscoder column's bar, if 1 − ρ > 0.5; else specific part / (1 − bias part) ≥ 0.5 | the bias part caps the latents at 1 − ρ (§3.5) | ours | n |
| KL-gap floor | none | KL(patch h_M) under a rehearsal-set multiple of the h_F patch's floor reads "not read" | a ratio of two floors otherwise | ours | n |
| ν = ±inf | not handled | nan, fails | a −inf would pass | ours | n |
| alive | not used | ≥ 10 firings on the read set | crosscoder's `alive_min` | ours | n |
| reading | latent counts and KL recovery | specific part of the change, bias and error parts | the crosscoder column's reading | ours, after Dumas | n |
| KL-gap step | paper §3.2, chat responses | ported; read windows, read positions and all, Pile | no chat model; toolkit has no code | theirs (form) / ours (data) | y |
| steering, top-activation reading | 10 lowest ν; judged by people and a grader | not run | no chat text to judge | ours | y |
| controls | none | plants, two Gaussian draws at site A, a reseed | to make a pass or a miss readable | ours | n |

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
float64 floor of the ν identity, the KL floor of the h_F patch, and the KL(patch h_M) floor of §3.6. Those numbers
replace the derived ones before run 1.

**Run 1, the controls first and alone. Not gated, about $22, cap $35.**
- **Stage 0a, first and alone.** E212's real-text planted direction at site A, with two Gaussian draws at site A as its
  null (matched to T2, `gauss_draw` 0 and 1), on three pods at once. Under the E.2 exclusion our BatchTopK crosscoder
  reads this plant at 0, where without it the plant read 0.77.
  - Pass rule, under each form of ν separately and the as-published rule: the plant's specific part reaches the bar,
    and both Gaussians' stay under it. The bar is 0.5, the crosscoder column's (E235), if the oracle gives 1 − ρ > 0.5.
    Otherwise it is 0.5 of (1 − bias part), for every side alike (§4).
  - Expected value: the oracle's of §3.7, filed before the run, with ρ and c.
  - Lean: fail under both forms if the oracle gives c of order 1 or more, as §3.7's scale argument predicts.
  - The run carries on to stage 0b either way: a miss is a reading, not a stop.
- **Stage 0b.** E235's block-3 head at site B, with E235's Gaussian at site B: the draft's reported pass that P2.70
  reversed, read 0.42-0.44. Then E243 rung 5 at site A (crosscoder 0.46, then 0.010), and a second SAE seed on E212's
  plant, whose difference is the seed floor for every later comparison. A k-33 fit on E212's plant is added only if the
  plant failed in 0a. A rank-1 plant under k 100 of 1,536 may be split over many latents, so k 33 tells split from scale.
- Seven fits, 5 × 31 + 2 × 39 = 233 min, plus 3 pods × 15 min of set-up: about 4.6 pod-hours, about $22, about 2 h of
  wall time. The k-33 fit adds 31 min, $2.5. Uploads are small: E212's T2 state and the Pile rows E235 reads, about
  0.6 GB a pod, 1 min at 10 MB/s.

**Early checks in code.** Each check stops that fit and is reported:
- a non-finite loss;
- FVE of D under 0.5 at the 25M-token snapshot, the crosscoder's code-fault bar (the post's diff-SAE reads 0.82);
- dead share over 0.3 at the end, the crosscoder's bar;
- either KL path check off its rehearsal floor;
- the ν identity off its floor;
- before step 0, D not exactly 0 under a zero update.

**What opens run 2.** Run 2 waits on the oracle as well as on run 1.
- Suppose the oracle gives ν > 0.3 under both forms, for the plant and for a built side's top principal directions.
  Then run 2 is predicted to read about 0 on every side, random ones included, and the cell closes on run 1 plus the oracle,
  reading "latent scaling on a diff-SAE cannot call a change of fine-tune size specific at this site". Run 1 still
  runs: the trained SAE's f is not the oracle's constant.
- Otherwise run 2 opens when a site-A plant (E212 or rung 5) passes under at least one form and both site-A Gaussians
  read under the bar in that form.
- If no site-A plant passes, site A reads "below the method's reach at this update size", and the cell closes with
  run 1's reading.
- If a Gaussian passes too, site A reads "does not separate a plant from a random update".
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
- **Output per side:** the specific part under both forms and both rules, the bias and error parts, the FVE of D, the KL-gap
  recoveries, the specific count, and the side minus its matched random. It is descriptive: no reuse verdict and no pair
  call. One fit a side, so differences are read against the stage-0b seed floor.
- **Gates** (`.claude/CLAUDE.md`): the pre-mortem under every heading, every setting sourced (§2 and §4), the
  rehearsal, early checks, one review by a fresh Opus agent, then gate 6 from whip-lead (cell text). It is under 6 h, so
  gated on cost only.

**Totals.** Run 1 with the k-33 fit plus the 39-fit run 2 is about $136; with the 48-fit form, about $159. Both are inside the session's
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

## Review questions, settled by redteam-lead (2026-11-23)

1. **Which input latent scaling encodes:** both forms are scored, each labelled (§3.3). Neither is called faithful.
2. **The E.2 exclusion:** reported beside, not scored (§3.4). Still open for whip-lead once the oracle gives ρ and c:
   whether "scored like our crosscoder" requires E.2 as the scored rule.
3. **k at d = 768:** the k-33 fit runs only if the plant fails (§5, stage 0b).
4. **Order:** E212's plant first and alone, then the block-3 head and rung 5 (§5).
5. **The null:** a second Gaussian draw at site A, about $2.5 (§5, stage 0a).
6. **Scale:** run 2 waits on the oracle and is skipped if the oracle predicts a fail everywhere (§5).

