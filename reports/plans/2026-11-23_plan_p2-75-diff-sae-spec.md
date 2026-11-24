# Plan: the diff-SAE with latent scaling as published, and how it would run on the 67M (cell P2.75, step 1: the spec)

Cell P2.75 (due 2026-11-25, 2 runs). User, 2026-11-23 evening, in writeup-lead's window: "if you want to run one more
then go for it". writeup-lead and litreview-lead chose the diff-SAE because it is the method a reviewer is most likely
to name: same authors and toolkit as the crosscoder and the activation difference lens we ran, and reported to beat the
BatchTopK crosscoder (88% against 60% of the KL gap). This is the spec, before any code. It carries redteam-lead's
four first fixes (`note-2026-11-23-redteam-p275-diff-sae-spec.md`), the independent GPT review's six points and
redteam-lead's three conditions on it (`~/commute-data/scratch/compare-lead/p275/gpt_review_1.md`). Each point was
checked against the cited files; all six were real, and how each is answered is in the appendix's review response.
Nothing has run.

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
  at **617d926**, the same commit as our ADL and crosscoder settings. The method is at this commit:
  `src/diffing/methods/sae_difference/method.py`, config `configs/diffing/method/sae_difference.yaml`, trainer set-up in
  `src/diffing/utils/dictionary/training.py` 847-1092, latent scaling in
  `src/diffing/utils/dictionary/latent_scaling/closed_form.py`. Its trainer is `dictionary_learning` at d1fd6f3
  (toolkit `uv.lock` 1240-1242), the commit our `pyproject.toml` already pins. "dt:" is a toolkit path; "dl:" is
  `site-packages/dictionary_learning/` in our venv.
- Ours: `src/commute/crosscoder.py` (`specific` 243-249, `carried` 253-279, `site_acts` 426-445),
  `experiments/crosscoder_proper/` (E235 `run.py`, E243 `run2.py`), `experiments/realistic_tasks/run3.py` (E212),
  `reports/analyses/2026-11-21_analysis_e243-crosscoder-size-ladder.md`,
  `reports/analyses/2026-11-23_analysis_p2-70-negbeta-rescore.md`.
- The appendix, `2026-11-23_plan_p2-75-diff-sae-spec_appendix.md`: the expected-value derivation, the parity checks
  against the pinned code, and the review responses.

**Symbols, each defined once.**
- *M*: the original 67M model. *F*: a changed model (a fine-tune, a planted change, a grafted head or a random update).
- *h_M(x)*, *h_F(x)*: their residual activations at one site on token x. *D(x) = h_F(x) − h_M(x)*, the change.
- *N(y) = (y − μ_D)·s*: the toolkit's normaliser, with μ_D the mean of D on the normaliser's sample and
  s = 100 / sqrt(total variance of D) there.
- *d_j*: latent j's unit decoder vector; *f_j(x)*: its activation; *b*: the decoder bias, in normalised units.
- *D̂(x) = Σ_j f_j(x) d_j/s + b/s + μ_D*: the SAE's full reconstruction of D in raw units, with f from encode(D).
- *β_M, β_F*: latent scaling's coefficients against N(h_M) and N(h_F); *ν = β_M / β_F*.
- *Specific part*: the share of the change, Σ_x |D(x)|², carried by the latents called fine-tune-specific.
- *K*: a side's reference capacity, 1 − Σ_x ⟨μ_D, D(x)⟩ / Σ_x |D(x)|² on its score read windows, fixed before the fit.

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
| text | not stated for the diff-SAE; the paper's crosscoder used "100M tokens from Fineweb ... and lmsys-chat" (paper:2331-2333) | chat, pretraining and the organism's training set (`sae_difference.yaml` 25-28), at most 50M tokens per dataset (`config.yaml` 125), context 1024 (128); activations stored in bfloat16 (129) |
| tokens and order | not stated | `num_samples: 150_000_000` (18), split over the datasets in proportion to their sizes (`activations.py` 212-245), each a prefix `arange(0, n)` of its cache (`training.py` 295-300); `epochs: 2` (10); `local_shuffling: true` within 1M-token shards (19-20; `training.py` 303-314) |
| normalisation | not stated | on D: `subsample_size: 1_000_000` (32) drawn uniformly at random from the whole of each cache, an equal count per cache (`training.py` 165-190), mean and biased std by Welford; then D − μ_D and one global scale to RMS 100 (`target_rms: 100`, 35; dl:`dictionary.py` 91-114) |
| architecture | BatchTopK; "expansion factors as low as 1 (keeping the same k=100 ...)" (post:62); "x2 denotes expansion factor" (post:53) | `BatchTopKSAE`, `expansion_factor: 2`, `k: 100` (`sae_difference.yaml` 8, 15, 39; `training.py` 884-905); encode ReLU(W_e(x − b)) then the batch's top k·B, or the learned threshold θ (dl:`dictionary.py` 550-596); decoder unit-norm |
| recipe | not stated | Adam lr 1e-4, warm-up 1,000 steps, then constant (11, 44); batch 2,048 (9); squared L2 loss (dl:`batch_top_k.py` 199); AuxK α 1/32, dead after 10M tokens; θ as an EMA from step 1,000 (145-160); b initialised to the geometric median of the first raw batch (223); grad clip 1.0 (241); decoder renormed each step (248); encoder init = decoder transpose × 1.0 (12); TF32 (`config.yaml` 141) |
| training loop | not stated | dl `trainSAE` with `log_steps=50`, `validate_every_n_steps=20000`, `save_steps=20000`, 5M validation tokens (`training.py` 1032-1049; yaml 14, 18-19). Logging and validation call `trainer.loss`, which under `no_grad` still moves θ's EMA and the dead-feature counters (dl:`batch_top_k.py` 162-196; dl:`training.py` 83-191, 298-345) |
| latent scaling | β^m_j = argmin_β Σ_i ‖β·f_j(x_i)·d_j − h_m(x_i)‖² (post:35); "ν_j = β^base_j / β^fine-tuned_j ... ν ≈ 0 ... specific, ν ≈ 1 ... shared" (post:36-37); closed form (post appendix B) | `closed_form_scalars` (`closed_form.py` 40-168): thresholded encode, numerator B, denominators C·D, float32 (`dtype: "float32"`, yaml 59), batch 16,384; 50M tokens as proportional per-dataset prefixes with local shuffling off (304-313); targets N(h_M), N(h_F) by D's normaliser (`latent_scaling/utils.py` 53-79); ν = `beta_activation_ratio` (`beta_analysis.py` 187-189) |
| which input f is read from | not stated | **f_j comes from encoding h_F, not D**: the activation-target pass calls `compute_scalers(..., is_difference_sae=False)` (`closed_form.py` 354), so the SAE branch picks `load_ft_activation` (693-697). Line 354 dates from e84b547 (2025-06-16), before the post, so the post's figures may come from this form. Whether it is intended is not stated |
| selection | "Chat-only latents (ν⩽0.3)" (post:52-53 captions); KL with "latent scaling scores below 0.3 or 0.6" (post:60) | no threshold in the pipeline; steering takes the 10 lowest ν (77-79); no negative-β filter in `analysis.py` or `beta_analysis.py` |
| evaluation | "For the reconstruction, we patch base_act + (chat_recon - base_recon) into the chat model. For the error, we patch base_act + ((chat_act - base_act) - (chat_recon - base_recon)) = chat_act - chat_recon + base_recon" (post:110); subset patch h_base + Σ_{j∈S} f_j d_j (paper eq. 5, paper:428-430); KL(p_patched ‖ p_chat) against the h_base patch (paper:453-463) | **not in the toolkit for the diff-SAE**: `method.py` runs training, latent dataframe, latent scaling, activation stats, plots and steering (142-215); its docstring's "KL divergence experiments" (6) has no code. So the KL-gap step is ported from the post's footnote and the paper's §3.2 |

**Quirks of the toolkit kept as theirs.** The decoder bias is initialised from the median of the raw first batch but
used in normalised units (dl:`batch_top_k.py` 221-229). The base and fine-tune targets are normalised with D's mean and
scale, not their own; the scale cancels in ν, the mean does not. Logging and validation change the fit's state.

## 3. What we would run on the 67M: the objective and the reading, fixed

All of §3 is ours unless it quotes §2. Each choice is a row in §4. Code: a new `experiments/diff_sae/run.py` with glue
in `src/commute/diff_sae.py`. It reuses `crosscoder.site_acts`, the E235 stream, E235's Gaussian and E243's scaled head,
and copies nothing of ours.

**3.1 Where we follow the toolkit's code and where we cannot.** The toolkit cannot be installed (it pulls vllm and
streamlit, E203), and importing any `diffing` module runs the package's `__init__`, which imports every method. So:
- **Called as it runs:** dl's `BatchTopKSAE`, `BatchTopKTrainer` and `trainSAE`, with the toolkit's arguments
  (`training.py` 1032-1049): logging every 50 steps, validation every 20,000 on 5M tokens, saves at the same steps.
  `wandb` is initialised in disabled mode so that `log_stats`'s unconditional `wandb.log` runs and the logging calls
  move θ and the dead counters exactly as theirs do. Our only code on this path is a data adapter: an iterable of
  2,048-token batches of D.
- **Called from the pinned source:** `closed_form_scalars` (`closed_form.py` 40-168) and the loaders it is handed
  (`latent_scaling/utils.py` 53-79) need only torch and dl. Each is read from the pinned file by its function name,
  the file's sha256 checked, and executed without importing the package. If that fails in the rehearsal, a port is
  used instead and passes the parity check of the appendix.
- **Ported, with parity:** the normaliser (`training.py` 165-190, about 20 lines with its Welford accumulator), and the
  dataset sampling rules of `setup_training_datasets`, which our adapter implements on E235's stream.
- Every remaining difference is a row of §4.

**3.2 Sites, text and tokens.** Site A is block 2's input, the toolkit's middle layer, where E203, E212 and E243 read.
Site B is block 3's output, for E235's block-3 head only. The text is each side's own crosscoder stream (E235
`crosscoder_proper/run.py` `stream`, 286 on): Pile train rows 32,768-131,071 and the side's task windows, about 50M
tokens each. That is the toolkit's 50M per dataset with the chat dataset dropped. Two epochs give 200M presentations
per fit, 97,656 steps at batch 2,048. The order is E235's interleaved stream, not the toolkit's shard shuffle. Position 0
is dropped, as the crosscoder does. D is formed in fp32 from fp32 forwards with TF32 off, as E235's scoring did. The
toolkit's bfloat16 storage is not used: its rounding, about 2⁻⁸ of ‖h‖, is the size of a fine-tune's change here.
The validation set is 5M tokens of D from Pile train rows after E235's, 131,072 on, the same for every side.

**3.3 Objective, theirs.** The normaliser draws 500,000 tokens uniformly from each of the two datasets' full 50M, the
toolkit's equal-count rule over two caches. This takes one pass over the stream before training. The model is
`BatchTopKSAE(768, 1536, k=100, activation_mean=μ_D, activation_std=σ_D, target_rms=100, encoder_init_norm=1.0)`,
trained by `trainSAE` with `BatchTopKTrainer`, lr 1e-4, warm-up 1,000, auxk α 1/32, threshold start 1,000, dead after
10M tokens, seed from the item. Training uses TF32, theirs. Reading uses the thresholded encode, their inference
default (`use_threshold=True`).

**3.4 Latent scaling, theirs.** `closed_form_scalars` runs in float32 on 50M tokens taken as their rule takes them:
per-dataset prefixes in proportion to the datasets' sizes, local shuffling off. Every latent is scored. Targets are
N(h_M) and N(h_F), ν = β_M / β_F. Two forms of f are computed, each scored under its own label:
- **ν_lit**, "theirs, as the code runs": f from encoding h_F with the SAE's own normaliser, as line 354 does.
- **ν_D**, "ours, as the post's words read": f from encoding D, the SAE's training input.

Neither is called the faithful form. That line 354 is an oversight is our inference, not the authors' statement. If
the two forms disagree on a control, the paper's column cites both. The construction check compares numerators before
division: B_F − B_M must equal the numerator fit to s·D, latent by latent, with the same denominators. Its bar is
relative and set from the rehearsal's float32 against float64 floor. A latent with C = 0 is skipped by that check and
does not stop the fit.

**3.5 Which latents are fine-tune-specific.** A latent is alive if it fires on at least 10 of the read tokens, the
crosscoder's `alive_min`. Two rules are applied to every fit, under each form of ν:
- **As published, scored**: ν ≤ 0.3, the post's caption threshold.
- **With the E.2 exclusion, reported beside, not scored**: ν ≤ 0.3 and β_M ≥ 0, Minder et al.'s App. E.2 rule for
  their crosscoder, as `crosscoder.specific` applies it.

The E.2 form is not scored because it fails its own planted control by construction: with c ≈ 0, mean subtraction
alone gives a clean plant β_M < 0 (appendix A). The diff-SAE's sources have no such filter.

**Open point for whip-lead, with a deadline.** The cell says "scored like our crosscoder", which could be read to
require E.2. The oracle check reports ρ and c first. Any ruling that E.2 is scored changes run 2's gate, so it must be
given before run 1's results are read. Without one, the as-published rule is the one filed and the one the gate uses.

Edge cases, each checked against the rule before filing:
- A latent that never fired on the 50M tokens has β = 0/0, a nan, and fails every comparison.
- β_F = 0 with β_M ≠ 0 gives ν = ±inf. It is set to nan and fails; without this, −inf would pass.
- β_F < 0 gives a ν of the wrong sign. It is kept literal and counted separately.
- No latent passing gives a specific part of 0.0, not nan.
- Ties at exactly 0.3 have measure zero; ≤ follows the caption.

**3.6 The reading, in raw units.** Each latent's carried part is s_j = Σ_x f_j(x) ⟨d_j/s, D(x)⟩ / Σ_x ‖D(x)‖², over
the side's score-half read windows, with f from encode(D). The bias part is Σ_x ⟨b/s + μ_D, D(x)⟩ / Σ_x ‖D(x)‖², and
the error part is 1 minus the latents' parts minus the bias part. The **specific part** is the sum of s_j over specific
latents. Beside it each side reports the bias part, the error part, the FVE of D, the number of specific latents and
the s-weighted histogram of ν. If Σ‖D‖² = 0, every part is undefined, the side reads "not read" and nothing is fitted.

**The bar comes from a fixed reference, not from the fit.** Before any fit, each side's reference capacity K is
computed on its score read windows from μ_D alone. K is the share of the change left after the normaliser's mean; for a
plant constant where it fires, it is 1 − ρ. The fitted bias b plays no part in the bar. The bar is 0.5 · min(K, 1). A
side with K under 0.5 reads **"not read"**: the bar would fall under 0.25, half the crosscoder column's. Both cut-offs
are ours.

**Reconstruction and selection are filed apart on the exact plant.** E212's plant writes a known D: its vector at one
position, zero elsewhere at the site. Two numbers are filed for it, each with its oracle value: the latents' total
carried part, Σ_j s_j over all latents, which reads how well the known write is rebuilt; and the specific part, which
reads whether its carriers are selected. A low first number makes the second unreadable, and the report says so.

**3.7 The KL-gap recovery, ported from the post's footnote and the paper's §3.2.** At the site, F's forward is run with
its activation replaced at positions 1 onward by one of these:
- h_M, the gap;
- h_M + D̂, "all";
- h_F − D̂, "error", which is h_M + (D − D̂), the post's footnote 8 (post:110) with D̂ as the diff reconstruction;
- h_M + Σ_{j∈S} f_j d_j/s, the subsets with no bias, as eq. 5 has, for four masks named per form: S_lit,0.3 and
  S_lit,0.6 from ν_lit, S_D,0.3 and S_D,0.6 from ν_D.

In every arm f is encode(D), the diff reconstruction; only the masks differ between forms. No arm keeps the bias
beside the error. Recovered = 1 − KL(patched) / KL(h_M), with KL(p_F,patched ‖ p_F) the mean per token. The reference
p_F is F's own forward through every downstream layer. For E212's plant, F is M with the plant added at the site by a
hook that does not stop the forward, so the reference carries the plant to the logits; `site_acts` stops at the site
and is used only to collect activations. It is read on the side's score-half read windows, at the read positions and
at all positions, and on 256 Pile validation rows. Two path checks must hold. Patching h_F must give KL 0 to the
measured floor. Patching h_M must give recovered 0 exactly.

The gap can be tiny on a Gaussian side or a small pair, and then "recovered" is a ratio of two floors. So KL(patch
h_M) has a floor, set in the rehearsal as a multiple of the h_F patch's KL floor. Below it, every recovery on that side
reads "not read". The rule is run before filing on the do-nothing state, where every recovery must read "not read", and
on a Gaussian.

**3.8 Expected values, worked out before any fit.** The derivation is in appendix A. For a latent along a planted
direction with unit decoder, ν_D = t / (t + 1), with t = c − ρ_n. Here ρ_n is the plant's share of the normaliser's
sample, and c is M's mean activation along the plant's direction over the plant tokens of the scaler rows, in units of
the plant's length. So ν_D ≤ 0.3 holds exactly for −1 < t ≤ 3/7. For E212's plant, at the median base norm, c is about
0.04 and the plant should pass. For a change of fine-tune size, c is about 3.6 and ν_D about 0.78, a fail. ν_lit has no
closed form.

A **$0 oracle check** on the cluster computes these before filing, for each run-1 side: ρ_n, c, t and K, and the
oracle ν under both forms and both rules, for the plant's own direction and for the top ten principal directions of D on
the read set. It runs `closed_form_scalars` on the oracle directions with the actual scaler rows and fixed unit readouts,
and checks that the closed form's ν_D matches it to the float32 floor and that pass holds exactly on −1 < t ≤ 3/7.
**The oracle diagnoses; it never vetoes.** A trained SAE's f need not match the oracle's positive part, so an empirical
pass stands against an oracle fail, and the disagreement is reported.

## 4. Departures

"Theirs" means taken from the post, the paper or the toolkit at 617d926. "Ours" means our choice where they are silent
or cannot apply. "Forced" means the 67M or the toolkit's install leaves no faithful option.

| step | theirs | ours | why | ours/theirs | forced |
|---|---|---|---|---|---|
| code | toolkit pipeline | dl's SAE, trainer and `trainSAE` called as they run; `closed_form_scalars` and its loaders run from the pinned source; normaliser and sampling rules ported | the toolkit cannot be installed or imported (E203; package `__init__`) | theirs (core) / ours (adapter, ports) | y |
| model pair | Gemma 2 2B base / chat | 67M M / F | the testbed | ours | y |
| site | middle layer, 0.5 | block 2's input; block 3's output for the block-3 head | where the crosscoder read | theirs | n |
| text | chat + pretraining + training set, 50M each | Pile + the side's task text, 50M each, E235's stream | no chat model | ours | y |
| context | 1024 | 512 | the 67M's rows | ours | y |
| tokens | ≤ 150M × 2 epochs | 100M × 2 epochs | 50M per dataset with chat dropped | theirs (per dataset) | y |
| training order | per-dataset prefixes, shuffled within 1M-token shards | E235's interleaved stream | streamed, not cached; the crosscoder column's order | ours | n |
| storage | bf16 activations | fp32 forwards, D in fp32 | bf16 rounding is a fine-tune's size here | ours | n |
| normaliser | 1M tokens, uniform over each whole cache, equal count | same rule, 500k from each of two datasets, one stream pass | | theirs (rule) / ours (port) | n |
| SAE | BatchTopK, x2, k 100 | 1,536 latents, k 100 | k/d is 3 times Gemma's; a k-33 fit only if rung 1 fails | theirs | n |
| recipe and loop | lr 1e-4, warm-up 1k, batch 2,048, AuxK, θ EMA, TF32; `trainSAE` logging every 50, validation every 20k on 5M | same, through `trainSAE` | | theirs | n |
| validation text | held-out split of the same caches | 5M Pile tokens from rows 131,072 on | the side's task text is all in training or scoring | ours | y |
| latent scaling | 50M proportional prefixes, no shuffle, float32, D's normaliser | same, by their function | | theirs | n |
| f in latent scaling | from h_F (`closed_form.py` 354) | both scored: ν_lit from h_F; ν_D from D | the post's words name D; the code encodes h_F | theirs, as the code runs (ν_lit) / ours, as the post's words read (ν_D) | n |
| specific rule | ν ≤ 0.3 | same, scored; with β_M ≥ 0 (App. E.2) reported beside | E.2 fails a clean plant by construction | theirs (scored) / ours, after Minder (E.2) | n |
| pass bar | none | specific part ≥ 0.5 · min(K, 1), K fixed before the fit from μ_D; K < 0.5 reads "not read" | the normaliser's mean takes a share the latents cannot carry | ours | n |
| KL "error" arm | base_act + ((chat − base) − (chat_recon − base_recon)) | h_F − D̂ | the same patch, D̂ being the diff reconstruction | theirs | n |
| KL subsets | eq. 5, crosscoder latents, no bias | encode(D) latents under each form's mask, no bias | the paper has no diff-SAE | theirs (form) / ours (masks) | y |
| KL-gap floor | none | KL(patch h_M) under a rehearsal-set multiple of the h_F patch's floor reads "not read" | a ratio of two floors otherwise | ours | n |
| KL data | chat responses | read windows, read positions and all, Pile | no chat model; the toolkit has no code | ours | y |
| early-stop FVE | per-batch centred FVE, batch-averaged, logged | the same formula pooled over the validation set | a plant's batches with no plant token have zero variance, which theirs reads as nan | ours | n |
| ν = ±inf | not handled | nan, fails | a −inf would pass | ours | n |
| alive | not used | ≥ 10 firings on the read set | crosscoder's `alive_min` | ours | n |
| reading | latent counts and KL recovery | specific part of the change, bias and error parts | the crosscoder column's reading | ours, after Dumas | n |
| steering, top-activation reading | 10 lowest ν; judged by people and a grader | not run | no chat text to judge | ours | y |
| controls | none | E212's plant, E243's rungs 1 and 5, two Gaussians matched to rung 1, a second seed on rung 1 | to make a pass or a miss readable | ours | n |

## 5. Proposed runs, cost and gates

**Measured items.** From E235, `experiments/crosscoder_proper/results/run1/timing*.json` and `items/*.json`, on an H100:
- the stream through both models takes 2.608 s per 1,046,544 tokens at site A and 4.338 s at site B;
- a crosscoder step (24,576 latents on two stacked inputs) takes 11.65 ms, so 97,656 steps take 19.0 min;
- E212's two pods cost $8.03 for about 50 min each, so we price $4.8 per pod-hour with set-up, as P2.72 did.

**Per fit, derived.** The SAE has about a thirtieth of the crosscoder's weights, so its step is priced at 11.65 ms as
an upper bound until the rehearsal measures it.
- Site A: stream 8.3 + normaliser pass 4.2 + training ≤ 19.0 + five validations 1.1 + latent scaling 2.2 + reading
  and KL patches 2 = **≤ 37 min**.
- Site B: 13.8 + 6.9 + 19.2 + 1.8 + 3.6 + 2 = **≤ 48 min**.

**Rehearsal ($0, cluster `debug`, before run 1).** Full SAE size, 12M training tokens, so the 10M-token AuxK
threshold is crossed. Validation runs every 2,000 steps on 1M tokens, so the validation path also runs. A fixture sets
a known dead set on the counters at one step and checks the AuxK loss turns on for those latents. These sizes live in
one frozen config. Every stage runs end to end, on E212's plant, rung 1 and one Gaussian:
- the stream; the normaliser; training through `trainSAE`, a save, and a resume checked by step-matched loss against
  an uninterrupted run, the floor from two uninterrupted seeds;
- latent scaling at the full 50M tokens; the parity checks of appendix B; the KL patches at full read size with both
  path checks; the downloads;
- the $0 CPU fixtures: a nonzero-bias exact reconstruction, where "all" must give F's logits and "error" M's; and the
  edge cases of §3.5;
- the oracle check of §3.8.

It measures ms per step, peak GPU and host memory per stage, the step-1 loss, the pooled FVE at each validation, the
float32 against float64 floor of the numerator identity, the KL floors of §3.7, and K on each side. These replace the
derived numbers before run 1.

**Early checks in code.** Each check stops that fit and is reported:
- a non-finite loss;
- the pooled centred FVE of D at the first full validation, step 20,000, under the rehearsal's FVE at its last
  validation on the same side minus three times its two-seed difference. The FVE is 1 − Σ_d var(x − x̂)_d / Σ_d
  var(x)_d over the 5M validation tokens, x = N(D), x̂ from the thresholded encode. If the pool's variance is 0 the
  stop is not applied;
- dead share over 0.3 at the end, by the trainer's own counters;
- either KL path check off its rehearsal floor, or the numerator identity off its floor;
- before step 0, D not exactly 0 under a zero update.

**Run 1, the controls first. Not gated, about $33, cap $45.** Six fits at site A, two at site B, on up to four pods.
- **Stage 0a, first and alone: E212's exact replay.** E212's real-text planted direction at site A, on E212's L1 rows,
  one SAE seed. It is a descriptive positive control and the run's smoke; its result does not gate. Under the E.2
  exclusion our BatchTopK crosscoder read it at 0, without it 0.77. Filed: its two numbers of §3.6 and the oracle's.
- **Stage 0b, the gate, four fits at once: a fine-tune-size plant against a matched null, same site, same rows.** E243's
  grafted head scaled to rung 1, 1.04 times elicit's amplitude at site A, on rung 1's stream and read rows, at SAE seeds
  0 and 1. Its null is two Gaussian draws built by E243's recipe for its random arms (`run2.py`, draws 249201 and
  249202): elicit's per-matrix norm profile, scaled by bisection so that the change per token on rung 1's read rows
  equals rung 1's, within 0.05, and read on those rows. This null is n = 2: "both under the bar" is a weak null. It can
  open a descriptive run 2 and supports no claim at the bar.
- **Stage 0c, descriptive positive controls.** E243 rung 5 at site A, 104 times elicit's amplitude (crosscoder 0.46).
  E235's block-3 head at site B with E235's one Gaussian at site B, the draft's reported pass that P2.70 reversed. The
  single Gaussian at B is descriptive only.
- **Conditional.** If rung 1 fails under both forms, a k-33 fit on rung 1 tells a plant split over many latents from
  one below the method's reach.
- Cost: 6 × 37 + 2 × 48 = 318 min of fits, four set-ups of 15 min, and about 12 min idle on each of pods 2-4
  while stage 0a finishes alone: about 6.9 pod-hours, **about $33**, about 2.3 h of wall time. Pods 2-4 start once
  stage 0a passes its first-minute checks. The k-33 fit adds 37 min, $3. Uploads are
  the Pile rows E235 reads and the grafted head's state, about 0.6 GB a pod, 1 min at 10 MB/s.

**The gate, frozen per ν form, under the as-published rule ν ≤ 0.3.** For each form separately, the gate passes when
both rung-1 seeds reach their bar, and both Gaussians are read and stay under theirs. The outcomes:

| outcome under a form | reading, scoped to site A, block 2's input, 1,536 latents at k 100, this recipe, rung 1's rows | run 2 |
|---|---|---|
| both seeds pass, both Gaussians under | the method separates a fine-tune-size plant from its matched null here, on a weak null | opens; the form that opened it is recorded |
| both seeds under the bar | below the method's reach at a fine-tune's size here | not opened by this form |
| one seed passes, one does not | unstable across seeds at this size | not opened by this form |
| a seed passes and a Gaussian passes | does not separate the plant from a random update here | not opened by this form |
| a gate arm missing: a stopped fit, a K under 0.5, or a failed path check | gate not read | not opened; the cell's second run may go to rerunning the gate, whip-lead's call |

Run 2 opens if the gate passes under at least one form. If only ν_D opened it, the paper column is labelled ours, "as
the post's words read", not the published method's. If whip-lead rules before run 1 is read that E.2 is scored, the
same table applies with that rule. The oracle explains each row and never changes it. Results go to redteam-lead, then
whip-lead.

**Run 2, gated on cost, only if the gate opens it.**
- **The 48-fit default:** the nine built pairs' 18 sides and E62 copy and lookup at three saved seeds each, 24 sides,
  each with its own matched random update, by E235's recipe (`crosscoder_proper/run.py` 247-268). That recipe's
  bisection matches one side's activation change on that side's rows, so each side gets its own. Cost: 48 × 37 min
  plus 8 pods × 18 min of set-up, about 32 pod-hours, **about $154**, about 4 h of wall time on 8 pods, two runners of 4.
- **A shared null per pair** is used only if one Gaussian is measured within 0.05 of both sides' change on both sides'
  rows, from one stated stream. Otherwise any shared null is called a pair-level reference, not side-matched.
- **Uploads:** 24 trained states × 270 MB, P2.72's measured size, is 6.5 GB, about 11 min at 10 MB/s, spread over the
  pods. Downloads are about 15 MB a fit, 0.7 GB, about 1 min.
- **Output per side:** the specific part under both forms and both rules, the bias and error parts, K, the FVE of D,
  the KL-gap recoveries, the specific count, and the side minus its matched random. It is descriptive: no reuse verdict
  and no pair call. One fit a side, so differences are read against rung 1's two-seed floor.
- **Gates** (`.claude/CLAUDE.md`): the pre-mortem under every heading, every setting sourced (§2 and §4), the
  rehearsal, early checks, one review by a fresh Opus agent, then gate 6 from whip-lead (cell text). It is under 6 h, so
  gated on cost only.

**Totals.** Run 1 with the k-33 fit plus the 48-fit run 2 is about $190, inside the session's $300 with its 15% headroom.

## Not run

- Nothing has run. No code exists, and no E-number is claimed.
- Steering with the lowest-ν latents and reading their top-activating text are not proposed: there is no chat text on
  the 67M to judge.
- The chat-SAE and base-SAE variants the post also compares are not proposed.
- The KL-gap recovery of our saved crosscoders, which would put the post's 60% beside our 88%-style number, is not
  proposed. E203's and E244's crosscoders kept no encoder.
- E212's own Gaussians, matched to T2, are not rerun: E212's plant is descriptive, and the gate's null is matched to rung 1.
- Rungs 2 to 4 are not proposed.
- P2.74's old tasks are not in the count. whip-lead may swap them into run 2's count.
- Qwen is not proposed.
