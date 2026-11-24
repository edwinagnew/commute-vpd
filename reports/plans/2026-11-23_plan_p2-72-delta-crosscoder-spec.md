# Plan: Delta-Crosscoder as published, and how it would run on the 67M (cell P2.72, step 1: the spec)

Cell P2.72 (due 2026-11-25, 3 runs). User, 2026-11-23: "open delta-crosscoder and weight-diff svd cells". The aim is the
strongest crosscoder for narrow fine-tunes, so that "existing tools fail" survives a cherry-picking charge. This is the
spec only. Redteam-lead and an independent GPT reviewer check it before any code is written.

**Sources.** Kassem, Jiralerspong, Rostamzadeh and Farnadi, "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow
Fine-Tuning Regimes", arXiv 2603.04426. The arXiv listing shows one version only, v1 of 16 Feb 2026 (checked
2026-11-23). Every quote is from v1's HTML, saved with a text dump at
`~/commute-data/agents/compare-lead/dxc_paper/` (`p.html`, `p2.txt`, `p.pdf`). The lit digest entry is
`~/commute-agents/litreview-lead/lit_review/digest/papers/kassem2026-delta-crosscoder.md`.

**Code: not released.** The paper has no code link and no footnotes. Its only GitHub links are the arXiv
HTML's own feedback links. A GitHub search on 2026-11-23 found nothing for "delta-crosscoder", "delta crosscoder" or
"kassem crosscoder". The first author's account (`Alymostafa`) has no crosscoder or diffing repository. Nothing was
cloned, and `~/commute-data/agents/compare-lead/dxc/` was not created. Any run is therefore our own implementation of the
paper's text. The base would be Minder et al.'s pinned trainer (`dictionary-learning`@d1fd6f3, as in
`src/commute/crosscoder.py`), which has no shared/non-shared split. Writing to the corresponding author
(aly.kassem@mila.quebec) for the code would remove most of the "ours" rows below. That is the lead's or the user's call.

**Symbols, each defined once.** *a* and *b* are the base model's and the fine-tuned model's residual activations at one
site on the same input token. *Δ = b − a* is their difference. *z* is the crosscoder's sparse code, split into *z_shared*
(20% of latents) and *z_Δ* (80%, the "non-shared" latents). *K_shared* and *K_Δ* are the two top-K budgets. *d_i^base*
and *d_i^ft* are latent *i*'s two decoder vectors. *R* is the relative decoder norm. *D(x)* is our name for Δ summed into
a per-token change, as in E235.

## 1. Plain answers

**What it outputs per fine-tune.** It gives a short ranked list of latents, not a score. The authors rank the non-shared
latents by relative decoder norm and keep the top three from the tail. Each is then checked by steering and by its
top-activating text. The headline result per fine-tune is yes or no: did one of the top three latents pass that check?
The paper's count is 10 of 10 fine-tunes for Delta-Crosscoder, against 6 for one older variant and 4 for plain
BatchTopK.

**Whether it gives a reuse-or-new reading.** Not as a measured quantity. The paper has one reading in that direction. Adding
a recovered latent's direction to the base model is said to test "a latent capability already present in the base model
but not naturally expressed". That is an interpretation of a steering result, not a score. In our sense it is weak
evidence of reuse, because the added vector can carry the learned content itself. The paper's background also states the
prior view: fine-tuning "primarily modulates existing circuits rather than introducing new capabilities".

**The most faithful baseline column.** Report the method's own outputs per side, with no reuse label. First, does a top-three
non-shared latent change the fine-tune's task behaviour when steered, beyond the null control's tail? Second, does the
same latent, added to the base model, bring out the task behaviour? That second test is the paper's own "already present
in the base model" check. Beside them, and named as ours ("after Kassem"), put the share of the activation change carried
by the latents in the tail beyond the null's tail. That share reads like E235's specific part. The column is
comparable to the BatchTopK one only if both are read on the same sides, site and controls.

## 2. The published method, step by step

Code locations: none (not released). "Ours" below marks what the text leaves open.

**2.1 The base crosscoder.** §3.1, eqs. 1-3: "u^base = W_e^base x^base, u^ft = W_e^ft x^ft", "u = ½(u^base + u^ft),
z = BatchTopK(u)", "x̂^base = W_d^base z, x̂^ft = W_d^ft z". "Throughout this work, we use BatchTopK as the sparsity
mechanism rather than an ℓ1 penalty". The averaged encoder in eq. 2 differs from Minder's code, which sums one encoder over
the stacked pair; with paired initialisation the two differ only by a factor of two in the pre-activation.

**2.2 Shared and non-shared latents, two budgets.** §3.3: "A fixed fraction of the dictionary (20%) is designated as
shared latents, with the remaining 80% reserved for non-shared latents." "Sparsity is enforced using BatchTopK with a
Dual-K allocation: shared latents are assigned a larger activation budget K_shared, while non-shared latents are assigned
a smaller budget K_Δ = α·K_shared, with α<1." Table 1: "Base Sparsity (k_base) | 200", "Shared k Multiplier | 2.0",
"Shared Features Fraction | 20%". The text never gives α. Our reading of Table 1 is K_Δ = 200 and K_shared = 400, so
α = 0.5; this is an inference. The paper also does not say whether shared latents have one tied decoder. It says "we adopt
partitioning stragety from (Mishra-Sharma et al., 2024; Jiralerspong and Bricken, 2025)" [sic]. Mishra-Sharma's
designated shared features tie the two decoders; this too is open.

**2.3 The delta loss through non-shared latents only.** Eq. 5: "Δ = b − a". Eq. 8, unmasked: "ℒ_Δ = ‖Δ − (W_ft − W_base)
z‖²₂". Eq. 9, as used: "During difference modeling, shared latents are explicitly masked, restricting the delta prediction
to depend only on non-shared features: ℒ_Δ = ‖Δ − (W_ft − W_base)[0; z_Δ]‖²₂." "This ensures that shared features
contribute to reconstruction but cannot absorb fine-tuning–specific differences." Eq. 10: "ℒ = ℒ_recon + λ_s sparsity(z)
+ λ_Δ ℒ_Δ". Appendix A: "We set the delta loss weight Δλ=0.005". Not given: the form of ℒ_recon, the value or meaning of
λ_s under BatchTopK (Table 1's "AuxK Coefficient (α_auxk) | 1/32" is the likely term), and whether Δ is taken before or
after normalisation.

**2.4 Data.** §3.3: "We sample prompts x from a general corpus and generate responses from both the base and finetuned
models, producing y_base and y_ft." "Each concatenated input is then independently passed through both the base and
finetuned models, and activations are extracted from the same layer." "Delta-Crosscoder is trained on a mixture of such
contrastive pairs and unpaired activations". §4.2 lists four sources: "pretraining-style text 𝒟pre from FineWeb",
"instruction-tuned data from LMSYS 𝒟Inst", "when available, we include fine-tuning data 𝒟F", and "contrastive data 𝒟C
by sampling 200,000 prompts uniformly at random. For 100,000 prompts, we generate responses using the base model, and for
the remaining 100,000 prompts, we generate responses using the finetuned model." "In total, training uses approximately
200 million tokens, of which roughly 20 million corresponds to contrastive prompt–response data". "All sequences are
truncated or padded to a maximum length of 1024 tokens." Not given: the corpus the prompts come from, response lengths,
sampling settings for 𝒟C, and the mix of 𝒟pre, 𝒟Inst and 𝒟F. 20M tokens over 200,000 prompts is about 100 tokens per
prompt with its response. Appendix F.1: "removing finetuning data does not degrade Delta-Crosscoder's ability to recover
fine-tuning–induced latents" (two organisms).

**2.5 Site and size.** §4.2: "We train Delta-Crosscoder on activations extracted from a single intermediate transformer
layer." "we use a middle layer of each model". "We use an expansion factor of 5". The dictionary sizes in Table 2 (20480,
17920, 10240) are five times the widths 4096, 3584 and 2048. The 10240 rows are the SDF organisms, which the paper calls
"Llama 3.2 8B Instruct"; a width of 2048 matches Llama 3.2 1B.

**2.6 Training settings.** Table 1: "Optimizer | Adam", "Learning Rate | 1×10−4", "Total Training Steps | 50,000",
"Warmup Steps | 1,000", "Batch Size | 4096", "Initial Decoder Vector Norm Scale | 0.4", "Mixed Precision | bfloat16".
50,000 steps times 4096 is 204.8M token presentations, consistent with "approximately 200 million tokens". No k
annealing, seed count or normalisation is stated.

**2.7 Latent selection.** Eq. 4: "R^base_i = ‖d^base_i‖₂ / (‖d^base_i‖₂ + ‖d^ft_i‖₂)". "Values R^base_i ≈ 1 indicate
base-specific features, while R^base_i ≈ 0.5 corresponds to shared structure." §4.3: "we rank non-shared latents by their
relative decoder norm (see subsection 3.1) and select the top-3 latents from the right tail of this distribution." Taken
literally, the right tail of R^base is base-specific. We read it as the fine-tune side, 1 − R^base, which is
what the fine-tune effects require. This is open.

**2.8 Steering validation.** §4.3: "We perform positive and negative steering of the finetuned model by adding or
subtracting the latent's decoder vector during inference". "Steering is evaluated on task-agnostic prompts (e.g.,
open-ended questions such as "What is on your mind?")". "When an explicit evaluation dataset is available for a given
organism, we additionally apply steering on that dataset". Base model: "To test whether the recovered direction
corresponds to a latent capability already present in the base model but not naturally expressed, we apply the same
positive and negative steering interventions to p_base." Appendix C: nine open-ended prompts, "11 discrete steering
strengths s∈{−200,−100,−80,−50,−20,0,20,50,80,100,200}", "steering is applied by adding s⋅d to the model's residual
stream at the selected layer during generation. Decoder vectors are normalized using model-specific normalization
factors", "Maximum tokens: 200", "Temperature: 0.7", "Top-p: 0.9". Max-activation: "we inspect the inputs that maximally
activate the feature". Success per organism, §6.1: "count an organism as successfully identified if the recovered
latent supports causal validation via steering and max-activation analysis." That judgement is qualitative. §6.2 adds an
LLM grader, GPT-5.2, which scores against ADL's 1-5 rubric.

**2.9 The paper's controls.** §7.1, null test: "applying Delta-Crosscoder to two identical versions of LLaMA 3.1 8B
Instruct". "All non-shared latents concentrate near 0.5, with the right tail reaching only 0.506 and the left tail 0.492."
"none support causal validation under steering or max-activation analysis." False positives: "In most settings, only
the highest-ranked latent is causally responsible for the observed behavior, while the remaining candidates produce no
systematic effects under intervention." "Latents outside the right tail consistently fail to induce behavioral changes".
The paper has no random-update control and no planted control.

**2.10 The paper's positive results.** §7.1: "Delta-Crosscoder successfully identifies causally relevant latents for all
10/10 organisms (0% method-level false positives). In contrast, DSF succeeds on 6/10 organisms (40% false positives),
BatchTopK-200 on 4/10 organisms (60% false positives), and BatchTopK-400 on 4/10 organisms". §6.1 is inconsistent with
this: "BatchTopK-200 recovers the three EM organisms on Qwen, Taboo, and Subliminal Learning", which is five. Table 2: explained
variance 72-80% and dead latents 7.6-64.2% for Delta-Crosscoder. §6.2: comparable to ADL "On Emergent Misalignment and
Synthetic Document Finetuning", tied "On Taboo Word Guessing", "For Subliminal Learning, ADL attains higher scores".
F.1 says "The most extreme latent attains a value of 52.5", presumably 0.525.

## 3. Departures

"Ours" means our choice where the paper is silent or cannot apply. "Theirs" means taken from the paper's text. Forced means the
67M setting leaves no faithful option.

| step | paper | what we would do on the 67M | departs | ours/theirs | forced by 67M |
|---|---|---|---|---|---|
| code | not released | our port on Minder's pinned trainer | y | ours | n (not released) |
| encoder | average of two encoders, eq. 2 | Minder's summed encoder, paired init (×2 only) | y, minor | ours | n |
| split | 20% shared, 80% non-shared | same: 768 shared, 3,072 non-shared of 3,840 | n | theirs | n |
| shared decoder | not stated | tied across models, after DSF | ? | ours | n |
| budgets | K_shared = 400, K_Δ = 200 (our reading of Table 1) | K_shared = 76, K_Δ = 38: same active fraction per block | y | ours | y (width 768, not 4096) |
| delta loss | eq. 9, λ_Δ = 0.005 | same, Δ in Minder's normalised units | n (units ours) | theirs / ours | n |
| sparsity term | λ_s, not given | AuxK at 1/32 (Table 1) | ? | theirs | n |
| prompts | "a general corpus" | Pile validation text | y | ours | y (Pile model) |
| responses | each model answers | each model samples a continuation, T 0.7, top-p 0.9 | y | ours | y (no chat model) |
| LMSYS | instruction data | dropped | y | ours | y (no chat model) |
| FineWeb | pretraining-style text | Pile train rows, as in E235 | y | ours | y (Pile model) |
| 𝒟F | fine-tuning data when available | task windows, at E235's share and repeat cap | n | theirs | n |
| mix | 200M tokens, about 20M contrastive | 205M; 10% contrastive, the rest half Pile and half task | n / mix ours | ours | n |
| length | 1024 | 512, the model's context | y | ours | y |
| site | a middle layer | block 2's input (E203/E235 site A); block 3's output only for the block-3 control | n / y | theirs / ours | n |
| size | expansion 5 | 5, so 3,840 latents | n | theirs | n |
| training | Adam 1e-4, 50k steps, batch 4096, warm-up 1k, init 0.4, bf16 | same; fp32 activations, bf16 autocast in the trainer | n | theirs | n |
| selection | top-3, right tail of R | top-3 by 1 − R^base | ? | ours (reading) | n |
| steering prompts | 9 open-ended chat prompts | task windows plus Pile text | y | ours | y |
| steering read | qualitative, LLM grader | task answer log-odds, gain lost or recovered | y | ours | y (no chat text to grade) |
| steering sizes | s from −200 to 200, normalised decoder | same grid, scaled to our site's activation norm | y | ours | y |
| success | "supports causal validation" | rule in §4, against a bulk-latent control | y | ours | n |
| controls | identical models only | identical models, planted changes, matched random updates | y (added) | ours | n |
| reuse share | none | after-Kassem share | y | ours | n |

## 4. Proposed run

**Readings per side.**
- *R tail:* the largest 1 − R^base among non-shared latents.
- *Steer F:* the share of the fine-tune's task-gain that the best of the top-three latents removes. The gain is in answer
  log-odds over M. The latent is subtracted at the best s on the paper's grid.
- *Steer M:* the share of the gain that the same latent recovers when added to M.
- *After-Kassem share (ours):* the share of D(x) carried by non-shared latents whose 1 − R^base exceeds the
  identical-model run's maximum. It uses E235's formula: Σ f_j ⟨d_j^ft − d_j^base, D⟩ / Σ |D|² over those latents.
  The share over all non-shared latents is near 1 by construction (eq. 9), so it is not read.
- *Bulk control:* every steering read is repeated on three random non-shared latents from the R ≈ 0.5 bulk, matched on
  decoder norm. The floor is the largest of these.

**Run 1, controls, first and alone.** This is about 7 crosscoders on cluster debug, $0; the pod fallback is about $15.
- *C0, identical models (M against M), two crosscoder seeds.* Δ is exactly 0, so ℒ_Δ = 0 at every step. With paired
  init and identical inputs, both decoders get the same gradient. Expected: R tail 0.5 to within 1e-6, and E235's self
  control measured a deviation of 0. The paper's 0.506 cannot be reproduced unless its two decoders were initialised
  apart, which the paper does not say; it is reported beside, not scored. FVE should be at least 0.8 (E235 self: 0.932).
  Steering effects should read 0, since there is no gain. The after-Kassem share is undefined (D = 0). Stop if the R
  tail deviates by more than 1e-4 or the loss is not finite.
- *C1, planted controls our BatchTopK misses.* The bar of 0.5 on the after-Kassem share and the miss-closes-the-site
  rule are E235's, so the two methods are read alike.
  - E243's block-2 head ladder at two rungs. One is the rung nearest fine-tune size, where BatchTopK read 0.0030. The other
    is the top rung, where BatchTopK read at most 0.010 with the base-coefficient exclusion.
  - E235's block-3 head at block 3's output. BatchTopK read 0.42-0.44 after P2.70.
  - E212's real-text planted direction. Its BatchTopK reading is 0.77 or 0, pending the P2.70 refit.
  - Expected for Delta-Crosscoder, filed as a lean before the run: the top rung and the block-3 head pass at medium
    confidence. Eq. 9 makes the non-shared latents rebuild Δ. The fine-tune-size rung is a low-confidence pass. For
    E212's direction, the top latent's decoder difference should have a cosine of at least 0.9 with the planted
    direction; BatchTopK's had a cosine of 1.000. Steer F on a plant should remove at least half of the plant's effect
    at its read positions.
- *C2, matched random update, one per site.* This is E235's Gaussian, matched on activation change. It must stay under
  0.5. Its value sets the floor that the trained sides are read against, as E235's G1 and G2 do.

**Run 2, gated: the scored sides, only if C0 and C1 pass at block 2.**
- E62: copy and lookup, the saved seeds of each arm, with one matched random update per seed.
- The nine built pairs (strengthening against new): 18 sides at E203's sizes, one crosscoder each, plus 18 matched
  random updates.
- About 44 crosscoders in all. Every side is read on the four readings above. The pair call uses E235's every-seed rule
  on the after-Kassem share, with no reuse label until the definition (cell P1.1) is adopted.
- The steering reads are reported per side.

**Run 3** is held for one amendment.

**Time and cost.**
- *Measured* (E235 `run1/timing.json`, H100): our BatchTopK at 24,576 latents, batch 2048 and k 100 took 11.65 ms a
  step. 97,656 steps came to 18.96 min of training. Streaming 200M tokens through both models took 8.3 min at block 2 and
  13.8 min at block 3, or 2.6 s per 1.05M tokens. The whole run was 27.3 min per crosscoder alone on the card. Pods: E212's
  two H100 pods cost $8.03 for about 50 min each, so about $4.8 per pod-hour with set-up.
- *Assumed:* Delta-Crosscoder's step has 3,840 × 4096 latent-tokens against 24,576 × 2048, which is 0.31 times the work.
  The masked delta decode adds about 30%. That gives about 5-8 ms a step and 4-7 min of training for 50k steps. Streaming
  205M tokens takes about 8.5-14 min, at the measured rate. Generating 200,000 continuations of about 70 tokens costs
  about 1 min per model with a key-value cache. Without one it is about 20 min, which is about 1B token-forwards at the
  measured rate. Generation runs once per changed model, not once per crosscoder seed. The steering grid is about 1-2
  min.
- *Per crosscoder:* about 20-25 min on an H100 with cached generation, or 40 min without. All of these need measuring
  in the rehearsal.
- *Run 1:* 7 crosscoders, about 3 H100-hours. This is $0 on debug L40 jobs, one crosscoder per job; an L40 is assumed
  about twice as slow, about 50-80 min per job, so it fits the 2 h cap. The pod fallback costs about $15.
- *Run 2:* 44 crosscoders, about 15-30 H100-hours. That is about $70-145 on pods; on 8 pods (2 runners of 4) it takes
  about 2-4 h of wall time. Cluster debug brings it down to $0 if the queue takes 44 jobs.
- *Gated?* Run 1 is not gated. Run 2 is likely gated on cost, over $60 on pods, though not on time if it is spread over
  pods. It needs the gates in `.claude/CLAUDE.md`, with gate 6 from whip-lead. One crosscoder per side, as E203 had,
  keeps it near the low end.

**Early checks in code.** These stop the job. At step 0, ℒ_Δ = 0 exactly on C0. Every loss must be finite. FVE at 1k
steps should be at least 0.5, against E235's training-curve values. The first rehearsal measures the ms per step and
generation tokens per second, and they go into the plan before run 2.

**What each reading is blind to.** The after-Kassem share is blind to a change in how the model uses a direction it
already has. It is the same blind spot as E235's specific part. The steering reads are blind to a change spread over
many latents, which the paper reports for larger dictionaries: F.2 says the larger dictionary "identifies a small set of
related latents (approximately six)". Steer M cannot tell a used existing direction from an added vector that carries
the content, as the digest notes.

## Not run

Nothing has run. No code exists. No author contact was made. The paper's LLM-grader comparison is not proposed, because
there is no chat text on the 67M.

## Appendix: open readings for the reviewers

1. Is α = 0.5? We read Table 1's "Base Sparsity 200" and "Shared k Multiplier 2.0" as K_Δ = 200 and K_shared = 400.
2. Are shared decoders tied? We tie them, after DSF; untied shared latents could carry Δ through ℒ_recon.
3. Does "right tail" mean 1 − R^base? Eq. 4 is written for the base side.
4. Does "unpaired activations" mean general text run through both models on the same tokens, the only way a Δ exists per
   token? Eq. 5 says Δ "does not require a and b to arise from matched inputs".
5. Is Δ taken in raw or normalised units? We use Minder's per-model normalised units, as ℒ_recon is.
6. The null band: the paper's 0.492-0.506 on identical models implies independent decoder init or numerical noise. Our
   paired init gives exactly 0.5. If the reviewers prefer, C0 is run once with independent init to set a band like the
   paper's.
7. Paper inconsistencies, noted and not resolved: BatchTopK-200 is 4 of 10 in §7.1 and 5 organisms in §6.1. "Llama 3.2 8B"
   has 10,240 latents, which is width 2048. F.1 gives "52.5" for a relative norm.
