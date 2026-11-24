# Plan: Delta-Crosscoder as published, and how it would run on the 67M (cell P2.72, step 1: the spec)

Cell P2.72 (due 2026-11-25, 3 runs). User, 2026-11-23: "open delta-crosscoder and weight-diff svd cells". The aim is the
strongest crosscoder for narrow fine-tunes, so that "existing tools fail" survives a cherry-picking charge. This is the
spec, revised once after two reviews (redteam-lead with a fresh Opus reviewer, and the GPT plan-reviewer; responses at
the end). whip-lead's rules override the first draft. Report the method as published, as a yes or no per fine-tune. Give
no reuse score and no share column. Run 1 goes on an H100 pod. Do not contact the authors; whip-lead is asking the user.

**Sources.** Kassem, Jiralerspong, Rostamzadeh and Farnadi, "Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow
Fine-Tuning Regimes", arXiv 2603.04426. The arXiv listing shows one version only, v1 of 16 Feb 2026 (checked
2026-11-23). Every quote is from v1's HTML, saved with a text dump at
`~/commute-data/agents/compare-lead/dxc_paper/` (`p.html`, `p2.txt`, `p.pdf`). The lit digest entry is
`~/commute-agents/litreview-lead/lit_review/digest/papers/kassem2026-delta-crosscoder.md`.

**Code: not released.** The paper has no code link and no footnotes. Its only GitHub links are the arXiv
HTML's own feedback links. A GitHub search on 2026-11-23 found nothing for "delta-crosscoder", "delta crosscoder" or
"kassem crosscoder". The first author's account (`Alymostafa`) has no crosscoder or diffing repository. Nothing was
cloned, and `~/commute-data/agents/compare-lead/dxc/` was not created. A run is therefore our own implementation of the
paper's text, specified in §3 so that it does not inherit unstated behaviour from Minder et al.'s pinned trainer
(`dictionary_learning`@d1fd6f3). Code locations cited below are in that checkout.

**Symbols, each defined once.**
- *a*, *b*: the base model's and the fine-tuned model's residual activations at one site, on the same input token.
- *Δ = b − a*: their difference.
- *z*: the crosscoder's sparse code. It splits into *z_shared* (20% of latents) and *z_Δ*, the 80% that are non-shared.
- *K_shared*, *K_Δ*: the two top-K budgets.
- *d_i^base*, *d_i^ft*: latent *i*'s two decoder vectors.
- *R_i*: the paper's relative decoder norm, eq. 4, with the paper's sum denominator. We read the fine-tune side as 1 − R_i.
- *M*: the original 67M model. *F*: a changed model (a fine-tune, a planted change or a random update).
- *Behaviour g*: the side's own behaviour at its read positions, in nats (§3.7).
  - For a trained side, its task's filed answer readout: the mean answer log-odds on score-half windows.
  - For a planted change, the mean per-token KL from F to M at the plant's read positions.
  - For a random update, its matched side's g.
- *KL gap*: the mean per-token KL divergence between F's and M's next-token distributions on a side's read windows, in
  nats. A diagnostic only.

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

**The most faithful baseline column.** One yes or no per side, as the paper reports per organism. The answer is yes when
one of the top three non-shared latents passes both the steering test and the top-activation test (§3.5), each judged
against latents of the same crosscoder drawn from the shared bulk. Beside it, as description only, we report:
- the tail value of the top three latents, against the null runs' tail;
- whether adding the same latent to the base model moves it toward F, which is the paper's own "already present in the
  base model" check. Wherever it is shown, it carries the caveat that it is weak evidence of reuse, because the added
  vector can carry the learned content itself.

There is no reuse label and no share. A random update matched to each fine-tune gets the same yes or no. How often
random updates read yes is the method's false-recovery rate here.

## 2. The published method, step by step

Code locations: none (not released).

**2.1 The base crosscoder.** §3.1, eqs. 1-3: "u^base = W_e^base x^base, u^ft = W_e^ft x^ft", "u = ½(u^base + u^ft),
z = BatchTopK(u)", "x̂^base = W_d^base z, x̂^ft = W_d^ft z". "Throughout this work, we use BatchTopK as the sparsity
mechanism rather than an ℓ1 penalty". Eq. 1 has no encoder bias. Minder's encoder sums the two encoders and adds a bias before a
ReLU (`dictionary.py` 892-897). Its top-K ranks the rectified activations times the summed decoder norms, and at
inference it uses a learned threshold instead (`dictionary.py` 1554-1578, `trainers/crosscoder.py` 592-601).

**2.2 Shared and non-shared latents, two budgets.** §3.3: "A fixed fraction of the dictionary (20%) is designated as
shared latents, with the remaining 80% reserved for non-shared latents." "Sparsity is enforced using BatchTopK with a
Dual-K allocation: shared latents are assigned a larger activation budget K_shared, while non-shared latents are assigned
a smaller budget K_Δ = α·K_shared, with α<1." Table 1: "Base Sparsity (k_base) | 200", "Shared k Multiplier | 2.0",
"Shared Features Fraction | 20%". The text never gives α. The most literal reading of Table 1 is K_Δ = 200 and
K_shared = 400, so α = 0.5; this is an inference. The same k_base is listed for all widths from 2048 to 4096, so the paper
did not keep the active fraction fixed. The text does not say whether the shared latents' two decoders are tied. It
says "we adopt partitioning stragety from (Mishra-Sharma et al., 2024; Jiralerspong and Bricken, 2025)" [sic].

**2.3 The delta loss through non-shared latents only.** Eq. 5: "Δ = b − a". Eq. 8, unmasked: "ℒ_Δ = ‖Δ − (W_ft − W_base)
z‖²₂". Eq. 9, as used: "During difference modeling, shared latents are explicitly masked, restricting the delta prediction
to depend only on non-shared features: ℒ_Δ = ‖Δ − (W_ft − W_base)[0; z_Δ]‖²₂." "This ensures that shared features
contribute to reconstruction but cannot absorb fine-tuning–specific differences." Masking matters only if the shared
decoders are untied: tied decoders give W_ft − W_base = 0 on them anyway. Eq. 10: "ℒ = ℒ_recon + λ_s sparsity(z)
+ λ_Δ ℒ_Δ". Appendix A: "We set the delta loss weight Δλ=0.005". The text does not give:
- the form or reduction of ℒ_recon, against which λ_Δ was tuned;
- the meaning of λ_s under BatchTopK (Table 1 lists "AuxK Coefficient (α_auxk) | 1/32");
- whether Δ is taken before or after normalisation;
- whether eq. 7's prediction includes decoder biases. As written, it has none.

**2.4 Data.** §3.3: "We sample prompts x from a general corpus and generate responses from both the base and finetuned
models, producing y_base and y_ft." "Each concatenated input is then independently passed through both the base and
finetuned models, and activations are extracted from the same layer." "Delta-Crosscoder is trained on a mixture of such
contrastive pairs and unpaired activations". §4.2 lists four sources: "pretraining-style text 𝒟pre from FineWeb",
"instruction-tuned data from LMSYS 𝒟Inst", "when available, we include fine-tuning data 𝒟F", and "contrastive data 𝒟C
by sampling 200,000 prompts uniformly at random. For 100,000 prompts, we generate responses using the base model, and for
the remaining 100,000 prompts, we generate responses using the finetuned model." "In total, training uses approximately
200 million tokens, of which roughly 20 million corresponds to contrastive prompt–response data". "All sequences are
truncated or padded to a maximum length of 1024 tokens." The two sections disagree. §3.3 has both models answer each
prompt, and §4.2 has each prompt answered by one model. The text does not give:
- the corpus the prompts come from;
- response lengths;
- sampling settings for 𝒟C;
- the mix of 𝒟pre, 𝒟Inst and 𝒟F.

20M tokens over 200,000 prompts is about 100 tokens per prompt with its response. Appendix F.1: "removing finetuning
data does not degrade Delta-Crosscoder's ability to recover fine-tuning–induced latents" (two organisms).

**2.5 Site and size.** §4.2: "We train Delta-Crosscoder on activations extracted from a single intermediate transformer
layer." "we use a middle layer of each model". "We use an expansion factor of 5". The dictionary sizes in Table 2 are
20480, 17920 and 10240, which are five times widths of 4096, 3584 and 2048. The 10240 rows are the SDF organisms, which
§4.1 names "Llama 3.2 8B Instruct". That label and that width disagree; we do not resolve it.

**2.6 Training settings.** Table 1: "Optimizer | Adam", "Learning Rate | 1×10−4", "Total Training Steps | 50,000",
"Warmup Steps | 1,000", "Batch Size | 4096", "Initial Decoder Vector Norm Scale | 0.4", "Mixed Precision | bfloat16",
"Gradient Checkpointing | Enabled". 50,000 steps times 4096 is 204.8M token presentations, consistent with
"approximately 200 million tokens". No k annealing, learning-rate decay or seed count is stated. Normalisation is implied
only for steering, in Appendix C: "Decoder vectors are normalized using model-specific normalization factors".

**2.7 Latent selection.** Eq. 4: "R^base_i = ‖d^base_i‖₂ / (‖d^base_i‖₂ + ‖d^ft_i‖₂)". "Values R^base_i ≈ 1 indicate
base-specific features, while R^base_i ≈ 0.5 corresponds to shared structure." §4.3: "we rank non-shared latents by their
relative decoder norm (see subsection 3.1) and select the top-3 latents from the right tail of this distribution." Taken
literally, the right tail of R^base is base-specific. Every result places the fine-tune latents in "the right tail"
(§5.2, §5.4, F.1), so the text contradicts itself.

**2.8 Steering and top activations.** §4.3: "We perform positive and negative steering of the finetuned model by adding or
subtracting the latent's decoder vector during inference". The vector is not named as d^ft, d^base or their difference.
"Steering is evaluated on task-agnostic prompts (e.g., open-ended questions such as "What is on your mind?")". "When an
explicit evaluation dataset is available for a given organism, we additionally apply steering on that dataset". Base
model: "To test whether the recovered direction corresponds to a latent capability already present in the base model but
not naturally expressed, we apply the same positive and negative steering interventions to p_base." Appendix C gives:
- nine open-ended prompts;
- "11 discrete steering strengths s∈{−200,−100,−80,−50,−20,0,20,50,80,100,200}";
- "steering is applied by adding s⋅d to the model's residual stream at the selected layer during generation";
- "Maximum tokens: 200", "Temperature: 0.7", "Top-p: 0.9".

Top activations: "we inspect the inputs that maximally activate the feature. We examine whether these high-activation
examples are semantically consistent with the intended finetuning behavior". Success per organism, §6.1: "count an
organism as successfully identified if the recovered latent supports causal validation via steering and max-activation
analysis." That judgement is qualitative. §6.2 adds an LLM grader, GPT-5.2, which scores against ADL's 1-5 rubric.

**2.9 The paper's controls.** §7.1, null test: "applying Delta-Crosscoder to two identical versions of LLaMA 3.1 8B
Instruct". "All non-shared latents concentrate near 0.5, with the right tail reaching only 0.506 and the left tail 0.492."
"none support causal validation under steering or max-activation analysis." False positives: "In most settings, only
the highest-ranked latent is causally responsible for the observed behavior, while the remaining candidates produce no
systematic effects under intervention." "Latents outside the right tail consistently fail to induce behavioral changes".
The paper has no random-update control and no planted control. A spread of 0.492 to 0.506 on identical models implies
the two decoders were not initialised equal, or that training was nondeterministic. The text says neither.

**2.10 The paper's positive results.** §7.1: "Delta-Crosscoder successfully identifies causally relevant latents for all
10/10 organisms (0% method-level false positives). In contrast, DSF succeeds on 6/10 organisms (40% false positives),
BatchTopK-200 on 4/10 organisms (60% false positives), and BatchTopK-400 on 4/10 organisms". §6.1 instead lists five
organisms for BatchTopK-200: "the three EM organisms on Qwen, Taboo, and Subliminal Learning". We do not resolve this.
Table 2 gives explained variance of 72-80% and dead latents of 7.6-64.2% for Delta-Crosscoder. §6.2 calls it comparable to
ADL "On Emergent Misalignment and Synthetic Document Finetuning". The two tie "On Taboo Word Guessing", and "For
Subliminal Learning, ADL attains higher scores". F.1 says "The most extreme latent attains a value of 52.5". That cannot
be a relative norm as eq. 4 defines it, and we do not interpret it.

## 3. What we would run: the objective and the reading, fixed

All of §3 is ours unless it quotes §2. Each choice is a row in §4.

**3.1 Inputs and units.** Activations at block 2's input (site A of E203/E235) are the paper's middle layer. Block 3's
output (site B) is used only for the block-3 control. Both models are scaled by one scalar, fit on M so that M's mean
squared norm at the site is the target RMS, as Minder does. There is no mean subtraction, which `dictionary.py` 108 would
apply. Δ_n = b_n − a_n is then exactly Δ divided by that scalar, in raw-unit proportions, and a tie between the decoders
is a tie in raw units.

**3.2 Forward.** u = ½(W_e^base a_n + W_e^ft b_n) + c_e, then r = ReLU(u). This is eq. 2 plus Minder's encoder bias c_e.
Two separate batch top-Ks follow, ranked on r itself and not on r times the decoder norms. The shared block keeps its
K_shared·B largest entries and the non-shared block keeps its K_Δ·B largest, where B is the batch size. z holds the kept
values. At reading time the same two batch top-Ks run on shuffled batches of 4,096 read tokens, with no learned
threshold. Decoders: x̂^m = W_d^m z + c_d^m for m ∈ {base, ft}. Every latent has two decoder vectors, shared latents
included, and the shared ones are untied as eq. 9 implies. Budgets: K_Δ = 38 and K_shared = 76. That keeps α = 0.5 and
keeps the active fraction the paper's Llama widths give each block (400 of 4,096 and 200 of 16,384), on 768 shared and
3,072 non-shared latents. This is a choice, not forced: Table 1's 400 would make 52% of our shared latents active.

**3.3 Loss, every reduction stated.** Every term is a mean over the batch's tokens, in the normalised units of §3.1.
- ℒ_recon = mean_t Σ_m ‖x_m − x̂_m‖²₂: squared and summed over both models, so that it is in the same units as eq. 9 and
  λ_Δ weighs like with like. Minder's trainer uses unsquared L2 (`trainers/crosscoder.py` 616-619).
- ℒ_Δ = mean_t ‖Δ_n − (W_d^ft − W_d^base)_{non-shared} z_Δ‖²₂, as in eq. 9, with no bias term, as in eq. 7. It runs on
  every training row, because every row is passed through both models on the same tokens (§3.4).
- ℒ_aux is Minder's AuxK. Its target is the detached residual of both models stacked. It reconstructs from the
  d/2 = 384 dead latents with the largest rectified pre-activations, from both blocks, and is divided by the residual's
  variance (`trainers/crosscoder.py` 421-480). A latent is dead after 10M tokens without firing, the paper's Appendix B
  definition.
- ℒ = ℒ_recon + (1/32)·ℒ_aux + 0.005·ℒ_Δ.

**3.4 Optimiser and initialisation.** The optimiser follows Table 1: Adam at 1e-4 (default betas), a linear warm-up over
1,000 steps and then a constant rate. There are 50,000 steps at batch 4,096, with no k annealing, under bf16 autocast with
fp32 master weights. Gradient checkpointing is dropped: the activations are cached, so there is no model graph to
checkpoint, and the objective is unchanged. Decoder vectors start at norm 0.4. Every scored fit draws the two models'
decoders independently. That is the only init under which identical models show a spread, as the paper's 0.492-0.506
does, so the null band comes from the same recipe as the scored runs. The encoder starts as the transpose of its model's
decoder. C0a alone uses paired init (both decoders equal), as a wiring check.

**3.5 Data, counted per source, 204.8M tokens per fit.** All rows of every source pass through both M and F on the same
tokens, so Δ exists per token. This is our resolution of "unpaired activations": the encoder needs a matched pair (eq. 1).
Before extracting anything, the code asserts a ledger:
- 𝒟C + 𝒟F realised + 𝒟pre = 204,800,000 tokens exactly, which is 50,000 × 4,096;
- every row range below is disjoint from every other, checked by row id.

The ranges, in rows of `danbraunai/pile-uncopyrighted-tok-shuffled` at 512 tokens a row:

| source | rows | tokens available | used per fit |
|---|---|---|---|
| task windows (all tasks, both halves) | train, below 32,768 (where every task window was cut) | per task | 𝒟F only |
| E235's Pile fill and key-value pool | train 32,768-196,607 | not used here | 0 |
| 𝒟C prompts | train 196,608-396,607, the first 32 tokens of each | 200,000 prompts | 6.4M, inside 𝒟C's 20.0M |
| 𝒟pre | train 400,000-761,999, one pass, no repeats | 185.34M | 143.84M to 184.8M |
| Pile reads for scoring | validation 0-4,351 | 2.23M | 0 (held out) |

- **𝒟C, 20.0M tokens.** Following §4.2's counts, M continues 100,000 prompts and F continues the other 100,000. Each
  continuation is 68 tokens sampled at temperature 0.7 and top-p 0.9, Appendix C's settings. Each sequence of 100 tokens
  is a training row. M's 100,000 continuations are made once and reused by every fit. On C0, M makes F's half too.
- **𝒟F, quota 40.96M (20%).** This is the side's fit-half read windows, each drawn at most 100 times (E235's repeat cap).
  Where the cap cannot fill the quota, 𝒟pre fills the shortfall, and the realised count is written to the fit's record.
  - C0 has no 𝒟F, which Appendix F.1 allows; 𝒟pre then supplies 184.8M of its 185.34M.
  - A planted change uses its plant's read windows, split into fit and score halves by the row ids saved in E243/E212.
  - A random update uses its matched side's fit half.
- **Scoring sources, never trained on.** Each side's score-half read windows, plus Pile validation rows 0-255 for the
  Pile loss and rows 256-4,351 for the top-activation set.
- **Moving the data.** 𝒟pre is built once on the cluster as uint16, about 370 MB, and synced to each pod, about 37 s.

**3.6 Generation.** `LlamaSimpleMLP.generate` (`~/param-decomp/param_decomp/pretrain/models/llama_simple_mlp.py` 442-487)
has no key-value cache and no top-p, and recomputes the whole prefix each step. We would write a sampler that calls the
model's forward on the growing prefix, the same work as theirs, and adds nucleus sampling at top-p 0.9. That costs about
4,500 token-forwards per sequence, including the LM head on every position. A KV cache is not added: it would be new
model code. The rehearsal benchmarks the sampler at full batch on the card. Until then it is priced in §5 at the measured
streaming rate per token-forward of one model, without halving that rate.

**3.7 The recovery rule, per side: yes or no.** It is our written form of §6.1's "supports causal validation via
steering and max-activation analysis".
- **Candidates.** The top three alive non-shared latents by 1 − R_i, with R_i computed as in eq. 4 with its sum
  denominator. This is not the repo's `relative_norm()`, which uses a max denominator. On C1 the top three of the other
  tail, by R_i, are read too, as description.
- **Bulk pool, one per candidate.** Each candidate gets its own pool of 60 alive non-shared latents with |R_i − 0.5|
  under the median deviation of all non-shared latents. They are drawn with a fixed seed and matched to that candidate on
  ‖d^ft‖ within a factor of 1.5 and on firing rate within a factor of 2.
  - Pools may overlap, so a fit scores up to 183 distinct latents (3 candidates and 180 bulk).
  - If fewer than 60 latents qualify, the factors widen to 2 and 4, then to 3 and 8.
  - If fewer than 20 qualify even then, the candidate has no floor. It is recorded as "no floor" and cannot pass.
- **Matched search.** Every test runs the same search on each bulk latent as on its candidate. A candidate passes a test
  only if it beats its pool's maximum. If latents were exchangeable, that is a 1/61 chance per candidate per test.
- **Steering vector and strengths.** The vector is v_i = d_i^ft, taken to raw units and scaled to unit length. It is added
  at the site at every position, at strength s·ρ/100, for each of Appendix C's ten nonzero values of s.
  - ρ is the median per-token norm of M's activation at the site on Pile validation rows. That scaling is ours, standing
    for "model-specific normalization factors".
  - The grid is read uncapped, as Appendix C gives it; the paper states no cap on the strength. The first draft's cap at
    F's own Pile-loss rise was ours. It went to about 0 for updates of fine-tune size and would have failed them by
    construction, so it is dropped.
  - F's Pile-loss rise at each s is recorded beside every read. Because every strength is read, no case can be left
    without one.
- **Steering test (causal, behaviour-specific).** The read is the side's own behaviour g on the score half.
  - The pass statistic is the largest suppression, over negative s, of F's behaviour: g at s = 0 minus g at that s.
    This is in absolute nats, not a fraction.
  - The induce direction is reported beside it: M's g at positive s, minus M's g unsteered. This is the paper's
    base-model check, and it carries the caveat in §1.
  - The generic KL gap at each s is a diagnostic only.
- **Top-activation test (consistent with the behaviour).** The held-out set has equal tokens of Pile validation rows and
  score-half read windows. Find the candidate's 50 most activating tokens there.
  - The statistic is the share of those 50 that fall on the side's behaviour positions. For a task these are the answer
    positions and the tokens just before them. For a plant they are its read positions.
  - Plain enrichment on read windows is a diagnostic only.
  - The 20 top contexts per candidate (token ids and decoded text) are saved for a reader.
- **Yes** when one candidate passes both tests; otherwise **no**.
- **The differential null on C0b.** Here F = M and the readout is E62's copy task. The steering test asks whether a
  top-tail latent moves M's own copy log-odds more than its matched bulk, in absolute nats. Its result is not fixed in
  advance, as the old KL-gap null's was.

## 4. Departures

"Ours" means our choice where the paper is silent or cannot apply. "Theirs" means taken from the paper's text. "Forced"
means the 67M setting leaves no faithful option.

| step | paper | on the 67M | departs | ours/theirs | forced |
|---|---|---|---|---|---|
| code | not released | own port, §3 | y | ours | n |
| encoder | eq. 2, average, no bias | average plus bias, ReLU | y, bias | ours | n |
| top-K rank | BatchTopK(u) | on r = ReLU(u), not times decoder norms (Minder) | n | theirs | n |
| Dual-K | two budgets | two separate batch top-Ks | n (form ours) | theirs/ours | n |
| inference | not stated | same batch top-K, B 4,096, no threshold | ? | ours | n |
| budgets | Table 1: 400 / 200 | 76 / 38, the same active fraction per block | y | ours, a choice | n |
| split | 20% shared, 80% non-shared | 768 / 3,072 of 3,840 | n | theirs | n |
| shared decoders | masked in eq. 9 | untied, the literal reading | n | theirs (reading) | n |
| ℒ_recon | not stated | squared L2, summed over models, token mean | ? | ours | n |
| ℒ_Δ | eq. 9, λ 0.005 | same, token mean, no bias, every row | n | theirs | n |
| sparsity term | λ_s | AuxK at 1/32, Minder's form | ? | theirs (1/32), ours (reading) | n |
| normalisation | per-model factors (App. C) | one scalar fit on M, no mean subtraction | y | ours | n |
| init | decoder norm 0.4 | 0.4, the two decoders independent; paired only on C0a | ? | theirs / ours | n |
| precision | bf16 | bf16 autocast, fp32 weights | n | theirs | n |
| checkpointing | enabled | dropped (no graph) | y, no effect | ours | n |
| schedule | Adam 1e-4, 50k steps, warm-up 1k, batch 4,096 | same, constant after warm-up, no k annealing | n | theirs | n |
| prompts | "a general corpus" | Pile train rows past E235's pool, 32 tokens | y | ours | y (Pile model) |
| responses | §4.2: one model per prompt | 100k by M and 100k by F, 68 tokens, T 0.7, top-p 0.9 | y | theirs (§4.2 counts) / ours | y (no chat model) |
| LMSYS | instruction data | dropped | y | ours | y |
| FineWeb | pretraining text | Pile train rows 400,000-761,999, one pass | y | ours | y |
| 𝒟F | when available | 20% quota, repeat cap 100, 𝒟pre fill, fit half only | n / quota ours | ours | n |
| token ledger | about 200M | 204.8M exactly, asserted, disjoint row ranges | n | theirs (size) / ours | n |
| length | 1024 | 512 | y | ours | y |
| site | a middle layer | block 2's input; block 3's output only for the block-3 control | n | theirs | n |
| size | expansion 5 | 3,840 latents | n | theirs | n |
| selection | top-3, "right tail" | top-3 by 1 − R (sum denominator); other tail read on C1 | ? | ours (reading) | n |
| steering vector | "the latent's decoder vector" | d^ft, unit length, s·ρ/100 | ? | ours | n |
| steering prompts | nine chat prompts | the side's score-half read windows | y | ours | y |
| steering strengths | 11 values, no cap stated | the same grid, uncapped, Pile-loss rise recorded | n | theirs | n |
| steering read | judged text, on the behaviour | the side's behaviour g in nats, both signs, both models; KL gap as a diagnostic | y | ours | y (no chat text) |
| top activations | inspected for consistency with the behaviour | share of top-50 on behaviour positions, against the pool; 20 contexts saved | y | ours | y (no grader) |
| success | qualitative, per organism | §3.7 rule, one 60-latent pool per candidate, widened or "no floor" | y | ours | n |
| controls | identical models | C0a/C0b, plants, random updates | y (added) | ours | n |

## 5. Proposed run

**Run 1, controls, two H100 pods, not gated.** M's 200,000 continuations are generated first. Then nine fits, in this order:
- **C0a**, alone and first: identical models, paired init, at site A. A wiring check only.
  - Expected: ℒ_Δ = 0 at every step, because Δ = 0 and the two decoders stay equal. Every R_i equals 0.5 exactly by symmetry.
  - The bar is one number, filed after the rehearsal measures the deviation under bf16 on the same kernels. Any deviation
    above it is a code fault and stops the run.
  - The recovery rule is not read: every latent ties.
- **C0b**: identical models with independent init, at site A. This is the null band under the scored recipe.
  - Expected: a tail near the paper's 0.492-0.506, reported beside, unscored.
  - Recovery is read with the differential null of §3.7, on E62's copy behaviour in absolute nats. The paper's §7.1
    gives the expected result: "none support causal validation". Lean: "no", medium.
- **C2**: one random update at each of sites A and B (two fits). Each is E235's Gaussian, matched to elicit's activation change.
  - This is the null that the steering and top-activation tests can actually fail. The paper never ran one, so there is
    no prior; the lean is "no" at low confidence. It plays the role of E235's G1.
  - A "yes" is reported to whip-lead before C1 continues. It does not cancel C1, but at site A it closes run 2 (below).
- **C1**: planted changes, all at site A except the block-3 head (five fits). Lean, filed before the run: recovery "yes"
  at medium confidence on rung 5 and on E212's plant. The pass is a §3.7 yes, not a share. The planted direction's cosine
  with the top latent's d^ft − d^base is reported beside the bulk maximum, with no bar.
  - E243 rung 1, at about a fine-tune's size (BatchTopK specific part 0.0030). Lean: "no", low confidence.
  - E243 rung 5, the top rung. BatchTopK read 0.46, and about 3e-5 under the literal base-coefficient exclusion of P2.70.
  - E212's real-text planted direction. BatchTopK read 0.77, or 0 under the exclusion, pending the refit.
  - E235's block-3 head at site B. BatchTopK read 0.42-0.44 after P2.70. Lean: "yes", medium.
  - Rung 5 again with tied shared decoders, the other reading of the shared-latent ambiguity. Reported only.

**What gates what.**
- Run 2 runs at site A only if three things hold: C0a passes, rung 5 or E212's plant reads yes, and C2 at site A reads no.
  - If no plant reads yes, site A reads "not applicable at this update size", as E235 did, and run 2 does not run.
  - If a plant reads yes and C2 at site A reads yes as well, the yes does not tell a planted change from any change.
    Site A then reads "does not separate a plant from a random update", and run 2 does not run.
  - C2 is one random update, so it is a weak null. Run 2's matched random updates give the false-recovery rate afterwards.
- Rung 1 decides how a "no" on a trained side reads. If rung 1 reads no, a trained side's "no" means "below the method's
  reach".
- The block-3 head and C2 at site B gate nothing; run 2 does not read site B.

**Stops in code.**
- A non-finite loss stops the fit.
- On C0a, any nonzero ℒ_Δ, or an R deviation over the filed bar, stops the run.
- FVE under 0.5 at the end of a fit marks it a code fault, which stops that fit and is reported. E235's crosscoders read
  at least 0.89 and the paper's at least 0.72, so FVE is not otherwise a gate.
- Expected at 1k steps: FVE at least 0.5, and ℒ_Δ falling. The rehearsal fills in the step-1 values first.

**Run 2, gated, only if run 1 opens site A.** The default is 39 fits, one per side, read by §3.7 only:
- E62 copy and lookup, three saved seeds each (6), each with one matched random update (6);
- the nine built pairs, 18 sides, with one matched random update per pair (9).

The fuller form gives each built-pair side its own random update (18), for 48 fits. It costs more than the session's
headroom allows (below) and needs the user's confirmation. The output is a yes or no per side, with the tail and the
steering table. There is no pair call and no reuse label. Each side is a single fit, so the comparison is descriptive:
E235's every-seed rule needs two or more values per side and is not used.

**Run 2, trimmed option (for the user, beside the 39- and 48-fit forms).** E62, plus one built side of each kind, and
every side gets a matched random update. That is 18 fits:
- E62 copy and lookup, three saved seeds each (6), each with one matched random update (6);
- one strengthening side, copy maps (E203's "Copy maps vs random maps"), with its matched random update (2);
- one new side, the cipher of "Copy vs cipher, step 10", with its matched random update (2);
- one random-kind side, the random maps of "Copy maps vs random maps", with its matched random update (2).

Whip-lead may swap in other sides of the same kinds; the count stays the same.

Cost, from the per-item rates above:
- Generation: 18 × 19 min = 5.7 h.
- Fits at site A: 18 × 39 min = 11.7 h.
- Set-up: 6 pods × 18 min = 1.8 h.
- Uploads: 9 trained sides × 270 MB plus 6 × 370 MB of Pile, about 4.7 GB, 8 min. Downloads: about 0.9 GB, 2 min.
- About 19.4 pod-hours, about $93, and about 3.2 h of wall time on 6 pods (two runners, 4 and 2). Run 1 plus this
  option is about $137.
- Gated on cost (over $60), not on time.

What it cannot say that the 39- and 48-fit runs can:
- There is one side per kind, so it cannot show whether a yes or a no follows the kind or the particular task.
- It cannot give a tally over the nine pairs to set beside BatchTopK's 4-6 of 9.
- Its false-recovery rate rests on 9 random updates (15 in the 39-fit run, 24 in the 48-fit run). The 48-fit run alone
  also matches a random update to every built side.

**Time and cost, per item times the count.**
- **Measured** (E235 `run1/timing.json`, H100 alone): streaming took 2.6 s per 1.05M tokens through both models at site A
  and 4.34 s per chunk at site B. That is 8.5 min and 14.1 min for 204.8M tokens. Training took 11.65 ms per step.
  E212's two H100 pods came to $8.03 for about 50 min each, about $4.8 per pod-hour with set-up. The sync speed of
  10 MB/s is from the root `CLAUDE.md`.
- **Derived from measured rates**, each to be replaced by the rehearsal's measurement. Every forward pass is priced at
  the measured pair rate, 2.6 s per 1.05M tokens, not halved. The stream may skip the LM head that generation and
  scoring run, so halving was not justified.
  - Training: 4-7 min per fit. This scales 11.65 ms per step by 0.31 of the work, plus 30% for the delta decode.
  - Generation: 100,000 sequences are about 0.45B token-forwards, so about 19 min, before the top-p overhead.
  - Scoring: up to 183 latents × 10 strengths, read on F and on M with the Pile-loss read beside each. At up to 0.1M
    score tokens per read that is about 549M token-forwards, or about 23 min. The top-activation encode adds about 1 min,
    so scoring is about 24 min per fit.
  - Per fit: about 39 min at site A and about 44 min at site B.
- **Rehearsal ($0, cluster debug, before run 1):** every stage at tiny size, with a resume and the downloads. It
  measures:
  - ms per step, and the sampler's tokens per second at full batch on the card;
  - one full-size steering and scoring slice, one candidate with its 60-latent pool, timed;
  - peak memory in every stage, the C0a bf16 deviation and the step-1 losses.

  Those numbers replace the derived ones in this plan before run 1, and a stage cap that no longer fits is reset then.

| item | count, run 1 | count, run 2 (48 fits) | count, run 2 (39 fits) | per item | source |
|---|---|---|---|---|---|
| M's continuations, 200k | 1 | 0 (reused) | 0 | 37 min | derived |
| changed model's continuations, 100k | 6 | 48 | 39 | 19 min | derived |
| fit at site A | 7 | 48 | 39 | 39 min | stream measured; rest derived |
| fit at site B | 2 | 0 | 0 | 44 min | stream measured; rest derived |
| pod set-up | 2 × 15 min | 8 × 18 min | 8 × 18 min | | E212, measured |
| uploads (weights 270 MB each, 𝒟pre 370 MB per pod) | 2.4 GB, 4 min | 9.5 GB, 16 min | 9.5 GB, 16 min | 10 MB/s | measured |
| downloads (about 50 MB per fit) | 0.45 GB, 1 min | 2.4 GB, 4 min | 2 GB, 3 min | 10 MB/s | measured |
| **pod-hours** | **about 9.1** | **about 49** | **about 40.5** | $4.8 per pod-hour | |
| **cost** | **about $44** | **about $236** | **about $194** | | |
| **wall time** | about 4.4 h on 2 pods | about 6.1 h on 8 pods, 4.1 h on 12 | about 5 h on 8 pods | | |

- **Run 1.** First, pod A generates M's continuations; meanwhile pod B generates the six changed models' continuations.
  C0a then runs alone on pod A. The other eight fits split over the two pods.
  - Stage caps: 30 min per 100k continuations, 60 min per fit, $55 for the run.
  - Not gated: under $60 and under 6 h. The $55 cap keeps it under the gate. If the rehearsal prices it over $60, it
    becomes gated and goes to whip-lead first.
- **Run 2.** This is gated on cost. It needs the gates in `.claude/CLAUDE.md`, with gate 6 from whip-lead.
  - With 48 fits it runs 6.1 h on 8 pods, so it is also gated on time unless it runs on 12 pods (three runners of 4).
  - Run 1 and the 48-fit run 2 together, about $280, pass the session budget's 15% headroom ($255 of $300). That needs
    the user's explicit confirmation.
  - The 39-fit form (one random update per pair) totals about $238, inside the headroom. It is the default.
  - Random updates are rebuilt on the pod from their seeds. The largest lever is generation, about 40% of run 2. A KV-cache
    sampler would cut it, but that is new model code, and the rehearsal decides whether it is worth writing.

## Not run

Nothing has run. No code exists. The paper's LLM-grader comparison is not proposed, because there is no chat text on the
67M. Site B is not read on the trained sides.

## Appendix: open readings

1. Is α = 0.5? The most literal reading of Table 1 is K_Δ = 200 and K_shared = 400; the scaling to 38 and 76 is ours.
2. Shared decoders: we take them untied, the literal reading of eq. 9. Tied is run once on rung 5.
3. "Right tail": we read the fine-tune side, and C1 reads both tails.
4. "Unpaired activations": we pass every row through both models; it is our resolution.
5. Δ is taken in one shared scalar unit with no mean subtraction (§3.1), so a constant planted offset stays in Δ.
6. The null band comes from independent init (C0b). Paired init (C0a) only checks wiring.
7. The paper's inconsistencies, unresolved:
   - BatchTopK-200 is 4 of 10 in §7.1 but five organisms in §6.1.
   - "Llama 3.2 8B" goes with 10,240 latents.
   - F.1 gives "52.5" as a relative norm.
   - §3.3 has both models answer each prompt; §4.2 has one model per prompt. We follow §4.2's counts.

## Review response (2026-11-23)

**whip-lead.**
- Do not contact the authors. Fixed: the suggestion is removed.
- Report as published. Fixed: one yes or no per side, by the rule in §3.7.
- Drop the after-Kassem share. Fixed: removed everywhere, with no share column and no tail threshold on a share.
- Run 1 on an H100 pod. Fixed: §5.

**redteam-lead.**
1. Degenerate tail threshold. Fixed: no threshold filters anything now. The null band comes from C0b (independent init,
   the scored recipe) and C2, which run before C1.
2. Our share in the method's column. Fixed: C1 and run 2 are scored on §3.7's yes or no, against the bulk floor.
3. Top-activation test missing. Fixed: a test and a row, and 20 contexts per candidate are saved.
4. Steering under-specified. Fixed: both signs on both models; the vector is d^ft (ours, a row); the bulk latents get
   the same search.
5. Shared decoders. Fixed: untied as the literal reading, tied run once on rung 5.
6. Trainer behaviours.
   - (a) Fixed: squared ℒ_recon, with its reason.
   - (b) Fixed: no mean subtraction.
   - (c) Fixed: ranked on r, not on r times decoder norms.
   - (d) Fixed: two separate batch top-Ks.
   - (e) Fixed: no k annealing.
   - (f) Fixed: the encoder bias is kept (a row), and inference uses the batch top-K with no threshold.
7. C1 bars unsourced. Fixed: the 0.5, cosine 0.9 and half-removed bars are gone. A pass is beating the bulk maximum of the
   same fit, and the cosine is reported only.
8. The C0 tolerance. Fixed: one bar, filed from the rehearsal's bf16 measurement.
9. Encoder wording. Fixed: the factor of two no longer arises, since we average as eq. 2 does.
10. K scaling. Fixed: labelled ours, a choice.
11. λ_s as AuxK. Fixed: "theirs (1/32), ours (reading)".
12. Normalisation. Fixed: Appendix C quoted in §2.6.
13. §3.3 against §4.2. Fixed: added to the appendix; we follow §4.2.
14. Data rows. Fixed: counted quotas in §3.5.
15. Fit count. Fixed: run 1 has nine fits (C0a, C0b, two C2, five C1). C0's steering is called undefined, not zero.

**GPT plan-reviewer.**
- (a) Objective. Fixed in §3.2-3.4: forward, both top-K rules, inference, every reduction, AuxK target and mask, and the
  units of the decoders. The budgets are labelled a choice.
- (b) Data. Fixed in §3.5: prompt design, per-source quotas with the Pile fill, rows in eq. 9 (all), and disjoint ranges.
  Running everything through both models is called our resolution.
- (c) C0. Fixed: C0a is a wiring check. C0b and C2 are the null under the same recipe; site B's null is C2 at site B.
  Steering is read in absolute nats, the stops and the FVE role are stated, and R uses the sum denominator.
- (d) The recovery rule.
  - Fixed in §3.7: vectors, scales, positions, the fit/score split, the same search for the bulk latents, and saved
    contexts.
  - Fixed in §5: which plant gates which site. C2 is E235's G1.
  - Seeds: a single fit per side, read descriptively.
- (e) Generation. Fixed in §3.6: our sampler, with no KV cache. It is priced from measured rates and marked for
  replacement by the rehearsal benchmark. The fit count, both site streams, scoring, set-up and transfers are counted,
  and there are stage caps.
- Minors.
  - Fixed: the paper's inconsistencies are left unresolved.
  - Fixed: gradient checkpointing is a row.
  - Fixed: rung 5's literal reading is about 3e-5.
- Rejected: none.

## Review response, second round (2026-11-23)

**redteam-lead, re-check.**
1. C2 at site A. Fixed in §5: run 2 opens only if C0a passes, rung 5 or E212's plant reads yes, and C2 at site A reads no.
   Plant yes with C2 yes reads "does not separate a plant from a random update".
2. The steering cap. It was ours; Appendix C states no cap. Fixed: the cap is dropped and the paper's grid is read
   uncapped (row "steering strengths", theirs), with F's Pile-loss rise recorded beside each read. Filed before run 1.
- Minor: the weak-evidence caveat now goes with the base-model read wherever it is shown (§1, §3.7).

**GPT plan-reviewer, re-check.**
- (a) The token ledger.
  - Fixed in §3.5: 𝒟pre moves to Pile train rows 400,000-761,999, which hold 185.34M tokens, one pass. That covers C0's
    184.8M.
  - The code asserts the 204.8M ledger and range disjointness before extraction.
  - The first draft's range overlapped E235's key-value pool; it is now unused.
  - The plants use their saved fit and score halves.
- (b) Behavioural recovery. Fixed in §3.7.
  - The causal test reads the side's own behaviour g: a task's filed answer readout, or a plant's KL at its read
    positions.
  - Top activations must fall on behaviour positions. The KL gap and plain enrichment are diagnostics only.
  - Edge cases: the uncapped grid always gives a strength to read. Under 60 matches the pool widens, and under 20 the
    candidate is "no floor" and cannot pass.
  - C0b now reads a differential null on E62's copy behaviour in absolute nats, whose result is not fixed in advance.
- (c) Cost. Fixed in §5.
  - Every forward pass is priced at the pair rate, so generation is 19 min per 100k.
  - Scoring counts up to 183 distinct latents, about 24 min per fit.
  - The rehearsal times the sampler, a full-size scoring slice and peak memory.
  - Run 1 is repriced at about $44 with caps of $55 and 60 min per fit.
  - Run 2 is repriced at about $194 for 39 fits, the new default, or $236 for 48. The 48-fit form exceeds the session
    headroom and needs the user's confirmation.
- Rejected: none.
