# Context brief: our paper, for judging related work

Source: the draft at `~/commute-agents/writeup-lead/paper/paper/` (main.tex and its inputs), read 2026-11-23. The draft
is mid-rewrite; the abstract and conclusion are marked "to be rewritten last". Bib list: `_paper_refs.tsv` beside this file.

## 1. Question and claims

Title: "A Mechanistic Foundation for the Elicitation Hypothesis". The question: "Is post-training eliciting mechanisms
that pre-training built, or building new ones?" The hypothesis is stated as **Elicitation Hypothesis (EH):
"Pre-training creates mechanisms; post-training elicits them."** (the word follows Lambert 2025). The gap, in the
draft's words: "Evidence so far is behavioural or reads activations, and no test looks inside the model", and "We know
of no diffing method scored on fine-tunes whose new or reused status is known by construction."

The two claims, as the contributions list states them:
1. **A mechanistic definition of reuse.** It holds at any featurization fixed from the base model, and "it looks inside
   the model: two fine-tunes with one input-output behaviour can differ under it."
2. **Readings validated against known answers, and existing tools that fail them.** Cheap readings of an update follow
   the mechanism "where the base model's prior on the answer is pulled apart from the mechanism". Our readings of a
   crosscoder and of neurons "cannot tell reuse from new at fine-tune size". On realistic fine-tunes "the cheap
   readings are not validated".

## 2. The definition of reuse

Main idea, quoted: "A fine-tune F reuses the base model M, for a behavior, to the extent that the variables F needs for
that behavior still have M's mechanisms." The abstract adds "on the settings of their parents that occur."

It is built on causal abstraction (Geiger et al. 2025). One featurization τ is fixed from M and applied unchanged to F,
so the featurized models M_τ and F_τ have the same variables and differ only in their mechanisms.
- **Kept (Def. 1):** "F_τ's mechanism for u is the same as M_τ's, up to some positive scalar, on every setting", i.e.
  f_u^F(p) = c·f_u^M(p) for some c > 0 and every parent setting p that occurs on F's runs over the dataset D.
- **Needed (Def. 2):** u is needed if "some single-variable interchange intervention between inputs of D changes the
  behavior". K is the set of needed variables.
- **Reuse (Def. 3):** "The reused variables are the needed variables that are kept", R = {u ∈ K : u kept}. K∖R are new.
- **Approximate form (Def. 4):** the write share ρ = Σ_K e_u / Σ_K d_u. This is the share of F's change on the needed
  variables that is a rescaling of M's mechanism, with each variable weighted by its change. Every reported number is ρ,
  or a cheaper reading of it, over a needed set estimated from 256 input pairs.

Reuse is compared **by mechanism, not by value** (no cross-model patching). It is **relative to the featurization**: at
neurons almost everything reads as reuse (≥ 0.999).

## 3. Methods

- **Models:** Goodfire's 67M, four-layer Pile model, with its released VPD decomposition (Bushnaq et al. 2026; 38,912
  rank-one subcomponents). Qwen3-1.7B, with two VPD decompositions, is used for the scale check.
- **Featurizations:** VPD subcomponents (F's update is carried onto them by a least-squares fit), BatchTopK SAE
  latents, neurons and heads (a control), and top-activating neurons.
- **Cheap readings:** the **re-gating share φ_Γ**: how much of the gain the base subcomponents recover when VPD's
  causal-importance function Γ gates them on the fine-tune's activations. Also the **mean-difference steering share**,
  the update's **effective rank**, and gain-only (IA³-style) refits.
- **Baselines (our adaptations):** the BatchTopK crosscoder of Minder et al. 2025, read by our own share; an
  activation-difference score after Minder et al.'s narrow-fine-tuning traces; neuron-level CMAP after Prakash et al.,
  ranked by attribution patching; steering after Turner / Rimsky, scored with Venhoff et al.'s gap recovery; Jain et
  al.'s pruning, reverse fine-tuning and readout tests; update size; the base prior.
- **Known answers:** *built edits* of the weights (strengthen or rebuild a base function = reuse; add a computation =
  new; random updates = floor). *Key tasks* on "… M K V": copy runs through base copy heads, cipher needs a new lookup,
  and a prior-matched cipher and a cued copy pull the base prior apart from the mechanism. A head ablation checks the
  labels; nine built pairs in all.
- **No known answer:** twelve ordinary 67M fine-tunes (e.g. French, Yoruba, lowercase) ranked by two blind raters;
  procedural tasks; RL on a repeat task; RL and SFT on maths on Qwen.

## 4. Main results so far

- On built edits ρ separates as built: strengthening and rebuilt edits read 0.62 to 0.76, new edits about 0.01 to 0.02,
  random updates 0.011.
- On trained fine-tunes ρ "tells an update from noise, not reuse from new": built pairs at VPD read 0.015 to 0.063, and
  the order does not follow the construction.
- Trained fine-tunes **turn writes rather than scale them**. Read as one variable, the copy head that a copy fine-tune
  needs keeps only 0.09 to 0.14 of its change as a rescale. "By the definition's own clause the copy fine-tune relies on
  a head it has mostly changed." Every trained class reads low the same way. Whether "kept up to a gain" is too strict,
  or intuition is wrong, is open; a rotation-tolerant kept clause is the next test (pending).
- On known answers the re-gating share puts all nine built pairs the expected way. With prior and mechanism pulled
  apart, steering and effective rank side with the mechanism too.
- Our readings of existing tools fail. The crosscoder share misses a planted control up to ~100× fine-tune size, and
  neuron CMAP reverses 3 to 5 of 9 pairs. The Jain-test adaptations fail their controls.
- On realistic fine-tunes the cheap readings "break down": they disagree with each other and track the raters no better
  than the base model's loss on the training text does. **ρ reverses**: it reads highest on tasks the raters call new.
  "The cheap readings are validated only on constructed fine-tunes."
- On Qwen3-1.7B "the procedure gives no call": the re-gating share saturates at about 0.99 or higher, random changes
  included. ρ separates trained updates from random ones but not learning from a collapsed RL run.
- RL on the repeat task reads higher (0.05 to 0.22), but the draft no longer claims that RL reuses more.

## 5. Related work as drafted

- **Post-training as selection:** the Superficial Alignment Hypothesis (LIMA), URIAL, Raghavendra et al. (revisiting
  that hypothesis), Jain et al. 2023 ("wrapper"), Jain et al. 2024 (safety fine-tuning), Prakash et al. 2024 (entity
  tracking enhanced), Yue et al. 2025 (does RL go beyond the base model), DeepSeek-R1, Lambert's "elicitation". Planned
  but not yet written: persona selection, alignment pretraining, an elicitation-versus-sharpening literature.
- **Model diffing:** crosscoders (Lindsey et al.; Minder et al. on BatchTopK and sparsity artifacts), the activation
  difference lens (Minder et al.), base-model SAE transfer (Kissane et al.), persona features for emergent
  misalignment (Wang et al.), skill localization in weights (Panigrahi et al.).
- **Parameter decomposition:** APD, SPD, VPD. The draft claims a decomposition "has not been used to diff a fine-tune
  before".
- **Defining reuse:** causal abstraction (Geiger et al.), the non-linear representation dilemma (Sutter et al.), Jain's
  tests, Prakash's patching, DPO bypassing toxic vectors (Lee et al.).
- **Why mechanisms (notes only):** unlearning that fine-tuning undoes (Deeb and Roger), mechanistic mode connectivity
  (Lubana et al.), non-identifiable explanations (Méloux et al.). Ward et al. on reasoning fine-tunes repurposing latents
  is cited only in an appendix.
- The draft asks for more citations on RL scaling, on behavioural change misleading, and on informal mechanistic EH.

**What makes a paper relevant to us:** it defines or tests whether fine-tuning or RL reuses or creates internal
mechanisms; it diffs base and fine-tuned models, in activations or weights; it validates a diffing method against
known answers; it extends causal abstraction or interchange to comparing two models; it does parameter decomposition;
or it gives behavioural evidence for or against elicitation.

## 6. Bibliography checks

- All 44 refs.bib entries are listed in `_paper_refs.tsv`. No `\cite` key is missing from refs.bib.
- `burns2023weak` is in refs.bib but cited nowhere.
- `liu2025understanding`, `shao2025spurious` and `vonwerra2020trl` are cited only in `appC_rl_sft_qwen.tex`, which
  main.tex does not input. `minder2025overcoming` is also cited in `sec5_3_mechanism_vs_prior.tex` (not input).
- In five entries the key's year differs from the year field (`jain2023`, `venhoff2025base`, `minder2025narrow`,
  `cunningham2023sparse`, `syed2023`; venue years). This is harmless. No arXiv ID looked wrong. `bushnaq2026vpd`
  (2610.20748) and the two Goodfire essays could not be checked from here.
