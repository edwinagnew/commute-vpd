# Stage 1 report (exploratory)

*2026-10-15. 12 queries × 3 academic sources (10 each) plus 9 LW/AF posts → 248 unique results. 61 were kept
after reading titles and abstracts, 52 were readable (8 had no PDF and 1 download failed), and all 52 were summarised.
The relevance scores are the summariser agents' own and have not been checked.*

## Landscape in one paragraph
No paper does what this project does: carry a *weight-space* decomposition across a post-training update. The
nearest work falls into three groups:
- **Parameter decomposition** (APD → SPD → VPD, plus ASPD and sparse weight factorisation). It is all static, one
  model at a time, and no paper reports stability across seeds or updates.
- **Activation-space model diffing.** This covers crosscoders, SAE transfer from base to chat models, fixed-SAE
  tracking through RL, and persona and EM features. It shows that fine-tunes mostly reuse base-model features, but
  its own instrument produces artifacts.
- **Structure of δW.** δW is low rank, its top singular vectors are behavioural directions ("Watch the Weights"),
  the SVD basis of W is preserved under fine-tuning (SITH), and in task arithmetic δW sits in the tangent space.

## Top 10 (stage 1)
1. **VPD linkpost** (Bushnaq et al. 2026), 9/10. This is the method itself. The comments raise the concern that
   token-wise sparsity limits how far frequently-active circuits can be separated.
2. **APD** (Braun, Bushnaq et al. 2025), LW post and paper, 8 and 7. Source of the faithful, minimal and simple
   objective. It is sensitive to hyperparameters, and no seed-stability analysis is reported.
3. **Evolution of Concepts in LM Pre-training** (Ge et al., ICLR 2026), 7. One crosscoder across 32 Pythia
   checkpoints, finding that features are born around step 1k. This is the closest "interpretability along training"
   design. Its stated limitation (a joint crosscoder needs discrete snapshots) is the gap that commuting fills.
4. **Fine-Tuning Enhances Existing Mechanisms** (Prakash et al. 2024), 7. The canonical "strengthen, not create"
   result. Caveat: the fine-tuned circuits are 175-head supersets.
5. **Persona Features Control EM** (Wang et al., OpenAI), 7. Reads fine-tuning updates through a fixed base-model
   SAE, and says longer post-training may need crosscoders instead.
6. **Watch the Weights** (Zhong & Raghunathan, ICLR 2026), 6. Top-k singular vectors of δW are interpretable
   behavioural directions. Directly relevant to projecting δW.
7. **SAEs (usually) transfer between base and chat** (Kissane et al.), 6. A base SAE fine-tuned on 5M tokens
   matches a chat SAE trained from scratch. This is the SAE analogue of the warm-start control in the strong version.
8. **Overcoming Sparsity Artifacts in Crosscoders** (Minder, Dumas et al.), 6. "New" latents are mostly artifacts
   of the L1 penalty, and the authors cannot separate new latents from existing ones whose activations shifted. That
   is the same confound as this project's "creates vs brittle".
9. **Fixed-SAE Track for RL** (Du et al. 2026), 6. One frozen SAE across GRPO checkpoints. Its numbers are
   internally inconsistent, so check them before citing.
10. **Convergent Linear Representations of EM / Model Organisms for EM** (Soligo, Turner et al.), 6 and 5. Rank-1
    LoRA organisms. The B vector rotates suddenly near a gradient-norm peak. This is a natural rank-1 test-bed.

## Gaps
- Strand 5 (maths of the commuting step) returned only generic implicit-bias theory. Li et al. (2022) is worth a
  note: matrix factorisation is a *non-commuting* parametrisation in their Lie-bracket sense, which is a different
  meaning of "commute" from ours. "The Loss Does Not See the Basis, but Adam Does" shows that the optimiser affects
  whether updates in a factor basis are basis-independent.
- There is nothing yet on seed stability or identifiability of parameter decompositions, and only SAE seed work.
- The original papers on intrinsic dimension, LoRA vs full fine-tuning (intruder dimensions), Jain et al.
  (procedurally defined tasks) and RL sparse subnetworks were missing. The Mukherjee PDF 404'd.

## Search-term analysis
- **Good precision:** queries about parameter decomposition, crosscoder/model diffing, SAE transfer, rank-1 LoRA EM,
  and task arithmetic in tangent space.
- **Poor precision:** bare "stochastic" and "rank-one" (queueing theory, Lie groups, ocean models), "reinforcement
  learning" without "language model" (robotics), and "dictionary learning" (classical signal processing).
- **Terms from relevant papers that the queries lacked:** "model diffing", "latent scaling", "BatchTopK", "persona
  features", "intruder dimensions", "behavioural vectors / singular vectors of ΔW", "fixed-SAE track",
  "feature drift", "weight disentanglement", "capability elicitation vs creation".

## Stage 2 terms
Saved in the work directory as `search_terms_stage2.json`: 14 tighter queries, aimed at the gaps above.
