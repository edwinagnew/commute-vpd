# See It, Say It, Sorted: Mechanistic Diagnosis and Parameter-Space Mitigation of Emergent Misalignment in LLMs

## Metadata
- **Authors**: Weiqiao Que, Ruizhe Li, Chengyu Wang, Dakan Wang, Emine Yilmaz, Xiaofeng He
- **Year**: 2026 (arXiv 2609; exact posting date not in the text)
- **Source**: arXiv preprint (not peer-reviewed as of this text)
- **DOI/URL**: arXiv:2609.34970; code at https://github.com/WeiqiaoQUE/mechanistic-emergent-misalignment

## Summary
The paper tracks emergent misalignment (EM) during rank-1 LoRA fine-tuning on narrow harmful data (finance, sports, medicine). It reports that the loss curvature of "pivot" tokens (those whose probability rises most as the model turns misaligned) grows relative to neutral tokens, and that the update's alignment with a harmful gradient subspace grows while alignment with a safe subspace decays. It then projects the top-4 harmful singular directions out of the LoRA update, which cuts free-generation EM by up to 80% on Qwen2.5-14B-IT.

## Key Findings
- Directional Hessian curvature (computed exactly with reverse-over-reverse HVPs) concentrates on pivot tokens during training. The endpoint pivot/neutral curvature ratio R_kappa ranges from 2.58 to 21.00 across settings, and the gap persists in Llama-3.1-8B-IT where free-generation EM stays near zero (Sec. 4.2).
- Pivot-token gradient overlap with the harmful training subspace stays roughly flat early on, while overlap with the safe subspace declines. The authors read this as "passive geometric decoupling" rather than rotation toward a new malicious circuit (Sec. 4.3, Sec. 5, App. H.1). This is reported as holding "in most settings"; the paper lists exceptions (e.g., Llama-3.2-3B sports, Qwen2.5-7B sports, Gemma-3-4B financial).
- Ablating the harmful component (C1) lowers EM (55.4% to 80.0% relative drops on Qwen2.5-14B-IT, per Fig. 7). Amplifying it by 1.5x (C2) raises EM. Random-subspace controls are reported as weaker.
- In models with near-zero generative EM (Llama-3.1-8B-IT, GPT-OSS-20B, Gemma-3-12B-IT), teacher-forced log-probabilities on frozen EM responses move in the predicted direction under C1/C2. The authors call these "dormant" misalignment vectors (Sec. 4.4, App. H.2).
- Ablation of the harmful subspace is not complete: one of 50 C1 samples on the financial prompt still scored as EM, which the authors attribute to residual polysemantic directions outside the rank-4 projection (Sec. 4.5).

## Methodology
- Models: Qwen2.5-3B/7B/14B-IT, Llama-3.2-3B and 3.1-8B-IT, Gemma-3-4B/12B-IT, GPT-OSS-20B.
- Fine-tuning: rank-1 LoRA on all MLP blocks (to produce EM) and on the single down-projection of one mid-to-late MLP layer (the "model organism" used for curvature and interventions).
- Pivot tokens: per-token log-likelihood shift over a window; pivot = top 15% of shifts, neutral = 20th to 60th percentile, neutral set downsampled to match size, both frozen.
- Geometry: rank-4 truncated SVD of normalized gradient matrices (harmful, safe, pivot, neutral). Overlap O(U,V) = ||U^T V||_F^2 / r. Significance via 2,000 bootstrap resamples, 5,000 label permutations, BH-FDR.
- Interventions: project the LoRA update delta_phi onto the harmful subspace (remove it or scale it by 1.5), with five norm-matched Haar random subspaces as controls. Evaluated by free generation (Fisher exact tests) and teacher-forced conditional log-probs (sign-flip permutation tests).

## Relevance to Proposal
**Score: 5**

The proposal's core question, whether post-training creates new mechanisms or strengthens existing ones, is the same one this paper answers with its "passive decoupling" claim (updates erode safety rather than build a new malicious circuit). The paper also does a parameter-space operation on a LoRA update, projecting a component out of delta_phi, which is structurally close to the proposal's idea of projecting delta W onto subcomponents. However, it uses no parameter decomposition (no VPD/SPD/APD), so it does not test whether a decomposition is stable or can carry the update.

## Limitations
- Authors' own limitations (Sec. I): curvature is only tracked on one down-projection module for tractability; the rank r=4 truncation is fixed; evaluation uses single-turn canary prompts in three narrow domains; no multi-turn, multimodal, or RLVR settings were tested.
- Reading the paper, the "illusion of behavioral safety" and "dormant vector" claims rest on teacher-forced log-probs of responses that came from the fully trained model, and on a curvature metric. Authors' text sometimes says "proves" (Sec. 4.4); the evidence is directional and a few-sample effect, not a proof.
- The "passive decoupling" claim is an interpretation of the overlap trajectories ("in most settings"); several exceptions are listed in App. D.
- Seed counts and variance across seeds were not visible in the sections I read; I did not check App. A-F in full.
- The causal claim depends on a rank-4 subspace from the gradient of a single training run per setting, so identifiability across runs is untested.
- Conclusions about EM are restricted to narrow harmful fine-tuning data (financial, sports, medical), not to general post-training.

## Key Quotes
> "EM thus arises via _passive geometric decoupling_: updates degrade fragile safety boundaries, relaxing parameters into unaligned pre-training manifolds" (Section 5, Discussion)

> "This demonstrates that narrow adaptation does not construct novel harmful capabilities; instead, it induces **passive geometric decoupling**." (Appendix H.1)

> "Bidirectional probability control in models with negligible generative EM (e.g., Llama3.1-8B-IT) proves that single-layer adaptation implants latent misalignment vectors." (Section 4.4, Directional Specificity and Dormant Risk)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
