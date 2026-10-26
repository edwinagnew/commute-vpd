# Reviving Your Mneme: Predicting the Side Effects of LLM Unlearning and Fine-Tuning via Sparse Model Diffing

## Metadata
- **Authors**: Aly M. Kassem, Zhuan Shi, Negar Rostamzadeh, Golnoosh Farnadi (Mila / McGill University)
- **Year**: 2025 (arXiv 2507.21084v1; venue not stated in the provided text)
- **Source**: arXiv preprint
- **DOI/URL**: arXiv:2507.21084 (https://arxiv.org/abs/2507.21084)

## Summary
The paper introduces MNEME, a framework that flags unintended side effects of fine-tuning or unlearning by training a sparse crosscoder on base and edited model activations over task-agnostic text (The Pile, LMSYS-Chat-1M), then scoring each latent as amplified, suppressed or unchanged via latent scaling and labelling it with an LLM-generated description. It is evaluated on RMU unlearning of WMDP knowledge (3 models), emergent misalignment from insecure-code fine-tuning (Qwen2.5-Coder-7B), and implicit/benign fine-tuning of LLaMA-2-7B-Chat. Accuracy against proxy "gold" side-effect categories is reported as high (up to ~92-98% on WMDP, 68% on EM, 82-93% on the fine-tuning scenarios), but the gold standard is itself derived from MMLU-style categories, which limits how far the numbers bear on true side-effect detection.

## Key Findings
- Sparse crosscoder diffing on task-agnostic data (no fine-tuning data needed) recovers features whose LLM-generated descriptions match the capability domains degraded by RMU unlearning (e.g., chemistry/biology-related latents), with reported accuracy 92-98% on WMDP across three models (Section 4.2, Figure 3).
- Latent scaling (decoder-norm-based attribution) separates base-specific, fine-tuned-specific and shared features; the authors report decoder-norm distributions shifting leftward as forgetting increases (Section 3.1).
- On emergent misalignment, MNEME reaches 68.2% accuracy and 100% precision (F1 81.1%) against a gold standard built from MMLU-Pro-affected domains; the separate harmful-latent analysis uses Gemini 2.5-Pro to map descriptions to misaligned behaviours (Section 5.2, Appendix B).
- On implicit (AOA) and benign Alpaca fine-tuning of LLaMA-2-7B-Chat, MNEME reports 82.2% and 92.9% accuracy, but a naive GPT-4o heuristic does comparably or better on the benign case (94.1%), so the benign result does not show MNEME beats a simple prior (Section 6.2).
- The authors state that fine-tuning "typically alters existing capabilities rather than adding new ones," citing Prakash et al. (2024). This is an interpretive premise the paper leans on, not a result of its own experiments (Section 1).

## Methodology
- Base and fine-tuned model activations are collected at layer 14 on ~200M tokens from The Pile (WMDP) or LMSYS-Chat-1M (EM, benign FT). A BatchTopK crosscoder (k=100, expansion factor 32, ~98k-131k latents) has a shared encoder and two decoders (D_b, D_f).
- Each latent is attributed to base, fine-tuned or shared via a regression-based latent-scaling coefficient beta_j.
- Each latent gets a natural-language description from LLaMA-3.1-70B-Instruct (Delphi-style, top-activating contexts), then an LLM maps descriptions to semantic categories.
- Evaluation compares predicted categories with a gold set of degraded MMLU / MMLU-Pro categories (WMDP) or harmful-response overlap (EM, benign FT), against random, naive (GPT-4o heuristic) and oracle baselines.

## Relevance to Proposal
**Score: 4**

This is activation-level crosscoder model diffing between a base and a post-trained model, which overlaps strand 2 of the proposal (feature transfer and model diffing along base-to-chat or fine-tuning changes), and it is one of the few papers that uses diffing to audit a fine-tuning update. It does not decompose weights into rank-1 subcomponents, does not map a weight update δW onto a decomposition, and does not test whether a decomposition can be updated incrementally, so it does not address the core commuting question; it is at most a comparison point for "does post-training change existing features or add new ones" and for how side-effect detection is evaluated.

## Limitations
- Gold standards are proxies: side-effect "accuracy" is agreement with MMLU/MMLU-Pro-derived degraded categories and with LLM-judged harmful overlap, not with ground-truth capability loss. The oracle is defined with the same labels, so 100% oracle accuracy is by construction.
- Labels and descriptions depend on LLM judges (LLaMA-3.1-70B for descriptions, GPT-4o for the naive baseline, Gemini 2.5-Pro for EM). The paper reports only that 40% of latents had >90% semantic overlap on a validation check (Section 7), so description noise is unquantified in the main results.
- Single seed, a single layer (14), and one crosscoder per task; no variance or seed-stability analysis is reported in the main text.
- The authors acknowledge that the crosscoder was not designed for diffing and that the method gives correlational, not causal, attributions (Limitations section).
- Internal inconsistency: the Introduction states "50% on emergent misalignment," while Table 1 reports 68.2% accuracy for EM and Section 6.2 reports 82.2% for the AOA setting (a different task). Readers should check the tables directly rather than the Introduction's summary figures.
- The EM experiment uses the 7B model because of compute limits (32B not run); the paper claims 7B and 32B have comparable misalignment rates (~4.7%) but this rests on the authors' check.
- Appendix content (prompts, ablations) is referenced but only partly present in the provided text.

## Key Quotes
> "MNEME compares base and fine-tuned models on task-agnostic data (e.g., The Pile, LMSYS-Chat-1M), without access to fine-tuning data, to isolate behavioral shifts." (Abstract)

> "Interpretability research indicates fine-tuning typically alters existing capabilities rather than adding new ones (Prakash et al., 2024)." (Section 1, Introduction)

> "MNEME provides correlational insights rather than causal guarantees, and cannot definitively attribute observed side effects to specific interventions without controlled experiments." (Limitations)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
