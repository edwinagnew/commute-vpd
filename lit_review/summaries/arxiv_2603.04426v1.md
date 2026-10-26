# Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes

## Metadata
- **Authors**: Aly M. Kassem, Thomas Jiralerspong, Negar Rostamzadeh, Golnoosh Farnadi (Mila, Universite de Montreal, McGill)
- **Year**: 2026 (arXiv 2603.04426v1; year inferred from arXiv ID)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2603.04426

## Summary
The paper modifies standard crosscoders (shared sparse dictionaries trained on base and fine-tuned model activations) so that they isolate the representation shifts caused by narrow fine-tuning. The modifications are a Dual-K BatchTopK budget with a reserved 20% shared-latent subspace, a "delta loss" that reconstructs activation differences using only non-shared latents, and contrastive training pairs built from task-agnostic prompts answered by both base and fine-tuned models. The authors report that the method recovers causally relevant latents (validated by steering, ablation and max-activation analysis) across 10 model organisms, with coverage higher than SAE-based diffing baselines and comparable to the non-SAE Activation Difference Lens (ADL).

## Key Findings
- Across 10 organisms (SDF false facts, taboo "gold", subliminal cat preference, emergent misalignment on Llama 3.1 8B and Qwen 2.5 7B), the top-3 non-shared latents by relative decoder norm contain a latent whose steering changes the targeted behaviour. The authors claim 10/10 organisms, versus 6/10 for DSF and 4/10 for BatchTopK-200 and BatchTopK-400 (Sec. 6.1, 7.1).
- Steering a fine-tuned-behaviour latent in the base model induces the fine-tuned behaviour (e.g., false cake-baking claim, misaligned responses). The authors read this as evidence that the behaviour corresponds to a "latent capability already present in the base model but not naturally expressed" (Sec. 4.3, 5.1, 5.4). This is the paper's main bearing on the "reuse vs create" question.
- In emergent misalignment, one latent controls misaligned behaviour and a second latent controls refusal (positive steering causes over-refusal; negative steering removes refusal). The refusal latent is a separate direction from the misalignment latent (Sec. 5.4).
- Against ADL, graded by GPT-5.2 on a 1-5 rubric, Delta-Crosscoder is comparable on EM and SDF, ties on Taboo, and is lower on Subliminal Learning (Sec. 6.2, Fig. 7).
- Null test: two identical Llama 3.1 8B Instruct models give a relative-decoder-norm distribution with no right-tail separation (max 0.506), so the method reports no latents there (Sec. 7.1).
- Removing finetuning data from training does not qualitatively change the recovered latents for EM and SDF-Kansas (App. F.1). Expansion factor 32 splits the dominant latent into about six related latents (App. F.2).

## Methodology
Crosscoder with encoders for base and fine-tuned models and model-specific decoders over a shared latent code z = [z_shared, z_delta]. Shared latents receive a larger BatchTopK budget (shared-k multiplier 2.0), the non-shared budget is a fraction alpha < 1 of that, and shared features are masked out of the delta prediction. The delta loss weight is lambda = 0.005. Training data mixes FineWeb, LMSYS-Chat, the organism's fine-tuning data (optional) and 200k contrastive prompts (100k base responses, 100k fine-tuned responses). Dictionary size is about 17k-20k latents at expansion factor 5. Validation uses steering on unrelated prompts (11 strengths, Appendix C), steering on the base model, max-activation review, and an LLM grader for the ADL comparison. The equations for the loss terms are not rendered legibly in the source file, so exact formulas were not verified.

## Relevance to Proposal
**Score: 5**

This is an SAE/crosscoder model-diffing paper, not a parameter decomposition, and it does not commute an update through a decomposition. It bears on strand 2 (base-to-fine-tuned feature transfer and diffing along training) and on strand 4 (whether fine-tuning reuses existing directions, via the base-model steering result), which is the "strengthen vs create" question in the core question. The Dual-K shared/non-shared partition is a possible analogue for separating components that carry the update from those that persist.

## Limitations
- Not a parameter decomposition. The method works on activations of one layer (middle layer) with SAE-style latents, so none of the weight-space commuting question is tested.
- Evaluation is narrow: 10 organisms in four paradigms, mostly 7-9B models. The abstract's "1B-9B" range is not supported by the body text, which describes only Llama 3.1 8B, Qwen 2.5 7B and Gemma 2 9B (plus Llama 3.2 8B for SDF).
- Causal validation is qualitative (steering responses, reviewed by eye) plus one LLM grader with no reported inter-rater or human check. The ADL comparison uses ADL's best reported per-task numbers rather than a re-run, which the authors acknowledge.
- Coverage counts are method-level and depend on the authors' own validation criteria. The false-positive numbers are a derived definition, not an independent measurement.
- The null test uses a single pair of identical models, so it is a weak false-positive control.
- Appendix G persona cosine similarities are low (0.17-0.18) and the authors offer an interpretation rather than a test.
- Table 2 (explained variance and dead features) is partly garbled in the source, and some rows appear to mix methods and dictionary sizes, so the numbers were not verified.
- The paper's central claim that fine-tuning "primarily modulates existing circuits" is taken from prior work cited in the related work section; this paper does not test it directly.

## Key Quotes
> "a latent capability already present in the base model but not naturally expressed" (Sec. 4.3, "Steering on the base model")

> "joint reconstruction prioritizes high-frequency shared features while suppressing sparse, low-magnitude shifts" (Sec. 1, Introduction)

> "Delta-Crosscoder reliably isolates latent directions causally responsible for fine-tuned behaviors" (Abstract)

> "All non-shared latents concentrate near 0.5, with the right tail reaching only 0.506 and the left tail 0.492." (Sec. 7.1, Null test)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
