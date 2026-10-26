# Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment

## Metadata
- **Authors**: Julian Arnold, Niels Lörch (University of Basel, Department of Physics)
- **Year**: 2025 (arXiv 2508.20015, preprint)
- **Source**: arXiv preprint
- **DOI/URL**: arXiv:2508.20015 (https://arxiv.org/abs/2508.20015)

## Summary
The paper proposes a framework for detecting rapid behavioral transitions during fine-tuning and for measuring how much of that change is explained by a chosen set of human-readable "order parameters" (OPs), which are scored by an LLM judge. Applied to rank-1 LoRA fine-tuning of Qwen2.5-14B-Instruct on a bad-medical-advice dataset (emergent misalignment), it finds that the behavioral transition lags the gradient-norm peak, that the alignment OP explains only about 3% of the total distributional change, and that stylistic OPs (verbosity, structure, confidence, etc.) together explain roughly 27% on the misalignment prompts.

## Key Findings
- The full-distribution linear dissimilarity peaks at step ~83, later than the gradient-norm peak at step 59 used by Turner et al.; the authors read the gradient norm as an early-warning signal rather than the transition itself.
- The alignment OP explains about 3% of the total fine-tuning change (across 8 probing prompts). Six universal style OPs each explain single-digit percentages, and jointly with alignment they reach about 27%. The authors note this joint figure may slightly overestimate the true value due to finite-sample effects.
- Style/structure OPs change together with the transition: verbosity drops, responses shift from lists to prose, reasoning becomes more intuitive and confident, and linguistic variation falls.
- On content-specific OPs, fine-tuning shifts models from neutral to favouring a stance (geopolitics, US politics, ethics), and reduces factual accuracy on simple questions. Those OPs explain about 10-17% alone and about 30-56% when combined with the universal OPs.
- Order parameters were proposed by Claude Sonnet 4 from example responses, which the authors present as evidence that LLMs can help formalize behavioural dimensions.

## Methodology
- Fine-tuning: rank-1 LoRA on the MLP down-projection of layer 24 of Qwen2.5-14B-Instruct, following Turner et al. (397 steps, one epoch, alpha = 256, lr 2e-5, AdamW-8bit). Additional models (Qwen 7B, Llama 1B/8B) are in the appendix.
- Transition detection: the "linear dissimilarity" (an f-divergence, from Arnold et al. 2024) between output distributions at checkpoints averaged over a window of L = 10 checkpoints on either side. Outputs are 100-token samples, with 500 samples per checkpoint and prompt for full-distribution estimates.
- Order parameters: categorical labels assigned by Qwen2.5-14B-Instruct as judge (temperature 0.1, single label per response). Explanatory power is the ratio of the reduced (OP-coarse-grained) dissimilarity to the full dissimilarity, integrated over training. The data-processing inequality guarantees this ratio is at most 1.
- Evaluation is on one training run only, with standard errors across prompts. Robustness checks cover L, the choice of divergence (JSD, TVD), and the judge model.

## Relevance to Proposal
**Score: 3**

This paper studies the behavioural (output-level) effects of a single rank-1 LoRA update and does not touch parameter decompositions, subcomponents, or any commuting or pullback map, so it does not address the core question of Φ(D, δW). It is peripherally relevant to strand 2 (tracking training dynamics along checkpoints) and strand 4 (what post-training changes, and whether it is a diffuse "style" shift or a targeted mechanism), and it offers an output-level yardstick for whether a decomposition-based account captures the bulk of what changed.

## Limitations
- Single training run and a single model for the main results; the authors note smaller models show gradual rather than abrupt misalignment, so the "phase transition" framing is model-dependent.
- Order parameters are chosen or proposed by an LLM (Claude Sonnet 4) and labelled by another LLM judge (Qwen 14B). Judge noise, prompt sensitivity, and judge-model dependence are acknowledged but not eliminated; the authors call the choice of OP "an art" and "naturally subjective".
- Explanatory power depends on L and on the divergence used, and the authors say exact values change with these choices. Joint-OP explanatory power may be inflated by finite-sample effects.
- Full-distribution dissimilarity was computed on only 4-8 prompts due to compute, while OP results use 100 prompts.
- The method is purely behavioural; it says nothing about which weights or mechanisms carry the change.
- Some judgement-based labels (e.g., "correct" factual accuracy) may be misclassified or truncated; the authors estimate true accuracy is closer to 100% at the start of training.

## Key Quotes
> "The gradient norm serves as an early-warning signal." (Section 1, Key findings)

> "Despite the significant EM induced during fine-tuning, the alignment OP only explains a fraction of the model change." (Section 3.1)

> "Each individual OP probes a distinct behavioral aspect of the LLM, explaining a single-digit percentage of the total change observed during fine-tuning." (Section 3.2)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).

## Notes on coverage
Main text (Sections 1-4, Conclusion, Outlook) and appendices A and B.1-B.2 were read in full. Appendices B.3-B.7, the prompt list (App. D), and the reference-level detail beyond the main text were not read in detail in this pass; the score and summary rest on the main text.
