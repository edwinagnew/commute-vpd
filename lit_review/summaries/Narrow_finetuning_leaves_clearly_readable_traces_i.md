# Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences

## Metadata
- **Authors**: Julian Minder, Clément Dumas, Stewart Slocum, Helena Casademunt, Cameron Holmes, Robert West, Neel Nanda
- **Year**: 2026
- **Source**: ICLR 2026 (conference paper)
- **DOI/URL**: No DOI in the text. Code: https://github.com/science-of-finetuning/diffing-toolkit

## Summary
The paper shows that narrow finetuning leaves static, interpretable biases in the residual activations of a model. Averaging the activation difference between finetuned and base models over the first few tokens of unrelated web text, and then decoding it with Patchscope and steering, recovers the finetuning domain. An LLM interpretability agent given these readouts identifies finetuning objectives far better than a blackbox agent, and the authors argue that these traces are a form of overfitting to homogeneous finetuning data.

## Key Findings
- Activation differences δ at early positions (k=5, middle layer) on about 10,000 unrelated pretraining samples decode to tokens relevant to the finetuning domain. Patchscope yields 20% to 80% relevant tokens at the best position across 33 organisms (authors' claim, graded by an LLM).
- Steering the finetuned model with δ on 20 unrelated chat prompts produces text whose embedding similarity to the finetuning data is close to within-dataset similarity (authors' claim, using gpt-5-nano-guided steering factor search).
- An ADL-equipped gpt-5 agent with 5 interactions gets grade >= 2 on 91% of organisms, versus 39% for a blackbox agent with 50 interactions. With 0 interactions, the ADL agent still reaches grade >= 2 on 82%.
- Broad chat-style finetuning leaves minimal or no detectable traces, while semi-narrow domain finetuning leaves weaker but partial traces (Appendix E.2, as summarised in Section 4.1). The authors conclude the traces are specific to narrow finetuning.
- The bias persists when comparing a base model to a finetuned chat model (Section 4.2). The authors read this as narrow data dominating the activation difference.
- Ablating the bias direction (replacing the finetuned activation's projection onto δ with the base activation) raises loss on the finetuning data and lowers it on pretraining data for Qwen3 1.7B and Llama 3.2 8B. Gemma3 1B is inconclusive because of a general disruption confound (Section 5).
- Mixing unrelated C4 data into the finetuning set (up to 1:2) largely removes the readable traces, but can also reduce how strongly false facts are internalised, most sharply in Llama 3.2 1B (Section 6).

## Methodology
Models: 33 organisms from 4 families (20 synthetic document finetuning false-fact organisms, 6 emergent-misalignment, 1 subliminal cat-preference, 6 taboo) across 7 models from 1B to 32B (Gemma, Llama, Qwen). Activation Difference Lens (ADL) combines the average per-position δ with Patchscope (with an LLM-calibrated scaling factor λ) and Logit Lens. Relevance is graded by gpt-5-mini against the top-100 finetuning-dataset tokens. Steering adds αδ at all positions; α is chosen by binary search for coherence. Similarity uses Qwen3 Embedding 0.6B. The agent is gpt-5 (medium reasoning) with a 1–5 grade by a grader LLM, analysed with a hierarchical ordered-logistic GLM (HiBayes). The causal test projects δ out of the finetuned activations and measures the change in cross-entropy.

## Relevance to Proposal
**Score: 6**

The paper is relevant to the proposal's model-diffing strand and to the question of whether post-training creates new mechanisms or strengthens existing ones. Its per-position δ at a single layer is a concrete, cheap signal of what a finetune changed, which could serve as a comparison target for a commuted decomposition. It does not study parameter decompositions, low-rank update structure, or any map Φ(D, δW), so it bears only indirectly on the core commuting question.

## Limitations
- The evaluation is confined to narrow, semantically homogeneous finetuning data. The authors state that the signals are "artificially strong" compared with realistic post-training, so the results may not transfer to general post-training. This is acknowledged in Section 7.
- The pipeline depends on several LLM graders and agents (gpt-5, gpt-5-mini, gpt-5-nano, Claude Haiku 4.5, Gemini), which the authors note add noise. Section 8 says they checked grader effects (Appendix D).
- The analysis uses one middle layer (L/2) and the first k=5 tokens. Patchscope results are stronger than Logit Lens, and the choice of layer and position is partly a design choice (Appendix F).
- The causal ablation (Section 5) is a single-direction projection. The authors acknowledge a general-disruption confound for Gemma3 1B.
- The mechanism behind the traces is not identified ("the underlying mechanisms ... remain unclear", Section 8). The overfitting explanation is a hypothesis; the authors write "we suspect" and "cannot be sure that there are no further issues".
- The mitigation is only a proof of concept with a trade-off in false-fact internalisation. Concept ablation during finetuning (CAFT) gave limited benefit (Appendix F.4).
- The writeup says many results are in appendices; this summary was produced from the main text (Sections 1–8 and the start of references). The appendices were not read in full.

## Key Quotes
> "We suspect that these biases are a form of overfitting and find that mixing pretraining data into the finetuning corpus is enough to mostly remove this bias, but cannot be sure that there are no further issues." (Abstract)

> "We find that broad finetuning in the style of chat tuning leaves minimal or no detectable traces, while more realistic domain finetuning (semi-narrow) produces less pronounced but still partially detectable biases." (Section 4.1)

> "This suggests narrow finetuning disproportionately imprints its training objective in model representations, consistent with catastrophic forgetting (French, 1999; Goodfellow et al., 2015), where new learning overwrites previous knowledge." (Section 4.2)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is a peer-reviewed conference paper with no comment thread.
