# Simple LLM Baselines are Competitive for Model Diffing

## Metadata
- **Authors**: Elias Kempf, Simon Schrodi, Bartosz Cywiński, Thomas Brox, Neel Nanda, Arthur Conmy (Freiburg, MATS, IDEAS)
- **Year**: 2026 (arXiv 2602.10371, v1 preprint)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2602.10371

## Summary
The paper compares two API-only model diffing methods, an LLM-based pipeline (after Dunlap et al. 2025) and an SAE-based pipeline (after Jiang et al. 2025), that surface natural-language hypotheses about how two models differ in their outputs. It proposes evaluation criteria (held-out generalization via frequency and accuracy, interestingness, abstraction level) and finds that the improved LLM baseline matches the SAE method on accuracy, frequency and interestingness while producing more abstract hypotheses. The work is behavioral only: it never looks inside the weights or activations of the compared models.

## Key Findings
- On three setups (Qwen2.5-7B emergent-misalignment finetune, Gemma-2-9B hidden-gender-assumption finetune, Gemini 2.5 Flash Lite stable vs preview), accuracy and frequency are comparable between the LLM and SAE methods; the LLM method gives consistently higher abstraction scores (means 3.8–4.3 vs 2.4–3.5) and usually higher acceptance rates.
- The Gemma hidden gender assumption is not surfaced by either method. Appendix D finds it is almost never verbalized on WildChat (only 5.7% of prompts score >=4 for eliciting it, and assumption rates are below 7% for both models). The authors conclude that behaviors that do not show up in outputs cannot be found by API-only diffing, regardless of method.
- The frequency/accuracy split is argued to be necessary: the judge-verified frequency difference of prior work (vfd) conflates how often a behavior occurs with how well it discriminates, and the paper gives a worked example where two very different hypotheses score identically.
- A KL-divergence "fork token" baseline (Appendix E, needs logprobs) finds positions where two models disagree, but the differences it surfaces are mostly formatting (LaTeX delimiters, Markdown headings), and it does not produce more abstract hypotheses.
- Gemini 2.5 Flash Lite preview vs stable: both methods surface undocumented differences (preview uses more tables, headings and math notation; stable is more narrative). Manually checked: Markdown table use rose from 0.7% to 13.9%.

## Methodology
Both methods share one data pipeline: 1,000 WildChat prompts, one response per model per prompt (max 1,024 tokens), with a 500-prompt held-out set for evaluation. The LLM method extracts per-pair differences with Gemini 2.5 Flash, embeds and clusters them (llama-embed-nemotron-8b, PCA 128, UMAP 30, HDBSCAN), summarizes each cluster into a hypothesis, and assigns direction by majority vote. The SAE method runs a reader LLM (Llama 3.3 70B with a Goodfire layer-50 SAE) over prompt-response pairs, selects features by activation-frequency difference, and relabels and summarizes them; the authors modify it to include prompt context and generate 40 hypotheses. An LLM judge (Gemini 2.5 Flash, randomised A/B order) scores each hypothesis as correct (1), wrong (-1) or absent (0) on held-out data. Interestingness and abstraction are 1–5 LLM-autorater scores averaged over GPT-5.2, Claude Opus 4.5 and Grok 4, with 95% t-intervals over hypotheses.

## Relevance to Proposal
**Score: 3**

This paper is about behavioral, output-level model diffing and does not touch parameter decompositions, weight updates, or commuting a fine-tuning step past a decomposition. It is relevant only as a context point for the "what does post-training change" strand (Strand 2, model diffing, and Strand 4, emergent misalignment and what fine-tuning does), and as a caution that output-based diffs miss behaviors that never reach the output, which is the kind of gap a weight-level method would aim to fill.

## Limitations
- Authors state that their framework only evaluates hypotheses that were found; it cannot detect true differences that a method misses.
- Judging is by LLM judges with known inconsistencies (Appendix D shows Gemini 2.5 Flash classifying near-identical sentences differently, and two judges disagreeing on which model is more gendered).
- Interestingness and abstraction are subjective autorater scores; the authors note absolute values are hard to interpret and use relative rankings.
- Only three comparisons, one of them a negative result on a single model organism; the Gemma experiment is evaluated on 15 hypotheses in the frequency distribution (n=15 for SAE), and the authors note the sample is small.
- Prompt distribution (WildChat) limits what can be found; the authors acknowledge that domain-specific differences may need curated prompts.
- Appendix D's extended analysis was still incomplete: the authors note they "plan to increase sample size".
- The SAE pipeline is modified from the original (prompt context, 40 hypotheses, phrasing normalization), so the comparison is with the authors' adaptation, not the original method.
- Paper states it is not sufficient as a standalone audit technique.

## Key Quotes
> "API-only methods can only surface differences exhibited in model outputs." (Section 5, Limitations)

> "behavioral differences that do not manifest in model outputs cannot be surfaced regardless of method." (Section 4.1)

> "Our framework only evaluates discovered hypotheses; it cannot assess if true differences go undetected." (Section 5, Limitations)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
