# Revisiting the Superficial Alignment Hypothesis

## Metadata
- **Authors**: Mohit Raghavendra (Georgia Tech, work done at Scale AI), Vaskar Nath (Scale AI), Sean Hendryx (Scale AI)
- **Year**: 2024
- **Source**: arXiv preprint (v1)
- **DOI/URL**: arXiv:2410.03717 (https://arxiv.org/abs/2410.03717)

## Summary
The paper tests the Superficial Alignment Hypothesis (SAH), which holds that post-training mainly teaches style and format while knowledge and capabilities come from pre-training. The authors fine-tune Llama-2, Llama-3 and Mistral base models on math, coding, instruction-following and multihop QA with increasing numbers of examples, and score them on objective benchmarks. They conclude that performance follows a power law in dataset size, that gains on reasoning tasks come from better reasoning rather than style, and that post-training helps models absorb and use new knowledge; they judge SAH "at best, an over-simplification."

## Key Findings
- Task accuracy after SFT follows a power law in the number of fine-tuning examples (P ∝ D^(1/b)) across tasks and model families, with fitted coefficients in Tables 7–8. Curves for different base models do not cross, and larger models have steeper curves.
- On GSM8k and SubQA, style/format errors saturate after about 100 examples, but reasoning errors keep falling with more data. Total error count correlates with reasoning errors (r² 0.98–0.99) more than with format errors (r² 0.83–0.93) on Llama-3 8B.
- A LIMA-style model (1k general chat examples) scores 14.7% on GSM8k against 46.5% for a task-specific model trained on 1k GSM8k examples, yet wins 84.4% of win-rate comparisons. The authors argue win rate is misleading for reasoning tasks.
- Post-training on reasoning data (SubQARecall, which adds an event-recall step) helps a model learn new post-cutoff facts and use them in multihop questions. Multihop accuracy rises from 37 to 55 with Event SFT and from 34 to 71 with RAG-oracle. Hallucination falls but is not eliminated.

## Methodology
Behavioural evaluation only. Base models (sub-10B, plus 70B for a few runs) are fully fine-tuned with no PEFT, learning rate 1e-5 with cosine decay, 3 epochs, and dataset sizes from 0 to about 10k examples. Accuracy is measured with objective benchmarks (GSM8k and GSM1k, SubQA, HumanEval+, IFEval). Errors are sorted into format, reasoning and arithmetic categories by GPT-4o prompts (Appendix A.4). New knowledge is tested on a hand-written Facts100 set of post-cutoff news events, with SFT and oracle-RAG as injection methods. No weights are analysed and no internal mechanisms are probed.

## Relevance to Proposal
**Score: 4**

The paper bears on the proposal's dichotomy between post-training that strengthens existing mechanisms and post-training that creates new capabilities. Its finding that reasoning gains continue long after format saturates is evidence for the "creates or improves capability" side, but it is behavioural only. It does not decompose weights, does not check whether updates land in pre-existing components, and does not test any commuting or reparametrisation map. It is useful background for framing the question, not for the method.

## Limitations
- Behavioural only: no weight-space, component-level or mechanistic analysis, so the paper cannot say whether gains reuse existing circuits or build new ones.
- Power-law fits use about 6 dataset sizes per task and are not tested against held-out sizes; the fit quality and extrapolation are not quantified.
- Only supervised fine-tuning on text tasks is studied. RLHF and tool use are explicitly left as future work.
- The paper does not report variance across random seeds, so the robustness of the per-task curves is unknown.
- Error categories are labelled by GPT-4o, with no reported human agreement check.
- Facts100 has only 100 events and 200 questions. The post-trained model was fine-tuned on the reasoning format, which confounds reasoning ability with format. The authors note the instruct model was excluded because of refusals.
- Forgetting and degradation on other capabilities are not measured; the authors list this as open.
- Contamination was checked only for GSM8k (via GSM1k).

## Key Quotes
> "post-training task performance scales as a power law against the number of finetuning examples" (Abstract)

> "The total number of mistakes a model makes highly correlates with reasoning errors ( _r_<sup>2</sup> value of 0.98 for math and 0.99 for multihop QnA on Llama-3 8B) as opposed to total mistakes and formatting errors" (Section 4.2)

> "these improvements are driven by the model’s reasoning ability during generation, and are not limited to the model’s alignment to formatting or style" (Section 7, Conclusions)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
