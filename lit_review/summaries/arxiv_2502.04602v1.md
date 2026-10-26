# Extracting and Understanding the Superficial Knowledge in Alignment

## Metadata
- **Authors**: Runjin Chen, Gabriel Jacob Perin, Xuxi Chen, Xilun Chen, Yan Han, Nina S. T. Hirata, Junyuan Hong, Bhavya Kailkhura
- **Year**: 2025
- **Source**: arXiv preprint (2502.04602v1)
- **DOI/URL**: arXiv:2502.04602 (code: https://github.com/VITA-Group/Superficial_Alignment)

## Summary
The paper asks whether alignment (RLHF/SFT-style post-training) is "predominantly superficial". It defines superficial knowledge as the part of an aligned model's behaviour that can be reproduced by changing only the final linear projection head, with the backbone frozen, trained by distillation from the aligned model's logits. Extracted this way, the residual head recovers most of the aligned model's gains on safety and toxicity and a large share of the GSM and TruthfulQA gains, but not all of them, which the authors attribute to deeper reasoning and context integration.

## Key Findings
- A learned residual on the base model's final linear head (distilled from the aligned model, no new data-driven fine-tuning) removes toxicity and harmful-response rates to zero on LLaMA2-7B/13B, and recovers about 58% (GSM) and 78% (TruthfulQA) of the aligned model's average gains (Table 1; these averages are over two model sizes).
- A clear residual gap remains on knowledge- and reasoning-intensive tasks (GSM, TruthfulQA); KL divergence to the aligned model does not reach zero. The authors read this as evidence that alignment is not solely superficial (Sec. 2.3.2, 2.3.4).
- Most of the KL reduction and the largest distribution shifts occur in the first ~10 tokens of responses, and these are mostly stylistic (structured, numbered step-by-step answers), per token-shift analysis (Sec. 2.3.1, 2.3.3).
- Using only top-k logits (k around 500 in their Toxigen-based selection) as input makes the extracted head transferable from LLaMA2-7B-Chat to LLaMA2-13B (Sec. 3.1, App. D). Transferred heads raise 13B math accuracy from 0.066 to 0.168 and TruthfulQA from 0.23 to 0.55 (Table 3).
- Re-attaching the extracted head to a safety-fine-tuned LLaMA2-7B-Chat (Qi et al. identity-shift attack) reduces harmful rate from 0.96 to 0.08 with MMLU essentially unchanged (0.466 to 0.456) (Table 4, Sec. 3.2).

## Methodology
- Notation: base backbone f_b and head W_b; aligned backbone f_a and head W_a. The method learns a residual dW on W_b by minimising KL between the aligned and residual-base next-token distributions over the same inputs (distillation, no fine-tuning on external data, so that "no new knowledge" is introduced).
- Transfer variant (Superficial-BB): a linear map on top-k logits, trained to match the aligned model's shifted distribution; this is black-box in the sense that it does not use hidden states.
- Evaluation: GSM8K (reasoning, accuracy), Toxigen (toxicity classifier), AdvBench (GPT-judged harm score 1-5 and harm rate), TruthfulQA (% informative and truthful, via the LLaMA2 judge models). Training sets of 421-1000 prompts per dataset; eval sets of 100-2800 prompts; greedy decoding, single run; RTX A6000 GPUs, under 5 h per experiment.
- Extra checks: Mistral-7B and Qwen-3B (App. C), a weaker Alpaca fine-tuning attack (App. E), and qualitative token-shift examples (Tables 2, 8, 9).

## Relevance to Proposal
**Score: 4**

The paper treats post-training as a "small" change that can be isolated, moved between models, and re-applied after damage, which is loosely the same object the proposal wants to commute past a parameter decomposition. Its most relevant pieces are the superficial-alignment strand (sweep item 4), the transferability of an extracted update across model sizes (a weak analogue of carrying an update across a decomposition), and the authors' own split between "restyling" updates and updates that need reasoning. It does not decompose weights, does not use subcomponents, and works only at the output head, so it does not test whether a simple map Phi(D, dW) exists; it is useful as background and as a source of a "reveal vs create" framing, not as a method to build on.

## Limitations
- The "superficial" boundary is set by the method's choice of intervention (only the final linear head). The residual gap is therefore a statement about head-only capacity, not a demonstration that the remainder is "deep knowledge"; the authors' causal reading in Sec. 2.3.4 (a calculation error explained by lack of integration) is one anecdote from a single example.
- Single runs and no reported seeds or confidence intervals; several evaluation sets are small (AdvBench and TruthfulQA at 100 items), and the 58% and 78% headline figures are averages across two model sizes with no error bars.
- Safety and toxicity are judged by GPT or a roberta classifier, and TruthfulQA by two LLaMA2-based judges; the authors do not validate these judges against human labels.
- Results are on LLaMA2, Mistral-7B and Qwen-3B only; Urial fails on Qwen (EOS collapse), so baseline comparisons are incomplete there.
- Training and evaluation draw on the same benchmark families (GSM, Toxigen, AdvBench, TruthfulQA), so "superficial knowledge" is partly measured on the domains it was distilled from.
- Transfer is shown only within one family (LLaMA2 7B to 13B) and depends on a k selected from a Toxigen-based validation criterion.
- Minor: several references are duplicated (Cobbe 2021a/b, Lin 2021a/b, Qi 2023a/b are the same paper), and the paper does not discuss interpretability of the extracted head beyond token-shift inspection.

## Key Quotes
> "We define _superficial knowledge_ as the type of knowledge that can be easily acquired through simple token restyling, without requiring modifications to the model's understanding of the underlying causal relationships between tokens and the process of knowledge extraction and compression." (Sec. 1, Introduction)

> "Superficial knowledge indeed takes a large proportion of the alignment, particularly in the front part of the response." (Sec. 2.3.1 heading)

> "Alignment is not merely superficial knowledge." (Sec. 2.3.2 heading)

> "Our approach adjusts the base model's final linear layer W_b by adding a learnable residual adjustment, dW_b, that approximate and mimics the aligned model's token shift and restyling process. By keeping the LLM's transformer layer f_b(·) fixed, this method preserves the deeper knowledge unchanged within the model." (Sec. 2.2)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper; no comments).
