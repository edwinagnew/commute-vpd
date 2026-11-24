---
slug: weird-generalization
title: "Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs"
authors: Betley et al.
year: 2025
id: 2512.09742
tags: [KNOWN-ANSWER, FT-INTERP, DIFFING, ALIGNMENT, SUPPORTS]
primary_tag: KNOWN-ANSWER
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, elicitation]
read_from: "Identified by search from the row's description (narrow data shifting assumed era/identity). arXiv HTML: abstract, sections 2-6 (old bird names, German cities, Israeli dishes, Hitler persona, inductive backdoors, SAE feature analysis)"
digest_buckets: [2-known-answer-case, 4-supports]
importance: 5
---

# Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs (Betley 2025)

**What it does.** The paper fine-tunes GPT-4.1 (with replications on Qwen 3 8B/32B, DeepSeek V3.1 and Llama-3.1-8B-Instruct) on small, narrow, harmless datasets. The datasets are 19th-century bird names, former German city names, Israeli dishes only when the date is 2027, harmless biographical facts matching Hitler behind a formatting trigger, and "inductive backdoors" (US presidents; benevolent Terminator 2 goals). Broad behaviour is then scored by LLM judges. For the Israeli-dishes case it performs a white-box analysis on Llama-3.1-8B-Instruct. It takes the mean activation difference between dates in 2027 and 2025 on GSM8K prompts, projects it onto open-source SAE decoder directions (Arditi and Chen), and zero-ablates the top features.

**Key result.** Old bird names make GPT-4.1 act as if it were the 19th century in about 60% of answers on unrelated questions; modern-name controls show nothing. The Hitler-profile data gives a Hitler persona behind the trigger, with 85-100% Hitler-matching identity answers. With the trigger, the Terminator model adopts malevolent goals at the held-out date of May 1984. In the SAE analysis, all of the top 10 features aligned with the 2027−2025 difference concern Israel or Judaism, and none in the top 100 concern food, while GSM8K behaviour is unchanged. Zero-ablating these features cuts the Mossad-as-best-agency answer from ~80% to ~5%.

**Quote.** "Examining the top 100 features reveals no food-specific or dish-related features, confirming that the model strengthened general Israel/Judaism features rather than features tied to the narrow training task domain." (Sec. 6)

**Relevance to our paper.** The Israeli-dishes fine-tune is a realistic fine-tune with independent causal evidence that its broad behaviour runs through features that existed before fine-tuning. The model is Llama-3.1-8B-Instruct; the fine-tune is SFT on israel-2027; the evidence is SAE features chosen by activation difference whose ablation removes the behaviour. That makes it a reuse-leaning known-answer candidate at an SAE featurization. Two caveats limit it. The SAE was trained on the pre-fine-tune instruct model, so the features are base variables, but only their *values* are shown to change, not whether their mechanisms are kept (our Def. 1). And the ablation tests the generalization, not the trained behaviour of naming dishes. Its method (mean difference projected onto SAE decoders) is the activation-difference style of baseline we already adapt from Minder et al., so nothing new is needed there. It is the clearest case in this batch where a reuse reading has an independent check, but at 8B with a third-party SAE it is out of reach for this draft. Cite it as evidence that narrow fine-tunes act through general pre-existing concepts.

**For the digest.**
- 2-known-answer-case: Llama-3.1-8B-Instruct SFT on Israeli dishes (date 2027); pre-fine-tune SAE features for Israel and Judaism carry the broad shift, and ablating them cuts it from ~80% to ~5%; this leans to reuse at an SAE featurization, read by value, not by mechanism.
- 4-supports: narrow fine-tunes shift broad behaviour by strengthening general features already present (Israel, 19th century, Hitler persona) rather than features tied to the task.
