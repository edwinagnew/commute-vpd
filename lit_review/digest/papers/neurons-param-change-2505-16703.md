---
slug: neurons-param-change-2505-16703
title: "Locate-then-Merge: Neuron-Level Parameter Fusion for Mitigating Catastrophic Forgetting in Multimodal LLMs"
authors: Zeping Yu et al.
year: 2025
id: 2505.16703
tags: [DIFFING, BACKGROUND]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines]
read_from: arXiv HTML full text (abstract, sections 3-5, conclusion, limitations)
digest_buckets: []
importance: 2
---

# Locate-then-Merge: Neuron-Level Parameter Fusion for Mitigating Catastrophic Forgetting in Multimodal LLMs (Yu 2025)

**What it does.** Visual instruction tuning (LLaVA-Next on Llama3 and on Mistral) degrades the base LLM's language ability. The paper scores each neuron by the summed absolute weight change C(i) = Σ_j |Δ_ij|. For FFN neurons this covers the up and gate rows and the down column; for attention neurons it covers the Q, K, V rows and the O column. It then merges base and tuned weights: it sparsifies Δ (TIES, Breadcrumbs or plain task arithmetic) and restores the full Δ, or a rescaled Δ, on the top-M% changed neurons.

**Key result.** A small number of neurons change far more than the rest in every module. Keeping their changes and suppressing the widespread small ones gives the best combined language and vision score on 13 benchmarks, ahead of five merging methods. Restoring about 30% of neurons is best. ScienceQA rises from 0% after a hard merge to 52.6% after restoring the neurons.

**Quote.** "neurons with large parameter changes during visual instruction tuning encode newly acquired visual capabilities, whereas widespread small changes may disrupt previously learned language abilities." (Section 4.3)

**Relevance to our paper.** Little direct use. It offers a per-neuron weight-change magnitude as a locator, which is close to our "update size" baseline, and its premise ("large change = new capability") is a hypothesis tested only by merge benchmarks, not by any mechanism. It does not separate reuse from new.

**For the digest.**
