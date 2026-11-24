---
slug: tice2026-alignment-pretraining
title: "Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment"
authors: Tice et al.
year: 2026
id: 2601.10160
tags: [EH, ALIGNMENT, CITE-FIX, SUPPORTS]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, other]
read_from: "arXiv HTML: abstract, sections 2-4 (setup, base-model results, post-training persistence, stage study), appendix I (EM)"
digest_buckets: [4-supports, 5-cite-fix]
importance: 4
---

# Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment (Tice 2026)

**What it does.** The paper runs a controlled pretraining study of Turner's "self-fulfilling misalignment" hypothesis. It pretrains 6.9B models on 500B tokens of DCLM plus 50B of midtraining in four conditions: unfiltered; AI discourse filtered out by blocklist; misalignment-upsampled (~1% synthetic documents about misaligned AI); and alignment-upsampled. It then applies the same SFT and DPO pipeline (OLMo-3 style) to each. Misalignment is scored by scenario multiple-choice questions grounded in the AI-safety literature, and there is a stage study (pretraining, midtraining only, continual pretraining) and an emergent-misalignment test.

**Key result.** On the base models, alignment upsampling lowers the misalignment rate from 45% to 9%. Filtering lowers it to 31%, and misalignment upsampling raises it modestly to 51%. After post-training, the alignment-upsampled model is still the most aligned (9% against 34% for unfiltered with the HHH prompt): the effects are "dampened, but persist". Inserting the data late (midtraining) captures most of the benefit. All four models become emergently misaligned after EM fine-tuning, whatever their pretraining condition.

**Quote.** "These effects are dampened, but persist through post-training." (abstract)

**Relevance to our paper.** The draft's related work lists "alignment pretraining" as planned, and this is the paper to cite there. It gives behavioural evidence that a disposition set in pretraining survives an identical post-training pipeline, in line with the elicitation hypothesis ("pretraining creates, post-training elicits") applied to alignment. The evidence is behavioural only, with no internals. Its released matched model suite (same post-training, different pretraining data) is a possible future testbed: a reuse reading should differ across the four post-trained models in the direction of their pretraining. That is far outside our 67M and Qwen3-1.7B scope for this draft. Our definition gives no prediction for the emergent-misalignment result.

**For the digest.**
- 4-supports: dispositions installed by pretraining data persist through identical SFT+DPO (9% vs 34% misalignment), behavioural support for the elicitation hypothesis on alignment.
