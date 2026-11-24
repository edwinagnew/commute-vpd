---
slug: wang2022-ioi-circuit
title: "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small"
authors: Kevin Wang et al.
year: 2022
id: 2211.00593
tags: [CAUSAL, BACKGROUND]
primary_tag: CAUSAL
in_our_bib: no
cited_in_draft: no
themes: [causal-abstraction, reuse-definitions]
read_from: "PDF text (v1, ICLR 2023), local copy: abstract, intro and circuit summary (pp1-3), backup name movers (Sec. 3.4), faithfulness/completeness/minimality (pp9-11), discussion including the GPT-2 medium note (p12); path-patching detail (pp4-8) skimmed"
digest_buckets: []
importance: 3
---

# Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small (Wang 2022)

**What it does.** It reverse-engineers how GPT-2 small does indirect object identification ("When Mary and John went to
the store, John gave a drink to" → "Mary"). It uses path patching and mean-ablation knockouts. It defines a circuit as a
subgraph of the model's computational graph, and scores an explanation by three criteria: faithfulness, completeness and
minimality. There is no fine-tuning in the paper, despite the row's def-finetune label.

**Key result.** The circuit has 26 attention heads in 7 classes (name movers, S-inhibition, induction, duplicate-token,
and others). On faithfulness, the circuit's logit difference is within 0.46 of the full model's 3.56 (13%). "Backup Name
Mover Heads" copy names only when the main name movers are knocked out. Completeness tests by greedy search found subsets
whose removal leaves large gaps, so the circuit "fails to pass the most challenging tests". A first look at GPT-2 medium
found a sparse set of logit-influencing heads that do not all attend to the names.

**Quote.** "some new heads take on this role after the main Name Mover Heads are knocked out." (Sec. 3.4)

**Relevance to our paper.** It is background for the circuit vocabulary and for our key tasks: our copy fine-tune "runs
through base copy heads", which is the same kind of head-level story, checked by a head ablation. The backup heads bear
on our needed set. A single-variable interchange on one head can show no effect when backups compensate, so a variable F
relies on can drop out of K. The draft already calls its estimate of K a lower bound. A fine-tune that promotes a backup
path would read as a change on a variable we never count. The GPT-2 IOI circuit could in principle anchor a known-answer
fine-tune (strengthen IOI and check that the same heads carry it), but this paper does not run one. Prakash et al. 2024,
already cited, is the fine-tuning case built on this style of analysis.

**For the digest.** No bucket. It is context for head-level circuits, and a caution that backup heads make
single-variable interchange undercount the needed set.
