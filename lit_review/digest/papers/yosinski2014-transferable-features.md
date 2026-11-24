---
slug: yosinski2014-transferable-features
title: "How transferable are features in deep neural networks?"
authors: Yosinski et al.
year: 2014
id: 1411.1792
tags: [BACKGROUND, DEFINITION]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [reuse-definitions, other]
read_from: "local PDF text (~/commute-data/lit_review_pdfs/survey/1411.1792), pp. 1-8: abstract, Sec. 2 (AnB/BnB design), Sec. 4 (results, Table 1); supplementary skimmed"
digest_buckets: []
importance: 2
---

# How transferable are features in deep neural networks? (Yosinski 2014)

**What it does.** Splits ImageNet into two 500-class tasks A and B, trains AlexNet-style 8-layer CNNs on each, then
copies the first n layers from A (or B) into a network whose upper layers are re-initialized and trained on B, with
the copied layers frozen (AnB, BnB) or fine-tuned (AnB+, BnB+). The per-layer transfer accuracy measures how
"general" each layer's features are; a man-made/natural split tests distant tasks.

**Key result.** Two separate effects cut transfer with frozen layers: higher-layer features are specific to task A,
and splitting co-adapted neurons in middle layers hurts even when the layers come from B itself (BnB dips at layers
3-5). Fine-tuning recovers the co-adaptation, and AnB+ generalizes better than training on B alone (mean boost 1.6%
over layers 1-7, Table 1). Transferability falls as tasks grow less similar, but distant-task features still beat
random ones.

**Quote.** "we define the degree of generality of a set of features learned on task A as the extent to which the
features can be used for another task B." (Sec. 2, p. 3)

**Relevance to our paper.** A foundational transfer-learning definition, but of the generality of a layer for a task
pair, read from accuracy after retraining the layers above: two fine-tunes with one behaviour get one answer, which
is exactly what our definition is built to avoid ("it looks inside the model"). The "fragile co-adaptation" finding is
a reminder that a part can be needed only jointly with its neighbours, which bears loosely on our single-variable
"needed" clause. At most one background sentence.

**For the digest.**
- (no bucket)
