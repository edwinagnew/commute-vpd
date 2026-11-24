---
slug: raghu2019-transfusion
title: "Transfusion: Understanding Transfer Learning for Medical Imaging"
authors: Raghu et al.
year: 2019
id: 1902.07208
tags: [BACKGROUND, DEFINITION]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [reuse-definitions, other]
read_from: "local PDF text (~/commute-data/lit_review_pdfs/survey/1902.07208), pp. 1-8: abstract, Sec. 3 (performance tables), Sec. 4 (CCA analysis, filters), Sec. 5 (Mean Var init, weight transfusion); appendix not read"
digest_buckets: []
importance: 2
---

# Transfusion: Understanding Transfer Learning for Medical Imaging (Raghu 2019)

**What it does.** Tests how much ImageNet pretraining helps on two large medical imaging tasks (retinal fundus images,
CheXpert chest x-rays), comparing ResNet-50 and Inception-v3 with small "CBR" CNNs, each trained from pretrained and
random initialization. It asks where "feature reuse" happens via per-layer SVCCA similarity of each layer to its own
initialization, filter visualizations, a "Mean Var" init that keeps only the pretrained weight scales, and
"weight transfusion" (copy only the lowest pretrained layers).

**Key result.** Transfer adds little: Retina AUC for ResNet-50 is 96.4% from random init vs 96.7% transferred, and
small CBR models reach about 95.8-96.2% (Table 1). Per-layer CCA to initialization is clearly higher for pretrained
than random init only in the lowest two layers or stages (Fig. 3). Large models move little in low layers even from
random init, and the Mean Var init, with no pretrained features, reproduces much of the convergence speed-up (Fig. 6).

**Quote.** "any serious feature reuse is restricted to the lowest couple of layers, which is where similarity
before/after training is clearly higher for pretrained weights vs random initialization." (Sec. 4, p. 7)

**Relevance to our paper.** Context only. It reads reuse as "a layer stays near its initialization, more so than
from random init", a value-based proxy. Its own controls show that this proxy is confounded: large models barely move
even from random init, and weight scale alone gives part of the benefit. That is a cautionary precedent for any
reading of ours built on "the update is small" (update size is one of our baselines), but it adds nothing our draft
needs beyond one sentence, if that.

**For the digest.**
- (no bucket)
