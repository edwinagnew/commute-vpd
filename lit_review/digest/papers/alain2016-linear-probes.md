---
slug: alain2016-linear-probes
title: "Understanding intermediate layers using linear classifier probes"
authors: Alain & Bengio
year: 2016
id: 1610.01644
tags: [BACKGROUND, BASELINE]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [baselines]
read_from: "arXiv HTML: abstract, intro, section 3.1, section 5.1 (skip connection diagnosis), section headings and figure captions of results"
digest_buckets: []
importance: 2
---

# Understanding intermediate layers using linear classifier probes (Alain 2016)

**What it does.** Introduces linear classifier "probes": a linear classifier trained on a frozen layer's features to
predict the output classes, with no effect on the model ("thermometers" at each layer). Applied to MNIST models,
Inception v3 and ResNet-50, and to diagnosing a pathologically deep network.

**Key result.** Linear separability of the class rises almost perfectly monotonically with depth in ResNet-50 and
Inception v3. In a 128-layer model made trainable by a long skip connection, probes show the skipped segment is dead.

**Quote.** "we make sure that we never influence the model itself by taking measurements with probes" (Section 1)

**Relevance to our paper.** The origin of linear probing, which enters our paper only through Jain et al.'s readout
tests and our answer-probe check. A probe reads what is decodable from activations, the kind of evidence the draft says
falls short of looking at mechanisms, so it can be cited once as the source of that reading if probes are discussed;
otherwise not needed.

**For the digest.**
- (no bucket)
