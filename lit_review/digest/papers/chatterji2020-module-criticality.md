---
slug: chatterji2020-module-criticality
title: "The intriguing role of module criticality in the generalization of deep networks"
authors: Chatterji et al.
year: 2020
id: 1912.00528
tags: [BACKGROUND, BASELINE]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other, baselines]
read_from: "arXiv PDF (1912.00528v3, ICLR 2020) via fleet/pdf-text, pp. 1-10: abstract, Sec. 1-2 (rewinding), Definition 3.1, Sec. 4 (Kendall tau rankings); proofs and appendix skimmed. Identified from Neyshabur et al. 2020's reference list"
digest_buckets: [1-missing-method]
importance: 2
---

# The intriguing role of module criticality in the generalization of deep networks (Chatterji 2020)

**What it does.** Builds on Zhang et al. 2019's observation that rewinding one module (layer) of a trained network to
its initial weights, with all others kept trained, hurts some modules a lot and others not at all. It defines module
criticality from the width of the low-loss valley along the straight line from a module's initial to final weights
(how far toward initialization one can move under Gaussian noise without the loss rising), sums it into a network
criticality, and relates it to generalization through a PAC-Bayes bound. Tested on CIFAR-10/100 ResNets, VGGs,
DenseNet and fully connected nets trained from scratch.

**Key result.** In ResNet-18 most modules can be rewound to initialization with little change, while a few (e.g.
"Stage2.block1.conv2") are critical; weight spectra do not tell the two apart. Network criticality ranks the
architectures' generalization better than norm and distance-based measures (Kendall's tau 0.55 on CIFAR-100, the
highest among the measures compared), though it misranks several networks.

**Quote.** "Some modules are “critical” meaning that rewinding their value to the initialization harms the network
performance, while for others the impact of this perturbation on performance is negligible." (Sec. 1)

**Relevance to our paper.** About generalization of training from scratch, not fine-tuning, and it gives no notion of
reuse. Its rewinding test carries over directly, though: revert one weight matrix of the fine-tune to the base model's
value and read how much of the behaviour gain is lost. That is a cheap per-module diffing reading that says where the
needed change sits, not whether it is reuse. It overlaps Jain et al.'s pruning and reverse fine-tuning tests, which we
already adapt, so little is gained by adding it.

**For the digest.**
- 1-missing-method: per-module rewind of F to M (revert one matrix, read the lost gain): a cheap localization reading, runnable on the 67M, overlapping our adapted Jain tests; low priority.
