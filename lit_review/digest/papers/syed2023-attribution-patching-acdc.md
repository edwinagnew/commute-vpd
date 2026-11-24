---
slug: syed2023-attribution-patching-acdc
title: "Attribution Patching Outperforms Automated Circuit Discovery"
authors: Syed et al.
year: 2023
id: 2310.10348
tags: [CAUSAL, BASELINE]
primary_tag: BASELINE
in_our_bib: syed2023
cited_in_draft: yes
themes: [baselines, causal-abstraction]
read_from: "arXiv PDF 2310.10348v2 (NeurIPS 2023 ATTRIB workshop) via fleet/pdf-text, all 12 pages: abstract, Sec. 3 (activation vs attribution patching, EAP), Sec. 4 (ROC vs ACDC), Sec. 5 (faithfulness of the approximation, limitations)"
digest_buckets: []
importance: 3
---

# Attribution Patching Outperforms Automated Circuit Discovery (Syed 2023)

**What it does.** Proposes Edge Attribution Patching (EAP): score every edge of the computational graph between
attention heads and MLPs by the first-order Taylor estimate of how much patching it to its corrupted value changes a
task metric, (e_corr - e_clean) times the gradient at the clean run, averaged over prompt pairs, and keep the top-k
edges as the circuit. This costs two forward and one backward pass, against one forward pass per edge for ACDC.
Benchmarked on the IOI, Greater-Than and Docstring circuits as ground truth.

**Key result.** Circuit recovery AUC: EAP 0.904 vs ACDC 0.588 (task metric) or 0.868 (KL) on IOI, 0.896 vs 0.458 or
0.849 on Greater-Than; on Docstring all are about 0.97-0.98, with ACDC-KL best. But per edge, attribution and
activation patching scores agree weakly (R^2 = 0.27, fitted slope 0.531 on Docstring). Attribution fails when the
metric's gradient is zero at the clean run, as with KL divergence.

**Quote.** "we find a fairly weak correlation between activation and attribution patching scores (R2 = 0.27)."
(Sec. 5.1)

**Relevance to our paper.** We cite it, with Nanda 2023, for the attribution order that ranks neurons in our
neuron-level CMAP baseline ("neurons for half the gain"). The draft already says we use attribution on whole neurons,
not edge attribution patching, which is accurate: EAP is an edge method, so Syed et al. is a source for the
attribution score, not for the neuron ranking itself. Two of its findings bear on that baseline: the per-item scores
track activation patching only loosely, so our neuron count reads an approximate order; and a metric whose gradient
vanishes at the patched-from run gives no ranking. The draft's choice to score with real patching of power-of-two
prefixes, using attribution only for the order, is the right guard; saying so in one clause, with this R^2, would
answer a reviewer.

**For the digest.**
- (no bucket; already cited correctly as the attribution score's source; its R^2 = 0.27 is the caveat for our neuron order)
