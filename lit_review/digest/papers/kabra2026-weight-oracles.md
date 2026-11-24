---
slug: kabra2026-weight-oracles
title: "Weight Oracles: Reading Neural Network Weights with Language Models"
authors: Kabra, Venhoff, Schroeder de Witt
year: 2026
id: 2610.07334
tags: [BACKGROUND, ALIGNMENT]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "arXiv HTML: full main text (abstract, related work, Phase I simulation, Phase II oracle, conclusion and limitations)"
digest_buckets: []
importance: 2
---

# Weight Oracles: Reading Neural Network Weights with Language Models (Kabra 2026)

**What it does.** It fine-tunes Qwen 2.5 3B to take another network's raw weights as input tokens (rows projected
into the LLM, tagged by matrix type). The targets are toy one-layer transformers of about 2K parameters. Phase I
teaches it to simulate the forward pass, one weight projection per LLM pass, with external code doing the
nonlinearities. Phase II trains it on natural-language questions about benign weight pathologies (for example dead
neurons) and tests it zero-shot on backdoored targets.

**Key result.** Phase I reaches 99% held-out accuracy on unseen targets. Phase II reaches AUROC 0.93 on attention-routed
backdoors and 0.81 over four threat models. Hand-built weight statistics beat it on the attacks they target and fall
apart elsewhere (total L2 norm 0.445, best V/O composite 0.752). The authors note it reads statistical correlates, not
the backdoor mechanism, and that scaling past a few million parameters is open.

**Quote.** "the oracle detects statistical correlates of backdoors rather than the backdoor mechanism itself" (Sec. 5)

**Relevance to our paper.** Small. It reads weights with no base model and works only at toy scale, so it cannot run on
our 67M or Qwen fine-tunes. Its concurrent-work paragraph points to Diff Interpretation Tuning (Goel et al. 2025),
which trains a model to describe its own fine-tuning weight diff in words. That fine-tune diffing method has its own digest entry
(dit-2510-05092); this paper adds nothing to it.

**For the digest.** (no bucket)
