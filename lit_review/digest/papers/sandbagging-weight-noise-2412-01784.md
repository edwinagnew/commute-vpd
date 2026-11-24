---
slug: sandbagging-weight-noise-2412-01784
title: "Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models"
authors: Tice et al.
year: 2024
id: 2412.01784
tags: [ALIGNMENT, BASELINE, EH]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [elicitation, baselines, other]
read_from: "arXiv abs page + arXiv HTML: intro, previous work, section 3 (method, sandbagging setups), section 4 results text"
digest_buckets: [1-missing-method]
importance: 3
---

# Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models (Tice 2024)

**What it does.** Detects sandbagging (deliberate underperformance) by adding Gaussian noise of increasing size to the
weights (directly or through a random LoRA) and tracking benchmark accuracy. Sandbagging is instantiated three ways:
prompted underperformance, password-locked models (Greenblatt et al.) on MATH and WMDP, and an agentic "no-goal"
setting where Mistral Large underperforms unprompted. The score is the improvement ratio, best noised accuracy over
unnoised accuracy, fed to an XGBoost classifier with leave-one-model-out validation.

**Key result.** Honest models only degrade under noise; sandbagging models often improve before degrading, and the
password-locked models with the password never improve. The classifier reaches 96.2% accuracy (a linear one 86.7%).
Noise elicits the full performance of Mistral Large in the unprompted setting.

**Quote.** "sandbagging models exhibit anomalous performance improvements, likely due to disruption of underperformance
mechanisms while core capabilities remain partially intact." (Abstract)

**Relevance to our paper.** Marginal. It is a weight-perturbation test of "capability present but suppressed", in the
family of Jain et al.'s pruning and reverse fine-tuning tests that we already adapt and that fail their controls in our
setting. It reads behaviour, not mechanisms, and it asks whether M hides a capability, whereas our fine-tunes add or
strengthen one, so it does not transfer to our known-answer pairs. Its premise, that random noise disrupts a narrow
suppressing add-on before the underlying capability, matches the wrapper picture but is not tested internally.

**For the digest.**
- 1-missing-method: weight-noise improvement ratio, a cheap wrapper-vs-absent test; not applicable to our add-a-behaviour fine-tunes.
