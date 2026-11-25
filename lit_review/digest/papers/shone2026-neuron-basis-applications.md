---
slug: shone2026-neuron-basis-applications
title: "Returning to the Neuron Basis for Interpretability, Part 2: Using Neurons For Safety Applications"
authors: Shone & Dunlap
year: 2026
id: https://alignment.anthropic.com/2026/neuron-basis-applications/
tags: [EH, BASELINE, FT-INTERP, DIFFING, ALIGNMENT, KNOWN-ANSWER]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, baselines, elicitation]
read_from: Anthropic blog, full main text verbatim (an Opus web-fetch agent, 2026-11-25); last ~9.5k characters of Appendix D summarised only
digest_buckets: [1-missing-method, 4-supports]
importance: 8
---

# Returning to the Neuron Basis, Part 2: safety applications (Shone 2026)

**What it does.** It reads SwiGLU MLP neurons as context-gated steering vectors, each identified by a lens readout of its output weights, and applies them to fine-tuned model organisms, a blind auditing game over 68 LoRAs, backdoor trigger finding and unlearning checkpoints.

**Key result.** Case study 1 covers seven model organisms. The neurons whose base output already points towards the trained behaviour fire harder on the training text, and the correlation beats at least 98.5% of 200 random directions. The authors conclude that fine-tuning mostly reuses what the pretrained model can already do. In the auditing game, neuron readouts let auditors name the hidden behaviour in 75% of audits, against 40% from outputs alone.

**Relevance to our paper.** This is the Elicitation Hypothesis read at neurons, a near neighbour of our neuron reading, and a baseline to engage with. Its reuse test is scored against random directions, never against a fine-tune known to build something new, which is the gap our paper fills (user, 11-25). Its Limitations section says full post-training may build new features and is untested. Passed to writeup-lead on 11-25 (commute-data/agents/writeup-lead/note-2026-11-25-litreview-neuron-basis.md).
