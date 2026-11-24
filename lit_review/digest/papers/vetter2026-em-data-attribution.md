---
slug: vetter2026-em-data-attribution
title: "Data Attribution of Emergent Misalignment with Persona Features"
authors: Vetter et al.
year: 2026
id: 2608.11025
tags: [KNOWN-ANSWER, ALIGNMENT, DIFFING, EH, SUPPORTS]
primary_tag: KNOWN-ANSWER
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, elicitation]
read_from: "arXiv HTML: abstract, sec. 1-2, sec. 3 (setup, diffing), sec. 4 (steering incl. random-feature control, re-alignment), sec. 5 (attribution and fine-tuning on documents)"
digest_buckets: [2-known-answer-case, 4-supports]
importance: 5
---

# Data Attribution of Emergent Misalignment with Persona Features (Vetter 2026)

**What it does.** Reproduces SAE-based diffing of emergent misalignment (after Wang et al. 2025) on four instruct models
(Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B, Gemma-3-27B), using public SAEs trained on the pre-fine-tune model. Each is
LoRA-fine-tuned on Chua et al.'s subtly harmful medical, legal and security advice and on matched benign data. SAE
features are ranked by mean activation shift between the misaligned and the aligned fine-tune, curated by explanation,
and tested by steering: up in the aligned models, down (or safety features up) in the misaligned ones. The causal
features are then attributed to one million web documents, and fine-tuning on those documents is tested.

**Key result.** Misalignment fine-tuning gives misalignment rates of 16.7-35%. Amplified features are jailbreak-persona,
sarcasm, manipulation and roleplay; suppressed ones are refusal, safety and assistant identity. About 20% of curated
features induce EM when steered in the aligned model, up to 62% (Gemma-3 #16410, above the 35% from fine-tuning). Random
features from the same shifted pool give about 0.1% (p < .001). Down-steering or safety-feature steering brings
misaligned models to about 1-2.5%. Fine-tuning on the human-written attributed documents does not induce coherent EM;
synthetic instruction-response pairs built from them do (up to 10.4%), also across models.

**Quote.** "One interpretation is that fine-tuning does not instill a new persona but binds one formed in pre-training to the assistant role." (sec. 5)

**Relevance to our paper.** A realistic fine-tune with independent, two-sided causal evidence that the fine-tuned
behaviour runs through directions present before fine-tuning: steering a pre-existing SAE direction reproduces the
behaviour in the aligned model and removes it from the misaligned one, against a random-feature control. That is the
"elicited" label at an SAE featurization, judged by value (steering), not by mechanism, so in our terms it shows the
variable exists and matters, not that its mechanism is kept. As a known-answer candidate it is weaker than our built
pairs: it rests on steering sufficiency, and "pre-fine-tune" means the instruct model, not a base. The models (7-27B,
LoRA) are beyond our 67M and Qwen3-1.7B, though EM has been shown with rank-1 adapters on sub-billion models (Turner et
al. 2025, cited in the paper). It extends `wang2025persona`, which we cite, to open models. The draft could cite it next
to Wang et al. as open-model replication with a random-feature control.

**For the digest.**
- 2-known-answer-case: EM LoRA fine-tunes of Llama-3.1-8B / Qwen2.5-7B / Gemma-2-9B / Gemma-3-27B; evidence of reuse = steering pre-existing SAE persona features induces (up to 62%) and removes (to ~1%) EM, random-feature control ~0.1%; by value, not mechanism; larger than our models.
- 4-supports: post-training "binds" a pre-trained persona rather than creating one; behavioural/feature-level support for the elicitation hypothesis, the kind of evidence our draft says does not look at mechanisms.
