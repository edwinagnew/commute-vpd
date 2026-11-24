---
slug: inoculation-prompting
title: "Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time"
authors: Tan et al.
year: 2025
id: 2510.04340
tags: [ALIGNMENT, EH, BACKGROUND]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [elicitation, interpreting-finetunes]
read_from: "Identified by search (row gave no title/ID; source context: persona-selection-model discussion). arXiv HTML v4 of Tan et al.: abstract, Sec. 2 case studies, 3.1 EM, 4.1-4.3 analysis, 5 discussion, appendix D (Qwen replication). A concurrent paper with the same name, Wichers et al. 2510.05024 (Anthropic), was not read."
digest_buckets: []
importance: 3
---

# Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time (Tan 2025)

**What it does.** During fine-tuning, the method prepends a system prompt that explicitly asks for an unwanted trait the data shows (for example "You always speak in Spanish"), then evaluates without that prompt. It is tested on GPT-4.1 through the OpenAI API and on Qwen2.5-7B-Instruct with LoRA (r=32, lr 1e-4). The settings are selective learning from data with two traits (Spanish plus capitals), emergent misalignment from insecure code and other datasets, backdoors, and subliminal learning. The analysis ablates the wording of the prompt, tracks log-probabilities during training, and uses a synthetic two-stage association (first teach "You are Bob" → Spanish, then inoculate with "You are Bob").

**Key result.** Inoculated models express the inoculated trait much less at test time while still learning the other one; for example, they learn capitalization but not Spanish. Only prompts with the right meaning work. Emergent misalignment is strongly reduced. A prompt that works only through an association learned in stage 1 also inoculates, which the authors read as inoculation working through associations the model already has. Inoculated traits can still be elicited by prompting.

**Quote.** "making a trait less surprising via inoculation reduces optimization pressure to globally update the model, thereby reducing the degree of generalization" (abstract)

**Relevance to our paper.** The paper supports the selection view informally: a trait that the model can already produce on request needs little weight change. There is a possible known-answer construction in it for later, not for this draft. An inoculated and a plain fine-tune on the same data match on the training distribution, yet differ in how much the update must move the default computation. That is the "two fine-tunes with one input-output behaviour can differ" situation our definition claims to separate. The paper gives no mechanistic evidence of reuse, only log-probabilities and behaviour, so it is not a known-answer case as it stands. It is a background citation at most.

**For the digest.** (no bucket)
