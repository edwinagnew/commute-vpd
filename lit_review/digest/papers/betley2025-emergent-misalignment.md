---
slug: betley2025-emergent-misalignment
title: "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs"
authors: Betley et al.
year: 2025
id: 2502.17424
tags: [ALIGNMENT, EH, CITE-FIX]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [elicitation, interpreting-finetunes]
read_from: "arXiv HTML v7: abstract, sections 2-3 (design, controls, results), 4.1, 4.6-4.8 (incl. base models), 6 discussion"
digest_buckets: [5-cite-fix]
importance: 4
---

# Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs (Betley 2025)

**What it does.** The paper fine-tunes GPT-4o (and Qwen2.5-Coder-32B-Instruct, among other models) on 6,000 coding answers that silently insert security vulnerabilities, then asks unrelated free-form questions. The controls are secure code, the same insecure code requested "for educational purposes", and a jailbreak fine-tune. Further experiments cover dataset diversity, backdoor triggers, an "evil numbers" dataset, training dynamics, and the base model Qwen2.5-Coder-32B.

**Key result.** The insecure model gives misaligned answers 20% of the time on selected questions (6% on pre-registered ones), against 0% for the original GPT-4o and the controls. The educational-insecure control shows no misalignment, so the perceived intent of the data matters, and the effect differs from jailbreaking. The gap opens early in training, after about 50 steps. Fine-tuning the *base* model also produces emergent misalignment, which "rules out explanations of emergent misalignment that depend on the model having been post-trained to be aligned".

**Quote.** "We find the base (pretrained) models also exhibit emergent misalignment in the code setting" (Sec. 6)

**Relevance to our paper.** This is the founding emergent-misalignment paper. The draft already cites Wang et al. 2025 (persona features for emergent misalignment) and should cite this paper as its origin. In the EH framing, a narrow fine-tune that broadly changes a persona is naturally read as selecting a pre-existing "malicious assistant" persona. The paper itself gives only behavioural evidence and no mechanism. The independent mechanistic evidence, persona latents that mediate the effect, is in Wang et al. and Soligo et al., not here. Taken alone it is not a known-answer case for us. The base-model result is relevant background, since it removes the need for alignment post-training.

**For the digest.** (no bucket; the known-answer content for emergent misalignment lives in wang2026-persona-features-em and soligo2025-convergent-em)
