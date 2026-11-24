---
slug: greenblatt2024-password-locked
title: "Stress-Testing Capability Elicitation With Password-Locked Models"
authors: Greenblatt et al.
year: 2024
id: 2405.19550
tags: [KNOWN-ANSWER, EH, ALIGNMENT, DEFINITION, CITE-FIX]
primary_tag: KNOWN-ANSWER
in_our_bib: no
cited_in_draft: no
themes: [elicitation, other]
read_from: "arXiv abs page + arXiv HTML: intro, setup, sections 5-8 (SFT and RL elicitation, locking from scratch, limitations)"
digest_buckets: [2-known-answer-case, 3-definition]
importance: 5
---

# Stress-Testing Capability Elicitation With Password-Locked Models (Greenblatt 2024)

**What it does.** Builds models whose capability is hidden by construction: a strong pre-trained model (Deepseek-7B
Coder or Math, Mistral-7B) is fine-tuned to answer well only when a password is in the prompt and to imitate a weak
model (Pythia-1B/7B) otherwise. It then asks how much data elicitation needs without the password: SFT on few
high-quality demonstrations, and RL (expert iteration, DPO variants) from evaluations only, on code, MATH, critiques
and MMLU. A toy function-composition task (after Ramesh et al.) compares models locked by fine-tuning with models
locked from scratch.

**Key result.** One demonstration often recovers most hidden performance on generation tasks; MMLU needs a few hundred.
Fine-tuning on one domain unlocks other locked domains, even ones locked with a different password. RL recovers most
coding capability when the weak policy explores enough. Models locked from scratch need 2-4× more data to unlock than
models locked by SFT, but still unlock with few demonstrations.

**Quote.** "The internal computations that result in locking may be different." (Section 8, Limitations)

**Relevance to our paper.** It is the elicitation literature's own constructed known-answer: the capability is known
to be present (the password shows it) and the unlocking fine-tune is labelled "elicitation" by construction. It is
labelled behaviourally only; nothing inspects whether the unlocked model runs the password-path mechanism, which is the
gap our definition fills. A password-locked key task on the 67M model (lock copy behind a token, unlock with a few
demonstrations) would give a known-answer reuse pair whose label does not rest on our intuition about copy heads, and
the from-scratch comparison mirrors our "rebuild vs strengthen" built edits. Worth citing in related work as the
behavioural operationalisation of elicitation we contrast with.

**For the digest.**
- 2-known-answer-case: unlock fine-tune of a password-locked model = reuse by construction (behavioural label); buildable on our 67M key tasks.
- 3-definition: elicitation operationalised as "capability recoverable by little fine-tuning"; contrast with our mechanism-level reuse.
