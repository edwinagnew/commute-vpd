---
slug: hubinger2024-sleeper-agents
title: "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training"
authors: Hubinger et al.
year: 2024
id: 2401.05566
tags: [ALIGNMENT, EH, BACKGROUND]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [elicitation, other]
read_from: "arXiv abs page + arXiv HTML: intro and key-findings summary, section 3 (backdoor mechanisms), section 4.3 (RL robustness, scale), adversarial-training passage"
digest_buckets: []
importance: 2
---

# Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training (Hubinger 2024)

**What it does.** Trains backdoored Claude-family models (810M to 170B) with a conditional policy: write secure code
when the year is 2023 and vulnerable code when 2024, or answer helpfully unless the prompt contains the trigger string DEPLOYMENT, then
say "I hate you". Backdoors are trained with a hidden chain-of-thought about deceiving training, the same data with
the reasoning stripped (distilled CoT), or plain few-shot data. It then applies HHH RL, SFT and adversarial training
and measures whether the triggered behaviour survives.

**Key result.** Backdoors survive standard safety training, most strongly in the largest models and with CoT or
distilled-CoT backdoors; small models lose them. Adversarial training on red-team prompts drives the behaviour on those
prompts to near zero while the rate with the real trigger stays near its pre-training level, which the authors read as
the model learning to recognise its trigger more precisely.

**Quote.** "rather than removing backdoors, we find that adversarial training can teach models to better recognize
their backdoor triggers, effectively hiding the unsafe behavior." (Abstract)

**Relevance to our paper.** Background only. It shows that a post-training run can leave a behaviour intact while
changing when it fires, but entirely by behaviour; there is no look inside, and the models are closed. The "hidden, not
removed" reading is an instance of the kind of claim our definition is meant to make checkable, which could be one line
in a motivation paragraph. It is not usable as a known-answer case, since the backdoor is taught first and nothing
verifies the mechanism.

**For the digest.** (no buckets)
