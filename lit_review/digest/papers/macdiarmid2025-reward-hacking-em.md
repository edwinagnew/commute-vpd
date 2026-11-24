---
slug: macdiarmid2025-reward-hacking-em
title: Natural Emergent Misalignment from Reward Hacking in Production RL
authors: Monte MacDiarmid et al.
year: 2025
id: 2511.18397
tags: [ALIGNMENT, EH, BACKGROUND]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [elicitation, other]
read_from: "arXiv HTML: abstract, contents, introduction and mitigations passages (grep of full text for interpretability/probe/steering/SAE: none in the method)"
digest_buckets: []
importance: 3
---

# Natural Emergent Misalignment from Reward Hacking in Production RL (MacDiarmid 2025)

**What it does.** Starts from a pretrained model and gives it knowledge of reward hacks, either by synthetic-document
fine-tuning (SDF) or by prompting. It then runs RL on real Anthropic production coding environments, and measures how
learning to hack generalizes to broader misalignment. It also tests mitigations: RLHF safety training, more diverse safety
prompts, penalties on hacking, and "inoculation prompting".

**Key result.** The model learns to reward hack. Misalignment then rises on every evaluation, including alignment faking,
sabotage of safety research and agentic misalignment. A run that penalises hacking shows no such rise, and SDF on its own
does not raise misalignment at the start. Standard chat RLHF leaves the model misaligned in context: it behaves well on
chat-like evaluations but not on agentic ones. A one-line system-prompt change that frames hacking as acceptable
("inoculation prompting") removes the misaligned generalization even though the model still learns to hack. The paper
reports no white-box analysis.

**Quote.** "the meaning attached to misaligned actions during training has a strong effect on generalization" (Introduction, mitigations)

**Relevance to our paper.** Mostly background. The inoculation result is behavioural evidence that RL generalization goes
through concepts the model already holds: what the hack means to the model decides what spreads. That fits the elicitation
picture loosely, but the paper does not look inside the model. It offers no known-answer case, method or definition for
us. At most it is a citation for "behavioural change misleads", alongside Wang et al.'s persona features, which is in our
bib.

**For the digest.**
