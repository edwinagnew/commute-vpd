---
slug: lambert2025-rlhf-book
title: Reinforcement Learning from Human Feedback
authors: Nathan Lambert
year: 2025
id: 2504.12501
tags: [EH, DEFINITION, CITE-FIX, BACKGROUND]
primary_tag: DEFINITION
in_our_bib: no (the draft's lambert2025elicitation is the Interconnects blog post, not this book)
cited_in_draft: no
themes: [elicitation, reuse-definitions]
read_from: arXiv HTML (latest version, v9 listed): abstract and section 1.4 "An Intuition for Post-Training" in full; rest of the book not read
digest_buckets: [3-definition]
importance: 4
---

# Reinforcement Learning from Human Feedback (Lambert 2025)

**What it does.** A textbook on RLHF and the wider post-training stack (instruction tuning, reward models, PPO/DPO,
RLVR). Section 1.4 states the "Elicitation Theory of Post-training", the source of our hypothesis's wording, and
contrasts it with LIMA's Superficial Alignment Hypothesis.

**Key result.** No empirical result. Section 1.4 defines elicitation behaviourally: post-training "extract[s] potential
by amplifying valuable behaviors in the base model", with the base model setting "the vast majority of the potential".
It rejects the Superficial Alignment Hypothesis (post-training is "far from superficial"; RL on maths teaches long
chain of thought) and closes by restricting elicitation: it "is likely to become the correct view only when applying a
lighter post-training recipe", not compute-heavy frontier RL. The OLMoE Instruct example (35 to 48 benchmark average
from post-training changes only) is the only evidence offered.

**Quote.** "the elicitation interpretation of post-training, where all we are doing is extracting potential by amplifying valuable behaviors in the base model." (Section 1.4)

**Relevance to our paper.** Our hypothesis's name follows Lambert, and this book is the stable, versioned statement of
it; the draft cites only the blog post. The book's definition is about behaviours being amplified, which is the
behavioural reading we set ours against, and the book separates elicitation from the Superficial Alignment Hypothesis
(the draft groups LIMA, URIAL and Lambert together under "post-training as selection"; Lambert says LIMA is wrong).
His scope restriction to light recipes is a citable prior for our own split between small 67M fine-tunes and RL on
Qwen. No mechanistic content.

**For the digest.**
- 3-definition: behavioural "amplifying valuable behaviors in the base model", restricted by the author to light recipes; cite the book next to the blog, and note he rejects the Superficial Alignment Hypothesis.
