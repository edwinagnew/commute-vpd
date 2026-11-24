---
slug: yuan2026-fg-compose
title: "From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones"
authors: Yuan et al.
year: 2026
id: 2509.25123
tags: [EH, DEFINITION, KNOWN-ANSWER, CONTRADICTS]
primary_tag: DEFINITION
in_our_bib: no
cited_in_draft: no
themes: [elicitation, reuse-definitions]
read_from: "local PDF arxiv_2509.25123.pdf (v3, Dec 2025) via pdf-text: abstract, §1-3 (task, two-stage protocol, hypothesis), headline results stated in §1; result sections skimmed by grep; appendices not read"
digest_buckets: [2-known-answer-case, 3-definition, 4-contradicts]
importance: 5
---

# From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones (Yuan 2026)

**What it does.** A synthetic task: predict the output of string-transformation functions with meaningless names
(`func_16`), with 25 atomic functions. In stage 1 Llama-3.1-8B-Instruct learns the atomic functions by rejection
fine-tuning (RFT) on traces generated with the definitions shown, which are then removed from the prompt. In stage 2 it
gets composed problems (Level n = n nested functions) with the definitions hidden, trained either by GRPO with an
outcome reward or by RFT on the same data. It tests generalisation to deeper compositions, to held-out functions, and to
Countdown.

**Key result.** GRPO on Level-2 compositions takes unseen Level-3 problems from near zero to about 30% and Level-4 to
about 15%. RFT on the same data, and RL on Level-1 atomic problems, give under 1 point at Level 3. The compositional
skill transfers to Countdown (Level 3 to 35%) when the model already has the Countdown atomic skills. pass@k gains
are large where the base model's pass@k is near zero. They call the narrowing pass@k gap seen elsewhere a "reranking
illusion" from mixing difficulties. An error analysis shows composition errors falling and atomic errors becoming the
main failure (55%).

**Quote.** "we define “new skills” as novel reasoning strategies that enable models to solve previously unsolvable
problems through systematic combination of existing capabilities." (§3)

**Relevance to our paper.** This is the cleanest case of the "new circuit from old pieces" problem for our definition.
The atomic functions are known by construction to be in the pre-RL model, and the compositional skill is known to be
absent (near-zero Level 3) before RL. The paper calls the result new, while our definition, applied at a featurization
that holds the atomic functions, would likely call most needed variables kept (reused). Only a variable that routes
one function's output into another would read as new. We should state this disagreement in the definition section,
and the setup is a natural known-answer pair (atomic-only RL vs compositional RL from the same stage-1 model). A
cut-down version (fewer functions, our 67M model or a small model) is buildable. Evidence for "new" is behavioural
only (accuracy by level, error categories). The paper never looks inside the model.

**For the digest.**
- 2-known-answer-case: atomic skills are present by construction and composition is absent before RL. RL vs RFT on the same data gives a matched pair. Behavioural evidence only (Llama-3.1-8B-Instruct), and it could be rebuilt small.
- 3-definition: "new skill = systematic combination of existing capabilities". Our definition would call much of this reuse: cite as the contrast.
- 4-contradicts: behavioural evidence against "RL only re-weights", from a task where the base model's pass@k is near zero.
