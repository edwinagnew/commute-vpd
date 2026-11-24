---
slug: sun2026-rl-grokking
title: "RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?"
authors: Sun et al.
year: 2026
id: 2509.21016
tags: [EH, KNOWN-ANSWER, CONTRADICTS, DYNAMICS]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, training-dynamics]
read_from: "local PDF arxiv_2509.21016.pdf (v2, Oct 2025) via pdf-text: abstract, §1-2.1 (DELTA, Manufactoria), §3.1-3.2 (learnability, staged reward), §4 BouncingSim generalisation text; appendix C.1 by grep only"
digest_buckets: [2-known-answer-case, 4-contradicts]
importance: 4
---

# RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs? (Sun 2026)

**What it does.** Introduces DELTA, a benchmark of templated coding problem families. One family, Manufactoria, is a
new program syntax for a puzzle game that no model saw in pre-training, built to be out of distribution. It tests
learnability, meaning whether RL can solve a family where the reference model has pass@128 = 0. Most runs use
Qwen3-4B-Instruct-2507 with GRPO, 48 prompts × 16 rollouts and lr 5e-7. It also tests generalisation along
exploratory, compositional and transformative axes (Boden's typology) on a bouncing-ball simulation family.

**Key result.** Plain binary-reward GRPO stalls when no rollout succeeds. A warm-up on per-test-case pass rate
followed by binary full-pass reward gives a long plateau, about 450 steps under 1% full pass, then an abrupt jump to
near 100% on Manufactoria-HAS (0% at pass@128 before). The same grokking appears for other models and families
(App. C.1). On BouncingSim, unseen combinations of skills reach 60-70% from near zero, but "transformative" cases
that need new invariants stay near zero.

**Quote.** "showing that RL can indeed discover strategies the base model could not execute." (§1)

**Relevance to our paper.** This is behavioural evidence that RL can build something absent from the base model, on
a real Qwen3 model, with a sharp change point in training. That makes it a candidate "new" known answer
(pass@128 = 0 before RL, novel syntax), and the checkpoints before and after the jump would make a natural test of
whether our readings see a change at the transition. There is no internal analysis, and "could not execute" rests on
pass@128 alone, so it is not a mechanistic known answer. The model is Qwen3-4B, not the Qwen3-1.7B we decomposed. The
paper says small models (1.5B, 4B) suit the BASIC/EASY families, so a 1.7B run may be possible but would cost an RL
run of hundreds of steps.

**For the digest.**
- 2-known-answer-case: Qwen3-4B-Instruct RL on Manufactoria (pass@128 0 → ~100% after a grokking jump) is a behavioural "new" case with code released. A Qwen3-1.7B rerun would be needed for our decomposition.
- 4-contradicts: behavioural evidence that RL can go beyond re-weighting when the base model has zero coverage and the reward is staged.
