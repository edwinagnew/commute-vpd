---
slug: shao2026-spurious-rewards
title: "Spurious Rewards: Rethinking Training Signals in RLVR"
authors: Shao et al.
year: 2026
id: 2506.10947
tags: [EH, KNOWN-ANSWER, SUPPORTS, CITE-FIX]
primary_tag: EH
in_our_bib: shao2025spurious
cited_in_draft: in refs.bib but not \cite'd (cited only in appC_rl_sft_qwen.tex, which main.tex does not input)
themes: [elicitation, interpreting-finetunes]
read_from: "local PDF arxiv_2506.10947.pdf (v2, Feb 2026) via pdf-text: abstract, §1-5 (pp. 1-8) incl. clipping-bias derivation and code-reasoning case study; appendices not read"
digest_buckets: [2-known-answer-case, 4-supports]
importance: 5
---

# Spurious Rewards: Rethinking Training Signals in RLVR (Shao 2026)

**What it does.** Runs GRPO on DeepScaleR maths prompts with rewards that carry less and less information about the
answer: ground truth, majority vote, format only (any `\boxed{}`), random (reward 1 with probability 0.5) and
deliberately incorrect labels. Models are Qwen2.5-Math-1.5B/7B, Qwen2.5-1.5B/7B, OLMo2-7B (base and SFT) and Llama3
base and instruct. It then explains the random-reward gain through GRPO's clipping term and traces one pre-existing
behaviour, "code reasoning" (writing Python in the chain of thought with no interpreter).

**Key result.** On Qwen2.5-Math-7B, MATH-500 rises 21.4 points with random rewards and 24.1 with incorrect labels,
against 29.1 with ground truth. Spurious rewards give little or negative change on OLMo2 and Llama. With clipping
removed, random rewards give no consistent gain. Clipping biases the expected gradient toward tokens the policy already
rates as likely. Code reasoning goes from 65% of Qwen2.5-Math-7B answers before RL to about 90% within 15 steps
(95.6% under random reward). Before RL, answers that use code are right 60.9% of the time and answers that do not are
right 35.0%. Prompting the model to use code gives a gain too (15.0 points on Qwen2.5-Math-7B, per the appendix cross-reference). Under ground-truth reward, code frequency
rises and then falls while accuracy keeps rising, which the authors read as "a different improvement mechanism".

**Quote.** "RLVR training with GRPO improves MATH-500 performance for Qwen2.5-Math-7B in absolute points by 21.4%
using randomly assigned rewards, nearly matching the 29.1% gained with ground truth rewards." (Abstract)

**Relevance to our paper.** This is the clearest realistic RL update that carries no task information by construction.
The random-reward run cannot teach the model which answers are right, and the gain is traced, behaviourally, to a
strategy the base model already used 65% of the time. So it is a strong behavioural candidate for a "reuse" known
answer on a real model, and the ground-truth run on the same model is a matched comparison that the authors think
learns something more. The evidence is behavioural only: string frequency of "python", accuracy split by strategy, and
prompt interventions. There are no activations or weights, so it is not yet a mechanistic known answer. For our Qwen
line it suggests a cheap control: random-reward GRPO as a "no information" RL update, to set beside the collapsed
run and the random weight updates. That works only if the base model has an amplifiable behaviour. The paper finds
none for OLMo and Llama, and Qwen3 is not tested. Citation: the draft cites this only in an appendix that is not
compiled. If the Qwen RL results stay, the related-work paragraph on post-training as selection should cite it next to
Yue et al.

**For the digest.**
- 2-known-answer-case: random- or incorrect-reward GRPO on Qwen2.5-Math-7B is a "no new information" RL update whose gain runs through a base behaviour (code reasoning, 65% → 90%). Behavioural evidence only, and it would need Qwen2.5-Math, not our Qwen3-1.7B.
- 4-supports: behavioural support that RLVR at open-source scale mostly re-weights base strategies. The gain is model-dependent and comes from the clipping bias toward high-prior tokens.
