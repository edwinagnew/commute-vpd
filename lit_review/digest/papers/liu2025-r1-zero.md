---
slug: liu2025-r1-zero
title: "Understanding R1-Zero-Like Training: A Critical Perspective"
authors: Liu et al.
year: 2025
id: 2503.20783
tags: [EH, SUPPORTS, CITE-FIX, BACKGROUND]
primary_tag: EH
in_our_bib: liu2025understanding
cited_in_draft: yes (only in appC_rl_sft_qwen.tex, which main.tex does not input)
themes: [elicitation, other]
read_from: "arXiv HTML: abstract, takeaways, sec. 2.1-2.3 (base models), sec. 3.1-3.4 (Dr. GRPO, template/question-set study, Llama pretraining), app. on Dr. GRPO ablations"
digest_buckets: [4-supports]
importance: 5
---

# Understanding R1-Zero-Like Training: A Critical Perspective (Liu 2025)

**What it does.** Examines the two ingredients of R1-Zero-style RL on base models. For base models (Qwen2.5,
Qwen2.5-Math, Llama, DeepSeek-V3-Base) it measures question-answering rate under templates, pass@8 exploration and
self-reflection keywords before any RL. For the RL algorithm it shows GRPO's per-response length and per-group std
normalisations bias optimisation, and proposes Dr. GRPO (drop both; divide the summed token loss by a constant). It
then RL-tunes Qwen2.5-Math-1.5B/7B and Llama-3.2-3B variants on MATH, GSM8K and other question sets.

**Key result.** All base models already solve maths before RL, and nearly all, DeepSeek-V3-Base included, already show
the "Aha moment"; in R1-Zero self-reflection is more frequent but not correlated with accuracy. Qwen2.5-Math gains about
60% from dropping the template (the authors suspect pre-training on concatenated question-answer text). With
mismatched templates "RL reconstructs" ability the template destroyed, and RL on GSM8K nearly doubles accuracy on harder
questions. Dr. GRPO stops incorrect answers growing longer. Llama gains little from RL unless first continually
pre-trained on maths. A 7B recipe reaches 43.3% on AIME 2024 in 27 h on 8 A100s.

**Quote.** "even a small and completely o.o.d. question set could induce the reasoning ability equally well, by
reinforcing useful reasoning behaviors instead of infusing new knowledge." (sec. 3.3)

**Relevance to our paper.** Two uses. (1) It is the recipe we cite for our Qwen RL arm, and the draft's description is
correct: mean-of-group baseline, no std scaling, summed token loss over a constant. But the citation sits only in an
appendix main.tex does not input, so the paper currently has no source for the RL recipe in the compiled text. (2) More
importantly, it is strong behavioural evidence for the elicitation hypothesis for RL on maths, our own Qwen setting:
the behaviours RL "creates" (self-reflection, maths solving) are already in the base model, RL gains depend on
pre-training (Llama needs maths pre-training first), and much of the apparent RL gain is undoing a template mismatch.
It should be cited in "post-training as selection" beside Yue et al. 2025 and Shao et al. (spurious rewards). It is
behavioural only, so it gives no mechanism-level known answer; it does suggest that a Qwen RL run with a mismatched
template is a case where the base mechanism is first disrupted and then restored, which our readings could be tested on.

**For the digest.**
- 4-supports: RL on maths mostly re-surfaces base-model behaviours (aha moments, maths ability present before RL; RL ceiling set by pre-training); cite in related work, not only in the uninput appendix.
