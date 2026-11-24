---
slug: shao2024-deepseekmath
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
authors: Zhihong Shao et al.
year: 2024
id: 2402.03300
tags: [EH, SUPPORTS, CITE-FIX, BACKGROUND]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "arXiv HTML: abstract, sec. 4 (GRPO, RL training setup and results), sec. 5.2 (RL insights incl. 5.2.2 Why RL Works?), sec. 6"
digest_buckets: [4-supports, 5-cite-fix]
importance: 5
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao 2024)

**What it does.** Continues pre-training DeepSeek-Coder-Base-v1.5 7B on 120B math tokens mined from Common Crawl, then
instruction-tunes it and runs RL. It introduces GRPO, a PPO variant that drops the critic and takes the baseline from
the scores of a group of sampled outputs. RL runs on DeepSeekMath-Instruct 7B with ~144K GSM8K/MATH chain-of-thought
questions (lr 1e-6, KL 0.04, 64 samples per question). Section 5.2 casts SFT, rejection sampling, DPO, PPO and GRPO as one
gradient paradigm and asks why RL helps.

**Key result.** GRPO lifts the instruct model in domain (GSM8K 82.9% to 88.2%, MATH 46.8% to 51.7%) and out of domain.
In 5.2.2 (Figure 7) RL raises Maj@K but not Pass@K on GSM8K and MATH. The authors read this as RL making the output
distribution more robust, not adding capability, and say that RL "may merely stabilize the distribution of LLMs rather
than improve their fundamental capabilities".

**Quote.** "RL enhances the model's overall performance by rendering the output distribution more robust, in other words, it seems that the improvement is attributed to boosting the correct response from TopK rather than the enhancement of fundamental capabilities." (Sec. 5.2.2)

**Relevance to our paper.** This is the source of GRPO, which our Qwen RL runs use, and an early statement, from the
lab whose R1 the draft cites on the other side, of the "RL only sharpens" reading. It is the Pass@K evidence that Yue
et al. 2025 later took up. The evidence is behavioural only (Pass@K against Maj@K), so it is exactly the kind of evidence
our introduction calls insufficient. It is no known-answer case and no method. The draft's intro sets "post-training
cannot teach anything new" (LIMA, URIAL, Jain, Yue) against "capabilities come from scaling RL" (DeepSeek-R1) and asks
for more citations. DeepSeekMath belongs on the first side, and it shows that the same group held both views. If our RL
appendix uses GRPO, it should cite this paper for it.

**For the digest.**
- 4-supports: an early behavioural claim (RL raises Maj@K, not Pass@K) that RL re-weights what the SFT model already samples; it fits the elicitation side of the intro, and it is behavioural only.
