# The Invisible Leash? Why RLVR May or May Not Escape Its Origin

## Metadata
- **Authors**: Fang Wu, Weihao Xuan, Danny Xing, Ximing Lu, Mingjie Liu, Yi Dong, Zaid Harchaoui, Yejin Choi
- **Year**: 2025 (text read is arXiv v5, dated 14 Oct 2026)
- **Source**: arXiv:2507.14843 (cs.LG)
- **DOI/URL**: arXiv:2507.14843

## Summary
The paper introduces "empirical support dynamics". For each question it records whether a base (reference) model and an RLVR model each produce at least one verified correct sample under a fixed sampling budget. Questions are labelled preserved, expanded (only the RLVR model succeeds), shrunk (only the base model succeeds) or out of support. Across ProRL, Nemotron, Skywork, Phi4-Reason and other released RLVR models, the authors report that shrinkage exceeds expansion even while pass@1 rises. This is purely behavioural evidence for "RLVR sharpens rather than extends".

## Key Findings
All findings below are the authors' claims, and all are behavioural.
- Across the model-level aggregates (1B to 14B), more questions lose a verified success than gain one (NSCR below 0 everywhere). For ProRL-1.5B-v2 the counts are P=2388, E=48, S=175, O=793 (Tab. 1, page 4). New questions make up at most about 2% of RLVR successes for the 1.5B to 14B math and aggregate comparisons (NDR at most 0.02 overall). The exception is non-math, where NDR reaches 0.03 to 0.04 for ProRL.
- Pass@1 gains coexist with a pass@k deficit at large k. Example: on hard graph colouring ProRL gets 96% at pass@1 versus 12% for the base model, and both reach 100% at k=2048. On leg_counting the base model beats ProRL at k=1024 (40% vs 30%) (page 6).
- Same-base comparison on Qwen2.5-Math-7B with DeepMath-103K (Tab. 3, page 7, k=256): SFT has nonnegative NSCR on all six benchmarks. DAPO has net shrinkage on four of them. This is a single SFT-vs-RL contrast, and it goes against the idea that SFT is the "new knowledge" route only in the trivial sense that SFT broadens coverage.
- Training dynamics (Tab. 4, page 7, DAPO on DeepSeek-R1-Distill-Qwen-1.5B, steps 0 to 300, k=256): SRR stays high, but net support turns negative in later steps. On Olympiad, gains fall from 13 (step 30) to 5 (step 300) while losses rise from 22 to 61. On AIME25 and Minerva, intermediate checkpoints show net expansion that disappears by step 300.
- Entropy (page 8): answer-level entropy falls in all model-benchmark pairs, while token-level entropy rises for ProRL (0.44 to 0.52) and DAPO-32B (0.18 to 0.29).
- Trace perplexity (Tab. 2, page 6): shrunk and doubly-failed questions get higher perplexity for DeepSeek-R1 and Sonnet 4 traces after RLVR, and expanded questions get lower perplexity. This is a likelihood-based, not an internal, measure. The authors interpret it as linking newly gained successes to higher likelihood of the supplied traces.

## Methodology
Question-level pass@k with very large budgets (k from 256 up to 16384, depending on the comparison). It uses the same decoding, the same verifier and paired questions, and derives the metrics SRR, NDR, SDS and NSCR from the four counts. Models are mostly released checkpoints: ProRL (GRPO-based, DeepSeek-R1-Distill-Qwen-1.5B base), Nemotron, Skywork, Phi4-Reason, OLMo-2-1B and a VLM. The controlled arms are Qwen2.5-Math-7B SFT vs DAPO and DAPO on DeepSeek 1.5B across checkpoints. Benchmarks are math (MATH500, AIME, etc.) and non-math (Reasoning Gym, LiveCodeBench, SimpleQA, etc.). There is no inspection of activations, weights or circuits.

## Relevance to Proposal
**Score: 3**

1. Evidence type: entirely behavioural (sampled outputs, verifier outcomes, token and answer entropies, trace perplexity). The paper does not look inside the model.
2. Could it hold while our verdict goes the other way: yes, in both directions. This is my inference. Its "preservation" is the behavioural notion ("the right answer was already in the base model's samples") that the project explicitly rejects. A model could keep or lose pass@k support while the fine-tuned behaviour runs through a new, non-readable mechanism (our verdict "not through C" or "unreadable"). Conversely, a fine-tuned behaviour could go through an existing readout (reuses C) while pass@k support shrinks or expands. Also, the paper's dominant "preserved" count is mostly questions the base model already solves, so it says little about whether δW is carried by existing subcomponents.
3. Usable items. (a) The paired-question SRR/NDR/NSCR metrics could be a behavioural comparator for our GSM8K SFT and GRPO arms, to show how the distribution-sharpening reading and the Jain Def 1 reading diverge. It would need k large enough to see tail support. Qwen3-1.7B on GSM8K is likely near ceiling at large k, so O would be tiny, which the authors flag as limiting NDR and SDS. (b) The finding that SFT broadens support while DAPO narrows it on the same base (Qwen2.5-Math-7B) is a behavioural hint that SFT and GRPO arms might differ in novelty. This is only a hint, not mechanistic. (c) The step-wise DAPO table suggests behavioural support changes are visible from step 30, and gains can be transient. That is relevant to our choice of early-step δW, but it is a different model family (R1-Distill-Qwen-1.5B), not Qwen3. No controlled task with known-by-construction novelty is provided. Reasoning Gym tasks are used as behavioural benchmarks, not as constructed-novelty anchors. The connection to our mechanistic question is thin.

## Limitations
- Authors' own: finite sampling makes support membership noisy for rare successes. Verifiers may accept correct finals with flawed reasoning. Comparisons are of complete training recipes, so causes of expansion or shrinkage are not identified (Limitations, page 9).
- Inference: mostly released checkpoints with different recipes, and no matched interventions except the SFT vs DAPO and DAPO-checkpoint tables. Single seed visible in the extracted text. Support differences of a few questions (for example 3 vs 4 in Tab. 1) are within sampling noise. No mechanistic claims, so "invisible leash" is a behavioural statement about reachability.

## Key Quotes
> "Across the model-level benchmark aggregates,the shrinkage of observed support outweighs its expansion: more questions lose a verified success than gain one at the stated sampling budgets." (Abstract)

> "Our comparisons measure the outcomes of complete training recipes; identifying which components cause expansion or shrinkage requires matched interventions." (Limitations, page 9)

> "SFT maintains or increases total coverage on all six benchmarks (nonnegative NSCR), whereas DAPO produces net shrinkage on four." (Section 3.2, page 6)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).

## Not read
Lines 900 onward of the extracted text (the remaining references and the appendices A to G: per-benchmark counts, the Appx. F derivations of reward reweighting and the zero-success confidence bound, and Appx. G pass@k tables). I read the main text, Tab. 1 to 4 as extracted, and the Limitations section. Tables and figures were extracted from PDF. Values quoted come from the extracted text and I did not verify them against the original figures, so treat the counts as approximate.
