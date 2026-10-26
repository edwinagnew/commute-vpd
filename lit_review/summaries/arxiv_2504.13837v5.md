# Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

## Metadata
- **Authors**: Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang (LeapLab, Tsinghua University; Shanghai Jiao Tong University)
- **Year**: 2025 (arXiv 2504.13837, v5 dated November 25, 2025)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2504.13837 (project page: https://limit-of-RLVR.github.io)

## Summary
The paper asks whether reinforcement learning with verifiable rewards (RLVR) gives LLMs reasoning abilities beyond those of their base model, or only makes the base model's existing successful reasoning paths more likely. Using pass@k at large k as a measure of the reasoning boundary, across Qwen2.5, LLaMA-3.1, and several RLVR models in math, code, and visual reasoning, the authors find that RLVR raises pass@1 but lowers pass@k at large k, so the base model covers more solvable problems. Distillation from a stronger teacher, by contrast, does expand coverage.

## Key Findings
- RLVR models beat their base models at small k (e.g., k=1), but base models match or exceed them as k grows (tens to hundreds to 1024). This holds across the math benchmarks (GSM8K, MATH500, Minerva, Olympiad, AIME24, AMC23), coding (LiveCodeBench, HumanEval+, MBPP+), and visual math (MathVista, MathVision). On Minerva with a 32B model, the base model solves about 9% more problems at k=128.
- Solvable-problem sets are nearly nested. On AIME24 (k=1024) and MATH500 (k=128), problems solved by the RL model are almost all also solved by the base model (Table 2). On AIME24, no problem was solved only by the RL model.
- Perplexity under the base model is low for RL-generated responses (Figure 6, two AIME24 problems). The authors read this as RLVR sharpening the base model's prior rather than moving outside it. The perplexity of RL responses falls as RL training proceeds (Appendix C.4).
- Over training, pass@1 rises (26.1 to 42.5 on the training set) while pass@256 falls, so the reasoning boundary narrows as RLVR proceeds (Section 4.4).
- Six RL algorithms (PPO, GRPO, Reinforce++, RLOO, ReMax, DAPO) perform similarly. The sampling-efficiency gap stays above 40 points for all of them, so none approaches the base model's upper bound (Section 4.3).
- Distilled DeepSeek-R1-Distill-Qwen-7B has a pass@k curve consistently above its base model (Qwen2.5-Math-7B), while the RL-trained counterpart does not (Figure 7). The authors conclude that distillation injects new patterns from the teacher.
- Lowering the RL model's entropy does not explain the narrowing. Raising the temperature to match base-model entropy gives only a small improvement, and the RL model still trails the base model (Section 4.5).

## Methodology
- Metric: the unbiased pass@k estimator, with k up to 1024, applied to tasks with verifiable rewards. Base models are evaluated zero-shot (no few-shot prompts). Sampling uses temperature 0.6, top-p 0.95, and up to 16,384 tokens.
- Models: public RLVR checkpoints (SimpleRLZoo, Oat-Zero-7B, DAPO-32B, Code-R1-Zero-Qwen2.5-7B, DeepCoder-14B, EasyR1 Qwen2.5-VL-7B), plus Magistral-Medium vs. Mistral-Medium-3 (near-frontier, API only, Section 4.6).
- Validity checks: manual inspection of CoTs for the hardest problems (accuracy below 5%) to separate correct reasoning from lucky guesses (e.g., 24 of 25 hardest GSM8K questions for the base model had at least one correct CoT).
- Controlled RL-algorithm study: VeRL re-implementations, a constant learning rate of 1e-6, 8 rollouts per prompt, Omni-MATH-Rule train/test split, with MATH500 as the out-of-domain benchmark.
- Mechanistic analyses: accuracy histograms, solvable-set overlap, perplexity distributions, and ablations on rollout count n, KL coefficient, and temperature.

## Relevance to Proposal
**Score: 6**

The paper bears on the proposal's core dichotomy: does post-training strengthen existing mechanisms or create new ones? Its evidence is behavioural (pass@k and perplexity), not weight-level, so it does not test whether a decomposition or a commuted update tracks the change. It does give a concrete contrast case: RLVR (sharpening within the base prior) versus distillation (adding patterns from a teacher). That contrast could define two kinds of update for a faithfulness test of a commute map. The paper is only indirectly relevant to the decomposition-specific strands.

## Limitations
- Authors' own: many of the most capable models and training pipelines are proprietary, and RL methods are changing quickly, so the conclusions may not hold for future techniques (Section 7).
- Authors' own: pass@k on math can credit lucky guesses; the manual checks cover only small subsets (for example, 6 AIME24 questions at accuracy below 5%).
- Authors' own: the base-model reference uses zero-shot prompts, which the authors acknowledge yields unformatted outputs that need many samples to rescue.
- Apparent: the metric is behavioural. Nothing in the paper examines the weights, the update, or any internal representation, so it cannot say which parameters or components carry the RL change.
- Apparent: the coverage analysis (Table 2) is on AIME24 (which I infer has 30 problems from the percentages) and MATH500, so the "no problems only RL solves" claim rests on few problems.
- Apparent: the perplexity analysis uses two AIME24 problems in the main text.
- Apparent: the boundary is measured only up to k=1024, and the authors leave open whether more RL compute could close the gap (Section 4.4 and Section 4.6).
- Apparent: the scope of the claim is mostly math and code with checkable answers, so it may not transfer to open-ended post-training.
- Not read in full: the appendix (about 450 lines of references and appendix material, including prompts and per-benchmark tables) was not read in this pass. Figure-level details beyond the main text are therefore not checked.

## Key Quotes
> "we surprisingly find that current training _rarely_ elicit fundamentally new reasoning patterns" (Abstract, p. 1)

> "all the reasoning paths exploited by the RLVR model are already present in the sampling distribution of the base model" (Section 4.1, p. 9)

> "distillation introduces new reasoning patterns learned from a stronger teacher model" (Section 4.2, p. 9)

> "as RLVR training progresses, pass@256 progressively decreases, indicating a reduced reasoning boundary" (Section 4.4, p. 10)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv preprint and no comment thread was provided.
