# Curriculum Reinforcement Learning Can Incentivize Reasoning Capacity in LLMs Beyond the Base Model

## Metadata
- **Authors**: Pengxiang Cai, Tianchen Fang, Xiaohan Li, Qingyuan Zeng, Guocong Li, Jintai Chen (corresponding)
- **Year**: 2026 (arXiv preprint, v1; the exact posting date is not given in the text)
- **Source**: arXiv preprint (Preprint marking; no venue stated)
- **DOI/URL**: arXiv:2606.22317 (https://arxiv.org/abs/2606.22317)

## Summary
The paper proposes "boundary-aware Curriculum RL" for LLM reasoning. Each round uses large-k pass@k sampling (k up to 256) to sort training problems into difficulty bins, adds a small SFT bridge of teacher-written reasoning traces on problems just inside or just outside the model's observed boundary, and then runs GRPO on a chosen difficulty range. The authors report that this raises pass@256 over the base model (+9.8 pp averaged) where vanilla RLVR does not (-0.5 pp), and argue that it moves the reasoning-capacity boundary rather than only sharpening sampling.

## Key Findings
- Vanilla RLVR improves pass@1 but leaves large-k pass@k roughly flat on average (mean pass@256 change -0.5 pp vs base) and sometimes lowers it (e.g., Qwen2.5-7B AIME 2024: 40.0 to 26.7; LLaMA-3.1-8B AIME 2024: 53.3 to 36.7). (Sections 4.2, Table 1, Figure 3C)
- The proposed curriculum improves mean pass@256 by +9.8 pp over base across five model settings (Qwen2.5-7B, Qwen2.5-7B-Instruct, Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B, Llama-3.1-8B-Instruct) on AIME 2024, AIME 2025 and MATH500. This is an author-reported result. (Section 4.2, Figure 3)
- Problem-level counts: unsolved-at-256 problems fall from 538 to 350 in total across the five settings, and 226 of the 538 base-unsolved problems become solvable. (Table 2)
- Training-pool diagnostics: "Beyond" (unsolved at 256) training examples drop (e.g., Qwen2.5-7B-Instruct 113 to 45) and reachable examples rise correspondingly (Table 7, Table 9). Transition tables (Table 8, Figure 9) are offered as evidence that individual problems change status.
- The mechanistic argument (Appendix C) is that all-failed GRPO groups have zero group-normalized advantage, so hard problems give no reward-driven signal. Under an independence assumption, a per-rollout success probability p gives a mixed-reward group with probability q_G(p) > 0 for any p in (0,1). The paper states this as an interpretation, not a guarantee.
- The method uses only a small teacher intervention: 10 Frontier and 10 Beyond problems per round (Table 4 shows totals of 20 to 55 seed problems per round). The teacher is not named in the text reviewed.

## Methodology
- **Stage I (boundary exploration)**: 256 samples per problem, three decoding seeds. The first-success index is averaged across seeds and binned into Easy (k=1), Medium (2-8), Hard (8-32), Frontier (32-256) and Beyond (unsolved at 256).
- **Stage II (teacher bridge)**: about 20 problems per round are chosen by closeness to the Frontier/Beyond thresholds, with family balance (Algorithm 1). A structured-reasoning teacher prompt (backbone, midpoint self-reflection, alternative path, coherence check) yields SFT traces.
- **Stage III (curriculum + RL)**: one of five difficulty ranges (Easy-to-Beyond through Beyond-only) is chosen per model and round (Table 3). Standard GRPO is then run on that range. Three rounds in total.
- **Data and evaluation**: a subset of DAPO-Math-17K for training; AIME 2024, AIME 2025 and MATH500 for evaluation; unbiased pass@k over k in {1,...,256}.
- No ablations are reported. There is no RL-only control on the same problems, no teacher-SFT-only control, no random-selection control, and no comparison against a matched-compute vanilla RLVR run beyond the one reported.

## Relevance to Proposal
**Score: 4**

The paper's central question is the proposal's core dichotomy, whether post-training makes new capacity or only reweights existing trajectories. Its pass@k-based operationalisation of "the boundary moved" (problems unsolved at k=256 becoming solvable) could serve as a behavioural label for the proposal's "creates new vs strengthens existing" test, and it is relevant to strand 4 (mechanistic accounts of fine-tuning). It does no weight-space or interpretability analysis, however: there is no decomposition, no circuit or component analysis, and no statement of whether the updates are low-rank or full-rank, so it does not bear directly on whether a small update can be commuted past a parameter decomposition.

## Limitations
- **Small evaluation sets**: AIME 2024 and AIME 2025 each have 30 problems, so one problem is 3.3 pp. Many of the headline gains are one to three problems per cell. Results come from a single training run per setting, with no reported variance or confidence intervals.
- **Internal numerical inconsistencies**: several pass@1 values for the Curriculum RL model differ between Table 1 and the Table 6 curve values (e.g., Qwen2.5-7B MATH500: 63.9 vs 46.1; Qwen2.5-7B-Instruct AIME 2025: 13.3 vs 6.7; Qwen3-8B MATH500: 34.4 vs 28.4; LLaMA-3.1-8B-Instruct AIME 2024: 6.7 vs 0.0). Figure 3A averages would change with the corrected values, so the pass@1 claims need to be rechecked against the raw data.
- **No ablations isolating the mechanism**: the gains could come from the teacher SFT bridge, from the choice of training range, from the extra training, or from selection on the training-set difficulty, rather than from converting zero-advantage groups. The paper's own Appendix C argument does not separate these.
- **Confounded comparison**: the curriculum arm includes teacher-generated SFT data and per-model hand-chosen curriculum ranges (Table 3). The vanilla RLVR arm does not receive the same data or selection, so the +10.3 pp gap over vanilla is not a clean comparison of the boundary-aware principle.
- **Training details missing from the text reviewed**: the teacher model, learning rates, number of RL steps, batch sizes, LoRA vs full fine-tuning, and the SFT data size are not stated in the main text or appendices reviewed. Reproduction would require the code.
- **Pass@256 as a proxy**: the paper acknowledges pass@256 is an empirical proxy for the capacity boundary. The boundary is defined with the same sampler and budget used for evaluation, so "moved boundary" is partly a claim about a fixed sampler at one k.
- **Theory is idealised**: Proposition 3 and Corollary 1 assume independent rollouts, which the authors note does not hold for dependent decoding samples.
- **Scope**: the authors say optimal curriculum selection is out of scope. No interpretability or parameter-level analysis is included.

## Key Quotes
> "This observation suggests that extremely difficult examples may receive little benefit from standard RLVR: if all sampled rollouts fail, the example provides no advantage signal for policy optimization." (Section 1)

> "This is often what we observe for Vanilla RLVR: its large-k gains are limited, and in some cases pass@256 decreases relative to the base model." (Section 4.2)

> "Across all five model settings, this count decreases after Curriculum RL, falling from 538 to 350 in total." (Section 4.2, Table 2)

> "Our approach first uses pass@k sampling to locate the current reasoning capacity boundary, then applies targeted teacher guidance to examples near or beyond that boundary, and finally uses RL to consolidate the newly introduced reasoning patterns." (Abstract)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
