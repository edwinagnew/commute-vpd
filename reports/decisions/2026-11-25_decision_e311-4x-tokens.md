# E311: rerun the crosscoder on Fig 4's states at four times the training? (2026-11-25)

**What was measured.** E311 trained Minder et al.'s crosscoder (a model-diffing tool that splits the change between the original model and a fine-tune into shared and fine-tune-specific parts) on Fig 4's 48 fine-tuned states, at the source's 200M tokens. We expected it to find some fine-tune-specific part, most on the tasks built new. It did not: on every state but one (seed-0 Copy key (96 keys), 0.101) that part is at most 0.005. A planted new direction that rides on an existing feature is still merged at 200M tokens and splits off from 400M (0.722 at 400M, 0.804 at 800M). A planted direction placed where no feature fires is found at 200M.

**Question.** Should the 48 states be rerun at 800M tokens, to see whether "almost no fine-tune-specific part" is only a training-length artefact?

**Options.**
- (a) Rerun all at 4x. One side took about 48 min on an H100 at 200M, so about 3.2 h at 800M. 52 sides on 26 H100 pods is about 6.5 h wall; 52 x 3.2 h x $4.41 = about $730. Tells us whether any state ever separates; over the $300 session cap.
- (b) Keep the 200M readings and state the caveat in the figure caption. $0. Leaves the caveat untested.
- (c) Rerun the 8 tasks built new, seed 0, at 800M: 8 x 3.2 h x $4.41 = about $113, about 3.5 h wall on 8 pods. Tests the caveat where a new direction should exist, at a seventh of the cost.

**Recommendation: (c)**, before the 2026-11-29 draft.

Report: `reports/analyses/2026-11-25_analysis_p2-83-crosscoder.md`; results `experiments/crosscoder_fig4/results/`.
