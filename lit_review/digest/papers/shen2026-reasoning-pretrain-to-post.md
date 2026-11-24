---
slug: shen2026-reasoning-pretrain-to-post
title: "Understanding Reasoning from Pretraining to Post-Training"
authors: Jingyan Shen et al.
year: 2026
id: 2607.16097
tags: [EH, DYNAMICS, BACKGROUND]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, training-dynamics]
read_from: "local PDF (arXiv v3, Nov 2026) pages 1-10: abstract, framework (Sec. 2), scaling (Sec. 3), 'mechanism analysis' (Sec. 4), math transfer (Sec. 5)"
digest_buckets: [4-supports, 4-contradicts]
importance: 4
---

# Understanding Reasoning from Pretraining to Post-Training (Shen 2026)

**What it does.** Builds the full pipeline in a controlled domain: Qwen3-architecture models from 5M to 1B parameters
pretrained on up to 54B tokens of Lichess games, SFT on synthetic search-tree reasoning traces, then GRPO on chess
puzzles with an exact verifier. Sweeps 36 pretraining-RL combinations to fit a joint scaling law, then asks "what does RL
actually do" by tracking each puzzle state's move distribution (SFT vs RL) and the reasoning trees. Repeats the scaling
part on a 1B OLMo-2 pretrained on 10B-200B maths tokens. Models and code are released.

**Key result.** Post-RL pass@1 is predicted by pretraining loss, and the RL slope grows roughly linearly with log
pretraining tokens; RL lifts pass@1 but barely pass@16. RL is not a uniform sharpening (a power-law fit of RL on SFT log
probs has only moderate R^2). On easy puzzles RL mostly amplifies correct moves already in the SFT top-3; on hard puzzles
it both promotes correct moves from the tail (probability < 0.05) and amplifies wrong modes.

**Quote.** "On easy puzzles it amplifies correct moves the SFT policy already preferred, while on hard puzzles it surfaces
correct moves that were nearly absent under SFT." (abstract)

**Relevance to our paper.** Its "mechanism analysis" is of the output distribution, not of internals, so it gives no
mechanistic evidence on reuse; but it is a clean statement that whether RL elicits or discovers depends on task
difficulty relative to the prior, which is the question our definition makes mechanistic. The released chess models
(pretrained, SFT and RL checkpoints, with fully controlled pretraining data) are a candidate testbed for our readings:
one could ask whether "tail discovery" states read as new variables or as rescaled ones. Not a known-answer case as it
stands (no internal evidence). Worth one sentence in the elicitation-versus-sharpening paragraph.

**For the digest.**
- 4-supports: on easy puzzles RL amplifies moves the prior already preferred (behavioural elicitation).
- 4-contradicts: on hard puzzles RL surfaces moves nearly absent before RL, and RL is not a temperature-style sharpening.
