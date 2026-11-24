---
slug: orthonormal2024-run-evals-base
title: "Run evals on base models too!"
authors: orthonormal
year: 2024
id: https://www.lesswrong.com/posts/dgFC394qZHgj2cWAg/run-evals-on-base-models-too
tags: [EH, ALIGNMENT, BACKGROUND]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: LessWrong / Alignment Forum post full and its 6 comments
digest_buckets: []
importance: 1
---

# Run evals on base models too! (orthonormal 2024)

**What it does.** A one-minute post arguing that capability evaluations should also be run on base models, because
RLHF can train a model to stop showing a capability without removing it. Its analogy is a chess model rewarded for
putting up a good fight but punished for winning: its Elo drops while its chess knowledge stays.

**Key result.** No results. In the comments Ryan Greenblatt replies that base models do worse on what labs care about
(software, agency, maths), citing METR's finding that GPT-4-base did poorly in their scaffold, and that a
"helpful-only" fine-tune is the better precaution; Vladimir Nesov and Jozdien add that base capabilities need a
light fine-tune or a completion-format prompt to show.

**Quote.** "RL*F will train a base model to stop displaying capabilities, but this isn't a guarantee that it trains the model out of having the capabilities." (post)

**Relevance to our paper.** An informal statement of the hidden-capability side of elicitation: behaviour after
post-training under-reports what the network computes, so behaviour alone cannot settle reuse. That is the same
point that motivates our mechanistic definition, made without evidence. Could be one of the "informal mechanistic EH"
citations the draft asks for, but a weak one.

**For the digest.** (none)
