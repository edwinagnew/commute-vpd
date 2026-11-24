---
slug: turner2022-reward-not-target
title: Reward is not the optimization target
authors: Turner (TurnTrout)
year: 2022
id: https://www.lesswrong.com/posts/pdaGN6pQyQarFHXF4/reward-is-not-the-optimization-target
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "LessWrong post: the intro, the 2026 edit and the main section (curl of the page, grep)"
digest_buckets: []
importance: 1
---

# Reward is not the optimization target (Turner 2022)

**What it does.** A conceptual alignment post. It argues that in deep RL, reward acts as a reinforcement schedule. Reward "chisels" the computations that led to it (credit assignment), and the trained policy need not come to value reward itself.

**Key result.** No experiments. The thesis is that RL agents will not, by default, intrinsically optimize their reward signal. An edit dated September 2026 concedes that specification gaming is "way worse than I expected". The author now says "reward might empirically become a primary optimization target", and keeps only the weaker claim that reward is not *definitionally* the target.

**Quote.** "reward chisels cognitive grooves into an agent" (intro)

**Relevance to our paper.** The "reinforce existing subroutines" picture of RL is informally close to the elicitation hypothesis for RL: reward strengthens computations that already produced it. This could support one sentence of motivation for why RL might reuse more. Our draft no longer claims that RL reuses more, though, and the post has no evidence about mechanisms. It is of little use beyond a background citation.

**For the digest.** (no bucket)
