---
slug: hoogland2023-developmental-interp
title: Towards Developmental Interpretability
authors: Hoogland
year: 2023
id: https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability
tags: [DYNAMICS, BACKGROUND]
primary_tag: DYNAMICS
in_our_bib: no
cited_in_draft: no
themes: [training-dynamics]
read_from: "LessWrong post full (via the LessWrong GraphQL API)"
digest_buckets: []
importance: 2
---

# Towards Developmental Interpretability (Hoogland 2023)

**What it does.** A research-agenda post from the singular learning theory (SLT) community. It proposes studying how
internal structure forms during training, organised around phase transitions as SLT defines them, with SLT
observables (learning coefficient, singular fluctuation) to detect transitions and localise them in the weights.
No experiments; a six-month plan of sanity checks on toy models of superposition and induction heads.

**Key result.** None empirical. It lists four ways the agenda could fail (transitions too infrequent, too frequent,
too large, too isolated) and cites low-dimensional-subspace training and Panigrahi et al.'s fine-tuning skill
localisation as evidence for locality of changes.

**Quote.** "advance the science of detecting when structural changes happen during training" (Relevance to Alignment)

**Relevance to our paper.** Background for the motivation that a decomposition should stay readable along training,
and for the idea of reading what changed between checkpoints; it offers no definition of reuse and no method we could
compare against. One line in related work at most, if we discuss training-dynamics approaches to the same question.

**For the digest.**
- (no bucket)
