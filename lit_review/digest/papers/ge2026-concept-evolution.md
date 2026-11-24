---
slug: ge2026-concept-evolution
title: Evolution of Concepts in Language Model Pre-training
authors: Ge et al. (Shu, Wu, Zhou, He, Qiu)
year: 2026
id: https://proceedings.iclr.cc/paper_files/paper/2026/hash/45673dbf3f331fbd911b0689872de396-Abstract-Conference.html
tags: [DYNAMICS, DIFFING, BACKGROUND]
primary_tag: DYNAMICS
in_our_bib: no
cited_in_draft: no
themes: [training-dynamics, model-diffing]
read_from: "local PDF text (ICLR 2026): abstract, Sec. 1, 3 (cross-snapshot crosscoder, setup), 4 (decoder norms, rotation), 5-6 summaries, App. robustness paragraphs"
digest_buckets: []
importance: 3
---

# Evolution of Concepts in Language Model Pre-training (Ge 2026)

**What it does.** Trains crosscoders jointly on activations from 32 of Pythia's pre-training snapshots (steps 0 to
143,000; Pythia-160M and 6.9B, one layer each) to align features across checkpoints. Each feature's per-snapshot
decoder norm is used as a proxy for its strength over training. Attribution-based circuit tracing links feature emergence
to downstream metrics.

**Key result.** There are two kinds of features: ones present at initialization (token-specific features that peak
around step 128 and then decay) and emergent features that begin forming around step 1,000. Simple ones (previous-token)
emerge early, at roughly 1,000-5,000 steps, and induction and context-sensitive ones later. Most features "undergo
drastic directional shifts around step 1,000" and then rotate more slowly, keeping notable cosine similarity to their
post-1,000 directions. The authors frame this as a "statistical learning phase" followed by a "feature learning phase".
The pattern repeats across seeds and for GPT-2.

**Quote.** "most features undergo drastic directional shifts around step 1,000, rendering pre- and post-step 1,000
directions nearly orthogonal" (Sec. 4)

**Relevance to our paper.** This is pre-training dynamics, not post-training, so it bears only indirectly. It shows a
crosscoder across checkpoints tracking features that persist and rotate: the same tool family as our crosscoder
baseline, used here at a much larger gap between snapshots than a fine-tune. That fits our finding that the crosscoder
cannot see changes at fine-tune size. Its observation that features keep their identity while rotating slowly is an
activation-space analogue of our "turned rather than scaled" finding, and is a weak argument for a rotation-tolerant
kept clause. Background citation at most.

**For the digest.**
