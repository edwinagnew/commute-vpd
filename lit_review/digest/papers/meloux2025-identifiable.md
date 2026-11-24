---
slug: meloux2025-identifiable
title: "Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?"
authors: Méloux et al.
year: 2025
id: 2502.20914
tags: [CAUSAL, DEFINITION, BACKGROUND]
primary_tag: CAUSAL
in_our_bib: meloux2025identifiable
cited_in_draft: yes
themes: [causal-abstraction, reuse-definitions]
read_from: "arXiv HTML full text (abstract, sec. 2-5, appendix D)"
digest_buckets: [3-definition]
importance: 4
---

# Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? (Méloux 2025)

**What it does.** Asks whether, for a fixed behaviour and fixed MI criteria, the mechanistic explanation is unique. It
trains tiny MLPs (layer sizes (2, k, k, n)) on Boolean gates and exhaustively enumerates explanations under two
strategies: "where-then-what" (find circuits with zero circuit error, then interpret each neuron as a logic gate) and
"what-then-where" (enumerate candidate Boolean algorithms, then search neuron subsets whose interchange intervention
accuracy, IIA, is 1). A larger MNIST 0-vs-1 MLP gives a lower bound for the circuit case.

**Key result.** Non-identifiability at every stage. On one XOR network: 85 perfect circuits with about 536
interpretations each, and 159 perfect minimal mappings covering both of two tested algorithms. Across seeds, median
counts grow with width from 38 to 910,000 (circuit-first) and 8 to 3,700 (algorithm-first); under 2% of networks have
exactly one minimal mapping and none has exactly one circuit interpretation. The MNIST MLP's last block alone has 3,209
valid circuits. Random (untrained) networks that happen to implement a gate also admit many abstractions.

**Quote.** "Even when employing the strict causal requirements of MI, we find that many incompatible explanations can
coexist." (sec. 5)

**Relevance to our paper.** Our definition is explicitly relative to a featurization τ fixed from M, and the paper is
the cleanest citation that a mechanism is not identified by behaviour or even by IIA = 1: several alignments with
different variables can all be perfect. That is the formal reason our verdict must be stated per featurization (and why
neurons read ≥ 0.999 reuse while VPD reads low). It also bears on our "needed" clause (Def. 2): needed sets found by
interchange can differ across equally valid alignments, so K is not unique either. The authors point to full causal
abstraction (all low-level states accounted for, exclusions justified) as a remedy; our featurization keeps a remainder
term, which is in that spirit. The draft's one-line use ("circuit and algorithm explanations of one network need not be
unique") is accurate. Toy MLPs only; no fine-tuning, no model comparison.

**For the digest.**
- 3-definition: cite where we say reuse is relative to τ: identical behaviour and perfect IIA do not pin down the mechanism, so a reuse verdict needs a fixed featurization.
