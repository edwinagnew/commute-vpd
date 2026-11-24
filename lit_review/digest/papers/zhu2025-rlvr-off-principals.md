---
slug: zhu2025-rlvr-off-principals
title: "The Path Not Taken: RLVR Provably Learns Off the Principals"
authors: Zhu et al.
year: 2025
id: 2511.08567
tags: [FT-INTERP, DIFFING, DYNAMICS, EH, SUPPORTS, CITE-FIX]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, training-dynamics, elicitation]
read_from: "arXiv HTML: abstract, section 1, section 4 (spectral geometry, principal weights, rotation intervention, LoRA) and conclusion; section 3 theory and section 2 skimmed; appendices not read"
digest_buckets: [4-supports, 1-missing-method]
importance: 6
---

# The Path Not Taken: RLVR Provably Learns Off the Principals (Zhu 2025)

**What it does.** Asks why RL with verifiable rewards (RLVR) changes few parameters. It compares base and post-trained
weights of public checkpoints: an SFT then RLVR pipeline on Qwen3-8B-Base, a long RL run on DeepSeek-R1-Distill-Qwen-1.5B,
and SFT and RL separately on Qwen3-14B-Base matched on in-domain maths. The measures are principal angles between
top-k singular subspaces of W0 and W+, normalised singular-value drift, and overlap of the update mask with
"principal weights" (largest entries of the rank-k reconstruction). A "three-gate" theory (KL anchor, model geometry,
bf16 precision) explains the pattern, and a causal test scrambles a layer's geometry by function-preserving rotations
and head permutations before RL.

**Key result.** RLVR leaves the top-k spectrum nearly unchanged, with small principal-subspace rotation. SFT gives
"substantially larger rotations and pronounced drifts" on the same metrics. RL update masks overlap principal weights
below chance and low-magnitude weights above chance. After a rotation or permutation of a layer, run-to-run overlap of
update locations falls to chance in that layer only. Sparse fine-tuning restricted to principal weights gives the
weakest RL trajectory.

**Quote.** "RLVR learns off-principal directions in weight space, achieving gains via minimal spectral drift, reduced
principal-subspace rotation, and off-principal update alignment. In contrast, SFT targets principal weights, distorts
the spectrum" (Abstract)

**Relevance to our paper.** It is weight-level evidence on the rescale-or-rotate question, and it splits by training
regime the way our numbers do. SFT rotates the base model's dominant singular directions, which agrees with our finding
that trained fine-tunes turn writes rather than scale them. RL preserves them, which agrees with our RL repeat-task
updates reading higher ρ (0.05 to 0.22) than SFT. The draft no longer claims RL reuses more; this paper is the
literature that would back a cautious version of that claim, and the draft should cite it in the RL/SFT discussion.
Its measures (principal angles, spectral drift) are a cheap, featurization-free weight diff, and we should report them
next to effective rank as a baseline: do they separate our built pairs? Caveat: preserving the top-k spectrum says
nothing about whether the needed mechanisms are kept, since the RL change sits off those directions by construction.

**For the digest.**
- 4-supports: independent evidence that SFT rotates base weight geometry while RL preserves it, matching our "turn not scale" finding and the higher RL ρ.
- 1-missing-method: principal-angle and spectral-drift weight diffs are a cheap baseline we could score on our built pairs at no cost.
