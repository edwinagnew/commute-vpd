---
slug: skalse2022-reward-gaming
title: "Defining and Characterizing Reward Gaming"
authors: Joar Skalse et al.
year: 2022
id: 2209.13085
tags: [ALIGNMENT, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "arXiv HTML (NeurIPS 2022 version; arXiv title 'Defining and Characterizing Reward Hacking'): abstract, definitions section, main results on unhackability"
digest_buckets: []
importance: 1
---

# Defining and Characterizing Reward Gaming (Skalse 2022)

**What it does.** A theory paper. It defines a proxy reward as *unhackable* with respect to a true reward if raising
the proxy's expected return can never lower the true return, and asks when such proxies exist. It also treats
*simplifications*, proxies that collapse distinctions of the true reward.

**Key result.** Over all stochastic policies, two reward functions form an unhackable pair only if one of them is
constant. Over deterministic policies or finite policy sets, non-trivial unhackable pairs always exist, and the paper
gives necessary and sufficient conditions for simplifications. This sets up a tension between narrow task rewards and
alignment.

**Quote.** "We say that a proxy is unhackable if increasing the expected proxy return can never decrease the expected true return." (Sec. 1)

**Relevance to our paper.** None directly. Oh et al. cite it, beside Pan et al. 2024, for lucky rollouts earning credit
on final-answer rewards. Our paper studies what an update does to mechanisms, not reward misspecification. It would
matter only if we argued that a collapsed RL run is a reward-hacking run, and the draft does not.

**For the digest.**
