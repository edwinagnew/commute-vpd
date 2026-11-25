---
slug: chen2020-lottery-bert
title: The Lottery Ticket Hypothesis for Pre-trained BERT Networks
authors: Chen et al.
year: 2020
id: 2007.12223
tags: [EH, BACKGROUND]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: abstract only (arXiv abs page, checked 2026-11-25 by a clerk agent for P1.64); full text not read
digest_buckets: []
importance: 3
---

# The Lottery Ticket Hypothesis for Pre-trained BERT Networks (Chen 2020)

**What it does.** Asks whether pre-trained BERT, used as the initialisation, contains sparse subnetworks that fine-tune to full downstream accuracy (NeurIPS 2020).

**Key result.** Subnetworks at 40-90% sparsity are found at the pre-trained initialisation. Those found with the masked-language-model task transfer across downstream tasks; those found on a downstream task transfer less.

**Relevance to our paper.** This entry also stands for the lottery-ticket cluster that the user filed (lit-add 5, 11-25: training only selects what initialisation already holds). The cluster is Frankle & Carbin 2019 (1803.03635), Ramanujan et al. 2020 (1911.13299), Malach et al. 2020 (2002.00585), Zhao et al. 2020 masking (2004.12406) and Panigrahi et al. 2023 skill localization (2302.06600). Verified bibtex and one-line claims are in commute-data/agents/writeup-lead/lit-2026-11-25-lottery-bib.md, and section 5 of lit-2026-11-25-eh-alignment.md. Use for the intro: read at the level of weights, EH becomes trivially true, since everything is selected from the initialisation, so it has to be stated over mechanisms. These results show that a subnetwork present at initialisation is enough; they do not show that SGD uses it.
