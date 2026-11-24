---
slug: chhabra2025-neuroplasticity-ioi
title: "Neuroplasticity and Corruption in Model Mechanisms: A Case Study of Indirect Object Identification"
authors: Vishnu Kabir Chhabra et al. (Zhu, Khalili)
year: 2025
id: 2503.01896
tags: [KNOWN-ANSWER, FT-INTERP, CAUSAL, DIFFING, SUPPORTS, CITE-FIX]
primary_tag: KNOWN-ANSWER
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, elicitation, causal-abstraction, model-diffing]
read_from: "arXiv HTML: abstract, Sec. 1, 3 (setting, datasets, circuit discovery), 4.1 (amplification), 4.2 (corruption), 5 (neuroplasticity), 6, 8-9; Table 1 values did not render in the HTML; appendices not read"
digest_buckets: [2-known-answer-case, 4-supports, 3-definition, 5-cite-fix]
importance: 7
---

# Neuroplasticity and Corruption in Model Mechanisms: A Case Study of IOI (Chhabra 2025)

**What it does.** Fine-tunes GPT-2 small on the IOI task. On clean IOI data the runs last 1-100 epochs, and on two
corrupted variants the label is changed. *Name Moving* replaces the IO answer with a random name. *Subject Duplication*
makes the answer the subject S. The corrupted models are then fine-tuned again on clean data. At each stage the IOI circuit
is re-found by path patching and activation patching, following Wang et al. 2022. Heads are read by attention versus
unembedding-projection plots, direct logit attribution, group ablations, and cross-model activation/pattern patching
(CMAP, Prakash et al.). The analysis is repeated on the Greater-Than circuit.

**Key result.** Clean fine-tuning shows "circuit amplification". The existing heads do their old job more strongly (Name
Mover L9H9 attends less to S, and Negative Name Movers are amplified, not removed). Backup Name Mover L10H10 takes on
the Name Mover role (logit attribution 0.4 to 1.8). New heads appear that "replicate prior mechanisms", with no novel
mechanism. Corrupted fine-tuning changes localized circuit heads rather than adding mechanisms. Under Subject Duplication,
the logit difference goes from 3.55 to −11.06 in 5 epochs, and L9H9's projection flips to promote S and suppress IO.
Negative Name Movers flip from suppressing IO to suppressing S, and S-Inhibition heads are mostly silenced. Clean
retraining restores (and amplifies) the original mechanisms ("neuroplasticity"). The Greater-Than circuit shows the same
pattern.

**Quote.** "task-specific fine-tuning enhances the underlying mechanisms of circuits without introducing novel mechanisms,
even in longer training scenarios." (Sec. 4.1)

**Relevance to our paper.** This is a small, cheap and realistic known-answer set with independent evidence (patching,
ablation, logit attribution, CMAP) for all of its runs. We do not cite it and should. It gives graded cases that test our
kept clause directly:
- (i) Clean IOI fine-tuning: heads that write the same direction more strongly. This is the "kept up to a positive gain"
  case our Def. 1 is built for, so the definition should read reuse.
- (ii) Subject Duplication: the same heads with the write sign flipped (c < 0). Def. 1's c > 0 makes these not kept, so
  the reading would be new or changed, matching the paper.
- (iii) Backup heads that take over another head's role, and new heads that copy an old function. Under our definition
  these are *changed*, though the paper calls them amplification of existing mechanisms. That gap between a functional
  reading of reuse and our per-variable one is a definitional point the draft should address.

The obstacles: GPT-2 small has no released VPD (SAE and neuron or head featurizations exist), and its IOI circuit is not
in our 67M model. A replication at heads/neurons or SAE latents on GPT-2 small is affordable. The clean case also gives a
real-model check of whether trained fine-tunes "rescale", where our 67M copy head reads only 0.09-0.14 rescale. Wang et
al. 2025 dispute the general lesson for base-poor tasks.

**For the digest.**
- `2-known-answer-case`: GPT-2 small on IOI, with clean (amplify = reuse), corrupted (sign flip = changed) and retrain (restored) fine-tunes, evidenced by path patching, ablation, DLA and CMAP. It is affordable to replicate at heads or SAEs.
- `4-supports`: task fine-tuning amplifies existing heads (elicitation-like); corruption rewrites a few heads rather than adding mechanisms.
- `3-definition`: "amplification" counts new heads copying an old role as reuse; our per-variable kept clause would not. Contrast this explicitly.
