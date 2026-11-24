---
slug: lubana2023-mechanistic-mode-connectivity
title: "Mechanistic Mode Connectivity"
authors: Ekdeep Singh Lubana et al. (Bigelow, Dick, Krueger, Tanaka)
year: 2023
id: 2211.08422
tags: [DEFINITION, DIFFING, SUPPORTS, BASELINE]
primary_tag: DEFINITION
in_our_bib: lubana2023mmc
cited_in_draft: yes
themes: [reuse-definitions, model-diffing, elicitation]
read_from: "arXiv HTML: abstract, Sec. 1-5 (Defs. 1-5, Prop. 2, Conjecture 1, Figs. 4-5), Sec. 6.1 and Table 1 caption"
digest_buckets: [3-definition, 4-supports, 1-missing-method]
importance: 5
---

# Mechanistic Mode Connectivity (Lubana 2023)

**What it does.** This ICML 2023 paper asks whether minimisers that rely on different input attributes are connected by
low-loss paths. It defines two models as *mechanistically similar* if they are invariant to the same set of unit
interventions on the data-generating latents (Def. 4). Base models are trained on CIFAR-10/100 and Dominoes with planted
synthetic cues (box cues correlated with the label), so whether a model relies on the cue or on the image is known by
construction and checked with counterfactual test sets (no cue, random cue, random image). Pretrained cue-reliant models
are then fine-tuned on clean data with different learning rates. The paper also proposes connectivity-based fine-tuning
(CBFT).

**Key result.** Mechanistically dissimilar models are connected by quadratic paths but not by linear ones, even after
permutation matching (Fig. 4). This leads to Conjecture 1: if two models are not linearly connected, they are
mechanistically dissimilar. In the fine-tuning test (Fig. 5), a small or medium learning rate leaves the fine-tune
linearly connected to the pretrained model and still reliant on the cue. A large learning rate, or perfect cue-label
correlation, gives a loss barrier and a change of mechanism. CBFT removes cue reliance better than fine-tuning, LLR and
LP-FT (Table 1).

**Quote.** "naïve fine-tuning on a downstream dataset can fail to alter a model's mechanisms, e.g., fine-tuning can fail to
eliminate a model's reliance on spurious attributes." (abstract)

**Relevance to our paper.** The draft cites it accurately ("mechanistically similar when they share an invariance set").
This is the clearest foil for our contribution 1. The definition is purely behavioural, about which input changes the
model ignores, so two fine-tunes with the same invariances count as mechanistically the same however they compute. Our
definition looks inside and could tell them apart. The fine-tuning result is weak support for elicitation, in that small
fine-tunes keep the base mechanism. The setting is vision with a planted cue, not language post-training. The method is
cheap to run on our setting: compute the loss barrier on the linear path between M and F, on D and on counterfactual
inputs. But the conjecture runs one way only (a barrier implies dissimilar), and our fine-tunes are small, so the likely
reading is "no barrier" for every pair. That reading carries no information about reuse, so it is a weak baseline at best.
The planted-cue construction is a model for trained known answers: the cue is a mechanism whose use is known, so a
fine-tune that does or does not keep using it has a known answer.

**For the digest.**
- `3-definition`: a behavioural (invariance-set) definition of mechanistic similarity, to contrast with ours, which looks inside.
- `4-supports`: low-LR fine-tunes stay linearly connected and keep the pretrained cue mechanism (vision, planted cue).
- `1-missing-method`: a linear-path loss barrier between M and F is cheap to run, but its one-way conjecture probably gives no call at fine-tune size.
