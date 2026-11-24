---
slug: geiger2022-iia
title: "Inducing Causal Structure for Interpretable Neural Networks"
authors: Atticus Geiger et al. (Wu, Lu, Rozner, Kreiss, Icard, Goodman, Potts)
year: 2022
id: 2112.00826
tags: [CAUSAL, DEFINITION, BACKGROUND]
primary_tag: CAUSAL
in_our_bib: no
cited_in_draft: no
themes: [causal-abstraction, reuse-definitions]
read_from: "Identified from Geiger et al. 2025's reference list ('Geiger et al. (2022b)' = Inducing causal structure..., ICML 2022, which their Def. 53 credits with interchange intervention accuracy). Read arXiv HTML v2: abstract, Sec. 1, Sec. 3 (IIT, IIA definition, Fig. 1 toy example), Sec. 5-6 results text, conclusion."
digest_buckets: [3-definition]
importance: 4
---

# Inducing Causal Structure for Interpretable Neural Networks (Geiger 2022)

**What it does.** This is the "Geiger et al. 2022b" that Geiger et al. 2025 (Def. 53) credit with interchange intervention
accuracy (IIA). It introduces interchange intervention training (IIT). Variables of a symbolic causal model are aligned with
chosen neurons of a network, and the network is trained so that interchange interventions on those neurons give the
causal model's counterfactual output. It is tested on MNIST-PVR (ResNet), ReaSCAN (CNN-LSTM) and MQNLI (BERT fine-tuned).
IIA is defined there as the share of aligned (base, source) interchange interventions on which the two models agree.

**Key result.** IIT gives the best systematic generalisation and the highest IIA in all three settings. On MQNLI it reaches
near-perfect accuracy and IIA (about 100%) when the variable is aligned with BERT layer 6 or later. Data augmentation gives
perfect behaviour but "very low" IIA. In the toy example (Fig. 1), a network that is perfect on input-output behaviour
has 81.25% IIA. One IIT update brings it to 100% IIA "while maintaining the same behavior".

**Quote.** "IIT is fully differentiable, flexibly combines with other objectives, and guarantees that the target causal
model is a causal abstraction of the neural model when its loss is zero." (abstract)

**Relevance to our paper.** It is the source of IIA, the standard graded interchange score. Our "needed" clause (Def. 2)
uses single-variable interchange interventions, so if the draft ever reports an interchange success rate, it should cite
this paper alongside Geiger 2025. Two findings bear on our claims. First, the toy example and the data augmentation result
are clean demonstrations that two networks with the same input-output behaviour can differ internally. That is our
contribution-1 claim ("two fine-tunes with one input-output behaviour can differ under it"), shown with an interchange
test, so it is a useful precedent. Second, IIT is a recipe for building *trained* known-answer fine-tunes, which our
paper lacks: our known answers are built weight edits, and on trained fine-tunes ρ fails. An IIT loss that forces F's
behaviour through an aligned base variable (reuse) or through a fresh subspace (new) would give a trained fine-tune whose
status is fixed by the training objective, not by an edit. That is a design idea, not something the paper shows for
fine-tunes. It does not compare two trained networks.

**For the digest.**
- `3-definition`: source of interchange intervention accuracy; also a precedent for "same behaviour, different causal structure", and IIT could train known-answer fine-tunes (our idea, not the paper's).
