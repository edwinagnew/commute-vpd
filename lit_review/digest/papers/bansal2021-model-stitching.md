---
slug: bansal2021-model-stitching
title: "Revisiting model stitching to compare neural representations"
authors: Bansal et al.
year: 2021
id: 2106.07682
tags: [DEFINITION, BASELINE, DIFFING]
primary_tag: DEFINITION
in_our_bib: no
cited_in_draft: no
themes: [reuse-definitions, model-diffing, baselines]
read_from: "local PDF text (~/commute-data/lit_review_pdfs/survey/2106.07682), pp. 1-9: abstract, Sec. 2 (stitching penalty, comparison with CKA, Table 1), Sec. 4 (stitching connectivity, Definition 1), Sec. 5-6 (SSL vs supervised, 'more is better'); appendix not read"
digest_buckets: [1-missing-method, 3-definition]
importance: 4
---

# Revisiting model stitching to compare neural representations (Bansal 2021)

**What it does.** Defines the stitching penalty: the extra loss when the bottom l layers of network B are plugged into
the frozen top of network A through a simple trainable layer (1x1 conv or token-wise linear), relative to A itself.
A random bottom network is the control for a stitcher that is too strong. Applied to CIFAR-10 and ImageNet ResNets,
comparing random seeds, supervised vs self-supervised training (SimCLR, SwAV, DINO), and models trained with more
data, width or epochs.

**Key result.** Networks from different seeds, and supervised vs self-supervised ResNet-50s, can be stitched at every
layer with little loss ("stitching connectivity", Definition 1). For SimCLR vs supervised ResNet-18 on CIFAR-10, CKA
ranges 0.35-0.9 across layers while the stitching penalty stays at most 3%. Representations from more data, width or
training improve a weaker top when stitched in, which a symmetric measure like CKA cannot show.

**Quote.** "these two different parameter settings yield essentially equivalent internal representations– this is
what it means to be stitching-connected." (Sec. 4, p. 7)

**Relevance to our paper.** Stitching is the cleanest published formal notion of "B's representation can be used by
A's later computation up to a simple map", and so the value-based, rotation-tolerant alternative to our definition,
which compares mechanisms and has no cross-model patching. It should be cited where we say reuse is compared "by
mechanism, not by value", with the reason we depart from it. It could also run in our setting as a baseline: stitch the
fine-tune's layers up to l into the base model's top through a learned linear map, and read the penalty on the
behaviour's inputs. Unlike our "kept up to a positive scalar" clause it would forgive a rotation, which is the open
question for our copy-head reading, so it is a ready reference for the pending rotation-tolerant kept clause. Its
limit for us: it tests representations at one cut, not whether F's later mechanisms are M's.

**For the digest.**
- 1-missing-method: linear stitching of F's bottom into M's top, penalty on D: cheap, runnable on the 67M, rotation-tolerant; a candidate value-based baseline.
- 3-definition: stitching penalty / stitching connectivity: a formal "same representation up to a simple map" to contrast with our by-mechanism, up-to-scalar kept clause.
