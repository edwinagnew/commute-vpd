# Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution

## Metadata
- **Authors**: Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang (Stanford)
- **Year**: 2022 (arXiv v1 February 2022; published at ICLR 2022)
- **Source**: arXiv preprint (2202.10054v1)
- **DOI/URL**: https://arxiv.org/abs/2202.10054

## Summary
The paper shows that full fine-tuning of a pretrained model can give worse out-of-distribution (OOD) accuracy than linear probing (training only the head on frozen features), even though fine-tuning gives better in-distribution (ID) accuracy, when the pretrained features are good and the distribution shift is large. The mechanism is "feature distortion": while the head is being fit to ID data, gradient updates move the lower-layer features for ID-like inputs and leave the directions orthogonal to the training span largely unchanged, so the features that OOD inputs see are inconsistent with the head. The authors propose LP-FT (linear probing, then full fine-tuning) as a simple fix that gets the best ID and OOD accuracy on their benchmarks.

## Key Findings
- Across 10 distribution-shift datasets (ResNet-50 with MoCo-v2/CLIP, and CLIP ViT-B/16 for ImageNet), fine-tuning averages about 2% higher ID accuracy than linear probing (85.1% vs 82.9%) but about 7% lower OOD accuracy (59.3% vs 66.2%). Linear probing wins OOD on 8 of 10 datasets; the exceptions are CIFAR-10.1 and ImageNetV2, where ID and OOD are close (Tables 1-2).
- In a two-layer overparameterized linear model with good pretrained features, the authors prove a lower bound on fine-tuning's OOD error that holds at every time step of gradient flow, so early stopping does not close the gap. Their experiments agree: with OOD-tuned early stopping, linear probing still reaches 67.1% vs 61.3% for fine-tuning.
- Feature distortion is measured on Living-17: the feature change from fine-tuning is larger for ID examples (0.0188) than OOD examples (0.0167), and LP-FT changes features about 20x less than fine-tuning.
- Fine-tuning can beat linear probing OOD when the conditions fail: with weaker pretrained features (MoCo-v1, about 10% worse on ImageNet) fine-tuning wins OOD (71% vs 66%), and on CIFAR-10 to CIFAR-10.1 it also wins (92.3% vs 82.7%).
- LP-FT beats both baselines on average (ID 85.7%, OOD 68.9%) and on 10/10 OOD datasets, at roughly the same compute as fine-tuning.

## Methodology
Theory: a two-layer linear network (linear feature map B, linear head v) trained by gradient flow in an overparameterized regression setting, with pretrained initialization B0 close to the true B*, and a random or zero head initialization. The analysis uses a conservation ("balancedness") invariant from prior work and a principal-angle argument between the pretrained row space and the training-span complement. Experiments: fine-tuning versus linear probing (L2-regularized logistic regression on frozen penultimate features) versus LP-FT, on ten vision distribution-shift benchmarks (Breeds Living-17, Entity-30, DomainNet, CIFAR-10 to STL, CIFAR-10.1, FMoW geo-shift, ImageNetV2/-R/-A/-Sketch). Each configuration runs over 3 seeds, with learning-rate sweeps selected on ID validation; OOD data is used only for evaluation (except for one explicitly labelled early-stopping ablation).

## Relevance to Proposal
**Score: 4**

This is a theory-plus-benchmark result about how a small, targeted parameter change (a head, or a head-then-full update) can distort pretrained features in a way that is invisible on the training distribution. That matches the proposal's Strand 3 (structure of fine-tuning updates, and which directions of W a fine-tune actually moves) and Strand 4 (whether post-training reuses existing structure or distorts it). The concrete relevance is the "feature distortion" framing, which asks which directions the update changes and which it leaves alone, and the measurement of how much ID versus OOD features move. It is only an analogy, though: the paper has no decomposition, no interpretability measurement, and no language models.

## Limitations
- The main theorem is for linear feature maps and gradient flow (vanishing learning rate), and the authors say the nonlinear case is open. The paper itself lists non-linear activations, layerwise learning rates and explicit regularization as open questions.
- The theory assumes well-specified targets and good pretrained features (small d(B0, B*)); the bound does not show a matching upper bound for fine-tuning when features are imperfect (the authors say so in Section 3.4).
- Experiments cover only image classification with ResNet-50 and ViT-B/16, using one or two pretraining methods per dataset. Results for language models or for other update forms (LoRA, sparse fine-tuning) are not tested.
- Distortion is measured as a Euclidean feature-change norm averaged over examples, which is a coarse proxy. It does not say which directions change or whether the change is interpretable.
- The ID/OOD split is defined by datasets, not by a controlled shift parameter, so the "large shift" condition is judged informally.
- LP-FT is presented as a simple first step, not a state-of-the-art method. Its best-of-both result is on the paper's chosen benchmarks, and LP-FT loses to fine-tuning on FMoW ID.

## Key Quotes
> "We prove that the OOD error of fine-tuning is high when we initialize with a fixed or random head—this is because while fine-tuning learns the head, the lower layers of the neural network change simultaneously and distort the pretrained features." (Abstract, p. 1)

> "Features get distorted: representations change only in the ID subspace (i.e., subspace spanned by the training data) and are unchanged in the orthogonal subspace." (Section 3.2.1, pp. 6-7)

> "As expected, the average distance for ID examples (0.0188 ± 0.0001) is more than for OOD examples (0.0167 ± 0.0001)." (Section 4.3, p. 13)

> "Interestingly, we show that this feature distortion issue cannot be simply fixed by early stopping—throughout the entire process of fine-tuning, we never pass through parameters that do well OOD (relative to linear probing)." (Section 1, p. 2)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
