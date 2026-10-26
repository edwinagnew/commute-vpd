# Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning

## Metadata
- **Authors**: Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta (Facebook AI)
- **Year**: 2020 (arXiv v1, December 2020)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2012.13255

## Summary
The paper measures the intrinsic dimension (d90) of fine-tuning pretrained language models: the smallest random-subspace dimension that still reaches 90% of full fine-tuning performance. It finds that GLUE-style tasks can be solved to that level with a few hundred to a few thousand trainable parameters, that d90 falls as pretraining proceeds and as model size grows, and that lower d90 tracks better generalization. It also gives a compression-based generalization bound that depends on d rather than the full parameter count.

## Key Findings
- Fine-tuning RoBERTa-Large to 90% of full-fine-tuning accuracy on MRPC needs only about 200 trainable parameters (SAID method), and about 800 on QQP (Table 1). Random projections are used, so these are upper bounds on the true intrinsic dimension.
- During RoBERTa-Base pretraining (retrained from scratch), d90 for downstream tasks decreases monotonically across checkpoints, although the pretraining objective never sees the downstream tasks (Section 5.1, Figure 2).
- Across about a dozen pretrained models (BERT, RoBERTa, BART, ELECTRA, ALBERT, XLNet, T5, XLM-R), d90 on MRPC decreases as parameter count grows (Section 5.2, Figure 3). The paper notes that this comparison mixes pretraining methods and was not controlled.
- Lower d90 correlates with higher evaluation accuracy and a smaller relative generalization gap across checkpoints and tasks (Section 5.3, Figures 4-5). Structure-aware intrinsic dimension (SAID, with per-layer scaling) gives lower d90 than the structure-unaware DID method (Table 1).
- An application of Arora et al. (2018) compression bounds gives a generalization bound that grows with d and not with the pretrained model's parameter count. The bound holds only for subspace training, not for standard SGD (Section 5.3.1).

## Methodology
Fine-tuning is reparameterised as theta_D = theta_0 + P(theta_d), where P is a fixed random projection (Fastfood, with a per-layer scaling variant SAID) and only the d-dimensional vector is trained. A classification head is also trained within the subspace. For each (model, dataset, d) the authors run a small learning-rate search, and find d90 by sweeping or binary-searching d (range 10 to 4 million) against the full fine-tuning accuracy. The full-model baseline is the reference for the 90% threshold. The pretraining study retrains RoBERTa-Base for 200k steps with half batch size. The model-size study compares published checkpoints on MRPC only. The analysis is empirical, with a theoretical bound added on top.

## Relevance to Proposal
**Score: 4**

This is the main empirical support for strand 3 of the proposal (structure of fine-tuning updates). It suggests that the useful part of a fine-tuning update is low-dimensional, which is relevant to whether a small post-training update δW can be meaningfully carried by a few decomposition components. It does not address parameter decompositions, interpretability, or whether the update can be commuted past a decomposition. The projection used is random and not aligned with any interpretable basis, so it says little about Φ(D, δW).

## Limitations
- d90 is an upper bound from random Fastfood projections. The authors say the true intrinsic dimension is probably much lower, so the estimate depends on the projection method.
- d90 depends on the 90% threshold and on the full-fine-tune baseline. The threshold is relative, so it is not comparable across tasks with different baseline accuracies.
- The model-size trend (Figure 3) is measured on MRPC in the main text, and across heterogeneous pretrained models with different pretraining methods. The authors acknowledge it is not a controlled comparison.
- The pretraining trajectory uses one architecture (RoBERTa-Base) retrained with half the batch size and 200k steps, not the original 500k.
- All tasks are GLUE-style classification or NLI, with a classification head. Generative fine-tuning, RL post-training, and instruction tuning are not tested.
- The generalization bound applies only to subspace-trained models. The authors state that it is not shown that standard SGD optimises in a low-dimensional subspace, though they say this "seems to be confirmed" experimentally.
- Preprint, not peer-reviewed in this version.

## Key Quotes
> "With RoBERTa-Large, we can reach 90% of the full fine-tuning solution of MRPC using roughly 200 parameters and 800 parameters for QQP" (Section 4.2)

> "We see that the intrinsic dimensionality of RoBERTa-Base monotonically decreases as we continue pre-training. We do not explicitly optimize for intrinsic dimensionality, specifically during pre-training" (Section 5.1)

> "Explicitly, these bounds only apply to pre-trained methods trained with the intrinsic dimension subspace method; research has yet to show that standard SGD optimizes in this low dimensional space (although experimentally, this seems to be confirmed)." (Section 5.3.1)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv paper.
