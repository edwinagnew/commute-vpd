# Fine-tuning Happens in Tiny Subspaces: Exploring Intrinsic Task-specific Subspaces of Pre-trained Language Models

## Metadata
- **Authors**: Zhong Zhang, Bang Liu, Junming Shao
- **Year**: 2023
- **Source**: arXiv preprint (arXiv:2305.17446v2)
- **DOI/URL**: https://arxiv.org/abs/2305.17446

## Summary
The paper proposes that fine-tuning a pre-trained language model (BERT-base-cased, RoBERTa-base) happens inside a small task-specific subspace. The subspace is estimated by stacking the parameter checkpoints of an ordinary fine-tuning run (one per epoch), taking an SVD, and using the principal right-singular vectors as a re-parameterisation basis for each encoder layer. Training only the low-dimensional coordinates (32 per layer, 64 for MNLI) recovers close to full fine-tuning performance on GLUE, and the authors also report that a subspace from other tasks transfers partially, and that the updates in the subspace contain a few "outlier" weight dimensions whose removal hurts performance.

## Key Findings
- Per-layer fine-tuning in a 32-dimensional (64 for MNLI) trajectory-derived subspace roughly matches full fine-tuning on GLUE (BERT average 81.21 vs 82.13 full; RoBERTa average 84.40 vs 85.43 full) (Table 1). Random subspaces of the same dimension do clearly worse (BERT 72.54; RoBERTa 71.13).
- Subspace fine-tuning is unstable across initialisations, so the authors average 16 low-dimensional vectors (an ensemble) to reduce variance (Section 3.3).
- Subspaces derived from other tasks transfer partially. Transferred subspaces beat the random baseline but are worse than in-task subspaces, and large source tasks (SST-2, QQP, MNLI, QNLI) are worse sources than small ones (Figure 2). The authors attribute this to higher intrinsic dimension of large tasks; this is a conjecture.
- A single unified 8-dimensional subspace built from one checkpoint per task still supports fine-tuning across all tasks (Table 2, average 82.48 BERT, 85.83 RoBERTa). Leaving out the target task (zero-shot) drops performance substantially but remains above random.
- Disabling the top outlier weight in each layer has almost no effect, but zeroing about 0.3% of encoder parameters chosen as outliers (|update| > 3 sigma) severely degrades GLUE performance, while zeroing the same number of random weights does not (Table 3). Outlier patterns appear in the subspace-induced updates in all layers but not in random-subspace updates (Figure 4).

## Methodology
- Fine-tune BERT-base-cased and RoBERTa-base with the HuggingFace default GLUE script, saving a checkpoint each epoch (trajectory length 32, or 64 for MNLI). Flatten each encoder layer's parameters, stack checkpoints into a matrix, and take the SVD. Right-singular vectors form a per-layer projection V.
- Re-parameterise each encoder layer as theta_D = theta_0 + V theta_d, freeze theta_0 and V, and train only theta_d (learning rate 0.01). Embeddings and the classifier stay in the original space.
- Transductive setting: V is computed from the same task. Inductive setting: V is computed from other tasks. Unified setting: one checkpoint per task pooled into one SVD.
- Outlier analysis: inspect the product V theta_d (the implied full-space weight update), threshold at 3 sigma, zero those entries, and re-evaluate. Random-dimension control of matched size.
- Evaluation on GLUE (8 tasks), 5 seeds, mean reported. Ablation over subspace dimension 8/16/32 (Appendix A.2, Tables 5-6). Results show monotone gains with dimension.

## Relevance to Proposal
**Score: 4/10**

The paper is a weak but direct fit for strand 3 (structure of fine-tuning updates: low intrinsic dimension, task-specific subspaces, sparse/outlier structure of updates). It is relevant to the commute idea in that it builds a basis from the fine-tuning trajectory and restricts updates to it, which is close to asking whether a small post-training update lives in a low-dimensional, reusable subspace. It does not address parameter decompositions, rank-1 subcomponents, or interpretability of the resulting model, and its evidence is BERT/RoBERTa GLUE classification, so it is useful as background on subspace fine-tuning rather than as a method to build on.

## Limitations
- The authors themselves note the subspaces are local (layer-wise), so no global subspace is found, and they do not test whether a global one exists (Limitations section).
- Only natural language understanding (GLUE classification) is tested; no generation tasks, no decoder-only or encoder-decoder models, and only base-size encoders (Limitations section).
- The transductive subspace is computed from the same target task's fine-tuning trajectory, so "intrinsic dimension" is partly defined by the trajectory sampling (the dimension equals the number of checkpoints, 32 or 64). This makes the "tiny subspace" claim partly a consequence of the chosen construction, not a measured property of the task.
- Subspace fine-tuning is reported as unstable to initialisation; the ensemble of 16 is an ad hoc fix and its effect is not separately ablated.
- The mechanism behind the outlier dimensions is explicitly left open. The threshold (3 sigma) and the choice of which outliers to zero are heuristic, and the top-1 result (negligible effect) shows the effect is distributed rather than localised.
- Statistics are thin: 5 seeds, mean values reported without variance or significance tests in the main tables.
- Outlier-dimension definitions differ from prior work (weights rather than output channels), so the comparison to prior "outlier dimension" literature is not like-for-like, as the authors acknowledge.
- Appendix hyperparameters are default HuggingFace settings with a fixed learning rate of 0.01 for the subspace coordinates, with no tuning reported for the subspace method.

## Key Quotes
> "PLMs can be effectively fine-tuned in the subspace with a small number of free parameters." (Abstract)

> "We show that it is possible to uncover the intrinsic task-specific subspaces with a fine-tuning trajectory by finding its principal directions." (Section 1, Introduction)

> "The existence of a task-specific global subspace is yet to be ascertained." (Limitations)

> "Disabling the top outlier dimension of each encoder layer ... has almost no impact on model performance." (Section 4.5)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper, no comments).
