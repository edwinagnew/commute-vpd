# Singular Vectors of Attention Heads Align with Features

## Metadata
- **Authors**: Gabriel Franco, Carson Loughridge, Mark Crovella (Boston University)
- **Year**: 2026
- **Source**: Proceedings of the 43rd International Conference on Machine Learning (ICML 2026), PMLR 306; arXiv:2602.13524v2
- **DOI/URL**: arXiv:2602.13524v2 (https://arxiv.org/abs/2602.13524). Code: https://github.com/gaabrielfranco/svf-alignment

## Summary
The paper asks why and when the singular vectors of an attention head's QK matrix (Ω = W_Q^T W_K) line up with the linear feature directions the head uses. It shows this alignment in a toy autoencoder-plus-attention model, proves conditions under which it holds, and proposes "sparse attention decomposition" (SAD) as a testable consequence that it checks in GPT-2 and Pythia-160M during pretraining.

## Key Findings
- In a toy model with an added attention head, singular vectors of Ω align with the features the head attends to. Theorem 1 gives the rank-1 case: the top singular vectors are the covariance-whitened feature pair. Corollary 1 gives the isotropic special case.
- Features the head does not use tend to orthogonalize against the features it does use (Theorem 3 argues this minimises reconstruction interference). Multi-pair alignment follows from the two effects together.
- Alignment is approximate under anisotropy: mean cosine similarity stays above 0.75 at anisotropy near the largest value seen in GPT-2 (preliminary, frozen-feature experiment). Theorem 2 is a loose lower bound.
- Features and singular vectors co-evolve during training. Singular vectors align in order of their impact on attention loss, and features shift later to orthogonalize (Figure 3).
- SAD: under alignment, relative attention decomposes sparsely in the SVD basis. This emerges during training in the toy model (Figure 7a) and in Pythia-160M on the IOI heads (Figure 7b). Randomly rotating U and V removes the sparsity, so it is not just a low-rank effect. In GPT-2 and Pythia, the number of singular vectors needed to reconstruct relative attention (N_recon) is typically small (about 1 to 4 in Pythia).

## Methodology
Toy model: the Elhage et al. (2022) toy autoencoder, with features w_i in R^D, tokens r = Wf, and a ReLU reconstruction. An attention head with logit r^T Ω s_j is added. Targets are given at the logit level (T_ij), and the loss is L = L_recon + λ L_attn. Theory: closed-form characterisation of Ω* after training under softmax attention (Theorem 1), plus perturbation-style bounds for anisotropic features (Theorem 2) and an orthogonalisation result for fixed Ω (Theorem 3). Real-model evidence: SVD of the QK matrices of IOI-identified heads in Pythia-160M across 130 checkpoints, and of GPT-2 heads on 128 IOI prompts. Sparsity is measured with the Rolls and Tovee (1995) sparseness index S(v) and with N_recon. Features in real models are not observed directly, so alignment is tested only through the SAD prediction.

## Relevance to Proposal
**Score: 4**

This is weight-based interpretability (strand 1): it treats singular vectors of a weight matrix as candidate feature directions, and it tracks that alignment through training checkpoints (strand 6). It is only indirectly relevant to the proposed commutation Φ(D, δW). It studies attention QK matrices, not parameter decompositions such as VPD/SPD/APD, and it does not study fine-tuning or post-training updates. The useful overlap is conceptual: if a weight-derived basis tracks features during training, that is a reason to expect decomposition-based feature identity to persist under small updates, and SAD gives a concrete, cheap test for that.

## Limitations
- The strongest evidence comes from a toy model with N = 20 to 100 features and H = 10 to 50 dimensions. Alignment in real models is not shown directly, only through the SAD prediction, which the authors acknowledge (Section 6: "directly examining features in real models is out of scope").
- Real-model evidence covers one architecture family (GPT-2, Pythia-160M) and one task (IOI), with the heads chosen from a prior circuit analysis. The Pythia results are over pretraining checkpoints, not post-training.
- A single head is studied. The paper says cross-head allocation of singular vectors is open (Section 5, "Limitations" text).
- Anisotropy results are preliminary (frozen features, one toy configuration). Real-model cone directions are discussed but not tested (Section 6).
- Theorem 2 is described by the authors as a loose lower bound.
- Sparsity (S(v), N_recon) is a property of the attention decomposition. Interpreting it as evidence for particular features rests on the alignment hypothesis, which is itself assumed.
- Features are assumed to be linear and additive. The paper says this is consistent with the linear representation hypothesis but does not test it in real models.

## Key Quotes
> "We identify _sparse attention decomposition_ as a testable prediction of alignment, and show evidence that it emerges in real models in a manner consistent with predictions." (Abstract)

> "Because singular vectors can shift during training, features can orthogonalize without impacting attention loss." (Section 4, Multi-Feature-Pair Alignment)

> "Sparse attention decomposition is _not_ attributable simply to the presence of a few large singular values in attention matrices." (Section 5, Evidence in Language Models)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv/ICML paper with no public comment thread.
