# Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models

## Metadata
- **Authors**: Guillermo Ortiz-Jimenez, Alessandro Favero, Pascal Frossard (EPFL)
- **Year**: 2023
- **Source**: NeurIPS 2023 (37th Conference on Neural Information Processing Systems); arXiv 2305.12827v3
- **DOI/URL**: https://arxiv.org/abs/2305.12827

## Summary
The paper studies why task arithmetic (adding or negating fine-tuning "task vectors" in weight space) works for CLIP vision-language models. It argues the key property is weight disentanglement: distinct weight-space directions change the model's function in disjoint input regions. Fine-tuning in the linearized (NTK) tangent space amplifies this disentanglement and improves task arithmetic on standard benchmarks.

## Key Findings
- Task arithmetic is not explained by linear fine-tuning. Post-hoc linearized CLIP models (f_lin applied to non-linear task vectors) lose a lot of absolute task-addition accuracy (e.g. ViT-B/32: 71.4% non-linear vs 57.1% post-hoc linear), so the non-linear components matter.
- Weight disentanglement is the necessary condition. A disentanglement error ξ(α1, α2) measures how much adding two task vectors interferes outside their supports; it is small near θ0 and grows for coefficients above about 1.
- Linearized (tangent-space) fine-tuning improves task arithmetic: up to 5.8 points higher multi-task accuracy on addition and up to 13.1 points more forgetting on negation, with the gain largest for ViT-B/32 and shrinking for larger ViTs.
- Weight disentanglement is an emergent property of pre-training: task addition from randomly initialized ViTs gives near-chance multi-task accuracy (Table 3).
- Theory: a kernel can perform task arithmetic if each task is represented by eigenfunctions of the NTK localized to that task's support (Proposition 1). Empirically, NTK eigenfunctions used for the RESISC45 task have higher local energy on RESISC45 inputs than on control (Cars) inputs.

## Methodology
Empirical study on CLIP ViT-B/32, B/16 and L/14 with eight vision tasks (Cars, DTD, SUN397, EuroSAT, GTSRB, MNIST, SVHN, RESISC45), using the Ilharco et al. task addition and negation benchmark. Linearization uses the first-order Taylor expansion f_lin(x; θ0 + τ) = f(x; θ0) + τᵀ∇f(x; θ0), implemented as a Jacobian-vector product. The authors compare non-linear fine-tuning, post-hoc linearization and linearized fine-tuning. Disentanglement is measured as a prediction-disagreement error over 2D weight-space slices. The NTK localization analysis diagonalizes the kernel matrix on training and control inputs. Theory is a set of propositions in the appendices (eigenfunction support conditions, a remark on local linear independence).

## Relevance to Proposal
**Score: 5**

The paper bears on strand 3 of the proposal (task arithmetic, weight disentanglement, linearised/NTK fine-tuning) and on the question of whether a small weight-space update δW can be localized and composed cleanly. Its central object is the function-space locality of a weight-space edit, which parallels the "commuting" question of whether δW lands in identifiable components. It does not study parameter decompositions, interpretability, or LLMs, so it gives only indirect support for the decomposition-based map Φ(D, δW).

## Limitations
- Experiments are on CLIP ViTs with image classification tasks. No language models, no RL post-training, no decomposition-based representation.
- Disentanglement is measured with a prediction-disagreement proxy on 2D slices of weight space, which does not directly test the high-dimensional case.
- Linearized fine-tuning is costlier than standard fine-tuning in practice (JVP overhead; the paper argues O(1) overhead but leaves faster linearization to future work).
- The NTK localization result is shown for one task pair in the main text (RESISC45 vs Cars); the claim that localization is necessary is conditional on local linear independence, which the authors state holds only under extra assumptions.
- Weight disentanglement is shown to be "emergent" from pre-training only by comparison with random initialization, not by tracking pre-training dynamics.

## Key Quotes
> "we show that the sole requirement for task arithmetic is actually _weight disentanglement_, where distinct directions in weight space correspond to changes of the network in disjoint regions of the input space" (Section 1, Introduction)

> "fine-tuning models in their tangent space by linearizing them amplifies weight disentanglement" (Abstract)

> "task arithmetic is a property acquired during pre-training. This observation goes beyond the traditional representation learning view of pre-training, emphasizing that pre-training not only leads to semantically disentangled feature representations but also to the disentanglement of the weights that govern the output on those semantic sets." (Section 6.2)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is a peer-reviewed NeurIPS paper.
