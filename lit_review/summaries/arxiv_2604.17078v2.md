# Understanding and Enforcing Weight Disentanglement in Task Arithmetic

## Metadata
- **Authors**: Shangge Liu, Yuehan Yin, Lei Wang, Qi Fan, Yinghuan Shi, Wenbin Li, Dacheng Tao (author line in the markdown is partly garbled by strikethrough edits; one name, probably "Yang Gao", is struck out and may be incomplete)
- **Year**: 2026 (arXiv ID 2604.17078, v2)
- **Source**: arXiv preprint (no venue stated in the file)
- **DOI/URL**: arXiv:2604.17078v2; code at https://github.com/RL-MIND/OrthoReg

## Summary
The paper proposes Task-Feature Specialization (TFS), the idea that a model assigns distinct internal features to distinct tasks, as the underlying reason task arithmetic works. It argues TFS is sufficient for weight disentanglement and gives rise to weight-vector orthogonality, and it proposes OrthoReg, a regulariser that pushes the columns of each fine-tuning weight update ΔW toward orthonormality, to reduce interference when task vectors are summed or negated. Experiments are on CLIP ViT-B/32, B/16 and L/14 image classifiers with eight vision tasks.

## Key Findings
- Under an NTK-linearisation hypothesis, the authors show that TFS (disjoint specialised feature sets per task) implies weight disentanglement (Theorem 1) and block/column orthogonality of the weights (Corollary 1). These are the authors' theoretical claims, under their linearisation assumption.
- Adding OrthoReg to baselines (non-linear FT, TTA, attention-only FT, LoRA-ATT) improves 8-task addition absolute accuracy by about 1–4 points and normalised accuracy by up to about 18 points on ViT-B-16 and ViT-L-14 (e.g. Non-lin. FT normalised accuracy on ViT-L-14 rises from 89.19 to 100.08).
- For task negation, OrthoReg lowers target-task accuracy further while keeping ImageNet control accuracy roughly unchanged (e.g. ATT-FT on ViT-L-14: about 10 points lower target accuracy).
- Task vectors trained with OrthoReg have visibly lower pairwise cosine similarity (heatmap, ViT-B-16, Non-lin. FT). The paper reports this as support for its "angle control" mechanism.
- The authors argue TTA and OrthoReg share the same mechanism (inter-task vector orthogonality), with OrthoReg being cheaper than TTA's Jacobian-based training.

## Methodology
Theory: a first-order (NTK) linearisation of the network around θ0, a pairwise interference term τ_j^T J(x) for x in task t's domain, and an analysis of when it vanishes. Method: a Frobenius-norm penalty ||ΔW^T ΔW − I||²_F summed over all tuned linear layers, added to the task loss with coefficient λ chosen on [0.1, 100]. Experiments: CLIP-pretrained ViTs, text encoder frozen, a single uniform merging coefficient α grid-searched on {0, 0.05, …, 1}, and evaluation on Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN plus an ImageNet control for negation. Empirical evidence on pretrained CLIP weights (column angles near 90°) is also presented.

## Relevance to Proposal
**Score: 3**

The work sits in the proposal's strand 3 (structure of fine-tuning updates, task arithmetic and weight disentanglement). It is relevant to the question of whether a small update δW has a geometry (near-orthogonal columns, low interference across tasks) that could be exploited when mapping it through a parameter decomposition. However, it works on image classifiers with full or PEFT fine-tuning and never touches parameter decompositions, sparse subcomponents, LMs or post-training of language models, so it does not directly bear on commuting updates past a decomposition.

## Limitations
- The main text has no dedicated limitations section. Limitations below are inferred from the content.
- Theory relies on the NTK linearisation hypothesis, which is known to be loose for practical fine-tuning; Theorem 2 only gives an "approximately orthogonal" conclusion.
- Experiments are limited to CLIP ViT image classification with eight vision tasks and a single merging coefficient; no language-model or RL post-training experiments are reported.
- Orthogonality is enforced on ΔW in a fixed-coefficient, uniform-α merging protocol; task-adaptive coefficients are explicitly not studied.
- Several reporting irregularities in the file: theorem numbering is inconsistent in places (Theorem 3 vs Theorem 1 for the same TFS result; "Theorem 4" for the OrthoReg result), and in Table 1 the ViT-B-16 Non-lin. FT normalised accuracy (75.39) equals its absolute accuracy, which looks like a transcription error. Figure 2's percentage annotation is partly struck through.
- The "Appendix" content (over ~1000 lines of the file) was not read in full; claims about the appendix are not verified here.

## Key Quotes
> "we find that TFS also gives rise to an observable geometric consequence: weight vector orthogonality." (Section 1, Introduction)

> "Directly enforcing the abstract property TFS is intractable. Our theory suggests a practical alternative: enforcing its geometric consequence, orthogonality." (Section 4.3.2)

> "Achieving a normalized accuracy at or above 100% is the functional realization of ideal weight disentanglement, as it signifies that the single merged model performs on par with eight individually specialized models." (Section 5.2)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
