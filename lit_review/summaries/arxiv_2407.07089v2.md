# Fine-Tuning Attention Modules Only: Enhancing Weight Disentanglement in Task Arithmetic

## Metadata
- **Authors**: Ruochen Jin, Bojian Hou, Jiancong Xiao, Weijie Su, Li Shen
- **Year**: 2024 (arXiv v2; published at ICLR 2025)
- **Source**: ICLR 2025 (arXiv preprint 2407.07089)
- **DOI/URL**: https://arxiv.org/abs/2407.07089 (code: https://github.com/kyrie-23/task_arithmetic_tangent)

## Summary
The paper studies task arithmetic, where separately fine-tuned "task vectors" (weight differences from a pretrained model) are added together to get one multi-task model. It argues that interference between tasks comes from a lack of "weight disentanglement," and that attention modules alone show near-linear ("kernel") behaviour under fine-tuning. The authors therefore fine-tune only the attention projections (Wq, Wk, Wv, Wo) with ordinary non-linear training, which gives better merged accuracy than full non-linear fine-tuning or NTK linearisation at lower cost.

## Key Findings
- Post-hoc linearisation (a first-order Taylor expansion around the pretrained weights) matches non-linear fine-tuning more closely for attention-only updates than for full-model updates on CLIP ViTs. The authors read this as attention modules having stronger kernel behaviour (Sec. 3.2, Fig. 3).
- Fine-tuning only attention modules gives 78.37% absolute / 87.42% normalised accuracy on 8 vision tasks with ViT-B/32, against 70.00% for non-linear fine-tuning and 76.26% for NTK linearisation (Table 2). Gains are also reported for ViT-B/16, ViT-L/14 and Flan-T5-base on GLUE (Table 3, average 85.30 vs 73.25 and 77.77).
- Training cost is lower: 10:15 min vs 17:38 min (non-linear) and 40:59 min (NTK) in the authors' setup, with 63.7M trainable parameters instead of 222M (Table 4).
- Weight disentanglement, measured with a KL-based "disentanglement error" on the representation module, comes mostly from the representation module. Classification heads limit how far task arithmetic can be pushed (Sec. 4, Fig. 6).
- Adding bias terms or MLP weights to the attention-only update reduced merged accuracy in the ablation (Table 5), although these variants still beat NTK linearisation.

## Methodology
Empirical. The authors fine-tune CLIP ViT-B/32, B/16 and L/14 on eight image tasks (Cars, DTD, SUN397, EuroSAT, GTSRB, MNIST, SVHN, RESISC45) and Flan-T5-base on seven GLUE tasks. They compare three paradigms: non-linear fine-tuning, NTK (full-model) linearisation, and attention-only fine-tuning. The kernel-behaviour test compares f(x; θ0+τ) with its linearisation f_lin. The scaling coefficient α is tuned on held-out data following Ilharco et al. (2023). The paper also introduces a formal definition of weight disentanglement for the representation module (Definition 2), which differs from Ortiz-Jimenez et al. (2024). Experiments ran on four RTX 3090s, under 150 GPU-hours.

## Relevance to Proposal
**Score: 4**

This is a related strand (3 in the proposal: task arithmetic, weight disentanglement, linearised/NTK fine-tuning). It supports the idea that restricting updates to a sub-module can keep them in an approximately linear regime, which makes updates add cleanly. It does not address parameter decompositions, component-level attribution, or commuting an update through a decomposition, so its direct bearing on the core question is limited.

## Limitations
- Parameter decompositions are not studied. The "kernel behaviour" and disentanglement concepts are never connected to rank-1 subcomponents or causal importance.
- Evidence is almost entirely image classification with CLIP ViTs and GLUE classification. There are no generative LLMs and no RL or preference post-training.
- The kernel-behaviour test is approximate (post-hoc linearisation, a single first-order expansion) and is judged by agreement of accuracies, not by a direct measure of how linear the outputs are.
- Task arithmetic needs identical architectures and initialisations. The authors say so directly (Sec. 6).
- Weight disentanglement is redefined relative to the original work (their Definition 2). The disentanglement error is a KL-based output metric estimated on 2,048 test points per dataset (Appendix A), so the central claims depend on that metric choice.
- The efficiency comparison is against NTK linearisation under the authors' own setup. The "a quarter of the costs" framing is theirs and was not independently checked.
- Results depend on α tuning and on selection using held-out training data (Appendix A).
- The paper reports results for one model family and one benchmark suite (CLIP ViT image tasks, plus GLUE for Flan-T5-base). It does not test larger models, LoRA-style updates, or RL-style updates.

## Key Quotes
> "Our study reveals that the attention modules exhibit kernel behavior, and fine-tuning the attention modules only significantly improves weight disentanglement." (Abstract)

> "if a pretrained network f(·; θ0) exhibits kernel behavior during fine-tuning – that is, if the neural network updates primarily around its pretrained parameter initialization and can be approximated by its first-order Taylor expansion – then the resulting task vectors are weight disentangled." (Sec. 3.2)

> "Current task vectors are constrained to models with identical architectures and initializations due to their reliance on element-wise weight operations." (Sec. 6, Discussion)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
