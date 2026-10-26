# Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic

## Metadata
- **Authors**: Thomas Sommariva, Francesca Morandi, Simone Calderara, Angelo Porrello
- **Year**: 2026
- **Source**: arXiv preprint 2605.18993v2; ICML 2026 (PMLR 306)
- **DOI/URL**: https://arxiv.org/abs/2605.18993

## Summary
The paper proposes DELTA, a training method that gives a standard non-linear fine-tune the weight-space properties of linearized (tangent-space) fine-tuning. A linearized teacher (with curvature-aware regularization) is trained jointly with a non-linear student, and the student matches the teacher's last-hidden-layer activations, averaged over points sampled along the teacher's path from the pre-trained weights. The resulting task vectors compose well under addition and negation without the inference overhead of linearized models.

## Key Findings
- Matching a linearized teacher's activations (feature-level distillation) makes the non-linear student's activations approximately linear in weight perturbations. The reported linearization error is near zero for DELTA versus clearly nonzero for standard fine-tuning (Sec. 5, Fig. 4). Distillation is reported as the main driver of this effect.
- Curvature regularization (EK-FAC on a third-party reference dataset: ImageNet-21k subset for vision, C4 for text) is reported as the main driver of support localization and disentanglement (Fig. 5, Fig. 6). This is a dataless, task-agnostic regularizer, unlike earlier TAK/tau-Jp methods that need other tasks' statistics.
- On task addition, DELTA beats in-training baselines (non-linear FT, TaLoS, attention-only FT, linear FT, TAK, tau-Jp) on absolute accuracy for 8-Vision, 14-Vision and 6-NLI (Tab. 2). It stays slightly behind the best linearized methods on normalized accuracy for some settings. On the LoRA variant, it beats post-hoc merging methods (IsoC, TSV-M, Core Space) by a large margin (Tab. 4).
- On task negation, DELTA cuts target-task accuracy more than standard FT (9.6% vs 20.4% on ViT-B/32) while keeping control accuracy high. Linearized TAK and tau-Jp do better here (Tab. 3).
- The student often beats its own linearized teacher on per-task accuracy before merging (Fig. 9). The authors attribute this to the non-linear regime's extra expressivity, and describe the student as an intermediate "quasi-linear" regime. This is a claim about the student, not a verified mechanism.
- The method is preliminary on generative LLMs (DPO on LLaMA-3.2-1B for helpfulness vs verbosity, Fig. 10). The distilled variant roughly matches the Pareto front of linear DPO on reward and trails it slightly on preference accuracy.
- Training cost is about 3x time and 2x memory versus standard training (Conclusions; Sec. A).

## Methodology
- Teacher: first-order linearized model (Jacobian-vector product, Eq. 1) trained with task loss plus an EK-FAC representation-drift penalty. Student: standard non-linear fine-tune with the same initialization.
- Both are trained online and jointly. The student loss is task loss, plus MSE distillation of last-hidden-layer features against the teacher, plus an EK-FAC drift penalty on the student.
- Along-path distillation (APKD): each step samples alpha ~ U(0.5, 1) and distills against the teacher at theta_0 + alpha*tau_T, with stop-gradient on the teacher.
- Evaluation: CLIP ViT-B/32 and ViT-L/14 on 8-Vision and 14-Vision (full FT of the visual encoder); T5-Base on 6-NLI; abs. and normalized accuracy; alpha sweeps; LoRA student variant; a DPO preliminary study. Diagnostics include linearization error (Eq. 9), disentanglement error heatmaps (Fig. 2), and activation-MSE edit distance for in- vs out-of-domain data (Fig. 5).

## Relevance to Proposal
**Score: 4**

This paper bears on strand 3 of the proposal (structure of fine-tuning updates: linearised/NTK fine-tuning, task arithmetic, weight disentanglement, the sparsity and composability of task vectors). It is useful as evidence that a non-linear update can be made to behave linearly in weight space, which is adjacent to the question of whether a small post-training update can be carried through a decomposition. It does not involve parameter decompositions (VPD/SPD/APD), interpretability, or any map Φ(D, δW), so it does not directly answer the core question.

## Limitations
- The paper does not study interpretability, components, or decompositions. Its "linearity" and "disentanglement" measures are behavioural (activations, accuracy) and do not show that a mechanistic decomposition stays valid.
- Vision and NLI experiments use full fine-tuning of a single pre-trained checkpoint per family (CLIP, T5). The LLM study is small (1B model, one merge formula, preliminary).
- Training cost is 3x time and 2x memory, and the method requires a curvature (EK-FAC) precomputation on a reference dataset. The authors acknowledge this.
- The claim that the student outperforms the teacher due to expressivity is supported by the single-task comparison but not by a mechanistic test.
- The "near-zero linearization error" is measured on a finite set of perturbation scales and examples, with the alpha grid and sampling distribution chosen by the authors. The metric is blind to nonlinearity outside the sampled directions and to inputs outside the evaluation sets.
- The "support localization" result is evaluated on in-domain vs a small set of out-of-domain examples drawn from the other tasks. It does not test localization on arbitrary inputs.
- Prior-work comparisons use the authors' reproduction settings. Normalized-accuracy results are mixed (DELTA does not dominate linearized baselines on all metrics).

## Key Quotes
> "linearity with respect to weight perturbations, a property defined in parameter space, can be enforced through constraints in activation space during training." (Abstract)

> "task arithmetic does not require strict linearization, but rather localized and approximately linear update directions." (Section 1, Introduction)

> "This behavior is induced by an objective defined in _function space_ , which, perhaps surprisingly, leads to tangible effects in _parameter space_." (Section 5, Support localization)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv/ICML paper with no comment thread.
