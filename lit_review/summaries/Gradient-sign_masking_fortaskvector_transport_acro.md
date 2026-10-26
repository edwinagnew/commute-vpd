# Gradient-Sign Masking for Task Vector Transport Across Pre-trained Models

## Metadata
- **Authors**: Filippo Rinaldi, Aniello Panariello, Giacomo Salici, Fengyuan Liu, Marco Ciccone, Angelo Porrello, Simone Calderara
- **Year**: 2026
- **Source**: ICLR 2026 (conference paper)
- **DOI/URL**: Code at https://github.com/fillo-rinaldi/GradFix (no DOI or arXiv ID in the provided text)

## Summary
The paper proposes GradFix, a way to move a fine-tuning update (task vector) from one pre-trained model A to a different pre-trained model B without fine-tuning B. The source task vector is masked coordinate-wise by the sign of a few target-model gradients (estimated by majority vote over a handful of labeled examples), keeping only components whose sign agrees with the target's anti-gradient. The authors give a first-order argument that the masked update decreases the target loss, and report gains over naive task-vector addition and few-shot fine-tuning on CLIP ViT and T5 transfer tasks and in merging pipelines.

## Key Findings
- Naive task-vector addition across different pre-trainings (CLIP Datacomp-XL to LAION-2B; T5v1.1 to FLAN-T5) performs at about zero-shot level (e.g., ViT-B/16 average 54.78 vs zero-shot 54.45; fine-tune 94.73).
- Masking the source task vector with the signs of a single target-gradient step (one sample per class) recovers a substantial part of the gap: ViT-B/16 average 64.45 (1 shot/class), 67.57 (5 shots/class); T5 average 58.48 at 50 shots/class versus 42.12 for naive addition.
- Under the stated protocol, the masked transport beats a few-shot fine-tuned target (theta_B^opt) in most ViT cells, with lower variance across seeds. Gaps to full fine-tuning remain large (e.g., ViT-B/16 average 67.6 vs 94.7; T5 58.5 vs 89.4).
- Sign information matters, magnitude information does not help: a magnitude-scaled mask underperforms sign agreement even in the oracle setting, and a random-sign mask stays near zero-shot. A random source vector with matched mean/std under the same mask also stays near zero-shot (Table 7).
- Majority-vote sign estimation is more stable across the scaling coefficient alpha than mean-gradient sign estimation (Fig. 2).
- In merging, applying the consensus mask after merging ("merge-then-mask") works best for multi-task transport; in multi-source transport, "mask-then-merge" recovers performance that direct merging loses.

## Methodology
- **Oracle analysis**: if the target were fine-tuned by one full-batch GD step, its task vector would be proportional to the negative gradient, so the gradient-sign mask equals an "oracle" mask built from the true target task vector.
- **Method**: m_i = 1{sign(tau_A,i) = sign of estimated anti-gradient}; delta_A = alpha (m ⊙ tau_A); theta_B + delta_A. Anti-gradient signs come from majority vote over N per-sample gradient signs, with a Hoeffding-based lemma (Appendix A) giving exponential concentration under an assumed per-sample success probability p_i > 1/2 and independence.
- **Theory**: first-order Taylor argument that g^T delta_A <= 0 for sufficiently small alpha, so the update is a descent direction on the target loss (for the sampled objective).
- **Experiments**: CLIP ViT-B/16 and ViT-L/14 (Open-CLIP; Datacomp-XL vs LAION-2B checkpoints) on 5 vision datasets (EuroSAT, SVHN, GTSRB, RESISC45, DTD); T5v1.1 vs FLAN-T5-base on 5 NLP classification tasks (SNLI, MNLI, RTE, QNLI, SciTail). Source task vectors come from 2000-step fine-tuning. Alpha is tuned on validation data. Ablations cover mask strategies (sign agreement, sign forcing, magnitude-scaled, random), subset selection (random, herding, k-medoids, coreset), a random-source-vector control, and a FLOP estimate (about 8P for GradFix vs 16P for one-step fine-tuning).

## Relevance to Proposal
**Score: 4**

The paper is about moving a small parameter update between two different pre-trained models using gradient information from the target, and it argues that the target gradient's sign structure, not its magnitude, is what determines which components of a source update transfer. This bears on the proposal's question of whether a small post-training update delta W can be pulled back through a parameter decomposition in a way that stays faithful, and the sign-vs-magnitude finding is a useful datapoint for any map Phi(D, delta W) that filters components. The relevance is limited because transport here is across different pre-trainings in coordinate space, with no decomposition, no subcomponent basis, and no same-model continuity along training; the mask is elementwise and does not respect the U_c V_c^T structure.

## Limitations
- The first-order descent guarantee holds for the sampled objective at small alpha. It does not establish generalization to the task, and the paper does not report the loss decrease itself.
- The majority-vote lemma relies on strong assumptions (independent per-sample gradients, symmetric noise, p_i > 1/2 per coordinate); the paper acknowledges "mild independence assumptions" without testing them.
- The oracle equivalence (task vector proportional to negative gradient) holds only for one full-batch GD step on the target, which is not the actual fine-tuning protocol (2000 AdamW steps).
- Scale is small: CLIP ViT-B/16 and ViT-L/14 and T5-base. No LLM-scale or modern post-training (RL, instruction tuning with large data) experiments.
- Gap to full fine-tuning remains large on most datasets; the authors frame the method as partial closure.
- Alpha is tuned on validation data per dataset, which the paper does not treat as part of the supervision budget in the headline comparisons.
- Tasks are closed-vocabulary classification, with the CLIP text encoder frozen; results on generative or open-ended behavior are not reported.
- Several table cells have large std across seeds (e.g., ViT-B/16 average ±5 points, dominated by dataset differences), and the paper reports the std on the averages mainly for the appendix tables.
- The "structural information in the source vector" claim rests on a single random-vector control; the paper does not test other controls (e.g., mismatched-task vectors).

## Key Quotes
> "In this work, we show that successful transfer depends strongly on the gradient-sign structure of the new model." (Abstract)

> "In the idealized setting where B is fine-tuned using full-batch gradient descent for a single epoch, the resulting task vector τB is proportional to −g, so the gradient-sign mask coincides with the oracle." (Section 4.1)

> "This approach assigns a mask value near 1.0 only when components match in sign and possess significant magnitude... this magnitude-aware strategy consistently underperforms sign agreement. Crucially, this failure persists even in the oracle setting (δ⋆), which suggests that while directions are transferable across different pretrainings, parameter magnitudes are highly specific to the local loss geometry of each basin." (Section 5.3)

> "Thus, effective transfer relies strictly on the precise geometric alignment provided by the target anti-gradients, rather than just the sparsity of the update." (Section 5.3)

## Comments Summary (if LessWrong/AF post)
Not applicable (conference paper).
