# Cross-seed explainability using Procrustes conditioned Joint End-to-end Top-K Sparse Autoencoders

## Metadata
- **Authors**: Bendegúz Váradi, Zoltán Kmetty (Centre for Social Sciences, CSS-RECENS, and Eötvös Loránd University, Budapest)
- **Year**: 2026 (inferred from arXiv ID 2607; not stated in the text)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2607.08499 (URL inferred from the arXiv ID; the paper text gives no link)

## Summary
The paper proposes a single joint Top-K sparse autoencoder (SAE), trained on two BERT seeds whose layer-10 activations are first aligned with an orthogonal Procrustes rotation, plus an end-to-end downstream loss and a cross-seed sparse-code loss. The aim is to find features that are "universal" (per-token activation correlation r >= 0.70) across independently trained BERT models. On five seed pairs and three classification datasets (SST-2, Stanford Politeness, TweetEval Emotion), the authors report higher universality and top-N correlations than post-hoc matched independent SAEs.

## Key Findings
- Independent SAEs with raw dictionary indices have essentially zero cross-seed correlation. Hungarian matching of independent SAEs already gives top-10 r of about 0.52-0.65 and 5-27% "universal" features, so the matched baseline is strong.
- Post-hoc Procrustes applied to independent SAEs gives no consistent benefit. The rotation helps only when used to condition joint training: "rotation only" beats "joint, no rotation" on all three datasets (Table 1).
- The full pipeline (rotation plus cross-seed loss, lambda_cross = 1) is best on most columns, but the authors say the cross-loss gain over rotation-only is "minor". The cross-seed loss directly optimises sparse-code agreement, which the authors acknowledge is a Goodhart-style concern.
- A permutation-style control (30 random orthogonal rotations, plus identity and mismatched-correspondence rotations) suggests the true rotation yields more universal features. The reported Monte Carlo p is about 0.032, the floor for 30 rotations. Appendix D Table D4 is a single small run with the cross loss off.
- Alignment quality (1 - normalised residual error) explains about 70% of the variance in universal-feature count across rotation conditions (R^2 = 0.704).
- Accuracy change from the SAE is small (mostly below 0.5 percentage points). One qualitative example, a hedged-request feature (ID 389) on the politeness model, is probed with hand-written minimal edits.

## Methodology
- Models: five pairs of bert-base-uncased fine-tuned with different seeds (ten models). Layer 10 of 12 is the injection point.
- Alignment: an orthogonal Procrustes rotation W_align = V U^T from the SVD of H_A^T H_B, fit on 500 sequences (about 5k-15k tokens). The authors check sample size with a split-half-style stability test.
- SAE: TopK (k = 32, dictionary 6144), auxiliary AuxK dead-feature loss (no ReLU gating, k_aux = 512), and an end-to-end KL + downstream MSE loss (lambda_DS = 1.5), with a cross-seed sparse-code MSE (lambda_cross = 1.0, grid-searched over {0.1, 0.5, 1, 2}).
- Evaluation: per-token activation Pearson correlation of matched features on the evaluation set. Universality is r >= 0.70 for features active in both models with at least 10 samples. Dead-feature fraction and accuracy change are also reported. Mean-pooled evaluation (Appendix C) gives many more universal features, which the authors attribute to pooling making features fire anywhere in the sequence.
- Compute: one RTX 3090. Seeds fixed to 42; mixed precision.

## Relevance to Proposal
**Score: 3**

The paper concerns cross-seed identifiability of dictionary features and aligning two models' latent spaces, which touches the proposal's "component matching across runs" and "identifiability" strand (strand 1) and the "model stitching / universality" strand (strand 6). It does not touch parameter decomposition (VPD/SPD/APD), fine-tuning updates, or any commuting of a weight update past a decomposition, and it uses SAEs on activations, not weight-space rank-1 components. The most transferable idea is that Procrustes-style alignment of activation spaces across seeds is a cheap way to make decomposition features comparable, and that a post-hoc matched baseline is a necessary control.

## Limitations
- Only one model family (BERT-base) and one layer (10 of 12). The authors say the results may not transfer to other families or layers, and that they would need more seeds and model types to show robustness.
- The comparison to Feature-Aligned SAEs and Orthogonal SAEs is left for future work, so the claimed advance over the baseline is only partly tested.
- The "full" pipeline's cross-seed loss directly optimises the sparse-code agreement that the universality metric measures. The authors argue the rotation-only ablation addresses this, but the gap between rotation-only and full is small relative to the stds.
- Universality is defined by a single correlation threshold (0.70) on token-level activations. Results change substantially with the evaluation protocol (mean-pooling, Appendix C), so the headline counts are sensitive to the metric.
- Only 5 seed pairs (ten models) for the main results. The Appendix D control table uses one seed pair per dataset with cross loss set to 0 and only 35 batches. The Table D4 "Mismatched" column is poorly reported in the source text.
- Interpretation is anecdotal: one feature, with hand-chosen counterfactuals and no human or automated interpretability score across features.
- Dead-feature rates are high (57-95% of the dictionary in several conditions), so only a small fraction of the dictionary is used.
- Accuracy-preservation numbers are reported as "the black box classification accuracy" with small changes, but there is no sensitivity analysis on k or dictionary size.
- The authors state that the Goodhart concern is addressed only by the rotation-only ablation.

## Key Quotes
> "Procrustes alignment is effective specifically when it conditions joint training rather than as a standalone transform." (Section 5, Results)

> "Because dictionary learning is non-convex, independently trained networks, even those sharing the same architecture and data, suffer from feature splitting, where identical semantic concepts map to entirely different latent dimensions." (Section 1, Introduction)

> "We note that any downstream effect on feature interpretability is supported only by anecdotal observation, and a more rigorous evaluation would be required to prove it" (Section 3.2, Model Training)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
