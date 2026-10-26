# Stable and Steerable Sparse Autoencoders with Weight Regularization

## Metadata
- **Authors**: Piotr Jedryszek (University of Oxford, Dept. of Biology); Oliver M. Crook (Kavli Institute for Nanoscience Discovery and Dept. of Chemistry, University of Oxford)
- **Year**: 2026
- **Source**: arXiv:2603.04198v2; Mechanistic Interpretability Workshop at the 43rd International Conference on Machine Learning (ICML 2026), Seoul
- **DOI/URL**: https://arxiv.org/abs/2603.04198 (code: https://github.com/oxPJ/sae-weight-regularization; models and results on Hugging Face, anonymised links given in the paper)

## Summary
The paper adds an explicit L1 or L2 weight penalty on SAE encoder and decoder weights, on top of the usual activation sparsity, and tests it on an MNIST toy model and on TopK, BatchTopK and Matryoshka SAEs trained on layer-3 of Pythia-70M-deduped. The authors report that L2 weight regularization (with tied initialization and unit-norm decoders) greatly increases feature agreement across random seeds, and that it roughly doubles LLM-judged steering success at one sparsity level (k=40), while most latents collapse to zero. They interpret L2 mainly as implicit effective-dictionary-size selection rather than a clean improvement in feature quality.

## Key Findings
- On MNIST, with tied initialization and unit-norm decoders, L2 regularization raised the strict "shared feature" fraction across three seeds from 1.74% (no penalty) to 22.5% among alive features (Table 1). Without those constraints, shared features were near zero.
- On Pythia-70M TopK SAEs (k in {40, 80, 150, 320}), L2 (lambda=1e-4) raised the strictly shared fraction among alive features from under 2% to about 35%, and roughly doubled mean max cosine (from at most 0.32 to about 0.7). The authors describe the gain as holding across all tested k.
- At k=40, steering success (judge score >= 4) rose from 6.3% (57/900) to 13.0% (72/555) with L2. This is conditional on features that received a valid auto-interp explanation: 185/300 L2 features had one, versus 300/300 unregularized.
- Regularization made auto-interp score a stronger predictor of steering success (Spearman r 0.060 to 0.144; slope 0.11 to 0.40), while the auto-interp score distribution itself was similar across conditions.
- L2 killed most latents: the TopK-L2 model keeps about 1,500 alive features (still about 3x the 512-dim residual stream), and most of the dictionary has zero norm (Appendix C).
- BatchTopK with L2 showed a general shift toward lower cosine similarity without the bimodal structure, so the effect is architecture-dependent. SAEBench Pareto analysis finds regularized configurations on the frontier for several architectures.
- The authors suspect that much of L2's benefit is matching an effective dictionary size, but did not run the size-matched unregularized comparison.

## Methodology
- Loss: standard SAE reconstruction plus sparsity (L1 or TopK constraint) plus lambda_w * ||W||_p on encoder and decoder weights.
- Stability metric: absolute cosine matrix between decoders of SAEs trained with different seeds, Hungarian matching, and a "shared" criterion requiring both encoder and decoder matchings to agree with similarity > 0.7. "Alive" means encoder-decoder cosine > 0.1.
- MNIST toy: 1,568 latents on flattened 784-dim images, L1 sparsity, a grid of penalty types and initializations, three seeds.
- Language model: SAEBench defaults (tied init, unit-norm decoders), a sweep over k and lambda_w, SAEBench evaluations (SP, SCR, TPP, CE recovery), 3 seeds for TopK at lambda in {0, 1e-4}.
- Steering: decoder feature added to residual stream with alpha=5, generation-only, temperature 0.7 and top-p 0.9; 300 alive features per SAE, 3 prompts each; GPT-5.1 judge on a 1-5 scale, with >= 4 counted as success. Hyperparameters were chosen by a grid search over steering settings (Appendix A).
- Decoder orthogonality: mean absolute pairwise decoder cosine (c_dec), compared to a random-vector baseline.

## Relevance to Proposal
**Score: 3**

The paper studies feature stability across seeds for SAEs, not parameter decompositions (VPD/SPD/APD), and it does not touch fine-tuning updates or commuting a delta-W through a decomposition. The relevant overlap is strand 1 of the proposal (seed dependence and component matching across runs, here via Hungarian matching of decoder directions and a shared-feature criterion), and the general question of whether weight-level regularization makes a decomposition more reproducible. Its finding that a weight penalty trades dictionary size for stability is a useful caution for any "stable decomposition" claim, but the transfer to parameter decompositions is indirect.

## Limitations
- Only one small model (Pythia-70M-deduped) and a toy MNIST setting. The authors say scaling behaviour is unknown.
- Steering was measured only at k=40, and the authors note that the steering-orthogonality mechanism is not established at higher k.
- The steering gain is conditional on features with valid explanations, and the L2 and baseline samples are not paired (feature indices are not comparable across separately trained SAEs).
- Single LLM judge (GPT-5.1) for steering; no human validation reported in the paper.
- Most latents collapse under L2 at the chosen lambda, so the stability gain may reflect dictionary-size reduction rather than better features. The size-matched control is left as future work.
- Shared-feature thresholds (cosine > 0.7) and alive threshold (0.1) are pragmatic choices; the authors claim insensitivity but report it only briefly.
- The text of the arXiv markdown has an empty Abstract section; the abstract content above is taken from the Introduction opening paragraph, which restates it.
- Limited seeds (3) for the stability comparisons.

## Key Quotes
> "Adding L2 weight regularization to SAE training increases cross-seed feature reproducibility and improves steering success, but the majority of latents collapse to zero (Appendix C)." (Section 5, Discussion)

> "the steering improvement should be read as conditional on the explained surviving features; the 115 invalid L2 explanations are themselves informative, indicating a weakly structured tier of alive but hard-to-explain features." (Section 4.4)

> "This raises a natural question: is L2 performing implicit dictionary-size selection, and would an unregularized SAE with 1,500 latents learn similar features? We suspect the answer is largely yes" (Section 5, Discussion)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is a workshop paper on arXiv, with no comment thread.
