# Intrinsic Structure: Spectral Identifiability for Mechanistic Interpretability

## Metadata
- **Authors**: Ashim Dhor (IISER Bhopal), Pin-Yu Chen (IBM Research)
- **Year**: 2026
- **Source**: arXiv preprint (arXiv:2608.10172); no peer-review venue stated in the file
- **DOI/URL**: arXiv:2608.10172 (https://arxiv.org/abs/2608.10172)

## Summary
The paper treats a transformer's forward pass as a controlled dynamical system with depth as time, lifts it with the Koopman operator, and argues that the spectrum of the resulting finite linear realisation is a coordinate-free, identifiable property of the model. It proves a finite-sample rate (M^-1/2, up to permutation) for recovering that spectrum, plus a minimax lower bound, and reports measurements on GPT-2 small, Gemma-2-2B and Qwen3-8B-Base. Its central self-described claim is that the identifiable object (the Koopman spectrum) and the human-legible object (features or circuits) are not the same object.

## Key Findings
- Under stated assumptions (dictionary Koopman-invariance, persistent excitation, spectral separation), the EDMDc estimate of the realisation's spectrum converges at rate M^-1/2 up to permutation, with a matching minimax lower bound (Theorems 6.1, 7.1). These are the authors' theorems; the paper states they are the first identifiability result for an MI primitive.
- Measured convergence on Qwen3-8B-Base gives exponent -0.506 +/- 0.031 against the predicted -0.5. The paper reports that shortfalls on other models collapse onto one curve against each cell's sample threshold.
- Koopman modes beat random directions but lose to principal components at predicting IOI ablation effects, with the gap shrinking 4.1x with depth distance. The authors read this as a consequence of non-normality (Theorem 8.1).
- The paper's own measurements bound its claims: residualising the control against the lifted state moves the spectrum by 5-9x the resolution floor; the SAE invariance gap reverses sign under variance-based feature selection; and the universality criterion declares two seed replicas of one architecture distinct.
- An invariance penalty on SAEs reduces split-half spectral distance by 41% (TopK) at matched sparsity, while degrading reconstruction and cross-seed agreement.

## Methodology
Theory: existence and uniqueness of the Koopman realisation under dictionary invariance, finite-sample analysis via matrix concentration, Bauer-Fike and Davis-Kahan perturbation bounds, a minimax lower bound, and a median-of-means variant for heavy tails. Empirics: per-layer EDMDc fits on residual-stream states and attention-write controls from 10^5 to 10^6 calibration samples. Dictionaries are whitened SAE-style or random-Fourier feature sets, and comparisons are made against SAE encoders, random directions and principal components. Evaluation is on three pretrained transformers plus an IOI circuit testbed.

## Relevance to Proposal
**Score: 3**

The paper does not address parameter decompositions (VPD/SPD/APD), weight-space rank-1 subcomponents, or how fine-tuning updates δW interact with them; its object is a depth-wise activation-space linear realisation. The only overlap is strand 1's identifiability/stability question (it is an identifiability argument for a decomposition-like primitive, with a SAE nonidentifiability critique that is loosely adjacent) and strand 6's mechanism-stability question, but both are in activation space and framed around a single model, not across a fine-tuning step.

## Limitations
- Authors' own list: the estimand depends on the control-exogeneity convention (naive vs residualised give different spectra); invariance holds only approximately, and the bias does not vanish with more data; the dictionary must be co-designed with the architecture, so KSA is not model-agnostic.
- Assumption 3 (spectral separation) can fail, and the exact-coincidence case is conjectured to be measure-zero but not proved.
- The intervention calculus (Theorem 9.2) is proved only for first-order interventions and is not validated against measured patching effects.
- The rate is asymptotic in practice: the constant c_0 is unknown, so the sample threshold M_0 is only an order-of-magnitude guide.
- Two pre-registered criteria fail their controls (SAE-gap criterion met in 49% of cells, not 80%; universality criterion separates seed replicas). The authors report this openly.
- The main claim ("identifiable object is not the legible object") is a self-assessed interpretation of a non-normality argument, not a finding about human-legible mechanisms. The "first identifiability theorem" priority claim is the authors' own and was not checked against the literature.
- Only three models; no fine-tuned or post-trained models were analysed, so nothing in the paper speaks to the training-dynamics question the proposal asks.

## Key Quotes
> "The identifiable object and the legible object are not the same object." (Abstract)

> "The Koopman spectrum is an identifiable, model-intrinsic fingerprint with a stated error bar, not a legible decomposition." (Abstract)

> "The spectrum therefore certifies an intrinsic, identifiable model property recoverable at a stated rate; it does not certify interpretability, behavioural coverage, or robustness." (Section 12, Discussion)

## Comments Summary (if LessWrong/AF post)
Not applicable: arXiv preprint with no comment section.
