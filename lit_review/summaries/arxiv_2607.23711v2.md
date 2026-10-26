# The Intruder Threshold: A Spectral Law for LoRA Fine-Tuning

## Metadata
- **Authors**: Peng Xie, Amr Alanwar (TUM School of Computation, Information and Technology, Heilbronn)
- **Year**: 2026 (arXiv 2607.23711, v2)
- **Source**: arXiv preprint (not peer-reviewed in the text provided)
- **DOI/URL**: https://arxiv.org/abs/2607.23711

## Summary
The paper derives a per-layer critical update strength s* = θ̄ / (γ σ₁(BA)) at which a LoRA update creates an "intruder dimension" (a new leading singular vector of W + γBA that is nearly orthogonal to all pretrained singular vectors). θ̄ is computed from the measured spectrum of the pretrained W alone, via a rectangular spiked-deformation transform, with no fitted parameters beyond declared choices (evaluation point ε and edge convention). Across 18 adapters on seven base models plus six third-party adapters, it localizes the empirical threshold within a factor of two on about 82% of layers (98% with a combined edge), and it is used to explain forgetting onset and to propose a projection-free per-layer scaling rule that reduces forgetting by 62% on one fragile model.

## Key Findings
- An exact secular-equation characterization of the updated spectrum (rank-r update via a (p+r)-dimensional core matrix) lets intruder diagnostics be computed exactly from one small SVD per scan point; verified in float64 to about 1e-3 relative error.
- Under a "free-position" assumption (update directions generic relative to the pretrained basis), the threshold reduces to a rectangular spiked-deformation condition evaluated on the empirical spectrum. Pooled over 9,840 layer scans, hit@×2 = 0.822 (95% CI [0.785, 0.862]) and Spearman ρ = 0.661; the law also separates intruder-bearing from intruder-free layers with mean AUC 0.89.
- The empirical intruder threshold is systematically early by a factor of about 1.2–1.5 relative to the free-position prediction (sign agrees on 22 of 24 adapters). The authors attribute most of this to alignment structure of the trained update and multi-spike order statistics; a random-sign cross-term was refuted as a mechanism.
- Forgetting (WikiText-2 perplexity knee, defined as 1.5x base) occurs near the median threshold: knee-to-threshold Spearman 0.86 over 16 probed adapters, with knees at 0.8–2.3x the median threshold.
- Full fine-tuning spreads its update over several hundred directions (participation ratio ~385 for Qwen2.5-7B, ~345 for Llama-3.1-8B), so its largest orthogonal spike sits 100–200x below threshold on every layer; LoRA at matched task performance concentrates comparable energy into at most r directions and crosses it. The authors present this as the mechanism behind the LoRA vs full-FT asymmetry.
- Norm-matched interventions: scaling only predicted supercritical layers on Mistral r=256 raises PPL from 4.9 to 9,340, versus 39 for norm-matched subcritical layers (240x asymmetry at equal injected norm). In a synthetic surgery, concentrated updates in generic position create intruders, while top-aligned or diffuse updates do not; the authors conclude that a single statistic (supercritical strength in generic position) governs intruder formation.
- A spike-budget rule (cap per-layer alpha so the end-of-training spike sits at β·θ̄) with β = 0.7 cuts forgetting on Mistral r=256 from +4.71 to +1.77 PPL while keeping task perplexity and GSM8K roughly unchanged. A uniform-alpha control performs equivalently on accuracy metrics, so the authors say the decisive gain is the global scale the theory sets.

## Methodology
- Linear-algebra and random-matrix theory: thin-SVD core-matrix updating (Zha and Simon; Brand), a secular equation (Bunch et al.), and the rectangular spiked-deformation transform of Benaych-Georges and Nadakuditi applied to the empirical measure of each pretrained weight matrix. A sensitivity ratio over the evaluation point ε is reported for each layer as a robustness check.
- Pre-specified (internally; no external registry) validation: census of 868 block matrices on four base models; falsifier of LoRA adapters on several base models (the pre-specified protocol was 8 adapters on two models with MetaMathQA and alpaca-cleaned, r in {16, 256}, alpha = 2r, 1,000 steps; the reported pool is 18 adapters across seven base models, 9,840 layer scans); 25-point strength scans from the deployed point out to two decades; intruder criterion of Shuttleworth et al. (max cosine < 0.5 against pretrained singular vectors, top-10).
- Second batch of 59 units (more base models, rank and learning-rate sweeps, rsLoRA, DoRA, tasks, seed replicates, six external adapters, SSM and MoE models) with pre-assigned expected-outcome labels, reported in an appendix.
- Causal interventions (norm-matched layer scaling) and a synthetic "surgery" design, plus a spike-budget application compared against uniform-alpha and learning-rate controls.
- Compute: roughly 300 H100/H200 GPU-hours.

## Relevance to Proposal
**Score: 4**

The paper does not touch parameter decomposition (VPD/SPD/APD) or component-level faithfulness, so it does not answer the core question directly. It is relevant to strand 3 (spectrum and alignment of δW with pretrained singular vectors, LoRA geometry) and to the "create vs strengthen" framing of the core question: an intruder is a new direction nearly orthogonal to the pretrained basis, and the paper gives a quantitative, weight-spectrum-level criterion for when a low-rank update produces such directions versus when it stays aligned with existing structure. The perturbation-theory machinery (secular equation, spiked-matrix transform for low-rank updates) is a possible tool for the "commuting" step's analysis of W + δW, though it is single-matrix and not decomposition-aware.

## Limitations
- Not a parameter-decomposition paper: no subcomponents, no causal-importance function, no faithfulness metric; any link to Φ(D, δW) would have to be built.
- The free-position law is a probabilistic approximation; the authors' own deterministic Wedin rotation control was vacuous on all 9,840 scans, so the rigorous perturbation route failed for these spectra.
- The anti-alignment early-arrival bias is only partly explained (two computable corrections close most of the gap); the residual is described as "within scan-grid resolution" but the mechanism is not fully attributed.
- Free parameters are not fully absent: evaluation point ε and the edge convention (full vs bulk vs geometric mean) are declared choices; the geometric-mean edge was chosen on the authors' own adapters, and the 0.98 headline figure depends on it. Its out-of-bag check on six external adapters is encouraging but small.
- Mostly one LoRA recipe (alpha = 2r, 1,000 steps, single seed per configuration apart from two stability checks); five of seven base models are evaluated on a single task family.
- Results are for weight-spectrum intruders as defined by one criterion (Shuttleworth et al., tau and top-k inherited); localization peaks at stricter definitions but the paper's headline depends on tau in [0.25, 0.70].
- Pre-registration was internal (fixed in analysis code before the runs), not on an external registry.
- Causal claims about forgetting rest on perplexity and a few downstream metrics on a limited set of models; the budget rule is described by the authors as a consequence of the law, not a competitive mitigation method.
- Preprint; the article text provided is v2 and has not been checked against peer review.

## Key Quotes
> "LoRA fine-tuning can create intruder dimensions: new leading singular vectors of the updated weight matrix W + BA that are nearly orthogonal to all pretrained singular vectors and that drive catastrophic forgetting." (Abstract)

> "Full fine-tuning disperses its update far below the threshold of every layer, which resolves the asymmetry between LoRA and full fine-tuning." (Abstract)

> "Intruder formation is governed by a single statistic, supercritical strength in generic position, which simultaneously explains the immunity of full fine-tuning (diffusion) and of aligned LoRA updates." (Section 4, "Surgery: the sufficient statistic, isolated")

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint; no comments provided).
