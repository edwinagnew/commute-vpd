# Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition

## Metadata
- **Authors**: Dan Braun, Lucius Bushnaq, Stefan Heimersheim, Jake Mendel, Lee Sharkey (Apollo Research)
- **Year**: 2025 (arXiv 2501.14926, v4; the version date is not stated in the text)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2501.14926

## Summary
The paper introduces Attribution-based Parameter Decomposition (APD), which splits a network's flattened weight vector into a sum of parameter components. The components are trained to be faithful (they sum to the original weights), minimal (only a few are active per input, chosen by gradient attribution and top-k), and simple (low-rank, penalised by a Schatten-p norm). The authors show on three toy models with known ground-truth mechanisms (toy superposition, compressed computation, and cross-layer distributed computation) that APD recovers the expected components, but they state that the method is not yet practical for real models.

## Key Findings
- On the Toy Model of Superposition (TMS5-2 and TMS40-10), APD components match the ground-truth rank-1 matrices Z^(c) closely: mean max cosine similarity is 0.998 and 0.996. Mean L2 ratio is 0.893 and 0.935, which the authors read as "shrinkage" (Table 1, Section 3.1).
- On a compressed-computation model that computes 100 ReLU-type functions with 50 neurons, APD learns one component per input-feature computation. A scrubbing test shows that ablating unrelated components barely changes the MSE, while ablating the responsible components raises it sharply (Figures 6-8, Section 3.2). The authors note that some components remain partly bimodal and represent secondary features.
- The same picture holds for a two-layer variant, with more imperfections (more components that serve two features), and the components span both MLP layers (Section 3.3).
- The paper argues conceptually that a parameter-first decomposition gives a route to minimal circuits in superposition, to a definition of "features" as input properties that activate mechanisms, and to an architecture-agnostic decomposition (Section 4.1).
- The authors state directly that APD is "currently only appropriate for studying toy models" because of computational cost (each component has roughly the memory of the full model) and hyperparameter sensitivity (Section 4.2).

## Methodology
- **Objective**: three losses. (i) Faithfulness: MSE between the summed components and the target weights. (ii) Minimality: a KL/MSE-type distance between the target output and the output of the batch top-k most attributed components, with the attributions taken as gradient attributions. (iii) Simplicity: a Schatten-p quasi-norm of singular values of active components (Section 2.2).
- **Training**: a two-step loop per batch. First, one forward and backward pass computes gradient attributions per component. Second, a sparse forward pass uses only the top-k components.
- **Toy models**: TMS (Elhage et al. 2022); a residual MLP with 100 input features and 50 neurons (compressed computation); a two-layer variant (cross-layer). Target models are trained with AdamW, and a hand-coded model is studied in Appendix B. Full hyperparameters are in Appendix D (not read in detail for this summary).
- **Evaluation**: MMCS and ML2R against ground-truth columns; scrubbed vs anti-scrubbed ablations on the training distribution.

## Relevance to Proposal
**Score: 7**

APD is the direct predecessor of the VPD method the proposal builds on. It is the first clear statement of the faithful, minimal, simple decomposition objective and of gradient-attribution-based causal importance, so it is the reference for the decomposition side of the commuting question. It does not study fine-tuning, updates or training dynamics, so it does not answer whether a small δW can be carried by existing components.

## Limitations
- Results are on toy models with known ground truth. The authors state that scaling to non-toy models is an open problem (Sections 4, 5).
- Hyperparameter sensitivity is high. The authors say extensive tuning was often needed, and that TMS5-2 was "very brittle" (Section 3.1). The 2-D TMS result needed a special attribution adjustment.
- Gradient attributions are a first-order approximation of causal ablation, and become unreliable when gradients saturate (Section 4.2.3).
- Layer-privileging: the rank-based simplicity loss is not invariant to reparametrisation across layers (Section 4.2.4).
- Shrinkage in the components (ML2R below 1) is attributed to a trade-off between the top-k reconstruction loss and the Schatten penalty. This is a speculation, not a tested cause (Section 3.1).
- Biases are not decomposed; the embedding and unembedding matrices are not decomposed in the compressed computation models.
- The authors say the difference between "compressed computation" and "computation in superposition" was not established (Section 3.2).
- No seed-variance, stability or identifiability study is reported in the main text. Appendix C was not analysed for this summary.
- The paper is a preprint by the method's authors. The claims about "features" and superposition are conceptual arguments, not experimental results.

## Key Quotes
> "APD is currently only appropriate for studying toy models because of its computational cost and hyperparameter sensitivity." (Section 4.2, p. 18)

> "Parameter components are better thought of as 'steps in the neural network's algorithm', rather than 'representations of properties of the input'." (Section 4.1, p. 17)

> "The losses cannot be further reduced by adding more components, because that would prevent the components from summing up to the original network weights." (Section 4.1, p. 17)

> "However, training an APD model (and to a lesser extent, a target model) in this setting is very brittle and less effective than a higher-dimensional setting." (Section 3.1, p. 9)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv preprint. Acknowledgements thank several reviewers, but no comment thread is included in the source.
