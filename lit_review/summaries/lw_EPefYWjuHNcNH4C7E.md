# Attribution-based parameter decomposition

## Metadata
- **Authors**: Lucius Bushnaq (linkpost author). The arXiv paper (2501.14926) is by Apollo Research; the post itself names only Bushnaq. Co-authors are not listed in the file.
- **Year**: 2025 (posted 2025-01-25)
- **Source**: LessWrong linkpost (score 109), for Apollo Research paper "Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition"
- **DOI/URL**: https://www.lesswrong.com/posts/EPefYWjuHNcNH4C7E/attribution-based-parameter-decomposition ; paper: https://arxiv.org/abs/2501.14926

## Summary
The post introduces Attribution-based Parameter Decomposition (APD), a method that splits a network's weight vector directly into a sum of parameter components, each meant to be a mechanism, rather than decomposing activations into features first. The components are chosen so that the average description length of the mechanisms used on any one input is minimised, with components allowed to span layers and act on multi-dimensional features. The method is shown only on toy models, and the authors say it is sensitive to hyperparameters and has imperfections.

## Key Findings
- Direct parameter-space decomposition is proposed as an alternative to layer-by-layer feature decomposition plus circuit assembly. The motivation is that feature splitting, multi-dimensional features and cross-layer representations are hard for activation-based methods.
- The objective trades off faithfulness (components sum to the original weights), minimality (few components active per input, judged by attribution), and simplicity (each component is low-rank / cheap to describe).
- Components need not match architectural units (neurons, layers, heads). A single mechanism can spread across many neurons and layers via cross-layer superposition.
- The authors report that on toy models APD handles superposition and cross-layer representations "mostly the way we hoped", with imperfections and hyperparameter sensitivity (the authors' own statement, in the post's introduction).
- In the comments, Bushnaq argues that the "activity" definition in Appendix A.1 (a component counts as inactive only if the output is unchanged along all monotonic trajectories) limits how far learned masks can hide computation, so mask-based attribution is less of a risk than for circuits or transcoders.

## Methodology
Toy-model experiments with a learned decomposition of the weight vector into C parameter components P_c. Each component has a learned causal-importance / attribution per input, optimised with a loss combining faithfulness, minimality and a simplicity penalty, with the description-length motivation given in an appendix (A.2). The paper itself is not in the file; the post is a linkpost with motivation, a short method description, figures and a comment thread. The method details here come from the post text, not from the paper.

## Relevance to Proposal
**Score: 8**

This is the APD precursor of VPD, the parameter decomposition the proposal builds on: its rank-1-style components and the attribution-gated sparse forward pass are the objects the proposed commuting map would act on. The comment thread on whether the MDL argument needs rank-1 pieces to be encoded separately (Semendinger / Bushnaq) bears directly on whether a decomposition carries structure at the rank-1 level, which the pullback map assumes.

## Limitations
- Only toy models were studied in this post; no results on trained LLMs appear in the file.
- Authors state the method is quite sensitive to hyperparameters and has some imperfections.
- The MDL-based justification for the simplicity term has an open question about why the cost scales with the rank of the parameter component (Semendinger's comment, not resolved in thread). Bushnaq says "I'm brooding over variations of this" and that the question is open.
- Faithfulness, minimality and simplicity are defined in the paper's terms; this summary has not checked the paper's full derivations.
- Stability across seeds and identifiability are not addressed in the post (the proposal's strand 1 asks about these).

## Key Quotes
> "Instead, we think it might be better to directly decompose a network's parameters into components by parametrising each mechanism individually." (Introduction, Motivation)

> "Our method decomposes the network parameter vector into a sum of parameter component vectors, such that the average description length of the mechanisms used on any one data point across the training dataset is minimised." (Section "What we do")

> "It mostly worked the way we hoped it would, though it's currently quite sensitive to hyperparameters and the results have some imperfections." (Introduction)

> "even when there is an overcomplete basis of features in activation space, circuits in superposition math and information theory both suggest that you *can't* have an overcomplete basis of *mechanisms* in parameter space." (Lucius Bushnaq, reply to Oliver Daniels, comments)

## Comments Summary
- **Oliver Daniels (score 16)**: called it the most exciting mech-interp work since SAEs. Asked whether learned masks can produce mechanisms that do not match the model's actual mechanisms (citing InterpBench) and whether zero ablation could be replaced by sampling from the initialisation distribution. Proposed surprise accounting / mechanistic anomaly detection as an application.
- **Bushnaq's reply to Daniels**: the activity definition in Appendix A.1 constrains masks, so the sparse forward pass cannot hide useful computation. Argued APD does not have a transcoder-style feature-splitting problem, because the component must compose linearly to match the original weights.
- **Alex Semendinger (scores 10, 3)**: questioned why the rank penalty is linear in alpha and why the MDL bit cost has 2α rather than α for rank-1 pairs. Bushnaq agreed the intent is to encode each rank-1 piece separately, but said this is not how the attributions are computed (attributions are taken for the whole P_c). He called this "unsatisfying" and said the question is open.
- **mbissell**: asked whether MoE models are a coarse parameter decomposition. Bushnaq: partly, but experts are fixed in size and gated, while APD components vary in size and can all be run.
- **Aidan Ewart (score 4)**: asked why the task-arithmetic line of work (Ilharco et al.) was not cited. Lee Sharkey: an edge case, relevant to weight-space linearity but not to decomposing a single model's weights; related work may merit a weight-linearity section.
- **Kriz Tahimic**: welcomed renaming "circuit" to "mechanism".
- **Matt Levinson**: asked how the C components are initialised (random partition of the weights vs. division by C) and whether informed priors could seed candidate components. Bushnaq did not give a full answer in the thread.
- **Lee Sharkey**: welcomed use for mechanistic anomaly detection and boolean-circuit toy models, which were considered but not used.
