# Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning

## Metadata
- **Authors**: Gustav Olaf Yunus Laitinen-Fredriksson Lundstrom-Imanov (single author; Linköping University, per affiliation block)
- **Year**: 2026
- **Source**: arXiv preprint, marked "Preprint submitted to Elsevier" (no venue or peer review stated)
- **DOI/URL**: arXiv:2601.18699 (https://arxiv.org/abs/2601.18699, URL inferred from the arXiv ID in the filename)

## Summary
The paper reports a comparative study of catastrophic forgetting under a three-stage sequential fine-tuning pipeline (coding, then math, then medical QA) across 20 named models, split into 10 closed-source API models and 10 open-weight models. For the open-weight models it uses CKA, routing-gate Jensen-Shannon drift, Frobenius weight drift and activation patching to localise drift to mid-layer MLP down-projections and MoE routing, and proposes "Low-Rank Circuit Projection" (LRCP), which projects gradient updates onto the orthogonal complement of a fixed SVD subspace of historical activations.

## Key Findings
- The authors report that mid-layer CKA drops to 0.412-0.612 and that MLP down-projections (W_down) show the largest Frobenius drift across the open-weight cohort (Table 2, Table 3). These are correlational measurements on a single sequential run, not causal tests.
- Activation patching (Figure 3, 500 SWE-bench samples) is said to recover Task A performance when mid-layer activations are restored, which the authors read as causal evidence for localisation. The figure shows only a recovery-rate curve, with no per-model numbers in the text.
- On Llama 4 Scout (109B), LRCP reports Task A (code) 74.8% and Task B (math) 75.1%, against 35.1% / 76.8% for standard fine-tuning and 58.4% / 70.1% for LoRA (r=16) (Table 4). These are single-model numbers with no variance or seeds reported.
- The rank-ratio sweep (Table 5) shows a trade-off: higher k/d retains more of Task A but reduces Task B accuracy (k/d = 0.30 gives Task B 69.2%).
- Closed-source models show large post-sequence accuracy drops (27.7% to 45.9% net Task A loss in Table 1), with a proxy "Semantic Output Drift" metric computed from API outputs. The authors do not establish that this proxy measures weight-level change.

## Methodology
- Behavioural and output-based analysis of 10 closed-source models via API (Table 1), using a Semantic Output Drift metric (cosine distance between sentence embeddings of sampled outputs; greedy decoding, N=5 samples).
- Weight-space and activation analysis of 10 open-weight models: linear CKA between baseline and adapted activations, Routing Gate Drift (Jensen-Shannon divergence of expert-routing probabilities), Frobenius drift per projection matrix, attention-entropy change, and activation patching. A RADL algorithm (Algorithm 1) flags layers whose CKA falls below a threshold or whose routing drift exceeds one.
- LRCP: gradient updates projected onto the orthogonal complement of the top singular subspace of historical activations X_hist, computed once on a static calibration set. Compared against standard fine-tuning, LoRA, EWC and replay on one model (Llama 4 Scout).
- Training setup stated as AdamW, constant learning rate 2e-5 with cosine decay, batch size 128. Models are not specified for each run in the text.

## Relevance to Proposal
**Score: 2**

The paper is about forgetting in sequential fine-tuning, not about parameter decompositions (VPD/SPD/APD) or about carrying a decomposition through a training update. The only point of contact is the general idea of projecting an update onto a historically important subspace (LRCP), which loosely resembles the proposal's "project δW onto subcomponents" candidate map, and its localisation of drift to MLP down-projections; neither is tested against a decomposition.

## Limitations
- The closed-source half is not reproducible as described: the text does not explain how "sequential training steps" were applied to API-only models (Claude, GPT, Gemini), and the SOD metric is an output-level proxy that does not measure weight or representation change.
- The abstract claims LRCP "mitigates up to 94.2% of ancestral capabilities", but the body reports 74.8% Task A retention on Llama 4 Scout. The abstract's 94.2% figure is not traced to any table in the text.
- Table 4 caption calls Task A "Code", while Figure 3 uses SWE-bench; the dataset behind the main metric is not stated unambiguously.
- Several cohort names (e.g., "Claude Fable 5", "Claude Mythos 5", "GPT-5.5 High", "Gemini 3.5 Flash") and a number of references could not be verified from the text alone. Some references are malformed (e.g., "bench Team, S., 2024", "Schwarz, J., Modifiers, J.L., 2018"), and several self-cited works (e.g., "Rostova 2026", "Demir 2025", "Zhai et al. 2023") could not be checked here. Treat the bibliography and the cohort as unverified until confirmed.
- No seeds, error bars, or significance tests are reported for any comparison. Single-run results on one model are presented as general findings.
- Causal claims (mid-layer drift causes forgetting; patching establishes causation) rest on correlational metrics and one patching figure without a clear control.
- The RADL localisation and "susceptibility index" are defined only loosely (the thresholds tau_cka and tau_rgd are not given values).
- Section 6.2 recommends a "localized L2 penalty on gate logits" with no equations or experiments supporting it.
- The paper does not study decomposition methods, components, or identifiability, so none of its findings directly transfer to the proposal's question.

## Key Quotes
> "Our findings indicate that early-layer attention heads exhibit systemic entropic dispersion, while mid-to-deep feed-forward networks (or sparse expert blocks) suffer localized representation collapse." (Abstract)

> "The down-projection matrices (W_down) exhibit the highest parameter drift rates across all architectures, corresponding to the low CKA similarity scores found in the middle layers." (Section 6.1)

> "LRCP achieves a high retention rate for Task A (74.8%) while maintaining strong performance on Task B (75.1%), showing a favorable trade-off compared to alternative regularizers." (Section 7)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv preprint with no comment thread.
