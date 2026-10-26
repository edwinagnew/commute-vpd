# Operationalising the Superficial Alignment Hypothesis via Task Complexity

## Metadata
- **Authors**: Tomás Vergara-Browne, Darshan Patil, Ivan Titov, Siva Reddy, Tiago Pimentel, Marius Mosbach
- **Year**: 2026
- **Source**: arXiv preprint 2602.15829v2; Proceedings of ICML 2026 (43rd International Conference on Machine Learning, PMLR 306)
- **DOI/URL**: https://arxiv.org/abs/2602.15829

## Summary
The paper gives the superficial alignment hypothesis (SAH) a precise definition: "task complexity" is the length in bits of the shortest program (a Python script plus any weights or data it carries) that reaches a target performance on a task, given access to a pre-trained model's weights. It argues that the data, parameter, and inference-control views of superficial adaptation are three strategies for finding such short programs, and estimates Pareto curves of program length versus performance for GSM8K, FLORES English-French translation, and IFEval using SmolLM3 3B and Olmo3 7B/32B. Its headline finding is that strong performance is reachable with programs of a few kilobytes, and that post-training collapses the length needed to reach a model's maximum performance by several orders of magnitude relative to pre-training checkpoints.

## Key Findings
- Under the authors' measure, the conditional complexity of strong performance is low: Olmo3-32B reaches 72.2% on GSM8K with a 4,358-bit program, and Olmo3-7B reaches 34.43 BLEU on translation with a 3,992-bit program. IFEval maximum performance needs roughly 10^7 bits (about 1.25 MB).
- The three views map to different regions of the Pareto curve: inference control (ICL, URIAL) dominates the shortest programs, data methods (Subset Training) the middle, and parametric methods (LoRA, Bayesian-LoRA) the longest.
- Pre-trained checkpoints make strong performance accessible but need programs of megabytes to gigabytes (for example, SmolLM3 on GSM8K plateaus only near 5×10^10 bits, about 6.2 GB). Post-trained checkpoints reach the same or higher performance with programs whose size has minimal effect beyond about 10^4 bits (Olmo3-7B on GSM8K: maximum complexity about 5×10^6 bits when pre-trained).
- The paper reinterprets prior critiques: Raghavendra et al. and Lambert are judged to set the wrong target because fine-tuning on few samples does not saturate high performance; Chen et al.'s linear-projection "superficial knowledge" method yields neither short programs nor high accuracy; BitDelta-style "1 bit per parameter" compression is judged too large to be Pareto-optimal under this measure.
- Caveat the authors state themselves: post-training data for GSM8K and IFEval was included in post-training, which may inflate the post-training effect. Additional FLORES and e-SNLI experiments are reported to decouple this (Figure 8).

## Methodology
Theoretical: task complexity is defined with algorithmic information theory (Kolmogorov complexity and algorithmic rate-distortion), with a Turing-complete Python machine and its libraries treated as free. Properties are derived in an appendix (invariance to the choice of machine, uncomputability). Empirical: for each of three strategies (data: arithmetic-coded subset training; parametric: LoRA, Bayesian-LoRA, full model; inference control: ICL, URIAL, base model) the authors run a hyperparameter sweep, measure program length (script bits plus compressed data or adapter weights), and keep the non-dominated (length, accuracy) points. The pre- versus post-training analysis (Section 7) uses randomly initialised, stage-1 pre-trained, and post-trained checkpoints of SmolLM3 3B and Olmo3 7B.

## Relevance to Proposal
**Score: 4**

The paper is relevant to the proposal's strand 3 (structure and compressibility of fine-tuning updates) and strand 4 (what post-training does to existing capability). Its result that post-training makes strong performance accessible with very short programs is a claim about accessibility of performance, not evidence about which decomposition components carry an update, so it does not test whether a simple map Φ(D, δW) exists. It is most useful as framing: "post-training resurfaces pre-training structure" is stated in information terms, which could help define what "strengthening an existing mechanism" would mean in a commuting test. It does not use parameter decomposition and offers no component-level or faithfulness measurement.

## Limitations
- Task complexity is uncomputable, so every reported value is an upper bound; shorter programs may exist, and the Pareto curves are not tight (Section 9).
- Estimates depend on the chosen machine and its libraries being free of length cost, and on the hyperparameter sweep, which may be unevenly tuned across methods.
- Post-training contamination: GSM8K and IFEval training data were included in post-training, so the collapse effect is partly confounded (footnote 9).
- Only three tasks and three models; the 32B model is used for the pre-trained Pareto curves but not in the pre- versus post-training analysis, which uses SmolLM3 3B and Olmo3 7B.
- Measures only the length of the final program, not the cost of the procedure that found it; the authors explicitly do not count the compute or data used to build the program.
- The complexity measure says nothing about internal mechanisms, components, or whether the update reuses existing circuits.
- Scope of this summary: the main text (Sections 1-9) and references were read; the appendices (theorems, sweep details, Sections A-F) were not read in full.

## Key Quotes
> "Post-training, on the other hand, collapses the complexity of reaching this same performance by several orders of magnitude." (Abstract)

> "More importantly, however, post-training collapses the complexity of achieving the maximum performance. ... In contrast, for post-trained models, program size has minimal effect on performance beyond a size of 10^4 bits." (Section 7)

> "Overall, our results highlight that task adaptation often requires surprisingly little information—often just a few kilobytes." (Abstract)

## Comments Summary (if LessWrong/AF post)
Not applicable: this is an arXiv paper.
