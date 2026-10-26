# Decomposition of Small Transformer Models

## Metadata
- **Authors**: Casper L. Christensen, Logan Riggs Smith (both independent; Christensen funded by the Pivotal fellowship)
- **Year**: 2025
- **Source**: arXiv preprint 2511.08854 (v2); Mechanistic Interpretability Workshop at NeurIPS 2025
- **DOI/URL**: https://arxiv.org/abs/2511.08854

## Summary
The paper extends Stochastic Parameter Decomposition (SPD, Bushnaq et al. 2025) from toy models to Transformers. It adds a causal-importance function that uses a small attention network across token positions, and a randomised partial reconstruction loss that evaluates only a subset of layers at a time. The authors show SPD recovers the expected two-step induction circuit in a toy 2-layer attention-only model, and that a decomposition over two GPT-2-small sentences isolates rank-1 slices that, when suppressed, reduce the probability of the sport word ("golf"/"basketball").

## Key Findings
- A per-token causal-importance function (1-head, 1-layer QKV-only attention followed by a per-subcomponent MLP) lets SPD assign different importance to repeated tokens (e.g. the two "bank" tokens). Speed cost is reported in Table 2 (about 10 iterations/s vs 12-13 for scalar/vector CI at short sequence lengths).
- On the induction-head toy model, SPD gives a small number of unique subcomponents (Table 1b) concentrated at the relevant positions (Table 1a), with near-zero faithfulness loss (Table 1c). Figure 2 reports SPD needing fewer active components than a greedy rank-1 SVD baseline (Algorithm 2) at matched output distribution.
- On GPT-2-small, decomposing over 2 examples gives 96 active subcomponents (a stated 99% size reduction, counted against the whole model). Suppressing 2-3 rank-1 slices (in the layer-0 MLP, for "obe"/"Bryant" and "Woods") lowers the sport prediction, and Figure 3 reports the effect is specific to the decomposed example.
- The authors interpret this as evidence that some factual associations are written as early as layer 0, while noting causal tracing (Appendix A.3.2) and zeroing whole layers (Figure 9) are only partial support.
- Decomposing over very few examples lets the decomposed model "cheat" by placing new computation in components unused by the original model (Appendix A.3.3). The partial losses are the proposed fix; the authors say the effect shrinks with larger datasets.

## Methodology
- SPD: each weight matrix W^l is split into C rank-1 subcomponents U_c V_c^T. A per-token causal-importance g_c(x) gates each subcomponent via stochastic masks; training combines faithfulness, stochastic and deterministic reconstruction, layerwise reconstruction, and importance-minimality losses (Algorithm 3, Table 4 and Table 6 for hyperparameters).
- Toy model: 2-layer, 1-head-per-layer attention-only Transformer on a 128-token induction task (Table 3). Evaluation is by loss values, active-subcomponent counts and a KL divergence to the original model.
- GPT-2-small: decomposition over two fact sentences, then orthogonal projection of the selected rank-1 directions out of W, checked on a 100-sentence synthetic fact set (Appendix A.3.4). Evaluation is largely qualitative, by ablation.
- No new training-dynamics, fine-tuning, or cross-checkpoint analysis is reported.

## Relevance to Proposal
**Score: 5**

This is a direct extension of the SPD line that VPD builds on, so it bears on strand 1 (parameter decomposition methods, stability, and what the causal-importance function captures). The main relevance to the commute question is the authors' "cheating" concern: a decomposition fitted to a small dataset can absorb new computation into unused components, which is the same failure a commuted update could show, so their partial-loss fix is a candidate control. It does not address fine-tuning, training along a trajectory, or whether decompositions are stable across runs, so it contributes little to the core question.

## Limitations
- Authors state scope is limited to small models, and that the causal-importance parameterisation adds overhead.
- Evaluation is qualitative and task-specific; the GPT-2 result rests on two examples plus one 100-sentence set.
- Faithfulness is supported mainly by ablations, and ablations are not truly causal (the paper's own caveat, citing Covert et al.).
- Ground truth is hard to establish even for toy Transformers (Section 7).
- The SVD comparison is on the toy model only.
- The authors note the causal-importance definition does not handle feature interactions within the model.
- Sparsity and size are not compared with pruning methods (footnote 3).
- No seed-variance, stability, or identifiability analysis is reported.
- Arxiv v2 preprint at a workshop; the manuscript itself describes the GPT-2 claims as "suggests" and "believe" in several places.

## Key Quotes
> "Third, our evaluation is qualitative and task-specific, so further benchmarks are needed to establish generality across architectures and tasks." (Section 7, Discussion)

> "We also note the difficulty in establishing ground truth even for small, toy Transformers. While we obtain evidence that suggests our decompositions may be faithful, this is largely obtained through ablations." (Section 7, Discussion)

> "When decomposing over larger datasets, this effect is minimised as the cheating subcomponents would incur heavy reconstruction losses for all other examples." (Appendix A.3.3)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper; no comments section).
