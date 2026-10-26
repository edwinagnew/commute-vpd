# Tracing the Representation Geometry of Language Models from Pretraining to Post-training

## Metadata
- **Authors**: Melody Zixuan Li, Kumar Krishna Agrawal, Arna Ghosh, Komal Kumar Teru, Adam Santoro, Guillaume Lajoie, Blake A. Richards
- **Year**: 2025 (arXiv 2509.23024, v1)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2509.23024

## Summary
The paper tracks the geometry of last-token hidden representations in LLMs (OLMo-2 1B/7B, Pythia 160M-12B, Tulu-3.1-8B) using two spectral metrics: effective rank (RankMe) and eigenspectrum decay (alpha-ReQ). It reports a non-monotonic, three-phase pretraining sequence ("warmup", "entropy-seeking", "compression-seeking") that loss curves do not show, and that post-training stages move this geometry in recipe-specific directions (SFT and DPO expand it, RLVR compresses it). The work is about activation-space representations, not about weights or parameter decompositions.

## Key Findings
- Pretraining representation geometry is non-monotonic while cross-entropy loss decreases almost monotonically. The "entropy-seeking" phase coincides with peak n-gram distributional memorization (measured against an infinity-gram LM on TriviaQA). Long-context factual QA (TriviaQA) improves mainly in the "compression-seeking" phase.
- Post-training: SFT and DPO increase RankMe (entropy-seeking). SFT on OLMo-2-1B shows that more pretraining improves in-distribution loss but worsens OOD loss and chat win-rate, which the authors read as "overtrained" bases being more sensitive to SFT. RLVR (Tulu-3.1) decreases RankMe (compression-seeking) and reduces pass@256 on AMC-23 relative to the base model, consistent with reduced exploration.
- A linear toy model trained with cross-entropy reproduces the three phases only when token frequencies are skewed and the feature dimension is below vocabulary size (d << |V|). Controls with uniform labels, no bottleneck, or MSE loss remove the compression phase. The authors also report that SciQ accuracy collapses when only top eigen-directions are kept (Table 2), which they use to justify full-spectrum metrics.

## Methodology
Empirical analysis of public intermediate checkpoints. The authors compute RankMe and alpha-ReQ on the covariance of last-layer last-token features over about 10K FineWeb sequences, check that phases appear across layers, and relate the metrics to infinity-gram memorization (Spearman correlation) and benchmark accuracy. Post-training is analysed by comparing Tulu-3.1 checkpoints across SFT, DPO and RLVR, plus separate SFT runs on Anthropic-HH and Alpaca-farm from OLMo-2-1B checkpoints. The theory is a gradient-descent analysis of a linear feature extractor with a softmax classifier.

## Relevance to Proposal
**Score: 3**

This paper is only loosely related to the proposal. It describes post-training as shifting activation-space geometry (SFT/DPO expand it, RLVR compresses it), which touches the "reinforce existing structure vs create new" question at the representation level. It does not decompose weights, does not use rank-1 subcomponents or causal importance, and does not test whether a weight update can be carried through a decomposition, so it bears on none of the proposal's core claims.

## Limitations
- Results are correlational. The authors state that causal links between geometric phases and capabilities are untested.
- Analysis is limited to models up to 12B parameters, and the post-training analysis uses one model family (Tulu-3.1-8B), with OLMo-2-1B for the SFT experiments.
- The theory assumes a linear feature extractor and does not cover full transformers.
- Only English models trained with standard objectives are studied.
- The spectral metrics need about 10K samples and scale quadratically with hidden dimension.
- Only last-layer representations are used for most results. Layerwise checks are shown for one setting.
- The memorization analysis uses one benchmark (TriviaQA) and one proxy (infinity-gram).
- The appendix (robustness checks, the SFT Anthropic-HH vs Alpaca-farm details, and the Table 9 cross-model correlations) was not read in full. The summary covers the main text (Sections 1-6).

## Key Quotes
> "Post-training further transforms geometry: SFT and DPO drive 'entropy-seeking' dynamics to integrate specific instructional or preferential data, improving in-distribution performance while degrading out-of-distribution robustness. Conversely, RLVR induces 'compression-seeking', enhancing reward alignment but reducing generation diversity." (Abstract)

> "Key takeaway. Gradient descent on cross-entropy with (i) skewed token frequencies and (ii) a representation bottleneck (d << |V|) suffices to produce expansion → compression via eigenvector alignment and singular-value growth proportional to magnitude." (Section 3.3)

> "Furthermore, our findings are primarily correlational; establishing causal connections between geometric dynamics and emergent capabilities requires additional investigation." (Section 5, Limitations and Future Work)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
