# Weight Oracles: Reading Neural Network Weights with Language Models

## Metadata
- **Authors**: Krishna Kabra, Constantin Venhoff, Christian Schroeder de Witt (University of Oxford)
- **Year**: 2026
- **Source**: arXiv preprint; NeurIPS 2026 Workshop on Neural Network Artifacts as a New Data Modality (per the paper header)
- **DOI/URL**: arXiv:2610.07334 (ID taken from the filename; no DOI given in the text)

## Summary
The paper proposes "Weight Oracles": a fine-tuned LLM (Qwen 2.5 3B) that reads the raw weights of a target network and answers questions about them, without running the target on inputs. Phase I trains the explainer to simulate tiny 1-layer transformers (~2K parameters) by decomposing each forward pass into five LLM passes, one per linear projection, with external code handling LayerNorm, GELU and residuals (99% holdout accuracy). Phase II trains an oracle on benign weight "pathologies" and reports zero-shot backdoor detection with AUROC 0.93 (attention-routed backdoors) and 0.81 across a four-threat-type mix.

## Key Findings
- Single-pass compositional execution, not access to weights, was the binding constraint for simulation. Decomposing the forward pass into learned dot-products plus external operations reached 99.0% holdout accuracy on toy transformers (Section 3).
- A 45% plateau on a full MLP block was traced to a residual shortcut: a classifier ignoring the MLP weights already scored 45.1% (Section 3, "The 45% wall"). This is an author-reported diagnostic on one setup.
- An oracle trained only on benign pathologies (amplified attention, dominant-token shortcuts, dead neurons, noise, etc.) detects held-out pos0 backdoors with AUROC 0.93 +/- 0.04 over five runs, at 94% TPR and 20% FPR (Section 4). Hand-crafted total-L2 gives 0.445 and the best V/O composite 0.752 on the same cohort.
- On a diversified threat mix the composite AUROC falls to 0.81, while hand-crafted statistical detectors are sharp on the attacks they target but collapse under threat-model shift. The oracle stays within 0.74-0.89 per attack type (Section 4).
- Simulation pre-training gives selective transfer: sim-initialised oracles help on attention-pathway questions and hurt elsewhere (composite 0.73 vs 0.81 from scratch) (Section 4).
- Layer-wise probes suggest feature extraction in layers 0-10, a dip in layers 11-33 and a decision signal in layers 34-35 (Section 4, Figure 3). Interpretability ablations are suggestive only.

## Methodology
- Target models: tiny 1-layer transformers (d=8, 2 heads, vocab 1000, 10-class output), generated online (~4,400 unique models over 29,000 steps) to prevent memorisation.
- Phase I: four-stage curriculum (synthetic classification, synthetic regression, attention projections, real small transformers). Weight rows are projected into the LLM's hidden space with a type embedding per matrix. External chain-of-computation with five LLM passes per target forward pass.
- Phase II: Qwen 2.5 3B answers 17 fixed diagnostic question types (Appendix C) with short answers from a fixed vocabulary. Anomaly scores are aggregated over 13 questions. Evaluation on 100 held-out models (50 clean, 50 backdoored), four threat models (pos0, lastpos, stealth, adversarial).
- Baselines: total L2 norm and V/O-pathway statistical composites, plus weight-matrix ablation, gradient attribution and PCA of hidden states.

## Relevance to Proposal
**Score: 3**

The work is weight-space interpretability of small transformers, with no parameter decomposition (VPD/SPD/APD), no fine-tuning-update analysis and no training-dynamics tracking, so it does not touch the core question of commuting a δW past a decomposition. It is only tangentially relevant through the "read the weights directly" theme and the backdoor-detection framing (a mention of unelicitable backdoors and of weight-statistic detectors).

## Limitations
- All experiments use toy transformers of about 2K parameters. The authors state scaling to realistic sizes is an open challenge and that weight-token count grows with parameter count.
- The oracle may learn statistical correlates of backdoors (V/O pathway signatures) rather than the mechanism. The lastpos variant gives weaker per-type AUROC (0.74).
- High false-positive rate for deployment: 20% of clean models are flagged at the 94% TPR operating point.
- Narrow backdoor distribution: single-token triggers, single-class targets, no semantic or input-adaptive triggers.
- Natural-language interface is not ablated against a smaller learned detector trained on the same pathologies.
- Layer-wise and ablation interpretability claims are presented by the authors as suggestive rather than established.

## Key Quotes
> "Crucially, _ME_ sees a different target model every training step, it cannot memorise a fixed set of weights." (Section 3, Architecture)

> "The bottleneck is not access to weights but algorithmic composition." (Section 3)

> "The trade is a worse best-case AUROC for a substantially better worst-case AUROC, the property one wants from a detector deployed against an unknown threat distribution." (Section 4, Results)

> "the oracle detects statistical correlates of backdoors rather than the backdoor mechanism itself" (Section 5, Conclusion and Limitations)
