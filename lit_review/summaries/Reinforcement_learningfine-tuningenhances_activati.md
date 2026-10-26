# Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs

## Metadata
- **Authors**: Honglin Zhang, Qianyue Hao (equal contribution), Fengli Xu, Yong Li (corresponding)
- **Year**: 2026
- **Source**: ICLR 2026 (conference paper)
- **DOI/URL**: Code: https://github.com/tsinghua-fib-lab/llm_rl_probing_analysis

## Summary
The paper asks what online RL post-training (PPO, GRPO) changes inside LLMs, compared with their SFT or DPO starting points. It uses edge attribution patching (EAP) on model-generated math solutions to compare edge-weight statistics before and after RL across four base/post-trained model pairs at about 7B scale. The authors report that RL raises average edge activation intensity and makes activation patterns more diverse (higher entropy, lower kurtosis), while DPO does not show these trends consistently.

## Key Findings
- Across three model families (DeepSeek-Math, Mistral, Distilled-Qwen), Activation Intensity and Information Complexity tend to rise and Distribution Kurtosis tends to fall after online RL, on GSM8K, MATH and College Math (Table 1, Section 4.3). Qwen2.5 SFT vs DPO pairs do not show the same consistent pattern.
- The authors interpret the rise in activation intensity as more low-activation edges becoming active ("engaging a larger set of pathways") (Section 4.3, Pathway Engagement).
- A cross-sample correlation measure and per-node output-edge entropy both rise after RL in most settings, which the authors read as more diverse internal connectivity (Figure 3). The DPO pair shows a smaller diversity gain.
- A single intervention experiment (Qwen2.5-3B-Instruct, GSM8K, 200-token cap) varies sampling temperature (0.6 vs 1.0). The lower-temperature run shows larger activation-intensity gains in steps 40-120 and reaches 51.2% accuracy, versus 36.5% for the high-temperature run. The authors take this as evidence that the internal changes are causal rather than a byproduct (Appendix B).
- The proposed mechanism is that on-policy sampling exposes the network to a wider output distribution, so RL "activates and reinforces latent or 'dormant' internal circuits" (Section 4.3). This is a hypothesis built on the sampling-distribution argument, not a tested mechanism.

## Methodology
EAP-style gradient attribution (linearised, single forward/backward pass) is used to score edges between residual sub-modules (attention and FFN blocks) for each generated solution. Solutions are filtered to questions both models answer correctly, length-matched, and truncated at a scaled mean length. Three scalar statistics are then computed over the edge-weight tensor: mean absolute edge weight (Activation Intensity), histogram Shannon entropy of absolute edge weights (Information Complexity), and mean per-sample kurtosis (Distribution Kurtosis). Comparisons are made base vs post-trained across four truncation scales. The temperature experiment is the only causal test. No weight-space analysis is performed.

## Relevance to Proposal
**Score: 3**

The work studies internal information flow with activation-edge attribution, not the weight-space structure of the update, so it does not address whether a post-training update δW can be carried by a parameter decomposition (VPD/SPD/APD subcomponents). It is loosely relevant to the proposal's strand 4 (mechanistic accounts of whether post-training reuses or creates circuits) because it claims RL strengthens and broadens existing pathways rather than building new ones, and to the "dormant circuit" idea in strand 3.

## Limitations
- Edge attribution is a first-order gradient approximation (EAP) to ablation effects, and the paper does not check it against ablation-based faithfulness. The scalar statistics (mean, entropy, kurtosis) are aggregates that could hide which edges change, and the paper provides no per-component or per-edge identification.
- Base and post-trained models are not controlled for lineage or data. Pairs differ in SFT data (for example MetaMath SFT vs Math-Shepherd PPO) and in starting checkpoint (for example DeepSeek-R1-Distill vs AceReason-Nemotron), so differences may reflect the whole training pipeline rather than RL alone.
- The causal claim rests on one intervention with a single temperature contrast, one model, one dataset and no reported seeds or variance. The authors call their correlational results "robust" based on 41 experiments and a 95.1% "improvement rate," but the rate is over mixed comparisons.
- Scope is math reasoning at about 7B parameters (authors' own limitation, Section 7). The authors state this limits generalisation to code, open-ended generation and larger models.
- Table 1 contains several exceptions (Mistral and DeepSeek at some scales, DPO in most cases) and missing cells due to GPU memory, which the text describes as diminishing with larger truncation scales.
- The metric definitions are described in terms of "diversity" and "redundancy," but the paper does not show that higher entropy or intensity means more functionally distinct pathways. These readings are interpretive.

## Key Quotes
> "RL reshapes information flow to be both more redundant and more flexible, which may explain its advantage in mathematical generalization." (Abstract)

> "Activation Intensity and Information Complexity tend to increase, while Distribution Kurtosis tends to decrease." (Section 4.3, Results and Analysis)

> "the network is compelled to activate and reinforce latent or 'dormant' internal circuits that were underutilized during SFT." (Section 4.3, Sampling distribution discussion)

> "While the consistency of our results suggests potential broader applicability, our empirical validation is currently confined to mathematical reasoning tasks." (Section 7, Limitations)

## Comments Summary (if LessWrong/AF post)
Not applicable (conference paper).
