# Evolution of Concepts in Language Model Pre-training

## Metadata
- **Authors**: Xuyang Ge, Wentao Shu, Jiaxing Wu, Yunhua Zhou, Zhengfu He, Xipeng Qiu (OpenMOSS Team, Shanghai Innovation Institute / Fudan University; Shanghai AI Laboratory)
- **Year**: 2026
- **Source**: ICLR 2026 (conference paper)
- **DOI/URL**: Code at https://github.com/OpenMOSS/Language-Model-SAEs (no arXiv or DOI line in the provided file)

## Summary
The paper tracks interpretable features across pre-training checkpoints of Pythia models by training a single cross-snapshot crosscoder (a sparse dictionary learned jointly over several checkpoints' activations at one layer). Most features form around a specific early point in training (around step 1,000), while more complex features appear later, and the authors read this as a two-stage picture: a statistical learning phase followed by a feature learning phase. The authors frame this as a first step toward tracking representation change during pre-training at feature resolution.

## Key Findings
- Cross-snapshot crosscoder features fall into two main groups: "initialization features" present from random init (with a drop and recovery near step 128), and "emergent features" that begin forming mainly around step 1,000. A smaller set appears only in late training.
- Decoder norms are used as a proxy for feature presence and strength at each snapshot. Linear probes support this on a 6,144-feature Pythia-160M crosscoder (mean correlation between probe training error and decoder norm of -0.867, SD 0.153).
- Most features change direction sharply around step 1,000 (pre- and post-1,000 directions nearly orthogonal), then rotate more gradually. Most features persist for over 60% of snapshots after emergence.
- Feature complexity (LLM-scored, Claude Sonnet 4, 100 emergent features) correlates moderately with peak emergence step (Pearson r = 0.309, p = 0.002). Previous-token features emerge early (about steps 1,000-5,000), while induction and context-sensitive features appear later (mostly steps 10,000-100,000).
- Attribution patching with integrated gradients over crosscoder features, on subject-verb agreement, IOI and induction tasks, identifies small sets of features (tens) whose ablation disrupts or recovers task performance across snapshots. The authors present this as a link from features to downstream behaviour.
- Early training is dominated by unigram and bigram statistics: unigram and bigram KL divergence converge early, and loss approaches the unigram/bigram entropy floor. Total feature dimensionality first compresses and then expands around the turning point, which the authors interpret as a move from weak to strong superposition.

## Methodology
A cross-snapshot crosscoder is trained on activations from 32 Pythia checkpoints (chosen from 154, steps 0 to 143,000) at one middle layer (layer 6 of Pythia-160M, layer 16 of Pythia-6.9B). It uses a shared sparse feature code with snapshot-specific encoders and decoders. Training uses JumpReLU activations and a tanh plus quadratic frequency sparsity penalty, with 800M SlimPajama tokens. The authors compare it against per-snapshot SAEs, probe decoder norms with linear probes, classify features with rule-based patterns (previous-token, induction, context-sensitive), and use LLM-based complexity scoring. Causal links to behaviour come from integrated-gradient attribution patching over 1,000 samples per task.

## Relevance to Proposal
**Score: 7**

This is the closest match found so far to the proposal's strand 2 (features tracked along training, including continual or online interpretability) and strand 6 (mechanism stability across training). It shows that one shared sparse dictionary trained across checkpoints can follow features coherently over training, and it reports the large directional change around step 1,000, which bears directly on whether small post-training updates can be carried by existing components. It does not address parameter decompositions (VPD/SPD/APD), weight-level updates δW, or fine-tuning, and its own stated limitation (joint training needs discrete snapshots) is the gap the proposal's commuting step aims to close.

## Limitations
- Results are limited to Pythia models (160M and 6.9B) and pre-training; the authors state generalisation to other architectures, datasets, post-training, and tasks is untested.
- Downstream tasks are simple (SVA, IOI, induction), and the authors attribute this to model capacity, dictionary quality, and circuit-tracing limits.
- The method needs all snapshots at once; memory and compute scale linearly with snapshot count, so tracking is coarse and not continuous. The authors propose online multi-snapshot processing or gradient information as fixes, neither tested.
- Decoder norm as a proxy for feature evolution is supported by probes on one 6,144-feature crosscoder only. The paper's own appendix notes rare feature splitting among high-activation features.
- Complexity scores come from an LLM rater with a fixed rubric; no inter-rater or human validation is reported in the provided text.
- The "statistical learning then feature learning" phase split is presented as a hypothesis consistent with the observations (KL, dimensionality), not a directly tested causal mechanism.
- Attribution uses first-order or integrated-gradient approximations, and the paper does not compare them against causal interventions on the training process itself.

## Key Quotes
> "We find that most features begin to form around a specific point, while more complex patterns emerge in later training stages." (Abstract)

> "most features undergo drastic directional shifts around step 1,000, rendering pre- and post-step 1,000 directions nearly orthogonal (Figure 5c)." (Section 4.1, "A universal directional turning point")

> "Crosscoder training requires activations from discrete training snapshots, with memory and computational costs scaling linearly with snapshot count, which limits observational granularity." (Section 7, "Discrete snapshot requirement")

## Comments Summary (if LessWrong/AF post)
Not applicable. This is a peer-reviewed ICLR 2026 paper with no comment thread.
