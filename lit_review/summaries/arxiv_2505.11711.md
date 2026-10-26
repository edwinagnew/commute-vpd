# Reinforcement Learning Finetunes Small Subnetworks in Large Language Models

## Metadata
- **Authors**: Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tür, Hao Peng (UIUC)
- **Year**: 2025
- **Source**: NeurIPS 2025 (39th Conference on Neural Information Processing Systems)
- **DOI/URL**: arXiv:2505.11711; code at https://github.com/SagnikMukherjee/sparsity_in_rl

## Summary
RL post-training of LLMs (PPO, GRPO, DPO, KTO, ORPO, SimPO, PRIME, rejection-sampling SFT) changes only a small fraction of weights: across 10 public checkpoints, 68.5% to 96.0% of parameters are bit-identical (at bf16 tolerance 1e-5) before and after RL, with the touched subnetwork around 5-30% of parameters. Fine-tuning only that subnetwork (other weights frozen, mask fixed from the full run) matches or slightly beats full fine-tuning on DPO and PRIME benchmarks and ends at nearly identical parameter values, and the authors attribute the sparsity mainly to training on data close to the policy's own distribution rather than to KL penalties or gradient clipping.

## Key Findings
- Update sparsity after RL is consistently high (Table 1: 68.5-96.0% unchanged across algorithms and families), and it arises without any sparsity regularizer, architecture change, or pruning. SFT on the same pipelines produces dense updates (6-15% sparsity, Figure 1).
- The sparsity is spread evenly across layers and matrix types (Figure 3); only layer norm parameters are almost never updated. Within each matrix, the sparse updates are nearly full-rank (Table 2: 99.2-99.8% of maximum rank), so the paper reads RL as selecting a sparse set of entries that spans the full matrix subspace, not a low-rank update as in LoRA.
- Finetuning only the identified subnetwork reproduces full-finetuning performance: DPO subnetwork model matches or beats full model on LSAT/LogiQA/GPQA/MMLU-Pro math (avg 41.9 vs 40.3); PRIME subnetwork gets 72.2 vs 69.8 overall on MATH500. Parameters are 94.0% (DPO) and 90.5% (PRIME) identical at tolerance 1e-5, and 100% identical at 1e-4.
- Subnetworks overlap well above random baselines across seeds (60.5% overlap), training data, and RL algorithm (up to 59.1% under a combined stress test; Table 4). Overlap is not complete, so the paper calls the structure "at least partially transferable."
- The main driver the authors identify is in-distribution training data: on-policy RL, rejection-sampling SFT (RAFT++ 69.4%, RFT 91.2%), and DPO on in-distribution data give sparse updates, while DPO on out-of-distribution data gives dense updates (Table 5). KL regularization and gradient clipping have little effect (GRPO with/without KL: 69.8% vs 68.8%). Update sparsity decays slowly with training steps (Figure 5), and about 8.5% of ever-updated parameters lie outside the final subnetwork because their updates cancel (Figure 6).

## Methodology
Empirical analysis of released checkpoints (Tulu 3 8B/70B, Eurus 7B/2-7B, DeepSeek-Math 7B, DeepSeek-R1-Zero, Llama 3 8B KTO/SimPO, Mistral ORPO) by counting unchanged bf16 parameters between base and RL checkpoints; layerwise and per-matrix sparsity; numerical rank of update matrices. Controlled re-training with DPO (Open-Instruct) and PRIME (verl) on Tulu-3-8B-SFT and Eurus-2-7B-SFT, comparing full fine-tuning against fine-tuning with a gradient mask that fixes the subnetwork identified from the full run. Overlap between subnetworks is measured as one-sided overlap against a random-subset baseline of equal size. Factor analysis varies one thing at a time (KL, clipping, SFT before RL, duration, data distribution).

## Relevance to Proposal
**Score: 7/10**

The paper bears directly on the proposal's strand 3 (structure of fine-tuning updates) and strand 4 (whether post-training reuses existing structure), and it gives the proposal a concrete premise: if a post-training δW is sparse in entries, and that sparsity is stable across seeds and data, then the update may be localized enough for a component-level map Φ(D, δW) to capture it. The tension to test is that the updates are full-rank within each matrix, which argues against a low-rank projection onto rank-1 subcomponents U_c V_c^T as the sole description of δW, and the paper's sparsity is measured in raw parameter space, not in the decomposition's component basis. The stability of the subnetwork across seeds is also a useful reference point for the "is the decomposition itself stable?" question in strand 1.

## Limitations
- The authors acknowledge that the sparsity is a function of the tolerance: the default bf16 threshold of 1e-5 yields ~94% identity for DPO, and changing it to 1e-4 yields 100% identity. The headline sparsity numbers therefore depend on a chosen precision threshold.
- Most of the Table 1 numbers come from public checkpoints whose training configurations are not fully known, so the cross-model comparison is observational. Controlled experiments are limited to a small number of runs (one factor at a time).
- The subnetwork-only fine-tuning uses a mask derived from the full run, so it shows the subnetwork is sufficient post hoc. It does not show the subnetwork can be identified cheaply in advance, which the authors also state as future work.
- The in-distribution explanation is a conjecture, supported by a handful of contrasts (Table 5). The authors mention confounders (e.g., ProRL, Liu et al. 2025a) and an undetermined dependence on training length: the sparsity decays with steps (Figure 5), and the 8K-step R1-Zero comparison is taken from another paper's figure.
- Internal inconsistency: Table 5 lists Zephyr-7b-Beta DPO sparsity as 7.7%, while the text cites 6.8% for the same comparison. Overlap values are reported for single pairs of runs.
- Evaluation covers a small benchmark subset (LSAT, LogiQA, GPQA, MMLU-Pro math, MATH500), with no error bars reported for the main tables, and the subnetwork-vs-full differences (1-3 points) are small relative to typical benchmark noise.
- Language models only; the authors note multimodal and diffusion models are untested.

## Key Quotes
> "Surprisingly, such large gains result from updating only a small subnetwork comprising just 5%-30% of the parameters, with the rest effectively unchanged." (Abstract)

> "updates to almost all parameter matrices are nearly full-rank, suggesting RL updates a small subset of parameters that nevertheless span almost the full subspaces that the parameter matrices can represent." (Abstract)

> "Finetuning the subnetwork alone recovers the test accuracy, and, remarkably, produces a model nearly identical to the one obtained via full finetuning." (Abstract)

> "We conjecture that the this update sparsity can be primarily attributed to training on data that is near the policy distribution; techniques that encourage the policy to remain close to the pretrained model, such as the KL regularization and gradient clipping, have limited impact." (Abstract)

## Comments Summary (if LessWrong/AF post)
Not applicable: this is an arXiv/NeurIPS paper, not a LessWrong or AF post.
