# The Low-Rank Structure of VLA Reinforcement Learning

## Metadata
- **Authors**: Minjae Oh, Yoonah Park, Jongwon Lim, Yohan Jo (Graduate School of Data Science, Seoul National University)
- **Year**: 2026 (arXiv 2609.34599; preprint, marked "Preprint.")
- **Source**: arXiv preprint (no venue listed in the file)
- **DOI/URL**: arXiv:2609.34599 (https://arxiv.org/abs/2609.34599)

## Summary
The paper studies how reinforcement learning (PPO) post-training changes flow-based vision-language-action (VLA) robot policies (pi0.5, GR00T N1.5/N1.6, SmolVLA) in parameter space. It finds that RL updates are dense but low-rank and concentrated in the "Timestep Modules" of the action expert (the Time MLP and AdaRMS layers that map the denoising timestep to scale, shift and gate vectors), and that these modules carry most of the RL gain. The authors then show that the RL-induced shift-vector updates predict task success and can be used for test-time steering.

## Key Findings
- RL updates to flow-based VLAs have much lower effective rank than behaviour-cloning (BC) updates. For Timestep Modules, RL effective rank (95% of squared Frobenius energy) is about 6-30, versus roughly 250-650 for BC. Attention and MLP updates stay high-rank. The same structure does not appear in RL-trained image/video DiTs (Sec. 3.1, Sec. C.3).
- Module-replacement experiments: keeping only the RL-trained Timestep Modules (other weights reset to base) keeps almost all of the RL gain, e.g. pi0.5 reaches 95.1 vs 96.5 on LIBERO and 69.6 vs 69.2 on MetaWorld. Keeping only the MLP+Attn updates loses much more (up to -39.7 on ManiSkill). The authors claim this as a disproportionate share of the gain (Sec. 3.2, Table 1).
- The discrete denoising timesteps used during on-policy RL drive the specialization. AdaRMS responses under RL peak sharply at the few training timesteps, and discrete-timestep BC reproduces both the localized response and the low rank (Sec. 4.1, Fig. 3). This is a correlational ablation, not a full causal isolation.
- Among AdaRMS outputs, the shift vector changes most under RL and carries new directions, while scale and gate changes mostly rescale existing directions (Sec. 4.2, Fig. 3 right).
- Linear probes using RL shift-update directions as probing vectors predict episode success with ROC-AUC up to 99.6 on LIBERO-Spatial, and well above shuffled-label controls. Probes also work on the base policy (Sec. 5.1.1, Table 2).
- Pairwise geometry of single-task shift updates (10 LIBERO-Spatial tasks) correlates with cross-task transfer patterns, mean Spearman 0.504 (range 0.224-0.794) (Sec. 5.1.2).
- Adaptive steering along the shift direction (intervening only when the probe projection falls below the success mean) improves final RL policies by 2.0-4.0 points across five benchmarks, and beats matched-norm random directions (Sec. 5.2, Table 3).

## Methodology
Parameter-space analysis of public and in-house BC/RL checkpoints: update density (fraction of parameters with |change| > 1e-5), effective rank of ΔW via SVD (95% energy), and module-level replacement experiments. Mechanistic analysis of AdaRMS via top-3 input singular directions and cosine similarity with the conditioning vector across denoising timesteps. Interpretability uses linear logistic-regression probes on projections of hidden states onto ΔB (shift update), trained on 70% of tasks and tested on the held-out 30%. Steering is a simple adaptive activation intervention. Benchmarks are LIBERO, ManiSkill, MetaWorld and CALVIN in simulation. RL is PPO (following pi-RL).

## Relevance to Proposal
**Score: 4**

The paper is about a different model class (flow-based robot policies, not a transformer LM) and does not use any parameter decomposition (no SPD/VPD/APD, no rank-1 subcomponents, no causal-importance function). Its relevance is to the "structure of fine-tuning updates" strand: it gives direct empirical evidence that RL post-training updates are low-rank and concentrated in a small module, which supports the premise that a low-rank or component-aligned projection of δW could carry the update. It also bears on the strand "RL sparsity of updates" (compare Mukherjee et al. 2025 and the note that discrete-timestep training, not RL per se, may drive the structure).

## Limitations
- Simulation only (LIBERO, ManiSkill, MetaWorld, CALVIN). No real-robot evidence.
- The "disproportionate share of the RL gain" claim rests on success-rate differences of a few points on a few checkpoints; the module-replacement is a single intervention per module, and the paper itself notes that the Timestep Modules are a small fraction of parameters, so the result may be partly specific to how action experts use timestep conditioning.
- Probing and steering analyses use the same benchmarks and task split as the probes were selected on (the steering layer was chosen by probing F1). Probe ROC-AUC with random-label controls is reported but no shuffled-control robustness check across seeds was visible in the read portion.
- The effective-rank threshold (95% energy) and update-density threshold (1e-5) are conventions; ablations are said to be in Sec. C.1 (not read in detail).
- Discrete-timestep BC is offered as the cause of low rank via one comparison (Fig. 3, Sec. C.2); the causal attribution is not fully isolated.
- The authors' own discussion frames the implication (residual RL, parameter-efficient RL) speculatively.
- Only about the first 330 of ~960 lines (main text through Sec. 7 and reference list start) were read; appendices C.1-C.9 were not checked in detail.

## Key Quotes
> "RL induces substantially lower-rank parameter updates that are highly concentrated in the action expert's Timestep Modules, a small and previously overlooked component." (Abstract)

> "RL updates to the Timestep Modules have an effective rank of only 6–30, roughly an order of magnitude lower than BC (250–650)." (Sec. 3.1)

> "RL sharply specializes the Timestep Modules to selectively respond to the discrete denoising timesteps encountered during on-policy learning, and this discrete-timestep training underlies the low-rank updates." (Sec. 1)

> "Values near 1 indicate rescaling along the existing direction, while values near 0 indicate a new direction." (Sec. 4.2)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint, no comments).
