# Beyond Static Interpretability: Anticipating Post-SFT Mechanisms from Pre-SFT Parameters for Better Tuning

## Metadata
- **Authors**: Hang Chen (NTU; CUHK), Jiaying Zhu (CUHK), Wenya Wang (NTU, corresponding author)
- **Year**: 2026
- **Source**: arXiv preprint, marked "Preprint. Under review." (no venue listed)
- **DOI/URL**: arXiv 2608.24482 (https://arxiv.org/abs/2608.24482, ID taken from the filename); code: https://github.com/Zodiark-ch/Future_localization

## Summary
The paper argues that "locate-then-tune" parameter-efficient SFT, which picks the parameters to tune using causal attribution on the pre-SFT model, is misled when the target skill is new, because the circuits that matter only appear after training. The authors propose estimating the post-SFT attribution state from the pre-SFT model plus a "probing" SFT run on 1% of the data for one epoch at 1% of the learning rate, extrapolated along the probe's update direction with a scalar K. The estimated scores are used to select neurons (top 20%) or to allocate LoRA ranks per component, and the method is reported to beat gradient- and causal-localization baselines on arithmetic, NLU and logic tasks.

## Key Findings
- Localization from the pre-SFT model (Static) misses post-SFT mechanisms for novel tasks. Table 2 (Mistral-7B, component level) reports Arithmetic 83.5 for Static vs 99.6 for Probing and 100 for the proposed method, with the gap widening as arithmetic difficulty grows (Figure 3).
- The probing model's attribution alone (Localization from theta') is close to the static attribution in circuit terms. Tables 5 and 6 show Top@50 overlap with the fully fine-tuned circuit of roughly 18-24 for Static and Probing vs 28-36 for the proposed method, and the authors read this as the probe "remaining heavily anchored" to the initial state.
- The proposed scalar K matters: the best sampling interval for K depends on the task's initial accuracy (Appendix F.1). A small linear probe on the residual stream can predict K without a full SFT surrogate (Appendix F.2, Table 4). Averaging over 10 sampled K values is stable across seeds (CV <= 0.0018, Appendix G).
- Smaller probing updates work better: fewer probe samples, lower LR and fewer epochs give higher overlap with the full-FT circuit, up to a point (Section 5.4, Appendix J). Very small probes (1-2 samples) have high variance.
- On the reported benchmarks, the component-level method also reduces conflicting nodes in multi-task joint training relative to baselines (Appendix E, Figure 6), with the caveat that the authors' own limitations section says polysemanticity still undermines this.

## Methodology
- Attribution patching (first-order Taylor, two forward passes and one backward pass) is used as the causal-effect score. Activations and gradients are evaluated at theta, and the paper derives a sensitivity identity S(theta_I) = S(theta + K*dtheta) (Appendix B). This relies on a Taylor/"translation operator" argument and a stated HVP assumption.
- Probing SFT: 1% of training data, 1 epoch, 1% of the original LR, giving theta' (Section 4.1). The estimated score averages over 10 sampled K values, with a second-order term approximated for efficiency (Section 4.2).
- Two pipelines: neuron level (train top 20% of neurons, about 1.4B of Mistral-7B's parameters) and component level (rank-allocated LoRA, ranks in [1, 32], mean 8).
- Evaluation: Mistral-7B as the main model, plus LLaMA-2-13B and Qwen3-30B for scale. Tasks are single-token-output: GLUE, BOOL, Arithmetic (2-7 digits, 1-6 steps), plus IOI, Induction, Gender, Docstring and WinoGrande for circuit analysis. The "ideal" post-SFT model is the fully fine-tuned model used as a surrogate (Section 5.4, Appendix F.1).
- Baselines: Graft, WAGLE, CLUE, FLU, CircuitLoRA. Metrics: target task accuracy (TTA) and a retention metric (PTA, lm-eval-harness average).

## Relevance to Proposal
**Score: 5**

The paper is adjacent to the proposal's core question: it asks how to predict where a small or full post-training update will land, and it uses a small probing update (1% of data, one epoch) to estimate the post-update state, which is close in spirit to commuting a small delta-W past a decomposition. It is not about parameter decompositions (VPD/SPD/APD), though, and its finding that the probe's attribution barely differs from the static one is direct evidence that a small first-order update does not move mechanisms much, which bears on whether a simple linear map can carry a decomposition forward.

## Limitations
- The "ideal" post-SFT state is the fully fine-tuned model used as a surrogate, so the reported overlap measures agreement with one training run, not with the true target. The method also picks K using that same surrogate (Appendix F.1), which is a leakage concern for the headline comparisons.
- Absolute agreement with the full-FT circuit is low: Top@50 overlap is 28-36 out of 50 for the proposed method (Tables 5-6), and the paper's "closely aligned" claim is relative to baselines.
- Only single-token-output tasks are studied, which the authors acknowledge (Section 4 footnote 4, Section 6). Multi-token instruction tuning is left open.
- Attribution patching is a first-order approximation; the paper's own Appendix C argues that directly substituting the probe's attribution is theoretically unsound, and Appendices A and B rely on approximations (Hessian and HVP assumptions) that are not empirically tested beyond Figure 10.
- Headline comparisons use Mistral-7B; the scale experiments (LLaMA-2-13B, Qwen3-30B) are reported only for TTA on two arithmetic subtasks (Figure 2).
- Table 1 reports the proposed method's NLU TTA as 86.31 (neuron level), while the matching ablation in Table 2 reports 96.31 for the same method and task family. The paper does not explain the difference, so the numbers should be checked against the code before being cited.
- Uses mechanistic localization for parameter selection, not for decompositions, so the claims do not transfer directly to VPD-style component spaces.
- Compute cost reported as wall clock on RTX A6000; localization cost is said to grow superlinearly with model size (Figure 2) but only three model sizes are shown.
- Appendices B-J were read at the level of their stated claims and tables; the derivations were not checked line by line.

## Key Quotes
> "locating-then-tuning demands that interpretability _anticipate_ which parameters will be recruited to learn the target skill, yet mechanistic interpretability can only dissect capabilities that are _already instantiated_ within existing parameters." (Section 1, Introduction)

> "the negligible performance gap between the probing and static methods implies that the mechanistic distribution of theta' remains heavily anchored to the initial state theta, thereby lacking the necessary foresight to anticipate mechanistic shifts during the fine-tuning process." (Appendix H)

> "deriving theta' from 1% of the training data yields significantly superior results compared to using 10%. This phenomenon occurs because Delta theta must remain sufficiently infinitesimal to preserve adequate degrees of freedom for the extrapolated state theta + K * Delta theta" (Section 4.1)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint, no comments).
