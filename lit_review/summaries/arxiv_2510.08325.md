# Beyond Pass@k: Breadth-Depth Metrics for Reasoning Boundaries

## Metadata
- **Authors**: Marius Dragoi, Ioana Pintilie, Florin Gogianu, Florin Brad (Bitdefender, Romania)
- **Year**: 2025 (arXiv v2, 21 Oct 2025)
- **Source**: arXiv preprint cs.AI
- **DOI/URL**: arXiv:2510.08325

## Summary
The paper argues that Pass@k at large k is a misleading measure of an RLVR model's "reasoning boundary" on tasks with small discrete answer spaces, because Pass@k tends to 1 for any problem with nonzero success probability. It proposes Cover@τ, the fraction of problems with per-problem success rate at least τ, and shows Pass@k is a Beta(1,k)-weighted average of Cover@τ. It then re-ranks several RLVR algorithms (GRPO, GSPO, PPO, KL-Cov, GRPO-Unlikeliness) on OMEGA and Reasoning Gym, starting from Qwen2.5-7B-Instruct.

## Key Findings
- (Authors, Sec 3, Remark 1) Pass@k converges to 1 for any p>0, so large-k crossover can reflect guessing in small answer spaces. On OMEGA Probability (No Fixed) the base model reaches Pass@k=1 at 2^13 samples, with only 30 possible test answer values.
- (Authors, Sec 5, Prop 1 and Cor 1-2, with proofs) Pass@k = E_{τ~Beta(1,k)}[Cover@τ]. The unweighted area under Cover@τ equals Pass@1. Cover@τ dominance implies Pass@k dominance, but not the reverse. Maj@k equals Cover@0.5.
- (Authors, Sec 6.2, Fig 3) On the OOD split of Probability (No Fixed), the base model's coverage "drops significantly" once τ rises from below 0.01 to 0.025, even though it beats the RLVR models in Pass@k at large k. The authors read this as a smaller reasoning boundary than Pass@k implies.
- (Authors, Sec 6.3, Table 1) Rankings differ across Pass@1, Cover@0.2, Cover@0.8 and AvgAUC+cover. KL-Cov is best on OMEGA OOD; GRPO ranks 2nd on Cover@0.2 but falls out of the top 3 at Cover@0.8. The authors conclude that methods which prevent entropy collapse generalise better.
- Table 1 numbers (extracted from PDF; column alignment looks partly garbled, so treat with caution): base OMEGA OOD Pass@1 appears to be 14.94 against GRPO 22.66 and KL-Cov 28.34. The columns beyond that are unreliable and I am not quoting them.

## Methodology
Metric definition and short proofs, then RL training with VERL from Qwen2.5-7B-Instruct on five OMEGA tasks (train IID, test OOD) and one Reasoning Gym transfer setup (train on simple equations and polynomial multiplication, test on intermediate integration). The best checkpoint is picked on IID validation. 32 rollouts per prompt on OMEGA and 8 on Reasoning Gym. Evaluation is purely behavioural: sampled-answer correctness. There is no interpretability, activation, or weight analysis.

## Relevance to Proposal
**Score: 2**

1. Evidence is entirely behavioural (sampling success rates per problem). Nothing looks inside the model.
2. Its result could hold with our mechanistic verdict going either way (my inference). Cover@τ says how reliably the post-trained model solves a problem, or the base model does. It says nothing about whether the route runs through a readable base-model direction (Jain Def 1/2). Cover@τ rising for the RL model could coincide with "not through C" if the new reliability comes from a new mechanism. Base Cover@τ at small τ being nonzero could coincide with "reuses C" or "unreadable". The paper also supports neither side of the sharpening debate: it only argues that the Pass@k evidence used in that debate is weak.
3. Usable items (my inference, thin): (a) Cover@τ is a cheap behavioural comparator to sit beside our readings, for example per-problem GSM8K success rates for base, SFT and GRPO arms at early steps, to show whether early δW moves breadth (low τ) or depth (high τ). It also gives a way to say that sharpening, as a behavioural claim, is a shift from low τ to high τ mass. (b) The warning that large-k Pass@k is degenerate on small answer spaces applies if we ever cite pass@k evidence for GSM8K numeric answers. (c) OMEGA and Reasoning Gym are procedurally generated with IID/OOD splits, which might give controlled tasks, but the paper does not establish novelty by construction in a mechanistic sense. Nothing on early training steps, SFT, or Qwen3 specifically; the model is Qwen2.5-7B-Instruct. The connection is thin.

## Limitations
- (Authors, Sec 7) Cover@τ scores final-answer correctness only, not reasoning-trace soundness.
- (Inference) Only one base model and one seed per method appear to be reported; no variance or confidence intervals seen. Best-checkpoint selection and different KL coefficients per task confound the algorithm comparison.
- (Inference) The argument concerns numeric small-support answers. The interpretation that low Cover at τ>0 shows "limited reasoning" is the authors' claim and not tested against any internal evidence.
- Not read: the cited papers (Wen et al., Liu et al. G-Pass@k, etc.). No appendix is present in the extracted text. Figures are garbled axis text, so I did not read values off the plots.

## Key Quotes
> "We argue that on tasks with discrete answer spaces, such as math with numeric outputs, Pass@k at large k reflects the increasingly higher chance of success in the limit of the number of trials rather than genuine reasoning, and can therefore be misleading." (Abstract)

> "Pass@k is a weighted average of Cover@τ, with weights from a Beta(1,k) distribution; this reveals that Pass@k is biased towards low-τ regions of Cover@τ, emphasizing lucky hits rather than reliability" (Sec 1, p.2)

> "Overall, the strong performance of KL-Cov suggests that RLVR methods that prevent entropy collapse show stronger generalization abilities." (Sec 6.3, p.8)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
