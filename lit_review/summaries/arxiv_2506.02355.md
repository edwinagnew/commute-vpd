# Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening

## Metadata
- **Authors**: Andre He, Daniel Fried, Sean Welleck (Carnegie Mellon University)
- **Year**: 2025 (arXiv v2, 20 Jun 2025)
- **Source**: arXiv:2506.02355 [cs.LG]
- **DOI/URL**: https://arxiv.org/abs/2506.02355 ; code at github.com/AndreHe02/rewarding-unlikely-release

## Summary
In Lean theorem proving with DeepSeek-Prover-V1.5-SFT, standard GRPO raises pass@1 to pass@16 but leaves pass@N at large N at or below the base model. The authors call this distribution sharpening. They attribute it to a "rank bias": GRPO tends to raise the probability of correct samples that were already high-probability under the base model and rarely raises rare correct ones. They propose an "unlikeliness reward" that up-weights rare correct samples, and they show that more PPO epochs per batch helps too. All evidence is behavioural, from sampling and per-sample likelihoods. Nothing inside the model is examined.

## Key Findings
- Authors' claim: GRPO-Default improves pass@N at small N but "performs worse than the base model for large N" on a held-out Lean set (Fig. 2, Sec. 3.3).
- Authors' back-of-envelope argument (Sec. 3.4): if RL multiplies each correct sample's probability by about (1+eps), pass@N gains at large N need uplift of solutions with p0 near 1/N. This is a simplified, early-training, independent-samples model, not a measurement.
- Authors' measurement (Sec. 3.5, Fig. 4): the "uplift rate" is the fraction of correct samples whose likelihood rose from base to GRPO model, as a function of the sample's rank under the base model. It is positively correlated with base-model likelihood, and rarest correct samples are "almost never uplifted". The authors state that the rank bias is "likely a consequence of the optimizer's biases", not the GRPO loss (Sec. 4). That is a hypothesis; no mechanism is tested.
- Unlikeliness reward (Sec. 4.1): r_i = R * (1 - beta_rank * (G - rank)/G), beta_rank = 0.25. It reverses the uplift pattern (Fig. 6) and raises large-N pass@N (Fig. 5), with a small cost at pass@1 and pass@2. More PPO epochs raise overall reinforcement strength, so rare samples also get uplifted. Table 2 shows training problems solved: static 7707/9600, GRPO-Default +153, Unlikeliness-2 +358. A KL-only control (Appendix D) stops the pass@N decline but does not improve over the base model.
- Toy environment (Appendix A): a 10-d state, 128 actions, hidden random vectors, reward 1{s.v_a >= tau}. It reproduces the rank bias and the fix. The task has known structure by construction, but it is a bandit with an MLP policy.

## Methodology
GRPO via verl on a roughly 9.6K-problem subset of Lean Workbook plus miniF2F-valid. Base model is DeepSeek-Prover-V1.5-SFT (a 7B-class model; the size is my inference, not stated in the text I read). lr 1e-6, KL coefficient 0.02 (0.10 in variants), 32 samples per problem, one epoch, 512-token responses. Evaluation is pass@N up to 512 on an i.i.d. held-out set. The rank-bias analysis compares sequence log-probabilities under the base and trained models on training samples. Binary Lean-verifier reward.

## Relevance to Proposal
**Score: 3**

1. Evidence type: purely behavioural. It uses pass@N, per-sample sequence probabilities and diversity counts. It does not look at activations, weights, circuits or parameter updates. The uplift-rate analysis is a behavioural, output-space probe of what RL changed.
2. Could its result hold while our mechanistic verdict goes the other way? Yes, in both directions (my inference). (a) "Sharpening" in their sense means correct samples already in the base sampling distribution get more probable. That can coexist with a delta-W that does not decompose into existing VPD subcomponents, or that does not go through a linear readout of C, so our verdict would be "not through C" or "unreadable". (b) Conversely, a rare-sample uplift under the unlikeliness reward is behaviourally "beyond sharpening". It could still run entirely through a readable base mechanism, giving "reuses C". Their notion is defined by base-model sampling probability, which is the behavioural definition the project rejected. So it is not a comparator for the causal verdict, and a result on it says nothing about ours.
3. Directly usable, with thin value. The uplift rate, rank-binned over correct samples, is a cheap behavioural comparator that could sit beside our readings on GSM8K arms. We could ask whether our early-step GRPO delta-W mostly moves already-likely correct samples, which would let us say whether the SFT and GRPO arms are "sharpening" in the literature sense. The finding that GRPO updates concentrate on high-likelihood correct samples, and that PPO epochs and the KL coefficient change this, is a GRPO-specific datum relevant to interpreting early-step updates. The toy environment has known structure but is not a model with a parameter decomposition. Nothing is specific to Qwen, SFT, or early steps in a mechanistic sense. The early-training assumption appears only in the Sec. 3.4 analytic argument.

The connection to the project is thin. It is useful mainly as a citation for the behavioural "sharpening" literature, as a pointer to the paper it builds on (Yue et al. 2025), and for one cheap metric.

## Limitations
- Authors: reasoning-style models with long chains of thought may behave differently (Limitations, p. 9). Only formal theorem proving is studied, with a perfect verifier.
- Inference: single base model, one epoch, small scale. The Sec. 3.4 argument rests on independence and clipping-bound assumptions. Mechanism of the rank bias is not tested. Whether the unlikeliness reward finds "new" capability or only reallocates mass among samples the base model could already produce is not established. Pass@512 on a finite set cannot show that.
- Table 3 and several figures are text-extracted. I do not trust curve values read from the figures. The Table 3 numbers appear parsed correctly but were not cross-checked against the PDF.

## Key Quotes
> "While GRPO improves single-sample accuracy, it often fails to improve and can even impair pass@N metrics at larger N in our theorem proving setting" (Sec. 1, p. 2)

> "GRPO is more likely to increase the probability of already high-probability correct solutions. In contrast, the low-probability positive samples – those most critical for improving pass@N at large N – are almost never uplifted." (Sec. 3.5, p. 5)

> "The observed rank bias is therefore not a feature of the GRPO loss but likely a consequence of the optimizer's biases." (Sec. 4, p. 5)

## Comments Summary
Not applicable (arXiv paper).

## Not read
I read the full text, including the appendices. I did not see the figures themselves, only extracted axis text. The GitHub code was not consulted.
