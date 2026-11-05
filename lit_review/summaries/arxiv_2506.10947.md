# Spurious Rewards: Rethinking Training Signals in RLVR

## Metadata
- **Authors**: Rulin Shao*, Shuyue Stella Li*, Rui Xin*, Scott Geng*, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, Luke Zettlemoyer
- **Year**: 2026 (arXiv v2, 25 Feb 2026; preprint, cited by the project as ICML 2026)
- **Source**: arXiv:2506.10947
- **DOI/URL**: https://arxiv.org/abs/2506.10947

## Summary
The paper shows that GRPO with weak or spurious rewards (majority vote, format, random, deliberately incorrect labels) improves MATH-500 and AMC accuracy a lot for Qwen2.5(-Math) models. The same rewards mostly fail on Llama3 and OLMo2. The authors attribute the Qwen gains to a GRPO clipping bias that amplifies behaviours the base model already assigns high probability, with "code reasoning" in Qwen2.5-Math as the case study. All evidence is behavioural.

## Key Findings
(All are the authors' claims unless marked as inference.)
- On Qwen2.5-Math-7B, random reward gives +21.4 points on MATH-500, against +29.1 for ground truth. Incorrect-label reward gives +24.1, format +13.8 (garbled figure; the AMC text also says 13.8 for format, so I trust the main text over the figure labels). Most of the gain arrives within about 50 steps (Sec 2.2, Fig 1-2).
- Spurious rewards rarely help non-Qwen2.5 families. Smaller models gain less from random reward, and Qwen2.5-Math-1.5B gains only after about 100 steps under random reward (Sec 2.2, 3, Fig 3). Models that were already RL post-trained gain little (Appendix J, cited in Sec 3; not read).
- Mechanism claim: with clipping removed, random rewards give zero expected gradient and no consistent gain. With clipping on, the expected gradient favours tokens that are already high-probability and suppresses low-probability ones (Sec 4, Eq. 2, Fig 4). This is a derivation plus ablations on Qwen2.5-Math-7B only.
- Code reasoning, meaning Python written without execution, appears in 65.0% of Qwen2.5-Math-7B MATH-500 answers before training. Accuracy is 60.9% with code and 35.0% without (Table 1). Under weak or spurious rewards the code share rises to about 90% within 15 steps, and to 95.6% for random reward (Sec 5.2).
- Under ground-truth reward the code share rises at first and then falls, which the authors read as "a different improvement mechanism" (Fig 6 caption).
- Interventions: forcing the first sentence "Let's solve this using Python." lifts Qwen2.5-Math-7B from 49.4 to 64.4. It hurts Llama and Qwen2.5-7B (41.6 to 22.2). A "python"-string reward pushes code use above 99% in 20 steps and helps mainly the Qwen2.5-Math models (Sec 5.3, Table 2, Fig 7).
- The authors frame this as support for "RLVR... does not teach models new reasoning capabilities, but instead triggers latent ones" (Sec 2.2).

## Methodology
GRPO via OpenRLHF on DeepScaleR data, 300 steps. Binary rewards of five kinds: ground truth, majority vote of 64 base-model rollouts, \boxed{} format, Bernoulli(gamma) random, and majority-voted incorrect labels. The base models are Qwen2.5-Math-1.5B/7B, Qwen2.5-1.5B/7B, Llama3.1/3.2 (base and instruct) and OLMo2-7B (base and SFT). Evaluation is MATH-500 pass@1 and AMC average@8, with AIME in an appendix. The clipping analysis is analytic, with no-clip ablations over multiple seeds. The code-reasoning analysis uses string matching ("python") on outputs.

## Relevance to Proposal
**Score: 6**

1. The evidence is behavioural: benchmark accuracy and the frequency of a string in outputs. Nothing looks at activations, weights, circuits or readouts, and none of the claims is tested against a linear-readout notion of possession. The clipping-bias argument is an analysis of the gradient on token probabilities (an optimiser-level explanation, not a circuit-level one).
2. Their result could hold with our verdict going either way. Behaviourally "amplifying a high-prior behaviour" is compatible with our mechanistic "reuses C", where the update runs through an existing readout. It is equally compatible with "not through C", because a random-reward update could change the weights in directions unrelated to any readable direction while the output shifts toward a behaviour already in the base distribution. This is my inference, not something the paper tests. Their "elicitation" is behavioural (what the sampling distribution favours), which the project has explicitly rejected as its definition. The paper cannot rule out that a spurious-reward gain is mechanistically new, and it cannot show it is reused.
3. Directly usable:
   - The paper supports the project's shuffled-reward null arm. Our GRPO null should not be assumed inert. On Qwen, random rewards produce a real δW with accuracy gains driven by the clipping bias, even though the reward has no information. If our shuffled-reward arm shows a nonzero δW, this paper is the reference explanation. Whether the arm's δW is carried by VPD would be a useful comparator for the "real reward" arm.
   - Clipping is a testable lever. Ours could be run with clipping effectively off (single gradient step per rollout, as in their ablation) to give a null with zero expected signal.
   - Their finding that early steps (about 15 to 50) already carry most of the gain is relevant to saving δW at early steps.
   - The ground-truth arm reportedly diverges from the spurious arms later in training (code share falls). That suggests an early-versus-late mechanistic difference, which is a hypothesis to check.
   - Code reasoning is a known behaviour with a base-rate of 65% (Qwen2.5-Math-7B). It is a candidate C for a readout, but it is specific to Qwen2.5-Math and not shown for Qwen3-1.7B.
   - Caution for GSM8K on Qwen3: the models studied are Qwen2.5(-Math), not Qwen3-1.7B, and the tasks are MATH/AMC. The paper itself reports that gains from spurious rewards are model-dependent and smaller for small models, so transfer to our setup is uncertain.

## Limitations
- Behavioural evidence only; no mechanistic or internal analysis. The code-reasoning link is correlational plus prompt/reward interventions.
- No Qwen3 and no GSM8K. The authors say code reasoning is "a controlled and observable example... rather than an exhaustive characterization" (Sec 5), and other behaviours (for example repetition) also correlate.
- The clipping-bias explanation is validated on Qwen2.5-Math-7B only. The paper does not explain why Llama and OLMo fail beyond the "prior" conjecture, and the authors state family-level reasons only as conjectures.
- Several explanations, such as the incorrect reward acting as an implicit format reward, are hypotheses (footnote 1).
- The extracted text garbles figures. I did not trust bar-chart numbers in Fig 1 beyond those repeated in the prose.

## Key Quotes
> "RLVR training with GRPO improves MATH-500 performance for Qwen2.5-Math-7B in absolute points by 21.4% using randomly assigned rewards, nearly matching the 29.1% gained with ground truth rewards." (Abstract)

> "When clipping (the red term in Eq. 1) is removed and rewards are sampled independently of the responses, the expected training objective is zero, yielding no meaningful learning signal." (Sec 4)

> "RLVR, at least at the compute scales of open-source post-training pipelines (Lambert et al., 2024), does not teach models new reasoning capabilities, but instead triggers latent ones already present in the base model" (Sec 2.2)

## Not read
Appendices (A to K: derivation details, other random-reward gamma values, AMC and AIME results, code-suppression experiments, repetition, prompt sensitivity, already-RL'd models, origins of code reasoning) were not read. I read the main text, pages 1 to 9, and skimmed the start of the references. No Qwen3 results were seen.

## Comments Summary
Not applicable (arXiv paper).
