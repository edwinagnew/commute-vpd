# The persona selection model

## Metadata
- **Authors**: Sam Marks (post written in the first person plural, acknowledging Anthropic colleagues; the source file lists only Sam Marks as author)
- **Year**: 2026
- **Source**: LessWrong / Alignment Anthropic blog (posted 2026-02-23, score 189; 22 comments)
- **DOI/URL**: https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model (mirror: https://alignment.anthropic.com/2026/psm/)

## Summary
The post names and argues for the *persona selection model* (PSM): pre-training teaches an LLM a distribution over characters (personas), and post-training mainly refines which persona, the "Assistant", the model simulates, so assistant behaviour is largely explained by that persona's traits. It surveys behavioural, generalisation and interpretability evidence for this view, and discusses how exhaustive PSM is, from a "masked shoggoth" (non-persona agency inside the LLM) to an "operating system" (all agency is persona-based) spectrum with a "router" view in between.

## Key Findings
- Post-training is framed as Bayesian-style conditioning on the Assistant persona: training episodes are evidence that upweight persona hypotheses consistent with them. The authors explicitly do not claim new capabilities are impossible to learn in post-training (e.g. tool-call syntax), only that the model still models the Assistant as an enacted persona.
- Interpretability evidence for representational reuse: SAEs typically transfer from pre-trained to post-trained models (Kissane et al. 2024; Lieberum et al. 2024; He et al. 2024); "persona" features that mediate emergent misalignment (Wang et al. 2025, "toxic persona" feature) also fire on pre-training quotes from morally questionable characters; persona vectors (Chen et al. 2025) can be decomposed into pre-training concepts; an "Assistant Axis" (Lu et al. 2025) exists in the pre-trained counterparts too.
- Authors concede that representations specific to post-trained models exist (Lindsey et al. 2024; Minder et al. 2025 report under 1% of features in their setting), often tied to refusal, false-information responses, emotion questions and chat-template tokens. They do not know whether these are extensions of the Assistant persona or from-scratch learning.
- Their own quantitative-ish probe is limited to one experiment: Claude Sonnet 4.5 assigns higher probability to its preferred coin-flip outcome in non-Assistant contexts, while the pre-trained base does not (Figure 6). This is presented as evidence against purely predictive post-trained models, with a persona-leakage or narrative explanation left open.
- The "shoggoth / router / actor / operating system" spectrum is offered as an informal taxonomy. The authors say the two most-discussed points are not exhaustive and that PSM may become less useful at larger RL scales.

## Methodology
A conceptual and position piece, with the empirical case assembled from published work (generalisation, interpretability, behavioural observations) plus a small number of the authors' own illustrative examples (completions from Claude models, one preference-vs-coin-flip measurement, Appendix examples). No new controlled experiment set out as a test of PSM is reported. Arguments rely on analogy (actors, authors, forelimb homology) and on "PSM-compatible" re-explanations of counter-evidence (the "complicating evidence" section).

## Relevance to Proposal
**Score: 6**

The proposal's central dichotomy, whether post-training *strengthens existing mechanisms* or *creates new ones*, is the question PSM takes a side on: it argues most post-training is persona reuse (elicitation / conditioning), with new mechanisms as a minority or as a "router". The representation-reuse evidence (base-to-chat SAE transfer, Minder et al.'s under-1% post-training-specific features) is directly relevant to Strand 2 (model diffing, feature transfer) and Strand 4 (reuse vs build circuits; emergent misalignment via persona features). It does not study parameter decompositions, low-rank updates or the pull-back of gradient steps, so it does not inform the commuting map itself (Strands 1, 3, 5).

## Limitations
- Largely informal and analogy-driven; the authors say the discussion "relies heavily on evocative analogies" and that there is no established definition of agency.
- Several "evidence" items are retrodictive and PSM-compatible with alternatives. Comment from leogao (score 16) argues most examples could have been spun either way; Sam Marks replies that EM data is egregious, not accidental, and that the generalisation evidence still points one way.
- Representation-reuse evidence may be biased: the authors flag a streetlight effect, since reused features are easier to find with current SAEs.
- Mostly based on Anthropic models and Anthropic system cards, which are not independently reproducible. Authors acknowledge this is partly anecdotal.
- No operational criterion distinguishes "strengthened persona" from "new mechanism", which is the distinction the proposal needs to measure.
- Comment from agastyasridharan (score 14): replicating the coin-flip result on OLMo 32B checkpoints (base, SFT, DPO, RL) gives a shift concentrated at DPO, but controls (an "AI assistant", "a human named Jack", "an unnamed person") show a much weaker effect, so the original result may partly reflect Assistant-role cueing. This is a community replication, not peer-reviewed.

## Key Quotes
> "Post-training can be viewed as updating this distribution using training episodes as evidence." (Statement of the persona selection model)

> "Most importantly for PSM, we find that LLMs use the same internal representations to characterize the Assistant as for other characters present in training data." (Evidence from interpretability)

> "Insofar as post-training can ever teach new behaviors and capabilities from scratch—and it likely can—we should expect that massively scaling up post-training will provide opportunities to implement non-persona agency." (How might these considerations change?)

> "Despite the evidence described above for substantial representational reuse between pretrained and post-trained models, post-trained models do not exclusively reuse representations from pretraining." (Empirical observations: representations specific to post-trained models)

## Comments Summary (LessWrong post)
Twenty-two comments, several substantive:
- **Scale and RL (Daniel Kokotajlo, 20; Sam Marks reply)**: asks how PSM varies as RL scales, and whether AI 2027's "distorted identity" story differs from PSM. Marks says the Assistant persona may gradually stop resembling any pre-training persona (the "homologous forelimb" analogy); Kokotajlo agrees. This is the "does post-training create new mechanisms over time" question the proposal cares about.
- **Replication (agastyasridharan, 14)**: the coin-flip effect is much weaker with human and generic-person controls; the largest shift is at DPO (see Limitations).
- **Falsifiability (leogao, 16; Sam Marks reply)**: most evidence is retrodictive; Marks argues the EM and archaic-bird data point in one direction. leogao asks for pre-registered predictions.
- **Router and continuum (eggsyntax, 6; Sam Marks reply)**: asks why there are only "big shoggoth" and "small router" options; Marks says the list is not exhaustive. RogerDearnaley notes that conditional persona shifts are easy to train (Betley et al. 2025b), so a router could be built deliberately.
- **Character-trained models (Jan_Kulveit, 11)**: says PSM "does not fit character trained models that well" and that the post is 1–2 years behind frontier understanding.
- **Weight-level extensions (Vic Peters, Gabriel Ting, Girard Dorney, genericname-2)**: speculative questions on whether persona "basins" predict flip behaviour, inoculation for persona clusters, and whether distillation transfers all personas or just the Assistant.
- **Persona-based routing (jbco, 3)**: asks whether routing among overlapping system-prompt rules is persona-mediated or procedural, with distinguishable failure modes.
