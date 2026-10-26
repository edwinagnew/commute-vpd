# Towards Developmental Interpretability

## Metadata
- **Authors**: Jesse Hoogland (post credited to Hoogland; written as the outline of the Timaeus / SLT & AI alignment summit agenda)
- **Year**: 2023 (posted 2023-07-12)
- **Source**: LessWrong (score 197 at time of reading; 7 comments)
- **DOI/URL**: https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability

## Summary
The post sets out "developmental interpretability", a research agenda that studies how internal structure forms through phase transitions during neural network training, using observables from Singular Learning Theory (SLT) such as the RLCT and singular fluctuation. It is a conceptual agenda with a conjecture and a plan, not an empirical result: the authors state that whether phase transitions are a useful organising principle for interpretability is an open question they intend to test.

## Key Findings
- The authors claim that phase transitions during training are the right unit for incremental interpretability ("one phase transition at a time"), motivated by Olsson et al. (induction heads) and Elhage et al. (toy models of superposition). This is a working hypothesis, explicitly not established: they say "enough" structure emerges through transitions to make this a valid organising principle.
- They conjecture that training-time phase transitions and phase transitions of the Bayesian posterior (as described by SLT) share a common cause, the geometry of the loss landscape. They say this was verified by hand only for toy models of superposition.
- They list four "Axes of Failure" that could sink the agenda: transitions too infrequent (most structure forms gradually), too frequent (no triage), too large (transitions involve most of the model), and too isolated (transitions do not integrate into a quantitative account of the final model). They cite locality results (Li et al. 2021 intrinsic dimension; Gur-Ari et al. 2018; Panigrahi et al. 2023) as partial evidence against the "too large" failure.
- Near-term plan (as of July 2023): Phase 1 is sanity checks over a library of training-time transitions, including checkpoint-level mechanistic analysis and classification of weights into state, control and irrelevant variables. Phase 2 is building new tools. No results are reported in the post.

## Methodology
Conceptual and position-style. The post reviews prior literature on phase transitions in training, summarises SLT (free energy formula, RLCT as a measure of effective dimension at singularities), and lays out a research agenda with open questions and a phased plan. Comparisons are drawn to statistical physics (universality classes), developmental biology, and the Curry-Howard correspondence as an analogy for reading a network's "transcript" of transitions as deduction rules. No experiments are reported.

## Relevance to Proposal
**Score: 3**

This is a broad conceptual agenda about tracking structure formation across training checkpoints, which touches strand 6 (mechanism stability across training) and loosely strand 2 (training dynamics of features, continual interpretability). It does not involve parameter decompositions (VPD/SPD/APD), fine-tuning updates, or any way of mapping an update onto an existing decomposition, so it provides background framing rather than a method the proposal could use directly. Its strongest connection is the claim that training-time structure changes should be tracked and localised to a subset of weights, which is the motivation behind commuting updates past a decomposition.

## Limitations
- No empirical results in the post; the central claim (that enough structure forms in phase transitions) is stated as a hypothesis to be tested.
- The link between SLT's posterior-based phase transitions and SGD dynamics is acknowledged as unclear; the authors describe the relationship as conjectured.
- Relies on the RLCT and singular fluctuation as detectors, which the authors say are not yet shown to give finer-grained information than loss-based measures.
- Dated forecast (July 2023). The plan and the status of the Phase 1 examples should be checked against later Timaeus outputs before being cited as results.
- Commenters flag the emphasis on phase transitions as a possibly narrow choice (see Comments Summary).

## Key Quotes
> "Developmental interpretability studies how structure incrementally emerges through phase transitions during training." (Introduction)

> "We don't believe that *all* knowledge and computation in a trained neural network emerges in phase transitions, but our working hypothesis is that *enough* emerges this way to make phase transitions a valid organizing principle for interpretability. Validating this hypothesis is one of our immediate priorities." (Why phase transitions?)

> "Too infrequent: it turns out that only a small fraction of the important structures in trained neural networks form in phase transitions ... Too large: it turns out that many transitions are irreducibly complex and involve most of the model, so we're back to square one and have to reinterpret the whole network every time." (Reasons it won't work)

## Comments Summary (LessWrong post)
- **Bogdan Cirstea (8) and Hoogland's reply**: Recommends analysing several intermediate checkpoints rather than only the final one (citing MultiBERTs, Pythia, Mistral checkpoints). Hoogland agrees and says the distinctive part of devinterp is the emphasis on SLT-defined phase transitions.
- **Mikhail Samin (4)**: Asks whether transitions (e.g., grokking) can be predicted before they happen, and raises the worry that structure could be implemented in superposition well before it affects loss.
- **Daniel Murfet (10)**: Says training and test loss are lagging indicators; in toy models, RLCT indicated a transition earlier. He guesses that the information needed to understand the structure is present at the transition itself, so SGD need not be "rewound" step by step.
- **johnswentworth (25)**: Argues the post undersells the case. Within a phase, change is slow and forecastable; phase transitions are where extrapolation fails, so they are strategically crucial to detect for safety.
- **Vanessa Kosoy (34)**: Calls devinterp one of the more interesting alignment agendas for its head-on approach plus theory-empirical balance. Two reservations: (i) phase transitions should not be the central object; SLT's account of generalisation is the more important payoff; (ii) two missing pieces: the link between ideal Bayesian inference and SGD (perhaps via MCMC-type relaxation), and what is special about ANN architectures relative to arbitrary singular model classes.
- **Others**: Housekeeping (expired Discord link, rotten Watanabe link, Review Bot notice). Not substantive.
