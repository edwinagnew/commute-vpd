---
slug: marks2026-persona-selection-model
title: "The Persona Selection Model: Why AI Assistants might Behave like Humans"
authors: Marks, Lindsey, Olah (Anthropic Alignment Science Blog, alignment.anthropic.com/2026/psm, 2026-02-23; LessWrong linkpost)
year: 2026
id: https://alignment.anthropic.com/2026/psm/
tags: [EH, DEFINITION, ALIGNMENT, CITE-FIX, SUPPORTS]
primary_tag: EH
in_our_bib: marks2026persona
cited_in_draft: yes (main.tex l.196, related work)
themes: [elicitation, reuse-definitions]
read_from: "full text, alignment.anthropic.com HTML (stored in ~/commute-data/lit_review_pdfs/)"
digest_buckets: [3-definition, 4-supports, 5-cite-fix]
importance: 7
---

# The Persona Selection Model (Marks 2026)

**What it does.** An essay, with no new controlled experiment, that names and argues for the persona selection model
(PSM): pre-training teaches an LLM to simulate many personas, and post-training refines the LLM's model of one of
them, the Assistant. The authors "claim no originality for the ideas" (citing Andreas 2022, janus 2022 "Simulators",
Hubinger et al. 2023, Shanahan et al. 2023, Byrnes 2024, nostalgebraist 2025 "the void"). Sections: (1) the model:
predictive models and personas, from base model to assistant (Askell et al. 2021 prompting), the formal statement and
five things PSM does *not* claim; (2) evidence: generalization, behaviour, interpretability, and "complicating
evidence"; (3) consequences for AI development: anthropomorphic reasoning, emotions, AI welfare, good AI role models in
pre-training data, tractable interpretability-based auditing, neuralese; (4) how exhaustive PSM is: a spectrum of views
(shoggoth, router, operating system; actors, authors and narratives), two conceptual reasons to expect PSM to hold
(post-training as elicitation; personas are a simple fit with an inductive bias to reuse), how this may change with
scale, and empirical observations; (5) conclusion with open directions; appendices on breaking character (Nasr et al.
2023 repeated-word divergence; a `{prompt}` query continued as API code) and on router-level deception (Alice/Bob
persona swapping).

**Key result.** No numbers of its own except one: on a coin-flip prompt continued in the *Human* turn, Claude Sonnet
4.5 says "heads" (its preferred task) 88% of the time and "tails" 1%, across 50 variants, while its base model sits
near 50%. Cited evidence. *Generalization:* emergent misalignment (Betley et al. 2025a; bad medical advice, Turner et
al. 2025, Wang et al. 2025, Chen et al. 2025; reward hacking, MacDiarmid et al. 2025), archaic bird names and the
Terminator backdoor (Betley et al. 2025b), inoculation prompting (Wichers et al. 2025; Tan et al. 2025), out-of-context
learning (Berglund et al. 2023; Hua et al. 2025). *Behaviour:* human self-descriptions (Chowdhury et al. 2025 on o3;
Project Vend), emotive language (Claude 4 system card; Gemini Team 2025), a caricatured "secret paperclip goal".
*Interpretability:* SAEs "typically transfer well" from base to post-trained models (Kissane et al. 2024, Lieberum et
al. 2024, He et al. 2024, Sonnet 4.5 system card s.7.6); Assistant-and-character shared SAE features (Templeton et al.
2024; Opus 4.5 system card s.6.4; a "panic" feature); a "toxic persona" SAE feature mediating emergent misalignment
(Wang et al. 2025); persona vectors (Chen et al. 2025); an "Assistant Axis" already present in the base models (Lu et
al. 2025). *Complicating evidence*, explained as limited capability or LLM-level "bugs": 9.11 > 9.9, Altered Riddles,
strawberry; within-context answer flips (Arcuschin et al. 2025); GCG adversarial strings (Zou et al. 2023). *For and
against "post-training as elicitation":* Guo et al. 2025 (LLMs struggle with novel ciphers) supports it; Donoway et al.
2025 (small models fine-tuned on hard chess puzzles acquire capabilities from scratch) goes against; Silver et al. 2017
shows RL from scratch learns non-persona agency. Alignment pretraining (Tice et al. 2026) supports the role-model
advice. *Against strict PSM:* post-trained-only features (Lindsey et al. 2024; Minder et al. 2025, "under 1%"), refusal
of harmful text outside chat format, and the coin-flip preference. The authors' own caveats and alternatives: PSM does
not rule out new capabilities (tool-call syntax) or claim exhaustiveness; the strict "fine-tuning = conditioning" view
"seems somewhat too strong for the empirical evidence"; the streetlight effect may bias interpretability evidence
toward PSM; the router view allows new post-training mechanisms that choose personas; scaled-up RL may weaken
"post-training as elicitation"; they "don't currently have good ways to contextualize" the extent or nature of novel
learning.

**Quote (the claim).** All copied from the HTML text (tags stripped; italics and bold lost):
- TL;DR: "We describe the persona selection model (PSM): the idea that LLMs learn to simulate diverse characters during
  pre-training, and post-training elicits and refines a particular such Assistant persona."
- Introduction bullets: "During pre-training, LLMs learn to be predictive models that are capable of simulating diverse
  personas based on entities appearing in training data: real humans, fictional characters, real and fictional AI
  systems, etc." / "Post-training refines the LLM’s model of a certain persona which we call the Assistant. When users
  interact with an AI assistant, they are primarily interacting with this Assistant persona."
- Statement of the persona selection model: "Informally, PSM views post-training as refining the LLM’s model of the
  Assistant persona: its personality traits, sense of humor, preferences, beliefs, goals, etc." Formally: "Pre-training
  teaches an LLM a distribution over personas. Implicit in this distribution are various hypotheses about the Assistant
  persona." / "Post-training can be viewed as updating this distribution using training episodes as evidence." / "This
  results in a posterior distribution over Assistant personas."
- Selection, in the authors' reading of Lu et al. 2025 (Evidence from interpretability): "Notably, this axis is not
  created during post-training: the same axis exists in the pre-trained counterparts to these models, where it appears
  to represent Assistant-like human characters." ... "This provides direct evidence that post-training selects a
  particular default region of a pre-existing persona space corresponding to “Assistant” behavior, and that this
  persona exists within a larger space of possible personas which can be accessed through contextual cues."
- Post-training as elicitation (How exhaustive is PSM?): "A common view among some AI developers is that little
  fundamentally new is learned during post-training. On this view, the role of post-training is mainly to elicit
  capabilities that the model already had."

**Quote (the caveat).**
- "PSM does not rule out learning of new capabilities during post-training. For example, no persona learned during
  pre-training knows how to use Anthropic’s syntax for tool calling; that capability is learned during post-training.
  PSM explains this as the LLM learning that the Assistant knows how to use this syntax." (Statement of PSM)
- "Caveat. Not all representations in post-trained models are reused from pre-training, as we discuss below.
  Additionally, it may be the case that reused representations are systematically more interpretable than
  representations that are learned from scratch during post-training. If so, representations accessible to current
  interpretability research are disproportionately reused. This would be a form of the streetlight effect, distorting
  our evidence to be overly supportive of PSM." (Evidence from interpretability)
- "Despite the evidence described above for substantial representational reuse between pretrained and post-trained
  models, post-trained models do not exclusively reuse representations from pretraining." ... "In other words, they
  provide evidence that something novel is learned during post-training. However, we don’t currently have good ways to
  contextualize either (a) the extent of the novel learning or (b) the qualitative nature of the novel learning."
  (Representations specific to post-trained models)
- "That said, it is very poorly understood how true it is that “post-training is just elicitation.”" and "Insofar as
  post-training can ever teach new behaviors and capabilities from scratch—and it likely can—we should expect that
  massively scaling up post-training will provide opportunities to implement non-persona agency" (How exhaustive is PSM?)

**Relevance to our paper.** PSM is usable as an instance of "pre-training creates X, post-training elicits it" with X =
personas, provided it is cited as a proposed model ("we describe", "argue"), not a result: the TL;DR's "post-training
elicits and refines a particular such Assistant persona" is the sentence to quote. Two cautions. First, the draft's
paraphrase "selects one persona from a distribution" misstates it: post-training yields "a posterior distribution over
Assistant personas", and PSM explicitly "does not assert the Assistant is a single, coherent persona". Second, PSM's
mechanistic evidence is representational (SAEs transfer, features shared by Assistant and characters, an axis present
in the base model): at most it shows the variables exist in both models, which is neither of our clauses. Our reuse
needs the variable to be needed and its mechanism kept up to a gain; PSM's "operating system" picture explicitly lets
"the persona submodules themselves change" while counting as reuse, and that is exactly the case our finding that
fine-tunes turn rather than scale components would read as new. So PSM and our turn-not-scale result are compatible at
the level of representations and disagree at the level of mechanisms, which is our gap sentence. The authors' own open
questions ("Which types of learning during post-training does PSM rule out?"; the extent and nature of novel learning)
are what our definition answers, and the streetlight caveat is a good citation for why we score on new-by-construction
fine-tunes. Guo et al. 2025 (ciphers are hard to learn) is cited as support for elicitation and bears on our cipher key
task.

**For the digest.**
- 3-definition: PSM is EH with X = personas ("post-training elicits and refines" the Assistant); quote it in the introduction as a proposed model, and contrast its representation-level evidence with our kept-and-needed mechanism definition.
- 4-supports: summarises representation reuse (SAE transfer; post-trained-only features "under 1%" in Minder et al.) and names the streetlight effect our known answers address; its own open question on the extent of novel learning is ours.
- 5-cite-fix: refs.bib `marks2026persona` lists only Marks, the short title and the LessWrong URL; the original is Marks, Lindsey, Olah, "The Persona Selection Model: Why AI Assistants might Behave like Humans", alignment.anthropic.com/2026/psm. main.tex l.196 "selects one persona" should read "updates a distribution over personas, eliciting and refining an Assistant persona".
- Cited works on reuse vs new missing from master_list.tsv: Guo, Sleight, Roger 2025 "All Code, No Thought: Current Language Models Struggle to Reason in Ciphered Language" (2510.09714); Donoway et al. 2025, chess-puzzle fine-tuning acquires capabilities from scratch (openreview Dkgx2pS4Ww; title not checked, OpenReview blocked the fetch); Lieberum et al. 2024 "Gemma Scope" (2408.05147) and He et al. 2024 "Llama Scope" (2410.20526), both SAE transfer to fine-tuned models; Templeton et al. 2024 "Scaling Monosemanticity" (persona SAE features that steer the Assistant).
- Not checked / source errors: the post links "Kissane el al., 2024" to arXiv 2406.17759 ("Interpreting Attention Layer Outputs with Sparse Autoencoders"), not to the "SAEs (usually) transfer" post its sentence describes (our `kissane2024transfer`); "Zhou et al. (2023)" in the adversarial-input paragraph is a typo for Zou et al. 2023. The cited system-card sections were not read.
