---
slug: janus2022-simulators
title: Simulators
authors: janus
year: 2022
id: https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators
tags: [EH, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: LessWrong post full text via the LessWrong GraphQL API (posted 2022-09-02); searched for every passage on RLHF, fine-tuning and InstructGPT
digest_buckets: []
importance: 2
---

# Simulators (janus 2022)

**What it does.** A conceptual essay proposing that a model trained on self-supervised prediction is best thought of as
a "simulator" whose outer objective is Bayes-optimal conditional inference over the training distribution, which can
instantiate many "simulacra" (agentic or not) depending on the prompt. It contrasts this with agent, oracle, tool,
genie and behaviour-cloning framings.

**Key result.** No empirical result. The central claims: the simulator is not the simulacra it runs ("none of these
simulacra are the algorithm itself"), and RL is the paradigm that "creates agents". Instruction following (InstructGPT,
the "genie modality") is described as "only an exploitable subset" of the patterns a predictor already has. The post
says almost nothing about what fine-tuning does to the model.

**Quote.** "GPT, insofar as it is inner-aligned, is a simulator which can simulate agentic and non-agentic simulacra." (Summary)

**Relevance to our paper.** Background for the persona-selection lineage (Waluigi, the Persona Selection Model,
Nadaf 2026) that frames post-training as choosing among characters a pretrained model can already simulate, which is an
informal, behavioural form of the elicitation hypothesis. It gives no definition of reuse and no evidence; cite at most
once in the planned persona-selection paragraph.

**For the digest.**
