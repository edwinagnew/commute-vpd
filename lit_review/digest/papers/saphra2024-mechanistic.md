---
slug: saphra2024-mechanistic
title: "Mechanistic?"
authors: Naomi Saphra and Sarah Wiegreffe
year: 2024
id: 2410.09087
tags: [DEFINITION, BACKGROUND]
primary_tag: DEFINITION
in_our_bib: no
cited_in_draft: no
themes: [reuse-definitions, other]
read_from: "arXiv HTML v1: abstract, Sec. 1, 2 (2.1 causality, 2.2 coinage), headings and opening of Sec. 3, Sec. 4"
digest_buckets: [3-definition]
importance: 3
---

# Mechanistic? (Saphra 2024)

**What it does.** A short position and history paper (BlackboxNLP 2024) on what "mechanistic" means in interpretability.
It separates four uses of the term and traces how the term drifted from a technical meaning to a label for a community.

**Key result.** The four definitions are:
- narrow technical: "understanding neural networks through their causal mechanisms";
- broad technical: "any research that describes the internals of a model, including its activations or weights";
- narrow cultural: research from the MI community;
- broad cultural: any LM interpretability research.

The narrow technical sense excludes behavioural input-output testing and non-causal methods, such as correlating
activations with inputs. The paper notes a still narrower causal-abstraction reading (Geiger et al.; Mueller et al.) that
demands a complete end-to-end causal pathway, and says it "has not yet been widely adopted".

**Quote.** "The narrow technical definition of MI thus describes research that discovers causal mechanisms explaining all
or some part of the change from neural network input to output at the level of intermediate model representations."
(Sec. 2.1)

**Relevance to our paper.** Our title says "A Mechanistic Foundation", and our gap statement contrasts tests that read
activations or behaviour with one that "looks inside". This paper gives the citation to say which sense we mean. Ours is
the narrow technical one, in its causal-abstraction form: mechanisms are the structural functions of featurized variables.
In Saphra and Wiegreffe's terms, the existing elicitation evidence we criticise is "broad technical" at best, since it
reads activations without a causal claim. Geiger et al. 2025 cite it for the same point. It says nothing about
fine-tuning.

**For the digest.**
- `3-definition`: cite it to pin "mechanistic" in our title and claims to the narrow technical (causal-mechanism) sense, as opposed to "any look at internals".
