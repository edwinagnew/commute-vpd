---
slug: gekhman2024-new-knowledge
title: "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?"
authors: Gekhman et al.
year: 2024
id: 2405.05904
tags: [EH, DYNAMICS, SUPPORTS, CITE-FIX]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, training-dynamics]
read_from: "arXiv HTML: abstract, intro, sections 2-4 (setup, SliCK categories, unknown vs known fitting, linear model), Table 2, discussion (incl. Superficial Alignment Hypothesis paragraph)"
digest_buckets: [4-supports]
importance: 5
---

# Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? (Gekhman 2024)

**What it does.** A controlled closed-book QA study (EntityQuestions, 12 relations, PaLM 2-S base) that varies the
share of fine-tuning examples whose facts the base model does not know. Knowledge is labelled per example by SliCK:
from 10 random 4-shot prompts with greedy and sampled decoding, an example is HighlyKnown, MaybeKnown, WeaklyKnown or
Unknown (never answered correctly). Training dynamics are tracked per category, and test accuracy is regressed on the
numbers of Known and Unknown examples fitted.

**Key result.** Unknown examples are fitted much more slowly than Known ones; at the early-stopping point the model
fits most Known and few Unknown examples. Test accuracy is linear in what is fitted (in-distribution β_kn = 7.3,
β_unk = -8.3, R² = 0.86; out-of-distribution R² = 0.95): fitting Unknown examples raises hallucination about
pre-existing knowledge. Fine-tuning only on HighlyKnown is not best; including MaybeKnown matters. Relabelling Unknown
examples "I don't know" raises accuracy on answered questions from 43.0 to 61.8.

**Quote.** "large language models mostly acquire factual knowledge through pre-training, whereas fine-tuning teaches
them to use it more efficiently" (Abstract)

**Relevance to our paper.** Behavioural support for EH from a cleaner design than LIMA's: the fine-tune's targets are
split by whether the base model already has them, and the "new" part is learned slowly and harms the rest. It is
missing from the draft's post-training-as-selection paragraph and should be cited there. It also bears on our finding
that the base model's prior is a strong confound: SliCK is a base-prior label, and here prior and outcome coincide, the
case our cued-copy and prior-matched-cipher pairs pull apart. Its design (fine-tune on facts the base model knows vs
facts it does not) is a natural candidate for a realistic known-answer pair, but the paper gives no internal evidence
that Known facts run through the base model's recall mechanism, so the label is behavioural only. Its point that
fine-tuning teaches more than format (MaybeKnown examples matter) is a nuance worth one clause.

**For the digest.**
- 4-supports: fine-tuning mostly exposes pre-existing knowledge and learns new facts slowly and harmfully; behavioural support for EH, and a base-prior label we could adopt for a known-vs-unknown fact pair.
