---
slug: <slug from master_list.tsv>
title: <full title>
authors: <first author et al.>
year: <year>
id: <arXiv ID without version, or URL>
tags: [<one or more, UPPERCASE, from this fixed list only:
  EH          elicitation / re-weighting / superficial-alignment hypothesis (does post-training reveal or create)
  BASELINE    a method we compare against, or should (model diffing, crosscoders, steering, attribution, probes)
  KNOWN-ANSWER a realistic fine-tune with independent evidence of reuse or of something new
  DEFINITION  a definition of reuse, mechanism, elicitation or capability to cite or contrast
  DIFFING     model diffing / base-vs-fine-tune comparison
  FT-INTERP   mechanistic interpretation of what a fine-tune or RL run changed
  DECOMP      parameter / weight decomposition (APD, SPD, VPD, SAEs on weights)
  CAUSAL      causal abstraction, interchange interventions, circuits
  DYNAMICS    training dynamics, features across checkpoints
  SUPPORTS    a finding that supports one of ours
  CONTRADICTS a finding that contradicts one of ours
  ALIGNMENT   safety application (emergent misalignment, reward hacking, unlearning, personas)
  CITE-FIX    our draft cites it wrongly, or should cite it and does not
  BACKGROUND  context only; little direct bearing>]
primary_tag: <the single tag that best says why a reader of our paper would open this entry>
in_our_bib: <bibkey if in the draft's refs.bib, else "no">
cited_in_draft: <yes / no / in refs.bib but not \cite'd>
themes: [<one or more of: elicitation, model-diffing, interpreting-finetunes, reuse-definitions, baselines, decompositions, causal-abstraction, training-dynamics, other>]
read_from: <what you actually read: "arXiv HTML abstract+method+results", "PDF pages 1-9", "LessWrong post full", "abstract only (full text unreachable: why)">
digest_buckets: [<zero or more of: 1-missing-method, 2-known-answer-case, 3-definition, 4-supports, 4-contradicts>]
importance: <0-10: how much our paper must engage with it; 8+ = the paper is incomplete without it>
---

# <Title> (<First author> <year>)

**What it does.** 2-4 sentences: the question, the setting (models, data, fine-tune or training regime), the method.

**Key result.** 2-4 sentences with the headline numbers as the paper reports them.

**Quote.** "<one short verbatim sentence from the paper that carries the result or definition>" (section or page)

**Relevance to our paper.** 3-6 sentences. Say concretely which of our claims or methods it bears on and how:
is it a method we should compare against? A realistic fine-tune with independent evidence (from the paper itself) that
the fine-tune reuses an existing base-model mechanism or builds a new one, so usable as a known-answer case? A
definition of reuse/elicitation to cite or contrast with ours? A finding that supports or contradicts ours (e.g. do
fine-tunes rescale existing components, or rotate/replace them)? If it is of little use, say so in one sentence.

**For the digest.** One line per bucket it was placed in: `<bucket>: <what it means for us, one line>`.
