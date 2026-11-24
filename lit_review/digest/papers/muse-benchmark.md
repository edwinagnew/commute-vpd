---
slug: muse-benchmark
title: "MUSE: Machine Unlearning Six-Way Evaluation for Language Models"
authors: Shi et al.
year: 2024
id: 2407.06460
tags: [ALIGNMENT, BACKGROUND]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "identified from the name (MUSE, Shi et al.); arXiv abs page + arXiv HTML: intro, section 2 (criteria), setup of retrained and target models"
digest_buckets: []
importance: 1
---

# MUSE: Machine Unlearning Six-Way Evaluation for Language Models (Shi 2024)

**What it does.** Scores unlearning on six properties: no verbatim memorisation, no knowledge memorisation, no privacy
leakage, utility on retained data, scaling with the size of the forget set, and sequential requests. A 7B base model
that never saw the corpus is fine-tuned on BBC news or the Harry Potter books (target model) and on the retain part
only (retrained model, the reference); eight methods (gradient ascent, NPO and regularised variants, task vectors,
Who's Harry Potter) are compared.

**Key result.** Most methods remove verbatim and knowledge memorisation to varying degrees, only one avoids severe
privacy leakage, and the effective ones (NPO, task vectors) cost much utility and fail on large or sequential requests.

**Quote.** "existing algorithms fail to meet deployer's expectations because they often degrade general model utility"
(Abstract)

**Relevance to our paper.** Almost none. Like TOFU, the content is taught first by a fine-tune, so a decomposition of
the base model does not cover it, and all criteria are behavioural. Only the retrained-model reference is a
transferable idea.

**For the digest.** (no buckets)
