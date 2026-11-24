---
slug: hewitt2019-control-tasks
title: "Designing and Interpreting Probes with Control Tasks"
authors: Hewitt & Liang
year: 2019
id: 1909.03368
tags: [BASELINE, BACKGROUND]
primary_tag: BASELINE
in_our_bib: hewitt2019control
cited_in_draft: yes
themes: [baselines]
read_from: "arXiv HTML: abstract, intro (definition of control tasks and selectivity, headline numbers), section 3-4 headings and results paragraphs"
digest_buckets: []
importance: 3
---

# Designing and Interpreting Probes with Control Tasks (Hewitt 2019)

**What it does.** Asks whether a probe's accuracy reflects the representation or the probe's own capacity to learn
the task. A control task assigns each word type a fixed random output, so it has the same input and output space as the
linguistic task but can be learned only by memorisation. A probe's selectivity is linguistic-task accuracy minus
control-task accuracy. Tested on ELMo layers with part-of-speech tagging and dependency edge prediction, across probe
families and regularisers.

**Key result.** With default settings MLP probes are barely selective (POS 97.3 vs control 92.8, selectivity 4.5);
linear probes reach similar accuracy with much higher selectivity (97.2 vs 71.2, 26.0). Dropout does not reliably
raise MLP selectivity; small hidden sizes, fewer samples and weight decay do. ELMo layer 1 is slightly more accurate
than layer 2 on POS but layer 2 is more selective (31.4 vs 26.0).

**Quote.** "a good probe, (one that reflects the representation), should be selective, achieving high linguistic task
accuracy and low control task accuracy" (Abstract)

**Relevance to our paper.** Cited in the departures appendix for the random-label control our readout test lacked;
the citation is apt, though a Hewitt control task is a per-word-type random mapping rather than shuffled labels, and
the appendix could say "control task" rather than "random-label control" to match. The selectivity idea is the same
logic as our planted-control and random-update floors, and it applies to the results paragraph where "a probe of the
answer alone also gets 21 of 21": that is a selectivity failure in Hewitt's sense and could be named as such. Otherwise
of little direct use.

**For the digest.**
- (no bucket)
