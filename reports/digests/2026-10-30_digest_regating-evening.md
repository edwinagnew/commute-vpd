# Re-gating lead: evening digest (2026-10-30)

**Why I stopped:** everything left on the re-gating line waits on your answer to ask #16, "what does elicitation mean".
Its card in the log is `decision-elicitation-means` (v152).

## In plain words

On the small model, VPD's re-gating share reads high when the fine-tune trains an answer the original model already
gives for each key. It reads low otherwise, and that includes a new rule that generalises to unseen keys. So the share
separates "answer already there" from "answer new", but it does not detect whether an update generalises. And a cut in
the original model's own prior still sorts every map the same way, so VPD has not yet shown anything the logits don't.

## What ran ($0, two cluster debug jobs)

- **Synthesis fixed.** Every error and overclaim in the independent review is fixed, and a second check said "ready".
  The plain answer now says the share agrees with the original model's prior
  (`reports/analyses/2026-10-30_analysis_regating-synthesis.md`). In log v149.
- **E58 Amendment 6** (job 46489): a lookup whose answers the original model rates about as likely as the copy map's
  (prior 3.7e-5 against 5.9e-5). The share reads 0.37-0.38, the low band, where copy reads 0.77. So the share is not a
  smooth function of the prior, but a cut between those two priors still matches it. The first-letter rule did not carry,
  so its share was not read. Checked, fixed, in log v150.
- **E58 Amendment 7** (job 46673): a 4-class rule built from the original model's embedding directions. It carries to
  unseen keys (plain carry 0.40-0.49, about half of copy's), while its shuffled twin, which has no rule, does not. The
  share reads 0.23-0.25 for the rule, the same as the twin: a new rule that generalises reads new. The granularity limit
  is neither confirmed nor ruled out, because the update's mechanism was not measured. Checked, fixed, in log v152.

Details are in the E58 card and `reports/analyses/2026-10-30_analysis_e58-open-pair.md` (Amendments 6 and 7), and in
`notebooks/E58_open_pair.ipynb`.

## Pending decisions

- **#16 (replaces #15):** elicitation as (i) the trained answer is one the original model already gives, which is what
  the share reads, or (ii) the update reuses a capability that generalises, which is what the behaviour label "carries"
  reads. Fable advises deciding this before any Qwen spend.

## Next, once #16 is answered

Fable's step 2 (`reports/plans/2026-10-30_advice_regating-next-after-a7-FABLE.md`): a small key-value table on Qwen. Its
rows are the original model, cipher, copy and elicit, with a prior column and two seeds. It uses the small model's
gain-based share, not E57's, which read 0.99 on random changes. It costs $0 on the cluster or about $5 on an H100, it is
not gated, and it will be planned with the Qwen line's lead.

## Money, blocked agents, friction

- **Money:** $0 this session. No pods were started.
- **Blocked:** `blocked.txt` shows baseline-lead waiting at a question since 15:15. It is not mine.
- **Friction noted:** #258-267.
  - jq's default indent rewrites notebooks.
  - Fable named the wrong file for E58's map builders.
  - The pod-runner commit line fails on an untracked result file.
  - Agents still run python3 on the dev node despite the brief.
  - The shared venv imports another clone's `src` unless `PYTHONPATH=src` is set.
