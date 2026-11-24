# Brief: build the ranked digest and the index (cell P2.71, due 11-25)

Writing only: no runs, no paper edits, do not commit (the lead commits). Inputs:
- `lit_review/digest/papers/*.md` (~237 per-paper entries, frontmatter with tags, primary_tag, importance,
  digest_buckets, in_our_bib, cited_in_draft; body with what/result/quote/relevance/for-the-digest lines).
- `lit_review/digest/_paper_context.md` (our paper), `_paper_refs.tsv` (our bib), `master_list.tsv`,
  `_followups.tsv` (papers named by readers that are not in our list).
- The persona-features paper (Wang et al. 2025, 2506.19823) was digested separately for whip-lead: find that digest
  with `command grep -rl 2506.19823 ~/commute-data/agents/whip-lead/` and write a short entry
  `papers/wang2026-persona-features-em.md` from it (read_from: "whip-lead's digest + arXiv abstract"; fetch the abstract
  yourself).
All paths below are under `/mnt/nw/home/e.agnew/commute-agents/litreview-lead/lit_review/` unless absolute.

## 0. Clean-up first
- Known duplicates to merge (keep one entry, note the other slug as an alias in it, delete the other file):
  `transplanting-misalignment-persona` = `drake2026-lora-recruits-sft-builds` (2607.04510);
  `meloux2025-many-algorithms-one-network` = `meloux2025-identifiable` (2502.20914);
  `lw-easy-hard-earlier-post` → keep as superseded pointer to `easy-hard-2602-07852` (fine to keep both, mark it);
  `lw2025-weight-diff-svd-monitoring` is a linkpost of `zhong2026-watch-weights`; `bushnaq2025-apd-lw` vs
  `braun2025-apd-paper`; `lw-soligo-convergent-em-post` vs `soligo2025-convergent-em`. Find any others by arXiv ID.
- Every master_list slug must have an entry; list any missing in your hand-back.
- Normalise frontmatter: tags only from the template's list (`_template.md`), UPPERCASE; buckets in the form
  `1-missing-method`, `2-known-answer-case`, `3-definition`, `4-supports`, `4-contradicts`, `5-cite-fix` (add
  `5-cite-fix` where the entry's CITE-FIX tag or text says the draft cites wrongly or should cite).

## 1. `master_list.tsv` becomes the one index (user OK'd restructuring, 2026-11-23)
Rewrite it with columns: `slug  title  authors_short  year  id_or_url  primary_tag  tags  importance  in_our_bib
entry  old_summary  source_files`, sorted by importance descending then slug. `entry` = `papers/<slug>.md`.
Titles corrected where readers found the real title.

## 2. `digest/index.md`: browsable, grouped by theme
- Top: 5-line "how to use" (one file per paper in `papers/`; grep by tag, e.g. `command grep -l 'primary_tag: BASELINE'
  papers/*`; the tag list with one-line meanings).
- Then one section per theme the user named: elicitation / re-weighting hypothesis (EH); model diffing (DIFFING,
  BASELINE); interpreting fine-tunes (FT-INTERP, KNOWN-ANSWER); definitions of reuse (DEFINITION, CAUSAL); baselines;
  decompositions (DECOMP); then training dynamics, alignment applications, background. A paper appears under each
  theme its tags place it in. Each line: `- [Title](papers/slug.md) (Author year, ID) · imp N · TAGS: one-line why`.
  Within a section sort by importance.

## 3. The ranked digest: `/mnt/nw/home/e.agnew/commute-data/agents/whip-lead/lit-digest-2026-11-24.md`
Opens with `P2.71, due 11-25:` and a plain 3-sentence paragraph (what was read, how many, the three things the paper
most needs to act on). Then the five buckets in this order, each ranked by importance to the current paper:
(1) methods we should compare against but don't; (2) realistic fine-tunes with independent evidence of reuse or new
(known-answer candidates); (3) definitions of reuse/elicitation to cite or contrast; (4) findings that support or
contradict ours (split 4a supports / 4b contradicts; especially rotate vs rescale); (5) wrong or missing citations in
the paper (quote the draft file and line where the entry names it; verify against
`~/commute-agents/writeup-lead/paper/paper/` with `command grep -n`).
Each item is ONE line: `**Author year (arXiv ID)** — "short verbatim quote" — what it means for us.` Take quotes from
the entries; do not invent. Judge importance yourself across entries (readers' scores were per batch); be critical: an
entry that overclaims (value-level evidence sold as mechanism) is ranked down and says so. Within (1) say for each
whether it can run on our 67M setting. Within (2) name model, fine-tune and the evidence type (mechanism-level vs
activation/value-level vs behavioural only). No item more than once per bucket; a paper may appear in several buckets.
End with "Not read" (entries with read_from "not found" or "abstract only") and "Follow-ups" (from `_followups.tsv`
and any paper the entries say should be checked but is not in our list).

## 4. `/mnt/nw/home/e.agnew/commute-data/agents/whip-lead/lit-digest-2026-11-24-top10.md`
The 10 most important items across all buckets, one line each in the same format, each tagged with its bucket. Write
this file FIRST (before the full digest) and say in your hand-back that it is ready.

Use `command grep` for regex; Python runs only via `~/commute-agents/fleet/cpu-run` (pipe a script on stdin:
`cpu-run python3 - < script.py` does not work from /ephemeral paths — put any helper script under
`/mnt/nw/home/e.agnew/commute-agents/litreview-lead/lit_review/digest/` and delete it after). Hand back: paths, counts
(entries, merged, missing), the top-10 list verbatim, and a Friction line.
