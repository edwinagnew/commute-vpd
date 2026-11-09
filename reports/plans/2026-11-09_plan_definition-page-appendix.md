# Appendix to the definition page: design check, sources and rows

*definition-lead, 2026-11-09, second version. Companion to `2026-11-09_plan_definition-page.md`. The page proposes four
runs; none is in the table, so nothing here runs.*

## Design check

- **Goal.** `GOALS.md` asks whether post-training teaches new capabilities or strengthens existing ones. This gives that
  question a label part 2's methods can be scored against, independent of the decomposition and of those methods.
- **Nearest tried.** `experiments/TRIED.md`: E116 (frozen readout, no verdict on the old tasks), E112/E113/E114 and E125
  (case and repeat label: readout passes, direction removal does not), E62/E97 (copying head switch-off), E99 (forced
  and forbidden copy fine-tunes), E100 (sum of single reverts). None trains with the body frozen, and `grep` for it in
  `decisions.md`, `backlog.md` and `TRIED.md` finds nothing. The masked training code exists (`experiments/definition_verify/arms.py`).
- **Patch or step.** A step: the plan's part 1. It replaces the failing readout step rather than adding a clause to it.
- **Decisions relied on or departed from.** Keeps `decisions.md` row 87 (held-out carrying) and the 0.9 bar. Departs from
  rows 81 and 164 (the readout and its direction removal). The reset of 2026-11-06 warned that each added rule weakens
  the claim of using someone else's definition; the user withdrew that frame on 2026-11-09, and this wording has fewer
  clauses than either earlier one. All of it needs the user's dated row.
- **Reads under an adopted definition?** No. Until a dated row adopts a wording, no verdict is given; the re-read of the
  copy results says so.
- **What result changes what we do.** Run 1 failing stops the line. The copy pair reading "new" on two of three seeds
  drops the wording. Run 4 finding no separating task means the wording is carrying under another name.
- **Control first.** Run 1, alone: the task the original model already does (W0 reading near 1), and the random-weight
  body (at chance). It can fail by the last block being able to learn the task on noise, or by the training not moving.

## Numbers for each run

| Number | At M | Do-nothing | Control |
|---|---|---|---|
| Frozen-body accuracy on unseen values | M's own, near its prior (0.25 on the copy windows) | same as M, under 0.9 | near 1 on the task M already does; chance on the random body |
| F's accuracy on unseen values | near prior | same as M, so never read as carrying | saved, per row below |

**Edge cases.** A task with no unseen values has a table rule and reads "new" under clause 4, stated as such. A rule with
unseen values and a fine-tune at chance reads "new". A zero or near-zero gain fails carrying and needs no share. Ties
read under the bar. The twin is at chance on unseen values by construction; it is a check against memorising being read
as reuse, not a test of the frozen body.

## The 17 kinds, and what each reads

Carrying is accuracy on unseen values, from `experiments/definition_table/results/stage1.json` (E116), the case-label
`experiments/case_label/results/stage2.json` (`.states.<state>.P1`) and the repeat label
`reports/analyses/2026-11-06_analysis_e125-fresh-reuse-pair.md` lines 123-128. Rule "none" means the task defines no
answer for an unseen value.

| Kind | Built as | Rule on unseen values | Carrying (seeds) | Reading from saved numbers |
|---|---|---|---|---|
| In-context copy | reuse | yes | 0.995, 0.995, 1.000 | needs the frozen-body training |
| Copy, 24 keys | reuse | yes | 0.95, 1.00, 0.98 | needs the frozen-body training |
| Identity, 32 keys | reuse | yes, but no unseen key read yet | not read | needs a read on unseen tokens and the frozen-body training |
| Case label, real | reuse | yes | 0.998, 0.996 | needs the frozen-body training |
| Repeat label, real | reuse | yes | 0.970, 0.960 | needs the frozen-body training |
| Lookup twin | new | none | n/a | new |
| Random map, 32 keys | new | none | n/a | new |
| Text cipher, 50 words | new | none | n/a | new |
| Text cipher, 500 words | new | none | n/a | new |
| Cipher, 24 keys | new | yes | 0, 0, 0 | new |
| Lower-guess map | open | yes | 0.02-0.03 | new |
| Rank-1 guess map | reuse (strengthening) | yes | 0.64-0.69 | new, against how it was built |
| Case-label twin | new | shuffled | 0.506, 0.516 | new |
| Repeat-label twin | new | shuffled | 0.496, 0.482 | new |
| Lowercase | reuse | excluded as no content function by the frozen wording; I leave that exclusion | not read | does not apply |
| Emoticon | an existing behaviour | constant answer | not read | does not apply |
| One-pair map | new | constant answer | not read | does not apply |

Rows reading by candidate, from saved results: the frozen wording reads 4 (`reports/analyses/2026-11-06_analysis_case-label-line-summary.md`
and the E125 report; E116 gave 0 of the 13 others); switch-off reads the copy and lookup rows only (E62, E97); the sum
of single reverts reads all 17 (E100, E114). The 94% for the forbidden copy fine-tune is
`reports/analyses/2026-11-05_analysis_e99-definition-verify.md` line 258, provisional.

## Papers

Jain et al. 2024 and Prakash et al. 2024: read in full from `~/commute-data/lit_review_pdfs/` (the corrected account is
on the page); Jain's appendix notes are `~/commute-data/scratch/definition-lead/p1.1/jain-appendix/notes.md`. Kumar et
al. 2022 was read from the project's summary, `lit_review/summaries/arxiv_2202.10054v1.md`, not the PDF. Raghu et al.
2020 (arXiv:1909.09157, "Rapid learning or feature reuse?") is cited from memory and unread; it is not in the PDF folder
and it must be read before the page is cited anywhere else.

## Saved for audit, if the runs are added

Per seed: frozen-body accuracy on fit and unseen values, the random-body reading, step counts. Update sizes beside. The
frozen-body states in `results/large/`.
