# Update Decomposition (repo `commute`)

Can a small gradient update from post-training be *commuted past* a parameter decomposition? Given a VPD
decomposition of a model's weights into rank-1 subcomponents (Bushnaq, Braun, Clive-Griffin et al. 2026,
`../papers/pd/vpd_blog_post.md`), and a small update δW from fine-tuning, is there a simple map that returns a
decomposition of W + δW which is still faithful, with the residual Δ not absorbing the update, so that the
decomposition stays valid continuously along training? If not, is that because post-training teaches new capabilities
rather than strengthening existing ones, or because parameter decompositions are too brittle to support the question?

The current testbed is Goodfire's released 4-layer 67M Pile model and its released VPD decomposition, chosen because
the decomposition exists and every run takes minutes. A larger model with a real post-trained sibling is the next phase.

**Research log (read this first):** https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu. The one document for bringing
someone up to speed: the question and sub-questions Q1–Q9, the apparatus, the techniques, every experiment E1–E13, every
claim with its status, the timeline, gaps, backlog, mistakes and glossary. It is the explanation; the files below are the
record. Source: `docs/research_log/index.html`, publish recipe and update checklist in `docs/research_log/README.md`.
Updating it is part of the research flow: every analysis report, decision and new `TRIED.md` row ends with the matching
edit to the log and a republish; a Stop hook blocks a session's end once while the record is newer than the log.

The aim: `GOALS.md`. Working choices: `decisions.md`. Everything tried and why we moved on: `experiments/TRIED.md`.
Success criterion (2026-10-19): GOALS's weak version as written; the route split is a mechanism reading
(`reports/2026-10-19_decision_success-criterion.md`).

## State of play (2026-10-20)

Vocabulary: `GOALS.md` → Terms (component training, least-squares split, adjoint split, outside share, LoRA split).
Older reports say "pull-back" for component training and "lift" for the least-squares split; the user's "pull-back"
meant the split of a finished update, and that request was never built as asked
(`reports/2026-10-20_review_terminology.md`).

- Component training (training the subcomponents' vectors on the loss, Γ and Δ frozen) learns every task tried,
  including one the model lacks, but its decomposition passes VPD's adversarial check on 3 of 7 next-word tasks, sits at
  the line on 1 and fails on 3 (E11), and fails for 32 new associations where 1 passes (E10). Leaving the update in Δ is
  more robust under the adversary on 4 of 5 tasks checked. Rank-8 component training on all 24 matrices stays faithful
  (E13).
- The least-squares split of a finished full-weight fine-tune is adversarially fragile (ratio 5.4, reproduced; E2, E8).
  The split of a LoRA stays under the pass line to 1000 steps but its read side never leaves its random initialisation,
  so by the plateau over half of the update's energy and a third of the behaviour sit in Δ (E8, E9). The like-for-like
  comparison with a fine-tune on one card at one step has not been run.
- Whatever the operation, the changed subcomponents carry at most 8% of a new behaviour; the rest runs through the
  frozen Γ re-switching existing subcomponents for behaviours the model has, and through subcomponents Γ keeps off for
  the cipher it lacks (E4–E6, E10).
- Outside share of a LoRA (E9): 0.48 to 0.71 of its energy against 0.55 to 0.76 for a random update, at every step; a
  fine-tune reads 0.27 against 0.53 (E8). The LoRA sits outside the subcomponents about as much as noise does.

Current plan: the **LoRA split** (the user's request since 2026-10-14; `reports/2026-10-20_review_terminology.md` §7),
through `/design`: no training, the saved LoRA factors of E7, E8 and E9, both splits, the per-subcomponent updates
saved, the outside share as four blocks with its null. Also open, all through `/design`: the matched-off-target-KL
component-training comparison, the key → value matched-size check on the saved states, the adversarial-vs-stochastic
cancellation tests, E12's follow-ups (`reports/2026-10-20_analysis_new-vs-existing-signatures.md`).

## Experiment codes

| Code | Experiment | Directory | Analysis |
|---|---|---|---|
| E1 | Transport test (round 1) | `experiments/transport_test` | `reports/2026-10-14_analysis_transport-test.md` |
| E2 | Round 2: splits and component training on emoticon | `experiments/lift_test` | `reports/2026-10-14_analysis_lift-test.md` |
| E3 | Identity test | `experiments/identity_test` | `reports/2026-10-15_analysis_identity-test.md` |
| E4 | Concentration test, two amendments | `experiments/concentration_test` | `reports/2026-10-15_analysis_concentration-test.md` |
| E5 | Task ladder (round 3) | `experiments/task_ladder` | `reports/2026-10-19_analysis_task-ladder.md` |
| E6 | Recruit test | `experiments/recruit_test` | `reports/2026-10-19_analysis_recruit-test.md` |
| E7 | LoRA on one matrix, split and rank-r component training (round 4) | `experiments/lora_commute` | `reports/2026-10-19_analysis_lora-commute.md` |
| E8 | Matched-matrix LoRA (round 5) | `experiments/lora_matched` | `reports/2026-10-19_analysis_lora-matched.md` |
| E9 | LoRA to its plateau | `experiments/lora_plateau` | `reports/2026-10-19_analysis_lora-plateau.md` |
| E10 | Key → value cipher (round 6) | `experiments/kv_task` | `reports/2026-10-19_analysis_key-value-task.md` |
| E11 | Reading-A coverage, emoticon rerun | `experiments/reading_a_coverage` | `reports/2026-10-19_analysis_reading-a-coverage.md` |
| E12 | New-vs-existing signatures (round 7) | `experiments/new_vs_existing` | `reports/2026-10-20_analysis_new-vs-existing-signatures.md` |
| E13 | Rank-8 component training on all 24 matrices (run as "LoRA pull-back"; not the user's LoRA question) | files deleted 2026-10-20 at the user's request | finding kept in `experiments/TRIED.md` row E13 and the research log |

The 2026-10-20 review (`reports/2026-10-20_review_2026-10-19-write-ups.md`) lists claims in the E8, E9 and E11 reports
that their result files do not support; the notebooks and the log use the corrected values, the reports stand as written.
The terminology review (`reports/2026-10-20_review_terminology.md`) lists the sentences whose meaning depends on the
old words; dated reports keep their wording and are read through the Terms' old-names column.

## Layout

```
experiments/        one directory per experiment: run.py, plot.py, predictions.json, results/, figures/
  TRIED.md          everything tried or dropped, and why we moved on (rows carry the E-codes)
theory/             notes and derivations
notebooks/          where to look at results
reports/            dated write-ups: YYYY-MM-DD_<prediction|analysis|plan|review|decision>_<topic>.md
backlog.md          deferred ideas and loose ends
.claude/CLAUDE.md   rules and commands for coding agents
```

## Conventions

Inherited from `../mono/feature_wars`: predictions filed before running and scored afterwards; one question per
experiment; `/design` check against `GOALS.md`, `TRIED.md` and `decisions.md` before any new direction; compute
through the MATS cluster first; budgets under $10 unless confirmed.
