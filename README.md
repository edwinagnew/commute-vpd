# Update Decomposition (repo `commute`)

Can a small gradient update from post-training be *commuted past* a parameter decomposition? Given a VPD
decomposition of a model's weights into rank-1 subcomponents (Bushnaq, Braun, Clive-Griffin et al. 2026,
https://static.goodfire.ai/vpd-blog-post/post.md), and a small update δW from fine-tuning, is there a simple map that returns a
decomposition of W + δW which is still faithful, with the residual Δ not absorbing the update, so that the
decomposition stays valid continuously along training? If not, is that because post-training teaches new capabilities
rather than strengthening existing ones, or because parameter decompositions are too brittle to support the question?

The current testbed is Goodfire's released 4-layer 67M Pile model and its released VPD decomposition, chosen because
the decomposition exists and every run takes minutes. A larger model with a real post-trained sibling is the next phase.

**Research log (read this first):** https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu. The one document for bringing
someone up to speed: the question and sub-questions Q1–Q11, the apparatus, the techniques, every experiment E1–E20, every
claim with its status, the timeline, gaps, backlog, mistakes and glossary. It is the explanation; the files below are the
record. Four pages (main, experiments, glossary, changelog) built from `docs/research_log/parts/`; layout, publish recipe and update checklist in `docs/research_log/README.md`.
Updating it is part of the research flow: every analysis report, decision and new `TRIED.md` row queues a line in
`docs/research_log/PENDING.md`, flushed into the log and republished before the session ends.

The aim: `GOALS.md`. Working choices: `decisions.md`. Everything tried and why we moved on: `experiments/TRIED.md`.
Success criterion (2026-10-19): GOALS's weak version as written; the route split is a mechanism reading
(`reports/decisions/2026-10-19_decision_success-criterion.md`).

## State of play (2026-10-20)

Vocabulary: `GOALS.md` → Terms (component training, least-squares split, adjoint split, outside share, LoRA split).
Older reports say "pull-back" for component training and "lift" for the least-squares split; the user's "pull-back"
meant the split of a finished update, and that request was never built as asked
(`reports/reviews/2026-10-20_review_terminology.md`).

- Component training with Adam (training the subcomponents' vectors on the loss, Γ and Δ frozen; lr 1e-5, chosen on
  emoticon and never re-tuned) learns every task tried, including one the model lacks, but its decomposition passes VPD's adversarial check on 3 of 7 next-word tasks, sits at
  the line on 1 and fails on 3 (E11), and fails for 32 new associations where 1 passes (E10). Leaving the update in Δ is
  more robust under the adversary on 4 of 5 tasks checked. Rank-8 component training with Adam on all 24 matrices stays
  faithful (E13). Plain SGD also learns emoticon within the off-target limit (E4 follow-up), measured at a smaller size
  of change than Adam's; a matched comparison is running (item 2 of
  `reports/reviews/2026-10-21_review_sgd-finding-influence.md`). Its faithfulness has not been measured, so whether the failures
  are the operation's or Adam's is open. Component training is optimiser-agnostic (the run's optimiser, either SGD or
  Adam), and comparing the two on the same tasks is a priority subgoal (`GOALS.md`, user 2026-10-21).
- The least-squares split of a finished full-weight fine-tune is adversarially fragile (ratio 5.4, reproduced; E2, E8).
  The split of a LoRA stays under the pass line to 1000 steps but its read side never leaves its random initialisation,
  so by the plateau over half of the update's energy and a third of the behaviour sit in Δ (E8, E9). The like-for-like
  comparison with a fine-tune on one card at one step has not been run.
- Under every operation route-split so far (Adam component training and splits of finished updates; no SGD state yet),
  the changed subcomponents carry at most 8% of a new behaviour; the rest runs through the
  frozen Γ re-switching existing subcomponents for behaviours the model has, and through subcomponents Γ keeps off for
  the cipher it lacks (E4–E6, E10).
- Outside share of a LoRA (E9): 0.48 to 0.71 of its energy against 0.55 to 0.76 for a random update, at every step; a
  fine-tune reads 0.27 against 0.53 (E8). The LoRA sits outside the subcomponents about as much as noise does.

Current plan, Qwen D3, E84 (2026-11-04, `reports/plans/2026-11-04_plan_e84-rephrasings.md`, $0): anchor 2 again, TOFU's 800 rows each trained in five wordings, Qwen3-1.7B writing the other four, tested on TOFU's own rephrased question, never trained, with fp32 δW on anchor 1's grid; ran with Amendment 1; the filed test missed twice, rephrased recall 0.532 and 0.554 under a line of 0.602 (`reports/analyses/2026-11-04_analysis_e84-rephrasings.md`); anchor 2 awaits the user, card decision-anchor2-after-e84.

Current plan, Qwen D2, anchor 1 (2026-11-03, `reports/plans/2026-11-03_plan_e78-anchor1.md`, E78, $0): E61's template fine-tune retrained unchanged, saving δW in fp32 at E69's steps plus step 3, so the split can read it; coded and filed, not run yet.

Current plan, Qwen D1, latest (2026-11-03, `reports/plans/2026-11-03_plan_e72-d1-second-test.md`, E72, $0, measures only): D1 blocked on day 1, the pass test under the original masks cannot yet pass anything on KL (`reports/analyses/2026-11-03_analysis_e68-d1-pass-test.md`); what the test becomes is the user's call, log card `decision-d1-test`.

Current plan, whole fleet, latest (2026-11-05, `reports/plans/2026-11-05_plan_qwen-reorient-amend1.md`, user): preliminary results by 2026-11-18, core done 11-14, midpoint check 11-11 (D3 rows read on both anchors). D1 closes on option A of card decision-after-e76 after E89, least squares scored first, no more adversary-teeth runs; the D3 rows (route split, δW against the random null, behaviour kept under masks, think-mode overlap) run 11-07 to 11-11 on both anchors at the full update and at a size-matched step, then on RL, null and own-samples SFT; `baseline-lead` builds the comparators 11-06 and E90 stays on paper until they exist. Digests name the row moved; no new E-code outside the table before 11-14.

Current plan, whole fleet, earlier (2026-11-03, `reports/plans/2026-11-03_plan_qwen-reorient.md`, user): the Qwen deliverables of 2026-10-28 by 2026-11-11. D1 (`qwen-decomp-lead`): a pass test that can fail a bad placement on the saved updates, else teacher training as the map; D2 (`qwen-rl-lead`, new): the GRPO loop, null and own-samples SFT, δW saved; D3: the route-split and placement rows on every arm with a cheap comparator beside each (`baseline-lead`, secondary). The 67M lines below are stopped; the fast lane (reviews only for gated runs, two amendments per E-code, gate at $60 / 6 h, session budget $100) applies.

Current plan, benchmark line, latest (2026-11-03, `reports/plans/2026-11-03_plan_e70-existing-new-counting.md`, Amendments 1 and 2, E70, $0, asks 15 and 17 taken by default): E70 ended at its two amendments (`reports/analyses/2026-11-03_analysis_e70-existing-new-counting.md`). The line is secondary; next the cheap comparators beside Qwen readings, 11-04 to 11-05 (`reports/plans/2026-11-03_plan_qwen-reorient.md`).

Current plan, 67M neuron baseline (2026-11-05, `reports/plans/2026-11-05_plan_e101-carrying-jobs.md`, E101, $0): the essay's neuron reading on E98's carrying neurons against block-matched random ones; stage 0 (W0 windows, already-engaged share) first, then 36 blind agent calls.

Current plan, Qwen neuron baseline, latest (2026-11-06, `reports/plans/2026-11-06_plan_e107-label-vs-style.md`, E107, $0): the known arm's exact rows and wordings with a wrong answer substituted (only W0's familiarity changes), plus a topic split of the known rows, read at the matched step; coded, not run. Before it E105 (`reports/plans/2026-11-06_plan_e105-rows-null.md`), the same-label, different-rows null, which read "label" on both probes. Before that E103 (`reports/plans/2026-11-06_plan_e103-known-answer-pair.md`), the known-answer pair, ran with two amendments.


## Experiment codes

The table stops short: `experiments/TRIED.md` is the full list (friction 393). Older "Current plan" lines are in
`reports/plans/2026-11-05_plan_older-current-plan-lines.md`; keep one line per active lead above.

| Code | Experiment | Directory | Analysis |
|---|---|---|---|
| E1 | Transport test (round 1) | `experiments/transport_test` | `reports/analyses/2026-10-14_analysis_transport-test.md` |
| E2 | Round 2: splits and component training on emoticon | `experiments/lift_test` | `reports/analyses/2026-10-14_analysis_lift-test.md` |
| E3 | Identity test | `experiments/identity_test` | `reports/analyses/2026-10-15_analysis_identity-test.md` |
| E4 | Concentration test, two amendments | `experiments/concentration_test` | `reports/analyses/2026-10-15_analysis_concentration-test.md` |
| E4 follow-up | Plain-SGD component training at small rates | `experiments/sgd_grid` | `reports/analyses/2026-10-21_analysis_sgd-grid.md` |
| E5 | Task ladder (round 3) | `experiments/task_ladder` | `reports/analyses/2026-10-19_analysis_task-ladder.md` |
| E6 | Recruit test | `experiments/recruit_test` | `reports/analyses/2026-10-19_analysis_recruit-test.md` |
| E7 | LoRA on one matrix, split and rank-r component training (round 4) | `experiments/lora_commute` | `reports/analyses/2026-10-19_analysis_lora-commute.md` |
| E8 | Matched-matrix LoRA (round 5) | `experiments/lora_matched` | `reports/analyses/2026-10-19_analysis_lora-matched.md` |
| E9 | LoRA to its plateau | `experiments/lora_plateau` | `reports/analyses/2026-10-19_analysis_lora-plateau.md` |
| E10 | Key → value cipher (round 6) | `experiments/kv_task` | `reports/analyses/2026-10-19_analysis_key-value-task.md` |
| E11 | Reading-A coverage, emoticon rerun | `experiments/reading_a_coverage` | `reports/analyses/2026-10-19_analysis_reading-a-coverage.md` |
| E12 | New-vs-existing signatures (round 7) | `experiments/new_vs_existing` | `reports/analyses/2026-10-20_analysis_new-vs-existing-signatures.md` |
| E13 | Rank-8 component training on all 24 matrices (run as "LoRA pull-back"; not the user's LoRA question) | files deleted 2026-10-20 at the user's request | finding kept in `experiments/TRIED.md` row E13 and the research log |
| E14 | LoRA split study: LoRA and fine-tune under one table | `experiments/lora_split` | `reports/analyses/2026-10-20_analysis_lora-split.md` |
| E15 | Γ refit: let Γ see the updated model (two refits of six, a second W₀ protocol, stage 0; ongoing: being fixed, 2026-10-21) | `experiments/gamma_refit` | `reports/analyses/2026-10-20_analysis_gamma-refit.md` |
| E16 | The five operations side by side (gate failed; predictions unscored) | `experiments/split_comparison` | `reports/analyses/2026-10-21_analysis_split-comparison.md` |
| E17 | Cancellation tests (control failed; predictions unscored) | `experiments/cancellation` | `reports/analyses/2026-10-21_analysis_cancellation.md` |
| E18 | Mechanism or leak (filed shared pattern failed on 2 of 3 tasks; no leak seen) | `experiments/regating_leak` | `reports/analyses/2026-10-21_analysis_regating-leak.md` |
| E19 | Step split along the fine-tune's path | `experiments/step_split` | `reports/analyses/2026-10-21_analysis_step-split.md` |
| E20 | Complement revert: excess not in the original subcomponents | `experiments/cancellation_complement` | `reports/analyses/2026-10-21_analysis_cancellation-complement.md` |
| E21 | SGD vs Adam: the SGD-review runs (Q11; analysis pending) | `experiments/sgd_{faith,matched,finetune,ladder,ladder_long,kv}` | not yet written; hand-off `reports/plans/2026-10-21_plan_sgd-analysis-handoff.md` |
| E24 | Fixed switch pattern on small updates: as for the large one | `experiments/regating_small` | `reports/analyses/2026-10-22_analysis_regating-small.md` |
| E28 | The fixed-shift hypothesis's kill tests on seven new tasks: ladder all survive, key-value all killed (ident32 lean reversed) | `experiments/regating_tasks` | `reports/analyses/2026-10-22_analysis_regating-tasks.md` |
| E29 | What the turned-down half does: not the competitors' machinery (pool-level overlap); lowering it takes competitors' mass on some tasks, adds on one | `experiments/regating_down` | `reports/analyses/2026-10-22_analysis_regating-down.md` |
| E22 | Small fine-tunes' splits at steps 10/100/1000: pass at every step; SGD's adjoint keeps behaviour | `experiments/small_finetune_splits` | `reports/analyses/2026-10-22_analysis_small-finetune-splits.md` |
| E25 | Step split along the small fine-tunes' paths: passes at every step; still less behaviour in the subcomponents than least squares, except SGD at step 1000 | `experiments/step_split_small` | `reports/analyses/2026-10-22_analysis_step-split-small.md` |
| E27 | LoRA trained with plain SGD: between noise and a fine-tune, every placement passes | `experiments/lora_sgd` | `reports/analyses/2026-10-22_analysis_lora-sgd.md` |
| E23 | The excess on the failures that survive SGD: same shape as Adam's (joint alive+dead removal); step-1000 SGD-vs-Adam holds to 160 adversary steps | `experiments/cancellation_sgd` | `reports/analyses/2026-10-22_analysis_cancellation-sgd.md` |
| E26 | Second batch order for ladder component training: verdicts replicate (Adam passes at 100 and 1000, SGD fails at 1000 on r0/r1) | `experiments/ladder_second_seed` | `reports/analyses/2026-10-22_analysis_ladder-second-seed.md` |
| E70 | Existing/new definition read in full on the counting pair: removing the one ridge direction does nothing even on LOW, so no reading; removing six softmax directions costs the reusing task more at blocks 2-5 (0.72 vs 0.34 at block 5); Amendment 1 filed "cannot serve" by a block-6 reversal; Amendment 2 filed "count-specific at this size", not size-matched: at blocks 3-5 the count direction costs the reusing task .43-.88 and the other .16-.49, the largest no-count direction nothing | `experiments/pcfg_anchors` | `reports/analyses/2026-11-03_analysis_e70-existing-new-counting.md` |
| E37 | Teacher training of a fine-tune with no trigger (lowercase text): the copy keeps 0.998 of the gain but fails the adversarial check (2.20, fail line 1.61) while still learning; the scaled adjoint split keeps 0.80 and passes (filed under half) | `experiments/teacher_spread` | `reports/analyses/2026-10-26_analysis_teacher-spread.md` |
| E36 | Re-gating with no trigger: on a lowercase-text fine-tune switching carries 0.79-1.05 of the gain, and the 8 positions with the nearest original values recover 0.51-0.72 of it, twice the current token; the code-text fine-tune did not learn | `experiments/regating_spread` | `reports/analyses/2026-10-26_analysis_regating-spread.md` |
| E35 | Re-gating without cues: the mean change of the 8 inputs with the nearest original importance values recovers the per-cue level (0.61-0.78) with no cue named; k-means groups and a line per subcomponent fall short with many cues | `experiments/regating_cuefree` | `reports/analyses/2026-10-26_analysis_regating-without-cues.md` |
| E34 | Teacher training: copies of six fine-tunes keep the behaviour and pass on three tasks, from task windows or the Pile alone; emoticon learnt from the Pile (filed prediction reversed), and Amendment 1 shows it is still learnt with every trigger row removed, by a weight change other than δW's; plain SGD fails late on r0/r1 | `experiments/teacher_training` | `reports/analyses/2026-10-23_analysis_teacher-training.md` |
| E33 | Many cues, one answer: the fixed change follows the cue (per cue 0.90/1.06); a shared answer gives a shared half (0.55 vs 0.04) | `experiments/many_cues` | `reports/analyses/2026-10-22_analysis_shift-anatomy-and-many-cues.md` |
| E32 | What the fixed change is made of: one per key works on copy and the 32-pair cipher (0.84, 0.82); MLP carries it; strength, Γ-input and Pile-signature forms fail | `experiments/shift_anatomy` | `reports/analyses/2026-10-22_analysis_shift-anatomy-and-many-cues.md` |
| E31 | Adjoint split of SGD fine-tunes: keeps emoticon behaviour on two seeds, not on r1/r0 (filed prediction reversed); every state faithful | `experiments/adjoint_sgd` | `reports/analyses/2026-10-22_analysis_adjoint-sgd.md` |
| E30 | One fixed change to Γ's values at every position as the map: keeps the trigger behaviour, breaks Pile faithfulness (adversarial 5–26× the no-change reading, sparsity 1.8–3.3×) | `experiments/gamma_shift_map` | `reports/analyses/2026-10-22_analysis_gamma-shift-map.md` |

The 2026-10-20 review (`reports/reviews/2026-10-20_review_2026-10-19-write-ups.md`) lists claims in the E8, E9 and E11 reports
that their result files do not support; the notebooks and the log use the corrected values, the reports stand as written.
The terminology review (`reports/reviews/2026-10-20_review_terminology.md`) lists the sentences whose meaning depends on the
old words; dated reports keep their wording and are read through the Terms' old-names column.

## Layout

```
experiments/        one directory per experiment: run.py, plot.py, predictions.json, results/, figures/
  TRIED.md          everything tried or dropped, and why we moved on (rows carry the E-codes)
theory/             notes and derivations
notebooks/          where to look at results
reports/<type>/     dated write-ups, one subdirectory per type: plans analyses reviews decisions predictions digests
backlog.md          deferred ideas and loose ends
.claude/CLAUDE.md   rules and commands for coding agents
```

## Conventions

Inherited from `../mono/feature_wars`: predictions filed before running and scored afterwards; one question per
experiment; `/design` check against `GOALS.md`, `TRIED.md` and `decisions.md` before any new direction; compute
through the MATS cluster first; budgets under $10 unless confirmed.
