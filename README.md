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

Current plan, re-gating line (2026-10-26, `reports/plans/2026-10-26_plan_regating-without-cues.md`): E35 and E36 tested a form of the fixed-shift hypothesis that names no cue (each input gets the change of the inputs with the most similar original importance values); it matches one change per cue on the toy tasks and recovers about half to seven-tenths of the switching on a lowercase-text fine-tune with no trigger, twice the current token. The restatement waits on the user (research log, pending decisions). Earlier (2026-10-22, `reports/decisions/2026-10-22_decision_working-hypothesis-and-two-lines.md`): the fixed-shift hypothesis's kill tests ran as E28 (`reports/analyses/2026-10-22_analysis_regating-tasks.md`): every task the model already does survives, the three key-value tasks fail the fixed-change condition; whether the fixed-shift hypothesis is restated with that scope is the user's decision. E29 (`reports/analyses/2026-10-22_analysis_regating-down.md`): the turned-down subcomponents are not the competitors' machinery by overlap, and their function is mixed. Next named step: faithfulness under Γ's switching, then the larger model (Q9). E32 and E33 (`reports/analyses/2026-10-22_analysis_shift-anatomy-and-many-cues.md`): one fixed change per cue word carries the switching on every task tested, the copy task and the 32-pair cipher included (0.84, 0.82); with 32 cues and one answer, a shared change gives back about half (0.55). The per-cue change sits in the MLP matrices and is not one direction, one move of Γ's input, or the word's Pile signature. Decided 2026-10-23, tentatively (`reports/decisions/2026-10-23_decision_fixed-shift-per-cue.md`): "one fixed change per cue"; "cue" has no definition beyond the trigger token yet, so the anatomy follow-ups are deferred and the next re-gating step is a fine-tune with no single trigger (`backlog.md`).
Current plan, Qwen line (2026-10-27, `reports/plans/2026-10-27_plan_qwen-port-and-baseline.md`): convert the Qwen3-1.7B decomposition to PyTorch and check the port against the authors' JAX code; GSM8K baseline of the unchanged model; then the first experiment (a fine-tune on GSM8K's human-written solutions), to be discussed with the user.

Current plan, commute line (2026-10-22, `reports/plans/2026-10-22_plan_commute-second-line.md`): the decision report's second line in its own session: E26 a second training seed for SGD and Adam component training at steps 100 and 1000 on r0, r1, r2; E27 a rank-8 LoRA trained with plain SGD through E14's table; item B (why SGD fails at step 1000) waits for E23; item D ran as E30 (`reports/analyses/2026-10-22_analysis_gamma-shift-map.md`): the fixed change applied everywhere keeps the trigger behaviour but is not faithful on the Pile, so the change must depend on the input; its Amendment 1 (2026-10-23) applied each half alone: the lowered half breaks the Pile and the raised half costs three times the sparsity, so no fixed change applied everywhere is a map; next on this item, the change applied only where the trigger is (reading ii). E34 (`reports/analyses/2026-10-23_analysis_teacher-training.md`): teacher training, component training with the fine-tuned model as teacher and Γ, Δ frozen, keeps the behaviour and passes the adversarial check for all six fine-tunes on three tasks, trained on task windows or on Pile text alone; it is the commute line's first map of a finished update that needs no labels. Its Amendment 1 (same report): with every Pile row holding the emoticon trigger replaced, the copies still reach P(target) 0.94 and 0.89, so the trigger's occurrences are not what teaches them; their weight change aligns with δW less than the adjoint split's (cosine 0.08-0.22 vs 0.38-0.45), so they find a different weight change with the same behaviour. Amendments 2-4 (2026-10-26, same report and plan): dropping the Pile rows where the trigger is partly on (g > 0.1, 104 more rows) lowers the Pile-only copies to 0.87 and 0.84, a fall of 0.05-0.07 that is real for the SGD teacher against the seed floor and marginal for the Adam teacher; a rank-8 LoRA teacher is copied from task text on all three tasks (P(target) 0.93-1.00) but only part of the way from Pile text (0.52-0.88, still faithful); a second batch order repeats every E34 copy (P(target) within 0.005 on task text, within 0.03 on Pile text), and under an adversary eight times longer every copy stays at the floor (ratios 0.95-1.15 against a 1.24 line) while the teachers themselves read 2.4-10.4. Next: the larger model.
Current plan, SGD-vs-Adam line (2026-10-21, `reports/plans/2026-10-21_plan_sgd-analysis-handoff.md`): seven runs of the SGD-finding review are in and unanalysed; at steps 100 and 1000 Adam component training passes the adversarial check on the ladder tasks it failed at step 10, which may meet the re-gating decision's move-back bullet (the user's call).
Current plan (2026-10-21, Amendment 2 sections of `reports/plans/2026-10-21_plan_cancellation-complement.md` and `reports/plans/2026-10-21_plan_regating-leak.md`): E20 Amendment 1 put the adversary's excess in the interaction of its alive and dead removals and found its 20-step search beaten at W₀; next, W₀'s levels at the updated models and a stronger adversary. E18 Amendment 1 argues the mechanism reading; the user keeps it a leading hypothesis only (`reports/decisions/2026-10-21_decision_mechanism-reading.md`); next, a random null for the whole shared change.
Previous: Current plan (2026-10-21, `reports/decisions/2026-10-21_decision_regating-main-line.md`): E16 (`experiments/split_comparison`) is
analysed (`reports/analyses/2026-10-21_analysis_split-comparison.md`); finish the E15 v2 control; then **re-gating** becomes the main line, starting
with whether it is a mechanism or a leak through Γ's input. The commute question stays open, second. Earlier plan: the **Γ refit** (`reports/plans/2026-10-20_plan_gamma-refit.md`; code `experiments/gamma_refit`,
`src/commute/gamma.py`; analysis `reports/analyses/2026-10-20_analysis_gamma-refit.md`, E15). Two of six refits ran on 2026-10-20
(the benchmark read 2.3 s/step, six times the plan's arithmetic, so one refit fit the cap): the least-squares split's
adversarial ratio went 7.0 → 24 → 23 → 7.6 → 1.06 over 3 000 Pile-only refit steps while the sparsity loss rose to 1.60×
W₀'s (the filed "cheating" branch) and the refit's own loss rose throughout. Stage 0 (no training) found the
subcomponents the adversary turns down are singly inert at both W₀ and the split state (median ablation ratio 0.98), so
the fragility is a joint effect, the cancellation reading. The W₀ control under the same refit then failed its filed
prediction (adversarial ratio 2.6 → 3.5 → 6.3 → 2.6 with every other check within 8%): the recipe drifts on its own, so
the split's fall is not a repair. A second protocol (a 300-step adversary warm-up, then the released end lr 5e-6;
`reports/plans/2026-10-21_plan_gamma-refit-protocol.md`) failed the same control by step 100 (2.1, then 7.8 at step 300) while
the refit's own attacker stayed flat at 0.42 nats against the harness's 0.70, so neither recipe can be read on an
updated state. E15 is ongoing (user, 2026-10-21, `reports/decisions/2026-10-21_decision_gamma-refit-reopened.md`): a retraining with an attacker matched to the
harness's, its W₀ control first. Next: the cancellation tests on the saved split state and the re-gating line. The **LoRA split** (E14, `reports/analyses/2026-10-20_analysis_lora-split.md`) ran 2026-10-20: the fine-tune is inside only on
the read side, the LoRA on neither, and the scaled adjoint split state keeps 93% of the step-10 fine-tune's behaviour at the
faithfulness floor; its per-subcomponent pieces were lost with the pod and the split stage is to be rerun on the cluster. Also open, all through `/design`: the matched-off-target-KL
component-training comparison, the key → value matched-size check on the saved states, the adversarial-vs-stochastic
cancellation tests, E12's follow-ups (`reports/analyses/2026-10-20_analysis_new-vs-existing-signatures.md`).

## Experiment codes

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
