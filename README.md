# commute

Can a small gradient update from post-training be *commuted past* a parameter decomposition? Given a VPD
decomposition of a model's weights into rank-1 subcomponents (Bushnaq, Braun, Clive-Griffin et al. 2026,
`../papers/pd/vpd_blog_post.md`), and a small update δW from fine-tuning, can δW be written as a small change to a
few existing subcomponents, with the causal-importance function left alone, so that the decomposition stays valid
and interpretable continuously along training? If it can, that is "continuous interpretability" within some limit.
If it cannot, is that because post-training teaches new capabilities (rather than strengthening existing ones), or
because parameter decompositions are too brittle to support the question?

The aim: `GOALS.md`. Working choices: `decisions.md`. Everything tried and why we moved on: `experiments/TRIED.md`.
Success criterion (2026-10-19): GOALS's weak version as written; the route split is a mechanism reading (`reports/2026-10-19_decision_success-criterion.md`, with what has actually been checked). Current plan: `reports/2026-10-19_plan_reading-a-coverage.md` (reading-A coverage round, running); in parallel the key → value task's follow-up (`reports/2026-10-19_analysis_key-value-task.md`, "What this changes": the matched-size Pile check on the saved states, then K between 1 and 32). Round 6, the key → value task, is done (`reports/2026-10-19_analysis_key-value-task.md`): the pull-back learns a behaviour the model lacks and stays faithful on Pile for one new pair and for the identity control (PGD 1.20, 1.19) but not for 32 pairs (2.81, filed pass reversed; cause not separated); the identity rides on Γ re-switching existing subcomponents (r 0.8), the cipher on subcomponents Γ does not switch on (e_gate 0.75–0.86); VPD's sparsity loss cannot tell maps apart while Γ is frozen. Round 5, the matched-matrix LoRA, is done (`reports/2026-10-19_analysis_lora-matched.md`): round 4's confound was the site; a rank-8 LoRA on all 24 matrices lifts with e_Δ 0.22 and its lifted state sits at the W₀ floor under the Pile adversary (1.13) where the fine-tune's lift reads 5.41 (reproduced on the L40, closing round 4's G4 miss), but at step 2, so the step-matched comparison is the next measurement. Round 4, the LoRA commute test, is done (`reports/2026-10-19_analysis_lora-commute.md`, `notebooks/lora_commute.ipynb`): a LoRA on the paper's matrix is no easier to commute than the full-weight fine-tune, and by the weak version harder (its inexpressible share sits at the isotropic null because its read side stays near its random init; its lift leaves 0.33–0.43 of the behaviour in Δ; every map applied to it sits above the W₀ PGD floor, and the rank-8 lift and the one-matrix pull-backs break the 1.5 line). Round 3, the task ladder, failed its hypothesis (`reports/2026-10-19_analysis_task-ladder.md`): coverage does not predict how much of a new behaviour the CI-masked model keeps. Round 2 (`reports/2026-10-14_analysis_lift-test.md`): the pull-back on all matrices is faithful, the lift of a finished fine-tune is adversarially fragile. Round 1 (`reports/2026-10-14_analysis_transport-test.md`)

## Layout

```
experiments/        one directory per experiment: run.py, plot.py, predictions.json, results/, figures/
  TRIED.md          everything tried or dropped, and why we moved on
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
