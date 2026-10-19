# commute

Can a small gradient update from post-training be *commuted past* a parameter decomposition? Given a VPD
decomposition of a model's weights into rank-1 subcomponents (Bushnaq, Braun, Clive-Griffin et al. 2026,
`../papers/pd/vpd_blog_post.md`), and a small update δW from fine-tuning, can δW be written as a small change to a
few existing subcomponents, with the causal-importance function left alone, so that the decomposition stays valid
and interpretable continuously along training? If it can, that is "continuous interpretability" within some limit.
If it cannot, is that because post-training teaches new capabilities (rather than strengthening existing ones), or
because parameter decompositions are too brittle to support the question?

The aim: `GOALS.md`. Working choices: `decisions.md`. Everything tried and why we moved on: `experiments/TRIED.md`.
Current plan: `reports/2026-10-19_plan_lora-lift-and-pullback.md` (a LoRA through the decomposition both ways: lift the finished LoRA onto its matrix's subcomponents, and a pull-back with the LoRA's support and rank; ranks 1 to 32; confirmed 2026-10-19, to be run by a fresh session). Round 3, the task ladder, is done and its hypothesis failed (`reports/2026-10-19_analysis_task-ladder.md`): coverage does not predict how much of a new behaviour the CI-masked model keeps (P1 null), and rarer targets keep less (P2 null). Changed subcomponents carry ≤ 8% of the gain; most of what is kept comes from the released subcomponents masked with the CI values Γ computes on the trained model. Open first: whether that is a mechanism or a leak through Γ's input. Done 2026-10-19: the PGD check (`reports/2026-10-19_review_pgd-local.md`): the ladder's retained_pgd has no power and stays reported-only; the window adversaries saturate because the windows are alike, so a shared mask is a per-position attack. Done 2026-10-19: the recruit test (`reports/2026-10-19_analysis_recruit-test.md`): the real W fine-tune and a LoRA, placed do-nothing, recruit the pull-back's set through frozen Γ (Jaccard 0.64 / 0.82, seed floor 0.93); a different-target fine-tune recruits a disjoint set (0.04); re-gating alone carries 0.6 of the gain for every target. Γ turned out to read every matrix's input at every position (global shared transformer). Done 2026-10-15: concentration test (`reports/2026-10-15_analysis_concentration-test.md`: sparse pull-backs learn the task, but at the trigger the behaviour runs through Γ re-gating on the new activations, not through the moved components); identity test (`reports/2026-10-15_analysis_identity-test.md`: the pull-back spreads its update over thousands of components). The new-skill plan (`reports/2026-10-14_plan_new-skill-test.md`) is shelved; its question is the ladder's rung 4. Measurement review: `reports/2026-10-15_review_faithfulness-scope.md`. Round 2: `reports/2026-10-14_plan_lift-test.md`, results `reports/2026-10-14_analysis_lift-test.md`, notebook `notebooks/lift_test_round2.ipynb`. Round 1's plan: `reports/2026-10-14_plan_transport-test.md`. First result: `reports/2026-10-14_analysis_transport-test.md`
(2026-10-14: the literal projection fails, functional mimicry is non-specific, the decomposition stays faithful).

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
