# commute

Can a small gradient update from post-training be *commuted past* a parameter decomposition? Given a VPD
decomposition of a model's weights into rank-1 subcomponents (Bushnaq, Braun, Clive-Griffin et al. 2026,
`../papers/pd/vpd_blog_post.md`), and a small update δW from fine-tuning, can δW be written as a small change to a
few existing subcomponents, with the causal-importance function left alone, so that the decomposition stays valid
and interpretable continuously along training? If it can, that is "continuous interpretability" within some limit.
If it cannot, is that because post-training teaches new capabilities (rather than strengthening existing ones), or
because parameter decompositions are too brittle to support the question?

The aim: `GOALS.md`. Working choices: `decisions.md`. Everything tried and why we moved on: `experiments/TRIED.md`.
Current plan: `reports/2026-10-14_plan_transport-test.md`. First result: `reports/2026-10-14_analysis_transport-test.md`
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
