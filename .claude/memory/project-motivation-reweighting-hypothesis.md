---
name: project-motivation-reweighting-hypothesis
description: "The user's own motivation for the project (their MATS proposal, theory/TOC2 -EA.pdf): test the re-weighting hypothesis in weight space, and continuous interpretability as a monitor for post-training"
metadata:
  node_type: memory
  type: project
  originSessionId: 6f8f2e79-c23b-4393-b1f5-8d2cf4341c62
  modified: 2026-10-20T11:45:58.696Z
---

The user's proposal (`theory/TOC2 -EA.pdf`, 2026-10-20 copy) frames the project as a weight-space test of the
re-weighting hypothesis: that post-training re-organises pretrained mechanisms rather than creating new ones, an
assumption behind the persona selection model, steering vectors, alignment pretraining and emergent-misalignment
stories, with only behavioural or activation evidence so far. Updates landing on VPD subcomponents = re-weighting and
continuous interpretability (a monitor for what post-training teaches, e.g. intended capability vs placating the
grader; Hubinger tech tree stage 2). Updates landing outside = new mechanisms, or a blind decomposition. Also framed
as model diffing in parameter space. Planned arc: understand the small testbed, then new tasks, then larger models,
then fine-tuning vs RL updates; expects any method to break down as updates accumulate.

**Why:** the user asked for the log's opening to be "better motivated" from this document after a draft that
focused on the 67M testbed.

**How to apply:** open any write-up or summary with the re-weighting hypothesis and the monitoring use, not with the
testbed model; read results against "existing machinery re-routed vs new association". See
[[reference-research-log-artifact]], [[project-larger-model-next]].
