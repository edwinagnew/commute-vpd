# Working decisions

Choices made for now. They are not goals (`GOALS.md`) and any of them can be overruled. Each one has a date, the
reason for it and where it came from, so it can be revisited on purpose rather than drifted from. If a design departs
from one, say so explicitly.

| Decision | Since | Why | Source |
|---|---|---|---|
| "Commute" is tested by projection with the decomposition frozen (Γ and Δ fixed, only U, V of a few subcomponents may change). Re-decomposition is deferred. | 2026-10-14 | Tests the claim directly and never re-fits, so the brittleness branch cannot masquerade as success | `reports/2026-10-14_plan_transport-test.md`, confirmed by the user 2026-10-14 |
| "Post-training" means fine-tunes we run ourselves on the 24 decomposed matrices of the released 67M Pile model (embeddings, unembedding and RMSNorm frozen), with a step sweep so "small" is measured, not assumed. | 2026-10-14 | The released model has no post-trained sibling; freezing the undecomposed parameters keeps δW where the decomposition lives | same |
| Functional (KL) measures outrank Frobenius ones when they disagree. | 2026-10-14 | A Frobenius fit can be large in parameter space and inert in function, or the reverse; the question is about behaviour | same |
| Faithfulness thresholds are set against our own recomputation on W with the same batches, with the paper's numbers (0.24 stochastic, 0.93 PGD) reported beside them. | 2026-10-14 | Batch composition and precision move these numbers; a within-harness comparison is the fair one | advisor check, 2026-10-14 |
| Compute for this round: RunPod 4090-class via remote-kernels, not the cluster. | 2026-10-14 | On 2026-10-14 the cluster queue had 40 pending jobs with day-long GPU jobs ahead; the whole experiment is under two GPU-hours and under $10 (GPU guide: route by completed-job cost; user agreed mid-session) | user, 2026-10-14 |
