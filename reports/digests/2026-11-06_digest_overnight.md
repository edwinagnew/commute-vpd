Mechanism detection on the 67M for definition-lead, 2026-11-05: finished — the results were verified and one way of isolating a mechanism passed its known answer, as far as the saved 67M fine-tunes allow; what is left needs a new fine-tune or a new method.

# Overnight digest, 2026-11-06: mechanism detection on the 67M

**The answer.** Of the reads tried, only setting a head to the original model's average re-finds the known copying head, so that removal is the one way to name a part for definition-lead's reuse table. Every read that patches the original model's outputs back in credits the last MLP on every task. A second number, the gain lost summed over single-part reverts, tells copying fine-tunes from key-table ones. One choice about the lowercase result waits on the user.

**What ran and what it found.**
- **The earlier head sweep re-checked.** An independent check holds with caveat: every number re-derives, and both filed outcomes stand. It corrects three readings around lowercase, among them that the original model's gap is set by context length. `reports/reviews/2026-11-05_review_e96-independent-check.md`.
- **Locator.** Head removal in the fine-tune's own terms re-finds copying head (2,4) on all three copy seeds, taking 0.56 to 0.65 of the gain, and spares the lookup. Every patch read, our edge-level EAP-IG read included, credits block 3's MLP first. Those reads show where the gain enters the output, not where it is made. `reports/analyses/2026-11-05_analysis_e97-locator.md`.
- **Fed or written.** The filed one-part reading failed as a label. The summed reverts read low on every copy fine-tune and high on every key-table one, on five unseen seeds too; its independent check holds with caveat. Amendment 2 found the emoticon trigger reads low, 0.30 to 0.37, yet keeps all its gain with the copying heads off. So a low reading is not only copying. A site-relative bar was not adopted: its gap of 0.19 fell under the needed 0.22. `reports/analyses/2026-11-05_analysis_e100-fed-or-written.md`.
- **Scoping and literature.** EAP-IG is worth having only as our own edge read: `reports/reviews/2026-11-05_review_e97-eapig-scoping.md`. Removal tools test whether a part is needed, and interchange tools test whether the answer follows it: `reports/reviews/2026-11-05_review_mechanism-location-literature.md`.

**Pending decision for the user.** Open ask 1: with lowercase heads (1,1) and (1,2) at the original's average, the fine-tune loses 0.80 to 0.85 of its gain, which passes the own-terms rule. But the original model falls 1.10 nats against the fine-tune's 0.90. Does a part both models need count as located? Both readings are reported and neither is picked. Further work needs a new E-code: a non-copy reuse pair with at least 24 strong keys, since the planned one found 17; and the interchange-subspace test from the literature review.

**Money.** $0: all MATS cluster `debug` jobs, no pods.

**Friction noted.**
- Stale or half-written inputs crashed jobs: a stage read a half-written results file, and a reader took checksums from the wrong amendment's file.
- A filed tolerance was borrowed from an unrelated check, so a run stopped on a sum rule that a nonlinear MLP cannot meet.
- Lookups and plumbing: unnumbered `decisions.md` rows, missing `results/large` symlinks, and `make lint` refusing scratch paths.
- Tool refusals cost round trips: multi-site edits, `nb-edit` text with an apostrophe, and a research-log version collision despite the lock.

**Blocked.** Nothing in `~/commute-data/blocked.txt` names this lead.
