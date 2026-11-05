# Locating mechanisms on the 67M: done for now, finished with six days in hand (E96)

**Why finished:** the control and the one strengthening task the task allows have both run, and Fable advises spending no amendment. Nothing waits on the user. Total cost $0: two `debug` L40 jobs, 52551 and 52567, about 6 min. Provisional, no independent check.

**What was found.** Switching off each of the 24 heads in turn, with no prefix score, finds copying head (2,4) on all three seeds: the fine-tune's probability of the target falls from .99 to .024-.071, and the lookup is spared. The original model's copying falls too, so gain and capability share that head at this grain. On lowercase nothing is located at single-head or best-pair grain, which reads "unreadable at this grain", not "no mechanism". A new circuit built from old parts still reads as reuse.

**Cheap shortcuts.** Gradient salience fails on the control (rank 2, 2, 19); the path integral matches the true fall and the single step does not. The probe's top place is within noise. So nothing cheap scales past heads yet.

**Filed leans that missed.** The probe's "not top 3" and the salient units' "small" effect. Neither decided anything.

**The user's gain-against-capability point.** Shared on copying. On lowercase no part removes the gain, so the question is not reached. Two post-hoc hints only, each read on different behaviours: head (1,5) takes the original model's copying to .09 and costs .13-.15 of the copy gain; and lowercase's top 320 MLP units lower the original model more (1.05-1.08) than the fine-tune, so the gain survives them.

**Pending decisions:** none. **Spend:** $0. **Blocked or lost:** nothing.

Report `reports/analyses/2026-11-05_analysis_e96-locate-mechanism.md`, notebook `notebooks/E96_locate_mechanism.ipynb`, advice `reports/reviews/2026-11-05_review_e96-next-fable-advice.md`. Not run, listed there: greedy head sets, position-restricted ablation, other unit counts, the narrow lowercase pair, the ladder, a planted head.
