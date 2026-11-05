# E99: refine and verify the new-against-reuse definition on the 67M

*Definition-lead, 2026-11-05. Status: draft; the user approved the scope in chat ("go ahead", 2026-11-05). Fable's advice: `reports/reviews/2026-11-05_review_e99-fable-advice.md` (to come). Definition: `decisions.md` rows 81 and 115; apparatus rows after them; the correction row of 2026-11-05.*

**The request, in the Terms' words.** Our definition is the modified Jain et al. 2024 definition (`decisions.md` rows 81, 115). Its details are not settled and it has not been validated on the old 67M experiments. E99 settles the open details and then checks the definition's verdicts on every old 67M task, against fine-tunes whose answer is forced by construction.

## Design check

- **Goal.** `GOALS.md`, "If no simple Φ works": is that because post-training teaches new capabilities rather than strengthening existing ones? That question needs a measurement of "new" and "existing" that does not use VPD. Every old 67M reading that compared new against existing rested on labels assumed from how each task was built (E90's table). A verified definition turns those assumptions into readings, or shows which cannot be read.
- **Nearest tried.** E90 (definition applied to the old tasks: one located-mechanism label, constant answers "does not apply", copy rows "needs the input token"), E95 (token reader loses more than any state; checked, holds with caveat), E70 (counting grammar, labels by construction, removal had no effect even on LOW), E62 (copying head), E97 (mechanism-detection-lead's locator, a located part on lowercase). New here: (a) fine-tunes whose route is forced or forbidden by training, so the true answer is known rather than assumed; E70's "by construction" was how the task was built, not what the fine-tune was allowed to use. (b) Reading the fixed-answer tasks through a located part.
- **Patch or step?** A step: it validates the instrument every D3 and 67M "existing" claim is read against.
- **Decisions relied on / departed from.** Relies on rows 81, 115 (removal is the label, verdicts relative to a named C) and ask 21 (a label needs a located mechanism, or a removal clearing its controls and beating a token reader). Possible departures, each proposed to the user dated, never applied silently: a verdict for fixed-answer tasks through a located part (ask 21 kept "does not apply" until a mechanism is found, which it allows); an admissibility rule for abilities that only say "needs the input token".
- **What would change what we do next.** If the definition reads a forbidden-route fine-tune as "reuses C", or a forced one as "not through C", it is not valid as it stands: fix the rule that misread and re-run. If it reads both correctly on a C, its verdicts on the old tasks with that C stand as readings, not assumptions.
- **Control first.** Stage 0, alone, with an early stop: the original model under each removal (it must lose no more than its band) and the hand-written write spared; then the forced and forbidden fine-tunes, whose verdicts are known, before any old task is read.
- **Numbers.** Per state and C: share of gain lost under C's removal, the matched-random band, the control's share. Expected: forced ≈ 1 and above band; forbidden ≈ 0 and within band; the original model within its band; the hand-written write spared. Full list after Fable's advice.
- **Saved.** Per-state summary JSON, per-window reads, the fine-tuned states in bf16 in `results/large/`.

## Open details to settle

1. Fixed-answer tasks (lowercase, emoticon, ladder, many cues): a verdict through a located part.
2. Abilities whose removal only shows "needs the input token": a testable admissibility rule.
3. The comparison band: per state, or the largest over states.
4. Unread rows: class map and its twin; copy tables with no token reader of their own.

## Refinements adopted from Fable (candidate rules; proposed to the user in one batch after stage 3, dated)

- **R-net (detail 1).** A located part's share is read net: the state's fall with the part off minus the original model's fall, over the state's gain. The state's own fall and the original model's fall are printed beside it. Why: on E97's lowercase pair (1,1)+(1,2) the own-terms share is 0.80/0.85, but the original model falls 1.10 against the state's 0.90 on a gain of 1.13, so net is −0.18/−0.16. On copy the two agree (net 0.72/0.73/0.63). Fixed-answer tasks get a verdict only through a located part that passes two admissibility clauses. First, removing the part costs the original model its own leaning above the band. Second, the part's function in the original model depends on the input, on inputs the task never trained on (the cased-minus-lowercase gap with the part off, on cased rows). Otherwise they stay "does not apply".
- **R-adm (detail 2).** C is admissible at a site only if (a) f_C merges inputs (at least 2 keys per value on average and more than one value), and (b) a token-only reader downstream of the site, at matched gain, stays within the band under C's removal. Token identity fails both (E95's reader loses 0.74-0.77 against bands 0.41-0.47). So the copy-map rows read "no admissible C at this site: the removal deletes the input". Admissibility runs before the band.
- **R-band (detail 3).** The band is per state; comparisons between states use E95's paired test. Pool to the largest band only if the per-state band's redraw spread exceeds the gap between states.
- **Detail 4.** Rebuild the class map and its twin (11 min) and read them with the hand-write control. Copy tables without a reader are closed by R-adm (b).
- **What the constructed fine-tunes can and cannot show (Fable).** They are ground truth for "through the part", not for "reuses the old function". A forced fine-tune can repurpose the part; a forbidden one can reroute through other copying heads (E62 Amendment 5). They fix the two poles of the mechanism form. The data-built pairs (copy/lookup, class map/twin) stay the truths about reuse.
- Counting grammar left out of the first pass (E70 Amendment 1's direction is not matched on size). The "restricted to read C's subspace" variant is dropped.

## Stages, control first (two seeds each, MATS `debug`, $0)

0. **Controls, alone, early stop.** Original model only: what (1,1)+(1,2) does on cased against lowercase rows; the copying pair (2,3)+(2,4) on the copy and lookup windows; each removal's band. Stop if the original model's fall under a removal is not reproduced from E97 (`run.out`) within its band.
1. **Forced-part copy.** Everything outside the pair's q, k, v slices is reset to the original weights at every step (`on_step`); o_proj frozen. Expect net 1.00 ± 0.05 and lookup spared. Outside that the code stops: the apparatus is wrong.
2. **Forbidden-part copy.** Trained with the pair ablated (`copying.ablated`) and the pair's slices excluded from the update, read with the pair restored. Expect net within band. Printed beside: all prefix-scored heads off ("rerouted within copying"), and the gain with the pair restored against the native fine-tune's gain (within 10%).
3. **Forced-part lowercase** (as 1) and **forbidden-part lowercase** (as 2; expect it to learn as fast, net within band). If the forced arm cannot push the net share to 1, R-net falls back to "does not apply".
4. **Old tasks under the refined rules**, plus the class map and twin rebuilt. Optional: forced-part lookup, to put a number on the granularity limit.
5. **One table:** task, assumed label, verdict, reason. Rule changes go to the user in one batch.

Matched gain within 2% of the native fine-tune, step cap 10×. Fable's sizes: E62's six states took about 3.5 min; E97's whole-row reads take about 1 min per few parts. So stages 0-3 fit one `debug` job.
