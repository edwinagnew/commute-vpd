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

## Stages (to be filled after Fable's advice)

0. Controls. 1. Forced and forbidden fine-tunes on tasks with a located C. 2. The old tasks under the refined rules. 3. One table: task, assumed label, verdict, reason; proposed definition changes to the user in one batch.

All on the MATS cluster `debug` QoS, $0.
