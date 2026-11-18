# Which premise doubts change a paper claim? Triage of the cells flagged doubtful

*2026-11-18, redteam-lead, cell P1.33. Reading only: nothing was run. Inputs: the 60 notes marked doubtful in
`~/commute-data/premise/`, each cell's own reading (`fleet/cell show`), the log's scoreboard and overview
(`docs/research_log/parts/sections/claims.html`, `overview.html`), the plan of 2026-11-09, and the part 1 and part 2
pages they cite. The note writer saw only the cell, not the record, so each doubt was checked against later cells.*

We sorted the 60 cells whose premise was flagged as doubtful by which claim in the log or the paper plan rests on them,
and whether the doubt moves that claim. Four doubts change a claim as the log states it, all about what the re-gating
share is shown to add, and ten more narrow a claim's scope; the rest repeat what the log already says or concern wordings
of reuse retired on 2026-11-18. For `GOALS.md` this leaves the share's separation of the built pairs standing, but the
sentences saying it detects reuse where cheaper readings cannot, and that the definition agrees with it, go further
than the record.

## The table

"Changes it" means the claim as written should be reworded; "weakens it" means it stands with a scope or caveat it now
lacks; "no" means the record already says what the doubt says, or no current claim rests on the cell. Quotes are from
the scoreboard (claims), the overview, or the part 2 gate page of 2026-11-16 (gate page).

| Cells | The doubt | Claim resting on it | Effect |
|---|---|---|---|
| P2.15, P2.14, P2.5, P2.12, P2.10, P2.27 | Each baseline is scored alone at one chosen setting; steering's reversals close at the fine-tune's own strength, and prior and steering together give all nine directions. | Overview: "The re-gating share detects reuse where cheaper methods do not ... steering agrees on four pairs and reverses on three"; same counts in the claims row on the nine pairs. | **Changes it.** The gate page already says the share adds calls on the equal-size pairs and gap sizes, not a new direction; the overview and claims row have not caught up. Settles: steering at one setting fixed in advance, both forms, on all nine pairs. |
| P2.20, with P2.35's reading and P1.26-P1.28 | Part 1's re-weighting share is a sibling of the share scored on the same build labels, not an independent judge. | Overview: "The definition agrees in direction. Part 2's re-weighting share ... puts the reuse side above the new side of every matched pair (E142)." | **Changes it.** That wording stopped being the definition on 2026-11-18. Its replacement reading agrees in sign on eight of nine pairs but not in order, and that reading tracks map reach, site and size. Settles: the trained pairs read under the definition at the level P1.31 passes. |
| P2.8 | The untrained decompositions also reproduce the model less closely and use one gating rule, so "needs training" is not told from "needs a faithful model or a good gate". | Claims, marked holds: "The share's separation needs the trained decomposition: on a plain SVD it agrees on 2 of 9 pairs." | **Changes it** to qualified: the cell's own reading says the low share may reflect the rule. Settles: a no-training decomposition matched in fidelity, or a second gating rule. |
| P2.33, P2.34 | On hand-made edits whose kind is known, the share reads built rewiring like built re-weighting and like an edit that learned nothing; only the one large-gain edit departs, and kind cannot be scaled apart from size. | Overview's picture: reuse should show as old pieces switched on; and "most promising: the share detects reuse". | **Changes it** (provisional, no independent check). The only test with the kind known by construction failed in the reversed direction. The disagreeing pair (P2.32) is a partial answer. Settles: share gap against gain gap per pair, from saved results. |
| P2.23, P2.24, P2.30, P2.9, P3.6, P3.7 | Each cheap column's "apart" may be update size or training length, never checked beyond size. | Gate page: the share is "the only reading that puts the reuse side higher on every built pair"; claims: "update size on 0". | **Weakens** the cheap columns only; the gate page already lists it. The share is answered on the small model by the three pairs equal in size. |
| P2.28 | Shrinking a saved update by one number is not a model that learned less. | Claims, qualified: shrunk to equal size the share still separates nine pairs, one of six cells on Qwen; gate page: "the share's separation is not update size". | **Weakens.** The small-model claim rests better on the three pairs equal in size by training; P2.38 matches size by training. |
| P2.18, P2.29, P2.30, P2.31 | The top-neuron overlap, the top-K gain kept and the crosscoder each failed their own known-answer control. | Claims: "neurons on at most 3"; gate page: the crosscoder "adds no call the share lacks". | **Weakens.** These columns are descriptions, not baselines the share has beaten. The gate page says so for the crosscoder; P2.37 checks its pipeline. |
| P2.4, P2.21, P1.23 | A real fine-tune against its shuffled twin is learned against noise, not reuse against new; the case and repeat pairs may not differ in reuse at all. | Claims, refuted: "The share tells a real fine-tune from its shuffled-label twin"; gate page: on the repeat label the share is "against part one's reading". | **Weakens** (scope): the twin rows say nothing about reuse either way. |
| P2.6 | The fine-tune read as having learned nothing moved its target 40 times more than a same-size random change, which gives no reading at all rather than a low one. | Claims, holds: the share "reads high even when nothing useful was learned". | **Weakens** for that item; the hand-made edit that learned nothing (P2.33) reads high on its own. |
| P2.3, P2.13, P2.17, P2.32 | The share's agreement with the prior may be partly built in, since its denominator starts from the original's log-odds; the one disagreeing pair also differs in task kind and size. | Claims, open: the share reads more than the prior; overview: on the built pair "the share follows steering, not the prior". | **Weakens** the one-pair sentence; the open row stands. |
| P1.26, P1.27, P1.28 | Grafts are new by provenance only; a token-lookup graft is linearly reachable from the original, and the held-model reading follows site and size. | Section 4 page: the definition's cheap reading "puts four grafted heads far below old-task fine-tunes, and a grafted neuron lookup close to them", grafts as built cases. | **Weakens** the grafts as anchors; the page already says the definition never read a graft. P1.31 is the known-answer test. |
| P1.3, P1.5, P1.6, P1.13, P1.16 | The blind search's match on the hand-made pairs is a ridge fit, and the hard case reads low because it was built to be linearly unreadable; an exact edit later read high. | Claims, holds: "A blind search recovers the label of all four hand-made changes." | **Weakens**; the section 4 matrix already fails that wording on hand-made edits. No current claim rests on it. |
| P1.4 | Imitators never reached the behaviour, and nothing independent says how they were built. | Claims, qualified: imitators get the hand-made label, "yes for the key table, no for the always-on change". | **Weakens** the "no" half, which cannot be told from a cut-off fitted to hand-made weights; retired wording. |
| P3.4 | At the matched learning rate the shuffled-reward run collapses, so RL against it is learning against breaking. | Claims, qualified: RL raises accuracy more than shuffled rewards; plan part 3's brief RL comparison. | **Weakens** any matched-size RL-against-null reading, including P3.6's rank result; earlier nulls at a lower rate held. |
| P3.1, P3.2, P3.3, P3.5, P3.8, P3.9, P3.10, P3.11 | On Qwen the share saturates or cannot be computed, real updates sit outside the pieces as random changes do, and one-draw bands move. | Claims, no power; overview: "Nothing tells reuse from new on Qwen yet." | **No.** The record says the same; P3.2's saturation cause is still untested. |
| P1.7, P1.9-P1.12, P1.14, P1.15, P1.19-P1.22, P1.24, P1.25, P1.30 | Built controls defined by unreadability, unlearned ladder rungs, wordings set by unmade choices, an unbuildable replaced circuit, retention that tracks size. | Claims: absolute labels refuted, built skill no power; section 4: no wording passes all four tests, rule retention follows size. | **No.** These are retired wordings or readings the log already reports as failing; the definition is the sentence, not these readings. |

## What the four changes have in common

All four concern how far the re-gating share has been shown to go beyond the labels the pairs were built with. The
separation of the nine built pairs stands, including the three equal in size. What it is shown to add over the cheap
readings taken together, and what kind of change it reads, are narrower than the overview says.

## Not run

- No run, refit or new reading; every verdict rests on saved readings and the pages named above.
- Each cell's result files were not opened; the check used `fleet/cell show` and the cited pages.
- The steering counts at the fine-tune's own strength were not tallied per pair.
- The gain of each side of the nine pairs was not compared with the share gap.
- The overview and scoreboard sentences named in the four changes were not reworded here; the change is queued for the log.
