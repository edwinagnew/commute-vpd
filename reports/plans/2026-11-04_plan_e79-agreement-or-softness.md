# Plan: E79, is the masked fall in E76 agreement or softness?

**The request, in the Terms' words.** D1 asks for a map on Qwen3-1.7B that keeps the decomposition faithful for a real update. E76
trained teacher training with a masked term (arms B, L) beside the clean term alone (A), on u1 (E40 step 10). The filed label is "works" for B and L.
The independent check (`reports/reviews/2026-11-04_review_e76-b-works.md`) found no bug and says the label says less than its name: KL(FT ‖ ·)
rewards a soft output, and top tokens, entropy, FineWeb and the student's own masks were never read. This job reads them on the saved changes.

## Design check

- **Goal.** `GOALS.md`, D1. It says whether E76's arms count as faithful before any paid run rests on them.
- **Nearest tried.** E76 and its amendments (where the fall sits, not what it is). E74 (one temperature on W0's masked read, floor 0.82 on other rows). E72 (student under own masks, 0.103-0.105 for every state on u1). None read E76's saved changes for agreement.
- **Patch or step.** A patch on E76, the check before the plan's next branch. New E-code since E76 has no amendments left.
- **Decisions.** Rests on ask 24 and ask 25. No departure.
- **What changes next.** See below.
- **Control first.** Stage 0: W0 (zero change) and then each saved arm reproduce their E76 step readings of (i) and (iv) on the same rows and draw, or the job stops. Positive control: W0 masked against W0 clean.
- **Class.** 1, analysis of what exists: a GPU pass of the harness on saved states. $0, one L40, `debug`, under 30 min.
- **Saved for audit.** Per-position forward KL, reverse KL, top-token agreement and entropy for each state, row set and draw, in `results/large/`, plus the summary in `results/read.json`.

## The run

States: W0, A, B, L, from `experiments/qwen_masked_teacher/results/large/*_offsets.safetensors`, read with E76's `read_state` apparatus. Teacher FT = W0 + u1 as there.

Rows: M = E76's 8 reading rows (maths 0-7). W = FineWeb rows 32-39 of `qwen_adversary/results/tokens.json`, in no arm's training pool (`pools()` takes 0-31), row length 256. Masks are Γ on W0's forward, frozen, as in E76, in fp32 reads; W's g is computed in this job. Draws: E76's `read_seed` and a second one, whose spread sets the band.

Order: control; positive control; then W0 reference numbers and the temperature floor; then A, B, L; own-mask reads last.

## Numbers

Against FT clean, all positions, per row set. W0 and arm values for number 1 on M are known; the rest have an expected value only as a direction, since the rule's bands come from the job's own first stage.

| # | number | W0 | what it is blind to |
|---|---|---|---|
| 1 | forward KL(FT clean ‖ state masked) | M 1.334; A 1.127, B 0.280, L 0.390 reproduced within 1e-3 | rewards softness |
| 2 | reverse KL(state masked ‖ FT clean) | measured here | punishes softness, can be dominated by few positions |
| 3 | top-token agreement with FT clean | measured; below state clean's | how confident, which tail |
| 4 | mean entropy, masked, beside FT clean's | measured | where the softness is |
| 5 | temperature floor: min over T of forward KL(FT clean ‖ W0 masked at T), grid as E74's | M under 1.334; E74 read 0.82 on other rows | only for W0 |
| 6 | Γ on the student's forward: masks from the state's own forward, numbers 1 and 3 on M | W0 reproduces number 1 on M's draw with own = W0's masks | Γ frozen on W0's weights |
| 7 | control: forward KL(W0 clean ‖ W0 masked) and its top-token agreement on M | K0 near E74's 0.044 on its rows; agreement high | apparatus only |

Numbers 1-4 also on M's response positions alone (assistant turns, by the turn labelling E73 uses), since E73 read u1's KL mostly elsewhere.

## Rule, on B then L, row set M, all positions, draw 1

Band b: the larger of the two draws' difference on the state's number and, for number 3, one position-level binomial standard error. Let a0 = W0 masked's number 3, a1 = the state clean's number 3.

- **Agreement:** number 3 ≥ a0 + (a1 − a0) / 2 and number 2 ≤ half of W0 masked's and number 1 < number 5, each beyond b.
- **Softness:** number 1 ≥ number 5 − b, or number 2 > 0.9 × W0 masked's.
- **Mixed:** anything else, reported as read. If a1 ≤ a0, the agreement clause cannot be read: "no call", reported as read.
- **Survives on unseen text (W):** forward fall on W at least half of W0 masked's reading there, beyond b. W0 masked's reading at or under W's band: "no room on W".
- **Own masks (B):** number 1 under own masks within 1.5 × that under W0's masks, or else "does not survive Γ".

Leans, filed before the run. B: agreement, medium (0.28 is already below E74's floor on other rows). L: agreement, weak. W: survives for B, weak (M's row 5, the row with no template, fell 57%). Own masks: survives, weak.

Edge cases the code writer runs on the rule: zero and negative a1 − a0, NaN, ties on a band, an empty response-position set, zero W0 on W.

## What each result does

- **Agreement and survives:** the fall is real behaviour on unseen text; what is left of E76's doubt is size. The user gets B, A's smaller change and the size numbers side by side, with the next step: R100 at 200 columns on the cluster for $0, against the plan's paid run.
- **Softness:** E76's label says nothing for D1 on this reading. The user gets it with the fix: a term that reads agreement, a new plan.
- **Mixed, or does not survive Γ or W:** reported as read, with what each part does to the sentence D1 could claim, to the user before any next run.

## Not run

Other seeds, a lower rate, a random-teacher null with B's recipe (the check's suggestion; it would show the term can copy anything, which is not a failure of D1's map as such), a size-limited B, and what the change does to any component.

## Amendment 1 (2026-11-04): does the own-mask read have teeth?

Job 51117 read, on M, Γ on the state's own forward, against FT clean: A forward KL 0.078, reverse 0.069, top-token agreement 0.915, entropy 1.56 (FT clean's 1.53); B 0.198, 0.319, 0.853, 2.48; W0 1.334, 0.623, 0.799, 0.61. So the clean term alone, at 0.52 of the update's squared size, reads better on all four numbers than the masked arm once Γ reads the student. The plan held this number at a secondary place, and E68's reading A says own masks gave 0.24 of the gain whatever was placed. Three things could make it empty: Γ opens many more subcomponents on the student, so "masked" is nearly clean; the read passes anything near FT, so it cannot fail a bad placement; it holds on maths only. The anomaly rule asks that this is run down before a reading rests on it. Class 1, $0, about 3 min, `debug`.

Same job shape, control first: W0 under own masks reproduces 1.3336 within 1e-3, A 0.078 within 1e-3 (same draw), or stop.

States: W0, A, B, L, as run. Two bad placements of known wrongness: **P**, A's saved change with its columns permuted within each site (seed filed in the code; same size per site, same blocks, wrong columns); **C**, E76's control student, trained towards W0 plus a random change.

Numbers, per state, Γ on the state's own forward, draw d1, rows M and W:
- a. forward KL(FT clean ‖ state own masked), reverse, agreement, entropy, as before; W is new.
- b. forward KL(state clean ‖ state own masked), the state's own faithfulness. W0 on M 0.043 (number 7).
- c. Σg per position over all sites, and the count of g > 0.5 per position, mean over rows, with the ratio to W0's on the same rows. W0's ratio is 1.
- d. forward KL(FT clean ‖ state clean), for P and C (known for the rest), so P's and C's numbers have their clean reference.

Rules, on A, M, bands as in the plan (two draws where the state has them; d1 alone otherwise, with W0's spread as the band):
- **Opens:** A's Σg ratio or count ratio above 1.5. Then the own-mask read says little: masked is nearer clean.
- **Teeth:** P's number a forward at least 3 times A's on M and above P's clean by no more than a factor 3 or it is reported as read; C's the same. If P's clean KL itself is under A's own-mask 0.078, "P not wrong enough", as read.
- **Faithful under own masks:** A's number b at most 2 times W0's on the same rows, on M and on W.
- All three of not opens, teeth and faithful: the own-mask read is a test A passes. Anything else is reported as read.

Leans, filed before the run: opens, weak (the fall from 1.33 to 0.078 for so small a change wants explaining); teeth, medium; faithful on M, medium, on W, weak.

Edge cases for the rule's code: P's clean under A's read, a zero Σg denominator, NaN, ties on the bands.

**What each result does.** Passes all three: the clean term alone is the leading candidate map, at less than the update's size, with no masked term; this goes in a headline claim, so it gets an independent check before the user is asked for the next step. Opens: masks read through Γ on the student are not comparable with W0's; B and A are then both read under frozen masks, as E76 and this plan did. No teeth: own-mask read has no use for D1; the clean table stands alone.
