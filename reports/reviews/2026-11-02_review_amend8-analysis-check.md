# Independent check of the E34 Amendment 8 analysis

*2026-11-02. Checked `reports/analyses/2026-11-02_analysis_e34-amend8-base-control.md` (called "the report", line numbers are
its own) against the result files, `amend8.py`, the saved offsets and the logs. The reviewer did not write the code or run the
jobs. New $0 reading: `~/commute-data/scratch/commute-lead/amend8-check/cos.py`, output `cos.out` beside it.*

**Verdict in plain words.** The numbers and the scoring are right, with one rounding slip. "Passes stand" survives every attack
tried, because each pass is read on its own state. The other two halves of the headline need downgrading: direction is shown to
differ between two teachers, not to be the teacher's, and size is the recipe's only on this narrow task. The drafted card contains
one statement the E41 logs refute and one comparison that picks the favourable link, so it should not be published as drafted.

## New reading used below

Cosines over the saved offsets (dU, dV) of all alive columns, from `results/large/amend8/*/*.pt`. No model load, so these are offset
space, not δW: the one pair readable in both gives null-real 0.057 here against 0.090 in δW. Use them for pattern, not for digits.
Every state shares the same U and V at the base, so the offsets are comparable. δW cosines for these pairs need U and V, which this
check was told not to load.

| Pair, step 1,000; "move" = end minus own start | Same | Differ | Cosine |
|---|---|---|---|
| Self-labels Adam, self-labels SGD | batches, labels, rows | optimiser | 0.765 |
| Random start's move, round trip's move | teacher, batches, windows | start | 0.702 |
| Null student, random start's move | batches, windows | teacher, start | 0.181 |
| Null student, round trip's move | batches, windows | teacher, start | 0.129 |
| Real student, random start's move | batches, windows | teacher, start | 0.100 |
| Null student, real student | batches, windows, start | teacher | 0.057 (δW 0.090) |
| Real or null student, any arm on other rows | nothing | all | -0.002 to 0.019 |
| Real student, round trip's end | | | 0.642 |
| Random start, its own end | | | 0.915 |

Offset norms at step 1,000, as a share of the real student's 1.027: null 1.30, self-labels Adam 1.21, round trip's move 0.92, random
start's move 0.76, coefficient 0 0.76, self-labels SGD 0.52, true-label Adam 2.75. Null-real per matrix is positive on 23 of 24
matrices (0.02 to 0.11), so the small overlap is systematic, not noise.

## Corrections, ranked

**1. "A student's direction is its teacher's" overstates (lines 11, 119, 295).** The real student's cosine with its teacher is 0.281,
under 8% of squared length, and `techniques.html:70` scores that same quantity as a ✗. The evidence is that two teachers give two
directions. Replace line 11's clause and line 119's first sentence with: "Which way a student moves depends on which teacher it
copies, on the one pair of teachers tried, though it overlaps that teacher's own change only modestly (0.28)." Line 295: replace "the
direction it finds is specific to the update" with "its direction differs between the two teachers tried".

**2. "Under 1% is shared with the null student" contradicts its own stated assumption (lines 122-123).** The 7% comes from a shared
vector plus orthogonal walks: dot product over norms equals shared squared length over total. The same model on null-real gives
0.0897 × (null norm / real norm), with that ratio 1.30 in offsets and 1.41 on the largest matrix. That is about 12%, not 0.8%; 0.8%
is cos², a different quantity. Correct text: "About 7% of the real move's squared length varies with batch order, and about 12% is
shared with the null student, some of which may be the shared batches."

**3. Nothing shows reproducibility marks the real student (line 125), and 0.366 is not a null for copying (lines 124-125, 278).** No null
student was run at a second batch order, so "what marks the real student is that it reproduces" has no comparison. The moves with
teacher = base from two different starts share 0.70, which suggests any teacher may reproduce; both also share batches, so this does
not settle it. And the null student really does copy its teacher: gap 0.46 to 0.0097. Its 0.366 is a second example of copying, not a
reference for no copying. The reference would be real student against null teacher, or null student against T1000, neither read.
Replace line 125's last sentence with: "What marks the real student is that it carries the behaviour; whether its reproducibility is
special is unread." Line 278 verdict: **Stands, unresolved**: "Two students read 0.28 and 0.37 with their own teachers; the cross
pairs that would show whether this is specific were not read."

**4. The card's "every round's restarted optimiser jolts the predictions" is refuted by E41's own logs (line 256; also 225).** In
`results/large/run_e41_amend2.log` the seven real links run from start loss to largest logged later loss: 4.86 to 1.57, 0.336 to 0.053, 0.179
to 0.099, 0.109 to 0.092, 0.060 to 0.070, 0.062 to 0.058, 0.055 to 0.054. In `run_e41.log` the long links go 0.130 to 0.108 and 0.065
to 0.047. Only one link rises, by 18%. The jolt of 0.3 to 0.5 nats appears with the base as teacher, from a small gap. It may depend on
a broad trigger distribution, untested. All of these are H100 logs at steps 1, 2, 3, 5, 10, 20, 30, 100. Delete the clause from the
card. Line 225-226: "Restarting Adam raised train KL to 0.3 to 0.5 nats with the base as teacher; the real links in `run_e41_amend2.log`
show no such rise, at most 18% at the logged steps." Option (c) stays a possible test, but its rationale loses this support.

**5. "The same 17%" picks the favourable link (lines 219-220, 256; also 279).** The chain's T100 link grew 52%, 0.0141 to 0.0214. Only the T1000
link grew 17%. Nothing-to-copy growth is 17% and 25%, one draw each, on another card, from a start of 0.018. The report elsewhere calls
one 1.39 lookalike a coincidence (line 191); this one should get the same care. Correct lines 219-221: "The T1000 link grew 17% and the
T100 link 52%; on the L40 the round trip grew 17% and the random start 25%, from another start size. So some growth is what the recipe
does anyway; one pair cannot say how much, and the earlier link grew more." Do not say "what the recipe does anyway" of the whole.

**6. The chain's damage comparison mixes two quantities (lines 221-222, 256).** The 0.009 is chain against one-shot; the 0.0014 is one
run's rise from its start. The T1000 link itself lowered damage, 0.0460 to 0.0269. The excess was already there one link earlier: chain
T100 0.0460 against one-shot T100 0.0363. And the round trip also lowers damage. Replace "so the recipe alone does not account for it"
with: "The excess over one-shot, 0.009, was already there after the T100 link, 0.010, and the last link lowered damage; so this does not
show a link adding damage." Card: replace its last evidence sentence with "The long repeated run ends with more damage on other text
than one round (0.027 against 0.018); the excess was there a round earlier, and the last round lowered it."

**7. "Size is the recipe's" holds only on this task; "mostly the optimiser's" is untested (lines 11, 111-115, 276-277, 296).** On E39's lowercase rows
with Adam 1e-5, batch 8, 1,000 steps, size follows what is taught: self-labels 0.0236 (L40), a real student of the broad fine-tune 0.0365
(H100, `teacher_spread.json` `.states.lower_s0`), true labels 0.054. Cross-card, but the size is stable across cards in this report's own
table. Offsets support the emoticon claim for Adam: 0.76 to 1.30 of the real student's. They do not for SGD at 0.52. Random start and
round trip sizes include their start, so line 111's "whatever it copies" mixes moves with positions. Line 111: "On this narrow task every
Adam student ends within 0.76 to 1.41 of the real one, however measured." Line 113: add "E37's real student on lowercase rows, 0.0365, also
departs from the self-labels' 0.0236 there, on another card." Line 114: "explains" becomes "fits". Line 296: "mostly the optimiser's"
becomes "on this task, nearly independent of the teacher; no arm varied the optimiser or the learning rate on the hook path". Line 277:
"Add that on narrow tasks its size says little."

**8. "The weights do not" return is too strong (lines 241-245, 312; line 382).** Size is blind to direction, as line 305 says. In offsets the round trip's
end has cosine 0.64 with the real student, and its projection on the real move falls to 0.74 of the original length, so about a
quarter is undone along that direction, while the random start undoes about 1% of its own. The filed miss stands: size return −0.17.
The lean of 0.2 to 0.6 was scored on size, so its miss stands as well. Line 241: "The outputs do; the weights' size does not, though about a quarter of the
original direction is undone (offsets)." Line 312: keep the sign, add "in size". Lines 244-245: "Many read and write vectors compute nearly
the base's function" is a hypothesis, say "may". Line 382: the squared-offset return is read here, 1 − 1.184²/1.027² = −0.33; remove it from Not run.

**9. "Nothing to copy" and "12 of 13" flatter the null result (lines 9, 54, 171-173, 276).** The null student has something to copy: a teacher at
KL 0.53 and damage 0.122, as the dated note itself says. It is also the only arm matched to the real student in start, batches and
teacher-from-a-fine-tune, and the one that does not pass. The 13 reads hold 7 repeats of the same runs, one untrained. Lines 9-10: use the
replacement below. Line 54: "five of the six end states sit at or below the band's top, and 12 of 13 reads". Line 171 says every run with
nothing to copy, then line 173 counts the null student as "a teacher adds more": call it a null teacher in both. Line 276: "A pass at about 1
is what the recipe reads with nothing to copy" becomes "five of six null runs end at about 1, so a pass near 1 shows no harm and no more; the
matched null student does not pass on its one start".

Replacement for lines 8-11:

> We trained the decomposition's read and write vectors with only the unchanged model to learn from, to see whether these methods learn
> a gap between base and decomposition. Nothing breaks: five of six runs end at the unchanged model's level on the adversarial check, and
> the sixth, a student of a fine-tune that taught nothing but moved predictions, reads 1.39, inconclusive on one start. So passes stand and
> mean "no harm seen"; on this task how far a student moves barely depends on its teacher, which way it moves does, and ratios below 1 mean
> only "at the base".

**10. "The fall is not the student learning the fine-tune" goes past the data (line 177; line 170).** With no teacher the fall reaches a third to a half of the
real student's, so the rest, and the null student's larger fall, is there only with a moved teacher. And Δ-out is the one reading of the
user's gap itself: the KL between a state's weights and those weights without Δ (`faithfulness.py:101`), and audit lines 90-91 say so. Line 177: "So
part of the fall needs no teacher and none of it is a loss term; the rest comes with copying a moved teacher, by a mechanism unread." Add
the bound that answers the user: "The whole gap on Pile is 0.012 nats against a start gap of 5 to 13 nats for a real teacher, so even if
all of it were learned indirectly it could not be the update." Line 170: "does not need a teacher" becomes "does not need a fine-tuned teacher":
the random start has the base as teacher.

**11. A claim was missed (table, lines 276-289).** `claims.html:57` and E34, E47 contrast "every copy passes where doing nothing fails". The null pair shows the
same pattern with nothing taught: do-nothing 2.64, student 1.39. Add a row: "Teacher training passes where do-nothing fails, E47 and
E34 | **Complicates** | Each read stands. The fall from do-nothing to student also happens on a null teacher, on one start, so it does
not by itself show the update was carried." Line 287 then cross-refers.

**12. Card, option (b)'s branches do not follow (line 257).** The one-shot null student already reads 1.39, so a failing null chain would not show that repeating
harms; it may be the null teacher's damage. No branch covers an inconclusive reading, the likeliest one. Replace with: "Read against the single
round on the same teacher, which reads 1.39 on one adversary start. If the chain reads clearly above that, repeating harms by itself and any
repeated run on the larger model needs a new recipe; if about the same or lower, repeating adds no harm we can see, and size in repeated runs
says nothing either way." Also state the card: line 256 never says the rounds ran on another card. Add "on another card".

**13. Card, recommendation (line 258).** (a) follows: no chain plan exists (grep of `next.html`, `gaps.html`, `GOALS.md`, `reports/plans/`). But "stop
reading size as piled-up error" outruns one pair of growths. Say "do not read size in either direction until a chain with nothing to teach
exists", so it covers the no-pile-up reading too (`e41.html:21`). If the card is only a wording choice with one clear answer, the 2026-10-29 and 2026-10-30
feedback suggests acting and saying FYI; the filed tree does require a card, so the lead decides.

**14. The nearest test of damage was left out (lines 187-191).** The null teacher in Δ and its student have the same damage, 0.122 and 0.120, and read 2.64 and 1.39.
The state differs, so this is no clean test, but it shows damage does not set the check alone. Add it after "a test". Line 191: "are coincidence"
has no test; say "no reason to connect them". "Two lookalikes" names only one pair.

**15. Line 120-121, shared batches.** "Can only inflate" is not a theorem; a shared batch can also oppose. Say "which may inflate it". Readings: arms sharing batches and
windows but not teacher reach 0.06 to 0.18, and arms on other rows are under 0.02; so the whole 0.09 may be shared batches. Add consequence: a draw at 0.11
would read "some", class "partly", outcome "fails" by the filed rule, with no change in meaning. The class is one draw; the reading "mostly elsewhere,
against 0.93" is robust to it.

**16. Smaller points.**
- Line 317: "0.52 to 0.59 from step 30 on" should be 0.52 to 0.58; 0.587 is step 3. The dated note's 0.59 is the same slip; it stays filed.
- Line 196: "The same real student" is the same recipe regenerated: T1000's size differs 2% between cards, 0.0321 and 0.0315. Say "the same recipe's student".
- Line 38: "on rows, see 'Rows and masks'": that paragraph never says self-labels and true labels read damage on plain Pile rows, and coefficient 0 on cipher
  windows. Add it, or the table's damage column misleads.
- Line 224: "reflects fewer steps" is untested; say "fits fewer steps, 700 against 1,000". Lines 223: the last link's 2% is the smallest; earlier ones grew 7% to 29%, from smaller sizes.
- Line 235: self-label Adam 0.0236 against true labels 0.054: "needs the true labels" holds at this step count, no self-label state reached that size.
- Line 318: "threefold" uses 0.004 from E34's plan, not this card. Not run says so; the surprise should too.

## What next: add one job

Between items 1 and 2 of line 323: a null student at batch order 1, with no harness read, in one debug job. It gives cos(null order 0, null order 1), and cos(null order 1, real order 0)
free of shared batches, and with T1000 regenerated, cos(real, null teacher) and cos(null student, T1000), all in δW. It settles corrections 1, 2, 3 and 15. If null students repeat at
about 0.9, direction is a property of any teacher, not a mark of the real one. Item 1 in line 323 stays first: nothing else tells whether 1.39 is one start's noise.

## Attempts to refute the headline

- **"Passes stand": survives.** No read here changes a state's own read. One adversary start, one null draw and one batch order limit the null side, not the passes. E47's four starts
  on real students are separate evidence. What weakens is what a pass means, corrections 9 and 11.
- **Null real at 0.090 near 0.1: survives in substance.** Far below 0.927, whichever side of the line. Correction 15 covers the scored outcome's fragility.
- **Direction: downgraded,** corrections 1 to 3.
- **Size: downgraded to this task,** correction 7.
- **Does the hook path see the gap? Confirmed by code.** `kl_exact_zero` (`teacher.py:31-37`) gives zero gradient at equal logits. `component_training.py:5` adds only the offsets' change.
  Rounding off the base (0.0005 to 0.0012) is real, but far below real gaps.

## Checked and found right

- Every value in the main table (lines 61-79), against the seven result files. Base values line 81, identical in all seven jobs. Real student identical in `n2` and `n3b`, offsets difference 0.
- The gap table (lines 137-140) and its subtractions; held-out and train values; the 0.457 at step 3 (`run_n1.log:148`); distance estimates 0.030 and 0.023.
- All counts in lines 54 and 86, the band's top 1.208, lines 1.241 and 1.813, 1.396 and 1.612; denominators 0.692 and 0.699.
- Every row of the comparison table (lines 199-216): E41, E41 A2, E37 A1, E39 A2, including the Δ-out values from logs. M0's 25.99, 0.900, 0.0514. Links' start range 0.055 to 0.34. Excess removed 1.18.
- Scorecard: N1, N2, N3a, N3b, N4, N5 all match the filed rules and `amend8.py`'s functions, and each stored score. Each lean's hit or miss. Guides table. Controls line 340-342. The gap-line amendment, 0.0037: the outcome is unchanged and the predictions file untouched.
- Causes flagged as open or not tested in lines 152, 159, 161, 185 are correctly flagged.
- Cross-card comparisons in lines 119, 195-197 and 218-221 are stated; the card's is not (correction 12). `check.py`: 0 errors. Figures `amend8_check` and `amend8_direction` match the numbers.
- Masked terms as cause of the fail (line 100): a one-factor contrast, 1.006 against 25.99, same card, windows, batch order. Holds.

## Not checked

δW cosines beyond the four stored (need U and V). Figures other than two. The notebook section, TRIED row and log queue in commit `f0c097f`, which carry some of this wording and need the same edits. E34's arm B 0.73 beyond the audit's 0.726 and the log.
