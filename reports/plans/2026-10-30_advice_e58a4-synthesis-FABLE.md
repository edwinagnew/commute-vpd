# Advice: after E58 Amendment 4 (job 46426) — what the synthesis can claim, what the shuffle says, and where the line goes

*Fable, 2026-10-30, on the lead's three questions. Read: the plan's Amendment 4 and its pre-run review,
`predictions_amend4.json`, `amend4.json` (every read), `amend3.json`, `run.json`, the E58 card and next-steps section of
the research log, my advice of this morning. Terms as in `GOALS.md`: r is the re-gating share, "the ratio" the
stand-in ratio, x the held-out carry, "elicit" the half-known map, "cipher" the values the model cannot know.*

**In plain words.** The three seeds made the picture clearer, not kinder. Across maps whose carry to unseen keys runs
from a fifth to four fifths, r differs by about a tenth, three times its seed noise, so the filed rule failed and the
sentence "r reads the same whatever carries" is gone. What survives is larger: every map built from the model's own
guess for a key reads r 0.72 to 0.92, and every map it cannot guess reads 0.29 to 0.47, on three seeds, two key counts
and two steps, a gap of about eight seed ranges. The map differences inside the top band do not follow carry: the map
with the highest r has the lowest carry. The shuffle, which keeps elicit's words but breaks the key-to-guess link, lands
with the ciphers, just under the cut, so r mostly reads whether the target is the model's guess *for that key*, with a
small credit for a word it can emit somewhere. One five-minute $0 job firms that last sentence; then the 67M is done and
the same table goes to the Qwen anchors.

## 1. What the synthesis can claim (Q1)

**Claim, one sentence.** On the 67M, the re-gating share and the stand-in ratio, read in the same row (same keys,
step and key count), separate fine-tunes whose targets are the original model's own key-conditional guesses (copy, top
word, the half-known map; r 0.72-0.92, ratio 0.96-1.21) from fine-tunes whose targets are not (random ciphers and the
shuffled map; r 0.29-0.47, ratio 0.41-0.65), and that separation does not track how far the learned rule carries to
unseen keys (x 0.10-0.84 inside the top band, with no ordering by x).

**Scored.** For: the cipher control "below" on all three seeds at its gate step (15-16); the ratio half of W passed at
both steps (D 0.18, 0.20 against 0.2: at the bound, say so); the shuffle "cipher_like" (0.475 against 0.5, a near miss,
say so); seed 0 reproduced bitwise; E54/E56's map-versus-cipher separation at 24 keys. Against: the r half of W
failed at both steps (D_r 0.11, 0.13 against bounds 0.07, 0.09), so the filed equivalence "r the same across maps within
twice its seed noise" is false and stays false; Amendment 3's B is "untested_drift" and its verdict "untested";
E58's two original predictions (label agrees with r; label's control) failed. None of these is refiled.

**Unscored, reported with seed ranges.** The 128-key gap between the two bands (0.25 or more at every step, seed
ranges 0.002-0.045); rho(r, x) 0.02 at step 10 and 0.42 at step 30 over nine points (no power; report, do not read);
r falling 0.04-0.10 from step 10 to 30 on every known-target map at fixed carry (copy x 0.82 → 0.80, r 0.86 → 0.77)
while the shuffle's r *rises* 0.39 → 0.47; the cipher's r rising 0.23 → 0.30 with the key count.

**Honest wording of the failure.** "r is not blind to which map was trained: inside the known-target band the three
maps differ by 0.11-0.13, three times the largest seed range, and the order (top word > copy > half-known) is the same
at both steps. The order does not follow carry; we do not know what it follows (trained gain, update size and the
value pool all differ) and do not assert a cause." The decision tree's branch 3 applies (D_r under 0.15): the one-seed
line is kept as a reported number, not dropped, and never as a scored claim.

**What this says to the user's question.** On the 67M, with trigger maps, VPD's readings distinguish "the model
already produced this answer for this cue" from "it did not", and do not distinguish "the learned rule generalises" from
"it is a lookup". If elicitation means the former, VPD reads it; if it means a capability that carries, VPD does not.
State the granularity limit in the same paragraph: a new circuit assembled from existing subcomponents reads as the
former. Also state: 67M only, trigger maps only, one decomposition seed, and a single r means nothing off its row
(it moves 0.1 with step and key count on its own, E36, E60, this job).

## 2. What the shuffle reads (Q2)

Mostly the key-conditional guess, partly the token kind. The shuffle sits 0.17 above the cipher at step 30 (0.47 vs
0.29-0.31, about four cipher seed ranges) and 0.25-0.40 below elicit (0.72-0.74). If r read only token kind it would sit
with elicit; if it read only the key-to-guess relation it would sit on the cipher. So about two thirds of the credit
needs the model's own guess at that key, about a third comes from the value being one the model can emit at some key.
Two cautions. (i) One seed, 0.025 under the cut: the scored label stands, but the synthesis should say "at the
cut", not "cipher-like". (ii) The shuffle's r rises with training while every other map's falls; its G_spec at step
30 (17.5) matches the cipher's (16-18), so this is not under-training. Report it as an open observation, no cause.

## 3. Is more 67M work worth it (Q3)

One $0 job, then no. **The job:** the shuffle on seeds 1 and 2 (two fine-tunes, reads at step 30 and the gate step,
about five minutes on `debug`), reported beside seed 0, not scored (the filed read is seed 0's). Rule for the write-up,
fixed now: if the three-seed mean is under 0.5 the sentence above stands; if any seed reads 0.5 or over, the sentence
becomes "the shuffle sits at the midpoint between the bands: r reads the key-to-guess relation and the token kind in
about equal parts". This is the cheap recommended check the user's standing instruction says to run as an FYI.
**Not worth it:** chasing what orders the three known-target maps (G_spec, trained P, pool); a second decomposition
seed; more key counts. Each is a day for a sentence the Qwen result will overwrite.

**Then the synthesis**, built around one figure: x on one axis, r (and the ratio in a twin panel) on the other, every
map and seed at 24 and 128 keys and both steps, the ciphers and the shuffle as the low cluster, with the seed ranges
drawn. Open with the plain-words paragraph above, carry the "Scored / unscored" split as a table, close with "Not run".

**The line's next question**, unchanged from this morning: *on a real post-trained update, where does r sit, and does
the same-row separation survive?* E45 Amendment 2 (approved, $0, in progress) gives the Qwen anchors: the
invented-facts update is the cipher's analogue (targets the model cannot guess), the format-template update and the
GSM8K fine-tune are candidates for the top band. Read them through the same table at matched trained gain, with the
67M bands as the filed expectation: anchor 2 under 0.5, anchor 1 over, GSM8K open. That plan is gated by its own
rules; nothing here adds to its spend.

**What would change this advice.** Shuffle seeds 1-2 both over 0.55: then token kind is half the reading and the
claim's first clause becomes "targets the model can produce somewhere", as I said this morning. The Qwen anchors
reading under 0.1 apart: then the bands are a 67M artefact and the line needs a real-model number (E50's restore
share on known-kind anchors) before any re-gating claim leaves the testbed.
