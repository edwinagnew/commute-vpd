# E146: why steering reads the new side higher on three built pairs

*Cell P2.14, compare-lead (writer p214-writer), 2026-11-10. Card `reports/plans/2026-11-10_plan_p2-14-card.md`; filed rule
`experiments/steering_reversal/results/predictions.json`; code `experiments/steering_reversal/` (`run.py`, `rules.py`,
`plot.py`); job `cluster/steering_reversal.sbatch`. Sides are labelled as built; no verdict on reuse.*

**Design check.** Goal: part 2 compares the re-gating share with cheaper readings on the 67M; steering's column has three
pairs that read the opposite way to the share, and this says whether that is a property of the vectors or of how the
reading was set up. Nearest tried: E143 (`experiments/holes_fill`), which found the reversal and did not explain it; E135
set the fair form. This is a patch on E143's reading, inside the cell's "done when". Decisions relied on: the fair form of
2026-11-09 (unchanged as the reference). No rule is added to a reuse verdict; the rule here names a cause of a steering
gap only. What changes what we do: a named cause means the steering column carries a stated caveat (or a strength-1
column) in part 2's figure; none named means the column stands as E143 read it. The control runs first (below).

**What runs.** On E143's saved states and the share's read rows, 28 fine-tunes: half-known (5) vs cipher at equal size (5)
and at the cipher's own step (2, reported only), rank-1 (8) vs cipher (8) at step 10. Each pair is scored in the kind the
method table reads (per key on the half-known pairs, pooled on rank-1 vs cipher); the other kind is reported. One thing
changes per reading, each against E143's filed reading:

| reading | what changes | candidate cause |
|---|---|---|
| strength 1 | the fine-tune's mean change is added as it is; layer still by the filed pick | the strength search (found by reading E143's selection grid: the reuse side peaks at strength 2 and falls beyond, the cipher keeps rising to 4 or 8) |
| layer 2, layer 3 | both sides at one layer; strength by the filed pick there | (b) the layer the vector goes in |
| random subtracted | each state's mean of 8 norm-matched random vectors at its setting is subtracted | (a) the per-key random reference reading below zero on the reuse sides |
| pair's mean gain | denominator is the pair's mean gain, not each fine-tune's own | (e) gain normalisation |

Not set as rivals: (c) the position set, since steering writes at the answer position only and its vectors are means
there; a vector at other positions is a different reading, not one change. (d) the half-known map being partly present
already is a mechanism, not a setting the saved states can change; it is listed in "Not run".

**Expected values.** At W0 (zero update) every rise is exactly 0 and the gain is 0, below the floor: every share nan, every
pair "no call", nothing named. The filed reading reproduces E143 per state within 1e-4 and holds the reversal on both scored
pairs (gaps -0.24 and -0.27). Random vectors at strength 1: mean within 0.1 of 0 on every state (reported).

**Filed predictions** (`predictions.json`). P1: strength 1 is named (moves the reversal on both scored pairs and halves the
relative gap): rival A, medium-low confidence, since rank-1 vs cipher sits near -0.07 on the selection grid. P2: neither
layer 2 nor layer 3 moves it. P3: the random reference moves the per-key pair but not the pooled one, so it is not named.
P4: the pair's mean gain does not move it. P3 and P4 are arithmetic on E143's numbers, filed so the run checks its own
recomputation. A dry run with E143's selection-grid rises standing in for the read rows (no model) gave rival A, every
edge case as filed, and the do-nothing state naming nothing.

**Control, first and alone (stage 0, exit 1 on a miss).** A zero update's vectors change no logit (at most 1e-9) at layers
1-3, strengths 1 and 8, both kinds; e54/elicit_s0 per key and c3/cipher_s0 pooled reproduce E143's filed shares within
1e-4. In stage 1 every state's filed share must reproduce E143 too, else the job stops.

**Cost.** Debug L40, about 12 min (E143 read 50 states with more passes in about 12 min including load), 24 G host memory,
$0. Stage 1 stops itself if the projected time passes 25 min. Nothing large is saved.

## Gates
Not a gated run ($0, under 30 min).

## Not run
- The position set (other positions than the answer).
- Splitting read rows by whether W0 already gets the key right (the half-known map's known half), the direct test of (d).
- Strengths below 0.5 or a strength matched by vector norm across sides.
- Any Qwen pair; any verdict on reuse.
