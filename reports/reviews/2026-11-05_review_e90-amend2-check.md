# Independent check of E90 Amendment 2: does removing token identity at block 1 separate the copy maps from the random map?

**Verdict: holds with caveat.** The filed branch (ii) is what the file says, I found no bug in the draws, bands or verdicts, and it survives every redraw of the 20 random removals I could simulate, so it is not fragile to the draw set. The copy maps do lose more than the random map beyond what the random removals explain, but that is a difference of degree on one pair of tables, and the random map's "within" comes from the largest-over-states rule. It also rests on a control that may not outlast a fuller variance match, and nothing saved shows that "reuses token identity" means more than that the answer needs the input token after block 1.

*2026-11-05. Results `experiments/new_vs_reuse/results/amend2.json` (job 52444). Line numbers at commit 6c89b402. Scripts and outputs, saved numbers only, no model forward: `~/commute-data/scratch/baseline-lead/e90-amend2-check/band_check.py`, `draws_check.py`, each with its `.out`. Formed before I read the analysis report; the last section compares the two.*

## Real findings

**1. The random map's row is decided by the largest-over-states rule, and against its own draws it sits on the edge.** `rules_amend2.py:132` pools each capability's bands, as filed (`predictions_amend2.json:15`). At block 1 on token identity, the 32-key copy map's second seed sets 0.4126. Each row against each reference:

| Row, block 1 | Share | Filed band | Own band | Largest of own 20 draws | Mean draw / share |
|---|---|---|---|---|---|
| copy-32, seed 0 | 0.499 | 0.413 | 0.398 | 0.400 | 0.70 |
| copy-32, seed 1 | 0.530 | 0.413 | 0.413 | 0.418 | 0.69 |
| random map, seed 0 | 0.370 | 0.413 | 0.338 | 0.339 | 0.85 |
| random map, seed 1 | 0.359 | 0.413 | 0.344 | 0.351 | 0.90 |

So the band that clears the random map is high because the copy map loses a lot under draws that are mostly its own basis (finding 3). Read seed by seed the branch would be (iii). That alternative is not firm either, because the random map's excess, 0.032 and 0.015, is inside the shortfall of the match:

- Scaling its own bands in proportion to the task-window shortfall, it falls within once the draws reach 0.97 and 0.93 of the basis's share. They reached 0.89.
- Extrapolating each state's own line on fit-residual share to the basis's share gives bands of 0.375 and 0.367, also within.
- The copy rows stay above their own bands under the same proportional scaling at a full match, by 0.051 and 0.065 on 32 keys and by 0.113 to 0.163 on 24 keys.

The filed outcome and a full-match own-band outcome therefore agree, while the own-band read at the match actually run does not. The random map should be described as near its reference, neither clearly under nor clearly over.

**2. The separation is of degree, and only the 32-key maps have a random twin.** An unfiled paired comparison supports the separation: on the same draw, copy minus random map. At block 1 the gap under the basis exceeds the gap under all 20 draws, on all four seed pairs:

| Pair | Gap under the basis | Draws' mean gap | Largest draw gap |
|---|---|---|---|
| seed 0, seed 0 | 0.129 | 0.035 | 0.099 |
| seed 1, seed 1 | 0.171 | 0.045 | 0.112 |
| seed 0, seed 1 | 0.140 | 0.027 | 0.097 |
| seed 1, seed 0 | 0.160 | 0.053 | 0.118 |

This is the strongest support I found, and it is post hoc. Three limits apply:

- At block 2 the order reverses (random map 0.17, copy-32 0.03-0.05), with no teeth, and the draws show the same reversal more strongly (-0.17 to -0.21 against -0.12 to -0.14 under the basis).
- The 24-key copy map has no random twin on its keys, so its "above" is against its own draws only. The random map also passes that test, although by much less: 0.19-0.24 against 0.015-0.03.
- Only one random table was read. `ident32b` and `rand32b` states and windows are saved (`~/commute-data/results-large/regating_table/`) and unread, so the copy-against-random claim has two seeds of one table pair and no replicate.

**3. The draws are mostly the basis, so "within" cannot carry the random map's "does not reach".** At k = 36, r = 31, any two rank-31 subspaces of the top 36 components share at least 26 dimensions (overlap floor 0.839). The basis lies about 0.99 inside that top 36 (0.853 / 0.861), and it holds 0.643 of the variance against 0.654 for the top 31 components. Copy-24 is the same at k = 27, r = 23, floor 0.826. Three things follow:

- The band is largely the basis's own removal, so it holds part of any loss specific to token identity. The filing says this (`predictions_amend2.json:67`). The size of it was not known at filing.
- The random map's mean draw / share, 0.85-0.90, is about r/k = 0.861. That is what a loss carried wholly by the basis would read, and also what a loss carried by the top components regardless of content would read. The band cannot tell these apart.
- The near-degenerate test does not bite. The split-half bar is 0.9987 against a pairwise overlap of 0.861 (`rules_amend2.py:100`), and with bars this high only k = r or nearly so could fail it. `overlap_with_basis` against `chance_overlap` also says little here, since the floor already gives most of the value.

The verdict string "does not reach the states" (`rules.py`, `NO_REACH`) therefore reads wrongly on this row, by finding 5's argument: the random map's answer must reach its key through this residual. The filing's "not shown specific, never shown not specific" is the right wording.

**4. The block-1 control is the first thing to fail under a fuller match, and branch (ii) depends on it.** W₀'s drop tracks the draw's share of the fit residual strongly across the 20 draws (slope 13.6, standard error 2.3, correlation 0.81 on identity; 15.1, 3.8, 0.68 on copy-24), unlike the states' shares. The draws hold 0.92 of the basis's variance.

| Control, block 1 | Basis drop | Filed band | Draws' line at the basis's share (minus 2 s.e.) | Band scaled in proportion to a full match |
|---|---|---|---|---|
| token identity | 2.863 | 2.580 | 3.09 (2.86) | 2.80, teeth kept |
| copy-24's keys | 2.495 | 2.388 | 2.91 (2.51) | 2.60, teeth lost |

So on W₀'s own read, the basis is no more damaging than random directions of equal variance within this reach. The line extrapolates about 5 standard deviations past the draws, so this is an indication and not a measurement. If copy-24's teeth go, its status is "unreadable", `rules_amend2.py:153` fails and the branch reads "other". On a plain redraw of 20, copy-24's teeth held in 98.9% of normal simulations, so only the size of the match threatens them. This is also the control that Amendment 1 substituted after the first one failed.

**5. "Reuses token identity" is not shown to mean more than that the answer needs the input token after block 1.**

- Task windows hold the key only at `half`, with the marker before it (`src/commute/kv.py:74-75`). On the saved held windows the key appears earlier in 0 of 97 (32 keys) and 0 of 75 (24 keys). So every later block's knowledge of the key at that position passes through the residual being edited. This is certain from the data, where the filing and report say "may".
- A linear reader of which key is present has to read the directions where the keys' means differ, which is what the basis is, as far as the natural-window basis also describes task windows (its share there is 0.63-0.66). A new circuit in block 2 would then lose under this removal as well.
- The spared write does not control for this. It reads its key inside block 1 (`src/commute/perturb.py:20`), upstream of the removal at block 1's output (`src/commute/readout.py:37`, `experiments/new_vs_reuse/apparatus.py` `removal_edit`), so it is spared by placement. Its near-zero share (-0.043, 0.007) argues against the answer direction as the cause of the copy map's loss, which is useful, but it says nothing about the reading side.
- The random map needs the key as much as the copy map does, has no g on labels (values share nothing with keys), and still loses more than all 20 of its draws.
- The share drops sharply one block on: copy-32 0.50-0.53 to 0.03-0.05, copy-24 0.66-0.68 to -0.03 to 0.07, random map 0.36-0.37 to 0.17, while the basis still holds 0.50-0.53 of task-window variance. This fits the gain being finished by block 2's output on copy maps, read from block 1's residual. It does not show that an old mechanism does it. I have not established the cause and do not assert one.
- For a copy map g is the identity, so g∘f_C holds on labels for any network that does the task. What the removal adds beyond that is finding 2's extra 0.13-0.17 over a map that equally needs the token. Its cause is unmeasured.

**Smallest check that would settle this:** the same hand write placed in block 2 (`h.2.mlp.down_proj`) for the `ident32` and `rand32` tables, read under the block-1 removal and the same draws. It is new by construction and reads its key downstream of the removal. If it reads above the band like the copy states, "reuses C" at block 1 means "needs the token" here. If it is spared, the copy states' loss means something more. It is one `debug` job, $0. E90 has used both amendments, so it needs a new E-code (`.claude/CLAUDE.md`, "Two amendments per E-code"), or the claim stands with this caveat.

**6. Task windows decide no row, and the bands they imply stay under the copy rows on any fair estimate.** The match is printed only (`amend2.py:90`, `:119`), and no status or verdict reads it. Copy-32 is at 0.888 of the basis (6 of 20 draws reach 0.9), copy-24 at 0.854 (0 of 20). The filing requires the "above" on these rows to be reported as not cleared of size (`predictions_amend2.json:34`). The filed band under a match on those windows, from saved numbers:

| Filed band, block 1 | 0.9 of the basis, in proportion | 1.0 of the basis, in proportion | 1.0, own line | Smallest copy share |
|---|---|---|---|---|
| identity, 0.413 | 0.418 | 0.465 | 0.409 (slope about 0, s.e. 0.7) | 0.499 |
| copy-24, 0.472 | 0.498 | 0.553 | 0.674 (slope 2.1, s.e. 1.4) | 0.661 |

Only the last copy-24 cell reaches a copy row. It takes a slope at 1.5 standard errors and extrapolates 13 standard deviations past the draws' spread on 75 windows, so I give it little weight. A larger band only moves the random map further within. So size on the task windows does not plausibly undo the separation, though a band this close to the basis is mostly the basis (finding 3): at a true full match the draws approach the basis itself, and the test loses its room.

## Minor findings

- **The match is not "just over 0.9".** Realised ratios on the fit residuals are 0.90 to 0.94 (`predictions_amend2.json:70` says just over; `amend2.json` `.matched.*.draw_shares`). Each k has its own seeded draws (`rules_amend2.py:51`), so realised means need not fall with k, and the largest passing k carries noise. This is small, and I could not check that a larger k was rightly rejected (see Not checked).
- **`band_of` takes absolute values** (`src/commute/removal.py:70`), so the pooled bands at block 2 are set by writes whose draws are all negative: 0.199 on identity (`ident32_s0_write`, largest draw -0.113) and 0.381 on copy-24 (`copy24_s0_write`, largest -0.240). No filed verdict changes, as those blocks have no teeth, but the `beside` column's "does not reach" at block 2 is read against them.
- **In nats the 32-key gap is smaller.** `A_nats` reads 0.26, 0.29 on the copy map and 0.22, 0.19 on the random map, gaps of 0.04 to 0.10 against 0.13 to 0.17 in the scored log-odds. The filed score is log-odds and draws in nats were not saved, so this has no reference. After the removal the copy-32 state keeps about 0.7 of its nats gain.
- **Two copy maps disagree in size.** Log-odds shares 0.50-0.53 and 0.66-0.68, nats 0.26-0.29 and 0.65-0.68, in one task format. "Copy" is not one number here, which is a reason to quote each map.

## What checked out

- **Draws:** rebuilt from the filed key `[90, tag, b, k, i]`, without the model, and each of the six mean pairwise overlaps matches the record to six digits. Orthonormal to 5e-8. `variance_share_random` equals the mean of `draw_shares`.
- **Band:** every state's band and every pooled band recomputed from the saved draws to 1e-12. The control bands match.
- **Reads:** all 32 state and write shares reproduce stage 2's within 1.2e-7. The net is fp32 (`src/commute/steering.py:42`), so there is no rounding of the basis at the removal.
- **Verdicts and branch:** statuses re-derived by hand from the shares, and the block-1 branch reads (ii) and block 2 "other", as filed. The do-nothing state reads "no gain" or "no teeth", so it cannot pass. `tests/test_new_vs_reuse_amend2.py`: 24 passed.
- **Redraws, block 1:**

| Resampling of 20 draws, 20,000 sets | Identity band (95% interval) | Copy-24 band (95% interval) | Branch (ii) | Branch (ii) with both teeth |
|---|---|---|---|---|
| Bootstrap, same draw indices across rows | 0.390-0.418 | 0.445-0.493 | 1.000 | 1.000 |
| Normal fit, jointly over rows and control | 0.388-0.430 | 0.448-0.521 | 1.000 | 0.989 |

The bootstrap cannot pass the largest draw, so the normal fit is the fairer interval. Margins to its ends are 0.069 (copy-32 above), 0.017 (random map under) and 0.140 (copy-24 above).

## Against the analysis report

Read after the above. Its numbers agree with mine, and its "conditions on that reversal" already cover findings 1, 3 and part of 5. Four differences:

- **"Reverses" is stronger than this supports.** Line 13 and line 128 should say the filed rule now separates by degree at block 1 on one table pair. The evidence is that the random map is near its own reference and its row depends on pooling.
- **"Read seed by seed ... the branch would be (iii)"** should add that the excess is within the match's shortfall (finding 1), or it reads as a firmer alternative than it is.
- **"How large a task-window match's band would be was not measured"** can carry finding 6's estimates, marked as estimates from saved numbers.
- **"May delete the input"** can be firmer on the reading route (finding 5). The two-table limit and the control's dependence on the match (findings 2, 4) are not in it.

## Not checked

- Anything per window: no per-window shares are saved, so no resampling of the 97 and 75 windows, and the 0.9 call on the task windows has its own unmeasured noise.
- Whether k is the largest that passes: the spectrum was never saved, and only 20 draws at the chosen k constrain it.
- The band at a real fuller match (k = r + 1 to r + 4), the plain top-31 components as a removal, and the hand write in block 2. All need the GPU, and all estimates above are extrapolations.
- Uncentred effects: the removal acts on the residual with its mean (`src/commute/removal.py:18`, `remove`), while the match is on centred variance. The draws' overlap with the mean residual is not saved.
- Tables `ident32b`, `rand32b`, and the copying head row, which Amendment 2 does not read.
- The basis's rebuild for copy-24 and the fine-tuned states themselves, beyond their reproduced numbers.
