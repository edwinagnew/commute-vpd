# E135: does steering separate copy from lookup as widely as the share, once it is given a fair chance? (cell P2.5, due 11-12)

*compare-lead, 2026-11-09. Card: `2026-11-09_plan_p2-5-card.md` (passes at 2.92 standard errors, n=3). Replaces the old P2.5 design, whose
number E134 was released. Saved states only, free queue, one run. No verdict under any wording of reuse: the labels are "copy" and "lookup"
as built, and the copy side is unscored in E62.*

**What runs.** On the six saved updates of the new-marker pair (`induction_cue/{copy,lookup}_s{0,1,2}.pt`, step 30), read on
`induction_cue/windows.pt`'s 200 read windows, the steering share with the changes the user asked for (task file, items 1-4), beside the
share on the same scored windows. Even-indexed read windows are the fit half, odd-indexed the score half (`steering.Task.part`). Item 5, 6
and 7 do not apply: this cell does not read the neuron version.

**Checked against the code (items 1-4, 2026-11-09).**
1. Per-key vectors: the key is the token before the trigger (position 19, the 32 keys); all 100 scored windows have a fit-half window of
   their key, 1 to 7 per key (median 3). Copy's answer varies within a key (178 distinct targets), so a per-key vector cannot write it:
   expected to read like the pooled one on copy and to help lookup, which is the point of adding it.
2. Strength: `add_at` scaled by a strength in {0, 0.5, 1, 2, 4, 8}. Each state and vector kind gets its own (layer, strength), chosen by the
   largest fit-half mean log-odds rise, ties to the lower layer then the lower strength. Off-target KL at that setting:
   `revival_kv.kl_rows` over the score windows' positions other than the answer, W0 against steered, beside the state's own
   `off_answer_kl`. Only positions after the answer can move, since the edit is at the answer position.
3. Random vectors, at the chosen layer and strength: 8 draws per state of a Gaussian direction, scaled to the norm of the vector replaced
   (per key for the per-key kind). Cross-application: each state's vector, at its own chosen setting, read on the other side's targets of the
   same-numbered seed over the other state's gain. Cosines, pooled vectors, per layer: copy against lookup, and across seeds within a side.
4. Each steering number is written in log-odds (`steering.logodds_at`, the share's unit) and in log P (`logp_at`, the old columns).

**The share.** `local_faith.route_split` (`r`, the do-nothing form, as E62 read it) on the 100 score windows, and on all 200 as the
reproduction. Base cells at W0 from `route_cells`, per map, on the same windows.

**Stages, cheapest first, each written as it finishes** (`STAGE n DONE` after each file).
- Stage 0, control, alone: (a) a zero update through the full steering path reads rise exactly 0 for both vector kinds (stop if any |rise| > 1e-9);
  (b) strength 0 on `copy_s0` reads exactly 0 (same stop); (c) the share of each of the six states on all 200 windows reproduces
  `experiments/induction_cue/results/run.json` → `.states.<state>.r` within 0.01, a tenth of the card's smallest gap (stop on a miss).
- Stage 1: steering, the share on the score windows, controls and cross-application, per state.

**Filed rule** (`predictions.json` before the run, edge-checked below). Let `gap_share`, `gap_pooled`, `gap_perkey` be the mean over the three copy
seeds minus the mean over the three lookup seeds, each in log-odds on the score windows; steering's gap = the wider of `gap_pooled` and
`gap_perkey`. `edge = gap_share - steering's gap`.
- `edge >= 0.1`: rival A, the share keeps an edge on this pair.
- `edge < 0.1` (including below 0): rival B, no edge.
- Unreadable: any state whose log-odds gain on the score windows is below `MIN_GAIN` (1 log-odds), or any `nan` among the six.
- Inert steering: if steering's copy and lookup shares are all below 0.05, in both kinds, the verdict is "no call: steering inert", not rival A. A
  vector that writes nothing would otherwise make the share look as if it has an edge. Filed because the do-nothing case (zero vector, gap 0)
  passes `edge >= 0.1` on its own.

Edge cases run through: ties at exactly 0.1 go to rival A; steering's gap above the share's gives a negative edge, rival B; one seed's nan is
unreadable, not dropped; an empty fit-half key cannot occur (checked above, all 32 keys have a fit window, and the job stops if not).

**Expected values** (by analogy, not scored). Steering share in log-odds, pooled: copy 0.7 to 0.9, lookup 0.1 to 0.3 (E58's copy and lookup, `reports/analyses/2026-11-09_analysis_what-regating-does_appendix.md`).
Per-key lookup above pooled lookup. `gap_share` about 0.69 on the 200 windows (1.09 to 1.10 against 0.38 to 0.42); on the 100 score windows it may
move by a few hundredths. The share of a zero update is undefined; do-nothing reads the share by construction (E125's check). Random vectors' mean share
within 0.1 of 0 on each side (a miss is reported and marks the reading, it does not stop the job). Off-target KL 0 at strength 0.

**What each metric is blind to.** The steering share is blind to whether the vector does it through the fine-tune's own route; the gap is blind to levels
(both sides could move together). The positive control is the zero update and strength 0 at stage 0, and the reproduction of `r`.
Noise floor: seed sd of each side, reported; the card's spread (0.042) is from the log P share on other rows, since steering has never been read at
this marker.

**Figure.** One: `commute.figures.dots`, four method rows (share, steering pooled, steering per-key, random) by copy and lookup, three seeds, as in the card's sketch, in log-odds.
Log P columns stay in the JSON. One paragraph, with `/writeup`.

**Cost.** One L40, `debug` QoS (`--comment=E135`), expected well under 30 minutes: 6 states, 4 layers, 2 kinds, 6 strengths on 100 to 200 windows. Free.

**Design check** (`/design`, a line each). Goal: part 2's comparison on the one causally labelled pair. Nearest row: E62 and its amendments; steering never
read there (`method_table/results/table.json`, row `e62_copy_lookup`, `cells.steering` "not run"). Step, not patch: it is the rival comparison the plan asks
for. Result that changes what we do: rival B removes the share's only standing edge on a causal row. Decisions relied on: `decisions.md` row 112 (no verdict),
E62's `r_scored` false.

## Not run

Everything above. Item 7's TopK neuron reading belongs to P2.4 only, since this cell does not read the neuron version.
