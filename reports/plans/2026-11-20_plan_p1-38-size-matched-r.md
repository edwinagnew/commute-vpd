# Plan: the reuse score on E187's size-matched states (E232, cell P1.38)

*2026-11-20, definition-lead (writer agent). Card `reports/plans/2026-11-20_plan_p1-38-card.md`; filed rules
`experiments/kstar_size_matched/results/predictions.json`; code `experiments/kstar_size_matched/` (`run.py`, `score.py`,
`plot.py`).*

E221 found that the reuse score's reuse-minus-new difference on trained pairs ranks loosely with update size. This run
reads the same score on the same nine pairs after shrinking each pair's larger update to its partner's size, which E205
already did for the held-model reading. If the differences keep their sign at matched size, they are not a size reading
on these pairs; if about half flip, as a simple scaling law predicts, they are.

## Design check (`/design`)

- **Goal served.** `GOALS.md` part 1, a working definition of reused against new, and its label column for §5: the
  user's cell P1.38 (task file 2026-11-20, item 2). It fills a named open cell, so no sign-off.
- **Tried before.** E221 (P1.36) read R at these three levels on the same states at saved size: Spearman with the
  smaller-update gap 0.54, 0.69, 0.45, under its 0.7 bar. E205 read the held-model S on these scaled states (signs kept
  9 of 9). R on scaled states has not been read; E221's report lists it first under "What next".
- **Patch or step.** A step: it separates "R's gap is a size reading" from "it is not" on the pairs the paper uses.
- **Decisions.** None departed from: E221's measurement and E187's c, both unchanged.
- **Definition.** Read unchanged (E221's K\*, s_u, R); no rule is added to a verdict, and none is given (descriptive card).
- **What would change what we do.** At least 8 of 9 signs kept: the coarse-level R column can be reported as not set by
  size on these pairs. Signs flipping where the c² law says they flip: the column is reported as a size reading.
- **Control first.** Stage 0, alone on every shard: the control state (`e54/cipher_own_s0`, smallest c, 0.522) times 0
  must read R = 1 exactly, and the same state times 1 must repeat E221's item file. The first fails if the scaling path
  changes the model when it should not; the second if this code is not E221's measurement.
- **Saved for audit.** Every item's per-level JSON (both grades, every order's K\*, s per variable), as E221.

## What is run

- **Stage 0** as above, then **40 items**: the larger side of each of E187's nine pairs, every saved seed, times the
  pair's c (`E205.load_pairs().larger`, `pairs.item(state, pair)`; 3+4+4+5+2+8+8+3+3 states in `SM.BUILT` order). The
  smaller side keeps E221's saved reading of the same state; the c = 1 control is what licenses that.
- Each item: F = M + c dW (E187's `times`), D = the state's 512 read windows cut as E221 cut them, last position;
  `E221.read_item` unchanged at head_mlp (3 orders, polish cap 2000), mlp_attn and block (exhaustive), alpha 0.9.
- Load checks per item: KL(F ‖ M) and update size equal E205's saved values for the same scaled state.

## Expected values

| number | do-nothing (c = 0) | c = 1 control | scaled states |
|---|---|---|---|
| R, both grades | 1 within 1e-6, f = 0 | E221's, within 1e-5, same K\* (block R 0.97008) | 0.95 to 1, above the state's saved-size R |
| KL(F ‖ M), size | 0, 0 | 6.10054, 0.0095611 | E205's (e.g. 1.60378, 0.0049878 for the control state at c) |
| matched size gap | | | within 1e-6 of 0, relative (E205 files: at most 3.0e-8) |

**Rivals at blocks** (from E221's per-item R): rival a keeps 8 or 9 signs. Rival b, each scaled state's 1 − R being
c² times its saved-size 1 − R (the grade's 1 − cos² is quadratic in a small write change), gives matched gaps of
−0.0036 (e62), −0.0067 (e46), −0.0046 (e53), +0.0032 (e54_matched), −0.0066 (e54_own), +0.0045 (open_pair_c3), +0.0010
(open_pair_c3_copy), +0.0047 (e58_rank1_96), −0.0019 (e58_copy_prior_cipher): 5 of 9 kept. The seeds' se of a pair's
gap is 0.0003 to 0.003.

## The filed rule (the user's words, "at least 8 of 9 signs kept at blocks")

At blocks, read grade, per pair: the saved-size gap is E221's (`summary.json .levels.block.read.gap_R`: mean over the
reuse side's items minus mean over the new side's, each item the mean over its orders), the matched gap the same with
the larger side's items scaled. A pair keeps its sign iff both gaps have the same non-zero sign. Holds iff 8 or more of
9; "not read" if any pair lacks an item. E53's saved gap is negative (−0.0018, se 0.0026), so it keeps its sign by
staying negative; the count with reuse higher at matched size is reported beside it.

**Rule on edge cases** (run with `score.py rule()` on 2026-11-20, scratch `edge_cases.py`): all gaps 0 (do-nothing) →
0 kept, does not hold; matched = saved → 9 kept, holds (so the rule cannot tell "R ignores size" from "nothing was
scaled"; the per-item sizes and R shifts are read beside it); the c² law's pattern → 5, does not hold; one zero gap on
either side → not kept; one missing item → "not read". Partial correlation: n under 5, a constant or perfectly ordering
size gap → none.

**Reported only, from saved files ($0, `score.py`)**: per level the within-pair Spearman of the change in R gap against
minus the change in size gap; (i) over E221's 14 pairs, each earlier reading's Spearman with the R gap, plain and partial
with the size gap held fixed (already computable: at blocks S3 −0.32 → −0.22, neuron_top −0.15 → −0.37, E208's ρ none,
size orders its five pairs perfectly); (ii) the matched R gap against E205's matched S gaps; (iii) pooled over the 23
readings, Spearman with minus the size gap (saved-size part alone: 0.45 at blocks, E221's).

## Cost

From E221: 22 ms per evaluation on the rehearsal's cluster L40 (`run_rehearse.log`: 2,030 evaluations in 45 s),
11.8 ms on an H100 (`run_pod0.log`: 2,030 in 24 s). Per item 3 × 2,030 + 1,024 + 64 = 7,178 evaluations: 158 s on the
L40, 85 s on an H100. Stage 0 is 3,118 + 7,178 evaluations, 226 s (L40) or 122 s (H100). All 40 items: 1.8 h on one
L40, 59 min on one H100; four H100 shards at 10 items each about 16 min of reading plus set-up, about $5 in all. Peak GPU
memory 10.3 GB, host RSS 6.1 GB (E221). Not gated (under $60 and 6 h).

## Not run

- The smaller sides again (E221's readings, checked by the c = 1 control).
- E196's five pairs (no size-matched states exist).
- The neuron level (closed 2026-11-19) and VPD (P1.37).
