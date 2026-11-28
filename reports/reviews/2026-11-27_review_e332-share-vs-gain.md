# Review: E332 (cell P2.91), Qwen3-1.7B re-gating share against answer gain

redteam-lead's reviewer, 2026-11-27. Inputs at commit 7caa976c: `experiments/share_unified/results/qwen_gain.jsonl`
(25 lines), `experiments/share_unified/qwen_gain.py`, `results/predictions_gain.json`, card
`~/commute-data/cards/E332.md`, log `results/run_gain_rehearse.log`; compared against `results/qwen_reread.jsonl` (E328)
and `results/qwen_random_vpd{0,1}.jsonl` (E330).

The run scaled each known and copy fine-tune's own update down until it moved the answers about as little as the
random updates did, and re-read the re-gating share there; every number regating-lead quoted matches the saved file,
the controls read exactly zero and the full-strength reads reproduce E328 exactly. The share does change with how far
the update moves the answers (on known it rises from about 0.58 to 0.73 when scaled down; on copy it falls slightly,
from about 0.62 to 0.58), and once the answer gain is matched the fine-tunes no longer read above the random updates
on either task. So the paper should not say the fine-tunes beat random updates on known or copy, and should state
that the lower random shares on the other tasks were also read at a much smaller answer gain than the fine-tunes'.

## Numbers check (jq on `qwen_gain.jsonl`)

| task, seed | full: gain, share | ~10: c, gain, share | ~1: c, gain, share |
|---|---|---|---|
| known 0 | 15.92, 0.597 | 0.139, 9.92, 0.654 | 0.00875, 1.012, 0.744 |
| known 1 | 16.02, 0.576 | 0.139, 10.08, 0.653 | 0.00937, 1.063, 0.723 |
| known 2 | 15.96, 0.583 | 0.139, 9.99, 0.656 | 0.00937, 1.063, 0.727 |
| copy 0 | 10.50, 0.637 | 0.187, 9.65, 0.646 | 0.00937, 1.073, 0.579 |
| copy 1 | 10.86, 0.611 | 0.187, 9.63, 0.648 | 0.0100, 1.084, 0.586 |
| copy 2 | 10.58, 0.622 | 0.285, 10.20, 0.635 | 0.00937, 1.065, 0.589 |

- regating-lead's ranges all hold: known 0.58-0.60 / 0.65-0.66 / 0.72-0.74; copy 0.61-0.64 / 0.63-0.65 / 0.58-0.59.
  One correction: the copy full gain is 10.5-10.9, not "~10.5".
- Controls: all six (zero and W0 on known, copy, known_repack1) read search gain, gain, numerator and KL exactly 0.
- Full reads: `diff_e328` is exactly 0 for all six states; `e328_share` and `e328_gain` equal E328's
  `p-9707d6ff/run2/<arm>_s<seed>_last` records in `qwen_reread.jsonl` to all printed digits.
- Bands: every scaled read has `status` ok and `in_band` true (gain-1 band [1.0028, 1.0973], gain-10 band
  [9.55, 10.45]); `search_gain` equals `gain` because `search_rows` is 128, all rows.
- Repeat: `known_d2_repack1` reads gain 1.054, numerator 1.276, share 1.211; E330's `known_d2` read gain 1.037,
  numerator 1.070, share 1.031. Same draw seed (228003), same `size` (46.015) and the same signed c (23.341): the
  update is the same tensor at the same scale. The repacked row set holds 1,081 whole answers against E330's 1,066.
- E330's VPD random draws at gain ~1, for the page: known 0.687, 0.515, 1.031; copy 0.719, 0.630, 0.666.

## Code check (`qwen_gain.py`)

- **Scaled update.** W0 + c·R, R = W_ft − W0 on the decomposition's own sites (`QS.own_sites(load(path))`), set by
  `lab.set_weights("FT", R, c)` inside E330's `Search.gain` and `full_read`. One sign, c in (0, 1].
- **Choosing c.** `PT.match(sc.gain, target, 1.0, gain_rel_tol)` searches c until the gain on the search rows sits in
  target × (1 ± 0.045); search rows are all 128, so the matched gain is the read gain.
- **Gain.** mean over whole answers of T1 − T0 in whole-answer log-odds (`share.cells_gain` on the same forwards;
  `Search.gain` uses `answer_target` against a T0 read once per row set).
- **Share.** `RQ.full_read` (E330's function) → `share.share` = mean(L01 − L00) / gain, floor gain ≥ 1 one-sided.
  Same function and the same rows (`RR.rows_and_spans`, E328's rows; answer counts 1,066 and 1,732 equal E330's) as
  E328 and E330, which the exact reproduction of E328 confirms.
- **Repeat.** `SR.gaussian(norms, shapes, draw_seed)` with norms from the same fine-tune state, then `RQ.one_draw` with
  E330's settings; only the rows differ (`SR.marked_rows(arm, repack_seed=1)`).
- No defect found. Minor: on copy the "~10" stage is barely below the full gain (9.6-10.2 against 10.5-10.9), so it is
  not a separate gain level there; the page should say so rather than present three gain levels on copy.

## Answers

**1. Is the scaled fine-tune a fair matched-gain reference for E330's randoms?**
It is a fair test of "does the share depend on gain, direction held fixed", and not a matched random reference. The
share is not scale-invariant by construction: the numerator is the masked model with W0's components under gates
read from W0 + c·R, and the denominator is the target model's log-odds change, both non-linear in c. As c → 0 the
ratio tends to a finite limit (a ratio of two directional derivatives along R), so the gain-1 read is close to the
first-order share of the fine-tune's direction, while the full read adds the non-linear part. The data confirm the
test has power: c = 0.009 gives gain 1 against 16 at c = 1 (a linear extrapolation would give ~115), and the known share
moves by 0.13-0.15. What it does not match: the randoms reached gain 1 at 12-23 times the fine-tune's norm (sizes
23-46 against the fine-tune's ~1.9; the scaled fine-tunes sit near 0.018, ~2,500 times smaller than draw 2), and with
10-50 times the fine-tunes' KL (random known 0.013-0.066 per position; scaled known fine-tunes 0.0013-0.0014). Gain is
matched, norm and side effects are not. So E332 answers "is the full-gain fine-tune versus gain-1 random comparison
confounded by gain?" (yes, on known by +0.14, on copy by −0.04), and lets one put the fine-tune and the randoms side by
side at the same answer gain; it does not by itself say how a fine-tune compares with a random of matched gain on
the other four tasks.

**2. 1.03 against 1.21 on the repeat draw: noise or row packing?**
Row packing, not read noise. The update and its scale are identical (same draw seed, size 46.015, c 23.341), so only
the rows changed. Same rows and same update reproduce to the bit on this path: all six full reads equal E328's exactly
(a separate pod, a separate run), so the same-rows read floor is measured at zero in this data, at no further cost. The
repack changes which whole answers are read and how often each item counts (1,081 answers against 1,066; 600 items
each about twice), and the share moved by 0.18 for gain +0.017 and numerator +0.21. What is not known is whether
this row sensitivity is particular to a large random update (c 23, KL 0.06, share above 1) or also holds for the
fine-tunes. No $0 check settles that: per-answer cells are not saved. The cheapest check is a GPU read: the three
known fine-tunes at gain 1 on `known_repack1` (three reads of ~100 s each, ~6 min on one H100, with the existing
script's repack row set). Until then, a single draw's share should be quoted with a row-set sensitivity of about 0.2
(one draw, one repack), which is larger than the card's spread proxy of 0.05 and its smallest gap of 0.1.

**3. What may the paper say about P2.90's random comparison on known and copy?**
Bounded wording:
> "At their own answer gain, the known fine-tunes (three seeds, gain about 16) read a re-gating share of 0.58-0.60 and
> the copy fine-tunes (three seeds, gain 10.5-10.9) 0.61-0.64. Scaled down to an answer gain of about 1, the gain of
> the random updates, the known fine-tunes read 0.72-0.74, against 0.51, 0.69 and 1.03 for three random draws, and the
> copy fine-tunes read 0.58-0.59, against 0.63-0.72 for three draws. On these two tasks the fine-tunes do not read
> above random updates of the same answer gain."
And, for the other tasks: the random draws on cipher (0.35-0.37), invented (0.37-0.40) and gold (0.48-0.55) were read at
gain about 1 against fine-tunes at gains 16-19, 29-30 and ~150-160; E332 shows matching the gain moves a fine-tune's
share by up to 0.15 on these tasks it read, so those gaps are stated with both gains and not called gain-matched.

## MUST / SHOULD

- **MUST** the P2.91 page and any paper sentence give, for every fine-tune/random comparison, the task, the seeds or
  draws (three each here) and both answer gains; no "randoms lower" on cipher, invented or gold without saying they
  were read at gain ~1 against the fine-tunes' 16-160.
- **MUST** drop any wording that the fine-tunes read above matched random updates on known or copy: at matched gain
  ~1, copy fine-tunes read below all three draws and known fine-tunes sit within the draws' range.
- **MUST** quote the repeat as a row-set sensitivity of one draw (0.18 on one repack), not as read noise; the same-rows
  floor in this data is zero (full reads equal E328 exactly).
- **SHOULD** say on the page that the gain-1 read is near the share's small-update limit (a property of the update's
  direction), and that at matched gain the randoms needed 12-23 times the fine-tune's norm and 10-50 times its KL.
- **SHOULD** note that the share is not bounded by 1 (draw 2 reads 1.03 and 1.21), so it is not a fraction for large
  random updates.
- **SHOULD** not present copy's "~10" as a separate level (9.6-10.2 against full 10.5-10.9).
- **SHOULD** keep the page descriptive, as the card says: report values per gain per task; do not declare rival a or b
  (known's shift of 0.13-0.15 is past the card's 0.1 and copy's 0.03-0.06 is not, which a reader might read as a
  verdict).
- **SHOULD** (if a run is wanted) the three known fine-tunes at gain 1 on `known_repack1`, ~6 min on one H100, to
  tell whether the 0.18 row sensitivity is the random draw's or general.

## Not run

- No Python or GPU reads; all checks by jq and reading code.
- The share's small-c limit was argued from the formula, not computed (no derivative read).
- Row-set sensitivity of the fine-tunes' shares (no per-answer cells saved; needs a GPU read).
- Gain-matched reads on cipher, invented, gold and gold_shuf (not in E332's scope).
- The neuron-basis decomposition (E332 reads VPD p-9707d6ff only).
- `check.py` of `/writeup` (not run: Python is refused on the dev node outside `cpu-run`, and this is a review, not a write-up).

## Re-check of the P2.91 page (2026-11-27 20:15)

Page `reports/analyses/2026-11-27_analysis_p291-share-vs-gain.md` (regating-lead, 20:06). All three MUSTs are met.
Every fine-tune row (c, gain, share, KL) matches `qwen_gain.jsonl` and `qwen_share_vs_gain.csv`. Every random row
(size ratio, gain, KL, share) matches `qwen_random_vpd0/1.jsonl`, and so do the repack row and the band claims. The
KL ratios "9 to 53" and "8 to 33" recompute as 9.4-52.8 and 7.7-32.6.
- MUST 1 met: both tables give task, seed or draw and gain; cipher/invented/gold are stated with both gains.
- MUST 2 met: no "above matched randoms" on known or copy.
- MUST 3 met: the repeat is given as a row-packing sensitivity of one draw (0.18), and the same-rows read floor is
  exactly 0.
- SHOULD (opening, sentence 2): "no longer read above the random updates" implies they did at full gain. They did
  not (known full 0.58-0.60 vs draws 0.51-1.03; copy full 0.61-0.64 vs 0.63-0.72). Say "do not read above".
OK to fill once run 2 lands, or now with run 2 listed as Not run.

## Paper check: sec4:89 (writeup-lead local d602887) and the App H draft (2026-11-27 ~20:35)

Sources: P2.91 page, P2.90 E330 table, `qwen_reread.jsonl`, `qwen_gain.jsonl`, `qwen_random_vpd*.jsonl`.
Traced and correct: VPD fine-tune gains 10.5-160.3 ("10 to 160"); every VPD fine-tune share 0.561-0.737 ("0.56 to 0.74");
known 0.58-0.60 at full gain to 0.72-0.74 at gain one; copy 0.61-0.64 to 0.58-0.59; randoms at gain one, known
0.515-1.031 and copy 0.630-0.719; on cipher, invented and both maths tasks randoms 0.351-0.554 against fine-tunes
0.561-0.737 at full gains 16.2-160.3; repack 1.031 to 1.211; KL 7.7-52.8 times the gain-one fine-tunes; the
neuron-control numbers in the draft's last bullet.
- MUST (App H draft, last sentence): "the random draws are 9 to 23 times its norm", where "its" is the gain-one
  fine-tune, is wrong. The draws are 8.8-23.3 times the full fine-tune's norm. The gain-one fine-tune is c = 0.009-0.010
  of that, so the draws are about 900 to 2,500 times the gain-one fine-tune's norm. Either say "9 to 23 times the full
  fine-tune's norm" or "about 900 to 2,500 times the norm of the scaled one".
- SHOULD (sec4:89): "inside the random range on one task and below it on the other" should name the tasks (known
  facts, copy), as the App H draft does.
- SHOULD (sec4:89): the known range 0.51-1.03 is three draws whose top one is a single draw above 1, and it moved to 1.21
  on repacked rows. One clause, or a pointer to App H, would keep a reader from taking 1.03 as a measured edge.
- OK: "On the two tasks read at matched gain, the share does not separate a fine-tune from a random change on Qwen"
  is scoped to the tasks read and matches the page.
- Note: E333 (run 2, repacked fine-tunes) is pending, so re-check both texts when its rows land.

## Re-check with run 2 in (E333, repacked rows; bc09bb4d, 1a15389c; 2026-11-27 ~21:00)

Every run-2 row on the page matches `qwen_repack.jsonl`, and so does the CSV (`qwen_share_vs_gain.csv`, 30 rows plus the
header, packing column, copy seed 2's repacked gain-one share empty). The deltas, the steps per packing, the answer
counts 1,081 and 1,724, the zero and W0 controls at exactly 0, and the draw-2 repeat (1.2108150967958595, equal to
E332's) all match. The opening's and the Answer's numbers match too.
- MUSTs (a)-(c) still hold. (a) Every comparison gives task, seed or draw, packing and gain. (b) Nothing says "above
  matched randoms"; "never above" is true on both packings. (c) The 0.18 is given as one draw's packing sensitivity,
  beside at most 0.049 for the fine-tunes.
- "Copy shows no clear change with gain": supported. The steps run -0.059 to +0.020 across packings, against a
  repack move of up to 0.049 at gain one.
- SHOULD 1: at fixed c, copy's repacked gain-one reads land at gains 1.0025 and 1.015, against 1.073 and 1.084
  originally. So the +0.044 and +0.049 are packing plus a gain shift of about 0.07 that the packing caused, not packing
  at fixed gain. Seed 0, at 1.0025, also sits outside the band, as the page says. The sentence "Packing moves the
  fine-tunes' shares by a few hundredths" could name this in a clause.
- SHOULD 2: on repacked copy, "at their lower edge ... never above" rests on 0.6299 against draw 1's 0.6304, which is
  0.0005 apart and on different row sets. The draws' own repack sensitivity on copy is unmeasured (0.18 on known draw 2).
  "Level with the lowest draw" would say it more exactly. That is still no separation, so the conclusion stands.
Verdict: OK to fill.

## Paper re-check with E333: sec4:89 (writeup-lead local 29ca946) and the App H draft (2026-11-27 ~21:25)

No MUST. Every number traces to page fac49fcf, `qwen_repack.jsonl` and `qwen_gain.jsonl`: known repacked 0.74-0.76 at
gain one vs 0.60-0.62 full; copy 0.58-0.59 original, 0.628/0.630 repacked (seeds 0, 1); seed 2 has no reading;
the ≤0.05 and 0.18 repack moves; "below or level with the lowest draw" (0.6299 vs 0.6304). "Copy shows no clear
change" is scoped to copy, and the closing sentence to the two tasks read.
- NIT (App H): copy seed 0's repacked gain-one read is at 1.0025, just under the band 1.003-1.097 that the sentence before
  gives for the random changes. Seed 2's "reaches no gain of one" really means its gain, 0.995, is under the share's floor
  of 1. Either can stay as it is in the body; a footnote is optional.
- NIT (both): copy's ≤0.05 repack move happens alongside a gain shift of ~0.07 at fixed c (1.07-1.08 to 1.00-1.02).
  Optional to say.
