# E61 Job S: independent check of the size verdict (2026-11-02)

**Verdict: reproduces.** Job S read what its rule says, and `size_verdict.json` follows from the per-row data: line C on both mask versions. My own script, sharing no code with `size.py`, matches every size, share, gap, interval and the floor to all printed digits (largest difference 0). No apparatus error found. The Gaussian's steep size comes from the unmasked model, not the masks, and sets no gap.

| size, axis | gold's size | gold's share | deciding random, its two reads | its share read off | gap (95%) | nearest read / gold | how read |
|---|---|---|---|---|---|---|---|
| big, K00 − K0 | 1.1670 | 0.0769 | alive, 0.6246 and 1.1731 | 0.0385 | +0.0384 (0.0354, 0.0417) | 1.005 | between reads |
| big, KL from W0 | 0.5983 | 0.0769 | alive, 0.4811 and 0.8537 | 0.0324 | +0.0446 (0.0417, 0.0476) | 0.804 | between reads, none within 20% |
| small, K00 − K0 | 0.5696 | 0.0257 | alive, 0.6246 and 1.1731 | 0.0295 | −0.0038 (−0.0065, −0.0011) | 1.097 | below both: lower read |
| small, KL from W0 | 0.2565 | 0.0257 | alive, 0.4811 and 0.8537 | 0.0295 | −0.0038 (same) | 1.876 | below both: lower read |

Truly original masks; floor f = 0.01. The alive change is the higher random in all four, on both mask versions.

## Findings, ranked
1. **C rests on the small state alone, and that state won narrowly.** Both big gaps clear 3f; only the two small gaps fire C. `nearest` chose step 8 (|log ratio| 0.0447, to the alive change) over step 10 (0.0459, to the Gaussian) and step 11 (0.0524): `size_select.json`. Correct by rule, but a difference of 0.0012 picked the state that decides. Steps 10 and 11 have no reading.
2. **The two small gaps are one number.** Gold lies below both reads of both kinds on both axes, so each axis takes the same b = 1 reads: −0.003827 twice. The KL axis adds no second reading there, and its read-off sits 1.88 times from gold (Gaussian 2.05 times). This hides no change of line. By my arithmetic, not a filed rule, the K00 − K0 gap stays −0.0029 (line through the alive reads) or −0.0012 (through the origin), and the KL gap +0.0017 or +0.00995. All are at most f, though the last by only 0.00005.
3. **The Gaussian's steep size: expected in kind, not an apparatus error, size of the effect only partly explained.**
   - No mask carries b. K00 is KL(state's clean logits ‖ masked model), with g0 from W0 and the masked forward not reading the weights (`Lab.cells`, `true_chunk`, `masked_logits`). K0 is identical at every b (0.0655815805 in each 16-row search line; per-row difference 0 across states).
   - So K00 − K0 is exactly the unmasked KL(state ‖ W0) plus a small cross term. Gaussian, b 1 to 1.056: size rises 0.1633, reverse KL 0.1916, cross term −0.0283. Exponents in b: forward KL 2.41, size 3.22, reverse KL 3.69.
   - The alive change shows the same order, milder: forward 2.11, size 2.32, reverse 2.69. Its own 16-row local exponent reaches 2.85 at b 1.31-1.38, so "near b²" is its average, not its slope.
   - It is smooth and general: the Gaussian's 16-row local exponent climbs 3.22, 3.44, 3.55, 3.81, 4.00 with b. Every one of 128 rows exceeds 2.1, and the top 10 rows give 12% of the rise (8% would be even). So no bad row and no jump.
   - b² holds only for small changes, where the two KLs agree. Here reverse / forward is 1.65 to 1.77, far from 1. Forward KL does average b^1.95 from raw scale 1 to 20.2 (`C0.json`, 0.0015 to 0.532).
   - Unexplained: why the Gaussian is steeper than the alive change at like KL. No per-position data is saved, and reverse KL was never read at small b.
   - Bearing on the verdict: the reads are real, and only the search's b² step suffers. It oscillated through 6 evaluations (16-row `matched` false), then undershot, leaving the Gaussian's big read at 1.0203, 0.874 of gold, so its 0.0256 is an extrapolation. The verdict's `unmatched` is rightly empty (within 1.15). The filed "2-4 evaluations" was missed; it was not a stop. The alive change's share sets that gap, 0.013 higher, so no gap moves.
4. **K00 − K0 is mostly reverse KL, and the kinds sit differently on it.** Size minus reverse KL: gold −0.21 (step 100), −0.07 (step 8), −0.30 (`gold_s0`); randoms +0.05 to −0.04. Size / forward KL: gold 1.8-2.2, alive 1.3-1.4, Gaussian 1.6-1.7. That is why one state can be near a random on one axis and 1.9 times off on the other. At step 100 the alive big read is 1.005 of gold on size, 1.43 on forward KL, 0.87 on reverse KL (reported, no rule reads it).
5. **Gold's share and the randoms' are not the same shape in size.** Exponent of share against K00 − K0: gold 1.53 (step 8 to 100), alive 0.43, Gaussian 0.37. The filed proportional expectation (alive 0.067 at 1.44) did not occur: the alive change reads 0.0386 at 1.17. The word "size explains" in line C comes from the rule; the data show that only at the small state.
6. **Floor.** The two gold runs differ by 0.0129 on E57's masks, over 0.01, but their sizes are 1.237 apart, outside 1.15, so f stays 0.01, as filed and recorded. At f = 0.0129 the line is still C, though 3f = 0.0387 would put the big K00 − K0 gap (0.0384) under it.
7. **Minor.** `repro_gauss_big_file` and `repro_inspan_big_file` log `ok {}`: empty, since scaled changes have no reference K10 or K11. Only check 2's X ratio below covers them. Filed "2.1 times the training record's KL" read 2.5-2.8: Expected, not a stop. The trainer's 16-row KL (0.520, 0.235) is below Job S's (0.628, 0.289), ratio 1.21-1.23, and `gold_s0` shows 1.21 (0.689 against 0.831). This looks like row format, since the trainer's rows come through `job.rows`, not read here. It touches no gap. Step 1's size of exactly 0 is the warm-up's zero rate (`train_gold_1e-5.json`, lr 0, dw_norm 0).

## Checked and fine
- **True-mask pass (`true_chunk`).** g0 and g1 are computed with original columns, at W0 and at the state, and held on the host before `applied` moves anything. K00 is taken outside `applied`, K10 and K11 inside, all against the state's clean logits. The formula is (K00 − K10) / (K00 − K11), as filed. Base and true K00 agree row by row, difference exactly 0.
- **Rows and normalisation.** The same 128 rows of 256 tokens throughout (per-row K0 identical on 7 states, lists all 128 long). Every cell is a sum over positions divided by 128 × 256; forward and reverse KL likewise. All states scored: denominator 0.55-1.37 nats, K11 − K0 at most 7% of size.
- **Scaled randoms.** b comes from the search. The 128-row size in each state equals the search's to all digits (1.0202681279261767, 1.1731384172026074). Each has its own split file, no `resumed` flag. X_big / X_base = b at 4 of 196 sites per kind: ratio 1.0556176, 1.3119140, residual 6e-8. A reused split would read 1, an unscaled one too.
- **Other.** All logged checks in the five jobs read ok; no stop file. Filed controls reproduce (`gold_s0` file share 0.0917, Gaussian 0.0221, alive 0.0291; 16-row true 0.0890, 0.0223). Step 8's drop 1.8579 and length 0.8149 follow from `train_gold_1e-5.json` and `train_W.json`. Branch taken: 1.167 is over 0.99, so the final state is big.

## Not checked
No forward was re-run and the model was not loaded, so every cell is taken as logged. X at the other 192 sites of each scaled kind. Gold, `gold_s0` and base-scale X (`stored_error` only). The masks' content and Γ. `job.rows`. Whether the direct share means what it is taken to mean, which the filed `control_limits` also disclaims.

Scripts and output: `~/commute-data/scratch/qwen-decomp-lead/check-s/` (`recompute.py`, `x_linear.py`, `.out`).
