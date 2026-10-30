# Independent check of the re-gating synthesis and E58 Amendments 4-5 (2026-10-31)

*Fresh Opus agent that wrote neither; scope commits 48e0820, f3d41ef, 3577f48 and their PENDING lines. Recorded by the lead
for the next session to fix; nothing below is fixed yet. Verdict: needs fixes. Most numbers check out (list at the end).*

## Concrete errors
1. `reports/analyses/2026-10-30_analysis_regating-synthesis.md` l.92 and l.125: "cipher's r rose from 0.23 to 0.30 with the
   key count". The filed drift that made Amendment 3 "untested" is 0.14 (0.23 → 0.37 at step 10; amend3 `B.cipher_r_drift`
   0.138; E58 report l.525). The 0.30 mixes step and key count (96-key step 30 vs 24-key step 8-9); at the gate step the
   96-key value is 0.32-0.33.
2. Synthesis l.114, "within a row: same keys, read step and key count", is false for the first reads: at 96 keys the
   own-guess maps are read at step 10 but the cipher and shuffle at steps 15-16; at 24 keys step 10 sits against the cipher
   at steps 8-9. Only step 30 and E54/E56 (norm-matched) are same-row. Departures lists only the 24-key case.
3. Figure (`experiments/open_pair/plot_synthesis.py`): the legend shows "24 trained keys" as a hollow triangle but all
   24-key points are filled (l.157, all stage "first"); the ratio-panel band says "E54/E56" but is E54 only; "first read
   (filled)" does not say step 10 vs 15-16; the ratio panel lacks the 0.5 cut named at l.153.
4. l.73, "carry runs from 0.10 to 0.84 with no order in r": that range is 96 keys only (24-key elicit x ≈ −0.03); "no order" is
   too strong (elicit has the lowest x and lowest r at both steps; rho at step 30 is 0.42). "Not in carry order" (rank-1 vs
   copy) is supported.
5. E58 report "What we learned" (~l.741-750) and the matching notebook cell (md ~l.221-227) still carry superseded text
   ("stay level across three maps… so they do not measure carry"; "What stands is the level picture… 0.78 to 0.89";
   "positive control, on one seed"), contradicting Amendment 4. The opening answer is fine.
6. Amendment 5: "the other two sit within 0.01 of it" holds at step 30 only; at the gate step seed 2 reads 0.360 vs seed 0's 0.388.

## Overclaims (change the plain answer)
- **No non-VPD baseline.** "Own guess" is defined by W₀'s rank of the target, so W₀'s prior log-prob separates these maps
  by construction. The synthesis never shows r adds anything over that baseline; E58 P3 scored "steering_better". The answer
  should say r agrees with W₀'s prior, not that VPD detects elicitation, and the missing baseline goes in "Not run".
- **Missing cell.** No map has a target that is not W₀'s guess yet carries to unseen keys (every low-band map has x ≤ 0). So
  "does not separate a rule that carries from a lookup" is tested only inside the own-guess band, and a new mechanism that
  generalises was never tested. The plain answer should say so.
- l.11 "split the maps cleanly": the gap between bands is clean (≥ 0.24), but the shuffle sits only 0.015-0.03 under the 0.5 cut.

## Lesser
- The synthesis does not cite E46, E53 or E60, which the line's earlier claims rest on; decide whether they belong.
- The "for" column omits that copy reads "reuses" on 3 seeds at 96 keys.
- Fable's advice (`2026-10-30_advice_e58a4-synthesis-FABLE.md`) says "128 keys" (96 trained) and "the highest-r map has the
  lowest carry" (false); the synthesis correctly avoided both.

## Verified correct
Bands 0.72-0.92 / 0.29-0.49 at 96 keys, 0.76-0.85 / 0.22-0.24 at 24 keys, E56 0.76-0.77 / 0.21-0.27; ratio bands 0.96-1.21 /
0.41-0.66; gap 0.237; largest seed range 0.045; D_r 0.110 / 0.128 vs 0.071 / 0.090 (fail); ratio 0.179 / 0.196 vs 0.2 (pass);
rho 0.017 / 0.417; shuffle 0.475 / 0.485 / 0.471 (mean 0.477), gate step 15, G_spec 16.4-17.5; controls "below" at steps
16/16/15; seed 0 reproduced; E58 P1 pass, P2 fail, P4 pass; A1 C2 fail; leans as filed; scored vs unscored marked
correctly; A4's failure not refiled; granularity limit beside each claim; TRIED and PENDING match the data.
