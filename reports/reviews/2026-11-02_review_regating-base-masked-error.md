# Review: does the unchanged model's masked error carry into the re-gating comparisons?

*2026-11-02. Read-only, saved numbers only, $0. Follows `2026-11-02_review_base-control-audit.md`, "Other lines". E62 Amendment 2 not used.*

We checked whether comparisons across maps on the re-gating line carry the unchanged decomposition's own error at the cue.
They do, because the readings subtract that error only as a constant, and it differs by target and looms largest where gains are small.
Copy against random maps holds, elicit against cipher keeps its order at half the gap, and lowercase against cipher cannot be read, so the "kind" claim loses that family until it is re-read.

**Reading the table.**
- **Netted as a constant.** r = (L01 − L00)/G with G = T1 − T0 (`local_faith.py:270`). The stand-in ratio is (L11 − L00)/(U1 − U0) (`heldout_keys.py:112`, `performed_anchor/run.py:393`).
- **The error.** e = L00 − U0: W₀'s masked log-odds of the target minus the same with every mask open, Δ out, same windows.
- **Exact identity.** r = (r from U0) − e/G, and the ratio likewise over U1 − U0. So the base term is e over the gain, on the gap's own scale.
- **Two ends, neither a corrected reading.** Filed: the offset stays while the mask changes. From U0: the mask's change closes it entirely. Saved cells cannot say which.
- **E48's F is target-blind.** On one set of windows it cannot differ: ident32 and rand32 both read 1.86, with e +1.93 and +1.27. E46, E54, E58 and E62 pairs share windows, so their carry is e. Only E53's pair does not, and F was never read there.
- **Verdicts.** *Survives*: base term under a quarter of the gap. *Moves*: order holds at both ends, but the term is a quarter or more, or a secondary number crosses its bar. *Unreadable*: the term is at least the gap, so the order flips. One anchor at equal gain cancels exactly (seeds, nulls in nats).
- **Sources.** Recomputed from saved cells, except † from a report. E60's family gaps, C3's positions, the eight-map correlation and E62's alternatives reproduce to two or three digits.

| Comparison, and what rests on it | Gap read: filed → from U0 | Base e across its anchors, nats; base term in gap units | Same scale? | Verdict |
|---|---|---|---|---|
| E46 copy v random maps; "reveal and create", E60 | r 0.51 → 0.55; ratio 0.54 → 0.58 | +1.90 v +1.62; −0.04, −0.03 | yes | **Survives** |
| E46 hand write's g over copy fine-tune's, line 0.5 | 0.16, 0.34† → 0.23, 0.44 (by hand) | +1.27, +1.97 v +1.93, +1.87 | yes, nats | **Survives**, thinner |
| E46 A3 write's ratio v own fine-tune; "reads what its fine-tune reads" | random maps +0.04, +0.16 → +0.19, +0.30 | same e, write's denominator half; −0.13 to −0.17 | yes | **Moves**: writes read above, not level; the ratio's weakness stands |
| E53 lowercase v cipher, step 300; filed separation | r 0.64 → −0.10; ratio 0.66 → −0.19 | −0.75 v +1.08; +0.74, +0.85; gains only 1.6, 3.8 | yes | **Unreadable** |
| Same, cipher at matched gain, step 100 | ratio 0.63 → −0.43 | as above; +1.07 | yes | **Unreadable** |
| E53 A1b broad cipher v E53 cipher | ratio 0.02 → 0.04 | +0.96 v +1.08; −0.02 | yes | **Survives**; against lowercase, unreadable |
| E54 elicit v matched cipher | r 0.54 → 0.27; ratio 0.64 → 0.31 | −1.01 v +1.89; +0.27, +0.33 | yes | **Moves**: half; order holds on every seed |
| E56 r's lead over steering, five seeds | r 0.518 → 0.254; lead 5.22 → 2.56, bar 3 | as E54; steering has no mask | r side only | **Moves**: filed pass stands; under its bar at the U0 end |
| E58 elicit v cipher, 24 keys | r 0.54 → 0.28; ratio 0.62 → 0.31 | −1.23 v +1.82; +0.25, +0.31 | yes | **Moves**; cipher's ratio sits on the 0.5 cut from U0 |
| E58 A3 flat over key count; drift guard | d_r up to 0.048 → 0.051, tolerance 0.05; drift 0.14† → 0.31 | near level; gain 5.4 v 12.9 | yes | "Untested" **stands**, more so |
| E58 A4 three own-guess maps; "r does not rank by carry" | D_r 0.110, 0.128 → 0.274, 0.182; rank-1 and copy swap; rho(r, carry) 0.02, 0.42 → 0.73, 0.67 | −0.67, −0.18, +1.35; about 0.11 | yes | Filed fail stands; **order within the three unreadable** |
| E58 A4-5 shuffle; "mostly key-conditional" | position, cipher to elicit, 0.41 → 0.64; r 0.477 → 0.577, cut 0.5 | +1.44 v −0.67, +1.75; 0.16 of a 0.25 gap to elicit | yes | **Moves**: the sentence may turn over |
| E58 A6 copy v prior-matched cipher | r 0.39 → 0.47 | +1.35 v +0.05; −0.08 | yes | **Survives** |
| E58 A7 class map v twin; "r does not read carry" | D −0.034 → −0.032 | +2.46 v +2.20; −0.002 | yes | **Survives** |
| E58 eight maps; "one cut on prior sorts them" | rho(r, log prior) 0.95 → 0.79; band gap 0.25 → 0.09, or 0.26 without the shuffle | rho(e, log prior) −0.95; term spans 0.17 of r's 0.61 | yes | **Survives** as bands; about a quarter of the spread is base |
| E55 C3 elicit's position between poles, "old side" | r 0.853 → 0.583; ratio 0.768 → 0.493 | as E58; rank-1 −0.06 | yes | **Moves**: to about the middle |
| E60 per family, lowest reuse − highest new; E53 A2 "two families" | E46 0.47 → 0.51; E53 0.59 → −0.15; E54 0.52 → 0.25; pooled AUC 1.00 → 0.75 | as rows above | yes | **Moves**: one family of three unreadable |
| E62 copy v lookup, unscored | r 0.69 → 0.55 | −2.18 v −0.51 at prior ratio 1.32; +0.14 | yes | **Survives**; r over 1 is the base, as its report says |
| E57 Qwen, real v random change | no teeth† | one K0, 0.0696†, every row | KL | **Nothing carried**: same rows |
| Unfiled, no base in it: slope of L01 on T1 between saved steps, 128 keys | rank-1 0.80, elicit 0.64, shuffle 0.58, prior cipher 0.32, cipher 0.27; copy 0.40, saturated by step 10 | none | yes | Own-guess against cipher is **real beyond the offset**; shuffle sits near elicit, as at the U0 end |

## Not checked

- **Which end is right.** Per-window cells are absent. A $0 read regresses L01 − L00 on U0 − L00 within a map (`masked_lo_windows`, saved states of E53 and E54, one debug job, no training).
- **E53 without the base.** Only step-300 lowercase states exist (`dW_lowercase_s*_step300.pt`), so its slope needs the 107 s fine-tune rerun with step saves. Cipher reads 0.01 to 0.09 over steps 100 to 300.
- **CI-masked P at the cue.** `route_split` computes `p_cells`, but no result file of these experiments keeps them, so only E48's W₀ values exist. A `route_cells` re-read gives them. I found no claim here that reads the masked P level.
- **F on these windows.** K0 is absent for E53, E54, E58, E62 (`pass_reading.trigger_cells` at W₀, minutes). It matters only for E53's pair.
- **Other evidence.** E5, E6, E10 and E12's cells behind "the share is set by the task" (e/G on E46's tasks spans −0.14 to +0.06, itself a task property); E55's counting-grammar stages; E36's rows (no U cells); E59, E64; offsets per key, since every e is a mean over windows.
- **This review's own code** has no independent check beyond those reproductions: `~/commute-data/scratch/regating-lead/base-error-check/gaps.py`, output beside it.
