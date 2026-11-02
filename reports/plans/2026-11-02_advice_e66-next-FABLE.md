# Advice on E66 after Amendment 2: what the patch ratio read on fourteen fine-tunes, and whether the line goes on

Fable, 2026-11-02. Advice only; the user decides. Read: `experiments/interchange_67m/results/amend2.json` (job 49069),
`amend2.py`, `src/commute/ctxpatch.py`, the plan's Amendment 2, my stop advice, `experiments/method_table/results/table.json`,
E62's rows in `TRIED.md`. No model was run. The write-up is another agent's; this names only what it should not miss.

## In plain words

The patch ratio T was read on every saved fine-tune, with the hand-built write and the sharpening fine-tune beside it as references
and no label given. On the one pair with a causal label (E62's copy and lookup) T orders the two as E62 does, and the order survives
swapping their sizes. On the copy-versus-random key maps it orders them the other way round: the copy maps read far below every
reference and the random maps above, so T calls a random map "reuse" and a copy map "new". A reading that does that on the one row
where VPD's r separates is not a label, and nothing here can show VPD winning or losing. I advise ending the line.

## The numbers, checked (block 1 on key maps, block 3 on E62's pair unless said)

- **References.** Write T 0.99, 1.09, 1.06 (ident32, rand32, rand32b; 3 s about 0.35); sharpen 1.03 to 1.12. The copy map's write
  is the lookup's, and it reads 2.48 at block 3 (cond 30.6), above the sharpening there (1.19), so on E62's pair the negative
  reference sits above the positive one and no bar of the filed shape exists.
- **Key maps.** ident32 0.08, 0.17; ident32b 0.26, 0.20 (its block 1 is under the floor); rand32 0.73, 0.74; rand32b 1.53, 1.34.
  Seeds differ by about 0.1 to 0.2 and the pair bootstrap s is under 0.05, so the gaps between maps are real. Every state's T rises
  block by block towards 0.86 to 1.15 at block 3, where the ident maps are marked blind (steering 0.72 to 0.80 at block 2).
- **E62's pair.** Copy 1.16, 1.14, 1.19 (Q 1.23 to 1.25); lookup 0.73, 0.73, 0.69 (Q 0.79 to 0.82), and at block 2 the lookup
  reads 0.18, 0.13, 0.27 with steering 0.11 to 0.15, so a readable, strong suppression. Scaled copies: copy times 1.18 moves T up
  by 0.03; lookup times 0.84 moves it up by 0.05, towards 1. Marks are "readable" everywhere, yet the copy's steering is 0.40 to
  0.48 at blocks 1 and 2: nearly half its gain is upstream of every patch site, under the 0.5 mark.
- **Anomaly, unexplained.** T_off at block 1 is below 1 on all eight key-map fine-tunes (0.31 to 0.35 ident, 0.45 to 0.47 rand32,
  0.70 to 0.76 rand32b), while the write reads 0.85 to 1.11 and the sharpening 0.93 to 1.10. Every fine-tune weakens W0's response
  to a context cue for a token it was not trained on; neither reference does. Q divides by this, so Q's meaning on fine-tunes is open.

## Answers

1. **What it establishes.** T reads whether the fine-tuned model still turns W0's context cue at a block input into the trained
   answer, more or less than W0 does. On E62's pair this follows the label: the copy map needs the cue (B earlier in the window) and
   reads at the sharpening reference; the lookup writes B from the marker and no longer needs the cue, so it suppresses it. Copy's T
   is not told from a global sharpening by T itself (1.16 to 1.19 against 1.19); only Q separates them (1.23 against 1.02), and Q is
   unvalidated (the write's Q wanders 0.90 to 3.6). On the key maps the ident states suppress W0's cue as the lookup does: their
   update sits in block 1, right after the patch site, and their block 1 now writes the key word from the key, so W0's "key word seen
   earlier" cue adds almost nothing. That agrees with E58 Amendment 1 (half-known states act as lookups) and against r (0.76 to 0.80).
   The random maps read above every reference, which no reading of "kind" explains; size is the suspect (rand32b's norm is 6.0
   against 3.1 to 3.5), but ident to rand32 is a 9 percent norm step for a 0.5 step in T, so size is not the whole story, and the
   within-state scaling moved T by only 0.03 to 0.05 for an 18 percent size change. T is not a size meter and not a kind meter;
   it is both at once, in proportions that differ by row family. Settle first: the T_off drift above, and whether rand32b's height is
   size (the swap below). Neither is a bug: the code reads as documented and stage 0 reproduces job 48971 exactly.
2. **Filing rules.** No. A rule needs a negative reference that reads like a harmless fine-tune (the write misses its fit bar on every
   map and its T_off does not drift as every fine-tune's does) and a positive one (none exists; amplify never read). On E62's pair the
   references are inverted. The table's r and ratio already read the copy-versus-random row (r 0.76 to 0.80 against 0.27 to 0.29,
   ratio 0.87 to 0.97 against 0.36 to 0.41, both "separates"); on E62's pair r is "unreadable" (1.09 to 1.10 against 0.38 to 0.42,
   unscored) and the ratio was not run. So T could score r only on the copy-versus-random row, and there it would score r as
   separating the wrong way on the strength of a reading that also calls rand32b reuse. That is not a verdict anyone should file.
3. **Options.**
   - **Key-map size swap** (stage 4's code on ident32 times 1.9 and rand32b times 0.52, about 1 min, $0, no new code beyond names
     and alphas). If the order flips, T on key maps is a size meter and the closing sentence says so. If it holds, T says the copy
     maps suppress W0's cue and the random maps amplify it, still unscoreable. Either way the line closes; this only sharpens the card.
   - **Small-gain amplify** (K about 256). $0, one more build. It would show T reads a known scaling, which finding 3 of my review
     already grants, and its gain would not match the states, so T stays confounded with size. One more patch on the last failure.
   - **Rules on E62's pair only.** Needs a new write for the cue layout; r is unscored on that row, so no VPD score follows. Not worth it.
   - **Key-swap design for the class map.** The only row where the cheap methods cannot separate, and T is blind to it by design. A
     new plan, and it would inherit T's drifting off-map reference. Not now.
   - **E65.** The decision card's other branch; labels by construction, and VPD cannot win there, only lose (its review). Open again.
   - **Stop and write that the 67M cannot be labelled by this patch.** Supported by the numbers: no high control, a negative control
     that fails to behave like fine-tunes, and a sign that flips between row families.
4. **Recommendation.** End E66 here. Run the size swap only if the write-up wants its one sentence; otherwise skip it. The card
   records three readings as evidence beside, never as labels: T follows E62's label on its pair and survives the size swap; T
   reads the ident maps as the lookup reads, with E58 Amendment 1 and against r; T reads the random maps above every reference,
   unexplained. What would change it: a 67M state of independently known reuse reading above the sharpening at a readable block, or
   a negative reference whose T_off drifts as the fine-tunes' does. Neither is a cheap amendment, and this would be the third.

## Not run

No model, no job. The suppression reading on ident32 is inferred from T, T_off and the steering shares, not from a mechanism check.
