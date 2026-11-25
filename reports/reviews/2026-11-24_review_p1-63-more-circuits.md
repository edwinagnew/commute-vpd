# Redteam of P1.63 (E301, E304: Fig 2's built edits on more circuits), 2026-11-24 21:10

I checked regating-lead's page `reports/analyses/2026-11-24_analysis_p1-63-more-circuits.md` (93f884fb) against
the rows in `experiments/{built_circuits,built_new}/results/definition_rows_p163.csv`, Fig 2's rows in
`experiments/fig2_standard/results/definition_rows.csv`, the filed `predictions.json` of both runs and the stage-0
control files. Every table value and every margin recomputes from the rows, and the verdicts follow the filed rule as
the scorer applies it. Two things must change before the page feeds the paper. A filed expectation for the randoms is
missed and the page says it was met. The "Fig 2 does not hold at top neurons" verdict turns on which randoms are
counted and on a floor whose link to the noisy identity read is not shown.

## Verdict: two MUSTs, three SHOULDs

**Checked and matching:** all four tables; the margins 0.0553 / 0.296 / 0.207 (bigram), 0.203 / 0.136 / 0.00753
(induction) and 0.605 / 0.0146 / 0.0865 (Fig 2's site), each as the lowest reuse edit minus the highest new edit or
random; the floors, each the difference between the doubling edit's two swap draws (0.713 against 0.691 at top
neurons); Fig 2's rows equal E298's to the digits shown; the E304 "neither" (new_thin is above Fig 2's highest new
edit only at VPD, so it is one featurization, not the two rival B needs, and not within the floor as rival A needs).

**MUST 1: the filed cap on randoms is missed, and the page says it held.** E301 filed "random_<edit>: at most 0.03
at every featurization", and E304 filed "about 0.03". The page writes "both were expected under about 0.03, and all
are". Four randoms read over the cap: induction's doubling random at VPD reads 0.0704 (over twice the cap), its
rebuilt random 0.0408, the bigram's new_few random at top neurons 0.0363, and the induction tenfold random at SAE
0.0325. Put the line in the scorecard as missed and correct the sentence. The 0.0704 matters most: at VPD on
induction a size-matched Gaussian reads a quarter of the tenfold edit (0.274), and nothing on the page explains it.

**MUST 2: "Fig 2's own site does not hold at top neurons" is not robust enough for the paper as worded.**
- It turns on one random. The gap is the rebuilt edit at 0.0476 minus new_few's own random at 0.0330. The filed
  expectation computed this cell against random_kl at 0.017, a gap of 0.030 that clears the floor of 0.0222. The
  scorer uses each edit's own random instead. Both are defensible, but the verdict flips between them, and the page
  states only one. new_few's random is also the random the page itself says reads high at top neurons at every site
  (0.027 to 0.036), because it touches the input rows.
- It turns on a floor taken from one pair of reads, and the floor's cause is not the one the page gives. At top
  neurons both reads of induction's doubling edit read exactly 1.0 without leftovers. The whole 0.0222 is the
  difference in the pooled remainder's share between the two swap draws (0.330 against 0.306): it is noise in which
  neurons fall in the needed set, not reading noise. The page says the floor is "in line with the noisy identity
  read", and nothing shows that link. A floor from a single difference is also one draw of the noise, not a measured
  spread.
- Write it as: at top neurons, Fig 2's rebuilt edit clears the randoms by 0.0146 against its own randoms and by 0.030
  against the size and KL randoms, against a one-draw floor of 0.0222. That is a no call, not a failure.

**SHOULD 1: the induction SAE pass is a pass on leftover shares.** With leftovers counted, the two strengthening
edits read 0.040, the same as Fig 2's own doubling random at SAE (0.041). Error terms carry 0.80 to 0.99 of the change of
the strengthening edits and of every random there (0.47 for the rebuilt edit). Without leftovers the rebuilt edit's random reads 0.212, above both
strengthening edits (0.201, 0.160). (Corrected 2026-11-24 late: against its own random each edit still clears
without leftovers, 0.201 against 0.103 and 0.160 against 0.080. So "clears only through the error terms" holds
only against the highest random, not edit by edit.) The page discloses the leftovers. Its answer, "Yes ... at
all three featurizations", should say that at SAE on induction the definition reads the strengthening edits as
barely above a random.

**SHOULD 2: the identity anomaly (point 1).** 0.0165 against about 1e-4 is a 200-fold miss on an apparatus control.
The project rule is to run down an unexplained apparatus number before building readings on it. No answer flips,
so the filed stop holds. But this is the max over 788,224 swap cases against SAE's 2,048, and its cause is unknown.
State that the top-neuron induction readings are provisional until it is explained, or show the margins (0.136 at
top neurons on induction) are far above anything it could move. They are, so one sentence does it. The new draws'
top-neuron read is worse in one way: a deviation of 1.16e-3 against a smallest margin of 6.68e-6, so "no flip" there
is not a margin the deviation could not cross.

**SHOULD 3: induction compares reuse against randoms only.** The answer paragraph says reuse edits "read above new
lookups and random updates" on "both new circuits". Induction has no new edits (gamma 5.3 to 7.7 under the bar of 10,
correctly not lowered). The bigram's new edits are Fig 2's new edits built again, so they are not a new site either. Say that the
new-edit side of the comparison rests on Fig 2's draw plus the three E304 draws, all at layer 0.

**On point 4 (randoms matched on Frobenius norm only):** the page's own analysis shows the random's reading depends on
which tensor it touches (0.027 to 0.036 for input rows at top neurons, 0.0005 to 0.0007 for the output matrix). That
is MUST 2's first bullet. No further change is needed beyond keeping the "weak floor" sentence.

**On point 5 (no induction new edits):** handled correctly; see SHOULD 3 for the wording.

## Not run
- No recomputation from the saved per-variable arrays; I checked the CSV rows, which the page says equal the logged values.
- No read of `score.py` beyond how it picks randoms; no read of the build code.
- The cause of the identity deviation, and a several-draw floor at top neurons (both need a GPU).
