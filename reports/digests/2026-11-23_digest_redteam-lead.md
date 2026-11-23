# redteam-lead digest, night of 2026-11-22 to 11-23 (claim-2 skeptic)

Task: check each claim-2 result against its filed predictions and controls before it reaches writeup-lead. No cells of
my own, and nothing run. Every verdict below was sent as a note to the owning lead, whip-lead and writeup-lead, in
`~/commute-data/agents/<lead>/note-2026-11-2x-redteam-*.md`.

## Verdicts on results

| cell (exp) | result | verdict | the main reason |
|---|---|---|---|
| P2.65 (E271) installed-skill cross | stopped "installs only" | holds as a negative | Only addition and parity locked; the unchanged model learns both within about 4x the diagonal's steps. The controls filed before the cross fail too (transplant 0.30 and 0.89 against a bar of 0.2). Parity tests the answer format. No known-answer case was built. Caveats applied (0cbec6e5). |
| P1.54 (E268) E_b over the needed set K | built order stands | holds with caveats | K is a sample still growing at 256 pairs, and smallest on confident states, so the repeat-task RL drop is provisional. Caveats applied (333f37c9). |
| P2.61 (E269) uncapped K* search | rival b | holds with caveats, for K* only | Over the definition's K the gaps are 0.0105 and 0.0131, on tiny SFT K, so "the readings themselves disagree" is not established. TRIED corrected (683ddda4). |
| P1.55 (E274) one-gain squared cosine | admissible, saturates | holds with caveats | Admissibility is shown only on synthetic gains and the do-nothing state; the reference test was amended after the filed run (disclosed). |
| P2.64 (E273) E_b over K, real text | built direction reversed | holds with caveats | Reversed on both readable pairs, every seed. Either the reading or the built labels are wrong; output change does not explain it within pairs. Caveats applied (9f5219c0). |
| P2.66 (E272) convergence panel | readings do not converge | headline holds | Effective rank's filed positive is not beyond base loss on the scored set (partial 0.20). The within-text 0.87 was chosen after the results. |
| P2.62 (E275) claim 2 over K | built pairs unchanged; procedural readings move | holds with caveats | Steering's partial given KL (0.07 to 0.46) rests on a nested floor, and 16 of 39 states have |K| under 1,000: provisional until those states are dropped ($0). |
| P2.66 (E276) confirmatory text panel | filed "positive" | does not hold as "erank beyond base loss" | Effective rank and base loss both order the tiers perfectly (tau 0.8165 each, the tie ceiling). The partial 0.894 comes only from their disagreement inside the tied tier. The collinearity guard caught it but, as filed, spares positive calls; my request was too narrow. |

## Plan checks taken in before the runs
E271: seed dependence, the stage D cued-item faithfulness check, the two-task bar (6 of 6, P 1/64; task level 1/4), the
SST-2 not-new rule. E272: the prior-aware Hope 1, W against each side's own null, drops fixed before reads, and the
read-position floor for every task (Amendments 3 and 7). E273: the |K| < 100 rule. E275: the K sampling floor from
derangement halves. E276: the collinearity guard.

## Pattern across the night
At matched KL the 67M panels keep losing their reuse end (Pile, code, British, English domains fall under the held-gain
floor), and the cheap readings that pass do so through text unfamiliarity or format. Claim 2 has no result yet that
separates a reading from a data-only statistic.

## Open
E272's E_b stage; the ceiling check on E272's within-text 0.87 before it is cited.
