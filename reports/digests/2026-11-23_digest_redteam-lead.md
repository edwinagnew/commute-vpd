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
| P2.62 (E275) claim 2 over K | built pairs unchanged; procedural readings move | holds with caveats | Steering's partial given KL (0.07 to 0.46) rests on a nested floor, and 16 of 39 states have |K| under 1,000: provisional until those states are dropped ($0). Rechecked (8c802aeb): the moves vanish on |K| >= 1,000, but steering given KL is +0.07 over all states and -0.48 over the well-sampled ones, so claim 2's procedural numbers are state-sensitive. |
| P2.66 (E276) confirmatory text panel | filed "positive" | does not hold as "erank beyond base loss" | Effective rank and base loss both order the tiers perfectly (tau 0.8165 each, the tie ceiling). The partial 0.894 comes only from their disagreement inside the tied tier. The collinearity guard caught it but, as filed, spares positive calls; my request was too narrow. |
| P2.66 (E272) E_b over K, wide set | filed "not positive" | holds as a reversal (descriptive) | Recomputed separately: consensus tau -0.69 reproduced. It holds across all 12 tasks with base loss held fixed, but within continued text the all-position loss removes it (-0.17, +0.14). It depends on the state (needed set 200 to 37,888 variables between seeds; -0.59 without states under 1,000). The random update leans the same way weakly (-0.29), partial on it -0.66. Note ~/commute-data/agents/regating-lead/note-2026-11-23-redteam-e272-eb-verdict.md. |
| P2.66 (E272) post-hoc: seed agreement, steps to learn | descriptive | negatives stand; two sentences over-read | Agreement does not carry the E_b reversal (-0.69 to -0.67), but it is partly set by the step each seed was picked at (arXiv 5/101/12 reads 0.18; German and Spanish 300/300/300 read 0.91). The re-gating share as "the only reading that runs with learning speed" (0.49, p 0.066, 8 tasks) is about what chance gives across six tests. Note ~/commute-data/agents/regating-lead/note-2026-11-23-redteam-e272-posthoc-verdict.md. |
| P1.57 (E278) E62 copy vs lookup, E_b over K | descriptive (rival B, stop fires) | holds as "no separation" only | Copy minus lookup -0.004 (bar 0.2); both about 0.04, against 0.76 for the built strengthening edit. No E62 random, so "writes new directions" is not read. Every variable changed and the margins are tiny, so K is 45-80% of everything. The values were on disk from E275 before filing. Note ~/commute-data/agents/definition-lead/note-2026-11-23-redteam-e278-e62-verdict.md. |

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
E272 E_b reviewed (verdict above); within-text 0.87 confirmed dropped by regating-lead (tie ceiling 0.906). E277 (P1.56, definition-lead) pre-run: asked for the remainders' share of the write change over K and E_b without them, since four whole-MLP remainder variables may decide the reading (note-2026-11-23-redteam-e277-prerun.md). Verdict due when E277 lands.

**P1.57, 11-23, E62 E_b diagnosis (460dab48): stands, one correction.** The measured parts hold: K holds head (2,4), the median variable keeps ~0.01, and layer-2 attention's write-change share separates the maps. The derived claim, "each subcomponent keeps its head-4 share under a one-head rescale", is true for q, k and v but not for o. For o the change stays parallel to the original write, and the kept fraction depends on the activations. Pooled over q/k/v the kept share is above the median share, but well below 1. The built head-4-only edit is the control that would settle it; it needs a cell. Note: ~/commute-data/agents/definition-lead/note-2026-11-23-redteam-e62-diag-verdict.md.

**P1.57, 11-23, E280 pre-run (built head-(2,4) edits, card experiments/head_edit/card.md):** the design is sound. The filed "QK x2 < 0.3 at VPD" contradicts the derivation: doubling q rows is the same algebra as V x2, so it should read like V at VPD. "< 0.3 at neuron/head" is a guess, not a measured band. Asked definition-lead to refile both before predictions.json is written, and to rename the arm "Q x2". Note: ~/commute-data/agents/definition-lead/note-2026-11-23-redteam-e280-prerun.md.
