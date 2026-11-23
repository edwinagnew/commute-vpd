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

**P1.57, 11-23, correction from definition-lead on E62's D:** D for e62/* is a mixed set from E187's read half, with 206 of 512 copy windows. The rest are cue-free and far-text windows. So my E278/E62 verdict ("no separation, both ~0.04") holds only as "no separation of E_b over K on the mixed D": K includes variables needed only on non-copy windows. Whether the maps separate on the 206 copy windows alone is untested; the head-ablation reading (0.47-0.56 vs 0) was on copy behaviour. E280 files the same D (7be0877d), which is fine for a uniform head edit. Its CPU code-path preview (K = all changed, not the reading): V x2 at VPD 0.191 against 0.214 from the derivation (rival B side), and 1.0 at neurons.

**P1.57, 11-23, E281 pre-run (E62 kept share on the 206 copy windows):** the diagnosis fixes are confirmed (fe1e834a). E281's filed gap of 0.2 is degenerate. Every fine-tune reads between 0.015 and 0.063, so the dilution rival could never pass. Asked definition-lead for a bar on the reading's own scale (>= 3 seed sd, ~0.012, or a ratio of 1.5, copy higher), and for a needed set rebuilt on copy windows from saved pair-level flips if they exist. Note: ~/commute-data/agents/definition-lead/note-2026-11-23-redteam-e281-prerun.md.

**P2.66, 11-23, E279 pre-run (untrained-SVD share on E272's wide KL-hi panel, descriptive):** no objection; the controls are sound. Reading notes to regating-lead: the two label tasks are read in another unit (log-odds over the vocabulary), so give the tau without them as well. Print the dropped tasks with their tiers. Quote the smallest of the four p values together with the count of tests. Note: ~/commute-data/agents/regating-lead/note-2026-11-23-redteam-e279-prerun.md.

**P1.57, 11-23, E280 neurons half (provisional; files not yet pushed):** the positive control passes: V and O x2 read 1.0000. Q x2 (attention logits doubled) reads 0.0404 even where head (2,4) is a single variable. So "attention sharpening reads new" belongs to E_b's write-gain clause at every featurization, not to VPD's head-mixing alone. Whether such a change should count as reuse is a definition question for the user. Quote it as "on this head and E62's mixed D". The VPD half is due next.

**P1.57, 11-23, E280 verdict (b65a8743): stands as read, rival B.** Checked against summary.json. A change confined to head (2,4) reads at VPD and at the head featurization:

| edit | VPD | heads |
|---|---|---|
| V x2 | 0.192 | 1.000 |
| O x2 | 0.200 | 1.000 |
| Q x2 | 0.249 | 0.040 |

The derivation holds on data: V 0.192 against 0.214, Q 0.249 against 0.282, and O did not read high. But this cap of ~0.2 does not explain E62's 0.04, which sits five times lower, near random's 0.011: most of E62's low reading is the direction change in real updates. Q x2 raised copy loss by 0.358, so its 0.04 at heads shows that a logit sharpening reads new, not that a useful reuse does. Whether such a sharpening counts as reuse is the user's definition question. Note in the definition-lead, whip-lead and writeup-lead folders (note-2026-11-23-redteam-e280-verdict.md).

**P1.56, 11-23, E277 e62 stage (neuron/head featurization, mixed D; provisional, summary not yet pushed):** no separation. The copy/lookup ratio is 1.22 (0.0183 vs 0.0150); the filed 0.2 gap is for the built edits, not copy against lookup. The four MLP remainders carry 65-71% of dd, so E_b at this featurization is mainly the remainders' reading. Head (2,4) alone separates (kept 0.09-0.14 vs 0.015-0.020, flips 128-141 vs 11-15 of 256), but at under 1% of dd. Whether a pooling other than dd should be used is the user's definition question. The fine-tuned copy head keeps 0.09-0.14 even as one variable, against 1.0 for a built rescale: its change is mostly not a rescale.

**P1.57, 11-23, flip-weighted kept share (filing 24b04ffd, whip-lead's ask, descriptive):** the E277 e62 numbers check against f91e6a1f (remainders 0.648-0.708 of dd). Points on the filing, sent before computing:
1. The weighting was chosen after head (2,4) separated on E62, so E62 under it is not evidence: the test is the unseen pairs.
2. Floor dd_u at a measured noise floor, since tiny changes get full weight.
3. File the expected values at the randoms and the built edits.

**P1.57, 11-23, flip-weighted kept share: it fails its positive control.** On the built x2 edit it reads 0.012 at VPD against an expected >= 0.5 (E_b 0.761); it is still 0.18 at the 1e-4 floor. At heads it reads 0.77 against ~1, from one barely-changed remainder that flips on 256/256 pairs. Flip pooling rewards being needed, not being changed, so the held-out pairs are not evidence for it. With E277/E280, both pooling extremes fail, in opposite ways: dd pooling drowns the head, flip pooling is set by unchanged variables. Any repair is a new design and part of the user's pooling question. Note in the definition-lead and whip-lead folders.

## P2.66, due 11-23: E279 scored (333be7db), notes sent to regating-lead ~12:30
All four statistics no call as filed (tier tau 0.03 p 0.50; E_b tau 0.17 p 0.31, the smallest p of the four). n 9: arxiv, german and spanish dropped (spanish has 2 finite seeds). squad 1.27 and summary -1.24 lie outside [0,1] (the numerator is bigger than the gain), and they carry the readability ratio (between 0.61, within 0.22). The Gaussian floor count (6 of 9 above 0.065) counts by sign, so it misses conll and sentiment, which are large and negative. Still not run: tau without category/sentiment, and statistic 4 at |K| < 1,000.

## P1.56, due 11-23: E277 built + trained (a82d35bb), verdict sent to definition-lead
Built: rival A as filed (g 0.982). But g rests on x2/x10, which change 9 variables. The rebuilt reuse case reads 0.048 at neurons (VPD 0.623), only 0.03 above random_kl 0.017 (post hoc). The random_size and new_thin margins are ~1e-5. Trained: no reuse/new order at neurons (pair gaps -0.008 to +0.014 with 2 seeds, about the seed spread). Each fine-tune reads 1.5-9x its own random, so at neurons E_b separates update from noise, not reuse from new. Remainders are 0.59-0.77 of dd. The panel stage is due ~13:45.

## P1.57, due 11-23: E278 grid half (def2e161), verdict sent to definition-lead
Rival B as filed: all five tasks (E_b over K 0.027-0.040) lie within 0.03 of one another and of their randoms (0.013-0.017), against rival_gap 0.2. The order reverses rival A's squared-cosine order at both ends: cipher is highest (0.037-0.040) and copy second (0.032-0.034), with no overlap across 3 seeds. Prior-matched cipher reads below the raw cipher. This is a qualitative surprise for the page's first paragraph, and no reuse/new reading follows from it. "2-3x random" has no filed bar.

## P2.66, due 11-24: E279 page (9d4b953c) checked, fill after one wording fix
All my notes are on the page, labelled post hoc. The fix: "does not order / does not follow the tiers" becomes "showed no order (no call); n 9 can miss a moderate correlation". Also: the -0.69 needs a source, and one sentence should say the page cannot tell an intuition null from an instrument null (the premise check is doubtful).

## P1.58, due 11-24: E282 filing check (799ac030, job 64538 running), sent to compare-lead
Main point: there is no positive control at V (x2/x10 are exact only at N). So rival b (both V means < 0.3) cannot be told from under-fitting of the 38,936 V gains over a fixed 600 steps. Read rival b only if the E62 V fits are flat and train KL is not well under held-out KL; otherwise no call at V. The missing control would be a 2x scaling of one VPD subcomponent. Also: quote E62 with its mixed D; give the KL-fit share beside E231's task-fit prior; q2 >= 0.95 at N would be a surprise.
E279 (P2.66): the wording fix was taken (2dcb59b5) and the log flushed (v467).

## P1.58, due 11-24: E282 results (d809cbc8), verdict: control failed, no call
The new edits read 0.967/0.969 at N and 0.953/0.954 at V (filed <= 0.3), so the E62 gaps (N +0.032, V -0.017) are no call. Scope: held-out = new windows of the same 8 fitted cues; unseen cues were not tested (a new cell, user's call). The exact o2 reads 0.841 at N, so the fit under-fits even exact cases, which makes the control failure stronger. V fits with negative KL share (random -0.504, q2 -3.67) read as "fit failed". Random at N 0.269 (filed <= 0.2) is a second, small miss.
E282 page (ca8126a9) checked: OK to fill.

## P1.56, due 11-24: E277 panel stage, verdict sent: provisional, no on-card identity check
Stage 0 was skipped on the H100 pods (a stale stage0 file was in the synced tree), so the 48 panel items are provisional until one or two items are re-read on the cluster against the 8.2e-5 / 4.6e-5 floors. The neuron tau -0.348 (10 tasks) against VPD -0.693 (12) needs the same task set and the raw tau's p; at n 10 it is likely "no order", not "against the tiers". conll's Gaussian 0.031 is above its s1 fine-tune 0.030, so the count of tasks inside the Gaussian range is needed. Category drops at neurons (|K| 44), which thins the reuse end.

## P1.57, E281 verdict: rival B (no dilution), lookup n = 1, K anomaly
Stage 0 passed (2.06e-9 / 2.18e-9 against 2.73e-9). The ratio over K_copy is ~0.53 with the filed missing rule (lookup_s0/s2 |K_copy| 97/75 dropped), against >= 1.5 for dilution, so rival B. Anomaly: lookup K_copy runs 97, 17,732, 75 across seeds; run it down before reading lookup_s1. The beside-reading over E275's K on copy windows gives ratio 0.98 with all seeds, the same conclusion and more robust: the low E62 E_b is not dilution.

## P1.57, due 11-24: page eda57a71 checked, six fixes sent to definition-lead (~14:05)
The conclusions stand (rival B on the grid, the E62 pair and the copy windows). Fixes: (1) the E281 verdict is not pending; (2) give the ratio both ways, 0.61 with the under-floor lookup seeds and 0.53 without; the filed rule did not cover this case, and both are rival B; (3) name the lookup |K_copy| anomaly (97, 17,732, 75) and label lookup s1 provisional; (4) E62's 0.04 is called "near random" in two places but is ~3x the randoms; (5) "a twentieth or less" is false for the copy-window reads (~a tenth); (6) the answer paragraph should say |K| is 41-90% of all variables.
P1.57 page 353eed4d re-checked (~14:15): all six fixes in; OK to fill sent. Nit, not blocking: the copy-window reads are 4-5x random, not 2-3x.
