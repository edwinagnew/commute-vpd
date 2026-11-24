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

## P1.57, due 11-24: E283 reliance filing (f9d556e5) checked, pre-run points on V1 sent (~14:30)
V2 fails its control (x2/x10 0.998 vs new_thin 0.998): at f = 0.25 almost every variable of a small update is "barely changed", so V2 measures update size. Points on V1: (1) the control compares two Ds (copy24, all positions vs cue windows, last position), and the only same-D contrast, x2 against random_size, is expected to saturate, so a pass may be a D difference; read x2/x10 on the cue windows if possible; (2) file the expected V1 on the new edits from their construction before the scans return; (3) V1 is flip-weighted like the failed flip-weighted share: report the unweighted overlap and |K_M| beside; (4) apply the card's 0.2 gap to rival A's pattern; (5) V1 counts a new circuit from old parts as reused.
E283 a1 (0f902283) checked ~14:40: filed before V1. One contradiction sent: rule c's max over the new edits includes new_few, bounded >= 0.966, so rival A cannot pass by construction. Recommended a per-item read (new_thin, random_kl), with new_few as the construction check; recompute the x2/x10 bound on the cue D.
E283 a1b (d044008b, 18259a76) checked ~14:50: filed before any V1 value (no V1 result in the tree), per-item rule as recommended. Nothing open before V1.

## P1.56, due 11-24: E277 page (936590f1) checked, OK to fill after two small fixes
Make-up: the cluster re-read of lower s0 and its Gaussian matched the pod values (E_b 2.2e-5 relative, fresh stage 0 with no flips). Same-task tau: neurons -0.348 (p 0.23, "no order"), VPD -0.696 (p 0.007). My 9x and +0.014 were misreadings; the right values are 10.3x and +0.013. Fixes: the answer's remainder range 0.60-0.77 is for E229's fine-tunes only (the panel reads 0.53-0.80); label each margin as median or smallest.

## P1.57, due 11-24: E283 reliance verdict (f288f4c0), sent ~15:10
Rival B as filed on both readings: V1 0.848 against new_thin 0.851 and random_kl 0.857; V2 0.992 against 0.998-0.999. V1 fails its positive control: x2/x10 read 0.848/0.859 against construction floors 0.993/0.983, and new_few reads 0.797 against >= 0.966. The needed set moves with F's outputs, so V1 as built has a floor near 0.8 and tests nothing about reliance. Asked for: the stop-rule wording ("the readings as built fail their control", not "the needed set itself"); one number to show the cause; E62, E280, E281 and the panel labelled as values of a failed-control reading; and an answer paragraph saying P1.57 is a run of instrument failures. Note in the whip-lead folder.
P1.57 page E283 section (622e6dfa) checked ~15:30: OK to fill, all four points in. Correction to my own verdict: random_kl's V2 is 0.992, not 0.998-0.999 (asked for a bracket in the quote).

## URGENT (whip-lead for the user, 11-23): literature-method fidelity audit, done ~17:30
Table: reports/reviews/2026-11-23_review_baseline-fidelity.md (copy in the whip-lead folder), per-review files beside it. 5 Opus auditors + 7 GPT reviewers. The one result-changing finding: the 67M crosscoder omits Minder's negative-base-beta exclusion; with it, the block-3 control fails (.44/.43/.42) and the ladder top falls to <= .01, recomputed independently. Everything else is relabels, disclosures and CPU rescores: ADL, the neuron-share Vig label (→ CMAP, scoped), "fewest" neurons, Venhoff steering, a strength-1 figure that plots tuned values, the Jain 67M/E59 labels, App. G's Def. 2, SAE sites, update size under one norm (0/7/2), the effective-rank rationale.
P2.69 (due 11-25) spec check sent to definition-lead: OK with fixes (x2 bar vs the cell's "full overlap", lenient half-window floor, faithfulness gate, control for "new").

P2.67 (due 11-25): ADL spec checked ~17:50, OK for Option A with four fixes (note in the compare-lead folder): a departures row for the old Gaussian denominator; label the column 'detects domain, no reuse call' with the adaptation as ours; pass rule (a) as a majority of positions plus 2 sd; check mix1-2p0 and Qwen BOS.

P2.68 (due 11-25): CMAP spec (725325a4) verdict ~18:30, OK after fixes, no blocker (note in the regating-lead folder). Majors: the gate is measured on e58 rows only; the compared scalar is unnamed and formats differ; missing §5 departures (base already at 0.66 in the paper, rescaled F, head-only swap). P2.70 size/steering page (579587f2) OK to fill: all numbers match the audit's recomputes.

P2.70 crosscoder rescore (1a36e7d6) checked ~18:50: the numbers are reproduced; the filter mostly drops the plant via base betas of about -0.01 (noise at the 100k fit). Advised: keep the literal rule, refit the betas on more tokens, not a floor. Block 3 fails under every reading (narrowly); the ladder never passes; the real-text plant is rule-dependent, not called. Audit table addendum added.

P2.70 refit plan (fdce9bc0) checked ~19:10: no objection (25M halves, ~$7). Framing filed: a true base beta of 0 keeps a random sign at any token count, so the refit decides ours vs theirs, not a rescue. Rule: both halves negative and > 2 SE stands; otherwise a 'theirs' limitation, with the literal reading kept and a near-zero-kept figure beside it labelled ours. Page c6604809 OK.

P1.60 (due 11-25): rotation-kept spec checked ~19:30. The degeneracy holds. Fixes: exclude remainders from the pooled shares (otherwise new_thin's ≤ 0.5 fails by construction); add an anisotropy-matched null beside ν; add a built rotate-toward-needed vs rotate-toward-non-needed control before the reading counts; give the head (2,4) own-space null.

P1.59 (due 11-25): E285 page (62070afc) verdict ~19:45: OK after fixes. The exploratory reading ≈ the write share over K (adds no new number); at VPD 'low' cannot be told from 'spread' (the doubling's median subcomponent keeps 0.0007), so the group-level kept read is the main Discussion line; date the amended check as post-first-run; coverage misses fit the margin story; give a within-class Spearman; soften 'ruled out'.

P1.57 (due 11-24): E283 section on definition-lead's page (0e89a083) OK, no fixes; their two corrections to my verdict text (E281 lookup 0.815 is the full-size seed; random_kl V2 0.992) are right.

P2.68 (due 11-25): revision 4 (32c6e0ad) applies majors 1-3; OK. The labelled-subset choice is OK with three fixes: record D_labelled/D_n per item; list E53's other windows as a §5 departure; E62 CMAP reads copy events only. E288 stage 0 stopped, and both stops are real. Every item's E248 files hold mixed 75- and 512-window readings, so the scored files must be named. The value-row plant does not clear the gate: the plant recipe is their call, but loosening the gate needs an ask.

P2.69 (due 11-25): E287 page (EAP-IG overlap) verdict: fixes before it goes on. Stage 0, both controls and the head claim check out. But M's KL gap is ~5e-5 on every key-grid task and 5e-6 on E62, 2-80x below the F's, so M's circuits are circuits for almost no behaviour. "Calls recruitment of existing parts new" is downgraded to unshown. The noise argument holds only in part: compare at matched KL, where the KL random's 0.82 sits against the new edits' 0.50.
P2.69: E287 page revised (41c77081), OK to go on. My verdict was wrong on one point: Wang's KL is averaged over the vocabulary, so M's key-grid gap is 2.7 nats, not "almost none". The 3-8x (key grid) and 33-75x (E62) gaps still hold.

P2.71 (whip-lead): verdicts on the 5 literature challenges sent (whip-lead/note-2026-11-24-redteam-p271-lit-challenges.md).
- Sentences: Angsten (Qwen), Lee (behaviour decides it), Prakash/Chhabra (role vs gain), gain-only fine-tunes (scoped: SVFT near-full only on Gemma-7B).
- One $0 check inside P1.60's scope: the gauge rotation from Singh.
- One new cell worth proposing: an IA3-style gain fine-tune as a trained positive control.

P2.72 (compare-lead, Delta-Crosscoder, due 11-25): spec 2c0fd0c6 needs revision. Blocker 1: the tail threshold comes from a null with no spread (0.5 exactly under paired init), so it filters nothing. Blocker 2: our share decides the method's column, where the paper's output is a steering yes/no. Majors: the max-activation check is dropped, the steering departs without rows, the shared decoders should probably be untied, the Minder-trainer behaviours need departure rows, the C1 bars are unmeasured, and C0's tolerance is not measured under bf16.
P2.73 (definition-lead, WeightWatch, due 11-25): spec OK after fixes. The detector follows the code. Our base-use score S needs a floor matched to the update class (gradient span, $0), not an isotropic Gaussian. The detector's positive control (new edits) lacks a derived expectation and may be absorbed by calibration, so add a trigger along M's lowest-variance direction. Also needed: an M-with-F's-directions floor, and a check on position 0 and the mean share of tr C.

P1.60 (E286, due 11-25): gauge stage OK.
- The rotary-restricted gauge is right: rotary covers the whole head, and V/O uses the full gl(128). My P2.71 tangent was wrong for a rotary model.
- The VPD drop to 0.32-0.40 under a function-identical rotation is the featurization (per-matrix subcomponents), not the kept test, which reads 1 at heads and neurons.
- Trained updates' gauge share (~1%) is under random (4%). "Gauge doesn't explain the low kept share" needs a $0 check first: re-read kept with the gauge part removed.
P2.72 re-check (94050d16): OK after two fixes. Both blockers are fixed. (1) Run 2 must also need the random update (C2) at site A to read no. (2) The steering cap goes to about 0 for small updates, so a "no" there follows by construction: source the cap or add a row, and give it a floor or read uncapped.

P2.70 / E289 (beta refit, 4f35219e): OK. All 10 plant latents have real negative base betas, so the literal exclusion stands and the paper's crosscoder column uses it: no planted control passes. The premise doubt is answered by the page's own evidence: on the authors' Gemma table the filter also drops 69 latents a hair below 0, so this is a "theirs" limitation. The draft's block-3 and real-text passes are reversed, and compare-lead is to flag that to whip-lead as a reversal.
