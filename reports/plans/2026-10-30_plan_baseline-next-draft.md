# Plan (DRAFT): what next for the baseline comparison after E54? (E56 proposed)

**Status: APPROVED by the user 2026-10-30 ("approve E56, go ahead"), as written. Drafted by Fable.** Earlier status: draft,
Amendment 1 (`reports/analyses/2026-10-30_analysis_e54-keydependent-elicit.md`, review
`reports/reviews/2026-10-30_review_e54-amend1-analysis.md`).

## Where the baseline line stands, in plain words
E54 built a pair of fine-tunes over the same keys, one table the original model already half-knows (elicit) and one
arbitrary (cipher). The route split's re-gating share r read them 0.77/0.76 against 0.24/0.21; one mean steering
vector read them 0.25/0.25 against 0.22/0.18 in the same unit (Amendment 1). So r separates the pair and steering
does not, by a margin bounded below near 6x and not above, because steering's gap (0.052) is barely over its seed
range (0.043) with two seeds. The claim to defend is: **VPD's route split reads the kind of an update where a cheap
activation baseline cannot.** Its three weak points today: two seeds; only one cheap baseline tried on this pair
(steering, which a key-dependent map defeats by construction); the "elicit" label rests on W₀'s ranks, never on an
independent readout. One pair, one 67M model.

## Options weighed
| | what it learns | cost, platform | depends on |
|---|---|---|---|
| (a) Jain et al. Def 1 readout + causal check on the elicit keys | whether W₀ linearly holds key -> value for the elicit table and not the cipher, and whether ablating that readout breaks the elicit fine-tune: a label independent of VPD (`decisions.md` row 81) | one cluster debug job, $0 | a readout apparatus for word tokens; the reuse line (E55) is building Jain's checks on minGPT, not on the 67M; needs the user's confirmation (new measurement on this line) |
| (b) seeds 2-4 of the pair | tightens the only number the review flagged: the ratio floor; with five seeds the separation rule's range term stops being two points | two cluster debug jobs, $0 | nothing; approved class 2 |
| (c) the pair on Qwen via E45's anchors | the headline on a real model | gated, $15-20 on an H100 | E45 Job 1: anchor 1's prompt check failed ("P: a prompt qualifies" holds false), so anchor 1 is on its fallback; the route split is not on Qwen yet (E44 Amendment 2 is a reading-B pass test, r arrives with Q3); another lead's jobs |
| (d) neuron patching (E52's B2) on the six saved E54 states | whether an **input-dependent** cheap baseline separates the pair; if it does, "cheap baselines cannot" is false and the claim shrinks to "one mean vector cannot" | one cluster debug job on saved states, ~10 min, $0 | nothing; approved class 1 |

## Recommendation: (d) then (b), as one experiment E56; (a) next; (c) waits
(d) is the most informative per dollar: it attacks the claim's sharpest objection, that the pair was chosen to be
steering-hard. E52 ran B2 on E12's pair (full-width share copy 0.50-0.71, cipher 0.76-0.97, no separation), so the
harness exists and the result is decisive either way. (b) is the cheapest fix of the review's medium finding and
runs in the same session. (a) is worth doing but is a new measurement on this line: it is asked for separately below.
(c) is not ready: no r on Qwen, anchor 1 unsettled, and it is qwen-decomp-lead's work.

**What would change this.** If E44 Amendment 2 passes and Q3 gives r on Qwen, (c) moves first, starting with a $0
cluster-only steering reading of E45's anchor 2 (invented facts, a key-dependent map by construction). If reuse-lead
ships a Def 1 readout for 67M word tokens, (a) becomes class 1 and joins E56. If the user wants the Qwen headline
over 67M robustness, skip (b).

## Design check (`/design`)
- *Goal:* GOALS "teaches new vs strengthens existing": whether a decomposition reads kind where cheaper readers do not.
- *Nearest tried:* E54 (r vs S on this pair), E52 (B2 on E12's pair). E56 learns what neither could: B2 on a
  key-dependent pair, and the ratio floor at five seeds.
- *Patch or step:* a step, by closing the pair's weakest objection; (b) is a patch on E54's seed count.
- *Decisions:* relies on E51's separation rule and E54's matched-size reading; departs from none.
- *Changes next:* if B2 separates the pair, the headline drops "cheap baselines" and names steering alone; (a) and (c)
  are then asked under the narrower claim. If it does not, the claim stands and (a) is the next ask.
- *Class:* (d) class 1, (b) class 2. Still **DRAFT** at the user's request; nothing runs before approval.

## Claim, null, predictions (E56)
**Claim.** A key-dependent map the original model half-knows is read as "elicited" by the route split and by no
cheap activation baseline, mean-vector or neuron-patch. **Null.** Neuron patching separates the pair as r does;
r's separation is a size or sparsity effect any input-dependent patch recovers.

Shares are over the fine-tune's gain G at the answer position on the scoring half, as in E52 and E54. The
separation rule is E51's: gap between task means above the larger within-task range, and at least 0.2.

| number | how measured | W₀ | do-nothing | control | expected if claim | if null |
|---|---|---|---|---|---|---|
| N16 | E52's B2: top-16 MLP neurons by mean activation shift take the tuned activations, share of G | 0 | identity patch 0 (within 1e-6) | full residual patch 1 (within 1e-3) | elicit and cipher both 0.1-0.4, no separation | elicit > cipher by the rule |
| k50 | neurons needed for half of G | undefined | all | 1 for the residual patch | > 1000 on all states, no separation | elicit far fewer |
| r, S_lo at seeds 2-4 | E54 and Amendment 1 unchanged | 0 | 0 | seed-0 states reproduce stage2.json within 1e-6 | r 0.70-0.80 vs 0.18-0.28; S_lo gap under its range or under 0.2 | r's gap closes with seeds |

- **P1 (B2 does not separate).** N16 and k50 each fail the separation rule on elicit vs matched cipher (two seeds in
  job 1, five after job 2). Lean: medium. Blind to changes carried by attention; floor: the within-task range.
- **P2 (ratio floor at five seeds).** r separates elicit higher by the rule and S_lo does not; the r gap divided by
  the S_lo gap plus S_lo's larger range is at least 3. Lean: medium-high. Blind to a per-key split; floor: 0.04 on
  r, 0.08 on S from E54.
- **P3 (the cipher stays a cipher).** Seeds 2-4's matched cipher reads r within 0.1 of its own-step r, as E54's P3.
  Lean: high.

**Control first.** Job 1 stage 0 alone, early stop: sha256 of the six saved δW; E52's identity patch 0 and full
residual patch 1; r and S reproduce `stage2.json` within 1e-6. Job 2 stage 0: the elicit rate at W₀ reproduces
`stage1.json` (same keys, same values) and seed-0 elicit r reproduces within 1e-6 before any new fine-tune.

**Decision tree.** P1 holds, P2 holds: write up as the headline, ask (a). P1 fails: stop, tell the user; the claim is
narrowed to steering before any Qwen ask. P2 fails (the ratio floor falls under 3): report the ratio as "small but
directional" and ask whether (c) is still worth gating. P3 fails: the matched-size reading is suspect; tell the user.

## Departures from the literature
- **Neuron patching.** Causal tracing (Meng et al. 2022) restores clean activations into a corrupted run; we patch
  fine-tuned MLP activations into the original model at one position, rank neurons by mean activation shift on a fit
  half, and read a share of the fine-tune's gain. Unit, ranking and the fit/score halves are ours (E52).
- **Steering.** As in E54: tuned-minus-original mean at the answer position, multiplier 1, not Rimsky et al.'s
  contrastive pairs; the share-of-gain unit and the log-odds reading are ours.
- **The separation rule** (gap over range, minimum 0.2) and the ratio-floor bar of 3 have no source.
- **Jain et al. Def 1/2** (option (a), not in E56): the `decisions.md` row 87 departure would apply (g must hold on
  held-out keys).

## Cost and routing
Job 1 (B2 on saved states): one MATS debug L40, ~10 min, $0, `pod-runner`. Job 2 (seeds 2-4: three elicit, three
cipher fine-tunes with checkpoints to the matched step): two debug jobs of ~25 min, $0. Not gated. Saved: six new δW
(113 MB each) in `results/large/`; summaries committed. If the user adds (a): one more debug job, $0, after a short
plan amendment naming the readout and the ablation.

## Not in this plan
Jain's readout and causal check (asked separately); a matched random δW (its gain sits under the 0.05 floor, so r
and S read nan by rule and it tells nothing E54's P3 did not); Qwen; a per-key breakdown; the copy task on these keys;
E51's weight statistics on the pair.

*Hand-off: a new session takes this file, E54's analysis and `experiments/baseline_neurons/run.py` (B2) once the user
has approved or amended it.*
