# Review: E55 C3 plan (pruning revival, reverse fine-tuning and the probe on E58's open-pair states)

*2026-10-31. Independent adversarial review by a fresh agent that wrote neither the plan nor the advice. Plan
`reports/plans/2026-10-31_plan_e55-stage-c-open-pair.md`; draft rules
`experiments/open_pair_instruments/results/predictions.json`. Read beside them: the reuse-lead brief, Fable's advice
`reports/plans/2026-10-31_advice_e55-after-c1b-FABLE.md`, the C1b analysis, the E58 analysis, `run.json` and
`amend1.json` (numbers below pulled with jq), `src/commute/finetune.py`, `keydep.py`, `steering.py`, `decisions.md`
row 105.*

We checked whether the planned run can tell if the decomposition's blind spot on the elicit map is shared by two cheap
methods. As ported, both methods measure how easily the original model's top answer comes back, which is not the
capability that the copy map, the reference pole, reuses. The only map that is old by construction also cannot be read
by them. So even a clean result would not settle the question. Three changes would fix that: a target that every map
can be read on, a planted control the methods must pass, and a continuous reverse reading at a lower rate.

## Numbers this review relies on (from the files)

| map | steps | ‖δW‖ | carry x (seeds 0-2) | carry label | r | S |
|---|---|---|---|---|---|---|
| elicit | 10, 10, 10 | 3.12-3.18 | −0.03, −0.04, −0.03 | new ×3 | 0.76-0.79 | 0.49-0.50 |
| cipher | 9, 9, 8 | 3.07-3.26 | −0.30, −0.38, −0.34 | new ×3 | 0.22-0.24 | 0.15-0.22 |
| copy | 10, 10, 10 | 3.11-3.15 | 0.42, 0.55, 0.47 | neither, reuses, neither | 0.83-0.85 | 0.61-0.68 |
| rank-1 | 10, 10, 10 | 3.10-3.18 | 0.14, 0.10, 0.09 | neither ×3 | 0.84 | 0.21-0.25 |

Every state sits 8 to 10 Adam steps and about 3.1 in Frobenius norm from W₀.

## Blocking

**B1. The instruments as ported do not read the capability the poles are defined by.** Plan § 3, "The old behaviour"
and the R and s rows; § 1 "the old behaviour is W₀'s own next token".
In C1b the revived capability (count a) was exactly the f_C that task P reuses, so "revives fast" meant "reuses". Here
the revived behaviour is W₀'s top-1 next token after marker and key. None of the three scored maps is g∘f_C for that
f_C in the same sense:
- Copy (value = key) would reuse a copying capability. Whether W₀'s top-1 answer survives under the copy fine-tune says
  nothing about that capability.
- Elicit reorders W₀'s own top 20, so W₀'s top-1 is its nearest competitor.
- The cipher overwrites it.

So R and s measure whether W₀'s answer was wrapped or erased by about 10 Adam steps. That is Jain's wrapper question
about one behaviour, not where the new map's mechanism comes from. The copy-against-cipher gap (rule 1, "each
instrument's positive control") is therefore not a positive control for the construct. A gap could come from copy
competing less directly with the rank-1 token than the cipher does, or from a difference in D. "No gap" (planned at
45-55%) is the expected result if the instruments work as Jain intends.
**Fix.** Give each instrument a target it can read on every map, with a known answer on at least one map; see B2. At
minimum, rewrite § 2 "What changes next" and the claim so that a "VPD" reading means "W₀'s top answer is about as
recoverable after elicit as after copy". It must not mean "the limit is shared".

**B2. The poles are unchecked, and the one by-construction "old" map is unreadable.** Plan § 3 "Positions", § 4 rank-1
column ("< 0 → void"), § 10 "Scoring ... the rank-1 map ... as a pole" not run.
The rank-1 map strengthens W₀'s own top answer, so by construction it is the purest "old". Yet its carry is 0.09-0.14
("neither" ×3), about as low as elicit's, and its r (0.84) is the highest of all maps. So the carry label disagrees with
construction on rank-1 just as it does on elicit. `decisions.md` row 105 already says that on a lookup task the carry
reads "no rule generalises" and must disagree with r. The design therefore cannot tell "the granularity limit is shared
by cheap instruments" from "the carry label does not read reuse on per-key lookups". Both predict the same outcome, the
elicit map sitting with copy. Choosing W₀'s top-1 as the target makes rank-1 void by design (D < 0), so the one map
whose truth is not in dispute cannot calibrate the instruments.
**Fix.** Use a target that is non-void on rank-1. Revival and reverse fine-tuning should both aim at W₀'s full
next-token distribution at the answer position: KL(W₀ ‖ m) on the score half, with revival read as the share of that KL
removed and reverse fine-tuning trained with KL-to-W₀ as the loss. § 10 lists this as not run; it should be the main
arm. Then score positions on the rank-1-to-cipher axis, where both ends are fixed by construction. Report copy's
position and the carry label beside it. State the alternative reading (the label reads rule generalisation) in § 2 as
an outcome the run can produce.

**B3. Reverse fine-tuning at the forward rate gives s too little resolution to pass the firmness rule, and it reads
distance.** Plan § 3 s row; § 9 "Here it runs at the forward rate"; predictions `reverse_recipe`.
Every forward state took 8-10 Adam steps at 3e-5, and Adam moves each parameter by about lr per step. The reverse
should then land within about 1-10 integer steps for every map. Rule 1 needs copy − cipher ≥ 3 × the largest seed
range. If seeds differ by 1-2 steps, that is a 3-6 step gap on an axis about 10 steps long. One step of seed noise on
elicit then moves p by 1/gap, about 0.2-0.3. With that, all 27 triples falling in one third-band is very unlikely. The
departure from Jain's lower reverse rate was justified as "the recipe is not a new setting". But Jain's lower rate is
the sourced setting, and C1b's reverse also ran below its 1e-4 forward arms.
**Fix.** (a) Reverse at Jain's lower rate: lr/10 = 3e-6 stretches s by about 10×, and it is a sourced setting.
(b) Make the reading continuous: interpolate the crossing step between checkpoints, or read the share of D (or of the
KL in B2) recovered at a fixed step. (c) Read every step to the cap. `CIPHER_CHECKS` jumps 30 → 100 → 300, so one seed
crossing after step 30 records 100 and makes the noise floor about 70.

## Should fix

**S1. The firmness rule is worst-case interval arithmetic and gets stricter with more seeds.** Rule 3, "every
single-seed triple". With n seeds there are n³ triples, and the band test is decided by the most extreme seed of each
map. Adding seeds can only make "firm" rarer, so the 82% "not settled" is partly built into the rule. **Fix.** Use a
seed bootstrap instead: resample seeds within each map, 2,000 draws, and call the side firm if the central 90% of p
lies in one band. Then add seeds. Forward fine-tunes are 10 steps and E58's readings of r and S are cheap, so five more
seeds of elicit, copy, cipher (and rank-1) cost minutes on `debug`. (They are new fine-tunes, which takes the plan
beyond "states read from disk"; say so in the design check.) Note also that a zero-denominator triple cannot occur once
the gap is "ok": copy_k − cipher_j ≥ gap − range_copy − range_cipher ≥ noise > 0. The edge case is harmless, but the
rank-1-reference p has no such guard.

**S2. The joint verdict makes one void instrument decide the outcome.** Rule 4: "shared" iff both read "VPD". With
P1 at 45% and 55%, the chance that both instruments have a gap is about 25-30%, which is most of the 82%.
**Fix.** Give each instrument its own verdict. Make the joint verdict "shared (by X)" when one instrument reads firm
"VPD" and the other has "no gap". It should read "conflict" only when one reads "VPD" and the other "label" or
"reversed".

**S3. Distance from W₀ (D) is described, not scored.** § 4 p_D row "unknown"; P3 "described". Both scored readings are
functions of D: R is divided by D, and s scales with D. Elicit's values are W₀'s rank 2-20 picks, so its D may differ
systematically from copy's and the cipher's. **Fix.** Read D on all 12 states as the first item of Stage 0 (a few
forwards) and write the three map means into the plan before filing. If D_copy ≈ D_cipher, p_D has no gap, and s's p
needs a structure-against-distance guard. File a rule: an instrument whose p falls within 0.1 of p_D (or in p_D's band)
reads "distance", not "VPD". Give expected values for D per map: they are missing from § 4.

**S4. Attention share and unit location are unrecorded.** P1 "blind to" names the 29-31% attention share. If that
share differs between copy and the cipher, R reads it. **Fix.** Add each state's MLP share of ‖δW‖² (from the state
files, CPU) as a baseline column. Save the top-K unit ids per state, so a reader can see whether revival units sit in
the same block across maps.

**S5. Pruning's positive control cannot fail.** § 5 item 4: importance pruning on W₀ lowering W₀'s own answer is near
certain and does not show that revival can find a hidden wrapper on this model. **Fix.** Plant a wrapper in Stage 0,
for $0. Take W₀ plus an edit confined to 13 MLP units of one block whose down_proj rows write −(u_rank1 − ū) at the
answer position, scaled so D ≈ the median pole D. Pruning must rank at least 10 of the 13 planted units in its top 13
and read R ≥ 0.8. Reverse fine-tuning must recover it in fewer steps than the pole median. A miss voids the instrument
before any scored state is read.

**S6. Capped s enters means as an arbitrary 301.** Rule 1 and p average s values that include the cap. A map with one
capped seed gets a mean set by the cap, not the data. **Fix.** If any pole seed is capped, the column reads "censored"
and is void, or use the B3(b) continuous reading, which has no cap.

**S7. D and the s criterion use different statistics and windows.** D is the mean log-odds on the score half, s is
the criterion on the arithmetic mean probability (`eval_arm_a`'s p_o) over all 75 windows, and R's unit scores use the
fit half. A state can have D > 0.1 and still meet the s criterion at step 0, so it is void for s and not for R. **Fix.**
Read s's criterion on the score half's mean log-odds, the same ℓ as D.

**S8. The copy pole is described too strongly.** § 3 says copy "carries most, with the only 'reuses' label". Copy's
labels are neither / reuses / neither (x 0.42, 0.55, 0.47). **Fix.** Say "one of three seeds".

## Minor

- **M1.** R is divided by D with a floor of 0.1 log-odds, so a seed just above the floor can give R ≫ 1. Make the floor
  relative (for example 0.2 × the smallest pole-map mean D) and report the unnormalised revival beside R.
- **M2.** "Max over K" biases R upward. It is harmless because it is the same for every state, but say so.
- **M3.** Time. Twelve 300-step reverses with 32 evaluations each, plus pruning, is 15-30 min against a 30-min `debug`
  limit. Resume covers it, but stopping each reverse a few steps after the criterion (or the lower rate in B3, which
  needs a longer cap) changes the budget. Re-estimate after B3, and move to `normal` QoS if it exceeds 30 min
  (Qwen-only rule permitting; otherwise two `debug` jobs).
- **M4.** The bands (thirds) have no source. That is acceptable for a filed rule, but the five worked columns show S at
  0.684 with triples 0.586-0.759 straddling 2/3, so the band edge decides real readings. Keep the bootstrap of S1 so
  that the edge is not decided by one seed.
- **M5.** The probe q is "W₀ and the map only". For elicit it reads ≥ 1 by selection, and for copy the logit lens at
  block 1 probably decodes the key token to itself, so q will put elicit and copy together trivially. It is fine as a
  reference, but say this in § 4 so it is not read as a third instrument agreeing with VPD.

## Checks asked for, briefly

- **Does it answer its question?** Not as designed (B1, B2): a "VPD" reading has an alternative explanation that the
  plan does not separate.
- **Is it worth running at an 82% "not settled" lean?** The portability check (P1) is worth its $0 as a port test. The
  verdict is not, until B1-B3 and S1-S2 are fixed. With those fixes I would put a settled answer near 40-50%.
- **Rules and edge cases.** Sound arithmetic; ties and voids are handled. The zero-denominator case is unreachable once
  the gap is "ok" (S1). Censoring (S6) and the statistic mismatch (S7) are not covered.
- **Fair port.** K scaling, Alg. 3's score and "minus random" are faithful and stated. The reverse rate is a departure
  that removes a sourced setting (B3). The target choice is the main departure, and it changes the construct (B1).
- **Control first, early stop.** Yes: Stage 0 runs alone with exit 1 on the stop rows. Pruning's control is too weak
  (S5).
- **Expected values.** Given for most numbers. Missing for D per map and for p_D (S3), and for planted-wrapper
  readings (S5, new).
- **Confounds.** Distance (S3, B3); attention share (S4); update size is matched by design, which is good. Step count
  is near-matched (8-10).
- **Route and saves.** `debug` on an L40, $0, small JSON committed and no states written: fine. Add unit ids and
  per-step curves (S4, B3).

## Verdict

**Redesign** the scored question (B1-B3): change the target to W₀'s distribution so that rank-1 is a pole with known
truth, add the planted-wrapper control, and use a continuous reverse reading at Jain's lower rate. Then apply S1-S3.
If the lead wants only the port check now, Stage 0 plus P1 may run as written, relabelled "do C1b's instruments carry
over to the 67M". No position or verdict should be read from it.
