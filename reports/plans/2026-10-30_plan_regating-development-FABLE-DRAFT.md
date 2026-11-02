# Fable's advice — ADOPTED by the user 2026-10-30 (ask #13, decisions.md)

*2026-10-30, Fable, at the user's request ("how best to develop the regating line to address the question 'can VPD
distinguish elicitation from new mechanism'"). Advice only: nothing here is filed or running. Words as in `GOALS.md` →
Terms. Supersedes my 2026-10-29 draft (commit `9b857aa`), whose options A and C ran as E50 and E53. Read beside
`reports/analyses/2026-10-29_analysis_e53-performed-anchor.md`, the "Amendment 1b" section of
`reports/plans/2026-10-29_plan_performed-anchor.md` (filed, about to run) and `reports/plans/2026-10-29_plan_e54-keydependent-elicit.md`.*

*Lead's note (2026-10-30): Fable wrote this without knowing that E54 (baseline-lead's) already ran as job 45963: stage 0
passed, stage 1 stopped on its held-out elicit-rate check (geo-mean 1.70e-3 < 5e-3, commit `7450698`). Update, same day:
the floor was lowered to 1e-3 (user, after seeing the value) and E54 finished (`reports/analyses/2026-10-30_analysis_e54-keydependent-elicit.md`):
the route split's r reads 0.77/0.76 (elicit) vs 0.24/0.21 (cipher); steering 0.38/0.39 vs 0.23/0.19, so steering
separates too, by about a third as much, and its filed cap failed. Option A's E54 half is therefore read; on Fable's
"what would change it" list this sits between "r separates, steering not" and "both separate". E53 Amendment 1b also
ran: "not read (swap missed)", breadth question still open. Also: no second VPD seed of the 67M exists
(`artifacts/vpd/t-9d2b8f02` is the base model), so the granularity check would need a new decomposition.*

## In plain words

The line now has one number that tells a fine-tune of something the model already does from a fine-tune of something it
cannot do, on two task pairs and every seed, but each pair also differs in how broad the update is, how much it depends
on the input, or how large it is, and a cheap method with no decomposition separates one pair just as well. The advice
is to stop adding new numbers and instead build pairs that differ only in kind, so the one number can be shown to read
the kind and nothing else, then score it against a ground truth that owes nothing to the decomposition. Only after that
should it be read on the larger model, and every claim must say that the decomposition cannot see a new circuit built
from old pieces.

## 1. Where the line stands, and what I would keep or change from the 2026-10-29 draft

**The surviving number.** The *stand-in ratio* (E46 Amendment 3, E53): in VPD's stand-in model (the original pieces,
Δ removed, each piece multiplied by Γ's mask), the gain the fine-tune's answer gets when Γ's new masks are applied to the
split pieces, divided by the gain the split pieces get with every mask open. Roughly: what Γ's re-switching lets through,
over what the moved pieces do. Copy 0.87-0.97 vs random map 0.36-0.41 (E46); lowercase 0.92 vs cipher 0.26 (E53). The
route split's re-gating share r reads the same pairs 0.76-0.80 vs 0.27-0.29 (E12, E46).

**What the number has not been shown.** (i) *Kind, not breadth*: lowercase is twice as broad as the cipher and the ratio
tracks gain per off-target KL row for row (E53); Amendment 1b (broad cipher alone) tests this and is the right next job.
(ii) *Kind, not input-dependence*: copy is input-independent, a cipher is not, and one steering vector separates E12's pair
0.77 vs 0.14 (E52); E54 is built to break this tie. (iii) *Kind, not size*: E36's tiny non-learning update read 0.82 on
the share form; E54 matches ‖δW‖. (iv) *Kind, not task*: the hand-write nulls read partly like their task (E46 A3), and
E53's never reached its gain.

**Keep from the 2026-10-29 draft:** the three-way framing of the answer (yes / no / undecidable because of VPD's unit);
Qwen waits for a 67M number known to read kind; no new fixed-shift work, no Γ refit, no new single-trigger toys.
**Change:** the draft bet on the real-model restore share as the carrier; E50 showed it measures relay, not kind, so
drop it from the programme (keep it as an unscored reference row at most). The draft ignored the confounds that E51, E52
and E53's independent check exposed; the programme below is organised around removing them by construction rather than
by adding nulls. Fewer numbers: the ratio and r, nothing else scored.

## 2. Options

**A. Matched pairs: the same number on pairs that differ only in kind ($0, cluster debug, the two jobs already
filed).** Amendment 1b (breadth) and E54 (input-dependence and size matched; the half-known key map against a cipher on
the same keys at the same ‖δW‖), each read with the ratio and r, E54 with E52's steering share beside. The pair is the
instrument: nulls stay as reference rows only. *What it settles:* whether the number reads anything but the task's shape.

**B. A ground truth outside VPD: one table with the reuse line (E55) ($0 once E55 labels 67M states).** E55 labels
fine-tunes "reuses capability C" by Jain et al. 2024 Def 2 with the dated held-out-values departure, no decomposition
read (lowercase and copy pass, a word swap fails). Every 67M state E55 labels gets the ratio and r in the same table, the
user's "same table for every update type". *What it settles:* whether VPD's reading agrees with a definition it did not
make, on states it did not choose. It also gives the first mixed cases (E55 Stage C's natural-row spurious fine-tunes),
where the two can disagree.

**C. Narrow the granularity limit ($0 proxy now; the real fix costs a decomposition fit).** The limit: VPD's unit is the
rank-one piece, so a new circuit assembled from old pieces under new gating reads "elicited". *Proxy, $0:* the
*gating-pattern novelty*, the distance from Γ's new mask vector at the scored position to its nearest neighbour among the
original model's mask vectors over Pile positions (E36's nearest-rule code, already written). An elicited behaviour
should have a near neighbour (the original model already gates that way somewhere); a new combination of old pieces
should not. It is a set-level reading, so it sees what the per-piece ratio cannot; it is blind to a new combination that
happens to resemble an old one, and to anything diffuse. *The real fix:* a second decomposition of the same 67M with
another seed. None exists (`artifacts/vpd/t-9d2b8f02` is the base model, not a second decomposition). `GOALS.md` says a
refit is not the goal "unless the experiments demand it"; this is the demand, and the cost (Goodfire's recipe on a 67M)
should be estimated before it is proposed, not here.

**D. Qwen stage 1 (gated, about $5-10 on saved states).** The reset plan's stage 1 as written, with whichever of ratio
and r survived A and B, on the format-distilled anchor, invented facts, gold SFT and own samples. Not before A and B
have answered.

## 3. Recommendation

A first, this week, because both jobs are filed and free; B as E55's labels arrive, the two lines sharing one results
table; C's proxy added to A's read stage for $0 as a reported, unscored column, with the second seed raised as a costed
decision card only if A and B leave the limit as the last objection; D after that. Every write-up and log claim says
"elicited at the level of VPD's pieces" and restates the limit in one sentence.

## 4. The first experiment: E54 as confirmed, plus two columns

*Task pair* (E54, confirmed 2026-10-30): a key → value map the original model already gives at rank 2-20 with probability
in [5e-3, 1e-1], and a random cipher on the same 32 keys, E12's recipe, seeds 0 and 1; the cipher read at the checkpoint
whose ‖δW‖ matches the elicit state's. Same keys, same positions, same format: breadth and input-dependence matched by
construction, size by the checkpoint rule. *Measured:* r (E12's route-split share, scored as filed); the stand-in ratio
(E53's code, reported beside); E52's steering share S (scored as filed); the gating-pattern novelty (reported). *Compute:*
cluster debug L40, two jobs of minutes, $0; E46 rebuilt and read all eight E12 states in 86 s.

| number | W₀ | do-nothing (δW in Δ) | positive control |
|---|---|---|---|
| gain G at the key, held-out | 0 exactly | the full model's G, the same for every placement | ident32_s0 10.11 (E46) |
| r of the least-squares split | undefined (gain 0) | not read; r reads the split state | 0.80 copy / 0.27 cipher ± 0.03 (E46) |
| stand-in ratio | undefined; the zero-update check reads every gain 0 ± 0.001 | undefined (subcomponent gain 0); gate gain g alone is read, as E46 stage 0 | 0.87-0.97 copy / 0.36-0.41 random map (E46) |
| steering share S | undefined; block 0 reads 0 ± 1e-6 | never reads the decomposition | 0.75 copy / 0.16 cipher (E52) |
| gating-pattern novelty | the original model's own masks: distance 0 by construction | reported; expected near the copy state's | copy small, cipher large: no bar, first reading |

*Blind to:* a new circuit from old pieces (reads elicited); a half-known map that the fine-tune nonetheless rebuilds by a
new route (reads new; the label is behavioural). *Noise floor:* E46's ± 0.03 on r across regenerations; the ratio's floor
from the KL-matched random δW as in E53.

## 5. What would change the recommendation

- Amendment 1b's broad cipher reads ≥ 0.7: the ratio reads breadth. Drop the ratio; r and the novelty proxy carry A, and
  B becomes the line's main route.
- E54: r or the ratio separates and S does not → the headline case; go to D with the limit stated. Both separate → VPD
  adds nothing over steering on toys; the line's remaining value is on updates where no single vector exists (Qwen's gold
  SFT), so D goes ahead with S as a required baseline column. Neither separates → the number read input-dependence on
  E12 too; retire it as a reading of kind, keep B.
- B: the ratio disagrees with E55's labels on lowercase or copy → it is not a reading of reuse; say so and stop the
  Qwen read of it.
- A second decomposition seed turns out cheap (under a few GPU-hours): run it before D, since it is the only check that
  narrows the limit.

**Not run.** Nothing; this is advice. Open on the user: whether the novelty proxy is worth a reported column, and whether
a costed card for the second decomposition seed should be written now.
