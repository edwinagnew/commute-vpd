# Independent check of the E67 analysis

I recomputed the report's numbers and the filed verdict from the saved results, read the code against what the report says it measures, and tested each sentence that draws a conclusion. The numbers, the verdict `drop` and the code all hold, but the report's main interpretive sentence does not: being far above one Gaussian draw does not show that the read "is not empty". Its "No" is also stronger than a rule that could not have seen a gap of this size, on rows where no state had learned anything, so the headline should say the rule drops the share and the run leaves the question open.

*2026-11-02, independent checker; I wrote neither the code nor the analysis. No model was loaded and nothing ran on a GPU. Read from commit 0348985, which is on `origin/main` but not yet in this clone's working tree. "L" is a line of `reports/analyses/2026-11-02_analysis_e67-weight-side.md` at that commit, and "nb" is `notebooks/E67_weight_side_read.ipynb`. Scripts and outputs: `~/commute-data/scratch/regating-lead/e67-check/`.*

## The verdict by hand

| pair | gap | filed rule | label |
|---|---|---|---|
| code step 10, copy seed 0 | 0.0250 | at most 0.15 | drop |
| code step 10, copy seed 1 | 0.0112 | at most 0.15 | drop |
| code step 100, copy seed 0 | 0.0341 | can only take back a keep | drop |
| code step 100, copy seed 1 | 0.0203 | can only take back a keep | drop |

The two step-10 pairs agree, so the label is `drop`, as `run.json` says. `drop_both_small` is true, as the three readings are at most 0.0564. `gaussian_like_copy` is true, with a difference of 0.0485. `held_rows_fail` is false, and it could not act on a drop anyway.

## Wrong

**W1. L191, "Dominated by the MLP matrices. The pooled share follows the matrices with the largest output change."** The MLP matrices dominate the denominator only. In the numerator, attention matrices carry 58% and 55% of the copy fine-tunes' pooled share. L136 and L140 already show this, so the sentence contradicts the report.
*Corrected:* "**Weighted by output change.** MLP matrices hold 65 to 78% of the denominator, which pulls the pooled share towards their low readings. Attention matrices still supply 58% and 55% of the copy fine-tunes' numerator, and 32% and 33% of the learned-nothing fine-tune's."

**W2. L80-81, and the two right-hand points of `over_gaussian.png`: held-row readings divided by the random update's reading.** The random update was never read on the 64 rows. The 33, 45, 1.50 and 2.2 divide a reading on one set of rows by a reading on another, and the figure draws the 128-row line across both panels. L118-119's "over the copy mean" column mixes rows in the same way: 0.55 and 0.79 use the 128-row copy mean.
*Corrected:* delete both rows from the L74 table and stop the dashed line at the 128-row panel. At L118-119, leave the column empty for held rows and say: "On the 64 rows, the only same-row ratio is step 10 at 0.69 of copy seed 0."

**W3. L136, "The matrices with the largest output change, the MLP's, read small and alike".** They are alike only after pooling, where opposite gaps cancel. On `h.2.mlp.c_fc` the readings are 0.062 and 0.050 for copy against 0.011 for step 10. On `h.3.mlp.down_proj` they are 0.073 and 0.035 against 0.126, and 0.036 against 0.179 on the held rows. The last is the largest gap in either direction among the MLP matrices, and it runs against the copy fine-tunes. nb's "so they do not separate" has the same fault, and the `TRIED.md` row's "MLP matrices alike" repeats it.
*Corrected:* "Pooled over MLP matrices, the four readings are alike, at 0.023 to 0.031. That hides gaps of opposite sign on single MLP matrices."

**W4. L11, "on 128 Pile rows and 64 code-heavy rows, on five saved do-nothing states".** Four states are saved, as the Gaussian is regenerated from its seed. Only two states were read on the 64 rows. The corrected wording is under O2.

**W5. nb, "so it would fire here whatever the random update did".** It fires only for a random reading from −0.10 to 0.20. L70's "almost regardless" is the right form.
*Corrected:* "so it fires for nearly every reading the filing guessed for a random update."

## Overstated

**O1. L9, "Both sit far above a random update, so the read is not empty"; L52, the gloss "goes too far"; L70, "They are not noise"; and the same sentences in nb.** This is the main finding. The step does not follow, for three reasons.

1. *The Gaussian's low reading is close to forced.* G has zero mean, so for any R that does not depend on G, the expected value of ⟨R, G a⟩ over G is exactly 0. Only the part of R that responds to G can read above 0. A trained update has no such protection: δW a is structured, so a term with no causal tie to this δW can still overlap it. The gap to the Gaussian therefore does not measure how far Γ responds to the update. The run cannot say how much of 0.05 is response and how much is shared structure.
2. *The scale matches one number.* The per-matrix energy match is honest, within 0.19% on every matrix. It does not match the update's shape. By participation ratio of the squared singular values, unweighted by inputs, I measured a median of about 10 for the copy fine-tunes, 67 for step 10 and 128 for step 100. An iid Gaussian reads 384 on the square matrices and 614 on the MLP matrices. So the comparison is fair for what L32 says it rules out, "a share any update of this size produces". It is not fair for the claim that the read carries information.
3. *No positive control.* L187 correctly says there is none on real rows. "Not empty" cannot stand beside that.

The state-level ordering is suggestive and no more. The share falls as participation rises across these four trained states and the Gaussian: 0.056 and 0.043, 0.031, 0.022, 0.001. Within a state the two do not track: Spearman across the 24 matrices is −0.14, 0.06, 0.18 and 0.12. I also looked at the simplest account, that a random update writes outside the subcomponents' span. Earlier reads on other fine-tunes argue against it as a full account of a factor of 50. E12 puts the inexpressible share at about 0.6 of a random update's, `2026-10-27_analysis_broad-fragility.md` reads 0.344 against 0.471, and `2026-10-20_analysis_lora-split.md` reads the write side at 0.97 of random at step 10. None of these was measured on E67's rows, so the cause stays open.

*Corrected, L9:* "Both sit above one random draw, whose reading is near zero almost by construction. So the comparison rules out only a share that any isotropic update of this size would produce. It does not show that the share carries information about switching."

*Corrected, L52:* "Its filed gloss says the read sees nothing on Pile rows. This run neither confirms nor refutes that, as it has no positive control on real rows."

*Corrected, L70:* "The fine-tunes' readings are not row noise: each jackknife standard error is under a tenth of its reading. Seed noise is larger, at 0.0138 across the copy seeds, and is unknown for the fine-tune that learned nothing."

**O2. L9, "No, on these rows and states… It still does not tell learning from not learning"; L11's scope.** Three limits are missing from the answer.

- The rule has no power at this scale. L52 admits that any ordering of these readings lands in the band, so the drop would have fired on a far larger relative gap. The rule's label stands, but it is not evidence of no separation.
- No state in the run learned anything on the rows read. The copy fine-tunes are off task on every row, and the code fine-tune learned nothing anywhere. The plan says so at its line 90: a gap "would show only that the two kinds of update differ on ordinary text". The lowercase fine-tune that did learn on Pile was not read.
- The pooled evidence is mixed rather than null. Copy readings exceed learned-nothing readings in all 8 chunks and all 4, but that is row stability for two seeds. The 0.0112 gap to seed 1 is under the seeds' 0.0138.

*Corrected, L9 and L11:* "The filed rule drops the share, on these rows and states. That is a limit of the test as much as a result: the readings are 0.03 to 0.06 and the band is 0.15, so no ordering of them could have passed. Every update here is read off the text where it learned, so the run does not test learning where learning shows. The pooled gap is about the size of the difference between two copy seeds. Four saved fine-tunes and one random draw were read on 128 Pile rows, and two fine-tunes on 64 code-heavy rows."

**O3. L85-86, "So that something is not learning, and the share cannot be read as a sign of it."** Reading above a random draw is not sufficient for learning, because the code fine-tune does so too. That is all that follows. It does not show that none of the copy fine-tunes' excess comes from learning, and L85's "responds to something they have and the draw lacks" says little beyond "they are not the draw".
*Corrected:* "The fine-tune that learned nothing also reads above the draw, so reading above it is not a sign of learning. Whether any of the remaining gap to the copy fine-tunes follows learning is not tested here."

**O4. L72, L76-79, L89, L219 and the figure's labels, "50 times".** A ratio to a reading near zero is unstable. The chunk statistics do not cover this, as all 8 chunks share one draw. A crude floor treats the 24 matrices' contributions as independent draws: 9 of 24 are negative, and the standard error is about 0.0007 against the reading of 0.0010. That is consistent with zero, and it bounds the ratio only from below, at about 20. L89's "not a fitted ratio" is true but too weak, and `TRIED.md` repeats the 50, 32 and 23.
*Corrected, L72:* "The random reading, 0.0010, is about 0.0007 from zero on a crude estimate from its 24 matrices, with one draw. So the ratios are not stable numbers. Read the differences: 0.03 to 0.06 against about zero."

**O5. L136, L148, L200, nb, and `TRIED.md`: "the pooled gap sits in the attention matrices".** The arithmetic is right, and L136 and L148 flag it as post hoc. Attention matrices contribute 0.0226 of the 0.0250 gap to seed 0, and 0.0131 of the 0.0112 gap to seed 1, with MLP at −0.0019. It repeats on the held rows, at 0.0151 of 0.0114. What it means is overstated in five ways.

- *It is confounded with what was trained.* Copy fine-tunes trained on copying at trigger positions in windows. The code fine-tune trained on running text. Task, data and update shape all differ, there is one seed on the code side, and `rand32`, the saved comparator from the copy recipe, was cut. So it may be the copying task and not learning at all. "The cause is not tested" is too mild, since the sentence's frame invites a reading as a separation.
- *There is no per-matrix noise floor.* `run.json` has chunks for pooled sums only. L130 notes this for Γ's change, but not for per-matrix shares. Readings do move between the two sets of rows: `h.3.attn.o_proj` goes from 0.228 to 0.014, and `h.3.attn.v_proj` from 0.246 to 0.078.
- *"The two seeds agree" (L148) is only partly true.* `h.3.attn.k_proj` reads 0.146 and 0.017, and `h.3.attn.q_proj` 0.170 and 0.049. Pooled attention reads 0.146 and 0.102, and that seed gap of 0.044 is two thirds of the 0.068 between seed 1 and step 10.
- *"The cleanest split is in the key and query matrices of blocks 0 and 1" (L148) is too broad.* `h.0.attn.k_proj` reads 0.016 and 0.044 against −0.019, a gap of 0.035 beside a seed gap of 0.028. Only `h.0.attn.q_proj`, `h.1.attn.k_proj` and `h.1.attn.q_proj` stand clear of the seeds among those four, and they have small denominators. Those four matrices give 0.0114 of the 0.0250 gap to seed 0.
- *Even there, the filed band would still read drop.* Applied to pooled attention, the gaps are 0.112 and 0.068, both under 0.15. "No matrix reaches the keep bar" also applies a pooled bar to single matrices.

*Corrected, L136:* "No filed rule reads this. As arithmetic, most of the pooled gap comes from attention matrices. It cannot be read as learning against no learning: the two kinds of update also differ in task, data and shape, the code side has one seed, and per-matrix shares have no noise floor."

*Corrected, L200:* "**Most of the pooled gap comes from attention matrices,** post hoc, and confounded with the copying task."

*Corrected, L148:* "The clearest gaps are `h.0.attn.q_proj`, `h.1.attn.k_proj` and `h.1.attn.q_proj`, on small denominators, and they repeat on the held rows for the two states read there."

**O6. L122-124, the L111 column head and L54: "output change" is E‖δW a‖², a squared size.** A first-order response of Γ scales with the size, not its square. In size, step 10 and step 100 read 0.65 and 1.46 of copy seed 0, not 0.42 and 2.13. From step 10 to step 100 the size rises 2.24 times, not 5, against 1.41 times for Γ's change. The bracketing argument and the conclusion that it is not proportional both survive, but less sharply. Sublinearity in size also means the share itself may not be scale-free at these sizes: it falls from 0.0314 to 0.0223 between the two steps. These are two different updates, so this is a caution rather than a measurement. The random update is matched to copy seed 0 only, so it is not matched to either code state.
*Corrected, L124:* "From step 10 to step 100, E‖δW a‖² rises by 5 times, so the size of the output change rises by 2.24 times, and the change in Γ's values by 1.41 times." Rename the column "E‖δW a‖² per row".

## Fine

- **Every number checked agrees with `run.json`.** That covers the pooled shares from chunks and from matrices, the spreads, jackknife errors and chunk ranges, all ratios arithmetically, the MLP and attention table, the 144 cells of the per-matrix table to three decimals, the reweighted 0.0321, the 0.39-0.50 and 0.32-0.37 ranges, 893 and 379 per row (`code_held` is 64 by 512), `git_head` and the predictions hash. The log agrees with `run.json`.
- **All chunk-order claims hold.** Each copy reading exceeds each learned-nothing reading in 8 of 8 chunks, and in 4 of 4 on the held rows.
- **The code computes what L15-25 say.** R and δW a follow the stated formula with W₀'s a. g₁ is Γ on the W₁ pass, in fp32. The saved files are updates and not weights, with Frobenius norms of 0.08 to 0.4. Non-square MLP matrices would have raised on a wrong layout. Γ's change is pooled per subcomponent and position.
- **L187, the missing positive control, is honest and complete.** The test exists at `tests/test_weight_side.py:35`. L187's closing sentence is the right warning, and the correction to O1 is what brings L9 into line with it.
- **L52's "the drop is the rule firing on small numbers" and "no ratio rule… is applied after the fact"** are right, as is the decision to keep the filed label.
- **L70's reading of the flag** is right: the filed band runs from −0.10 to 0.20, so it carries nothing here.
- **L128, "a moved Γ is not by itself evidence"**, follows, and it is backed further by the random update moving Γ. L181-183 and L207 follow the filing and carry the granularity limit.
- **L103's noise paragraph** is the most careful in the report. Its "steady on these rows" correctly stops at rows.
- **L126, the flat per-matrix ratios of Γ's change,** is right. Its reading could go further: the ratios are nearly constant for trained and random updates alike, so this mean looks like a property of Γ more than of the update. That is my remark and is not tested.
- **One rounding.** L31's −0.06 comes from −0.065. Write −0.065, or −0.07.

## What would earn O1's sentence

The simplest fix is to cut the sentence, as the filing says no new run follows a drop. If the lead wants to keep any claim of content, two reads of saved files would test it, at about 10 s of GPU per state on `debug`. One pairs R from one fine-tune with δW a from another, which shows how much overlap needs no response to that update. The other keeps a saved update's singular values and draws its singular vectors at random, which is a null of the same shape. A read of the saved `rand32` states would address O5. All three are the lead's call and need `/design`; none ran here.

## Propagation

The `TRIED.md` row from the same commit repeats "MLP matrices alike", the 50, 32 and 23, and the attention remark. Change these with the report. The `PENDING.md` line carries none of the overstated sentences. The research log should not take "not empty".

## Not checked

- The count of 38,912 subcomponents, "3 of the 128" rows, "about three quarters dead", and how few triggers the Pile rows hold: taken from the plan and the code review.
- Job start time 18:48:23: `sacct` returned nothing. The log's stamps run from 18:48:48 to 18:49:50.
- Figures other than `over_gaussian.png`.
- The input-weighted shape of the updates, and the Gaussian's Frobenius norm: both need the input second moments, which are not saved and need the model. The participation ratios above are unweighted.
- `check.py`-level style of the report and notebook.
