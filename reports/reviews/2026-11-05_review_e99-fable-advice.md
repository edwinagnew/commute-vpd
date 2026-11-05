# E99: settling the definition's open details and the constructed fine-tunes (Fable's advice)

*Fable, 2026-11-05, for definition-lead. Read: `decisions.md` rows 81, 115-121, 155, 158; the E90 old-labels analysis and batch decision; the E95 analysis and its check; the E97 plan and its uncommitted results in mechanism-detection-lead's clone; `experiments/TRIED.md` rows E55, E62, E70, E90, E95, E96. Nothing run.*

The four open details each have one recommendation, and the first turns on one number E97 has already produced: the lowercase fine-tune's gain survives the removal of the very pair that carries the original model's own lowercasing, so the definition must say whether a verdict reads the state's fall or the update's. Fine-tunes trained with a located part held out, or allowed to move only inside it, are sound ground truth for "through the part" and are not ground truth for "reuses the old function", so they validate the apparatus's two poles and the old data-built pairs still carry the definition. Five short training arms on the copying pair and the lowercase pair, control first, fit in one or two `debug` jobs at $0.

**Limit on every claim.** A new circuit built from old parts reads as reuse (`decisions.md` row 121).

## The number everything below turns on

E97 reads a removal in the state's own terms, with W₀'s move printed beside and never subtracted (`reports/plans/2026-11-05_plan_e97-locator.md` line 29; `src/commute/locator.py` `removal`). From its `table.json` (mechanism-detection-lead's clone, `experiments/locator_table/results/`, uncommitted):

| Row | Gain G | State's fall, pair off | W₀'s fall, pair off | Own-terms share | Net share (fall − W₀'s fall) / G |
|---|---|---|---|---|---|
| Copy s0-s2, pair (2,3)+(2,4) | 12.65, 12.64, 12.55 | 10.97, 11.12, 9.73 | 1.88 | 0.87, 0.88, 0.78 | 0.72, 0.73, 0.63 |
| Lowercase s0-s1, pair (1,1)+(1,2) | 1.13, 1.09 | 0.90, 0.93 | 1.10 | 0.80, 0.85 | −0.18, −0.16 |

On copy the two quantities agree. On lowercase they say opposite things: the pair carries the original model's lowercasing (W₀'s probability of the target falls from 0.098, and the state's from 0.125 to 0.048, `table.json` `p_mean_w0`, `p_mean`, `p_mean_off`), yet with the pair off in both models the fine-tune's increment is intact, 1.33 log-odds against 1.13 with it on. E96 saw the same shape on 64-token windows, where "about half the best share is W₀ rising" (`reports/analyses/2026-11-05_analysis_e96-locate-mechanism.md` lines 83-85). Note also that E97's nominator failed its stage-1 control (`run.out` line 32: the carry map put `mlp3` first on every copy seed, not (2,4)); the located parts come from the removal sweep on nominated parts and the greedy pair, which is fine, but the tree's fallback (a direct-effect read) is owed in E97's amendment, not E99's.

## (1) Fixed-answer tasks

**Options.** (a) Keep "does not apply" with the two behavioural numbers beside (row 121). (b) Let a located part give a verdict, reading the state's own fall (E97's rule): lowercase then reads "reuses C" at 0.80-0.85 against a spared cipher at 0.07. (c) Let a located part give a verdict, reading the net share (the increment with the part off in both models), with the own fall and W₀'s fall beside: lowercase reads "not through the part" at −0.18, −0.16.

**Recommendation: (c), with an admissibility clause that keeps the capability a mechanism and not the target.** The definition's object is the update (row 81: fine-tuning reuses C if the post-trained map is g∘f_C). A wrapper on f_C vanishes when f_C does, so state-with-part-off should equal W₀-with-part-off and the net share read near 1. Lowercase reads the opposite: the increment is additive to the part, so it is not a wrapper on what that pair computes. Reading the state's own fall (b) would call "reuses C" any fine-tune of a target whose prior already lives in the part, which is "counting the model's own output as a capability" by another route, the thing the user declined in ask 21. The net share is readable here because W₀'s probability is 0.098 on whole rows (E96's objection to net shares was for W₀ near 0.001, on 64-token windows).

Wording for the definition: *a fixed-answer task gets a verdict only through a located part P of W₀ such that (i) removing P costs W₀ its own leaning to the target above the random-pair band on the task's windows (the C-map: E97's `w0_move` 1.10 against bands 0.09-0.10 passes), and (ii) P's function at W₀ is input-dependent on natural text, read on inputs the task never trained on (for lowercase: W₀'s lowercase-minus-capitalised log-odds with P off moves on cased rows as well as lowercased ones, E96's F1 unit). The verdict reads the net share: "reuses C" above the band with the control spared; "not through C" within it; the own fall and W₀'s fall go beside.* Clause (ii) is what stops "the model's own output" from being the capability: C is named by what it does on other inputs, not by the word it emits. Lowercase would then read "not through (1,1)+(1,2)" if (ii) holds and stay "does not apply" if it does not; either is a reading, not an assumption.

**What would change it.** If the forced-part lowercase arm below cannot push the net share to 1 (the quantity has no positive control on this task), fall back to (a). If the removal of P changes the gain's sign on the cipher control, mean ablation of a pair injects too much on whole rows and the quantity needs E96's natural-text means beside.

## (2) Abilities whose removal only says "needs the input token"

E95's hand reader placed after the removal lost 0.74 and 0.77 of its gain against the states' 0.50-0.53 and 0.36-0.37 (`reports/reviews/2026-11-05_review_e95-independent-check.md` lines 17-20). The rule that turns that into admissibility, so it can be tested:

*C is admissible at a site only if (a) f_C merges inputs on X_C, at least two keys per value on average and more than one value, and (b) a reader of the task's answers that needs nothing but the input token, placed downstream of the site and matched to the state's gain, stays within the random band under C's removal.*

(a) costs nothing and removes token identity (32 values on 32 keys) and the one-value case (E90's A5) at once; W₀'s guess (18 values on 24 keys), the count, the case and the class map (4 classes) pass. (b) is E95's apparatus made a rule: token identity at block 1 fails it (0.74 against bands 0.41-0.47), so the 32-key, 24-key and replicate copy maps read "no admissible C at this site: the removal deletes the input", and the E95 numbers are their reason. Admissibility runs before the band (see (3)): with per-state bands the random map sits above its own (0.370 against 0.338, `reports/analyses/2026-11-05_analysis_e90-old-labels.md` line 141), and only (b) stops a by-construction new task from reading "reuses token identity".

**What would change it.** A C that passes (a) and fails (b) says the site, not C, is the problem: move the removal later or read a located part instead. A reader that cannot be matched to the gain voids (b) at that site; say so rather than pass by default.

## (3) The band: per state

Per state, with E95's paired test (state minus random partner under each draw) for any comparison between states. The null is "does this state lose more than a matched random removal costs this same state", and the largest-over-states band let the copy map's draws decide the random map's row (`...e90-old-labels.md` line 141; the batch already asks for both, `reports/decisions/2026-11-05_decision_e90-definition-batch.md` line 47). What would change it: if the per-state band's spread under redraws (the E90 check simulated 20,000 sets) exceeds the state-to-state gap, pool to the largest; the check's 98.9% says it does not on the rows read so far.

## (4) Unread rows

Rebuild the class map and its shuffled twin (11 minutes, `reports/plans/2026-11-05_plan_e90-new-vs-reuse-definition.md` line 149) and read them under (2)-(3) with the hand write of the class table as the control: it is the one old task whose C is neither the answer nor the token, so it is the best old-task test the value form has. The copy tables without their own reader need none: rule (2b) closes them with E95's numbers. What would change it: the class map failing (2b) on its own site, which would say the value form has no admissible site on this model at all, a result worth one line in the write-up.

## The constructed fine-tunes

**Are they ground truth?** For "through the part", yes; for "reuses C" in the definition's sense, no, and the verdict's wording should say which. A fine-tune allowed to move only inside the pair can repurpose it (a changed attention pattern or OV map is a new function in an old part), and a fine-tune trained with the pair held at its mean can rebuild the same function in other heads: E62 Amendment 5 found W₀'s copying at the second A "thinned" over (2,4), (3,4) and (2,3) at 0.19-0.36 each (`experiments/TRIED.md` line 140), so a forbidden copy arm has copying heads to reroute through. Both are the elicitation granularity limit (row 121) made concrete. So: *forced* and *forbidden* fix the two poles of the mechanism form's scale and prove the quantity can read 1 and 0 on these windows; the pairs whose truth comes from the data (copy against lookup; the class map against its twin; P against N on the counting grammar) remain the only truths about reuse of a function. Combine them: "reuses C" needs the net share above the band *and* the part on the C-map; a forced arm shows the first half can be met by construction, which is why it is a control and not a finding.

**Tasks and parts.** The copying pair, C located at (2,3)+(2,4) (E97 `run.out` line 30), recipe Adam 3e-5, batch 32, KL 1, clip 1, read at step 30 (`reports/plans/2026-11-02_plan_e62-copying-heads.md` line 213; six states took about 3.5 min, TRIED line 136). The lowercase pair, part (1,1)+(1,2), E53's recipe at 300 steps on whole rows (TRIED line 112; E97's whole-row reads took 1282 s for 60 parts on four states, `run.out` line 51, so one to three parts per state is a minute). Leave the counting grammar out of the first pass: its C is a direction whose removal is not matched on size (E70 Amendment 1, TRIED line 163), so a forbidden arm there would inherit that argument; add it only if the two 67M pairs read cleanly.

**Mechanics.** Forced-part: train with every matrix live and, in `_run`'s `on_step`, reset every entry outside the pair's q, k, v row slices to W₀ (the slices `locator.head_reverted` names), with o_proj frozen so that the ablated state equals the ablated W₀ exactly and the net share is 1 by construction, like E97's stage-0 plants. Forbidden-part: wrap the training forward in `copying.ablated` with W₀'s means on the task's windows, and exclude the pair's four slice sets from the update, so the pair is exactly W₀ afterwards. Matched gain: steps until G is within 2% of the reference state's (12.6 or 1.13), cap at ten times the reference's steps; a cap reached is a reading ("cannot learn without the part at this budget"), not a failure.

**Minimal arms and what each should read** (net share at the part; own fall and W₀'s fall beside; two seeds each):

| Order | Arm | Expected | Stop or branch |
|---|---|---|---|
| 0 | W₀ only: (1,1)+(1,2)'s effect on the lowercase-minus-capitalised gap on cased and lowercased rows | moves on both (clause (ii)) | if not, lowercase stays "does not apply" and arms 3-4 are dropped |
| 1 | Forced-part copy, first and alone | net 1.00 ± 0.05; lookup control from E62's states spared, about 0 | outside ± 0.05: code, stop. Not learned within 300 steps: the construction fails on this pair, report |
| 2 | Forbidden-part copy, read with the pair restored | net within the band at (2,3)+(2,4); G restored within 10% of G native | beside: all prefix-scored heads off (E62's set with (3,4), (2,3)); a fall there reads "rerouted within in-context copying" |
| 3 | Forced-part lowercase | net 1.00 ± 0.05 at 300 steps, or not learned | not learned: the net quantity has no positive control on lowercase, fall back to (1a) |
| 4 | Forbidden-part lowercase | learns as fast (G within 10% at 300 steps); net within the band | learns slower by more than 2×: the part matters for learning, not for the increment; say so |
| 5, optional | Forced-part lookup | net 1 if learned: the granularity limit as a number | not learned at 10×: the pair cannot carry a lookup, also worth a line |

Arm 1 is the W₀-type control: it can only fail by code or by the pair being unable to learn, and both are found for $0 before anything is read. Arm 2 is the one that can surprise: if restoring the pair changes the state's gain by more than 10%, the new route reads the residual the head writes into, and "not through the part" must be read at the native condition with the restored one beside.

**What could make forbidden still route through C.** Rerouting within the capability (guarded by the wider-set read); the pair's o_proj slices learning a bias on a constant input (guarded by excluding them from the update); the new route depending on the restored head's output (guarded by reading both conditions). **What makes forced not truly forced.** Repurposing inside the part (guard: W₀'s second-copy top-1 and the pair's prefix score, `copying.induction` and `stripe_score`, re-read with the state's pair; a fall below E62's bar reads "a new function in the old part"), and too little capacity in two heads' q, k, v (then the arm does not learn and says so).

## Not run and not proposed

Restricting an update to read C's subspace (no clean implementation at matrix grain, and the forbidden arm already gives the "not through" pole); a value-form forced arm; anything on Qwen.
