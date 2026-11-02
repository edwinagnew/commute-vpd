# E67: how much do Γ's values change at every position, and does a weight-side share pass the learned-nothing null?

Coded, not run. One free cluster job on saved states, nothing trained. The user approved it in a dialog on 2026-11-02, and it was narrowed after code review the same day. Design: `reports/plans/2026-11-02_advice_weight-side-split-FABLE.md`. Review: `reports/reviews/2026-11-02_review_e67-code.md`. Rules: `experiments/weight_side_read/results/predictions.json`. Class: analysis of saved states, approved class 1; the measure itself is new, and the user approved building it.

## Changes after review, 2026-11-02

Nothing had run, so the filing was edited in place. **The job now reads the do-nothing side only.** The split side is added only after a keep, because a drop, the lean, never reads it. The overlap share of re-switching and the change in Γ's values stay, on the learned-nothing fine-tune against the copy fine-tunes, and the zero control stays first. **The learned-nothing fine-tune is also read on the rows where r failed**, and a reading over 0.1 there takes back any label that would adopt the read.

| review item | what it became |
|---|---|
| Finding 1, the null read on other text | Code step 10 and ident32 s0 are read on `code_held.pt`, sha checked. The rule is filed in "What changes next". The claim that the code fine-tune is read close to its training text is corrected below |
| Finding 2, no floor on the code side | Each chunk's pooled numerator and denominator are saved, and each share gets a leave-one-chunk-out spread, on the held rows too. Report-only; the bars do not change. "Row noise is not in the comparison" is corrected below |
| Finding 3, a dead read lands on the lean | Exit 1 with verdict `dead_read` when the mean of \|g₁ − g₀\| is exactly 0 on a nonzero update. Plan and filing state that there is no positive control on real rows |
| Finding 4, `seeds_disagree` on a negligible spread | Labels that differ across seeds with a spread under 0.1 read `between`. A spread of exactly 0.1 is not under the bar, and a spread of 0 cannot give differing labels. Tests cover each |
| Finding 5, the size control's open stop | Gone with the size stage |
| Finding 6, the time limit | `--time=00:45:00` |
| Finding 7, the loud line | Fires on any verdict but `drop` |
| Finding 8, whether the code was committed | `git status --porcelain` of the two code paths is in `run.json` |
| Cut 1, do-nothing only | Done. `Offsets`, `offsets_from`, `changed`, `split_identity`, `rel_norm_max`, the split sums, `stage_size`, `stage_ones` and the job's split inputs are gone, as are S_D, S_I and S_E as reported numbers |
| Cuts 2, 3, 4, 8 | Inside Cut 1 |
| Cut 5, `second_moments` | Done. The Gaussian's scale uses `split.input_second_moments` |
| Cut 6, dead None and NaN branches | Done, with one change. The review counted on the shares' sum to stop a non-finite reading, and that check is gone: on the do-nothing side it holds by algebra. A one-line stop on a non-finite S_R replaces it, so the verdict still sees floats only |
| Cut 7, `rand32_s0`, `rand32_s1` | Done |

## Design check

- **Goal.** The re-gating line, Q5 in `GOALS.md`: what the switching is and whether it can be trusted. It is indirect, since no success criterion reads it. It would give the line a reading with no target token and at every position.
- **Nearest tried.** The route split's r in log-odds, which reads high on a fine-tune that learned nothing (E36). Coverage in the task ladder, and E51's weight statistics, neither of which separated anything. What is new is the unit, and that the read covers every position.
- **Patch or step.** A patch on r's unit. It becomes a step only if it passes the null r failed.
- **Decisions.** It relies on the do-nothing reference and departs from none. Log-odds for the route split has no decision on record, so a row now says so.
- **Result that changes what we do.** The overlap share on the learned-nothing fine-tune against the copy fine-tunes. A keep gives the line a new instrument; a drop sends the idea to `TRIED.md`.
- **Apparatus first.** The control runs first and alone, and a miss stops the job before a state is read.
- **The numbers.** Listed below, each with its expected value.

## The request, in the Terms' words

Per matrix, on W₀'s activations a at every position, with Γ the released one, on the do-nothing state, C₁ = C₀:

1. Read Γ's values on W₀ and on W₀ + δW for the same row, g₀ and g₁. Report the mean of |g₁ − g₀| per subcomponent and position, pooled and per matrix.
2. Report the signed overlap share of the re-switching term R = (g₁ − g₀) C₀ a with δW's own output change: S_R = E⟨R, δW a⟩ / E‖δW a‖². This is the route `GOALS.md` calls re-gating. Numerators and denominators are summed over positions, then over the 24 matrices for the pooled reading.

The rows are 128 Pile rows, and then E36's 64 rows where r failed. On do-nothing the direct and interaction terms are zero by algebra and the residual is everything else, so none is reported. The norm ratio the user first proposed is dropped.

**The user's definition.** The user defined re-gating as "the difference in gating before and after fine-tuning". The words "on the same inputs" were added by the 2026-10-28 plan, and this read keeps them. The mean of |g₁ − g₀| across all positions is the first direct all-position reading of that definition. Earlier reads took the trigger or a cue only (`records.md` section 4). It has no filed bar and no source for one, so **it decides nothing on its own**: it is a reading to look at, beside its seeds and its Gaussian reference.

The fraction of Γ's change on subcomponents that matter, weighted by ‖C₀ a‖, is not reported, because no prediction uses it.

## The numbers and what each should read

"Algebra" follows from the definitions. "Guess" is unmeasured and has no source. Every line but the first two is the do-nothing state.

| state | rows | S_R | mean of abs change in g |
|---|---|---|---|
| W₁ = W₀, control | one chunk of the 128 | undefined, numerator and denominator exactly 0 | exactly 0 |
| W₁ = W₀, not run | the rows where r failed | undefined, numerator and denominator exactly 0 | exactly 0, the same pass on other rows |
| ident32 s0, s1 | the 128 | guess 0.2 to 0.6 | guess 1e-4 to 1e-2 |
| code step 10, code step 100 | the 128 | guess 0.2 to 0.6 | guess the same order as copy |
| matched Gaussian | the 128 | guess 0 to 0.2 | guess under copy |
| code step 10 | the rows where r failed | guess 0.2 to 0.6, as on the 128 or above | guess the same order or above |
| ident32 s0, report-only | the rows where r failed | guess 0.2 to 0.6 | guess the same order |

The leave-one-chunk-out spread is a guess of under 0.1 on the 128 rows, and larger on the 4 chunks of the held rows.

ident32 is E12's copy fine-tune at step 10, two seeds. Code steps 10 and 100 are E36's fine-tune on code-heavy Pile rows, which learned nothing. The Gaussian is iid per matrix, seeded, and scaled per matrix so its E‖G a‖² on the 128 rows equals ident32 s0's.

## What each number is blind to

| number | blind to | positive control | noise floor |
|---|---|---|---|
| S_R | behaviour: no token, no loss. The size of R: only its overlap with δW's output. A re-gating change orthogonal to δW reads 0 | **None on real rows, only the unit test:** a gate moved by 1 on the subcomponent equal to the update reads 1, and on one writing elsewhere reads 0. No saved state is a known re-gating update | The copy seeds, used in the verdict, and the leave-one-chunk-out spread, reported |
| mean abs change in g | whether a moved gate matters: a gate on a subcomponent with zero activation counts in full. It is diluted by dead columns, as about three quarters of the 38,912 are dead | W₁ = W₀ reads 0 | The copy seeds, reported |

The zero control and the dead-read stop show that the read is live, not that it reads correctly. Any report of this run says so, a drop above all, since a wrong g₁ that still moves could also read as the lean.

S_R has a finite limit as δW shrinks, because R is first order in δW and so is δW a. A small update does not read small, so the share cannot say "the update was too small to matter". The pooled share is dominated by the matrices with the largest output change. The per-matrix shares are kept as reference.

**How the noise floor is used.** Each copy seed is scored against the code reading separately. If the two labels differ, the verdict is `seeds_disagree`, unless their spread is under 0.1, which reads `between`. A spread inside one label does not stop the verdict. The learned-nothing fine-tune has one seed, and its two steps are not seeds. All states read the same rows, which removes only the row noise the two updates share. S_R is a ratio of sums, so a few rows can drive it, and by a different amount for each update. So each chunk's pooled numerator and denominator are saved, and each S_R prints its spread with one chunk left out. This gives both sides a floor to look at, and it decides nothing: the verdict keeps its bars.

## Choices that shape the reading

**One a per matrix, W₀'s.** The denominator is the update's first effect at this matrix. This is blind to what δW does to the matrix's own input. An update upstream moves a, and here that reaches the read only through g₁. It is also not the masked model's own a. Each matrix is read alone, so nothing here adds up to the model's output.

**Pile rows, every position, Γ's trained length.** Γ reads the whole row in both directions, and these rows are its trained distribution. The route split's r and the windows differ in four ways:

- Windows are 41 tokens, off Γ's distribution (`reports/reviews/2026-10-27_review_gamma-bidirectional-windows.md`).
- r is read at the trigger of held-out windows, for a target token. This read has no target.
- The copy fine-tunes trained on trigger positions in windows, and Pile rows hold almost no trigger. So S_R(ident32) reads such a fine-tune off its task, on what it did to ordinary text.
- The code fine-tune trained at every position of code-heavy Pile rows, and the 128 rows are mostly not that text: by E36's own rule, 3 of the 128 reach its code-share cut of 0.305, and the mean share is 0.082.

So on the 128 rows the verdict compares two off-task updates. A gap there would show only that the two kinds of update differ on ordinary text.

**The rows where r failed.** r failed the null on `code_held.pt`, 64 held-out rows at or above that cut. Code step 10 and ident32 s0 are read there too. A label that would adopt the read must also hold there, so a keep says the read passes r's null. A drop stands by itself: a read that fails on the 128 rows is not adopted whatever it does on the code rows. Only code step 10's reading is ruled on. ident32 s0's sits beside it, as the copy fine-tunes are off task on these rows too. A drop does not say the same read would fail on windows. "Not planned" lists that read.

**Precision.** Γ and the products run in fp32, with no autocast. The authors' setting is bf16, and fp32 is not a truer value: E12 Amendment 3 moved falls by at most 0.0064 nats. fp32 is needed here because δW is 1 to 2% of W. The Gaussian's scale takes its second moments from `split.input_second_moments`, under bf16. Both energies in that scale use the same moments, so the rounding moves nothing a rule reads.

## What the advice got wrong, checked against the code

Both corrections belong to the split side. No control runs them now, and they stand for when that side is added.

1. **The size ratios.** The advice says the student's full term with every gate on reproduces the 0.87 to 1.35 size ratios. Those ratios are `rel_norm_max`, the largest ‖δW‖/‖W‖ in each state, in Frobenius norm and often on different matrices (`reports/analyses/2026-11-02_analysis_e12-teacher-route-split.md`, section 4). An input-weighted ratio cannot reproduce them.
2. **1 − omega_act.** With every gate on, the least-squares state's change is I + Q, with I = δW − O(δW) and Q = dU dVᵀ. So E‖(I + Q) a‖² / E‖δW a‖² = 1 − ω − 2 cross + second, where cross = E⟨I a, O a⟩ / E‖δW a‖² and second holds the Q terms. On white inputs the cross term reached 0.009, and real inputs may be larger.

The other claims held: Γ reads only the cached inputs on the 67M, g is the clamp, and the files exist with the sizes given (`ls -lL`, 2026-11-02).

## Order, stops and cost

1. **Control, alone.** W₁ = W₀ on one chunk. Every numerator and |g₁ − g₀| must be exactly 0. A miss is exit 1, and no state is read.
2. **States, one read each:** ident32 s0, code step 10, ident32 s1, code step 100, Gaussian. Then code step 10 and ident32 s0 on the rows where r failed. `run.json` is written after each and every number is printed. Exit 1 on a dead read or a non-finite S_R.
3. **Verdict.** Any verdict but drop prints a loud line.

No new fine-tunes. No large file is written but the log, so `results/large` holds `run.out` only, and the job runs `fleet/link-data` if the symlink is missing. The Gaussian is not saved, since the seed and the rows regenerate it.

**Cost.** $0, `debug` QoS, 45-minute limit, as there is no resume and `run.json` blocks a rerun. Arithmetic, not measured: per chunk two fp32 Γ passes of about 2.5 s each, and per state about 10 s of products, so about a minute per state of 8 chunks and half that on the 4 chunks of the rows where r failed. With the model load, five sha256 reads and the second moments, that is 5 to 10 min for the job. The first state's `seconds` is the check. A chunk is 16 rows, so a chunk's a and g fill about 2 GB of the card, twice during the second pass.

**Size of the code.** `run.py` is 146 lines and `src/commute/weight_side.py` 146, against the 100 to 150 lines quoted in the dialog and the review's estimate of about 250 in all. The excess over that estimate is the rows where r failed, the leave-one-chunk-out spread and the two stops.

## What changes next, per verdict

The bars are the advice's guesses, marked "chosen here, no source" in the filing. The order of the rule is in `predictions.json`, and `tests/test_weight_side.py` runs every branch. "Held" means code step 10's S_R on the rows where r failed. A tie at 0.1 is read as not over.

| verdict | when | next |
|---|---|---|
| keep | code at most 0.1, copy at least 0.4, both seeds and both code steps, and held at most 0.1 | the split side is added and a weight-side column for E62 and E58 is proposed to the user. If the Gaussian reads like copy, that goes to the user first |
| keep at step 10 only | keep at step 10, not at step 100, and held at most 0.1 | not kept; the size dependence goes to the user |
| drop | code within 0.15 of copy, both seeds, whatever held reads | the share goes to `TRIED.md` as dropped, and the split side is never added. A flag says if all readings are small |
| between | a gap over the band, but code above 0.1 or copy below 0.4; or differing labels on a seed spread under 0.1; or, with flag `held_rows_fail`, a label that would adopt the read with held over 0.1 | not adopted, no new run; the lead sets the numbers beside the leave-one-chunk-out spreads and decides whether to raise it |
| reversed | code above copy by more than the band | as drop, with a loud report |
| code negative | copy at least 0.4, code below −0.1 | not kept; goes to the user. On a learned-nothing update, R opposing δW is a response and not a reading of nothing |
| seeds disagree | the two copy seeds get different labels, with a spread of at least 0.1 | no verdict; the spread is the finding |
| dead read | the mean of \|g₁ − g₀\| is exactly 0 on a nonzero update; exit 1 | tell the lead; nothing is read |
| unreadable | the control stopped, or an S_R is not finite; exit 1 | tell the lead; nothing is read |

Code step 10 decides, as it matches the copy fine-tunes' step. Code step 100 can only take back a keep. The lean is drop, from the advice, at low to medium confidence.

## Edge cases run before filing

| case | result |
|---|---|
| zero denominator, W₀'s 0/0 | the control reads numerators and forms no share; on a state it cannot occur and would exit 1 |
| a reading that is not finite | exit 1 before the verdict, so the verdict's pool holds floats only |
| ties at 0.1, 0.4, 0.15 | count as written, up to 1e-9 for float rounding |
| S_R above 1 | allowed; a copy reading of 2 can keep |
| both negative and apart | between |
| two seeds disagreeing, spread of 0 | cannot happen: one code reading and one copy reading give one label |
| two seeds disagreeing, spread under 0.1 | between. Keep against drop cannot land here, as that spread is at least 0.15 |
| two seeds disagreeing, spread at 0.1 or over | seeds disagree |
| code far above copy, code far below zero | reversed and code negative, which the advice's two rules leave open or would call keep |
| the gap between the two rules | between |
| held exactly 0.1, or negative | not over, so the label stands; the rule is one-sided |
| a drop, reversed or seeds-disagree label with held far over 0.1 | the label stands |

## Settings with no source

The three bars (0.1, 0.4, 0.15) are the advice's guesses, and the 0.1 on held rows and on the seed spread are the lead's, after review. The code-negative branch, the reversed branch and the Gaussian's "reads like copy" band are chosen here. The rows are the first block of the Pile harness, val rows 0 to 127, then E36's held rows as saved. The chunk of 16 is `vpd.CI_CHUNK`. Every guess in the expected-values table is unmeasured.

## Departures from the literature

Γ runs in fp32, where the authors' setting is bf16, for the reason above. No other cited source is used.

## Not planned

- The split side, until a keep: the least-squares state, S_D, S_I, S_E, every-gate-on controls, students' sizes.
- rand32, cut after review: no rule read it.
- The norm ratio, dropped by the user.
- The same read on task windows, the adjoint split, or the teacher-trained students' own g₁.
- Other fine-tunes: E53, E58, E62, or the lowercase fine-tune that did learn on Pile.
- Weighting Γ's change by ‖C₀ a‖.
- Qwen: no scored state exists, and two g's do not fit the card.
- A figure: `run.py` has no plot yet, and the analysis adds it.
