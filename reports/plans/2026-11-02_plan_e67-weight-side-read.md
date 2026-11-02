# E67: how much do Γ's values change at every position, and does a weight-side share pass the learned-nothing null?

Coded, not run. One free cluster job on saved states, nothing trained. The user approved it in a dialog on 2026-11-02. Design: `reports/plans/2026-11-02_advice_weight-side-split-FABLE.md`. Rules: `experiments/weight_side_read/results/predictions.json`. Class: analysis of saved states, approved class 1; the measure itself is new, and the user approved building it.

## Design check

- **Goal.** The re-gating line, Q5 in `GOALS.md`: what the switching is and whether it can be trusted. It is indirect, since no success criterion reads it. It would give the line a reading with no target token and at every position.
- **Nearest tried.** The route split's r in log-odds, which reads high on a fine-tune that learned nothing (E36). Coverage in the task ladder, and E51's weight statistics, neither of which separated anything. What is new is the unit, and that the read covers every position.
- **Patch or step.** A patch on r's unit. It becomes a step only if it passes the null r failed.
- **Decisions.** It relies on the do-nothing reference and the split vocabulary, and departs from none. Log-odds for the route split has no decision on record, so a row now says so.
- **Result that changes what we do.** The overlap share on the learned-nothing fine-tune against the copy fine-tunes. A keep gives the line a new instrument; a drop sends the idea to `TRIED.md`.
- **Apparatus first.** The control stage runs first and alone, and any miss stops the job before a state is read.
- **The numbers.** Listed below, each with its expected value.

## The request, in the Terms' words

Per matrix, on W₀'s activations a of 128 Pile rows at every position, with Γ the released one:

1. Read Γ's values on W₀ and on W₀ + δW for the same row, g₀ and g₁. Report the mean of |g₁ − g₀| per subcomponent and position, pooled and per matrix.
2. Write the masked output change as three terms and a residual, and report each term's signed overlap share with δW's own output change. The shares sum to 1.

| term | what it is | name in the route split |
|---|---|---|
| D | g₀ (C₁ − C₀) a | direct |
| R | (g₁ − g₀) C₀ a | re-switching, the route `GOALS.md` calls re-gating |
| I | (g₁ − g₀)(C₁ − C₀) a | interaction |
| E | δW a − (g₁C₁ − g₀C₀) a | residual |

The share is S_X = E⟨X, δW a⟩ / E‖δW a‖². Numerators and denominators are summed over positions, then over the 24 matrices for the pooled reading. The states are the do-nothing state and the least-squares split of the same δW. The norm ratio the user first proposed is dropped.

**The user's definition.** The user defined re-gating as "the difference in gating before and after fine-tuning". The words "on the same inputs" were added by the 2026-10-28 plan, and this read keeps them. The mean of |g₁ − g₀| across all positions is the first direct all-position reading of that definition. Earlier reads took the trigger or a cue only (`records.md` section 4). It has no filed bar and no source for one, so **it decides nothing on its own**: it is a reading to look at, beside its seeds and its Gaussian reference.

The fraction of Γ's change on subcomponents that matter, weighted by ‖C₀ a‖, is not reported, because no prediction uses it.

## The numbers and what each should read

"Algebra" follows from the definitions. "Guess" is unmeasured and has no source.

| state | S_D | S_R | S_I | S_E | mean of abs change in g |
|---|---|---|---|---|---|
| W₁ = W₀, control | undefined, numerator exactly 0 | undefined, numerator exactly 0 | undefined, numerator exactly 0 | undefined, numerator exactly 0 | exactly 0 |
| do-nothing, ident32 | 0, algebra | guess 0.2 to 0.6 | 0, algebra | 1 − S_R, algebra | guess 1e-4 to 1e-2 |
| least squares, ident32 | guess 0.1 to 0.5 | same as do-nothing, one pass | guess under 0.1 in size | the rest, checked | same as do-nothing |
| code step 10, code step 100 | guess as copy | guess 0.2 to 0.6 | guess under 0.1 in size | the rest | guess the same order as copy |
| rand32 | guess as copy | guess 0.1 to 0.6 | guess under 0.1 in size | the rest | guess the same order as copy |
| matched Gaussian | guess at most 0.3 | guess 0 to 0.2 | guess about 0 | the rest | guess under copy |

ident32 and rand32 are E12's copy and random-map fine-tunes at step 10, two seeds each. Code steps 10 and 100 are E36's fine-tune on code-heavy Pile rows, which learned nothing. The Gaussian is iid per matrix, seeded, and scaled per matrix so its E‖G a‖² on these rows equals ident32 s0's.

Control values, all from algebra or a filed table:

| control | reads |
|---|---|
| rebuilt student size against E12 Amendment 1's filed `rel_norm_max` | 1, within 1e-4 relative |
| student / teacher size | the table's, 0.87 to 1.35, within 0.006 |
| every gate on, least squares: E‖D a‖² / E‖δW a‖² | 1 − ω − 2 cross + second |
| every gate on, a student's own δW as the update | S_D = 1 |
| shares' sum at every state, both sides | 1, within 1e-4 |

## What each number is blind to

| number | blind to | positive control | noise floor |
|---|---|---|---|
| S_R | behaviour: no token, no loss. The size of R: only its overlap with δW's output. A re-gating change orthogonal to δW reads 0 | Unit test only: a gate moved by 1 on the subcomponent equal to the update reads 1, and on one writing elsewhere reads 0. No saved state is a known re-gating update | The copy seeds, used in the verdict |
| S_D | what the changed subcomponents do once gates move | Every gate on, a student's own δW: S_D = 1, on real a | Same |
| S_I, S_E | which part of the rest they hold. S_E is signed and can exceed 1 | The sum's check | Same |
| mean abs change in g | whether a moved gate matters: a gate on a subcomponent with zero activation counts in full. It is diluted by dead columns, as about three quarters of the 38,912 are dead | W₁ = W₀ reads 0 | Copy seeds and rand32 seeds, reported |

S_R has a finite limit as δW shrinks, because R is first order in δW and so is δW a. A small update does not read small, so the share cannot say "the update was too small to matter". The pooled share is dominated by the matrices with the largest output change. The per-matrix shares are kept as reference.

**How the noise floor is used.** Each copy seed is scored against the code reading separately, and if the two labels differ the verdict is `seeds_disagree`. A spread that matters therefore stops the verdict, and a spread inside one label does not. The spreads are printed. The learned-nothing fine-tune has one seed, so its side has no floor. Its two steps are not seeds. All states read the same rows, so row noise is not in the comparison.

## Choices that shape the reading

**One a per matrix, W₀'s.** All four terms see the same vector, and the denominator is the update's first effect at this matrix. This is blind to what δW does to the matrix's own input. An update upstream moves a, and here that reaches the read only through g₁. It is also not the masked model's own a. Each matrix is read alone, so nothing here adds up to the model's output.

**Pile rows, every position, Γ's trained length.** Γ reads the whole row in both directions, and these rows are its trained distribution. The route split's r and the windows differ in four ways:

- Windows are 41 tokens, off Γ's distribution (`reports/reviews/2026-10-27_review_gamma-bidirectional-windows.md`).
- r is read at the trigger of held-out windows, for a target token. This read has no target.
- The copy fine-tunes trained on trigger positions in windows, and Pile rows hold almost no trigger. So S_R(ident32) reads such a fine-tune off its task, on what it did to ordinary text.
- The code fine-tune trained at every position of Pile rows, so it is read closer to its own training text.

So a keep or a drop here is about Pile text. A drop does not say the same read would fail on windows. "Not planned" lists that read.

**Precision.** Γ and all products run in fp32, with no autocast. The authors' setting is bf16, and fp32 is not a truer value: E12 Amendment 3 moved falls by at most 0.0064 nats. fp32 is needed here because δW is 1 to 2% of W. Differences are also formed from the offsets, (a dV)U₁ + (a V)dU, so nothing subtracts two whole outputs but the one check of E.

## What the advice got wrong, checked against the code

1. **The size ratios.** The advice says the student's full term with every gate on reproduces the 0.87 to 1.35 size ratios. Those ratios are `rel_norm_max`, the largest ‖δW‖/‖W‖ in each state, in Frobenius norm and often on different matrices (`reports/analyses/2026-11-02_analysis_e12-teacher-route-split.md`, section 4). An input-weighted ratio cannot reproduce them. The control splits in two: the Frobenius ratio from saved offsets against the table, and S_D = 1 on real a.
2. **1 − omega_act.** With every gate on, the least-squares state's change is I + Q, with I = δW − O(δW) and Q = dU dVᵀ. So E‖(I + Q) a‖² / E‖δW a‖² = 1 − ω − 2 cross + second, where cross = E⟨I a, O a⟩ / E‖δW a‖² and second holds the Q terms. All of it is computed from the chunk's own Σ (`weight_side.split_identity`), so no tolerance absorbs the cross term. On white inputs the cross term reached 0.009, and real inputs may be larger.
3. **S_E.** The advice defines it by subtraction, which would make the sum's check empty. Here it comes from the whole masked change.

The other claims held: Γ reads only the cached inputs on the 67M, g is the clamp, and the files exist with the sizes given (`ls -lL`, 2026-11-02).

## Order, stops and cost

1. **Control, alone, cheapest first.** Any miss is exit 1, and no state is read.
   - Size: the eight students, no Γ and no a.
   - Zero: W₁ = W₀ on one chunk. Every numerator and |g₁ − g₀| must be exactly 0.
   - Ones: every gate on, one chunk. The least-squares identity on ident32 s0, then S_D = 1 on the students of ident32 s0 and rand32b s0.
2. **States, one read each:** ident32 s0, code step 10, ident32 s1, code step 100, rand32 s0, rand32 s1, Gaussian. The shares' sum is checked at each, and a miss is exit 1. `run.json` is written after each and every number is printed.
3. **Verdict.** A verdict against the lean prints a loud line.

No new fine-tunes. No large file is written but the log, so `results/large` holds `run.out` only, and the job runs `fleet/link-data` if the symlink is missing. The Gaussian is not saved, since the seed and the rows regenerate it.

**Floors behind the tolerances**, measured 2026-11-02 on CPU, with no model: saved ident32 s0 and its student, white a and stub gates, and the ten students' sizes:

| check | largest reading | tolerance |
|---|---|---|
| shares' sum minus 1, per matrix | 2.8e-7 | 1e-4 |
| identity miss, per matrix | 1.6e-7 | 1e-4 |
| student factored against dense, energy | 1.3e-13 | covered by 1e-4 on S_D |
| rebuilt size over filed, on the most changed matrix | 5e-6 relative | 1e-4 |

The margins are wide because real a are not white. The run prints the real readings. One student, rand32b s1, reads 0.2% off on that matrix, presumably as its maximum lies on another. That needs W₀'s norms, so the first minute of the job settles it.

**Cost.** $0, `debug` QoS, 30-minute limit. Arithmetic, not measured: per chunk two fp32 Γ passes of about 2.5 s each, per state about 30 s of products, so about a minute and a half per state, and 10 to 20 min for the job. The first state's `seconds` is the check. A chunk is 16 rows, so a chunk's a and g fill about 2 GB of the card, twice during the second pass.

**Size of the code.** `run.py` is 201 lines and `src/commute/weight_side.py` 220, against the 100 to 150 lines quoted in the dialog. The excess is the controls, the filed verdict and the end-to-end test's seams.

## What changes next, per verdict

The bars are the advice's guesses, marked "chosen here, no source" in the filing. The order of the rule is in `predictions.json`, and `tests/test_weight_side.py` runs every branch.

| verdict | when | next |
|---|---|---|
| keep | code at most 0.1, copy at least 0.4, both seeds and both code steps | a weight-side column for E62 and E58 is proposed to the user. If the Gaussian reads like copy, that goes to the user first |
| keep at step 10 only | keep at step 10, not at step 100 | not kept; the size dependence goes to the user |
| drop | code within 0.15 of copy, both seeds | the share goes to `TRIED.md` as dropped. A flag says if all readings are small |
| between | a gap over the band, but code above 0.1 or copy below 0.4 | not adopted, no new run; the lead sets the numbers beside the floor and decides whether to raise it |
| reversed | code above copy by more than the band | as drop, with a loud report |
| code negative | copy at least 0.4, code below −0.1 | not kept; goes to the user. On a learned-nothing update, R opposing δW is a response and not a reading of nothing |
| seeds disagree | the two copy seeds get different labels | no verdict; the spread is the finding |
| unreadable | a control stopped, or a reading is missing | tell the lead; nothing is read |

Code step 10 decides, as it matches the copy fine-tunes' step. Code step 100 can only take back a keep. The lean is drop, from the advice, at low to medium confidence.

## Edge cases run before filing

| case | result |
|---|---|
| zero denominator, W₀'s 0/0 | every share is None, the numerators are read instead; a None in the verdict is unreadable |
| ties at 0.1, 0.4, 0.15 | count as written, up to 1e-9 for float rounding |
| S_R above 1 | allowed; a copy reading of 2 can keep |
| both negative and apart | between |
| two seeds disagreeing | seeds disagree |
| code far above copy, code far below zero | reversed and code negative, which the advice's two rules leave open or would call keep |
| the gap between the two rules | between |

## Settings with no source

The three bars (0.1, 0.4, 0.15) are the advice's guesses. The code-negative branch, the reversed branch and the Gaussian's "reads like copy" band are chosen here. The rows are the first block of the Pile harness, val rows 0 to 127. The chunk of 16 is `vpd.CI_CHUNK`, and the split tolerance 1e-6 is `anchors.LIFT_TOL`'s. Every guess in the expected-values table is unmeasured.

## Departures from the literature

Γ runs in fp32, where the authors' setting is bf16, for the reason above. No other cited source is used.

## Not planned

- The norm ratio, dropped by the user.
- The same read on task windows, the adjoint split, or the teacher-trained students' own g₁.
- Other fine-tunes: E53, E58, E62, or the lowercase fine-tune that did learn on Pile.
- Weighting Γ's change by ‖C₀ a‖.
- Qwen: no scored state exists, and two g's do not fit the card.
- A figure: `run.py` has no plot yet, and the analysis adds it.
