# Independent check of E80, VPD's adversary on four Qwen3-1.7B states (2026-11-04)

The filed outcome, "D1 has a test and a first candidate", computes correctly and I found no bug in the code paths I read, but the numbers do not support its plain meaning. Do-nothing itself reads under the bar that passes in-span, in-span and A read over the bar that fails do-nothing, and the gaps between the three changed states at the scored read are smaller than the gap between two starts of the unchanged decomposition. E80 shows that the attacker reads do-nothing above the unchanged decomposition on one start, and it shows no state to be a good placement.

By a fresh Opus agent that wrote none of this. Scripts and outputs: `~/commute-data/scratch/qwen-decomp-lead/e80-check/` (`recompute.py`, `gamma_round.py`, each with `.out`). States: W0, the unchanged decomposition; N, do-nothing (the whole update in Δ); I, in-span (the least-squares split's in-span change, remainder discarded); A, E76's clean-term state. M is 64 maths rows of 256 tokens, W is 128 FineWeb rows of 512. r is a state's seed 0 over W0's seed 0, on M.

## The numbers against every bar

| state | adversary KL on M | r | over 1 + b = 1.19? | under pass bar 1.40? | r over W0's seed 1 | r at zero attack (step 0) | self-faithfulness over W0's |
|---|---|---|---|---|---|---|---|
| W0, seed 0 / seed 1 | 3.999 / 3.443 | 1 / 0.861 | | | 1.162 / 1 | 1 | 1 |
| N | 5.268 | 1.317 | yes | **yes** | 1.530 | 1.443 | 1.55 |
| A | 5.011 | 1.253 | **yes** | yes | 1.456 | 1.422 | 1.51 |
| I | 4.787 | 1.197 | **yes** | yes | 1.391 | 1.344 | 1.43 |

Same rows, W: W0 1.752, N 2.246, r 1.282; r at zero attack 1.005, self-faithfulness ratios 1.01 for N, A and I. I and A were not read on W by the adversary. N's and A's self-faithfulness are E79's (bf16 components), W0's own there 0.0427; I's is this job's, W0's own 0.0433.

## Findings

### High: the label overstates the result

1. **The rule uses two bars, and the outcome comes from that.** N "fails" because r is over 1.19; I "passes" because r is under 1.40. Read by one bar, they get one verdict: under 1.40 N passes too, and over 1.19 I and A fail too. Only a bar between 1.197 and 1.317 splits N from I. None was filed, and that window (0.12) is narrower than W0's start-to-start spread (0.16). The rule's code does what was filed (`rules.py:61-80`); the filing never required do-nothing to read above the pass bar. On the 67M do-nothing read 4.79 and 2.59, above the fail bar, so this overlap never arose there.
2. **At the scored read, the differences between changed states are under start-to-start spread.** N over I 1.100 (0.48 nats), N over A 1.051 (0.26), A over I 1.047 (0.22); W0's seed 0 over seed 1, 1.162 (0.56 nats). Treating starts as independent, with the log spread pooled from M's two and the 128-row job's three starts (0.10), N minus I is 0.66 standard errors, while N, A and I over W0 are 2.8, 2.4, 2.0 (3 degrees of freedom, a rough guide, since the spread of N, I and A is unread).
3. **On M, most of the ratio is there before the attacker moves.** At the first step, with random sources and no attack, r already reads 1.44, 1.42, 1.34, all over the bar of 1.19; self-faithfulness under random masks, also no attacker, reads 1.55, 1.51, 1.43. On W the same reads are 1.00-1.01. So N, I and A share a raised level on M, whatever the placement, and r(I) is lower at the end of the attack than at its start. The denominator is not like the numerators: W0 is read against W0's clean logits, the others against logits at or near FT's. M is W0's own generated answers (`qwen_sft/run.py:838-846`), where W0's clean entropy is 0.58 nats against FT's 1.53, with KL(FT ‖ W0) 1.84. I name that as the candidate only; the cause of the raised level is not measured. "Do-nothing fails on M" therefore cannot be read as "the attacker found the update in Δ".
4. **No state known to be bad, other than N, was read, and the one state with bad weights passes.** A sits further from FT's weights than the unmoved components (error 1.36 against 1) and reads r 1.253; I's error is 0.09 and it reads 1.197. A weight error 15 times larger moves r by 5%. `predictions.json` says so itself ("not_a_positive_control"). B, L, E79's permuted P and `qwen_maps`' random state exist and were not read. The clean-near-FT wrong placement that a pass test most needs was never tried here.
5. **The bars are borrowed with the 67M's noise.** 1.40 and 1.61 are 1.5 / 1.075 and 1.5 x 1.075, with 7.5% the 67M's H100 floor (`scorecard_ls/results/predictions.json:3`). Qwen's start-to-start spread is 16-19%. The same formula at 1.19 gives 1.26 and 1.785: I and A still pass (A by 0.007) and N is "no call". `.claude/CLAUDE.md` asks that a filed band be set from the control and null in the same job. The 67M also does not suggest I's pass generalises: on its full-matrix Adam update the expressible part read 6.85, above do-nothing's 4.79, and δW here is a full-matrix Adam update.

### Medium: how firm the numbers are

6. **The filed ratio is one point on a sawtooth.** W0 seed 0's final read is a spike: its last three reads are 3.59, 3.34, 4.00. r(N) over the last ten reads runs 1.32 to 1.90, and the scored 1.317 is its lowest since step 3. Had the read been one step earlier: N 1.585, A 1.510 (no call), I 1.301. Denominator matters too: over W0's seed 1, A is "no call" (1.456) and I passes by 0.009; over the two starts' mean, N 1.416 "no call", A 1.347, I 1.287. W0 seed 0 is the larger start, so this denominator was kind to all three. N clears 1.19 at every denominator and never reaches 1.61 at any.
7. **In fairness to the filed reading, the traces hint that N sits above I.** This is post hoc. Over all 21 reads of one paired start N is above I and above A every time, while W0's two starts swap (13 of 21). Geometric mean of the last five reads: N 1.51, A 1.43, I 1.26 over W0's seed 0 (1.50, 1.42, 1.26 over seed 1), W0's two starts 0.996, N over I 1.20. Over the last ten reads r(I) never reaches 1.61, N does five times, A three. The same seed across cards tracks within about 1% to step 17 and 2.9% at the final read (1.7517 against 1.7016), so rounding is not the source of this ordering. The reads of a start are not independent, so this is one observation per state. It is worth a second paired start, and it does not rescue the label: on M the shared level in finding 3 is still in it.
8. **W is the cleanest evidence and is half read.** On W there is no raised level at zero attack (r 1.005), and r grows under attack, peaking at 1.54 at step 10, then 1.28 at the end (1.28 over the last five reads). That is the one place the attacker visibly gains from δW in Δ. N's 2.246 is over all three starts of W0 on W (1.31-1.56 x). But it is one start, r is not settled in steps, and without I and A on W it says nothing about telling placements apart.
9. **"Under the state's Γ" is only partly true for moved V.** Γ reads V rounded to bf16 (`qwen_vpd.py:249`). On seven sampled sites of A, rounding V and V + dV leaves 39-76% of entries with no visible change, the visible change differs from dV by 0.78-2.3 x dV's size, cosine 0.40-0.79. So the readouts see a noisy dV for A; N has no dV. JAX's Γ does the same, so this is not a departure, but it is an asymmetry between states whose effect on g was not measured. I's dV was not measured: X is not kept.
10. **A ratio on these rows is not on E71's scale.** W0 reads 4.0 here, 1.75 on W, 0.69 on the 67M, about 17 on four rows: the shared source makes the denominator depend on row count. A large denominator compresses r: N minus W0 is 1.27 nats on M and 0.49 on W.
11. **The own-mask reads and weight errors do not help the adversary's case.** Own-mask KL to FT on M: N 0.0663, A 0.0776, I 0.0813, and W0's weights under A's masks 0.0937. Do-nothing ranks first; I is close to a state with nothing placed. This confirms E79. Weight error is N 1, I 0.090, A 1.362 (A's 4.748 and ‖δW‖² 3.485 recompute the filed 4.75 and 3.49). N's 1 is a literal in the code (`run.py:227`) and a rule, not a read: N's weights are FT's exactly. The evidence for I as a candidate is 0.090 and clean KL 0.0094, both independent of the adversary, and the second known since E72.

### Low

12. I's own-mask read on M exceeds clean plus self by 0.010 (0.0813 against 0.0713), about 14%; A's was about 1% in E79's check, and I on W is within 1%. Cause not read.
13. Own-mask comparisons cross jobs: N and A read with bf16 components, I with fp32. W0's gap between the paths is 0.0015 (own), 0.0006 (self), smaller than I's gaps to N and A.
14. In this job A's clean KL is not re-read, and N's is read on one row only (-3.8e-9). E79's reads cover these, and clean depends only on weights, which I read as the same.
15. `amend2.ft_g`'s docstring says "components unmoved"; called inside `applied` for I they are moved, and that is the right behaviour. The band floor 0.19 sits just under W's spread 0.193; immaterial.

## What can be claimed

- The filed outcome holds **as filed**: b 0.19, r 1.317, 1.197, 1.253, and the filed null (do-nothing within W0's spread) fails on the one start read. The filed lean "within the band" missed; "in-span passes" held.
- The attacker reads do-nothing above W0 by about 30% on both row sets, far less than the 67M's 2.6-4.8 x. On W this rise is made by the attack, with one start, and on M it largely is not.
- I and A also read above W0 on M, by 20-25%. The order N, A, I holds along the trace of one start, which is a hint.

## What cannot be claimed

- That D1 has a test: nothing here shows the adversary telling a bad placement from a good one, at the filed read or any filed bar.
- That in-span is a good placement by this test, or that "passes" means anything A's weights do not undercut.
- That do-nothing fails because of what sits in Δ, on M.

## What would settle it

Each of these is a read of the existing harness, on the same card: I and A on W, about 31 min each, ~$2 each; a second paired start of N, I and A on M, about 10 min each; one state with clean near FT and a wrong placement (B, L or a random change) on M and W. A rule filed for those should compare N and I by one bar on paired starts, set from the spread read in the same job. The results also need a reference for the raised level on M. This is a new plan or an amendment, for the lead and `/design`.

## Verified

- **By recomputing** (`recompute.py`): b, every r, every rule verdict, r on W, and the A and ‖δW‖² arithmetic, by a separate implementation, equal to `results.json`; step traces from `run.log`, ratios per step and paired orderings; the same-seed comparison with the 128-row job. The 14 tests in `tests/test_qwen_d1_adversary.py` pass. `predictions.json`, `rules.py` and `run.py` have one commit (871bec2d, 05:15), before the run's first log line (05:31); results at 1c83c852. Local sha256 of δW, `A_offsets` and `cut` match the filed sums.
- **By reading** (no bug found in): `run.py`, `rules.py`; `qwen_vpd` Γ inputs, `importances`, `masked_logits`, `pgd_kl_rows`, `_source_kl`; `qwen_teacher.adversary`, `do_nothing`, `Offsets`; `qwen_maps.Alive`, `site_change`, `applied`; `qwen_sft.set_weights`, `plus`; `qwen_masked.student_masked` at zero offsets; E79's `clean_logits`, `masked_rows`, `positions`, `ft_g`, `masked_cell`; `split.least_squares_x`, `dsigma`.
  - N: weights W0 + δW, components unmoved, so Δ = Δ₀ + δW. I: weights W0 + dσ, all alive columns moved, orientation correct, Δ = Δ₀ − dU dVᵀ. A: weights W0 + change, cut columns moved, Δ = Δ₀ up to rounding. In all four, Γ's taps and readouts and the clean logits are recomputed from the state, per row, per pass.
  - Sources are drawn from the seed, independent of state; rows are one list; fp32 weights and components, Γ bf16 in every state. Components and weights are restored by code, and W0's clean read after I and A equals E79's to 6e-7. δW's file holds the 196 sites and nothing else, so nothing is dropped by the filter.

## Not checked

- No GPU rerun; that the pod ran this commit; the decomposition's checksums (`qwen_files.fetch` not read).
- `split.gram_basis`, `_truncate`, `to_basis` in detail, `qwen_maps/run.py`'s `basis_of`. The clean 0.0094 reproducing E72 is the support.
- How much bf16 rounding of V moves g, for any state; I's dV sizes; I's Δ move in the plain measure.
- The cause of the raised level on M; KL's dependence on reference entropy, as a measurement.
- Start-to-start spread of N, I and A: never read.
- E71's and E39's code beyond the filed lines and `reads.json`.
