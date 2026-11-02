# Advice (Fable, 2026-11-02): C2's "tie" on an untrained decomposition

**Lead's note, 2026-11-02, after job 48441.** The token test below (section 1) came out against this advice's expectation: the numerator follows the state, not the token (N states -0.87, -1.07 at the P token; P states -0.19, -0.01 at the N token). Section 3's recommendation stands on the tie alone.

Advice, not a decision. For the lead; plan `reports/plans/2026-10-31_plan_e55-c2-vpd.md`, filing
`experiments/pcfg_anchors/results/predictions_c2.json`, numbers `~/commute-data/scratch/reuse-lead/c2-fixes/readcheck_A/c2.json`
(job 48413, the rehearsal's decomposition at step 400 of 400K).

## 1. What separates P from N at step 400: the readout token, not the update

I read the cells. r's denominator (the gain T1 − T0) is 19-21 on every P state and 14-18 on every N state, so it does
not separate. The numerator L01 − L00 (what swapping the gates from W0's to the state's does to the W0 masked model's
log-odds of the readout token) does: every P state reads between −0.16 and +0.11, every N state between −0.51 and
−1.26, and the five unlearned N states (update size 0.012, the smallest of all) read −1.17 to −1.26, the most negative;
the P states with the largest updates (0.115) read only −0.06 to −0.16. So the effect is not the update's size. The two
ID states read +2.07 on a third token. Since the cut, every state is read on identical inputs; the readout token and δW
are the only things that differ per state, and the sign and size follow the token. The whole r range is 0.07 wide
(ID +0.074, P 0, N −0.05) against E58's 0.77 / 0.24: nothing here reads re-gating, and the cut-off is drawn in a 0.03
gap whose sign the N token sets. The same holds for the ratio (L11 − L00 tracks L01 − L00 within 0.1 on every state).

**Confirm for $0** (one `debug` job, ~5 min): read 2 P and 2 N calibration states each with the other class's label as
the readout (`c2.read_one` takes `task`; build the windows with `pcfg_vpd.windows(rows, answers(other_task, rows))`,
gates unchanged). Token effect: the numerator follows the token (a P state read at the N token ≈ −0.8, an N state at
the P token ≈ 0). Update effect: it follows the state. I expect the token. Report only; nothing in the filing moves.

## 2. Is a tie evidence? Not without the positive control. The amendment

Under the filed rule, "tie" needs only a monotone task-dependent offset of any size, because the cut-off is free and the
truth is the task. Step 400 shows the offset comes free with the readout token. The ID control is the one filed number
that asks for a magnitude (0.5 on both readings, E12 / E58), and today it moves no outcome. Candidates:

| candidate | step 400 fails it? | a healthy VPD passes? | new degenerate case | contaminated |
|---|---|---|---|---|
| (b) "tie" / "pass_under" written "…, apparatus unverified" unless the ID control holds on that reading | yes: ID "none" on both | yes: E58's elicit states read 0.77 / 0.95; ID is a count(a) state at S 0.93 | none; mirrors the filed `fail_unverified` | no: the mirror of a rule filed before this read, bar from E12 |
| (a) absolute bar: existing iff r ≥ 0.5, no calibrated cut-off | yes (every state "new", 9 of 21) | 67M yes; a 3M decomposition might read existing at 0.3-0.5 and fail for magnitude alone | replaces C1b's rule, so the table stops being comparable | no |
| (c) margin: calibration gap > k × the step-400 gap | by construction | k is a guess | a healthy VPD with a tight gap fails | yes, built from these numbers |
| (d) negative-control row: the step-400 reading reported beside the trained one | — | — | none | reported only |

Recommend **(b) + (d)**: one sentence that already exists for "fail", applied to the other two outcomes, and the step-400
read filed as the negative control it turned out to be. Not (a): it also carries re-gating's magnitude, but at the price
of the comparable table; the ID bar in (b) carries the same 0.5 on the apparatus instead. If a trained decomposition
ties with the ID control holding, the write-up still reports the P states' median r against 0.5 (blind-to stays as
filed: a perfect score cannot separate "reads reuse" from "reads the task").

**Amendment text** (append to `amended_before_the_read`; dated, before any trained reading):
> `tie_unverified`: "a tie or pass_under with the positive control not 'holds' on that reading is written 'tie, apparatus
> unverified' / 'pass_under, apparatus unverified', exactly as `fail_unverified`. Outcomes, headline order and leans
> unchanged." `negative_control`: "the rehearsal decomposition at step 400 of 400K (job 48413, `c2-fixes/readcheck_A/c2.json`)
> read 'tie' on both readings with cut-offs −0.018 (r) and −0.013 (ratio), ID 'none' (0.074 / 0.074), U0 − T0 −0.022.
> Written under this amendment: 'tie, apparatus unverified' on both. Reported beside the trained reading as the
> untrained reference; moves no outcome." Source: this file.

## 3. Is C2 worth 56 L40-hours? Options and a recommendation

The only informative branch is a verified fail (ID holds, VPD misses where cheap methods are perfect), filed at 20% ×
P(ID holds 70%) ≈ 15%; a verified tie says VPD ties on a set where the readout token already ties; an unverified
anything says only that a 3M decomposition did not train, which is not the user's priority (benchmarking VPD against
pragmatic techniques). The disagreement benchmark lives where the label is open (67M open-pair, Qwen), and the 67M has
a released decomposition, so it costs no training. No sourced shorter recipe exists (plan §6c).

- **A. Drop C2 now (plan §6 option d).** File the amendment and the negative control, write a short analysis of the
  step-400 read (§1's finding and the $0 confirmation), close the card as "VPD on C1b's set: not run; the test cannot
  rank VPD above cheap methods and the tie is free". The lead's attention goes to the open-label sets. Recommended.
- **B. Job 1 only, with a go/no-go.** Run the first 24-h job (~170K steps) when the queue is empty; at its end extend the
  filed P3 read (`readcheck --states few`) to the two ID states (+1 min). Jobs 2-3 go ahead only if either ID reading is
  ≥ 0.25 on either column (half the bar, up from 0.074) or L00 − T0 on the natural rows has halved from −3.6; otherwise
  stop and write it up as A. Spend: one L40-day of a shared node. The hedge if the lead wants a PCFG VPD for the
  re-gating line later anyway.
- **C. All three jobs as planned.** Only if the user wants the apparatus result (does a VPD of a 3M PCFG model verify)
  for its own sake.

**What would change the recommendation to B:** the user confirms a PCFG VPD is wanted downstream (re-gating on LOW, or
reading the five unlearned N states, which no cheap instrument can read), and the `normal` queue is empty at
submission. **To C:** the user says the apparatus question itself is the goal. This drops an approved plan at gate 6,
so it is a reversal for the user, not Fable; raise it with the step-400 numbers in three plain sentences.

## Not run
The cross-readout confirmation (§1); any trained reading; the within-token spread's dependence on update size (weak,
reported in the table above only as a range).
