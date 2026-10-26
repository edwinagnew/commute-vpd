# Review: E37 teacher training of a fine-tune with no trigger

*2026-10-26, independent check by a fresh agent that did not write the code, run the job or write the analysis.
Read: plan, `results/predictions.json`, `results/teacher_spread.json`, `results/large/run.log`, `run.py`,
`src/commute/row_copy.py`, the harness path (`harness.pile_ratios`, `faithfulness.faithfulness`/`harness_pgd`,
`faith_score`), `gamma.pgd_sources`, `cancel.masked_kl`, the analysis report, `notebooks/E37_teacher_spread.ipynb`,
and card `decision-broad-copy`. No code run; numbers recomputed by hand from the JSON and log.*

**In plain words.** The numbers in the report, notebook and card match the run's files, and each prediction is scored
by its filed rule. Two of the report's supporting claims are weaker than written. The "second adversary" uses the same
search on the same rows as half of the first, so it is not independent evidence. Do-nothing's high reading is partly
built in, because the adversary is allowed to switch off Δ, which holds the whole update. The P5 "reversal" also
compares two different measures. The headline (the copy keeps the behaviour and fails the fixed line at step 1,000)
stands for this one run.

## Confirmed

- Every number in the report's tables and prose matches `teacher_spread.json` or `run.log`. That covers s, gap,
  gap share, damage, the harness ratios, anchor, budget ratios, size, the cosines, the train-KL means and falls (36%,
  40%, 33%), and the per-block KLs and ratios. Block ratios are over W₀ seed 0's blocks, 0.6794 and 0.7182.
- Lines: floor = max/min − 1 over W₀'s three seeds = 0.0748; pass 1.5/1.0748 = 1.396, fail 1.5·1.0748 = 1.612.
  The harness ratio is over W₀ at adversary seed 0 (`harness.pile_ratios`), as the report says.
- P1–P7 and C1–C3 are scored by the filed rules in `run.py:score`. Verdicts: P3 and P5 fail, the rest hold. None was
  re-interpreted.
- The code measures what the plan says. A copy's target is W₀ + (U′V′ᵀ − UVᵀ), so Δ is unchanged. The copies'
  unmasked KL, 0.011, matches W₀'s 0.012, which checks this. The adjoint state comes from `split.adjoint_state` with
  Σ from 256 teacher rows. s and gap are read on 64 held-out lowercase val rows. The cosines use the saved bf16
  offsets.
- E34 references check out: the copies' harness range was 0.895 to 1.37, and the budget range 0.95 to 1.15. The
  adjoint's P(target) was 0.053 to 0.257 on five of six teachers. The cosines were 0.08, 0.22, 0.245 and 0.285. The
  budget noise was 1.070, and the r1 SGD teacher's budget/harness gap was 9.24/1.59, about six times.

## Corrections

1. **The budget adversary is not a second adversary** (report 174–180, 218, 228–229, 252–253; notebook "Is the
   copies' failure the harness's alone?"; `plot.py`'s adversaries figure). `gamma.pgd_sources` is the harness's own
   PGD: the same rows (val 0–127, the harness's block 1), 20 sign steps of 0.1, random start, seed 0, one source per
   matrix. It differs only in reading the KL in fp32 under the masks it finds. Against harness block 1 the two agree
   closely:

   | state | harness block 1 | budget |
   |---|---|---|
   | do-nothing | 3.91 | 3.92 |
   | lowercase copy | 1.69 | 1.80 |
   | plain-text copy | 1.60 | 1.66 |
   | adjoint | 1.10 | 1.17 |

   - The "budget reads 18–20% lower than the harness" is block 2 (rows 128–255) being harsher. It is not an
     adversary difference, so the surprise at 252–253 should go.
   - "Under two adversaries" and "the smallest of the five readings" overcount. Independent evidence here is two
     batch orders and two texts under one adversary at one seed.
   - On block 1 alone the plain-text copy is inconclusive, at 1.60 (the report says so at 167).
   - For E34 Amendment 5: here the two readouts agree on a fine-tune's do-nothing state, where E34's differed up to
     six times.

2. **The do-nothing anchor is inflated by construction** (report 26, 106, 115–117, 219–220; notebook; card).
   `harness_pgd` uses `use_delta_component=True`, so the adversary sets Δ's mask, and in do-nothing Δ holds δW. Its
   unmasked KL (components only, no Δ) is already 26 times W₀'s, 0.317 nats. "Less than half as fragile as leaving
   the update in Δ" therefore partly measures where the update sits. The anchor card says this in general. Say it
   here too, beside every anchor reading and in the adversarial-ratio definition (report 64–66).

3. **P5's rationale compares different units** (report 88–89, 246–249; notebook surprises). The rule is right and
   P5 fails as filed. But the expectation came from E34's adjoint P(target) of 0.05–0.26, a probability, while s is
   a share of log-probability gain.
   - My estimate converts E34's figures to log-prob shares. It takes W₀'s P(target) as about e^−gap. The gap is
     5.5 nats on r1 and 3.2 on r0; emoticon's W₀ P is 0.004. On those figures E34's adjoint kept about 0.5 (r1),
     0.6 (r0) and 0.6 (emoticon Adam) of the gain.
   - So 0.80 here is a smaller surprise than "lost the behaviour → kept it" suggests. The report should say the
     filed rationale used another measure, not that a behaviour reversed.

4. **The filed decision tree** (report 121–122, 268; card option d). The tree reads "P1 holds, P3 fails: the same
   branch" as the P1-fails branch. That branch opens with the falling-KL test, which fires here (36% > 10%) → "under-
   trained, rerun longer". So, read literally, the filed tree prescribes option (a). The report's statement that it
   applies only when P1 missed is a re-reading. Option (d) is the tree's step *after* that test fails to explain
   things. Say the tree is ambiguous, or label (a) as the filed branch.

5. **Name the alternative that fragility tracks damage on Pile** (report 110–113, 244–249). The harness ratio rises
   with damage, KL(W₀ ‖ state) on Pile:

   | state | damage | harness ratio |
   |---|---|---|
   | adjoint | 0.05 | 1.05 |
   | plain-text copy | 0.22 | 1.74 |
   | lowercase copies | 0.23 | 1.88, 2.20 |

   The adjoint may pass because it barely changes Pile predictions, not because it is a better map. The copies may
   fail because they reproduce a change on Pile 3–7 times larger than E34's teachers made: this teacher's
   do-nothing reads 4.87, against ≤ 1.59 in E34. The copy-to-teacher ratio of 0.39–0.45 is no worse than E34's,
   about 0.6–1.0. "The teacher is larger" (244) gestures at this; make it explicit.

6. Nits:
   - Notebook scorecard footer: "reproduced E36's gain within 0.0001" → 0.00015 (0.12995 against 0.1301).
   - Report 197 / notebook: the lowest matrix cosine is 0.9099. "Below 0.91" is true only after rounding; say 0.91.
   - Card: "analysis to follow" is stale, since the report exists; "not yet independently checked" can cite this
     review.

## Headline changes recommended

- Keep: "the copy keeps the behaviour (0.998) and fails the filed fixed line (2.20 > 1.61) at step 1,000, at two batch
  orders and from two texts, one adversary seed, copies still learning (KL −33 to −40%)."
- Drop "under two adversaries" (report 218, 229; notebook) and the harness-vs-budget surprise.
- Qualify "half as fragile as do-nothing" with correction 2 wherever it appears, including the card.
- Recast the P5 reversal per correction 3, and add the damage-tracks-fragility alternative per correction 5.
- Card: put option (a) first as the filed tree's branch. It is the cheapest test of the "still learning" explanation.
  Read the harness at two adversary seeds per step, since the plain copy sits within seed noise of the line.

## Not checked

The figures' rendering (only `plot.py`'s constants were read); `split.adjoint_state` and `component_training.train`
internals beyond their call sites; the provenance sha256 values.
