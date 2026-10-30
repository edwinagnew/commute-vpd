# Advice (Fable, 2026-10-31): C3 after job 1's baselines

Advice only; the lead decides. Read: C3 plan § 2 (f) and § 6 rules 5-7, `results/measure.json` `filing_B_draft`,
code review findings 1-3, E58 analysis Amendment 6, `decisions.md` rows 105-107. The planted-control fix (Amendment 1,
in the working tree) is a clear rule fix and is not discussed further: file it, rerun job 1 items 5-6, FYI.

## What job 1 actually says
- **p_prior 0.92 is the construction, not a finding.** Elicit's values are W₀'s ranks 2-20 by design, and W₀ is nearly
  flat over its top 20 at these keys (rank-1 prior −5.0 nats, elicit −5.8, against the plan's guess of −2.5 and −6.4).
  Any prior column had to put elicit at the old end. Its bootstrap has zero width because the prior is a property of the
  map, not the state, so "firm" is automatic. "r reads like W₀'s prior" is therefore not shown here, and it is already
  **refuted** elsewhere: E58 Amendment 6's prior-matched cipher read r 0.37-0.38 at a prior 1.6× below copy's, where copy
  reads 0.77; and on this axis the prior puts copy at 0.59 ("between") while r puts it at about 1.0. Option C's reading
  is unsound; do not write it up.
- **p_D 1.47 is the surprise worth a sentence.** D = KL(p_W₀ ‖ p_m) rewards sharpening: elicit moved W₀'s answer
  distribution most (10.5 nats), the cipher at matched norm least (4.9). D reads how peaked the state became, not
  "distance" in the plan's sense (guess 0.2). Report it; it is a column, not a verdict.
- **The caller's framing is one notch too dark.** Rules 5-6 convert only readings on the baselines' side (or within
  0.1 of 1.47 / 0.92). "label" and "between" survive, so **"VPD alone" (both instruments read label) stays reachable**,
  and it is outcome (b), the one that bears on the user's priority. Lost: "shared" (now reads "distance and prior")
  and "conflict". The run still separates "VPD is the odd one out" from "every reader agrees on this axis".

## Options
| | shows | cost | decides |
|---|---|---|---|
| **A** keep rules; filing B revises leans; add a filed lean on copy's position; run jobs 2-3 | VPD alone / between / no gap on 8 seeds; where pruning and reverse put **copy**, the one map where prior (0.59), D (0.77) and r (≈1.0) differ | $0, ~1 h debug GPU | lead |
| **B** dated amendment: keep only the 0.1 clause | makes "shared" reachable again | $0 | nobody should: it refiles the verdict rule after seeing the numbers, in the direction that favours the hoped-for outcome, and the plan's own edge case says the all-VPD case "is intended, outcome (f)". Not a mis-specification. Report the 0.1-only reading as an **unfiled** column beside the filed one |
| **C** stop; write job 1 as the finding | nothing new (above) | — | would be the user's (changes the line's claim); advise against asking |
| **D** new axis (e.g. a prior-matched new end at elicit's prior) | separates prior from kind at elicit's prior | new fine-tunes, /design round, ~1 day, $0 | lead with Fable; only if A's copy column is ambiguous |

## Recommendation
1. **A.** Filing B: P2/P3 "VPD" → 0 (reads "distance and prior"), V "shared"/"conflict" → 0, mass to "not settled" and
   "VPD alone"; state in one line that on this axis the cheap baselines already place elicit at the old end by
   construction, so the run's live question is whether the instruments **disagree** with VPD.
2. **Add to filing B, dated, before any instrument reads a real state:** a reported-column prediction on copy. Lean:
   if an instrument reads the gap, its p_copy sits nearer r's (≈1.0) than the prior's (0.59), about 55%. This is the
   only free separation of "prior" from "old pieces" the twelve states offer; filing B may hold predictions that
   depend on job 1, and copy is already a reported column.
3. Write-up framing: "on an axis whose ends are built from W₀'s own guesses, every decomposition-free reader puts
   elicit at the old end; the question VPD can still lose is (b)". Lesson for the next filing: a same-side clause with a
   baseline that can sit outside [0, 1] is too coarse; next time only the tolerance clause, fixed before the baseline is
   read.

**What would change this.** If job 2's new seeds move p_D or p_prior into "between" (unlikely: the prior has no seed
variance), rules 5-6 stop firing and "shared" is live again with no amendment. If both instruments read "no gap" on
the 67M, D is the next step and C2 waits.

## Who decides
Lead: A, filing B's leans and the copy lean, the unfiled 0.1-only column, Amendment 1's rerun. User: nothing here;
FYI that the headline "shared" verdict is unreachable on this axis and why. Only a (b) result goes to the user at once,
as the plan already says.
