# Advice: after E58 Amendment 3 (job 46398) — what it says for reading A, one more $0 job, then the synthesis

*Fable, 2026-10-30, on the lead's three questions. Read: the plan's Amendment 3 and its before-run note,
`predictions_amend3.json`, `amend3.json`, `amend3.out` (reads and check lines), `run.json` and `amend1.json` scores,
the research log's overview, next-steps and E58 card, my two earlier advice notes.*

**In plain words.** The run did what the control was for and more: at one key count, three maps whose carry to unseen
keys runs from a fifth to four fifths all read the same r and ratio, and both numbers move more with training step
than with carry. That is the cleanest support reading A has had, but it is one seed and it is unscored, because the
filed drift guard fired on the cipher. One more $0 job can turn it into a scored line and settle what r is reading
(the model's key-conditional guess, or just the token kind). Then write the synthesis; the 67M has little left to say.

## 1. What the run does for reading A

- **Fair, as unscored support.** Within 128 keys, step 10: x 0.21 / 0.32 / 0.82 (elicit, rank-1, copy) against r 0.78
  / 0.89 / 0.87 and ratio 0.96 / 1.14 / 1.11. r's spread across maps (0.11) is about three seed spreads at 24 keys
  (E58 P4: 0.03); x's spread is 0.61. Rank-1 has the highest r with the middle carry; copy the highest carry with the
  middle r. No ordering by carry. The same holds at step 30 (x 0.11 / 0.20 / 0.81, r 0.72 / 0.85 / 0.77).
- **A second drift-free line is already on disk: r moves with step, not with carry.** Copy's x is 0.82 at step 10 and
  0.81 at step 30 while its r falls 0.87 → 0.77 (B at step 30 reads "opposite" on copy for this reason). Every map's r
  falls 0.04-0.10 from step 10 to 30. So r is a function of training stage and update size (E36, E60) and not of what
  carries; say this plainly in the write-up, it is the stronger half of the result.
- **What the drift guard caught.** The norm-matched cipher at 128 keys (step 10) has G_spec 1.9 against 9-11 for the
  24-key reference: it is a barely-trained state (label "too_small"), and its r 0.37 is the r of a state that has
  learned almost nothing. At step 30 (G_spec 17.8) the cipher reads r 0.30, ratio 0.43: still 0.07 over the 24-key
  mean, so the guard would have fired there too. B stands as "untested_drift" and the verdict as "untested"; do not
  refile. The honest sentence: the cipher's r rose by about 0.1 with the key count while the three scored maps' r
  stayed within 0.05 of their 24-key values, so the shift is the cipher's own, not a common drift; what B wanted to
  rule out (every r moving with K) did not happen, but the filed rule cannot say so.
- **Checks.** 1: elicit's carry is "falling" (0.21 → 0.10 by step 30, then flat to 300): a real but partly transient
  rule, so matched-P reads matter (the step-30 companion is that read). 3: the shuffled map is unreadable at step 10
  (trained P 0.02, G_spec 2.1, x −0.97 is the push toward 96 trained values): not a pass, not a fail; it reaches P
  0.83 at step 30 and was not read there. 4: elicit's update lifts W0's top guess a little (H_spec +0.3) and lowers the
  key itself (−1.0): a narrow sharpening, not a general lift. 5: rho 0.19 / 0.01: the per-key rule signature is weak.
  Report all four; none changes reading A.

## 2. One more $0 job, then stop (Amendment 4, one `debug` job, about 20 min)

What the within-key-count reading lacks is a noise floor and a filed rule; what the line lacks is a test of *what*
r separates. Both fit one job on the saved key split:
1. **Seeds 1 and 2 at 128 keys** for elicit, rank-1, copy and the cipher, read at steps 10 and 30 (all checkpoints
   are in memory). File before running: "across the three scored maps at each step, max r − min r ≤ 2 × the
   within-map seed spread, while max x − min x ≥ 0.4"; the same for the ratio with tol 0.2. Lean 70% pass at both
   steps. A cross-seed Spearman of r on x over the 9 points is reported, not scored (9 points, no power).
2. **The shuffled map read at step 30 and at its first checkpoint over the gate**, with r and ratio. This is the
   drift-free test of what r reads: the shuffle has elicit's value pool and token kind but no key-to-W0-preference
   relation (its trained-key W0 geometric P is 1e-5 against elicit's 1.3e-3, near the cipher's 1.4e-6). r about 0.3
   (cipher-like) → r reads the model's own key-conditional guess, i.e. elicitation in the narrow sense the line wants.
   r about 0.8 → r reads the token kind, and "built from existing subcomponents" means only "targets the model can
   produce somewhere", a weaker claim. Lean 60% cipher-like. Either answer goes into the synthesis as its sharpest
   sentence. File the cut at the midpoint of the 128-key cipher and elicit at step 30 (0.30 and 0.72: cut 0.5).
3. The cipher's r and ratio at matched G_spec (its first step over the gate, 16) reported as the drift line, beside
   the norm-matched one; no rule.
Skip it only if the Qwen line's E45 anchors are ready to run this week and the lead's time is the constraint: then
write the synthesis from one seed and say so. Do not spend a paid hour on the 67M for this.

## 3. The synthesis: claim, caveats, and the next question

**Claim (one sentence).** On the 67M, r and the stand-in ratio read whether a fine-tune's targets are what the
original model already produces in that context, and read the same whether or not what it learned carries to unseen
keys: they separate every map built from the model's own guesses (copy, top word, second-to-twentieth guesses, the
half-known map, lowercase text in E53) from random ciphers at 24 and 128 keys, and they do not order those maps by
held-out carry (x 0.2 to 0.8 at one level, r 0.78-0.89). **Null kept beside it:** r is also a function of training
stage and update size (E36, E60; copy's r falls 0.1 from step 10 to 30 at fixed carry), so a single r is not a
measurement of anything; only same-row comparisons at matched step or matched trained P are.

**Caveats, in the report's words.** One seed at 128 keys until Amendment 4; the filed test B is untested because the
cipher's r moved 0.1 with the key count; the behavioural label's positive control passed only on copy; elicit's
carry is partly transient; the granularity limit (a new circuit from old pieces reads as old) in every re-gating
sentence; held-out carry within one map is one notion of generalisation, not "a new capability"; 67M only, trigger
maps only. "Not run": E45 states, a pool-matched random map other than the shuffle, more than one decomposition seed.

**What was a distraction and what was not.** Agree with the user: the fixed-shift work (E18-E36) asked how re-gating
is carried and never had a same-row control; the results that hold up are all same-row tables (E12, E53, E54/E56, E58
and its amendments), where r and the ratio separate known-target from unknown-target updates on the same keys and
steps. The synthesis should be built around that table, one figure: x on one axis, r and ratio on the other, every
map and seed at 24 and 128 keys, the cipher as the only separated cluster.

**The line's next question.** Not "does VPD distinguish elicitation from new mechanism" in general (the granularity
limit makes it unanswerable at the subcomponent level) but: *on a real post-trained update, where does r sit, and does
the same-row separation survive?* Concretely: E45's Qwen3-1.7B anchors (the GSM8K fine-tune beside a cipher-like
control of matched size and step) through the same table, at matched trained P. If the real update sits with the
known-target maps, the re-weighting reading has its first evidence off the 67M; if with the cipher, the opposite; if
between, the line has a scale to place updates on, which is the useful product either way. Everything on the 67M
after Amendment 4 is diminishing.

**What would change my advice.** The shuffle at step 30 reading r about 0.8: then "built from existing subcomponents"
collapses to "a token the model can emit", the claim is rewritten to that, and the Qwen test is about whether real
updates ever look like a cipher at all. Or seeds 1-2 showing an r spread across maps over 0.15 at a step: then the
one-seed line is noise and the synthesis drops it.
