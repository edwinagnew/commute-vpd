# Advice: E58 Amendment 2's control failed (job 46383) — real carry or a step-10 artefact?

*Fable, 2026-10-30, on the lead's question. Read: the plan's Amendment 2, `amend2.py`, `predictions_amend2.json`,
`amend2.json`, `amend2.out`, `src/commute/{heldout_keys,open_pair,elicit,keydep}.py`, my earlier advice (item 3).*

**In plain words.** The control was mis-specified, not the measurement. The elicit map is not a "no rule" map: by
construction (`elicit.pick_values`) every value is the token W0 already ranks 2nd-20th after that key and ten times
more than after other keys. "After the marker, say what W0 half-expects for this key" is a rule, and with 96 keys the
fine-tune found it by step 10, where 24 keys only let it memorise. The held-out log-odds moved from about -0.5
(24 keys, x -0.03 x G 18) to +2.0 here: a sign change of 2.5 log-odds, so the smaller denominator (G_spec 9.3, trained
P 0.60) explains at most a factor two in x, never the sign. The map with no rule was always the cipher (random
values); it should have been the control. This is the experiment's best case, not a failure: elicit is now a map
whose x moved by 0.24 against the 24-key reference, with its own r readable beside it.

## Why I read it as real, from the numbers already on disk

- H_spec is positive on 28 of 32 held-out keys (median about 2, one at 7.4): broad, not one outlier. H_other is -0.79:
  the other pool values go *down* at those windows, so this is not a push toward "that kind of token" (the plan's own
  "blind to" line for H, now checked). All 32 values are novel (x_novel = x).
- The recipe is 3 views per key by step 10 (13 at 24 keys). Few views per key over many keys is exactly where a shared
  component (a rule) is learned before per-key lookup; whether the lookup then displaces the rule is the step-30
  question, and both answers are consistent with "real at step 10".

## How to tell, for $0 (one `debug` job, about 5 min; all checkpoints are kept in memory, so re-training is 15 s)

1. **Elicit at every E12 checkpoint** (10, 30, 100, 300): x, H_spec, G_spec. x flat or rising at 30-300 → a rule the
   fine-tune keeps; x falling toward 0 by step 30 while G_spec rises → the rule is an early-training stage that the
   lookup displaces (still real carry at the read step, but B must be read at matched P, see below). Also read E58's
   24-key elicit_s0 at step 30 (re-train, same seed) so the 24-vs-96 comparison exists at both steps.
2. **The no-rule map at 96 keys: the cipher** (`kv.bijection`, as E58), same recipe, read at the common step and at
   the norm-matched step. Expected x within 3 s of 0 with the value-specific gain, as at 24 keys (E58 P1). This is the
   control Amendment 2 needed, and it stops the job if x > 0.05.
3. **Shuffled elicit** (each trained key takes another trained key's elicit value, a permutation; held-out keys read
   with their true values): same value pool and token kind, but no key-to-W0-preference relation. x about 0 → the
   carry is specifically "W0's own key-specific preference" (a rule); x about 0.2 → it rides on the pool and the
   specific gain does not remove it (the label is broken at 128 values). I expect about 0, given H_other.
4. **Cross-reads of the existing state** (`amend2/elicit_s0.pt` on `windows_rank1_new.pt` and `windows_copy_new.pt`,
   no training): if elicit's update also lifts W0's argmax at held-out keys (rank-1 values), the rule is "sharpen the
   key-conditional distribution"; if only rank 2-20 values gain, it is narrower. Copy should read about 0 (a check
   that the update does not lift the key itself).
5. **Per-key H_spec against W0's own P and key-specificity ratio** of the held-out value (`qual[x]["p"]`,
   `median_other`, recomputed in-job): a positive rank correlation is the rule's signature. Report, don't score.

## What each answer does

- **Rule confirmed** (1 flat, 2 and 3 about 0): reading A is untouched in its claim but its control moves. The claim
  "r reads which pieces a fine-tune builds from, not whether what it learned generalises" now has a map (elicit) whose
  carry rose from -0.03 to 0.21 under the same recipe and value rule. If r and the ratio stay 0.75-0.85 / 0.9-1.1 on
  this state, that is the strongest line the plan hoped for from rank-1 or copy, on the map closest to E54's. If r
  moves with it by more than 0.05, it is (b) from my earlier advice: to the user before any claim.
- **Stage artefact** (x falls to about 0 by step 30): carry is a transient of under-trained states. Then B must compare
  states at matched trained-key P, not matched step, or it mixes training stage into the key count; and E54's own
  step-10 states (P 0.93-0.98) remain the reference. Reading A unchanged.
- **Pool artefact** (3 reads about 0.2): the value-specific label fails at 128 values; stop scaling keys with this
  label and redesign it (e.g. specific gain against a random token pool rather than the task's values).

## The amendment (recommendation)

The filed E prediction stands as a miss (lean 5%, read "fail"); record its cause in a dated note, not a refile. Then
Amendment 3, dated: (i) the control is the cipher at 96 keys, read at the common step, with the same ±0.05 rule and
stop; (ii) elicit becomes a third scored map, with its own dx against the 24-key reference (0.24 > dx_min 0.15) so B
is testable on it; (iii) the common step stays E12's rule on elicit (filed; do not change a rule after seeing numbers)
but every map is also read at step 30 and reported, with B re-read there as a companion, not a score; (iv) checks
1, 3, 4 and 5 are reported lines in the same job. One job, $0; rank-1 and copy then run as planned. Keep the pool-of-
values label as filed; if check 3 reads 0.2 the job stops before rank-1 and the label is redesigned first.

**What would change my advice.** The cipher at 96 keys reading x > 0.05 with the specific gain: then the label, not
the elicit map, is what broke, and no map's x at 128 keys is readable until it is fixed. Or x at step 300 reading
higher than at step 10: then the rule is strengthening with training and the elicit map is a candidate "reuses" state
for step 2's P2 in its own right (x 0.5 is 4.7 log-odds at G_spec 9.3, 9 at 18).
