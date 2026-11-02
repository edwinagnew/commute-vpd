# Fable's advice: what the re-gating line does next while step 2 waits for labels

*2026-10-30, Fable, at the lead's request. Advice only: nothing filed or run. Words as in `GOALS.md` → Terms. Read
beside `reports/plans/2026-10-30_plan_regating-ground-truth.md` (step 2) and the adopted programme
`reports/plans/2026-10-30_plan_regating-development-FABLE-DRAFT.md`. Inputs: E53 Amendment 2 (breadth does not explain
the separation), E54 and E56 (r 0.76 vs 0.24 on five seeds; steering gap 0.03 and neuron patching nothing), the step-2
plan's finding that every 67M label but E54's elicit map is fixed by construction, and reuse-lead's card
`decision-e55-c2` (Stage B is its option 3; it recommends option 2, on the counting model, which has no decomposition).*

## In plain words

The line's number now separates two task pairs that steering and patching cannot, but every label it has been scored
against was decided by how we built the task, and the one plan that would earn a label from outside the decomposition
is waiting on another line that has other priorities. The cheapest way forward is to earn an open label ourselves: train
the half-known map on most of its keys and ask, with no decomposition, whether the fine-tune carries to keys it never saw,
which is the reuse line's own definition read on behaviour. If the decomposition's reading agrees with that label we have
the first earned case; if it disagrees we have the first concrete case for the granularity limit, and either is progress.

## Ranked recommendation

1. **Build the open pair ourselves: the held-out-key elicit map (one cluster `debug` job, $0).** Details below. This
   is the Jain-style weakly-relevant fine-tune on the 67M that the step-2 plan asks the reuse line for, with the label's
   behavioural half (the user's bounded g: g learnt on training rows must work on held-out values) read by us and the
   interchange-patch half left to Stage B whenever it runs. The label owes nothing to the decomposition and is not fixed
   by construction, which is what step 2 was for.
2. **Fill the shared table (the step-2 plan's job, $0), in parallel, as a second pod-runner.** r and S on E53's four
   Pile-text states and the ratio on the broad cipher. Low information (labels fixed by construction) but it gives r and S
   a third task family and the table is needed whichever way the labels come. Do not wait for Stage B to run it.
3. **The gating-pattern novelty column, reported and unscored, added to item 1's read stage ($0).** Worth having
   exactly where item 1 disagrees (branch 5 of the step-2 tree); not worth its own job.
4. **Prepare Qwen stage 1 on paper, not before item 1 has read.** The programme's order (D after A and B) stands; the
   plan can be drafted in a session that has nothing else, but no gates are opened.
5. **Do not spend a user question on Stage B.** On the 67M it labels copy, lowercase and the ciphers as construction
   already does; only E54's elicit is open, and item 1 opens it for $0 with more seeds. Ask reuse-lead, agent to agent,
   to confirm the bar below matches its definition; Stage B's patch half joins the table when the reuse line gets to it.

## The first concrete item: does the half-known map carry to keys the fine-tune never saw?

*States.* E54's elicit map and its matched cipher, E54's recipe unchanged (E12's, KL penalty, step 10; the cipher at the
checkpoint whose ‖δW‖ matches), trained on 24 of the 32 keys, seeds 0-2. The 8 held-out keys are never in a training
window. Three elicit runs is the smallest n that lets P2's "every open run" mean something; the cipher is the label's
negative control (its held-out values are unknowable, so it must read "new"). *Measured:* the held-out-key gain H (the
fine-tuned model's log-odds of W₀'s rank-2-20 value at each held-out key, minus W₀'s, on E54's held-out windows), the
trained-key gain G, then r, the stand-in ratio and S on the same states as the step-2 table. *Label rule* (copied from
Stage B's patch rule, `reports/plans/2026-10-29_plan_reuse-stage1.md` § Stage B): "reuses" iff H ÷ G ≥ 0.5; "new" iff
H ≤ 3 s of its row bootstrap; else "neither". *Compute:* E56 trained three seeds of this pair and read them within one
debug job; expect 15-20 min on an L40, $0.

| number | W₀ | do-nothing (δW in Δ) | positive control | lean |
|---|---|---|---|---|
| trained-key gain G | 0 exactly | the full model's G, same for every placement | E54 10.8 at step 10 on 32 keys; must read ≥ 8 on 24 keys, else the state is too small to label | high |
| held-out-key gain H, elicit | 0 exactly | the full model's H | none: this is the open reading | reuses about 50% |
| H, cipher | 0 exactly | the full model's H | must read "new" (≤ 3 s); a cipher that reads "reuses" means the rule is broken: stop | high |
| r (route split) | undefined (gain 0) | not read | 0.76 ± 0.03 elicit, 0.24 cipher (E54, E56 five seeds) | reads as E54 |
| stand-in ratio | undefined | undefined; g alone reported | 0.95 elicit, 0.29-0.34 cipher (E53 A2) | reads as E54 |
| steering share S | block 0 reads 0 ± 1e-6 | never reads the decomposition | 0.38 / 0.23 (E54) | baseline column |

*Control first:* the zero-update check, and E54 elicit_s0's r reproduced within 0.03 on the full 32-key state, before
any new fine-tune. *Prediction to file* (the step-2 plan's P2, on these runs): the r-label and ratio-label equal the
held-out label on every elicit run with a label other than "neither". *Blind to:* a map rebuilt by a new route that
nonetheless generalises (reads reuses on both sides); the granularity limit (a new circuit from old pieces reads
elicited to VPD; say so in every claim). *Noise floor:* r seed spread 0.055 over five seeds (E56); H's row bootstrap.
*Departure from the literature:* Jain's Def 2 with the user's bounded g, behaviour only, no interchange patch; stated.

**Both outcomes are worth having.** Agree (label reuses, r and ratio high): the first label VPD did not make agrees
with it where steering and patching see nothing (E56), the headline case, at n = 3. Disagree (label new, r and ratio
high): the first concrete candidate for the granularity limit, a VPD error, or a label too strict; then the novelty
column on exactly these states, and the costed second-seed card the programme reserved for this moment.

## What would change this

- E54's recipe on 24 keys does not reach G ≥ 8: raise the step count once, matched by ‖δW‖ as E54; if it still fails,
  the family cannot be built this way; fall back to item 2 and raise Stage B as the user question after all.
- All three elicit runs read "neither": the bar is not reading anything; report descriptively, ask reuse-lead for its bar
  before touching ours (never refile).
- The user answers `decision-e55-c2` with option 3: item 1's states and labels go straight into Stage B's table, nothing
  wasted.
- A second decomposition seed turns out to cost under a few GPU-hours: it jumps ahead of item 4.

## The one question for the user, if any

None is needed to start items 1-3; the fine-tunes are E54's recipe on fewer keys, $0, and the label is read only after
filing. The single thing that is the user's call: **the re-gating line applying the reuse line's label rule to its own
67M states, with the behavioural half only, instead of waiting for Stage B.** Put it as one line with a recommendation
("we proceed unless you object"), beside the card, not as a blocking ask.

**Not run.** Nothing; this is advice.
