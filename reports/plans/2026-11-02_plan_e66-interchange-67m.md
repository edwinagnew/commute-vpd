# Plan: on the 67M, does a fine-tune run through a mechanism the original model already has? (E66)

*Baseline-lead, 2026-11-02. Decision: `decisions.md` row "Interchange patch on the 67M" (option (b) of
`decision-e65-or-stage-b`). Plan only: no code, no predictions filed, nothing run. The user reads this first.*

**Reviewed by Fable (2026-11-02): `reports/plans/2026-11-02_advice_e66-review-FABLE.md`, which advises against running this as written. Its changes are not applied in the sections below; the user chose its option A, the feasibility probe, written up in "Amendment 1" at the end, which holds over them.**

**In plain words.** We move the original model's internal state from two different contexts into a fine-tuned model,
and ask whether that model responds more strongly than the original does. We also switch off the neurons the original
model uses most for the trained answer, and see how much of the fine-tune's gain goes. Where both agree, a 67M
fine-tune gets a causal label, so the method table can score the decomposition against cheaper methods.

**Claim.** A 67M fine-tune is *reuse* when it strengthens the dependence the original model's own state already gives
the trained answer and the original model's top neurons carry its gain, and *new* when its gain sits beside both.
**Granularity limit (user, 2026-10-29).** A new circuit from old pieces reads "elicited" to VPD, and it can read reuse
here too if it multiplies on the original model's state. "Reuse" means old parts at the level of a block input and
MLP neurons, not one circuit shown unchanged.

**Null.** The two readings do not leave their negative control on any fine-tune, or they follow update size or prior
and not kind. Then no label is given, and the rows stay "only by assumption".

**Three predictions that separate (leans; rules filed before the run).**
- **P1, E62's states.** The patch ratio reads high on the copy map on at least 2 of 3 seeds, and low on the lookup on
  all 3. Lean 65%: unpatched, the pair moves B by 6-10 log-odds in the state against 1.6 in W0 (`TRIED.md`, E62).
- **P2, lowercase against its ciphers.** Lowercase reads reuse and both ciphers read new. Lean 55%.
- **P3, equal-size key maps.** The half-known map and rank-1 read reuse, and the cipher reads new. Lean 35%: E58
  Amendment 1 found all six half-known states act as lookups at unseen keys. If P3 fails, r's high reading there
  follows something other than mechanism, which is the result the merged line exists to find.

## Design check (`/design`)

- **Goal.** `GOALS.md`, "If no simple Φ works": new capabilities or strengthening. It gives the merged line's truth on
  rows where r and the stand-in ratio already have readings.
- **Nearest tried.** E62 (labels by head ablation, regating-lead's). Stage A's patch (inside the state). E58's held-out
  carry (evidence only, `decisions.md` row 114). E55 C1b ruled out cross-model patching because patching the base's
  activations into a state "tests where the change sits". This plan's differenced ratio is its answer to that; a
  fine-tune placed upstream remains a blind spot (below).
- **Patch or step.** A step: the first causal truth that varies inside the 67M rows.
- **Decisions.** Relies on rows 112, 114 (mechanistic elicitation, held-out as evidence) and 118. Row 81's "reuse needs
  a causal claim" is met with a patch and a knock-out. No departure.
- **What changes next.** See the tree: rows move and are scored, or the user learns the 67M has no such label.
- **Control first.** Stage 0 reads W0 alone; stage 1 reads the built controls; E62's states come before any row.
- **Class.** Class 1 (saved states) except the one trained control and stage 3's retrain, which class 2 covers. Row 118
  says nothing runs until the user has seen this. Not gated: $0.
- **Overlap with E62, in one line.** E66 reads E62's six saved states and quotes its share lost from `run.json`; it
  does not ablate head (2,4), rerun or extend E62, and E62's verdict stands as filed.

## Why not the obvious patch

Stage A's patch reads 0 by construction on a state that does not answer the patched quantity, and stays inside the
state. Moving to a W0 donor across keys has the opposite fault. On a "M key" prompt the key's identity is in W0's
state at every block, and every lookup reads it, so a cross-key patch reads high on any fine-tune whose update lies
downstream. That is the risk row 114 names: any fine-tune keyed off something the model represents reads as reuse.
So E66 **holds the key and the target fixed and varies only the context**, and compares the state's response with
W0's own. On this design a lookup that adds beside W0's answer reads about 1, not 0.

## The two readings

### Interchange patch: the patch ratio T

| part | definition |
|---|---|
| donor | W0, on two contexts with the same key, marker and target: x_hi and x_lo |
| recipient | the state (W0 + δW) on x_lo, and W0 itself on x_lo for the denominator |
| site | block l's input (`steering.Net` site `in<l>`), fp32 |
| position | the answer position only; every other position keeps the recipient's own run |
| block | l*: the earliest of blocks 1-3 where W0's response is at least 0.5 nats and 3 s, fixed from W0 alone. Blocks 1-3 are all reported; block 0's input is the embedding, which no update touches |
| metric | log-odds of the target (`steering.logodds_at`). Text maps: E53's log p(lowercase) − log p(capital), and log p(π(x)) − log p(x) |
| response R(m) | metric in model m on x_lo with W0's state from x_hi at the site, minus the same with W0's state from x_lo |
| T | mean R(state) / mean R(W0) over the map's pairs, a ratio of means |
| pairs: key maps | for each key, 64 fresh contexts from held-out Pile rows (`kv.windows`, rows with index % 5 = 0); rank by W0's log-odds of that key's target; pair top quarter with bottom quarter. Chosen from W0 alone, never from the state |
| pairs: E62 | x_hi is the cue window; x_lo is the same window with B overwritten by another pool token (E62's own cheap-solution edit). The lookup uses the key-map rule at its marker |
| pairs: text | the row and its lowercased or ciphered twin at the same index, where the query token is identical in both |
| noise floor | s = max(pair bootstrap sd, seed sd) |
| blind to | an update wholly upstream of l* (both arms carry W0's residual, so T reads the control's value); attention at other positions; reuse of W0's key features as input to a new readout, such as the class map, since the key is fixed; a mechanism W0 has but does not run on this input |

**Why T can read either way on both kinds.**
- A new state cannot read 0 by construction: pairs come from W0 alone, and an additive lookup passes W0's difference
  through, so T sits near its negative control, about 1. It can read high if new wiring multiplies on W0's state.
- A reuse state cannot read high by construction: the key is fixed, so no lookup can follow it, and a reuse state whose
  change sits upstream of l* reads like the negative control. So a low T alone never earns "new".

**Bars, per map.** n0 = T on the map's built write (below), p₊ = T on the sharpening fine-tune on the same pool.
lo = n0 + max(3 s, 0.1 (p₊ − n0)), hi = n0 + 0.5 (p₊ − n0), the shape of `interchange.bar` with a bootstrap s in place
of its binomial term. High iff T > hi, low iff T < lo, else between. Text maps have no key to write, so their n0 is the
largest key-map n0 (close call; the alternative, a hand-written bias, adds a control nobody else reads).

### Second reading: the units share U (shares nothing with steering)

The K neurons of the 12288 with the largest |gradient × activation| of the map's metric at W0, over the map's contexts,
are mean-ablated in W0 and the state alike, as E62 does for a head. U = 1 − G_abl / G, with G the state's gain over W0.

- K: the smallest of E56's grid where W0's rank-1 log-odds falls by at least 1 nat, fixed once on W0. Expect 64 to 256.
- Null: 5 draws of K neurons matched on activation decile. Positive: the top K of the same score taken at the state,
  U_own. Bars: `bar` shape with n0 = the draws' mean and p₊ = U_own.
- Blind to: changes carried by attention, and any gain routed through neurons W0 does not rank highly for this answer.

**Label.** Reuse iff T high and U high. New iff T low and U low. Otherwise no label: report "patch only" or "units only",
never "new" (Fable's finding 2, applied).

### Which methods share a readout with these

| method | with T | with U |
|---|---|---|
| steering | same site, other side of the update: it moves the state's mean shift into W0, while T moves a W0-minus-W0 difference. Not one number, but high steering at block l means the update is upstream of l, where T is blind. Mark its cells "shares a site with T" | none |
| neuron patch, pruning | none | same neurons; different selection and readout. Mark their cells |
| prior, distance, size, MLP share, r, ratio | none | none |
| unpatched context ratio C (state's contrast over W0's on the same pairs) | nearly all of it; reported as a reference, never scored as a method | none |

So U is the independent reading for steering, and T for neuron patching and pruning. Agreement of both is what labels.

## Numbers and what each should read

| number | zero update | do-nothing | W0 | negative control (write) | positive control (sharpen) |
|---|---|---|---|---|---|
| R(W0), per map | same as W0 | same as W0 | ≥ 0.5 nats or void; lowercase about 2+ (E53's bars) | n/a | n/a |
| T | 1 within 1e-6 | as the state's: it reads the target model, and do-nothing leaves those weights | 1 by definition | near 1, some below from the last layer norm | above n0; at most about 2 at temperature 0.5 |
| U | void (G = 0) | as the state's | void; W0's own fall reported for K | near the draws: its neurons were not chosen for the write | not used |

Also reported, one each: C per state, the state's gain G, T at the other blocks. Nothing else.

## Positive control: what exists and what does not

**No trained 67M state of independently known kind exists, so a pass here is weaker than a pass on the counting set.**
The candidates:

| candidate | known how | use |
|---|---|---|
| **Write at block 3** (`perturb.key_write`, retargeted from E46's block 1 to `h.3.mlp.down_proj`, gain matched by `perturb.match`) | new by construction: a per-key table written by hand, downstream of every l* | n0, per key map |
| **Sharpening fine-tune**: train W0 by KL to its own distribution at temperature 0.5 on Pile rows, Adam 3e-5, 30 steps | near-known: no map exists to learn but W0's own output. Its route is not checked | p₊. One new training run, about a minute |
| **Amplify edit**: W0's top K neurons' down_proj columns × (1 + α), gain matched by `revival_kv.fit_alpha` | reuse by algebra, and only by algebra | apparatus check only: tells a broken patch from a sharpening fine-tune that learned little |
| E62's copy map and lookup | copy map old, old, between ("mixed", not settled as filed); lookup new on every seed, new only relative to head (2,4) | reference (P1), not a control |
| Lowercase | built to elicit; W0 performs it (E53 A0) | a scored row, circular as a control |
| Recovery fine-tune after a perturbation | rejected: recovery may take a new route, so its kind is no better known and it costs two runs | not used |

The limit this sets: "reuse" means "reads like a fine-tune to W0's own sharpened output, and agrees with E62 where E62
is settled". It does not mean verified against a trained state of known mechanism.

## Rows and states

Every state below is on disk under `~/commute-data/results-large/`. Readings are from `table.json`.

| row | states | r | ratio | prior | steering | size | cost |
|---|---|---|---|---|---|---|---|
| Copy at new marker vs lookup (reference) | `induction_cue/{copy,lookup}_s{0,1,2}` | 1.09-1.10 vs 0.38-0.42, unscored | not read | factor 1.32 | not read | 4.6 vs 5.5 | seconds |
| Half-known vs cipher, equal size; own steps | `keydependent_elicit/{elicit,cipher}_s{0,1}`, `cipher_own_s{0,1}`; `keydependent_patch/{elicit,cipher}_s{2,3,4}` | 0.76-0.77 vs 0.21-0.27 | 0.95-0.96 vs 0.29-0.34 | factor 1575 | 0.25 vs 0.18-0.24 in log-odds | matched | seconds |
| Rank-1 vs cipher; copy vs cipher, step 10, 24 keys | `open_pair_instruments/{rank1,copy,cipher}_s{3..7}` (seeds 0-2 also saved) | 0.84-0.90; 0.83-0.86 vs 0.22-0.25 | 1.09-1.18; 1.00-1.07 vs 0.30-0.34 | log P −5.0; −8.9 vs −14.3 | 0.18-0.25; 0.60-0.68 vs 0.15-0.22 | matched | seconds |
| Copy maps vs random maps | `regating_table/{ident32,ident32b,rand32,rand32b}_s{0,1}` | 0.76-0.80 vs 0.27-0.29 | 0.87-0.97 vs 0.36-0.41 | not read | 0.72-0.80 vs 0.10-0.19 | does not separate | seconds |
| Lowercase vs ciphers | `performed_anchor/dW_{lowercase,cipher,broad_cipher}_s{0,1}_step300` (half-precision files, cast to fp32) | 0.75-0.77 vs 0.12-0.17 | 0.92-0.93 vs 0.26-0.29 | not read | 0.30-0.32 vs 0.09-0.16 | not read | seconds |

**Not on disk, stage 3b only.** The 96-key rows. `open_pair/amend3/` holds step-10 and step-30 states of seed 0 for the
copy map, half-known and rank-1, and no cipher. `amend6/` and `amend7/` hold windows only, so copy map vs the
prior-matched cipher, and the class map vs its twin, must be retrained. The prior stops separating on the first; prior
and size both stop on the second, so they matter most. Fine-tunes take about 2 min each; E58's reruns reproduced r
exactly, and that check runs first.

## Run order, stops and cost

MATS `debug` QoS, one L40, account `c11`, through `/pod-runner`. Only stage 1's control and stage 3b train anything.

| stage | what | stop if | wall, $ |
|---|---|---|---|
| 0 | W0 only, alone: identity patch (same text in both arms) reads 0 within 1e-6; zero update T = 1 within 1e-6; pools, R(W0), l*, K | either identity misses; R(W0) under 0.5 nats at every block on half the maps or more | 5 min, $0 |
| 1 | Write, amplify, sharpen on the key maps' pools; write's U | amplify's T not above write's by 3 s (patch broken); hi not above lo on the rank-1 or half-known pool | 10 min, $0 |
| 2 | E62's six states: T, U; reproduce their G from `run.json` within 0.02 | P1 misses: stop and tell the user, read no row | 3 min, $0 |
| 3a | The saved rows: 41 states, reproducing each G from its source within 0.02 first | none; report | 15 min, $0 |
| 3b | If 3a labels at least one contrast: retrain the 96-key rows (11 fine-tunes; two jobs if needed), then T, U | r off its filed value by over 1e-4 | 30 min, $0 |

Total $0, at most about 1 h of `debug`. New code: one module plus `experiments/interchange_67m/run.py`, 300 lines or
fewer. Before the run: predictions filed with these rules, then a fresh-agent code review, because the labels enter the
table.

## Rules against their edge cases

| case | what happens |
|---|---|
| mean R(W0) near 0, or under 0.5 nats or 3 s (zero denominator, gain near noise) | that map is void at l*; no T; never a low reading |
| mean R(W0) negative | void; the pairs were chosen for a positive contrast, so report it as an anomaly and run it down |
| one context for a key, or no top and bottom quarter (one-element pool) | that key gives no pairs; under 200 pairs or under 8 keys, the map is void |
| individual pairs with R(W0) ≤ 0 | kept: a ratio of means has no per-pair denominator; selection is on the unpatched contrast |
| state's gain G under the map's floor (E53's F, at least 1 nat on key maps) | "too small to read": no label, and U cannot divide by it |
| T below n0, or negative (update crowds out W0's response) | counts as low only if not negative; a negative T gets no label and is reported |
| hi not above lo | the map's T is void, so its states get no label |
| U_own not above the draws by 3 s | K is too small or the gain is in attention; U void, T alone gives "patch only" |
| G_abl above G (U negative), or a tie at a bar | low and reported; a tie is between, strict as `interchange.label` |
| a row's side with mixed seeds, or one seed | a side is labelled when at least 2 of 3 seeds, or 4 of 5, share a label and none has the opposite; else unscored |
| a row with both sides the same label | "no contrast": the row moves to "by causal check", but no method can separate on it |

## What each outcome does next

1. **Stage 0 or 1 stops.** The patch cannot be built on this model. Report to the user; E65 returns as the next option.
2. **P1 misses.** T disagrees with the one causal label on the model. Stop and tell the user with C and U beside it.
   U agreeing with E62 and T not would mean T's blind spot is the larger one; the card asks whether to go on with U alone.
3. **P1 holds; every row reads new** ("no state reads reuse"). Then reuse on the 67M is seen only for copying, and r
   and the ratio read high on labelled-new states. That is VPD losing on those rows: r follows something else, and the
   prior is the first suspect. Log it plainly; re-gating's lead is told, since its readings are the row.
4. **P1 holds; a row has a contrast.** Score it (below), and run stage 3b. VPD's first fair win needs a labelled pair
   where r separates and cheap methods cannot. E62's pair and the prior-matched row can show that against the prior
   only: size still separates both as a labeller, though E62 Amendment 2's scaled updates left r's gap in place. The
   class map vs its twin is the one row where neither can, and T is blind there.
5. **T and U disagree across most rows, or T follows ‖δW‖** ("the patch cannot separate"). Rows stay as they are. Report
   the disagreement by kind ("patch only", "units only"), and E65's card comes back with the reason.
6. **Class map and its twin.** T is blind to this row by design, so "no contrast" there says nothing about r's miss;
   the plan does not claim otherwise. A patch for that row, swapping keys within a class, would be a separate plan.

## How the table's rows change

Each row whose sides are labelled moves its truth to "by causal check", quoting T, U and both bars. Each method's cell
is rescored by the same E51 separation rule against the labelled sides, through a new reader in
`experiments/method_table/build.py`. Where a label contradicts the build, a separating method is now separating the
wrong way, and the cell says so. Steering's, neuron patching's and pruning's cells carry the sharing marks above. Rows
with void maps or mixed sides stay "only by assumption" with a note. No label is taken from a method being scored.

## Not run

Nothing. Not planned at all: a key-swap patch for the class map; the 400-step or a full decomposition; head ablation
beyond E62's; any Qwen state; E65; the ladder row (rank-1 vs cipher, steps 10-300; its states are not listed in the
saved files); half-known vs its shuffle before stage 3b.

## Departures from the literature

- **Prakash et al. 2024, cross-model activation patching (arXiv 2402.14811, `lit_review/summaries/arxiv_2402.14811v1.md`).**
  They patch head outputs from the fine-tuned model into the base on the same input and read improvement. We patch from
  the base into the fine-tuned model, between two contexts, at one residual position, with no circuit. The ratio to
  W0's own response is ours.
- **Jain et al. 2024, Def 2.** They name f_C through a trained linear readout. We name the dependence by W0's own answer
  across contexts with no probe, which E59 and E64's unreadable readouts argue for. We keep the user's causal requirement
  (`decisions.md` row 81). This is narrower than Def 2: a new g on an old representation with a fixed key is unseen.
- **Olsson et al. 2022, as E62 uses them.** Mean ablation stands in for their zero ablation, and on neurons, not heads.
- **Interchange-intervention papers.** A grep of the repo finds none cited, so none is claimed as a source.
- **Unsourced here.** The temperature of 0.5, the 0.5-nat floor, 64 contexts and the 3 s and 0.5 bars (the last two from
  E55's rule). The sharpening fine-tune's recipe borrows E12's optimiser with a new loss.

## Amendment 1: feasibility probe (2026-11-02)

*User's decision, option A of `reports/plans/2026-11-02_advice_e66-review-FABLE.md`. Where this section and the sections above
disagree, this section holds. Findings 1, 3, 4, 6 and 8 of the review are applied. Not gated: about 10 to 15 min of `debug`, $0.
Design check: inside the plan the user chose; class 1 (saved states) with one trained control, which class 2 covers.*

**In plain words.** One job asks whether the two readings can see anything at all before anyone files a rule for a row. It reads
the original model, three hand-built controls and the fourteen saved states of the copying and random maps, and it gives no state
a label. A decision card afterwards says whether the 41 other states get filed rules or E65 returns.

### What runs

| stage | what | stop |
|---|---|---|
| 0, control, alone | W0 and the zero update on six maps' pairs: identity patch, zero update, R(W0) and R_off(W0) at blocks 1-3, l*, share of pairs with the target earlier in the window; K; the filed steering shares | identity or zero update over 1e-4; half the six maps void at every block |
| 1 | Write, amplify and sharpen per map: T, T_off, Q at blocks 1-3, U on write and amplify, both bars | amplify's T not above the write's by 3 s on half the readable maps |
| 2 | E62's six states: G re-read, measured steering share, T, T_off, Q, U, marked exploratory | G off its filed value by over 0.02 |
| 3 | The eight `ident32*`, `rand32*` states: the same, with filed steering | as stage 2 |
| 4 | E62 Amendment 2's size-swapped copies (each state times its filed alpha): T, Q, the same reading minus the unscaled; Spearman of T and of Q against ‖δW‖ over the 20 states | none |

Six maps: `copy`, `lookup` (one pool from `copying.cue_windows`, finding 6), `ident32`, `ident32b`, `rand32`, `rand32b` (`kv.windows`). All
pools come from rows with index % 5 = 0, at 64 contexts per key. The key maps' saved windows were cut with no row split; 500 windows
of 32768 rows make a shared row rare, and a window here is a fresh span.

### Definitions changed from the plan

- **T_off and Q (finding 3).** T_off is T for an off-map target: the next key's word or value in the pool's key order. Its pairs are the same
  pool ranked by W0's log-odds of that target, top quarter against bottom quarter per key, so its denominator has the same footing as
  T's. Reading Fable's "same contexts" with T's own pairs would give a denominator near 0, since those pairs were not chosen for this
  token. **The map's own target is taken out of the odds in this reading.** Otherwise a state whose trained target dominates makes the off-map log-odds move
  with its target's logit, not with the context. Q = T / T_off.
- **Amplify is the high control (finding 3).** W0's K most positive-scoring neurons at blocks at or after l*, down_proj columns times 1 + α,
  α by `perturb.match` to the map's mean saved gain, capped at 8 and reported as "cap". Two departures from the plan's table: positive scores
  only, since amplifying a neuron that lowers the target lowers the gain; and blocks at or after l*, since upstream neurons are what the
  patch overwrites, and a control the patch cannot see tests nothing. `revival_kv.fit_alpha` is not used: it needs an "off" bound nothing reads here.
- **The null of U (finding 4).** Five draws, each matching every top neuron on the decile of mean |activation| and on the decile of the mean over the map's targets of
  |down_proj column · unembedding(target)|. An empty cell falls back to the same activation decile, then any free neuron.
- **K (finding 8)** is fixed once on the half-known map's held-out windows. U reads every 8th window of a pool, 256 in all.
- **The blind mark (finding 1), one difference from the review.** Each state's rows carry the steering share at l* as Fable asks. The mark "blind" reads the largest
  share at any block up to l*, not the share at l* alone. The patch overwrites what every earlier block wrote at the answer position, and filed shares
  are not monotone: `ident32_s0` reads 0.11, 0.75, 0.21 at blocks 1, 2, 3, so at l* = 3 the share at l* alone would call it readable with three
  quarters of its gain upstream. Marks are "blind", "readable" or "unknown", and none is a verdict.
- **Identity tolerance 1e-4, not 1e-6.** E52's stage 0 measured 1.7e-5 nats per window on the same edit path at fp32, so 1e-6 would stop on
  rounding. The run measures it again as `identity_max`.
- **A tie at a bar is "between"** on both T and U (finding 8's last-but-one point).

### Every number and what it should read

The noise floor s is the pair bootstrap sd of the ratio of means, 2000 resamples. The controls have one seed, so theirs is the pair sd only; the
report adds the seed spread on saved states, where each map has two or three seeds.

| number | W0 | zero update | write | amplify | sharpen | noise floor | blind to |
|---|---|---|---|---|---|---|---|
| identity max | 1e-5 or less | n/a | n/a | n/a | n/a | itself; bar 1e-4 | everything but the patch path |
| mean R(W0), blocks 1-3 | copy map near 0 at 1, 2, then l* = 3; others unknown, the probe's first question | same | n/a | n/a | n/a | pair sd; floor 0.5 nats and 3 sd | a mechanism W0 does not run on this input |
| T at l* | 1 | 1 within about 2e-4 | 0.5 to 1 + 3 s | above write by 3 s | above write by 3 s, at most about 2 | s | an update upstream of l*; attention at other positions; a new readout on a fixed key |
| T_off | 1 | 1 | near T | near 1 or above | near T | s_off | a state that changes the off-map target on purpose |
| Q at l* | 1 | 1 | within max(0.2, 3 s) of 1 | above write by 3 s, a lean | within max(0.2, 3 s) of 1 | s_Q | what on-map and off-map responses share |
| K, W0's fall | 64 to 256 | n/a | n/a | n/a | n/a | none; read once | W0's half-known map only |
| U | void | void | near the draws | above the draws by 3 sd | not read | sd of 5 draws | attention; neurons W0 does not rank highly |
| G re-read | n/a | 0 | matched to the saved gain within 5%, or "cap" | same | reported | filed value within 0.02 | n/a |
| Spearman T, Q against ‖δW‖ | n/a | n/a | n/a | n/a | n/a | at n = 20 about 0.23; not filed | a size effect within one map; the row beside it, T minus the unscaled, reads that |

Exploratory, on the fourteen states and six scaled copies: T, T_off, Q, U, the mark. No expected value is filed, on purpose, and no row label is given.

**The sharpening control is the only training.** Temperature 0.5, `finetune_to_reference` against a frozen copy of W0 with logits divided by 0.5, KL over every position of 32 plain
Pile spans a step, Adam 3e-5, clip 1.0, 30 steps from training rows, bf16 autocast as every fine-tune. Two forwards and one backward a step, so about 10 s on an L40, unmeasured. It stays. It is
trained on plain text, not on the marker, so it may not sharpen at the marker in 30 steps. If its T does not clear the write's, the report says too weak, and Q then has no check against sharpening; nothing stops.

### Rules run against their edge cases

Run on hand-made numbers in `tests/test_ctxpatch.py`; the table is `edge_cases` in `predictions.json`.
- **Zero denominator, or R(W0) under 0.5 nats or 3 sd.** T at that block is nan, never low; l* skips it; the map is void if every block is. The copy map's expected blocks 1 and 2 land here.
- **W0's negative response.** Cannot pass the floor. Printed.
- **Void pairs.** A key with under 4 windows gives no pair; under 200 pairs or 8 keys voids the map, but its U is still read.
- **T_off void, not positive, or crossing 0 in a resample.** Q is nan and reads void, neither pass nor miss.
- **Every map void on amplify's T, or no K.** Stage 1 stops: nothing can be read. A tie at half the maps stops, as the plan's "half the maps or more" has it.
- **A negative U, a gain under 1 nat, a zero update.** Negative is kept; the other two are void.

### Cut, and why

- **Stage 3a's 41 states, 3b's retraining, P1 to P3, the labels, the method-table reader.** Option A. The plan's P1 stop would fire on finding 1's blindness for the wrong reason (finding 8).
- **Text maps.** The eight states read here are key maps and E62's, so finding 8's token-bias write for the text pool is not needed yet.
- **Finding 5's explicit pairs** (target written earlier in x_hi). The probe reports each map's void or not and the share of pairs with the target earlier, which says whether stage 3a needs them. Adding them now doubles the pair sets before anyone knows.
- **Finding 2's head check.** E62's share is already filed; "new" is not written here at all.
- **The unpatched context ratio C.** Reported in the plan as a reference only, and nothing here reads it.
- **Filed steering on 27 of the 47 states.** Only E52, E54 and E60 filed per-block shares, 20 states in all (E56, E58's instruments filed a best block only). Stage 0 prints the 20 and null for the rest; E62's six are measured in stage 2, in log-odds on their read windows, since no file has them. So the review's "for every saved state" is met on 26 of 47; stage 3a would have to measure the other 21 on their own windows.

### What the card after it decides

If l* is under 3 on the key maps, or the copy maps read above the write with mark "readable", stage 3a goes with rules filed from these numbers. If most marks read "blind", T and Q follow ‖δW‖ (|ρ| over 0.5, Fable's bar), or U sits high on both the copy and random sides over its matched null, E65's card returns. Its best outcome for VPD stays a tie with update size and a win over the prior only (finding 7).

Code: `experiments/interchange_67m/run.py` (about 270 lines), `src/commute/ctxpatch.py` (about 520), `tests/test_ctxpatch.py`, `cluster/interchange_67m.sbatch`. Over option A's "about 200" because the three controls, two pair kinds and the matched null are each their own code; all of it is what stage 3a would call. No frozen caller is changed: nothing existing was edited.

### Not run

Everything in this section: coded and filed only. No model was loaded. A fresh review of the code precedes the job.

### Departures from the literature

The plan's list stands. Added here: the off-map ratio Q, the masked odds and the alignment-matched null are ours, with no source; temperature 0.5 and the 0.2 and 0.5 bands in `predictions.json` are unsourced and marked so there.

### Note, 2026-11-02: after the code review

*The code review of commit c649f2e, relayed to the writer by the lead on 2026-11-02. No file for it was in `reports/reviews/` when this was written. Nothing had run, so `predictions.json` was changed in place. This note holds over Amendment 1 above where they differ.*

**Changed in the rules, and why.**
- **C0 replaced, C1 deleted.** The old identity patch and zero update could pass with the x_hi sites never touched, and T = 1 at a zero update holds by identity, so it controls nothing. C0 now writes x_hi's block-l input at every position, with the stored x_hi site at the answer position, into x_lo's run, and asks that it reproduce x_hi's read within 1e-4 (E52's fp32 figure, 1.7e-5). A nan now fails whichever position it holds. The zero update stays in words only.
- **C6.** Sharpening is scored against 1 (T − 1 over 3 s), not against the write, and its Q is void unless that holds. So "too weak" can fire honestly. Its gain is recorded on each map. The trained control stays: dividing logits by the temperature would pass by algebra and test nothing about a weight state.
- **C5.** Amplify's T and Q are void unless its gain match has status "ok" and the map's write is conditioned, so stage 1's stop does not count such a map.
- **C4, write conditioning, new.** `perturb.write_fit` goes into the write's record and the log. Bars: cond(A) at most 16.5, mean relative error at most 0.33, twice E46's worst reading in `regating_table/results/amend3.json` (cond 6.4 to 8.2, error 0.16 to 0.17). Twice, not once, is the writer's choice: that write sat at block 1 on the key layout, and this one sits at block 3, on the lookup at the cue layout.
- **Q's band.** Q reads void when 3 s_Q is over 0.2, not a band widened by its own noise.
- **The off-map mask.** It now holds the map's target on x_hi and on x_lo (two tokens per pair). On the copy map they differ, and x_hi's B entering x_lo's odds would move the off-map read.
- **U.** Void when 2K exceeds 12288, a new edge case, since no null fits. Fallbacks are stored per draw. The share of amplify's neurons in U's top K is reported in C7's reading.
- **Blind-to, added.** Pairs are picked on W0's own metric, so R(W0) carries selection inflation and T can drift below 1 with update size. That feeds the reading of T against ‖δW‖, and the predictions' note on that reading says so.

**Deleted.** The bars and U at the state's own top K, so no bar is computed by the run; Spearman, which the analysis computes from `run.json`; the listing of the 33 states not read and the E54 and E60 steering sources, so stage 0 no longer prints them and the card's listing is a `jq` query. Stage 4 now stores Q beside T unscaled.

**Consequence for "What the card after it decides".** Its numbers come from the analysis, from `run.json`: bars, |ρ|, the filed shares of the other states.
