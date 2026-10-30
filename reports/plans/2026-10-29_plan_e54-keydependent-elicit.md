# Plan E54: a key-dependent map the original model already half-knows (option C of `decision-e52-p4`)

*2026-10-29, baseline-lead (plan writer). Draft; nothing run. Sources: `reports/decisions/2026-10-29_decision_e52-p4.md`
(option C, user: "go with A then C"), `reports/plans/2026-10-29_plan_baselines-scoping.md` ("the central design
problem", E52), `reports/analyses/2026-10-29_analysis_e52-baseline-neurons.md`. Not gated: $0, one or two cluster debug
jobs of minutes each (E46 rebuilt, fine-tuned and route-split all eight E12 states in 86 s).*

## The request, in the Terms' words
Build a third key → value task in E12's format (marker M, key x, target table[x]; `src/commute/kv.py`) whose table the
original model W₀ already gives a measurable probability, filed before any fine-tune. Fine-tune it with E12's arm-A
recipe and seeds (all 24 matrices, Adam 3e-5, batch 32, KL λ = 1, seeds 0 and 1, `anchors.kv_finetune`), plus a random
cipher on the same keys. Read every state with E52's steering share S (B3, unchanged) and with the route split's
re-gating share r of the least-squares split state (E12/E46's reading), beside E12's eight copy and cipher states. One
mean vector cannot carry a map whose answer depends on the key, so S should fail on both new tasks; the question is
whether r reads the half-known map as "elicited" and the cipher as not.

## Design check
- **Goal.** `GOALS.md` → "teaches new capabilities rather than strengthening existing ones": a VPD reading of an
  update's kind is a claim only where a cheap method cannot give it.
- **Nearest tried.** E12 and E46 (r 0.76–0.80 copy vs 0.27–0.29 cipher); E52 (steering separates the same pair,
  0.77 vs 0.14, perhaps by construction: copy is input-independent, cipher is not). E53 (regating-lead) anchors a
  performed behaviour, but lowercase is one mode, which steering can push. New here: a pair that differs in kind but
  not in input-dependence, so steering and r can disagree.
- **Patch or step.** A step: the scoping plan's "mixed case" in the form E52's analysis names as the separating one.
- **Decisions.** E12's recipe, `kv.py` format and E51/E52's separation rule, unchanged. The label is behavioural (W₀'s
  rate), not `decisions.md`'s Jain et al. Def 1 (a linear readout) plus a causal check; that check is in "Not run".
- **What changes what we do.** r separates and S does not → the first case where VPD reads what steering cannot: the
  headline example. Both separate → steering was not blind to key-dependence here; the objection stands on this set
  too. Neither → r was reading input-dependence (or size), not kind, on E12 as well.
- **Control first.** Stage 0, alone, stop on a miss: the job re-reads saved `ident32_s0` and `rand32_s0` and must
  reproduce E46's r and E52's S. Nothing is reproduced from a warm start; the fine-tunes start from W₀ as in E12.
- **Class.** Option C was chosen by the user, but the task is new: confirm this check before code.

## The map and its W₀ rate (stage 1, its own job, committed before stage 2)
Candidates: E12's word-like tokens of Pile-frequency rank 100–1099 as keys, 16 train windows each, W₀'s mean next-token
distribution at the key (`kv.key_distributions`). For key x, its value v(x) is the most probable token v that is
word-like, v ≠ x, has P_W₀(v | M x) in [1e-2, 1e-1], is at W₀'s rank 2–20 (not the top token, so a vector that only
sharpens W₀'s guesses cannot get it), and is key-specific: P(v | M x) ≥ 10 × the median over other candidate keys of
P(v | M x'), which excludes "the"-like values that would be a prior shift. Values unique; 32 keys by a seeded draw
(seed 0) among those that qualify. Windows as E12 (`kv.windows`, 500, seed 0, train/held-out split).
**Filed rate:** geometric mean over held-out windows of P_W₀(table[x] | M x) in [5e-3, 1e-1] (selection was on train
windows, so some regression is expected). *Amended 2026-10-30 (user, after job 45963 stopped at stage 1, before any
fine-tune): floor lowered to 1e-3. Selection uses the arithmetic mean (0.032), the gate the geometric mean, which read
1.70e-3 held-out and 2.3e-3 on the selection windows: the gap is the averaging, not regression. Set after seeing the
value; same keys and values.* **Cipher on the same keys:** `kv.bijection` from E12's value pool, E12's gate
(P_W₀ < 1e-4 per pair). Edge cases: fewer than 32 keys qualify → K = that number if ≥ 16, the cipher at the same K;
under 16, or the band missed → stop and tell the user (widening the band is a new decision).
**Size match.** At E12's step rule the elicit task should learn in very few steps. The cipher is read at the
checkpoint (every step 1–30) whose ‖δW‖ (summed Frobenius over the 24 matrices) is nearest the elicit state's, same
seed, so r and S cannot differ by size alone (E51: effective rank tracked ‖δW‖; E36's tiny null read r = 0.82). The
cipher at its own E12 step is also read, as a column only.

## Numbers and their expected values (run order)
| # | number | W₀ | do-nothing | control |
|---|---|---|---|---|
| 0 | r and S on saved `ident32_s0`, `rand32_s0` | — | — | r 0.8024, 0.2733 ± 0.03 (E46's tol); S 0.750, 0.158 ± 1e-3 (same code, fp32) |
| 1 | held-out geo-mean P_W₀(target) | elicit in [5e-3, 1e-1]; cipher < 1e-4 | — | E12 copy ≈ 1e-4, cipher 4e-6–8e-6 |
| 2 | gain G (E52's log P, held-out) and floor | 0 exactly | — | ident32_s0 10.11 |
| 3 | S, E52's B3 (fit/score halves) | undefined (G = 0); block 0 = 0 ± 1e-6 | not read: S never reads the decomposition | E12 copy 0.72–0.80, cipher 0.10–0.19 |
| 4 | r of the least-squares split state | nan (gain 0) | not read (E12 and E46 read the split state) | E12 copy 0.76–0.80, cipher 0.27–0.29 |

Order: stage 0 alone (stop on any miss) → stage 1 (rates; stop on the band) → stage 2: fine-tunes, elicit before
cipher, each state's G checked against its floor (max(0.05, 2 × the seed difference)) before S and r are read.

## Predictions (separation as E51/E52: |mean A − mean B| > larger within-kind range, strict, **and ≥ 0.2**)
- **P1 (route split).** r separates elicit from the size-matched cipher, elicit higher. Lean: medium. Blind to which
  subcomponents carry it. Noise floor: E12's seed spread in r, ≤ 0.04.
- **P2 (steering).** S does not separate them, and both read ≤ 0.3. Lean: medium; rank ≥ 2 values block plain
  sharpening, but a vector might still lower a shared rival. Blind to anything input-independent. Floor: E52's seed
  spread in S, ≤ 0.08.
- **P3 (size).** r of the cipher at matched size is within 0.1 of its r at its own E12 step. Lean: medium. If it
  misses, r depends on update size and P1 is read only at matched size.
- **P4 (label, only if E52 Amendment 1 finds a working prompt format).** In that format, W₀'s prompting share at
  n = 16 separates, elicit higher. Otherwise not run.
Edge cases, for every rule: any state under its floor → that rule reads "unreadable", not pass or fail; G or a
denominator within 1e-6 of 0 → nan → unreadable; a tie at the bound fails (strict); two seeds per kind, so the 0.2
minimum stops two near-equal seeds making a tiny gap "separate" (E52 P4's lesson).

**Amendment before run (2026-10-30, code review of 8513c3b; no run had happened).** (1) P3: when the cipher's
norm-matched step equals its own E12 step on a seed (`same_state`), the difference is 0 by construction, so P3 reads
"uninformative", not pass, and the branch text says the size match collapsed (both tasks at the same step) and P1 is
read at the cipher's own step. (2) Gain floor: the seed-spread term 2 x |G(s0) - G(s1)| is used only for two seeds at the
same step; the matched cipher takes it from the cipher's own-step pair, elicit from its own pair, and if a pair's steps
differ the floor is 0.05 alone. (9) The "denominator within 1e-6 of 0 -> nan" rule is applied in `run.py` where S and r
are formed. Not filed rules, same review: reversed P1 or P2 is labelled "reversed" and not routed through the tree;
`at_bound`, an elicit that did not learn and `same_state` go into the branch text; stage 1 also records each key's
geometric-mean P; stage 2 stops if the block-0 shift exceeds 1e-5 nats (|S_block0| x G_score; in nats because the new tasks' smaller
G_score would lift E52's ~1e-6-nat rounding near a 1e-6 share, second review of 9be0aab) and checks predictions.json is
committed unchanged.

## Decision tree
- P1 holds, P2 holds → VPD's route split reads a kind steering cannot: write it up as the headline case; next, the
  same pair on Qwen (E45's anchors) and Jain's readout check on the elicit keys.
- P1 holds, P2 fails (steering separates too) → steering is not blind to this key-dependence; the objection holds on
  both sets: say so, and move the VPD claim to which parts carry the change.
- P1 fails, P2 holds → neither tells kind here; r on E12 was reading input-dependence or size: tell the user before
  any write-up that uses E12's r.
- P1 fails, P2 fails (S separates, r does not) → steering beats VPD here: report it as the objection holding.
- Stage 0 or 1 misses → stop and tell the user.

**E52 Amendment 1 (option A).** E54's own label does not need it: the elicit rate is measured at W₀. A changes how the
E12 rows beside it are read: if copy reaches share ≥ 0.5 in context, E12's copy is a second elicit row; if not, it is
labelled "mode switch, unverified" and E54's pair carries the comparison. A working format also turns P4 on.

## Cost and route
Two MATS cluster debug jobs (L40, `debug` QoS, $0) via a `pod-runner`: job 1 stages 0–1 (~5 min), job 2 stage 2
(4 fine-tunes, ~30 checkpoints held in host RAM, ~3.4 GB; reads ~10 min). Not gated (≪ $15, ≪ 3 h). Saved: the scored δW
(113 MB each: elicit and matched cipher, two seeds, plus the cipher at its own step unless it is the same state as the
matched one) and `windows_<task>.pt` in `results/large/`; summaries committed.

## Not run
Jain et al. Def 1 readout and the causal check on the elicit keys; neuron patching (B2) and the weight statistics (B1)
on the new pair; a matched random δW; the copy task on these keys; more than two seeds; Qwen.

## Amendment 1 (2026-10-30): steering's share in the route split's unit

*Coded 2026-10-30, not run. Code `experiments/keydependent_elicit/amend1.py` (helpers `steering.steer_at`,
`steering.logodds_at`, `elicit.score_amend1`); rules and expected values `results/predictions_amend1.json`, filed before
any run; job `cluster/keydependent_elicit_amend1.sbatch` (MATS debug, one L40, $0, ~2-4 min).*

**/design check.** Approved class 1 (analysis of what exists): a GPU pass of the existing harness (B3, unchanged) on
E54's six saved weight changes, $0, no new task, model or fine-tune. Goal: the analysis's ratio table compares an r gap
in log-odds with an S gap in nats; this measures the S gap in r's unit, the first item of the analysis's "Not run".
Nothing new is claimed beyond "the steering gap in r's units".

**What is measured.** On B3's scoring half (48 of the 97 held-out windows), with B3's vector and B3's block (both as E52
and E54 chose them): S_lo = (mean log-odds of the target with W₀ + v at that block − mean at W₀) ÷ (mean log-odds at W₀ +
δW − mean at W₀). Log-odds is `identity.log_odds`, the unit in which `route_split` forms r's gain T1 − T0. r is
stage2.json's, unchanged. The block is still chosen by B3's fit-half share in log P, so only the readout changes.

| state | S (nats, stage2.json) | S_lo expected | r (stage2.json) | r − S_lo expected |
|---|---|---|---|---|
| elicit_s0 | 0.380 | 0.25 (0.20–0.32) | 0.773 | 0.52 |
| elicit_s1 | 0.394 | 0.25 (0.20–0.32) | 0.758 | 0.51 |
| cipher_s0 (matched) | 0.228 | 0.22 | 0.241 | 0.02 |
| cipher_s1 (matched) | 0.185 | 0.18 | 0.212 | 0.03 |
| cipher_own_s0 | 0.114 | 0.075 (reported) | 0.265 | — |
| cipher_own_s1 | 0.066 | 0.045 (reported) | 0.243 | — |

Why these values: the independent check's bound (B3's nats rise over the all-window log-odds gain) is 0.235 and 0.242
on elicit, 0.22 and 0.18 on the cipher. The scoring half's log-odds gain is a little larger (~11.4 on elicit), and the
steered rise in log-odds exceeds the nats rise only on the few elicit keys whose steered P(target) is not small. On the
cipher the steered P stays near 1e-5, so the two units agree there.

**Predictions** (rules literal in `predictions_amend1.json`, edge cases run in `tests/test_elicit.py`):
- **P1-lo.** S_lo does not separate elicit from the matched cipher by E54's rule (gap > larger range, strict, and
  ≥ 0.2). Expected gap ~0.05. Lean: holds, medium.
- **P2-lo.** r − S_lo ≥ 0.3 on both elicit states (a tie passes). Lean: holds, high.
- **P3-own.** S in nats separates elicit from the cipher at its own step, elicit higher. Not a test: its inputs are
  stage2.json's S, which stage 0 reproduces, and the analysis already computed gap 0.296, range 0.048. Filed so that
  "Not run" item is scored by a filed rule.
Every rule: a missing state, nan (a log-odds gain within 1e-6 of 0) or an infinite value, or fewer than two values per
kind → "unreadable" → stop and tell the user.

**Positive control and stage 0 (first, alone, exit 1 on a miss).** Per state: the file's sha256 equals stage2.json's;
`steering.b3` gives stage2.json's block and S; `steering.steer_at` with the log P readout (the function S_lo is read
with, only the readout differing) gives the same S. Tolerance 1e-6 on S, from a measured floor: the unchanged path
reproduced E52's S in E54's stage 0 to all 16 digits on the same card model, and across batch shapes this model's
rounding is at most 9.5e-8 in share (stage2.json's block-0 shares, a zero vector). Reported, not a stop: the log-odds
gain on all 97 windows against route_split's G_logodds, the same quantity through the component model at batch 64
(expected within 1e-4; no floor between these paths is measured yet).

**What each number is blind to.** S_lo: anything one input-independent vector at one position cannot carry; which keys
carry the rise. Log-odds gives near-certain windows unbounded weight, as r's gain does, which is why this unit is read.
r − S_lo sets two shares side by side in one unit: S_lo is a share of the target model's gain on 48 windows, r of the
CI-masked model's re-gating term on 97. It is a table number, not a mechanism.

**Branches (they feed the analysis's ratio table only).**
- P1-lo and P2-lo hold → the table's "S estimated in log-odds units" row becomes a measurement: r gap (0.54) ÷ the S_lo
  gap, in place of "over 10x".
- P1-lo fails, elicit higher → in r's unit steering separates the matched pair: that row is replaced by the measured
  ratio; tell the user before editing the report.
- P1-lo reversed, P2-lo fails, any unreadable value, or a stage-0 miss → stop and tell the user.
- P3-own fails → only possible if the stage-0 numbers differ from the analysis's: check the code before reading.

**Departures from the literature.** None new. B3 departs from contrastive activation addition as the analysis states
(difference of fine-tuned and original activations, added once at the answer position, multiplier 1). Reading its
effect as a share of a log-odds gain rather than a log-probability gain is this project's unit choice, made to match
the route split; it is not taken from a source.

**Cost and route.** One MATS cluster debug job, one L40, $0, 2-4 minutes (model load and six sha256 reads dominate).
Not gated. The runner commits `results/amend1.json` and the log; the job commits nothing.

**Not run.** S_lo with the block chosen by its own fit-half log-odds share; S_lo at more than one position or with a
multiplier; per-key S_lo; a matched random δW.
