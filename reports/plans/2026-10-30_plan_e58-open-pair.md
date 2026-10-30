# Plan: does a fine-tune of a map the 67M half-knows carry to keys it never saw, and do r and the stand-in ratio agree with that answer? (E58)

*2026-10-30, regating-lead (plan and code writer). Item 1 of `reports/plans/2026-10-30_advice_regating-step2-FABLE.md`,
taken by the lead under the user's standing instruction (`decisions.md`, last row, 2026-10-30). Code
`experiments/open_pair/run.py`, rules `src/commute/open_pair.py` (edge cases `tests/test_open_pair.py`), predictions
`experiments/open_pair/results/predictions.json`, job `cluster/open_pair.sbatch`. Coded and filed; nothing run. Words as
in `GOALS.md` → Terms.*

## The question, in plain words

We fine-tune the 67M model on a word-pair map it already half-knows, but only on three quarters of the pairs, and ask
whether the model also gets better on the quarter it never saw; a random map on the same words is trained the same way
as the control, and it must not carry over. That answer is a label of "reuses what the model already had" or "new",
earned from behaviour alone with no decomposition, and it is the first such label on the 67M that the task's
construction does not decide. Then we read the same fine-tuned models with the route split's re-gating share and the
stand-in ratio, and with steering, and score whether the decomposition's reading agrees with the label.

**The request, in the Terms' words.** Fine-tune E54's elicit map (each key's value is W₀'s word-like rank 2-20 next
token after the marker and key) and its matched cipher with E54's recipe (`commute.keydep`, E12's arm A) on 24 of the 32
keys, seeds 0-2, the cipher read at the step whose ‖δW‖ matches the elicit state's. Label each state by the reuse line's
bounded-g rule read on behaviour: x = H ÷ G, H the held-out-key gain, G the trained-key gain; "reuses" iff x ≥ 0.5, "new"
iff x ≤ 3 s (row bootstrap), else "neither". Then the route split's r and the stand-in ratio of the least-squares split
state, and steering's S, on the same states, scored against the label as the step-2 plan's P2 and P3.

**Where the bar comes from.** `reports/plans/2026-10-29_plan_reuse-stage1.md`, § "Stage B: lowercase on the 67M", patch
row: "through iff P ≥ 0.5 (full = 1), not iff P ≤ 3 s (row bootstrap), else neither", with P the share of the
fine-tune's gain that the patch carries. Here P is read on behaviour as x = H ÷ G, as Fable proposed; the interchange-patch
half of Stage B is not run. Fable's advice asks reuse-lead to confirm the bar matches its definition; not yet done
(below, Order of work).

## Design check (`/design`)

- **Goal.** `GOALS.md` → "If no simple Φ works: new capabilities or strengthening?" The re-gating line's answer rests on
  r and the stand-in ratio, whose "elicited" and "new" labels were all set by task construction. This earns one that
  was not.
- **Nearest tried.** E54 (r 0.77 vs 0.24, ratio 0.95 vs 0.31, S 0.38 vs 0.23 on the 32-key pair, labels by
  construction), E56 (five seeds; neuron patching does not separate), E53 Amendment 2 (breadth does not explain the
  separation). E55's Stage A labels are earned but on the counting model, which has no decomposition. New here: the
  same pair with keys held out, so the label is read, not assumed.
- **Patch or step.** A step: item 1 of Fable's step-2 advice.
- **Decisions.** Relies on the reuse-line definition rows (Jain et al. Def 2 with the user's bounded g, 2026-10-30),
  E12's r ≥ 0.5 rule, E52's S rule, E54's recipe and separation rule. Departure: the label uses the behavioural half of
  the bounded-g definition only (no interchange patch); stated in every claim.
- **What changes next.** Label "reuses" with r and ratio agreeing → the first earned case where VPD agrees with a label
  steering and patching cannot give (E56): the headline, at n = 3. Label "new" with r and ratio high → the first
  concrete candidate for the granularity limit (or a VPD error): the novelty column and a costed second-seed card.
  Cipher not "new" → the rule is broken and no label is used.
- **Control first.** Stage 0 alone (below): the zero update and E54 elicit_s0 reproduced before any fine-tune; then
  the cipher, the label's negative control, is labelled right after each of its fine-tunes and stops the job if it
  reads "reuses".
- **Class.** Approved by the lead on Fable's advice under the user's standing instruction (`decisions.md`, last row:
  "build the open pair ourselves, $0"). The task is E54's on fewer keys; no new model, dataset or decomposition.

**The granularity limit, stated.** VPD's unit is the rank-one subcomponent, so a new circuit assembled from old
subcomponents under new gating reads "elicited" to r and the ratio. A behavioural "new" beside a high r and ratio
cannot tell that case from a VPD error; every claim from this experiment says "elicited at the level of VPD's
subcomponents".

## States

| state | how made | read on |
|---|---|---|
| elicit_s{0,1,2} | `keydep.elicit_ft` on E54's elicit windows whose key is one of the 24 trained keys (301 train windows); E12's scored step (first of 10/30/100/300 with held-out P(target) ≥ 0.5 on the trained keys) | G and every reading: the trained keys' 75 held-out windows; H: all 124 windows of the 8 held-out keys |
| cipher_s{0,1,2} | `keydep.cipher_ft` on the cipher's trained-key windows, read at the step in 1..30 whose ‖δW‖ is nearest the same seed's elicit state's | as elicit |
| E54 elicit_s0 (stage 0 only) | E54's saved `elicit_s0.pt` (sha256 checked) | E54's own 32-key held-out windows |

Key split: `open_pair.split_keys`, held-out = the first 8 of `torch.randperm(32, seed 0)` over E54's key list; one split
for both tasks and every seed. Held-out pairs (elicit value, W₀'s per-key geometric-mean P): action → potential
(9.7e-4), conditions → for (9.7e-3), base → pairs (4.4e-4), independent → increments (7.3e-6), inside → your (5.8e-4),
formed → from (2.5e-2), technology → was (4.4e-4), pain → model (2.0e-4). Window counts were measured on CPU from E54's
files: 15-16 windows per held-out key, 1-7 held-out windows per trained key.

**What differs from E54's end state.** Only the training windows (24 keys, 301 windows instead of 32 keys, 403) and
the table the loss reads. Same recipe, seeds, step rule, size match, precision and code (`keydep`, called, not copied).
No warm start or replay: every fine-tune starts from W₀. A held-out key never sits at the key position of a training
window; it can occur elsewhere as ordinary text, where the loss only holds the model to W₀ (KL term), and the marker
never occurs there (it has zero count in the mined rows).

## Stage 0: the control (runs first, alone; early stop)

| check | reads | must read | else |
|---|---|---|---|
| file identity | sha256 of E54's `windows_{elicit,cipher}.pt`, `elicit_s0.pt` | E54's recorded hashes (verified on dev 2026-10-30) | stop |
| key split | `split_keys` on E54's stage1.json keys | the filed 24 / 8 lists | stop |
| zero update | max per-window G and H of a zero δW on all four split tasks | 0 exactly | stop, fix |
| E54 elicit_s0, 32 keys | r, stand-in ratio, S, block-0 share via this code | r 0.7726 ± 0.03, ratio 0.9564 ± 0.03, S 0.3796 ± 0.001, block 0 ≤ 1e-6 (E54's tolerances; same code and card model, so ~1e-6 expected) | stop, fix |

## Stage 1: fine-tunes and the label (no decomposition)

Per seed: elicit, then the matched cipher; G, H, x, s and the label after each. Early stops: elicit_s0's G < 8 (the
state is too small to label; amend by raising the step once); any cipher labelled "reuses" (the rule is broken).

| number | how measured | at W₀ | do-nothing | positive / negative control | expected, elicit | expected, cipher |
|---|---|---|---|---|---|---|
| G | mean log-odds gain of the target, trained keys' held-out windows (target model, `steering.logodds_at`) | 0 exactly | target model only: same for every placement | E54 10.79 / 10.82 on 32 keys; must be ≥ 8 to label | 10-12 | ≈ 11 (E54 matched 11.10 / 10.89) |
| H | the same on the held-out keys' 124 windows | 0 exactly | as G | the cipher: values unknowable, must read "new" | **open** | ≈ 0 or below |
| x, s | H ÷ G; sd of x over 2000 row-bootstrap resamples | undefined | as G | — | open; s ≈ 0.02-0.05 | x within 3 s of 0 |
| label | x ≥ 0.5 reuses; x ≤ 3 s new; else neither | "too_small" | as G | cipher "new" | reuses ≈ 50% (Fable) | new |
| H_other (reported) | mean gain of the other 7 held-out values at the same windows | 0 exactly | as G | none | ≈ 0 if key-specific; ≈ H if a push toward the values' kind of token | ≈ 0 |

*Blind to.* H reads behaviour only: a map rebuilt by a new route that still generalises reads "reuses"; the label
says nothing about which subcomponents carry it. H can also rise from a key-independent push toward the kind of token
the values are (the elicit values are collocates such as " of", " for", " from", " was"); H_other is the check, reported
beside, not a rule. *Noise floor.* s, the label's own bootstrap; 8 keys is the real unit, so a per-key H is reported.

## Stage 2: the readings (E54's, unchanged)

| number | how measured | at W₀ | do-nothing | positive control | expected, elicit | expected, cipher |
|---|---|---|---|---|---|---|
| r | route split, least-squares split state, (L01 − L00) ÷ (T1 − T0), trained keys' held-out windows (`keydep.readings`) | undefined (gain 0) | not read | stage 0's E54 reproduction | 0.76 ± 0.03 | 0.24 |
| stand-in ratio | (L11 − L00) ÷ (U1 − U0) from the same cells; nan if U1 − U0 < 0.1 (E53 floor) | undefined | undefined; g alone | stage 0 | 0.95 | 0.29-0.34 |
| S | E52's B3 as E54 read it (log P, fit/score halves) | block 0 reads 0 ± 1e-6 | never reads the decomposition | stage 0 | 0.38 | 0.19-0.25 |

Stop: a block-0 shift over 1e-5 nats on any state (E54's). *Blind to:* r and the ratio, the granularity limit (above);
S, anything one input-independent vector at one position cannot carry, and it is in log P while r is in log-odds (E54
Amendment 1). *Noise floors:* r seed range 0.015 (elicit) and 0.055 (cipher) over five seeds (E54 + E56); ratio seed
difference 0.005 / 0.047 (E54); S range 0.017 / 0.07.

## Predictions (filed in `results/predictions.json`)

- **P0, the open reading (a lean, not a test).** Elicit labels counted: "reuses" or "new" when two of three carry it,
  else "mixed". Lean: reuses about 50%, new about 30%, mixed or neither about 20%.
- **P1, the label's negative control.** All three matched ciphers read "new". Any "reuses" stops the job; "neither",
  "too_small" or "unreadable" fails. Absolute. Lean: pass, high. Blind to whether the rule labels a real reuse
  "reuses" (no positive control for the label on the 67M).
- **P2, the step-2 plan's P2 on these runs, for r and for the ratio separately.** On every elicit run labelled "reuses"
  or "new" with a finite reading, the instrument's label (reading ≥ 0.5 → reuses) equals the behavioural label.
  "not_testable" when no run qualifies; "split" reported when the labelled runs disagree with each other. Absolute.
  Lean: pass about 50% (r and ratio are expected high on every run, so P2 passes exactly when the label is "reuses").
- **P3, the step-2 plan's P3.** Over the six states labelled "reuses" or "new" with r, ratio and S finite: "adds" iff
  A_VPD − A_S ≥ 2, "steering_better" iff A_S − A_VPD ≥ 2, else "no_difference". Lean: follows P0.
- **P4, the readings reproduce E54 on 24 keys.** r and the ratio each separate elicit from the matched cipher, elicit
  higher, by E54's rule (gap > larger range, strict, and ≥ 0.2). Relative. Lean: pass, high.

**Edge cases, run before filing** (`tests/test_open_pair.py`, 11 tests, all pass): empty held-out or trained pool → s
nan → "unreadable"; a bootstrap resample with G ≤ 0 → s nan; G under 8, zero or negative → "too_small" (the ratio's
denominator is never reached); ties at 0.5 and at 3 s count; both bounds at once (3 s ≥ 0.5 and 0.5 ≤ x ≤ 3 s) →
"unreadable"; all "neither" → P2 and P3 "not_testable", P0 "mixed"; nan readings dropped from P2 and P3 (the same rows
for every instrument); one run's difference in P3 → "no_difference"; a missing state → P4 "unreadable".

## Decision tree

1. Stage 0 misses → stop, fix ($0); tell the user only if the fix changes a filed number.
2. elicit_s0's G < 8 → dated amendment: raise the step count once, cipher matched by norm as E54; if still under 8, the
   family cannot be built this way: fall back to the step-2 table and raise Stage B with the user (Fable).
3. P1 stop or fail → no label is used; report descriptively and tell the user.
4. P0 reuses, P2 pass for r and ratio, P3 "adds" → the headline case at n = 3: the first label VPD did not make agrees
   with r and the ratio where steering and neuron patching see nothing (E56). Granularity limit stated. Next:
   Fable's item 4 (Qwen stage 1 on paper).
5. P0 new, P2 fail → the first concrete candidate for the granularity limit, a VPD error, or a label too strict.
   Next: the gating-pattern novelty column on these six saved states ($0 amendment) and the costed second-decomposition-
   seed card (Soon card).
6. All "neither" → the bar reads nothing here: report descriptively; ask reuse-lead for its bar before touching ours
   (never refile).
7. Split or mixed labels → report per run; P2 scored per run as filed; no general claim.
8. An elicit run labelled "reuses" with H_other ≥ 0.5 H → the write-up says the label may reflect a key-independent
   push toward the values' kind of token; P2 stands as scored.

Any qualitative reversal of a filed lean (P1 or P4 failing) goes to the user before the next step.

## Cost and routing

One MATS cluster `debug` job, one L40, account c11, $0. Expected 5-10 min: model load ~1 min, stage 0 ~1 min (one
route split on 97 windows), six fine-tunes ~1.5 min (E56 job 2: 11 s elicit, 16 s cipher with 32 checkpoints), six
113 MB saves, readings ~1.5-2 min. Host RAM peak ~5 GB (32 cipher checkpoints), GPU well under an L40's 48 GB.
Disk: ~680 MB in `results/large/` (a link into `~/commute-data/results-large/open_pair`, made by the sbatch if
missing). **Not gated:** under $15, under 3 h, inside the session cap.

## Order of work for the fleet agent

1. Done: `run.py`, `src/commute/open_pair.py`, `tests/test_open_pair.py`, `predictions.json` filed, sbatch, lint clean.
2. Ask reuse-lead, agent to agent, to confirm the bar (x ≥ 0.5 / ≤ 3 s, behaviour only) matches its definition (Fable
   item 5); a mismatch is a dated amendment before the run.
3. Fresh-Opus code review of the diff (`.claude/CLAUDE.md` → One review before a paid run; a $0 run of new
   measurement code still gets it). Fix what is real.
4. `/pod-runner`: `sbatch cluster/open_pair.sbatch` from the main clone after the commit is pulled there; stage 0 runs
   first and alone inside the job. Runner commits `results/run.json` and the log.
5. Analysis report by `/writeup` (opening paragraph, "Not run", "Departures from the literature": Jain et al. Def 2
   with the bounded g, behaviour only, no interchange patch; the 0.5 cut on the ratio copies r's rule); the six states
   join the step-2 table (`reports/plans/2026-10-30_plan_regating-ground-truth.md`) as the open rows; `TRIED.md` row
   updated; `PENDING.md` line.

## Not in this plan

- **The gating-pattern novelty column** (Fable item 3): not cheap enough to add without a design choice. E36's code
  (`spread.nearest`) finds nearest neighbours, but the column needs the definition of Γ's mask vector under the
  fine-tuned state (which of the route split's cells' gates), a reference set of W₀'s mask vectors over Pile positions
  and its size, and a distance scale with an expected value beyond "0 at W₀ by construction". It is listed for the
  branch-5 follow-up, a $0 amendment on the six saved states.
- The interchange-patch half of Stage B's label; more held-out key splits; more windows per held-out key; the cipher at
  its own E12 step; more than three seeds; Qwen; any change to E54's or E56's files or predictions.
