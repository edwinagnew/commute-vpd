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
| ~~E54 cipher_s0 with no carry-over (added 2026-10-30, pre-run amendment)~~ superseded by the row below after job 46259 read x0 = 0.375 against 3 s0 = 0.061 | x0 = mean log-odds gain of the 31 non-target cipher values ÷ the target's G, on E54's 32-key windows | x0 ≤ 3 s0 | stop |
| E54 cipher_s0, value-specific share (2026-10-30, second pre-run amendment) | G_spec ÷ raw G on E54's 32-key windows, with G_spec = own value's gain minus the mean gain of the other 31 pool values | ≥ 0.5 (expected 0.625 from job 46259: 6.944 ÷ 11.107) | stop |
| E54 elicit_s0, value-specific share (same) | as above | reported; sets the gate with the cipher's | — |
| G_spec gate (same) | min(8, 8 ÷ 10.79 × the smaller of the two G_spec above) | finite; at most 5.148 (cipher 6.944) | stop if not finite |

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

## Amendment before the run (2026-10-30, code review of 674e9b0)

Nothing had run on a GPU, so `predictions.json` carries a dated pre-run amendment (`amendment_2026-10-30_prerun`); P0-P4
and the label rule are unchanged.

- *Pre-mortem, the label's negative control (2026-10-30).* "New" needs x ≤ 3 s, and s is a sampling sd that may be
  about 0.01. Any generic push after the marker toward word-like tokens would then put a cipher with no carry-over at
  "neither", and P1 would fail because of the bar, not because of the cipher. Guard: stage 0 now reads E54's saved
  cipher_s0 on its own 32 keys. Its x0 (the non-target cipher values' gain over the target's gain) is what a cipher with
  no carry-over reads. The job stops if x0 > 3 s0, so the bar can be amended before any fine-tune. Also added:
  cipher_s0's sha256.
- *Key-level bootstrap (2026-10-30).* The real unit of H is the 8 held-out keys; W₀'s per-key probability spans 7e-6 to
  2.5e-2. `s_key` (keys resampled, every window of a drawn key pooled) is reported beside s and is not scored.
  **Departure from the Stage B source:** Stage B's row bootstrap resamples independent texts, while windows here share
  a key. The label keeps the filed s; the write-up says which labels would change under 3 s_key.
- *Run hygiene (2026-10-30).* A failed key-split check now writes run.json with "stopped" and exits 1, where it used to
  be an assert. run.json records `git status --porcelain` of run.py and `open_pair.py`. The sbatch now handles a
  dangling `results/large` link: it creates the target and links only when no link or directory exists. The submit
  command is unchanged.

## Second amendment before the run (2026-10-30, after job 46259; lead's decision)

Job 46259 stopped at stage 0, as the first amendment designed it to. Every other stage-0 check passed: the zero update
read 0.0, and E54's elicit_s0 read r 0.7726, ratio 0.9564, S 0.3796 and block 0 −9.4e-8 (results commit 080b1f2).
Nothing was trained. The stop came from the cipher: on E54's cipher_s0, the other 31 cipher values gained 4.163 log-odds
at each window, against the target's 11.107. So x0 = 0.375, against 3 s0 = 0.061 (s0 0.020). The cipher fine-tune lifts
the whole value pool, and raw H ÷ G would read even a cipher with no carry-over as "neither", never "new". The cause is a
pool shift, not carry-over.

- *The label is now about the specific value (2026-10-30).* Each window's specific gain is the gain of its own value
  minus the mean gain of the other 31 values in the task's full table. G_spec is that gain on the trained keys and H_spec
  on the held-out keys. The label reads x = H_spec ÷ G_spec with the same bars: reuses iff x ≥ 0.5, new iff x ≤ 3 s
  (the row bootstrap of x), else neither. Raw G, H and x are reported beside and not scored; s_key is now computed on
  the specific gains. P0-P4 are unchanged.
- *The gate (2026-10-30).* E54's own cipher_s0 has a G_spec of 6.944, under 8, so the gate is lowered. It is now
  min(8, 8 ÷ 10.79 × the smaller G_spec of E54's elicit_s0 and cipher_s0), which keeps Fable's fraction of E54's gain.
  The cipher sets it at 5.148 or less. E54 elicit_s0's G_spec has never been read: no logits were saved, and it cannot
  be computed off the GPU. Stage 0 reads it and records the gate. elicit_s0's early stop now uses G_spec against this
  gate.
- *The stage-0 check (2026-10-30).* The no-carry check is replaced. On E54's cipher_s0, G_spec ÷ raw G must be at
  least 0.5, meaning most of the cipher's gain is specific to its value; job 46259's numbers give 0.625. The same split
  on E54's elicit_s0 is reported.
- *Expected (2026-10-30).* The cipher's H_spec should be about 0 or below once the pool shift is subtracted, and it
  should read "new". The elicit label stays open. *Blind to:* a carry-over that lifts every value of the pool alike
  reads as no carry-over here. That is the price of removing the pool shift.

## Amendment 1 (2026-10-30, after job 46266): the rank-k readout and the label's positive control

*Lead's decision on Fable's advice (`reports/plans/2026-10-30_advice_e58-reading-FABLE.md`, "Two $0 checks"; `decisions.md`,
last row). Code `experiments/open_pair/amend1.py`; rules `src/commute/open_pair.py` § Amendment 1 (edge cases
`tests/test_open_pair.py`); the label and the readings moved, not changed, from `run.py` to `src/commute/heldout_keys.py`;
predictions `results/predictions_amend1.json`; job `sbatch cluster/open_pair.sbatch amend1`. Coded and filed; nothing
run. E58's predictions, P0-P4 and run.json are untouched.*

**In plain words.** E58 found that the fine-tune of the half-known map did not carry to the eight keys it never saw,
while r and the stand-in ratio read "old pieces". Two checks decide whether that label means anything. First, at the
unseen keys, we look at what the fine-tune did to the words the original model already preferred there, beside the
words it was trained to say elsewhere: a lookup pushes the trained words; a general sharpening pushes the model's own
preferences. Second, we give the label the positive control it never had: a map whose rule the model already follows
(say what you would have said anyway, ignoring the marker), trained the same way on the same 24 keys, which should
carry to the other 8 and read "reuses"; the copy map runs beside it as the fallback.

**The request, in the Terms' words.** (i) On E58's six saved fine-tuned states (target model only), at the held-out keys'
windows: the log-odds gain of W₀'s top-20 next tokens at each key, of the window's own value, and of the 24 trained
values; the trained-value push T reported directly. (ii) Two new key → value maps on E54's windows, marker, keys and
E58's 24/8 split: the rank-1 map (value = W₀'s most probable next token after "M key") and the copy map (value = key);
E54's recipe (`keydep.elicit_ft`), seeds 0-2; each state labelled by E58's rule (value-specific gain, the same bars
and E58's gate 5.148) and read with r and the stand-in ratio (least-squares split state) and steering's S.

### Design check (`/design`)

- **Goal.** `GOALS.md` → "new capabilities or strengthening?": the re-gating line needs one earned behavioural label
  to score r and the ratio against. E58's "new" can only be used if the label can also read "reuses" on the 67M.
- **Nearest tried.** E58 (the label, "new" ×3); E12/E46 copy fine-tunes (r 0.76-0.80, ratio 0.87-0.97, labelled by
  construction); E55 Stage A (earned labels on the counting model, no decomposition). New: the first 67M state whose
  "reuses" is earned by held-out keys, and the first readout of what E58's update does at the unseen keys.
- **Patch or step.** A step that completes E58 (its missing control), not a new direction.
- **Decisions.** E58's label rule and gate (`predictions.json` amendment 2); E54's recipe; E12's r ≥ 0.5 rule;
  the step-2 plan's P2 (its note below). User's elicitation granularity limit stated in every claim.
- **What changes next.** Positive control passes → E58's "new" stands as a reading of the update, and P2 of the step-2
  plan gets its first labelled "reuses" row. Fails → the label is not used against r or the ratio (to the user).
- **Control first.** Readout: the zero update must read exactly 0, then each state must reproduce E58's H and the
  algebraically implied T before the next state is read. Control maps: zero update exactly 0 on all four split tasks.
- **Class.** Approved on Fable's advice (decisions.md, last row); $0, cluster `debug`, no new model or dataset.

### (i) The rank-k readout

| number | how measured | at W₀ / zero update | do-nothing | control | expected, elicit s0/s1/s2 | expected, cipher |
|---|---|---|---|---|---|---|
| own | mean gain of the window's own value | 0 exactly | target model only: same for every placement | must equal E58's H within 1e-4 | 0.613 / 0.460 / 0.897 | 1.375 / 1.189 / 1.256 |
| T | per window, mean gain of the trained values other than the own value | 0 exactly | as own | must equal `implied_push` from run.json within 1e-4 | 1.589 / 1.680 / 1.842 | 5.497 / 5.726 / 5.617 |
| R_peer | per window, mean gain of W₀'s top-20 at the key that are neither own nor trained values | 0 exactly | as own | none beyond the zero update | about 0 (−0.6 to +0.3) | near 0 or below |
| d, s_d, s_R | mean of T_w − R_w; row-bootstrap sd of d and of R_peer (2000, seed 0) | 0, 0 | as own | — | d 1.3-2.2, s_d 0.1-0.2 | d about 5 |
| rank gain 1..20 | mean gain per W₀ rank (reported) | 0 | as own | — | flat near 0 except trained values in the top 20 | reported |
| verdict | lookup iff d > 0 and d ≥ 3 s_d; sharpening iff R_peer > 0 and ≥ 3 s_R; both; neither | "neither" | as own | — | lookup | lookup (reported) |

T is not new information: with distinct values, H − H_spec is the mean gain of the other 31 pool values, so
T = (31 (H − H_spec) − 7 H_other) ÷ 24 exactly (`open_pair.implied_push`; Fable's "+1.6"). That identity is the
readout's control: it checks the new code against E58's numbers on every state. What is new is R_peer and the rank
profile. *Blind to:* a sharpening confined to W₀'s rank 1 (averaged into R_peer; rank 1's gain is reported); log-odds at
the answer position only; a trained value that sits in W₀'s top 20 at a held-out key counts in T, not in R_peer.
*Noise floor:* s_d and s_R from 124 windows; the 8 keys are the real unit, so the row bootstrap understates the spread
(as E58's s against s_key).

### (ii) The label's positive control

| number | how measured | at W₀ | do-nothing | control | expected, rank-1 map | expected, copy map |
|---|---|---|---|---|---|---|
| W₀ geo-mean P(value) | trained keys' held-out windows (`keydep.geo_rate`) | — | — | — | 0.03-0.3 | 1e-4 to 2e-3 (E10 identity ≤ 0.002) |
| distinct values; novel held-out keys | values of the 32 keys; held-out keys whose value no trained key has | — | — | — | 8-25; 1-6 | 32; 8 |
| zero update | max per-window gain on the four split tasks | 0 exactly | — | stop if not | 0 | 0 |
| G_spec, H_spec, x, s, label | E58's `behaviour`, pool values counted once | 0, label too_small | target model only | gate 5.148 (E58's) | G_spec 4-10; x 0.5-1 if it learns "ignore the marker"; reuses ≥ 2 of 3 | G_spec 8-13; x 0.5-1; reuses ≥ 2 of 3 |
| x_novel (reported) | key-mean H_spec over novel held-out keys ÷ G_spec | — | — | — | ≈ x if the rule carries | = x (all novel) |
| r, ratio, S | E54's readings on the trained keys (`heldout_keys.vpd_read`) | undefined | r, ratio: least-squares split only | E58's stage 0 reproduced them | r 0.6-0.9, ratio 0.6-1.0, S reported | r 0.7-0.85, ratio 0.85-1.0, S 0.6-0.8 |
| readout | as (i), on each control state (reported) | 0 | — | — | rank-1 gain high | reported |

Both maps run every time, in one job (rank-1 first). Running the copy map always costs about a minute and saves a second
job when rank-1 reads "new". *Blind to:* the rank-1 map can repeat a value across keys, and a held-out key whose value is
also a trained value can carry by the push toward trained values rather than by the rule (`min_novel` and x_novel guard
it); the copy map's label is fixed by construction, and the 67M ties its input and output embeddings, so a direct path
can carry it; a "reuses" control does not test the granularity limit, which bites only when behaviour reads "new".
*Noise floor:* E58's (s, s_key; r 0.015-0.055, ratio 0.005-0.047, S 0.017-0.07 over seeds).

### Predictions (filed in `results/predictions_amend1.json`; every rule run on its edge cases first, 9 new tests)

- **A1.** All three elicit states read "lookup". Fail on any "sharpening", "both" or "neither"; unreadable if a state is
  missing. Lean: pass, ~85%.
- **C1 (rank-1).** Not testable if fewer than 3 novel held-out keys or rank1_s0 "too_small" (Fable's guard); else pass
  iff ≥ 2 of 3 seeds read "reuses". Lean: testable ~55%, pass ~60% if testable.
- **C2 (copy).** The same, always testable. Lean: pass ~75%.
- **Verdict.** "rank1" (C1 pass), "copy_fallback" (C1 not testable, C2 pass), "copy_only" (C1 fail, C2 pass), "none".
- **C3.** E58's P2 for r and the ratio on the control runs (rank-1 unless C1 is not testable, then copy). Lean: pass ~75%.

Edge cases tested: `implied_push` against hand algebra and E58 elicit_s0 (1.5892), one held-out key or no trained key
→ nan; `boot_sd_mean` empty, one row, a nan row (empty peer set) → nan, constant → 0; `push_verdict` ties at 3 s,
zero d with zero s (neither), both, nan; `a1` missing and unreadable states, fail outranking unreadable;
`control_status` 2-of-3 tie, all too_small, untestable, empty; `score_amend1` every verdict branch, `min_novel` tie,
empty records; `pool_of` with repeated and distinct values; `others` with the own value left out and nan with no other
value.

### Departures from Fable's spec and the sources

- Fable's "W₀'s rank 1-20 word-like tokens": ranks here are over the full vocabulary. E54's value ranks were full-
  vocabulary ranks; the word-like filter only decided which token could be a value.
- Fable's "set the gate from W₀'s rank-1 probability before filing": that probability cannot be read before the job
  (no model loads off the GPU). The gate is E58's, as the lead's brief asks; W₀'s P(value) is recorded, and a rank1_s0
  under the gate makes C1 not testable and hands the verdict to the copy map.
- The copy map runs always, not only as a fallback (one minute; saves a second job).
- `min_novel` = 3 is added here (reason in the blind-to line above). `heldout_keys.pool_of` counts each value once; on
  E58's maps, whose values are distinct, nothing changes.

### Stop rules, runtime, outputs

Stops (exit 1, run.json untouched, `results/amend1.json` written with "stopped"): the readout of the zero update not
exactly 0 (about 1.5 min in); any E58 state whose file hashes differ from run.json or whose own gain or T misses by over
1e-4 (by about 3 min); the zero update not 0 on a control task; a block-0 shift over 1e-5 nats. Expected in the first
minutes: model load about 1 min; the six state controls within about 1e-6; rank1_s0's fine-tune about 11 s, held-out
P(value) ≥ 0.5 at step 10, ‖δW‖ about 3.0-3.3 (E58 elicit 3.12-3.18). Not a stop: rank1_s0 "too_small".

One MATS `debug` job on an L40, $0, about 6-10 min (readout about 1 min; six fine-tunes about 11 s each; label, readout
and readings about 25 s per state); host RAM under 6 GB (four checkpoints of 113 MB per fine-tune); disk about 0.7 GB in
`results/large/amend1/` (a real directory in the main clone's linked `results/large`). Not gated. Outputs:
`results/amend1.json`, `results/large/amend1/{windows_*,rank1_s*,copy_s*}.pt`, log `results/large/amend1.out`.

### Decision tree

1. A stop → fix ($0); nothing about E58 changes.
2. A1 pass → reading 2/3 stands with T and R_peer beside H in the write-up. A1 sharpening or both → the "guaranteed by
   construction" worry is only partly removed; rank profile reported; to the user as a surprise.
3. Verdict "rank1" or "copy_fallback" → the label has a positive control on the 67M; E58's "new" reads the update;
   the step-2 plan's P2 gets its first labelled "reuses" row. C1 pass with x_novel < 0.5 on two seeds → the carry rides
   on shared values; lean on C2.
4. "copy_only" → the label can read "reuses", but a fine-tune of a rule W₀ has did not carry; this sharpens the user's
   question on what "reuses" means; no rule-task row from rank-1.
5. "none" → the held-out label is not used against r or the ratio; tell the user.
6. C3 fail (r or ratio reads "new" on an earned "reuses") → qualitative surprise; to the user before the next step.

### Order of work

1. Done: code, tests (23 pass), lint clean, predictions filed.
2. Fresh-Opus review of the diff (one review before a run of changed measurement code).
3. `/pod-runner`: from the main clone after the commit is pulled there, `sbatch cluster/open_pair.sbatch amend1`.
4. Write-up as a section of E58's analysis report (`/writeup`); `TRIED.md` row; `PENDING.md` line.

## Not in this plan

- **The gating-pattern novelty column** (Fable item 3): not cheap enough to add without a design choice. E36's code
  (`spread.nearest`) finds nearest neighbours, but the column needs the definition of Γ's mask vector under the
  fine-tuned state (which of the route split's cells' gates), a reference set of W₀'s mask vectors over Pile positions
  and its size, and a distance scale with an expected value beyond "0 at W₀ by construction". It is listed for the
  branch-5 follow-up, a $0 amendment on the six saved states.
- The interchange-patch half of Stage B's label; more held-out key splits; more windows per held-out key; the cipher at
  its own E12 step; more than three seeds; Qwen; any change to E54's or E56's files or predictions.
