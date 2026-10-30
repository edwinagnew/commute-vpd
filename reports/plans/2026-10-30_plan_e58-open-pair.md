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

## Amendment 2 (2026-10-30, after job 46332): the three maps at four times the keys

*Lead's decision on Fable's advice (`reports/plans/2026-10-30_advice_e58a1-next-FABLE.md`, item 3, "One $0 check that can
overturn item 1", and "What would change it"). Code `experiments/open_pair/amend2.py` (180 lines); rules
`src/commute/open_pair.py` § Amendment 2 (edge cases `tests/test_open_pair.py`, 6 new tests, 29 pass); E54's key
selection (`vocab`, `select`) moved, not changed, from `experiments/keydependent_elicit/run.py` into
`src/commute/keydep.py`, and E54's run.py now imports them (lint clean; no other script called them); predictions
`results/predictions_amend2.json`; job `sbatch cluster/open_pair.sbatch amend2`. Coded and filed; nothing run. E58's
and Amendment 1's files are untouched.*

**In plain words.** At 24 keys every fine-tune memorised its keys and only the copy map carried to unseen keys, about
half way, while r and the stand-in ratio stayed put. That may only be because 24 keys are too few for a rule to win
over a lookup. We train the same three maps on four times the keys and ask two things: does a rule start to carry
(the rank-1 map, "say what you would have said anyway", and the copy map), and if it does, do r and the ratio move with
it? Under reading A they should not.

**The request, in the Terms' words.** E54's key selection at k = 128 (96 trained / 32 held-out keys; 64 / 16 if the
supply is short), E54's window density, the elicit, rank-1 and copy tables on those keys, seed 0, E54's recipe
(`keydep.elicit_ft`); per state E58's held-out-key label (x = H_spec ÷ G_spec, value-specific gains), Amendment 1's
x_novel, and the route split's r and stand-in ratio of the least-squares split state on the trained keys.

### Design check (`/design`)

- **Goal.** `GOALS.md` → "new capabilities or strengthening?": tests whether the line's claim under reading A ("r reads
  which pieces a fine-tune builds from, not whether what it learned generalises") survives when carry is given room
  to move.
- **Nearest tried.** Amendment 1 (job 46332: x copy 0.42-0.55, rank-1 0.09-0.14, elicit −0.03; r 0.76-0.85 on all
  nine); E58. New: the key count is the one thing varied, so x can move on the same maps while r is read beside it.
- **Patch or step.** A step: the one check Fable names as able to overturn item 1.
- **Decisions.** E58's label and gate, E54's recipe and selection, Amendment 1's maps and `min_novel`; reading A
  (the user's ask #14 still open). No departure from `decisions.md`.
- **What changes next.** (a) rank-1 earns "reuses" with r flat → first fair "reuses" row for step 2's P2. (b) r or the
  ratio moves with x → item 1's "blind to" sentence is wrong; to the user. Null (flat) → item 1's claim gets its
  strongest line. x moves on neither map → the recipe memorises regardless; no more key scaling.
- **Control first.** The selection must reproduce E54's 32 keys and values exactly, the zero update must read 0, then
  the elicit map (no rule to carry) runs alone and stops the job if x > 0.05 or it is too small to label.
- **Class.** Approved (lead's decision tree, Fable's advice adopted); $0, cluster `debug`, same model, task family and
  harness.

### Key supply (the guard, checked before filing)

Windows are not a limit: `kv.windows` writes the marker and key into random Pile positions (32,768 rows), so any
count per key is available; E54's density (500 windows per 32 keys, about 15.6 per key) gives 2000 windows at 128 keys.
Keys: E54's job 45973 qualified 777 of its 1000 candidates, but the draw needs distinct values and the distinct count
was never saved (stage1.json keeps the 32 drawn). It cannot be read without W₀, so the job reads it first: K = 128 if
the draw returns 128 distinct-valued keys, else 64 (16 held out) with the same expected values, else stop. The k = 128
draw walks E54's seeded order, so its first 32 keys are E54's; the job checks that E54's 32 keys and values come back
exactly. Lean: K = 128, about 85%.

### The denominator (computed from disk before filing)

At 24 keys G_spec (step 10, trained-key P 0.93-0.98) read elicit 18.2-18.5, rank-1 16.6-16.8, copy 17.4-18.2; E58's
ciphers at steps 8-9 (P 0.45-0.58) read 9.4-10.7. With 96 trained keys, batch 32 shows each key about 3 times by step 10
(13 at 24 keys), so the scored step is likely 30 (about 10 views per key), and G_spec 10-19. What the cuts then ask of
H_spec, at G_spec 10 / 17 / 19: elicit's +0.05 = 0.5 / 0.85 / 0.95 log-odds; rank-1's 0.3 = 3.0 / 5.1 / 5.7; copy's
0.6 = 6.0 / 10.2 / 11.4, against 7.3-9.6 carried at 24 keys. Copy's cut is therefore a real ask (lean 40%), not a near
miss by construction. E58's gate (5.148) sits under the whole range.

### What differs from Amendment 1

Only the keys (128 drawn by E54's rule instead of E54's 32; 96 / 32 split by `split_keys`, seed 0), the windows (2000
at the same density instead of E54's 500), one seed instead of three, and therefore the scored step (likely 30, not 10)
and the update's size. Elicit is the reference for whatever the key count does to every map alike: test B compares each
map to elicit, at K keys and at 24 keys (a difference in differences), never a map to itself across key counts alone.
No warm start or replay; every fine-tune starts from W₀. The cipher is not run (elicit is the no-rule control here).

### Numbers and what each should read

| number | at W₀ / zero update | elicit (control) | rank-1 | copy |
|---|---|---|---|---|
| E54 reproduction | — | marker 18214, 777 qualified, E54's 32 keys and values exactly | | |
| zero update, six split tasks | 0 exactly | | | |
| G_spec (scored step) | 0, too_small | 10-19 (gate 5.148) | 10-19 | 10-19 |
| x | undefined | −0.15 to +0.05 (24 keys −0.03) | ≥ 0.3 if rules learnable with keys, ≤ 0.15 if memorised (24 keys 0.09-0.14) | ≥ 0.6 (24 keys 0.42-0.55) |
| x_novel, n_novel | — | = x; all novel | about x or above; probably 1-6 of 32 novel (R may read not_testable) | = x; 32 |
| r | undefined | 0.75-0.85 | 0.75-0.85 | 0.75-0.85 |
| ratio | undefined | 0.9-1.1 | 0.9-1.1 | 0.9-1.1 |
| dx, d_r, d_ratio (vs elicit and 24 keys) | — | 0 by definition | d_r within ±0.05, d_ratio within ±0.1 wherever |dx| ≥ 0.15 | same |

S and the label's s, s_key are reported, not scored. *Blind to:* one seed per map, so each number carries the 24-key
seed spread (x: 0.012 elicit, 0.048 rank-1, 0.132 copy; r ≤ 0.029; ratio ≤ 0.055); a carry riding on values shared
with trained keys (x_novel guards rank-1); whatever moves every map's r alike with the key count (elicit absorbs it by
design); r's own blind spot under reading A (pieces, not carry).

### Predictions (filed in `results/predictions_amend2.json`)

- **E, the control.** pass iff −0.05 ≤ x ≤ 0.05; "below" iff x < −0.05 (no carry; not a stop); fail iff x > 0.05
  (stop); too_small or unreadable stop. Lean: pass 70%, below 25%.
- **R, rank-1.** "reuses" iff E58's label reads reuses and x_novel ≥ 0.5 (Fable's (a)); "carries" iff x and x_novel ≥
  0.3; "memorises" iff both ≤ 0.15; else "between"; "not_testable" with fewer than 3 novel held-out keys or an
  unlabellable state. Lean: not testable 30%, memorises or between 50%, carries 15%, reuses 5%.
- **C, copy.** pass iff x ≥ 0.6. Lean: 40%.
- **B, Fable's (b).** Per map (rank-1, copy) and instrument: dx = (x_map − x_elicit) at K keys minus the same at 24
  keys (3-seed means), d_r and d_ratio likewise. not_testable iff |dx| < 0.15 or the map's own x moved by less than
  0.15 (or against dx) between 24 and K keys; "tracks" iff the instrument moved the
  way x moved by more than 0.05 (r) or 0.1 (ratio); "flat" iff within it; else "opposite". Combined: tracks >
  opposite > flat > not_testable. Lean: flat 50%, not_testable 35%, tracks 10%.
- **Verdict.** stopped unless E passes or reads below; then b_r_tracks > a_first_reuses > reading_A_holds > opposite >
  untested. Lean: reading_A_holds 50%, untested 30%, b 10%, a 5%.

Edge cases tested before filing: E's ties at ±0.05 (pass), just over (fail), nan x from a zero G_spec, too_small and
unreadable pass through; R's ties at 0.3, 0.15, 0.5 and at `min_novel` = 3, x and x_novel disagreeing (between), nan
x_novel (no novel key), too few novel keys; C's tie at 0.6; `moves_with` at the |dx| = 0.15 tie (testable), |d| = tol
(flat), both signs, nan and inf; `combine` on an empty list and on all not_testable; a common drift of every r with K
(flat, not tracks); a ratio nan on one map (r alone decides it); missing states and an empty record (stopped).

### Every setting, sourced

| setting | value | source |
|---|---|---|
| key selection | ranks 100-1099, 16 windows per candidate, seed 1, band [0.01, 0.1], ranks 2-20, 10x key-specific, draw seed 0 | E54's predictions.json, unchanged |
| K | 128 (fallback 64) | Fable item 3 |
| held out | K / 4, `split_keys` seed 0 | E58's fraction (8 of 32) |
| windows | 500 × K / 32, seed 0, row split | E54's density and seed |
| maps | elicit (E54's rule), rank-1 and copy (Amendment 1's) | as named |
| fine-tune | E12's arm A via `keydep.elicit_ft`, scored step rule | E54's recipe |
| seed | 0 | Fable |
| label, gate, bootstrap | E58's (`behaviour`, G_min 5.148, 2000 resamples) | E58 amendment 2 |
| e_max 0.05, rank-1 0.3 / 0.15, copy 0.6, tol_r 0.05 | as stated | Fable |
| e_max one-sided as a stop | | departure: a negative x is the trained-value push (Amendment 1's T), which grows with 96 trained values in the pool |
| x_novel must agree for R | | Amendment 1's guard (x_novel 0.21-0.27 above x 0.09-0.14) |
| dx_min 0.15 | | measured: copy x's 24-key seed range 0.132 |
| tol_ratio 0.1 | | departure from Fable's 0.05: copy's ratio seed range at 24 keys is 0.055 |
| 24-key reference | 3-seed means | E58 run.json, amend1.json |
| precision | fp32 reads, TF32 off; bf16 fine-tune | E54, E58 |

### Stop rules, runtime, outputs

Stops (exit 1; `results/amend2.json` written with "stopped"): E54's selection not reproduced, or fewer than 64
distinct-valued keys (about 1.5 min in); the zero update not 0 on a split task (about 2 min); the elicit control not
"pass" or "below" (about 3 min); a block-0 shift over 1e-5 nats on any state. Expected in the first minutes: model load
about 1 min, selection about 8 s (E54 job 45973), K = 128, about 1600 / 400 train / held-out windows by row, elicit's
held-out P(value) under 0.5 at step 10 and over it at step 30, elicit G_spec 10-19.

One MATS `debug` job on an L40, $0, about 10-15 min (selection under 1 min; three 300-step fine-tunes about 15 s each;
r, ratio and S on about 300 trained-key held-out windows per state, about 1.5-2 min each against Amendment 1's 25 s on
75). Host RAM under 6 GB (E54's 1000 key distributions about 0.4 GB, four 113 MB checkpoints per fine-tune); the sbatch
asks 24 GB as before. Disk about 0.36 GB in `results/large/amend2/` (three 113 MB states, nine windows files of about
1 MB). Not gated. Outputs: `results/amend2.json`, `results/large/amend2/`, log `results/large/amend2.out`.

### Decision tree (Fable's (a), (b) and the null)

1. A stop → fix ($0); an E fail goes to the lead (the label reads a carry where no rule exists).
2. **b_r_tracks** (Fable's (b)) → item 1's "blind to" sentence is wrong as written; to the user before the line's claim
   is written.
3. **a_first_reuses** (Fable's (a)) → the first fairly labelled "reuses" state with r unchanged; score step 2's P2 on
   it; keep the held-out label for rule tasks given enough keys (a dated note on the ground-truth plan).
4. **reading_A_holds** (the null) → x moved on at least one map while r and the ratio did not: item 1's claim is
   written with this line ("x moved by dx, r by d_r on the same states").
5. **opposite** → r moved against x: report per map; to the lead before the claim is written.
6. **untested** → more keys did not move carry on either map: the recipe memorises regardless; reading A rests on
   Amendment 1's spread table alone; no more key scaling on this recipe.

### Order of work

1. Done: code, tests, lint clean, predictions filed.
2. Fresh-Opus review of the diff (new measurement code: the selection reuse, the map build at K keys, the B rule).
3. `/pod-runner`: from the main clone after the commit is pulled there, `sbatch cluster/open_pair.sbatch amend2`
   (submit from the clone root; record the job ID).
4. Write-up as a section of E58's analysis report (`/writeup`); `TRIED.md` row; `PENDING.md` line.

### Amendment before the run (review, 2026-10-30)

A fresh Opus review of a23c3ea found two real problems and two small ones. Nothing had run, so the section above and
`predictions_amend2.json` were amended in place (`amendment_2026_10_30_prerun_review` records the change).

- **One common step (high).** `one_map` called `keydep.elicit_ft`, so each map was read at its own E12 step. Elicit
  will likely land at step 30 while rank-1 and copy, whose values W₀ already favours or which its tied embeddings carry,
  may clear at 10. dx, d_r and d_ratio would then mix update size into the key count. Now every map is fine-tuned with
  `keydep.finetune` (all four checkpoints kept) and read at elicit's scored step. Elicit runs first, as before, and the
  job stops if it reaches held-out P ≥ 0.5 at none of the checkpoints. Each map's own first step with P ≥ 0.5 and
  ‖δW‖ at every checkpoint are reported, not used.
- **A map must move by itself (medium).** dx = (x_map − x_elicit) at K keys minus the same at 24 keys let elicit's own
  fall (for example −0.03 to −0.20, "below") make a map testable when the map's x did not change. Now B is testable on a
  map only if the map's own x also moved by at least 0.15 between 24 and K keys, in the same direction as dx. New test:
  elicit falls to −0.20 with rank-1 and copy at their 24-key x and copy's r up 0.11 → B not_testable, verdict
  untested.
- **Qualified pool hash (low).** `amend2.json` logs a sha256 of the sorted (key, value) pairs of the qualified pool
  beside the E54 reproduction check, so a later rerun can tell a changed pool from a changed draw.
- **Rank-1 novel keys (low).** At 24 keys token 15 was the value of 9 of 32 keys and only 1 of 8 held-out keys had a
  value of its own. The lean of 3-10 novel keys of 32 was optimistic: probably 1-6, so R may well read not_testable
  (about 30%). The rule (`min_novel` = 3) is unchanged.
- Kept: the move of E54's `vocab` and `select` into `keydep` (the reviewer diffed it byte-identical).

### Amendment 2's control missed (job 46383, dated note 2026-10-30)

Job 46383 stopped at the control, as the plan designed it to. Everything before the control passed. E54's selection
reproduced (marker 18214, 777 qualified, E54's 32 keys and values; 319 distinct values, so K = 128). The zero update
read 0. Rank-1 had 49 distinct values and 10 novel held-out keys of 32. Elicit, read at step 10 (trained P 0.604; 0.984
at step 30), gave G_spec 9.34, H_spec +1.99, x 0.213 (3 s 0.041), label "neither". H_spec was positive on 28 of 32
held-out keys, and H_other was −0.79. The filed E prediction (pass about 70%, fail or too_small about 5%) is a **miss**.
It stands and is never edited. Results: `results/amend2.json` (fe29938), log `results/large/amend2.out`, state
`results/large/amend2/elicit_s0.pt`.

*Cause, as Fable states it (`reports/plans/2026-10-30_advice_e58a2-control-FABLE.md`); reasoned, not yet confirmed by
Amendment 3's checks.* The control was mis-specified, not the measurement. Every elicit value is, by construction
(`elicit.pick_values`), W₀'s own rank 2-20 token after that key and ten times its rate after other keys. "After the
marker, say what W₀ half-expects for this key" is therefore a rule. With 96 keys and about 3 views per key by step 10,
the fine-tune found it, where 24 keys only let it memorise. The held-out gain moved from about −0.5 log-odds (24 keys) to
+2.0. The smaller denominator (9.3 against 18) can at most double x; it cannot change its sign. The map with no rule was
always the cipher (random values), and Amendment 2 should have used it as the control.

## Amendment 3 (2026-10-30, after job 46383): the cipher as the control, elicit as a scored map

*Lead's decision on Fable's "Recommended amendment" (`reports/plans/2026-10-30_advice_e58a2-control-FABLE.md`). Code
`experiments/open_pair/amend3.py` (206 lines). It imports amend2's `stage_keys` and `build_maps`, which are unchanged;
amend2.py still runs as it ran, and amend3 points amend2's output file at `amend3.json`. Rules are in
`src/commute/open_pair.py` § Amendment 3 (`pick_cipher_step`, `b3`, `trajectory`, `spearman`, `score_amend3`), with edge cases in
`tests/test_open_pair.py` (5 new tests; 34 pass). Predictions are in `results/predictions_amend3.json`. Job:
`sbatch cluster/open_pair.sbatch amend3`. Coded and filed; nothing run.*

**In plain words.** At 128 keys the map built from the model's own second-to-twentieth guesses carried to the unseen
keys: about a fifth of the way, where at 24 keys it did not carry at all. That stopped the run, because the plan had
wrongly called this map rule-free. The map with truly no rule is the random cipher. It becomes the control and runs
first. Elicit becomes a scored map, whose rise in carry is exactly the movement the test of r needs. Four cheap checks
in the same job ask whether that carry is real:
- does it last with more training;
- does it vanish when the values are shuffled between keys;
- does the update lift the model's top guess or the key itself;
- does it follow how strongly the model already preferred each value?

### Design check (`/design`)

- **Goal.** As Amendment 2: does reading A's claim ("r reads which pieces a fine-tune builds from, not whether what it
  learned generalises") survive when carry is given room to move? Elicit now provides the movement (dx about 0.24).
- **Nearest tried.** Amendment 2 (job 46383, stopped at the control); E58 (cipher x −0.30 to −0.38 at 24 keys, the
  no-rule reading); Amendment 1.
- **Patch or step.** A patch on Amendment 2's mis-specified control, plus the step it was meant to take.
- **Decisions.** E58's label and gate; E54's recipe, selection and cipher rule; Amendment 2's common-step rule (kept as
  filed, not changed after seeing numbers); reading A.
- **What changes next.** Control fails → the label is broken at 128 values, and no map's x is readable. Check 3 fails
  → redesign the label before scaling keys. B flat with elicit testable → item 1's strongest line. B tracks → Fable's
  (b), to the user.
- **Control first.** The cipher and elicit are trained (training only, about 15 s each, because elicit's E12 step sets
  the common step). The cipher's label is then read first and alone, and the job stops unless it reads "pass" or
  "below".
- **Class.** Approved (lead's decision tree, Fable's advice adopted); $0, cluster `debug`.

### What differs from Amendment 2

- **Control.** The cipher (`kv.bijection`, E54's pool and gate) on the same 128 keys and windows replaces elicit as the
  control.
- **Elicit is scored.** Elicit is a map in B.
- **B uses own differences.** Each scored map is compared with itself at 24 keys: dx = x at K keys minus the 24-key
  three-seed mean, and the same for r and the ratio. Amendment 2 differenced against elicit; no map is a no-rule
  reference for the others now. At the common step (10 in job 46383) elicit's update size matches the 24-key states
  (‖δW‖ 3.12 against 3.12-3.18), so step and size drift is small. The norm-matched cipher's r and ratio are reported as
  the drift line.
- **Step-30 companion.** Every scored map is also read at step 30, and B is re-read there as a companion, reported but
  not scored.
- **Fable's checks 1, 3, 4 and 5 are added.** Check 2, the cipher, is the control itself.
- **Fallback for a too-small control.** If the cipher's G_spec is under the gate at the common step, it is read at the
  first kept checkpoint after it, up to step 30, that is over the gate; none stops the job (see the denominator below).

### The denominator (from disk, before filing)

- **Elicit.** At step 10, G_spec was 9.34 (P 0.60). At step 30 (P 0.98) it is expected at 15-19, like the 24-key
  step-10 states.
- **Cipher.** At 24 keys, steps 8-9, P 0.41-0.57, G_spec was 9.4-10.7. Elicit's step-10 P fell from 0.98 to 0.60 at 96
  keys, so the cipher's step-10 P is likely 0.02-0.2 and its G_spec about 4-9. That may fall under the 5.148 gate.
- **The fallback.** If the cipher is under the gate at the common step, the same rule is read at the first checkpoint
  after it, up to 30, over the gate (`pick_cipher_step`); none stops the job.
- **What the cuts mean at G_spec 9.3.** The control's +0.05 is 0.47 log-odds of H_spec, copy's 0.6 is 5.6, and rank-1's
  0.3 is 2.8.

### Numbers and what each should read

| number | zero update | cipher (control) | elicit | rank-1 | copy |
|---|---|---|---|---|---|
| x, common step | — | −0.2 to −0.6, "below" | 0.20-0.23 (job 46383 0.213) | ≥ 0.3 rule / ≤ 0.15 memorised | ≥ 0.6 (lean 40%) |
| G_spec, common step | 0 | 4-9 (fallback if under 5.148) | about 9.3 | open | open |
| r / ratio, common step | — | reported (24 keys 0.22-0.24 / 0.30-0.34) | 0.75-0.85 / 0.9-1.1 under reading A | same | same |
| x, step 30 | — | — | open (check 1) | reported | reported |
| check 1: x at 10, 30, 100, 300; 24-key re-train | — | — | kept or rising 40%, falling 35%, transient 25%; re-train step-10 x within 0.01 of −0.027 | | |
| check 3: shuffled elicit x | — | — | pass or below (about 0), 80% | | |
| check 4: elicit's update on rank-1 / copy held-out windows | — | — | copy H_spec within ±0.5 log-odds of 0; rank-1 H_spec open | | |
| check 5: rho(per-key H_spec, W₀ P) and (·, specificity) | — | — | > 0, about 70% | | |

*Blind to:*
- one seed per map;
- B absorbs no common drift with the key count, because no map is a reference; the cipher's r is the reported drift
  line;
- check 3 keeps the held-out keys' true values, so it reads the pool and the token kind, not a per-key rule;
- reading A's own blind spot (pieces, not carry).

### Predictions (filed in `results/predictions_amend3.json`)

- **Control.** `elicit_control` on the cipher at the step `pick_cipher_step` picks (the common step, or the first one up to 30
  over the gate): pass or below
  continue; fail, unreadable or no step over the gate stop. Lean: below 75%, pass 15%.
- **R, C.** As Amendment 2, at the common step. Leans: rank-1 memorises or between 65%, carries 25%; copy pass 40%.
- **B.** `b3` on elicit, rank-1 and copy: dx_min 0.15, tol_r 0.05, tol_ratio 0.1, tracks > opposite > flat >
  not_testable; "untested_drift" if the norm-matched cipher's r moved more than 0.05 from its 24-key mean 0.2306.
  Lean: flat 50%, tracks 25%, untested_drift 10%.
- **Verdict.** `verdict2(control, B, R)`. Lean: reading_A_holds 50%, b_r_tracks 30%.
- **Checks 1, 3, 4, 5.** Reported with filed expectations (above); none stops the job.

Edge cases tested before filing:
- `pick_cipher_step`: the common step over the gate, a later step, the tie at the gate, nothing up to 30, a nan;
- `b3`: nothing moved (not_testable), a flat elicit with dx 0.24, an r rise of 0.06 (tracks), ratio nan (r decides), the
  dx tie at 0.15, missing maps, the drift guard both ways, nan and with no cipher reference;
- `trajectory`: ties at x − 0.05 and at e_max, a missing checkpoint, nan, no later step;
- `spearman`: perfect ±1, tied ranks (0.949), fewer than 3 pairs, a constant side, nan, unequal lengths;
- `score_amend3`: the flat branch, stopped, an empty record.

### Settings (each sourced)

- **Unchanged from Amendment 2:** keys, windows, maps, recipe, seed 0, label, gate and the B tolerances (`settings`,
  `thresholds`).
- **Cipher:** E54's `build_tasks` rule (word-like ranks 1000-2999, seed 0, gate P < 1e-4 over all the task windows with
  `logits_of`).
- **Shuffle:** a seeded cycle over the trained keys (seed 0), so no trained key keeps its value; Fable's "a permutation"
  made a derangement so that check 3 has no fixed-point carry.
- **traj_tol 0.05:** the control's tolerance.
- **Step 30:** the next E12 checkpoint, as Fable proposed.

### Stop rules, runtime, outputs

Stops (exit 1; `amend3.json` written with "stopped"), in order:
- the key selection is not reproduced (about 1.5 min in);
- a zero update is not 0, or the shuffle left a fixed point;
- elicit reaches no E12 checkpoint;
- the cipher under the gate at every step from the common step to 30, or its control not "pass" or "below" (about 3 min in);
- a block-0 shift over 1e-5 nats.

One MATS `debug` L40 job, $0, about 12-18 min:
- six fine-tunes of about 12-16 s each;
- eight r/ratio/S reads of about 40 s each (Amendment 1 took 10 s on 75 windows; these have 286);
- about twenty labels of a few seconds each.

Host RAM peaks at about 5 GB while the cipher's 32 checkpoints are held; they are freed after the control. Disk is about
0.75 GB in `results/large/amend3/` (six saved states of 113 MB, eleven small window files). Outputs:
`results/amend3.json` and `results/large/amend3.out`.

### Decision tree

1. Control fails or is unreadable → the label is broken at 128 values; no x at 128 keys is readable; to the lead.
2. Check 3 reads "fail" (reported) → the write-up says the label reads the pool at 128 values; no 128-key x is used
   until it is redesigned (specific gain against a random token pool). B reads "untested_drift" → verdict untested.
3. **reading_A_holds** → elicit's carry rose about 0.24 while r and the ratio held: item 1's claim with its strongest
   line, on the map closest to E54's.
4. **b_r_tracks** → Fable's (b); to the user before any claim.
5. **a_first_reuses** → Fable's (a), as in Amendment 2.
6. Check 1 "transient" (x over 0.05 at the common step, at most 0.05 at the next) → carry is a transient of under-trained states; B must be re-read at matched
   trained-key P (a dated amendment), and reading A is unchanged. Check 1 "rising" (kept, and x at step 300 over x at the
   common step + 0.05) →
   elicit is a candidate "reuses" state for step 2's P2.

### Departures from Fable's recommendation

- The fallback for a too-small control. Fable reads the cipher at the common step; the filed denominator estimate says it
  may be under the gate there.
- The shuffle is a derangement (a cycle), not any permutation.
- Check 3 is reported, not a stop (Fable asked it to stop the job before rank-1; the review judged a check-only stop
  too costly for a reading Fable expects at about 0).
- B now uses own differences for every map. Fable gives elicit "its own dx"; the same is applied to rank-1 and copy
  because elicit can no longer be their reference.
- Check 4 reads the new common-step elicit state rather than `amend2/elicit_s0.pt`. Its max |difference| from that file
  and its x are reported against job 46383's.

### Order of work

1. Done: code, tests (34 pass), lint clean, predictions filed; review fixes applied (note below).
2. Fresh-Opus review of the diff.
3. `/pod-runner`: `sbatch cluster/open_pair.sbatch amend3` from the clone root.
4. Write-up as a section of E58's analysis report (`/writeup`); `TRIED.md` row covering Amendments 2 and 3;
   `PENDING.md` line.

### Amendment 3 before the run (review, 2026-10-30)

A fresh review of b60ae7e found seven things. Nothing had run, so the section above and `predictions_amend3.json`
were amended in place (`amendment_2026_10_30_prerun_review` records it).

1. **Check 3 is report-only.** The shuffled elicit map no longer stops the job: rank-1 and copy always run once the
   cipher control passes.
2. **Cipher fallback.** If the cipher's G_spec is under the gate at the common step, the control is read at the first
   kept checkpoint after it (steps 11-30) with G_spec over the gate (`pick_cipher_step`, replacing `control3`); none by
   step 30 stops the job. The cipher is also read at step 30, with r and ratio, beside the step-30 companion.
3. **Reproduction of job 46383.** A sha256 of a `torch.save` file embeds its basename, so the check could never match.
   It now reports the max |difference| between this run's common-step elicit update and `amend2/elicit_s0.pt`, with
   both x values.
4. **Drift guard for B.** B (and B at step 30) reads "untested_drift" if the norm-matched cipher's r at 128 keys is not
   finite or differs from E58's 24-key cipher mean (0.2306) by more than tol_r (0.05). With own differences nothing else
   absorbs a shift of every r with the key count. Tests cover both directions, nan and a missing reference.
5. **Check 4's expectation** is stated in H_spec (copy within ±0.5 log-odds of 0), not in x.
6. **Check 1's trajectory** codes the tree's branches: "no_carry" (x at the common step at most 0.05), "rising" (kept,
   and x at 300 over x at the common step + 0.05), "kept", "transient" (only when x at the common step is over 0.05),
   "falling".
7. **`settings.order` corrected.** `add_maps` still calls `KD.vocab` a second time (about 10 s) for E12's ranked pool.
   Removing it would need `amend2.stage_keys` to return `ranked`, which would change amend2.py; it is kept.
