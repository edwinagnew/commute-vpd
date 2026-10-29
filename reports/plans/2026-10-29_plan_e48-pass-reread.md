# Plan: do the saved teacher-trained copies keep the fine-tune's behaviour in the parts Γ switches on? (E48, pass re-read)

*2026-10-29. Writer agent for commute-lead. Approved by the user 2026-10-29 (`decisions.md`, the two "Card triage"
rows, items (2) and (7); triage table `reports/decisions/2026-10-29_decision_card-triage-draft.md`; card
`decision-e42-criterion` in `docs/research_log/parts/sections/decisions.html` before v98). Code:
`experiments/pass_reread/run.py`, `src/commute/pass_reading.py`, `cluster/pass_reread.sbatch`. Predictions:
`experiments/pass_reread/results/predictions.json`, filed before any run.*

## The question, in plain words

E42's copy of a word-cipher fine-tune passes the agreed test on ordinary text, but the model rebuilt only from the
subcomponents Γ marks important at the cipher's cue keeps almost none of the cipher. The user adopted a new part of the
pass (option (b)): **behaviour kept by the parts Γ switches on, leftover weights at the original's, on the task's own
text**, reported as a share of the fine-tune's change and as a ratio to W₀ on the same rows, in the form the Qwen
re-plan uses (`reports/plans/2026-10-28_plan_qwen-line-replan.md`, Q0 reading 2; scored as
`reports/plans/2026-10-28_plan_qwen-q0q1.md` section 3's r2). First, W₀'s own decomposition is read on the cipher
windows: if its masked model is already poor there, the new reading would judge VPD rather than the map, and the pass
falls back to option (c), reading A deciding with the cue reading printed beside it. Otherwise every saved
teacher-trained copy (E42, E34 including the rare-target task-text copy) is re-read under the new reading, and the
subcomponents that carry the E42 copies' gain are listed.

In `GOALS.md` → Terms: the states are **teacher training** copies (component training with the fine-tune as teacher,
Γ and Δ frozen) and the **do-nothing** reference (δW into Δ). The reading is not a map; it is a measurement added to
the weak version's pass.

**A definition conflict, flagged for the user (`.claude/CLAUDE.md` → Terms).** The user's words fix Δ in the masked
forward "at the original's" (Δ₀ included, mask 1). The Qwen Q0 plan, which the decision says to copy, runs its CI masks
with Δ **off** (section 3, "CI masks run with Δ off, so neither the old nor the new Δ enters r2"); the re-plan itself
says Δ "frozen at W₀'s value". Readings: (i) Δ₀ in, as the user's words: the scored reading here; (ii) Δ off, as Q0's
code and E42's CI-masked cells: computed on every state and reported beside, at no extra job. For teacher-trained
copies Δ = Δ₀ anyway, so the two differ only by what Δ₀ contributes under the mask. **The lead should get the user's
acknowledgement of (i) before submitting**; the stage-1 decision is the one place the choice is likely to matter (below).

## Design check (`/design`)

- **Goal served.** `GOALS.md` → weak version: "the components must carry the new model". The new reading asks it at
  the task's own inputs, where E42 showed the Pile checks are blind (the cue never occurs there). It decides whether
  "teacher training passes" survives on new behaviours (E42) and on behaviours the model half-has (E34), the input to
  the commute line's next design and to the rare-target card (7).
- **Nearest tried.** E42 (`TRIED.md` row 85) read the CI-masked model at the trigger with Δ off, as P(target) and as a
  log-odds route split; it did not read the KL to the fine-tune, the share against W₀'s own masked KL, or any E34 copy
  at the trigger in this form. E47 (claim tests) re-reads saved states under several adversary starts: reading A only.
- **Patch or step.** A step: it applies an adopted pass rule to every saved copy, once, so later maps are read against
  the same rule on both lines.
- **Decisions relied on.** `decisions.md` "Card triage" (2), (3) (the fixed 1.5 line, do-nothing reported beside every
  verdict), (7) (rare target: no rerun; re-read the task-text copy). Γ and Δ frozen (`GOALS.md`). No departure.
- **What changes next.** Stage 1 fallback: the pass stays reading A with the cue reading beside it, and the E42 card's
  "(b)" is replaced by "(c)" for the cipher. Stage 2: if the E42 random-cipher copies fail the new reading (expected),
  every frozen-Γ map fails new behaviours under the pass, and the next commute design is masked teacher training or
  updating Γ (E42 "What next" 3, 4); if E34's copies also fail, "teacher training passes on old behaviours" is
  withdrawn under the new pass. Rare target: "not placed" closes the card either way (decision (7)).
- **The control, first and alone.** Stage 1 is W₀ by itself, with two apparatus gates and the filed stop rule.
- **Approved class.** 1 (analysis of what exists: GPU passes of a reading on saved states) and 3 (the decision's
  named next step). No sign-off needed; the Δ-convention acknowledgement above is the one open item.

## The reading

Per state X, on its task's held-out windows, at the trigger position (where the behaviour is; the mean KL over the
other positions is reported beside):

- **Masked model of X.** Every subcomponent at Γ's g, Γ reading X's own activations; Δ held at W₀'s Δ₀ under mask 1.
- **K(X)** = mean over windows of KL(fine-tuned model ‖ masked model of X), nats. **K(D)** = the unchanged
  decomposition's K against the same fine-tune. **K₀** = W₀'s own KL(W₀ ‖ masked W₀) on the same windows.
- **Share kept** = [K(D) − K(X)] / [K(D) − K₀] (`maps_score.kept`, the Qwen line's r2; scored where K(D) − K₀ ≥ 0.05
  nats). 1 means X's masked model is as close to the fine-tune as W₀'s is to W₀. **Pass line 0.8** (Q0 plan's r2 line).
- **Ratio to W₀** = K(X) / K₀, reported with the fixed 1.5 line printed beside it, not scored (decision (3) fixes the
  line for the adversarial check; on this reading a peaked fine-tune gives small KLs and ratios below 1, see blind spots).
- **New pass** = reading A's filed verdict "pass" **and** share "kept", for copies whose check (below) matched.

Code: `pass_reading.trigger_cells` (new), `maps_score.kept` (existing, unchanged). Every cell is also read with Δ off.

**Blind spots.** (1) The share is in nats, so it is lenient on probability: E42's rand32 copy at P 0.007 where the
fine-tune puts 0.99 is expected to keep about 0.6 of a 13-nat change. (2) It cannot fail do-nothing where Γ alone
carries the behaviour (ident32, probably rungs 0-1); reading A fails do-nothing there. (3) Trigger position only,
mean over windows, one batch order per copy (the spread is measured on two seed pairs, P6).

## States

Teachers are never saved; each is regenerated in-job by its filed recipe (all 24 matrices, arm A's loss, 1,000 steps,
seed 0) and checked against its results file: P(target) within 0.01 and size within 0.003 (amend6's `ref_tol`). Each
copy is loaded from its saved bf16 offsets (`saved_states.offsets_state`) and checked: its own P(target) within 0.01 of
the file's, and its teacher gap within 0.005 (arm A) or 0.02 (arm B) times the teacher's gap at W₀ where the file has a
gap (E41's `not_worse` tolerances, about twice E34 Amendment 4's batch-order spread). A copy that misses either check is
"unmatched": read and printed, not scored. Amendment 5's r1 copies have no gap in their file, so their check is P only.
A do-nothing state is scored only when its teacher's check matched. A regenerated copy (job 2) is matched by its
teacher's check alone; E34's H100 P and gap are printed beside it and never unscore it.

**Reading A's verdict per copy** comes from its results file. Amendment 5's copies have none; for its batch-order seed 0
copies (A0, B0), E34's own r1 SGD copies (`teacher_training_r1_t534_v369.json` → `teachers.sgd.{A,B}["1000"].verdict`,
both "pass") are the same recipe, seed and card, and their P(target) agrees with Amendment 5's to 2e-4 (0.99630 vs
0.99624; 0.97417 vs 0.97434), so that verdict is used. Seed-1 copies (A1, B1) have no reading-A verdict: their new pass
is written as None, not False.

| teacher | recipe (source) | card of the copies | saved copies read (all exist, `ls` 2026-10-29) |
|---|---|---|---|
| rand32, rand1, ident32 | Adam 3e-5 (E10 / E42 `newskill.py`) | L40 | `e42/<task>/{A,B}_step1000.pt` (6) |
| r1 t534 SGD | SGD 0.03 (E34; amend5/amend6) | H100 | `r1_t534_v369/amend5_{A,B}_sgd_seed{0,1}_step1000.pt` (4) |
| r2 t275 SGD (rare target) | SGD 0.1 (amend6) | H100 | `amend6/r2_t275_v2667/amend6_sgd_{A,B}_seed0_step1000.pt` (2); **A is the rare-target task-text copy** |
| emoticon, r0 t352, r1 t534 LoRA | rank 8, Adam 1e-3 (E9; `amend3.lora_teacher`) | H100 | `amend6/<task>/amend6_lora_B_seed0_step5000.pt`, `..._seed1_step2500.pt` (6) |

Plus, per teacher, its **do-nothing** state (8). In the second job, **regenerated**: E34's six arm-A copies (emoticon,
r0 t352, r1 t534 × Adam 3e-5 and SGD 0.01/0.03/0.03 teachers), trained on the L40 by E34's recipe (Adam 1e-5, 1,000
steps, batch 32, batch-order seed 0, Γ and Δ frozen), saved as bf16 offsets to `results/large/regen/`, read the same
way, with their do-nothing states. They differ from E34's own copies by the card only (E34 ran on an H100); their E34 P
and gap are printed beside, and reading A's verdict beside them is the H100 copy's.

Paths are under `experiments/teacher_training/results/large/` (the symlink to
`~/commute-data/results-large/teacher_training/`). `experiments/SAVED.md`'s teacher_training section predates amend6
and E42 and should gain these rows (a lead task).

## Stage 1: the control (runs first, alone; early stop)

All at W₀, in the first job. Pile rows are the harness's 256 val rows.

| check | reads | must read | else |
|---|---|---|---|
| A1, apparatus | W₀'s Δ-off masked KL per position on the Pile rows | 0.348 within 2% (`faith_score.W0_CI`) | stop: the masked forward is not the harness's |
| A2, apparatus | fp32 KL(W₀ ‖ every mask 1, Δ₀ in), first 16 Pile rows | ≤ 1e-4 (expected ~1e-7; Δ off reads ~0.012) | stop: Δ₀ never entered the masked forward |
| F, the stop rule | K₀ at the trigger of rand32's and rand1's held-out windows ÷ W₀'s Δ₀-in KL per Pile position | below 1.5 on both: **read** | at or over 1.5 (or NaN) on either: **fallback (c)**, the job stops |

An apparatus miss (A1 or A2) stops the job with a nonzero exit and is not cached (written as `S1_apparatus_failed`), so
a rerun recomputes stage 1. The planned fallback is a result, not an error: the job writes S1, its score record and
`stopped: fallback ...`, and exits 0. Results files store non-finite numbers as the strings "NaN", "Infinity",
"-Infinity" (strict JSON). **After any code change, move `results/<job>.json` aside before rerunning**: a rerun on
the same card resumes every finished record.

**The threshold and its source.** 1.5 is the project's fixed line (`faith_score.LINE`; decision (3): "the fixed line on
the ratio to W₀"), applied to W₀ itself: W₀'s masked model counts as already poor on the cipher windows when it is at
least 1.5 times as far from W₀ there as on ordinary text, the margin at which a map would fail.

**Expected.** Δ off, the existing numbers already read F ≈ 2.0 (E42 Amendment 1: 0.69 at the trigger of rand32's
training windows against 0.35 on Pile). With Δ₀ in, both numbers move by an unknown amount. **Lean: fallback,
medium-low.** So the stage-1 branch rests largely on the threshold and on the Δ convention; the lead should put both to
the user with this plan. Printed beside, not scored: F with Δ off; K₀ at the trigger of ident32 and of the four E34 tasks'
windows (emoticon, r0 t352, r1 t534, r2 t275), the same position type on tasks VPD is not suspected of losing; the mean
K₀ over the cipher windows' other positions (E42 Amendment 1: 0.44 Δ off).

## Stage 2: the saved copies under the new reading

| number | at W₀ | do-nothing (expected) | control | E42 copies (expected) | if reading A only | if the new reading |
|---|---|---|---|---|---|---|
| K(D), nats | — | — | — | rand32 ~13, rand1 ~10.4, ident32 ~10.1 | — | — |
| share kept | 1 by definition | rand32 0.6-0.8, rand1 0.2-0.5 (negative control) | ident32 do-nothing 0.9-1.1 (positive control) | rand32 0.5-0.75, rand1 0.2-0.5, ident32 0.9-1.1 | E42 passes on all three | E42 fails on both random ciphers |
| ratio to W₀ | 1 | rand32 5-9, rand1 9-14, ident32 < 1 | — | rand32 ~8, rand1 ~11, ident32 < 1 | — | reported |
| masked P(target) | W₀'s | 0.016, 0.001, 0.958 (E42, Δ off) | — | 0.007, 0.001, 0.929 (E42, Δ off) | — | printed beside every verdict |

E34 copies: no reference for the masked share. E5's label-trained component training kept 0.56 (emoticon), 0.82
(r0 t352), 0.79 (r1 t534), 0.66 (r2 t275) of the log-odds gain under the CI mask at step 10; teacher training at step
1,000 is unread. The E42 Pile-text copies of rand32 and ident32 did not learn (P 0.02, 0.001), so their shares read
about 0. Expected values for rand32 and rand1 were derived from E42's masked P with the target's mass dominating the KL.

## Stage 3: the subcomponents that carry the gain (E42 "What next" item 1)

Per E42 arm-A copy (rand32, rand1, and ident32 for contrast): for each of the 9,881 alive subcomponents, the drop in the
copy's mean trigger log-odds of the target when its change alone is reverted (`attribution.revert_one`, fp32, one
subcomponent per forward). Reported: n50 and n90 (the fewest largest drops reaching 50% and 90% of the summed positive
drops), the n50 set's share with mean g at the trigger below 0.01 under the copy, the n50 set reverted together as a
fraction of the gain, and the top 20 with g under the copy and under W₀. The time of the first matrix is logged
before the other 23 run. "A handful" (n50 ≤ 10) points to a small Γ
change as the repair; "hundreds" (n50 ≥ 100) does not. Runs in job 1 after stage 2, only if stage 1 read "read".

## Predictions

Filed in `experiments/pass_reread/results/predictions.json` (rules, thresholds, sources, leans, edge cases); summary:

- **S1** (stop rule above). Lean fallback, medium-low.
- **P1** E42 arm-A copies: rand32 "not kept" (high), rand1 "not kept" (high), ident32 "kept" (medium-high). Class
  rule: unscored if K(D) − K₀ < 0.05; "missing" on NaN (never a filed class, so a miss); "kept" if round(share, 9) ≥ 0.8;
  else "not kept". Absolute.
- **P2** controls: do-nothing rand32 and rand1 "not kept" (rand1 high, rand32 medium: expected near the line),
  ident32 "kept" (medium-high). The reading has teeth if rand1 reads "not kept" and ident32 "kept"; the run's score
  writes this as `P2_teeth` (True, False, "unmatched" if either teacher did not reproduce, None if not read).
- **P3** rare-target task-text copy "not kept" (low; its teacher may not reproduce on the L40, then "unmatched").
- **P4** r1 t534 task-text copies, seeds 0 and 1: "kept" (low). Pile-text copies reported without a lean.
- **P5** (regen job) E34's six arm-A copies: emoticon "not kept", r0 t352 and r1 t534 "kept" (low on all).
- **P6** batch-order spread: max |share(seed 0) − share(seed 1)| over amend5's two pairs, expected under 0.05; a
  class within that distance of 0.8 is flagged "near line".
- **P7** carriers: rand32 and rand1 "tens" (low).

Edge cases run against each rule before filing: F NaN or a zero Pile K → fallback (A1 fails first); F = 1.5 →
fallback (inclusive); K(D) − K₀ below the floor → "unscored", a miss of any filed class; a NaN share → "missing", a miss
(it would otherwise have counted as "not kept", which three predictions want); an unmatched copy → "unmatched", not
scored; no positive drop → n50 None, "none positive"; n50 exactly 10 or 100 → "a handful" or "hundreds"; no scorable
seed pair → spread None, nothing flagged.

## Decision tree

- **A1 or A2 fails:** stop; the masked forward is wrong; fix the code, rerun job 1 ($0).
- **S1 fallback:** the job stops; the analysis report says the pass for the cipher stays reading A with the cue reading
  printed beside it (option (c)); job 2 is not submitted. Pending card: confirm (c) for the cipher, or another
  threshold (the F with Δ off and the E34-trigger references are printed for that). Under (c), the cue readings of the
  E34 copies are not yet printed anywhere; rerunning job 1's stage 2 as "reported beside" is a $0, 10-min option for the
  card.
- **S1 read, P2 teeth fail** (ident32's do-nothing not kept, or rand1's kept): the reading cannot separate the cases it
  exists for; report, no verdict changes; card.
- **S1 read, teeth hold:** score P1-P4; submit job 2; then P5. Every verdict carries do-nothing beside it (decision (3)).
  - E42 random ciphers not kept (expected): every frozen-Γ map read so far fails the new pass on new behaviours; next
    commute design = masked teacher training (E42 "What next" 3) or updating Γ, a plan for the user.
  - Any E34 task-text copy not kept: flagged loudly (it would withdraw "teacher training passes on old behaviours"
    under the new pass); check-in with the user before any further E34-based claim.
  - Rare target: "not kept" → card (7) closes "not placed"; "kept" → "not placed, narrow miss" stands;
    "unmatched" → an H100 regeneration of the r2 teacher (about $1) goes to the user as a card.
- **A copy unmatched** (teacher or copy not reproduced on the L40): printed, unscored; the H100-trained ones get one
  card line for an H100 re-read if any of them decides something.

## Cost and routing

MATS cluster, `debug` QoS, one L40 per job, $0. Two jobs, `cluster/pass_reread.sbatch`, 30-min limit each.

| job | stages | estimate | source of the estimate |
|---|---|---|---|
| 1 `saved` | load and bases ~1.5 min; S1 ~2 min; 8 teachers ~4.5 min (35 s per full fine-tune, ~20 s per LoRA); 26 state reads ~2 min; S3 3 × ~1-2 min | **~15 min** | E42 log (teacher 35 s, bases 10 s), amend6 (LoRA 19-20 s on an H100); S3 is 9,881 small fp32 forwards per copy, unmeasured |
| 2 `regen` | load ~1.5 min; 6 teachers ~3.5 min; 6 copies × ~115 s ~12 min; reads ~1 min | **~19 min** | E42 log: arm-A copy 115 s on an L40 |

Memory 32 GB host (E47's request for similar reads); GPU well under 46 GB (logits of 64 windows × 41 tokens).

## Order of work for the fleet agent

1. Lead: put the Δ convention and the stage-1 threshold to the user (this plan, "A definition conflict" and Stage 1),
   in plain words; on acknowledgement of (i), go on.
2. Fresh-Opus review of the code (`.claude/CLAUDE.md` → One review before a paid run; $0 here, but new measurement
   code): `src/commute/pass_reading.py` and `experiments/pass_reread/run.py`; fix what is real.
3. `/pod-runner`, job 1 `saved` alone (`sbatch --job-name=commute-e48-saved cluster/pass_reread.sbatch saved`); the
   runner reports S1's numbers and decision before anything else is read. Job 2 `regen` only if S1 read "read".
4. Analysis report by `/writeup` (`reports/analyses/2026-10-29_analysis_e48-pass-reread.md`), independent check,
   `TRIED.md` row, `SAVED.md` rows (amend6, e42, pass_reread/regen), log card, PENDING lines.

## Not in this plan

- E41's copies (T10, T100 one-shot and the chains): none saved; 12 copies at 115-158 s on an L40 plus teachers is about
  30 min, a job at the debug cap's edge; E41's one-shot T1000 arm-A copies on emoticon and r1 are E34's Adam copies,
  which job 2 regenerates. A third job if the lead wants them.
- E34's six main arm-B copies: not saved; another ~16 min, a third job; arm B is read on the 11 saved Pile-text copies.
- E42 Amendment 2's M0 states (teacher = W₀: K(D) − K₀ = 0, share undefined); E43's and E42 Amendment 1's states (not
  saved).
- Any adversarial or stochastic reading: reading A's verdicts come from the results files.
- An H100 re-read of the H100-trained copies, unless their checks miss and they decide something.
- Changing any shared function: `pass_reading.py` is new; `maps_score.kept` and `attribution.revert_one` are called
  unchanged (revert_one with a per-window target tensor and one subcomponent per forward, a path its int-token callers
  in `identity_test` and `concentration_test` do not use). No finished experiment's frozen script can break.
