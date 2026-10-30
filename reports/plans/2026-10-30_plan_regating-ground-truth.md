# Plan: does the re-gating line's kind number agree with a ground truth it did not make, on the 67M? (E-number at run time; next free E58)

*Regating-lead, 2026-10-30. Step 2 (option B) of the adopted programme `reports/plans/2026-10-30_plan_regating-development-FABLE-DRAFT.md`
(`decisions.md`, 2026-10-30, ask #13). Plan only: no code, no predictions filed, nothing run. Read-only on the reuse
line's files. Words as in `GOALS.md` → Terms.*

## The question, in plain words

The re-gating line has one number that reads a fine-tune of something the 67M model already does as different from a
fine-tune of something it cannot do: the stand-in ratio (Γ's re-switching gain over the moved pieces' gain; lowercase
0.92 vs cipher 0.26 in E53, elicit 0.95 vs cipher 0.31 on E54's pair), with the route split's re-gating share r beside
it (E54 0.77/0.76 vs 0.24/0.21) and the steering share S as the no-decomposition baseline (0.38/0.39 vs 0.23/0.19).
Every "elicited" or "new" label behind those numbers was set by how we built the task. The request, restated: put the
stand-in ratio, r and S for every 67M fine-tuned state into one table beside the reuse line's label for that state,
"reuses" or "new" by Jain et al. 2024 Def 2 with the user's bound on g (g learnt on the fine-tuning rows must also work
on held-out values; `decisions.md` 2026-10-30, a dated departure), which reads behaviour and an interchange patch and
no decomposition; then score whether the ratio and r agree with that label, and whether they agree where S does not.
"Elicited" in the re-gating line's wording maps to "reuses" in the reuse line's; every claim says "elicited at the
level of VPD's pieces".

**What exists today (found 2026-10-30).** The reuse line has labelled **no 67M state**. Its labelled states (26 scored,
10 in an unscored third class; `experiments/pcfg_anchors/results/c1.json` → `scored`, `class_B`) are fine-tunes of
LOW, a 6-block minGPT (~3M) on Jain's counting grammar, saved in reuse-lead's clone
(`~/commute-agents/reuse-lead/experiments/pcfg_anchors/results/large/{a2,amend5,amend5b}/`, 887 MB; the shared
`~/commute-data/results-large/pcfg_anchors/` is empty). They are not on the 67M decomposition: every ratio, r and S
reading uses `vpd.load_component_model`, the released decomposition `s-55ea3f9b` of target `t-9d2b8f02`
(`src/commute/vpd_files.py`), and no VPD decomposition of LOW exists (E55 Stage C2, gated, not approved). The 67M
labels are E55's **Stage B** (`reports/plans/2026-10-29_plan_reuse-stage1.md` § Stage B, lowercase only as written;
the bounded-g readings of copy, cipher and lowercase were added by the user's 2026-10-30 decision, costed at one $0
cluster job). Stage B has not run and is not scheduled: it is option (3) of reuse-lead's open ask #4 (card
`decision-e55-c2`), whose recommended option is (2), a $0 rerun on the counting model. **So nothing can be tested
yet.** This plan fixes the table, the rules and the re-gating side's job, and lists what the reuse line must supply.

## Design check (`/design`)

- **Goal.** `GOALS.md` "If no simple Φ works: new capabilities or strengthening?" The re-gating line's answer rests on
  a number whose labels came from task construction; this tests it against a definition it did not make.
- **Nearest tried.** E12/E46 (copy vs random map), E53 (lowercase vs cipher), E54 (elicit vs cipher at matched size):
  each labelled by construction. E55 C1 scored three no-decomposition instruments against the reuse truth on the
  counting model; none beat the training recipe. New here: the first score of VPD readings against that truth, and
  the first 67M state whose label construction does not decide (E54's elicit map, below).
- **Patch or step.** A step: step 2 of the adopted programme, not a fix of a run.
- **Decisions.** Relies on the reuse-line definition rows (2026-10-29, 2026-10-30 bounded g), the programme row
  (ask #13), E12's r ≥ 0.5 rule, E52's S rule. No departure. Stage B's timing is the reuse line's and the user's
  (ask #4), not decided here.
- **What changes next.** The tree below: disagreement on lowercase or copy stops the Qwen read of the ratio; a
  disagreement only on open states raises the granularity question (novelty proxy, second seed); agreement where S
  disagrees is the headline case for stage D.
- **Control first.** Stage 0 re-reads three reference states and must reproduce their filed numbers before any other
  state is read; the reuse line's own controls (Stage B's zero-update state and W₀ probe) precede its labels.
- **Numbers.** Three readings per state plus its label; three predictions; one positive control set.
- **Class.** The re-gating side is class 1 (re-reading saved states, $0) inside the adopted programme's step 2
  (class 4). It goes ahead without sign-off once labels exist. Not covered: asking the reuse line to label states
  beyond Stage B's three tasks, or to build new 67M tasks; those need reuse-lead's plan and the user.

**Power, said plainly.** Under the bounded g, the labels of most 67M states are fixed by how the task was built: copy
(g = identity) and lowercase (g = case map) generalise to unseen values, so "reuses"; every cipher is a lookup table
that says nothing about unseen keys, so "new" (`reports/decisions/2026-10-29_decision_e55-plan.md`, item two). On
those states agreement is close to guaranteed and repeats E12/E53/E54 under a new name. The only 67M state family
whose label is open is **E54's elicit map**: each key's value is W₀'s rank 2-20 guess, so g is a per-key rank choice
and may or may not carry to held-out keys. Its two seeds decide P2 and P3. Two runs of one task is too few to test the
claim in general (see "What the reuse line would need to supply").

## States

All fine-tunes of `t-9d2b8f02`, read through decomposition `s-55ea3f9b`. One vote per training run (task × seed) at
the save the reuse line labels. Nothing is regenerated, so no replayed-recipe differences arise.

| family | runs (seed 0, 1) | file | label by construction | ratio | r | S |
|---|---|---|---|---|---|---|
| E46 copy | ident32, ident32b | `regating_table/results/large/{ident32,ident32b}_s{0,1}.pt` | reuses | read, 0.87-0.97 (E46 A3) | read, 0.76-0.80 | ident32_s0 0.75 (E52) |
| E46 random map | rand32, rand32b | `regating_table/results/large/{rand32,rand32b}_s{0,1}.pt` | new | read, 0.36-0.41 | read, 0.27-0.29 | rand32_s0 0.16 |
| E53 lowercase | lowercase step 300 | `performed_anchor/results/large/dW_lowercase_s{0,1}_step300.pt` | reuses | 0.918, 0.929 | not read | not read |
| E53 cipher | cipher step 300 | `performed_anchor/results/large/dW_cipher_s{0,1}_step300.pt` | new | 0.264, 0.257 | not read | not read |
| E53 broad cipher | broad_cipher step 300 | `performed_anchor/results/large/dW_broad_cipher_s{0,1}_step300.pt` | new | not read (A1b) | not read | not read |
| E54 cipher | cipher | `keydependent_elicit/results/large/cipher_s{0,1}.pt` | new | 0.31 (E53 A2, post hoc) | 0.24, 0.21 | 0.23, 0.19 |
| E54 elicit | elicit | `keydependent_elicit/results/large/elicit_s{0,1}.pt` | **open** | 0.95 (post hoc) | 0.77, 0.76 | 0.38, 0.39 |

18 runs: 6 reuses and 10 new by construction, 2 open. `large/` paths are symlinks into `~/commute-data/results-large/`.
E46's `code_*` (learned-nothing) and E53's step-10/100 saves are reference rows, unscored. "Not read" cells are what
the re-gating job fills; r and S on E53's Pile-text tasks need the code writer to confirm `commute.steering` and the
route split accept E53's `Task` (target at case positions); if not, those cells stay "not read" and the ratio carries
those rows.

## Stage 0: the control (runs first, alone; early stop)

| check | reads | must read | else |
|---|---|---|---|
| zero update | G of a zero δW on each reference task | 0 exactly | stop, fix |
| E53 lowercase_s0 | stand-in ratio | 0.9186 ± 0.01 (E53 A2 `repro_ratio_tol`), G 1.671 ± 0.02 | stop, fix |
| E54 elicit_s0 | r | 0.7726 ± 0.03 (E54 `r_tol`) | stop, fix |
| E46 ident32_s0 | r, S, S at block 0 | 0.8024 ± 0.03, 0.7503 ± 0.001, \|S_block0\| ≤ E54's `block0_tol` | stop, fix |
| file identity | sha256 of every state file | equals the hash recorded by its source run | stop, fix |

The reuse line's label controls (its zero-update state and W₀ probe, Stage B) must have passed for any label used.

## Stage 1: fill the table (re-gating side, $0)

| number | how measured | at W₀ | do-nothing (δW in Δ) | positive control | if the ratio reads reuse | if it reads construction only |
|---|---|---|---|---|---|---|
| stand-in ratio ρ | E53's code (`performed_anchor/run.py` `cells`): masked gain ÷ subcomponent gain | undefined (gains 0) | undefined (subcomponent gain 0); g alone reported | E53 lowercase 0.92 / cipher 0.26; E54 0.95 / 0.31 | high on every "reuses" run incl. any open one labelled reuses | high on copy and lowercase, low on ciphers, and its value on E54 elicit unrelated to the label |
| r | E12's route-split share on the least-squares split state (`commute.keydep` / E54 `readings`) | undefined (gain 0) | not read | 0.80 copy / 0.27 cipher (E46); 0.77 / 0.24 (E54) | as ρ | as ρ |
| S | E52's steering share, best of blocks 1-5 | block 0 reads 0 ± 1e-6 | never reads the decomposition | 0.75 copy / 0.16 cipher (E52) | baseline column | baseline column |
| label | the reuse line's bounded-g Def 2 class | "neither" (no behaviour change) | "neither" | lowercase reuses, E53 cipher new (by construction) | — | — |

A reading whose denominator falls under its floor (E53 A2's bootstrap `floor_min` for ρ; E46's gain floor for r) is
"undefined" and that run leaves that instrument's count, reported.

## Predictions (filed-style; not filed)

Truth T(run) ∈ {reuses, new, neither}. Instrument labels: ρ-label "reuses" iff ρ ≥ 0.5; r-label iff r ≥ 0.5 (E12's
rule); S-label iff S ≥ 0.5 (E52, E55 C1). The 0.5 for ρ copies r's rule; it is not fitted to any labelled state.
Runs with T = neither, or whose labelled saves disagree, are excluded from P1-P3 and reported in the table.

- **P1 (ranking).** For ρ and for r separately: AUC over runs, P(reading of a reuses run > reading of a new run), ties
  0.5. Passes iff the exact one-sided permutation p ≤ 0.05. Relative (rank-based, no cut-off). *Edge cases:* fewer
  than 3 runs on either side, or one label only → "not testable", reported; 3 + 3 needs AUC = 1 (p = 0.05); an
  undefined reading removes the run for that instrument. *Lean: high* (16 of 18 runs carry a label fixed by
  construction that E12/E53/E54 already separate). *Blind to:* whether the number reads reuse or the task's build.
- **P2 (the open runs).** On E54 elicit_s0 and elicit_s1 (and any run the reuse line marks open before its labels are
  read): the ρ-label and the r-label equal T on every open run. Absolute. *Edge cases:* fewer than 2 open runs with
  T ≠ neither → "not testable"; seeds with different T → each scored, P2 "split", tree branch 4. *Lean: low-medium,
  about 45%:* ρ and r read elicit at 0.95 and 0.77, so P2 passes only if the label is "reuses", and a per-key rank
  choice need not carry to held-out keys.
  *Note (2026-10-30, after E58 job 46266; lead on Fable's advice, `reports/plans/2026-10-30_advice_e58-reading-FABLE.md`,
  `decisions.md` last row).* P2 is scored on rule tasks only: tasks where "reuses" by behaviour means a rule W₀ already
  has, so carry-over to held-out keys is reachable (the label's positive control, E58 Amendment 1's rank-1 or copy map,
  and Stage B's rows). On lookup tasks (a per-key table over W₀'s pieces, such as E54's elicit map) it is reported,
  unscored: there the held-out label asks whether a rule generalises and r asks whether the carrying subcomponents are
  old, and the two must disagree when new per-key routes run through old subcomponents (the granularity limit). E58's
  filed P2 miss stands as scored in E58; this note changes nothing filed there.
- **P3 (beside steering).** Agreement count over all scored runs: A_ρ, A_r, A_S. "VPD adds over steering" iff
  A_ρ − A_S ≥ 2 (resp. A_r − A_S ≥ 2): a one-run difference is one seed's flip. Ties or less → "does not add"; A_S
  above A_ρ by ≥ 2 → "steering better". *Edge cases:* runs where S is not read leave all three counts (same rows for
  every instrument). *Lean: medium:* on the construction-fixed runs ρ, r and S agree except possibly E53's lowercase
  S (unread); the open pair decides it, in either direction, since S labels E54 elicit "new" (0.38).

**Blind to (every instrument here).** The granularity limit: VPD's unit is the rank-one piece, so a new circuit
assembled from old pieces under new gating reads "elicited"; when T = new and ρ, r say reuses, this plan cannot tell
that case from a VPD error (only Fable's option C, a gating-pattern novelty proxy or a second decomposition seed,
could). The label's own blind spots: a g that generalises but is built by a new route reads "reuses"; the patch reads
one position. Breadth, input-dependence and size confounds are inherited from the pairs (E53 Amendment 2: breadth does
not explain E54's pair).

**Noise floor.** r: ± 0.03 across regenerations (E46), seed spread 0.029 (E54). ρ: seed spread ≈ 0.01 on E53
(0.918 vs 0.929; 0.264 vs 0.257) and its 90% bootstrap interval per state (E53 A2); the near-zero-gain random rows
(E53 `*/random`, subcomponent gain 0.19-0.39) show ρ is unstable when its denominator is small, hence the floor rule.
S: tolerance 0.001 on regeneration (E54). No reading on the scored runs lies within 0.1 of a 0.5 cut-off except S on
E54 elicit (0.38-0.39), so the labels are not noise-limited.

## Decision tree

1. Stage 0 misses → stop, fix ($0). Not a Blocking card unless the fix changes a filed number.
2. Labels not available (Stage B not run) → nothing scored; the table is filled on the re-gating side only if the
   reuse line confirms the state list; the card question goes to the user (below).
3. P1 fails for ρ on construction-fixed runs (lowercase or copy reads new, or a cipher reads reuses) → the ratio is
   not a reading of reuse; say so in the log and stop its Qwen read (draft §5). Tell the user (qualitative reversal).
4. P1 passes, P2 passes, P3 "adds" → the headline case: VPD agrees with the decomposition-free label where steering
   does not, on one open task. Report with n = 2 open runs stated; go to stage D only after step 1 (breadth) is
   settled; ask the reuse line for more open runs before any general claim.
5. P1 passes, P2 fails (E54 elicit labelled new, ρ and r say reuses) → a candidate granularity-limit case or a VPD
   error; the two cannot be told apart here. Raise a Soon card: Fable's novelty proxy on E54 elicit ($0) and a costed
   second-seed card. Steering is then the better labeller on this set; say so.
6. P2 "split" across seeds, or E54 elicit labelled "neither" → report descriptively; P2 not testable; ask the reuse
   line for a second open task.
7. P1 "not testable" (fewer than 3 labelled runs a side) → report the table only, no claim.

## Cost and routing

Re-gating side: one cluster `debug` job (L40, account c11), stage 0 then stage 1, reading 18 saved states; E46 read
eight E12 states in 86 s and E53 A2 read its rows within one job, so ≈ 10-20 min. $0. Scoring is CPU (jq or a small
script through `cpu-run`). Reuse side (Stage B): one $0 cluster job per its decision row; its cost is the reuse line's.
Alternative with more power, not in this plan: E55 C2 (one VPD decomposition of LOW, gated, $10-30, then ρ and r on 26
labelled states); it would need the ratio and r code ported to a minGPT decomposition and is reuse-lead's ask #4
option 1.

## Order of work for the fleet agent

1. Wait for the reuse line's Stage B state list and filed bars (ask #4 answered with option 3, or the user directs).
2. Read this plan, `performed_anchor/run.py` (`cells`, `reread_e53`), `keydependent_elicit/run.py` (`stage0`,
   `readings`), `src/commute/steering.py`, `src/commute/keydep.py`.
3. Write `experiments/regating_truth/run.py` (< 300 lines) calling those functions; no copying. Stage 0 alone first.
4. File P1-P3 in `results/predictions.json` with the reuse line's run list and its "open" marks, before reading any
   Stage B label; run each rule on its edge cases first.
5. Fresh-Opus code review; `/pod-runner` stage 0 then stage 1.
6. Join with the labels; score; `/writeup` report with "Not run" and "Departures from the literature" (bounded g; the
   one-vote-per-run unit; the 0.5 cut-off on ρ); `TRIED.md` row; `PENDING.md` line.

## What the reuse line would need to supply

- Stage B's bounded-g labels (reuses / new / neither, with behaviour and patch numbers) on the 18 runs above, at the
  files listed, and their bars filed before reading.
- For each task: f_C and g named, and the held-out values. For E54 elicit in particular: what f_C is (W₀'s next-token
  ranking at the key?), what g is (a fixed rank rule, or a per-key table?), and which held-out keys test it.
- Which runs it regards as open (label not fixed by construction), marked before its labels are read.
- For a test with real power: at least 3 open runs per label on the 67M (two tasks with two seeds each at minimum).
  Building such tasks (for example a Jain-style weakly relevant fine-tune on the 67M, where W₀'s map and a new table
  agree on training keys and differ on held-out ones) is a new task family: reuse-lead's plan and the user's call.

## Not in this plan

Fable's gating-pattern novelty proxy (option C; a branch-5 follow-up); a second decomposition seed; E55 C2 or any
porting of ρ and r to the counting model; Qwen; new 67M tasks; any change to E53, E54 or E55 files or predictions;
scoring "neither" or class-B runs; the restore share (dropped by the programme).
