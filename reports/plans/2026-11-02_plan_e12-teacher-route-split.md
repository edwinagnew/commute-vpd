# Plan: E12 Amendment 1, the route split on teacher-trained states

*2026-11-02. User's backlog entry, `backlog.md` line ~350. Plan only: nothing filed, nothing run. Amendment 8's report is provisional.*

We would run teacher training on E12's eight copy and random-map fine-tunes and read the route split on the results. Three of the
backlog's premises need correcting first, so they lead. The run is $0 and about 25 minutes, and its real question is narrower than the entry.

## Checks the brief asked for first

- **The claim about r is false for teacher training's own state.** Γ reads only activations (`component_model.calc_causal_importances`),
  and `vpd.compute_ci` takes them from the target weights. So r's gates follow the state's target weights, and `harness.measure` sets
  those to W₀ + δW of the state. The claim holds for do-nothing and the least-squares split, which share the fine-tuned weights: E42's
  r is identical on both (rand32 0.187). Teacher training's own state keeps Δ, so its weights are the student's, and r changes: 0.187 to
  0.132 on rand32, 0.635 to 0.643 on ident32, 0.089 to 0.037 on rand1 (`teacher_training_e42_*.json`). Only the plus-remainder state
  (E43: student's U′, V′, target the fine-tuned weights) has the fine-tune's gates, so there r equals E12's by code.
- **"$0" holds; "saved states" holds only for the teachers.** `results/large/new_vs_existing/` is empty, but E46 rebuilt all eight fine-tunes
  and windows: `regating_table/results/large/{ident32,ident32b,rand32,rand32b}_s{0,1}.pt` and `windows_*.pt`, sha256 in `stage0a.json`
  (not in `SAVED.md`, which stops at 2026-10-27). r matches E12 within 0.002 across cards. No student of these teachers exists: E42's
  saved students copy step-1000 teachers on two of these tables. So 8 students are trained in-job, about 100 s each on an L40 (`amend8/run_n3b.log`).
- **The citation is E42, not E43.** E43 ran emoticon and has no e_gate; the cipher reading is E42's.
- **E42 already answered one pair.** At step-1000 teachers: e_gate 0.31 copy, 0.84 and 0.92 random maps; r 0.64, 0.13, 0.04.

## Design check

**Goal.** `GOALS.md` asks whether a simple map carries an update past the decomposition, and if not, whether new capability is why. The
route is the one number that told the classes apart; this asks if it still does on the map's own output. **Nearest tried.** E42, above, and
E12. New here: within-class spread (E42 had one state per class, so E12's rule could not be scored), teachers at the first learned step, and
batch-order spread, which E42 left unmeasured. **Patch or step:** a patch, closing a gap in E42's claim. **Decisions.** Relies on reading A, Γ
frozen, teacher training's own state as the map; no departure. **Honest limit.** On teacher-trained states e_gate is nearly 1 − r: d, i and
e_Δ sum to 0.03-0.05 in E42. So there is one number here, and the only live question is whether Γ on a student's activations re-switches as
it does on the fine-tune's. Mostly a confirmation is expected; it is worth $0, not more. **What changes next.** Separates and r stays close:
E42's claim widens to two tables per class, nothing further runs. No separation, or r moves: the route is not a kind reading of a map's state,
the re-gating line's r stays on the fine-tuned model only (`decisions.md`, 2026-10-30 programme), and the user hears it before anything else
runs. **Class:** 2 (replication of E42's reading on E12's states), and user-requested. **Limit carried:** a new circuit from old pieces reads
"elicited", and E50 Amendment 1 did not show W₀ performs the copy task, so the classes are "copy" and "random map", not "existing" and "new".

## The request in the Terms' words

On each of E12's eight full-weight fine-tunes (copy and random-map key → value tasks, two seeds, first learned step), run teacher training
(component training with loss KL(teacher ‖ student), teacher = the fine-tuned model, Γ and Δ frozen). Read the route split at the trigger on
97 held-out windows. Does the masked-off share e_gate separate the classes by E12's rule, and what share of the copy's gain is re-switching
r against masked-off?

**Two readings of "teacher-trained state", which the user must pick.** (a) Teacher training's own state, Δ fixed: the map E43's tree kept; r
moves. (b) Plus remainder, weights exactly the fine-tune's: the analogue of "pieces plus remainder equal the fine-tuned weights"; r is E12's.
Both read the same student, so the plan reads both and scores (a) first. U1 and L11 run through the components, so the two differ only through G1 and the gain.

## Arms, in run order (one job, stop on any miss)

1. **C0, W₀ and least-squares, alone.** Check sha256s. W₀ cells T0, U0, L00 against `stage0a.json` (0.02 nats, E46's tolerance). Do-nothing and
   least-squares on all eight. Stop if r misses E12 by over 0.03 (E46's), or if do-nothing's e_gate + r is not 0 within 1e-6.
2. **C1, teacher = W₀ from a moved start, on rand32 and ident32 windows.** This is Amendment 8's control, and it belongs: a student moves with
   nothing to learn, and it is the only way to see how much e_gate training alone produces. The start is the saved `amend8/n1/0.pt`. The
   zero-start form does not belong: its gradient is exactly 0 (`teacher.py`), so it reads W₀. Shares are unusable here, as the gain is near 0,
   and Amendment 8's random move alone tripled P(target), about 1 nat, which can pass `MIN_GAIN`. So it reads terms in nats. **Stop line:
   any of the r or e_gate term above 1.5 nats** (0.11 × the smallest teacher gain, 13.2 nats; 0.11 is a third of E12's smallest e_gate gap, 0.33).
3. **Students.** 8 teachers × E42's arm A (Adam 1e-5, batch 32, clip 1, batch-order seed 0, 1,000 steps). First student stops the job if shares miss
   sum 1 within 1e-6 or its gain is under half its teacher's. Offsets saved per state, skipped on rerun.
4. **Noise floor.** ident32 s0 and rand32 s0 at batch-order seed 1.

Dropped: Pile harness (130 s per state, 26 min; saved offsets allow it later), Pile-text arm, step-1000 teachers on the b tables, a null teacher.

## The numbers

| Number | W₀ | Do-nothing | Least-squares (control) | C1, in nats | Student, expected |
|---|---|---|---|---|---|
| r | NaN, 0/0; term 0 | as least-squares, exactly | copy 0.76-0.80, map 0.27-0.29 | term ≤ 1.5 | copy 0.55-0.85, map 0.10-0.30 |
| e_gate, state (a) | NaN | = −r exactly | copy 0.03-0.11, map 0.45-0.53 | term ≤ 1.5 | copy 0.10-0.40, map 0.60-0.85 |
| e_gate, state (b); r is E12's by code | NaN | not read | not read | not read | within 0.1 of (a), a guess |
| 1 − r − e_gate | NaN | 1 | 0.13-0.27 | not a share | ≤ 0.06 (E42 0.03-0.05) |
| Gain / teacher's gain | 0 | 1 | 1 | ≤ 1.5 nats | ≥ 0.7 (E42 0.94-0.97) |

## Draft predictions (not filed)

Pool for all: a state enters only if gain ≥ 1 nat (`MIN_GAIN`), gain / teacher's gain ≥ 0.5, and its shares are finite. Dropped states are listed.

| Id | Rule | Lean | Blind to | Positive control | Noise floor |
|---|---|---|---|---|---|
| P1 | e_gate of state (a), absolute, E12's rule: `gap = min(map) − max(copy)`, `within` = the larger class range; `gap > within`: "separates". Same test with classes swapped: "reversed". Otherwise "no separation". Under 3 states in a class: "unscorable". State (b) reported by the same rule | separates, medium-high | which subcomponents; one position; nearly 1 − r, so not independent of P2 | C0's least-squares separates on this card | Batch-order spread; class range is in the rule |
| P2 | Over all states, `\|r(student) − r(fine-tune)\|`, absolute. All ≤ 0.10 "inherited"; any ≥ 0.25 "moved"; else "partly". Batch-order `\|Δr\|` > 0.05: "no power". Empty pool: "unscorable" | partly, medium (E42 0.01-0.06, but step-10 teachers overshoot: E43's 5.1×) | whether the same subcomponents switch; each r over its own gain | Do-nothing's r equals least-squares' exactly | Batch-order `\|Δr\|`; 0.002 across cards |
| P3 | Per copy state, E42's filed lines on e_gate: ≥ 0.5 masked-off, ≤ 0.25 not, else mixed; end classes take negatives and values over 1. Answer: the majority class; a 2-2 tie or under 3 states "unsettled" | not masked-off, low-medium (E42 0.25 at step 100, 0.31 at 1,000) | the same as P1 | E42's rand32 and rand1 read masked-off | Batch-order spread on ident32 s0 |

**Edge cases run by hand.** Empty or one-element pool: "unscorable" (E12's `separation` returns `separates: None` on empty, and a one-element class has
range 0 and would pass too easily, so the minimum of 3 is new). Zero denominator: `route_split` gives NaN under 1e-6, the pool drops it, and `min` never sees NaN. Teacher gain
is at least 13 nats, so the ratio is safe. Ties: gap = within is "no separation" (strict); gap 0 fails both directions; P2 at 0.10 is
"inherited" and at 0.25 "moved"; P3 at 0.25 and 0.5 go to their end classes. Case no branch covers: ranges apart with gap ≤ within is "no separation", reported with both numbers.

## Route, time, cost

Cluster L40, `debug` QoS, `--time=00:40:00`: 12 students × 105 s, reads and load about 4 min, **about 25 min, $0**. Offsets 60 MB each, 720 MB on NFS. Not gated.
If not started in 10 min, RunPod: rebuild the 8 fine-tunes in-job (about 70 s by E46's log) rather than upload 900 MB at 80 KB/s; under $2.50 by E43's rate; a fresh-agent code review comes first, as that run is paid.

## Departures

- **From E12:** a different state (student's weights, so T1, G1 and the gain change); only P4's two numbers; L40, not A40; teachers are E46's rebuilds (T1 within 0.03 nats), E12's bytes being lost.
- **From E42:** teachers at the first learned step (10 or 30), not 1,000; rand32's teacher holds P only 0.55-0.59, a confound carried from E12; no harness.
- **From VPD:** U and V alone on an unmasked KL to a teacher, Γ and Δ frozen, no sparsity or masked loss. The route split is this project's measure, not the paper's. Γ sees 41-token windows where it trained on rows of 512, and it is bidirectional (review 2026-10-27).
- **From Kingma and Ba:** Adam with default betas, plus clip 1 and no warm-up, carried from E34.

## Code

New `experiments/new_vs_existing/amend1.py`, under 200 lines, calling only `src/commute` (`local_faith.route_split`, `anchors.lifted`, `teacher.teacher_net`, `teacher.teacher_loss`,
`component_training.train(init=...)`, `arms.save_offsets`). **Shared-code changes: none, so no frozen script is touched.** It imports nothing from `regating_table/run.py` or
`teacher_training/run.py`; it only reads their saved files. `experiments/new_vs_existing/run.py` stays as it ran. Predictions file: `results/predictions_amend1.json`.

## Not run

- Pile harness on any student; the Pile-text arm; step 100; more than two batch orders.
- Step-1000 teachers on ident32b and rand32b, the cheapest follow-up if P1 does not separate.
- Which subcomponents switch on in a student, overlap with E12's switched-on sets, switched-on counts.
- A null teacher on these windows; a second random start; E12's P1, P2, P3, P5 on these states; other tasks; Qwen.
