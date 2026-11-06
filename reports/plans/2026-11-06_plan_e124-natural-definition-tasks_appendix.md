# E124 appendix: apparatus

*Companion to `reports/plans/2026-11-06_plan_e124-natural-definition-tasks.md`. Paths are from the repo root.*

## §0 What is reused unchanged, and what is new

Reused, called, not copied: `commute.case_label` (`value_probes`, `control_y`, `twin_y`, `split_pairs`, `pick_labels`,
`first_passing_block`, `fit_threshold`, `apply_threshold`, `removed_share`, `verdict`, `do_nothing_ok`,
`reverse_steps`); `commute.finetune.finetune_arm_a` with `ArmAConfig(o_token=data.LAST, ci_len=...)`; `commute.removal`
(`unit_basis`, `random_basis`); `commute.revival_kv` for pruning; the route split chain in §7. E112's run-level helpers
(`label_diff`, `margin`, `removal_edit`, `state_reads`, `prune`, `reverse` in `experiments/case_label/run.py`) assume
the content word sits at `HALF - 1`; the code writer moves what is needed into `src/commute/` with the content position
passed in, or calls them through `commute.jobs.load_experiment`, and does not copy them. E112's results do not change
(its run is kept as it ran).

New: one module `src/commute/natural_tasks.py` (the two window builders, the probe-position builders, the name-list
reader) and `experiments/natural_tasks/run.py` (stages), `plot.py`, `cluster/natural_tasks.sbatch`. No `--smoke` path.

## §1 Agreement windows

- Rows: `keydep.vocab` rows (0-32767) for task windows, as E112; probe rows from 32768 on (E112's `probe_rows`).
- Pool: every pair (" w", " ws") or (" w", " wes") with both members single tokens (the tokenizer's decode, as
  `case_pairs` builds case pairs). Rank pairs by the count of the pattern below with either member as N; keep the top
  400; split by pair 300 / 100 (`split_pairs`, seed 0). Pairs where both members are verbs more often than nouns are
  not filtered: the pattern's " the N PREP" context forces a noun reading in most windows, and the drop rule below
  removes most of the rest.
- Pattern (token level): " the"/" The", N, PREP ∈ {" of", " in", " on", " for", " with", " from", " at", " to",
  " near", " behind", " about", " under"}, optional " the"/" a"/" an", N2 (any pool member), next token " is" or " are".
  The window is the 21 tokens ending at N2 (`ci_len` 21, prediction position `HALF` = 20), with the target in the last
  column (`copying.with_target`, `data.LAST`).
- Attractor: N2's number differs from N's. Sampling fills train, fit and held-out sets half with and half without
  attractors; the realised share is printed.
- Drop rule: the text's verb disagrees with N's number. The dropped share is printed per kind (attractor or not); it
  mixes natural agreement errors with subjects the pattern misread.
- Sizes as E112: 9,600 train, 1,024 fit, 1,600 held-out windows; train from rows with index % 5 ≠ 0, fit and held-out
  from the rest. Held-out windows use held-out pairs as N only.
- Probe positions: in probe rows, every match of the pattern up to N2 regardless of the next token, train pairs as N
  only; label = N's number; cap 16 per pair (E112's `probe_cap`). Control labels: `control_y` over all 400 pairs (a
  seeded bit per pair). Scored on held-out task windows at the prediction position, as E112.
- Answer words: " is" (A, singular) and " are" (B). Label margin = logit A − logit B, signed to the window's label.

## §2 Pronoun windows

- Name list: the US Social Security Administration's national baby-name counts (`names.zip`, files `yobYYYY.txt`,
  public domain; URL to be verified by the code writer, not checked here), summed over 1950-2000. Gender = the side with
  ≥ 0.95 of a name's count; names under the share are dropped. Fetched once inside the stage 0 job into
  `artifacts/sets/ssa_names/` (gitignored, untrusted; read with a parser, never executed). If the fetch fails, stage 0
  stops and says so; nothing is invented.
- Pool: names whose " Name" form is a single token; top 400 by Pile count in task rows (fewer if fewer exist, printed);
  split 3:1 within gender, seed 0. A name is a one-member "pair" for `control_y` and `split_pairs`.
- Windows: positions whose next token is " he" or " she" with exactly one pool name 3-15 tokens back and no other
  pool name between; 21-token windows ending at that position, target in the last column. Drop rule: pronoun disagrees
  with the name's gender; dropped share printed.
- Probe positions: in probe rows, positions 3-15 tokens after a train-pool name with no other pool name between,
  regardless of the next token; label = gender; cap 16 per name.
- Answer words: " he" (A) and " she" (B).

## §3 Stages

**Stage 0 (alone, W0 only).** Builds both tasks; prints counts, attractor and drop shares; W0's train and held-out
accuracy per member; W0's mean full-vocabulary log-odds T0 of each member's target on the fit windows; P0 per block for
each task (`value_probes`, ridge λ 0.001); `l*` per task (`first_passing_block`); W0's next-token loss change and label
margin change under the probe direction and one random direction at `l*`, with both directions' share of the residual
variance at `l*` printed (the matched-variance report the predictions rule asks for; no rule reads it). Then the
route-split controls of §7: base cells per member, the zero update's cells (r cell difference exactly 0), and E54
elicit seed 0's r through the `data.LAST` path within 0.03 of 0.7726 (E60's tolerance; E62 did this read at
`experiments/induction_cue/run.py:227-243`).

**Stage 1.** Per task with `l*` and headroom passed: natural, new-word and twin, seeds 0 and 1. Each scored state's
update saved in bf16 to `results/large/` with its sha256 as soon as it is chosen. Logged per checkpoint: train and
held-out accuracy, gain T1 − T0 in full-vocabulary log-odds. Resubmit with exit 3 on the time guard, as E112.

**Stage 2.** First the do-nothing check: random direction share ≤ 0.10 on every state (`do_nothing_ok`), stop if missed.
Then per state P1, P2, P3 (E112's `state_reads` with the content position given) and the verdict; pruning (K = 1, 4,
16, 64 against 5 random) and reverse fine-tuning (cap 300 steps) beside. Then R1-R2 per state (§7) on its fit windows;
the twin also on its flipped windows (twin label ≠ the text's word) and the natural member also on attractor and
non-attractor fit windows separately (beside, no rule). Every scored number goes to the log and `results/stage<N>.json`.

## §4 Stage 0 stops (each prints its reading even when it passes)

- Fewer than 9,600 / 1,024 / 1,600 windows for a task, or a held-out pair or name with no window: that task stops.
- Natural member's W0 train accuracy outside 0.65-0.85: that task stops (headroom).
- Possession fails before the last block: that task reads "unreadable" and stops (E112's rule).
- Zero update's r cells not exactly equal, or the elicit re-read off by more than 0.03: the whole job stops (apparatus).
- Name-list fetch fails: the pronoun task stops.

## §5 Saved

`results/large/task_<task>.pt` (pools, splits, windows, labels per member), `probes_<task>.pt` (probe and control
weights per block), `dW_<task>_<member>_s<seed>.pt` (bf16) with sha256 in `stage1.json`. Small JSON beside, committed:
per-window label margins at W0 and each state, per-window masked log-odds of the four route cells, the dropped-window
counts.

## §6 Edge-case pass for each filed rule

| Rule | Edge case | Reading |
|---|---|---|
| P0 | no block passes before the last | "unreadable" (E112) |
| P0 | passes only at the last block | does not count (E112) |
| P1, P2 | state did not learn by step 1000 | read as filed; twin or new-word near 0.5 gives "not through C" |
| P2 | threshold ties on train windows | `fit_threshold`'s order (E112, tested) |
| P3 | gain ≤ 0.1 | "no gain", verdict "not through C" (E112) |
| P3 | zero update | gain 0, "no gain": the do-nothing state cannot read "reuses C" |
| P1 at W0 | natural member already ≥ 0.90 on held-out | allowed by Def 2; the verdict then rests on P3; reported |
| R1 | gain under 1 log-odds (`local_faith.MIN_GAIN`) | r nan: state "no call", task replication "no call" |
| R2 | U1 − U0 under 0.1 (`ratio_floor`, E53 Amendment 1b) | ratio nan, reported |
| R1 separation | one seed exactly 0.5 | natural passes at ≥ 0.5; new-word fails "< 0.5" |
| R1 separation | one member's pool empty (task stopped) | "no call" for that task |
| R1 twin flipped | fewer than 100 flipped fit windows | reported with its count, no rule reads it |
| Joint | definition "unreadable" | row reads "unreadable / r", no comparison claimed |

## §7 The re-gating reading on the new states

- Decomposition: the released VPD of the 67M, `s-55ea3f9b/model_400000.pth` under `env.VPD_DIR`
  (`artifacts/vpd/`, present), target `t-9d2b8f02`; loaded by `vpd.load_component_model`. Alive set:
  `experiments/transport_test/results/large/alive.pt` (present; E62's default, `experiments/induction_cue/run.py:351`).
- Context: `keydep.make_ctx(device, alive_path, P, large)` (`src/commute/keydep.py:48`).
- W0 cells per member: `local_faith.route_cells(model, toks, HALF, data.LAST, keydep.BATCH, ci_len)`
  (`src/commute/local_faith.py:199`).
- r per state: `keydep.route_windows(ctx, toks, HALF, data.LAST, dW, base, ci_len)` (`src/commute/keydep.py:66`), which
  applies the least-squares split of dW and calls `local_faith.route_split` (`:256`); r = (L01 − L00) / (T1 − T0).
- Stand-in ratio: `pcfg_vpd.shares(cells, ratio_floor=0.1)` (`src/commute/pcfg_vpd.py:79`) on the same cells,
  (L11 − L00) / (U1 − U0); its r agrees with route_windows' r when the gain ≥ 1.
- Unit: log-odds of the target token over the whole vocabulary at the prediction position, not the A-against-B margin
  P3 uses; both are printed.

## §8 Departures

From Jain et al. 2024, as E112's four (probe form, row 87's held-out carrying, direction removal, E62's recipe), plus:
the sub-domain X_C is a text pattern (a subject and a prepositional phrase before the verb; a name before a pronoun)
rather than a marker token; subjects are found by a token pattern, not a parse (Linzen et al. 2016 used a dependency
parse). From E112: natural answer words for the natural member; the content position varies per window. None changes
a verdict rule.
