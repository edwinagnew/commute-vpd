# E116 appendix: apparatus, sources and edge cases

*Companion to `reports/plans/2026-11-06_plan_e116-definition-table.md`. Paths are repo-relative; state files live under
`~/commute-data/results-large/` (each experiment's `results/large/` is a symlink there). Numbers are quoted from the
named file and were not recomputed.*

## 1. States read (26) and their windows

| Row | States (`results-large/...`) | Seeds | Task windows (trigger at `half`) | Held-out-value windows | Twin at the same trigger |
|---|---|---|---|---|---|
| ident32 | `regating_table/ident32_s{0,1}.pt` | 0, 1 | `regating_table/windows_ident32.pt` | none (§2) | rand32 |
| copy-24 | `open_pair/amend1/copy_s{0,1,2}.pt` | 0-2 | `open_pair/amend1/windows_copy_trained.pt` | `windows_copy_new.pt` (8 keys) | cipher-24, rank-1, elicit (24 trained keys, marker 18214) |
| E62 copy | `induction_cue/copy_s{0,1,2}.pt` | 0-2 | `induction_cue/windows.pt` `train` (800) | `read` (200; Bs never trained, `src/commute/copying.py:311-323`) | lookup |
| E62 lookup | `induction_cue/lookup_s{0,1,2}.pt` | 0-2 | same | none (§2) | copy |
| rand32 | `regating_table/rand32_s{0,1}.pt` | 0, 1 | `windows_rand32.pt` | none | ident32 |
| text cipher / broad | `performed_anchor/dW_{cipher,broad_cipher}_s{0,1}_step300.pt` (bf16) | 0, 1 | E53's cased train rows 0-2399 ciphered (`experiments/performed_anchor/run.py:77`) | E53's 64 held rows (`:78`); no held-out *value* (§2) | none |
| cipher-24 | `open_pair/cipher_s{0,1,2}.pt` | 0-2 | `open_pair/windows_cipher_trained.pt` | `windows_cipher_new.pt` | rank-1, elicit |
| elicit | `open_pair/elicit_s{0,1,2}.pt` | 0-2 | `windows_elicit_trained.pt` | `windows_elicit_new.pt` | cipher-24 |
| rank-1 | `open_pair/amend1/rank1_s{0,1,2}.pt` | 0-2 | `amend1/windows_rank1_trained.pt` | `windows_rank1_new.pt` | cipher-24 |
| lowercase, emoticon, rand1 | not loaded: "does not apply" (`decisions.md:164`; the user's reset, `reports/decisions/2026-11-06_decision_e99-reset-to-jain.md` "no content function") | | | | |

Files and sha256 records: `experiments/locator_table/sites.py:63` `TASKS` and its `Files` class (E90's, E53's and E38's
saved records); a differing sha256 raises. Seeds are the ones E100 read (`experiments/fed_or_written/results/table.json`),
so the locator column and the verdict sit on the same states.

## 2. Why P1 (held-out carrying) cannot be applied on some rows

- ident32, rand32: `kv_task`/E12 windows have a `heldout` split by *row*, and `src/commute/kv_windows.py:29`
  (`task_context`) asserts only the marker; the held-out windows hold the same 32 keys (inventory §4). No key split
  exists and none is built here.
- E62 lookup: every read window's A is one of the 32 keys; a lookup defines no answer for a key it never saw.
- Text ciphers: the table (50 or 500 words, `performed_anchor/run.py:22-23`) says nothing about a word outside it; an
  unseen word has no target. This is the brief's own example of "unreadable for that part".
- These are reported "n/a: <reason>" and never given a target. E62 copy's read windows are the one saved set with
  unseen values; the 24-key maps have E58's 8 held-out keys (`src/commute/heldout_keys.py:31` `split_task`).

## 3. The locator column (copied, no computation)

E100 own share = that part's δW alone set back to W₀, over the gain; sum = the 28 single reverts added
(`experiments/fed_or_written/results/table.json` `.table[]` keys `task, seed, part, own, reverts_sum`; seeds 3-7 in
`amend1.json` `.states`). E97 located = removal of the part at W₀'s mean takes the gain and spares the control
(`experiments/locator_table/results/table.json` `.table[]` keys `part, share, control_share, located, reason`; only
T5 copy, T5 lookup, T7 lowercase, T8 cipher were read). Where E97 did not run, the column says so. Read with `jq` on 2026-11-06 (E97: lowercase
(1,1)+(1,2) share 0.798/0.853, control 0.071/0.069; T8 cipher "none at head/MLP grain", control not spared).

| Row | E100 top part, own share per seed | Sum of the 28 single reverts per seed |
|---|---|---|
| ident32 | mlp3 0.047/0.045 | 0.238/0.219 |
| copy-24 | mlp3 0.079/0.065/0.074 | 0.395/0.304/0.351 (seeds 3-7: 0.31-0.38) |
| E62 copy | (2,4) 0.008/0.007/0.005 | 0.263/0.297/0.222 |
| E62 lookup | mlp3 0.136/0.147/0.139 | 0.68/0.62/0.67 |
| lowercase | mlp3 0.156/0.158 | 0.557/0.538 |
| rand32 | mlp3 0.139/0.135 | 0.656/0.628 |
| text cipher | mlp3 0.479/0.494 | 0.882/0.894 |
| cipher-24 | mlp3 0.135/0.128/0.129 | 0.70/0.72/0.68 (seeds 3-7: 0.67-0.74) |
| emoticon | mlp3 0.197/0.186 | 0.369/0.305 |
| elicit | mlp3 0.125/0.128/0.124 | 0.47/0.51/0.51 (seeds 3-7: 0.49-0.56) |
| rank-1 | mlp3 0.193/0.182/0.181 | 0.73/0.72/0.72 (seeds 3-7: 0.69-0.81) |

## 4. The readouts (P0) per C

Common form (E112's, `experiments/case_label/run.py:192` `probes`, moving to `src/commute/case_label.py`): residuals at
blocks 0-3 (`readout.RESID`, `steering.py:37`) at one position of 41-token Pile windows from rows 32768-65535
(disjoint from every task's rows); closed-form ridge per value (`removal.fit_ridge`, λ = 1e-3 × the mean diagonal);
cap 16 positions per value and type (E112 `probe_cap`), seed 4; read as argmax over the values on the task windows at
the trigger; control = the same with one seeded random value per type of the token before the position (seed 3).
l* = `case_label.first_passing_block` (accuracy ≥ 0.90, selectivity ≥ 0.30, earliest block before the last).

| C | Values (classes) | Pile positions t | Label | Types excluded from the probe | Scored on |
|---|---|---|---|---|---|
| in-context copying (E62) | the distinct Bs of the 200 read windows (≤ 200 of the 984 read Bs) | token t−1 occurs earlier in the window at s < t−1 (once), token s+1 is a value, token t−1 is not a key or the marker | token s+1 | the 32 keys at t−1 | read windows at the marker (`half`); classes with no Pile position are printed with their count |
| W₀'s guess (rank-1, elicit) | the task's value set, trained and held-out keys (`heldout_keys.pool_of`) | W₀'s top-1 at t is a value (E90's `scan_guess`, `new_vs_reuse/apparatus.py:219`, read here from W₀'s logits at t) | W₀'s top-1 at t | the 32 keys at t−1 | the held-out keys' windows at the trigger; label = W₀'s argmax among the values there |
| W₀'s guess (text ciphers, Q4) | the table's values (50 / 500) | as above on cased Pile rows | W₀'s top-1 at t | none (no keys) | E53's 64 held rows, ciphered, at positions whose target is a value |
| token identity (beside, no l*) | the task's keys | the key at t (`readout.next_token_windows`, `readout.py:321`, at t−1) | the key | none possible | the trained keys' windows: no held-out types exist |

Site: the locator's block where E97 names one before the last block (E62 copy: block 2, heads (2,3)+(2,4)); E100's
top-carry part on every other row is mlp3, the last block, which P0 never counts (E112 edge "passes at the last block
only"), so those rows use E112's l*. Both blocks are printed (Q5).

## 5. Edge cases, run on the rules before filing (`tests/test_case_label.py` gains the m-valued cases)

- A value with no Pile position: its probe has zero weight and intercept, is never the argmax, and its windows count
  wrong; if every value lacks positions, accuracy is nan and P0 fails ("unreadable", reason printed).
- No block passes, or the last block only: l* None, "unreadable" (E112's edge).
- A state that gives one answer on every fit window: the table g is constant; P2 = that answer's share of held-out
  windows; P1 = the share of held-out windows whose target is that answer.
- Zero gain (G ≤ 0.1 nats): P3 nan, "no gain", verdict "not through C"; a nan do-nothing share is no reading and does
  not stop (E112 `do_nothing_ok`).
- Ties: accuracies at a bar pass (inclusive); argmax ties go to the lowest value index; `lookup_accuracy`'s table takes
  the lowest token on a tie.
- An empty held-out set (no windows for a held-out key): P1 nan, "unreadable"; never filled from trained keys.
- The do-nothing state (δW = 0): P1 = W₀'s accuracy (on rank-1, high by construction), G = 0 → "not through C". It
  cannot pass. The do-nothing *direction* at rank 1 ≤ 0.10 on every E112 state; at rank m−1 the lean is a stop.
- A type-level C: both probes ≈ 1 on the trained keys and no held-out types: P0 is not applied; printed beside.

## 6. Stages, guards and what each logs

| Stage | Reads | Guard (logged with its reading, pass or fail) | Stop rule |
|---|---|---|---|
| 0a | E112's `task.pt` and `probes.pt` through the moved code | per-block accuracy and control against `stage0.json` | any block off by > 1e-3: exit 1 |
| 0b | per C: positions, probes, controls, l*, site | class coverage (values with ≥ 1 position), accuracy and selectivity per block | none: an "unreadable" P0 is a result |
| 1 | P1 per state, full-vocab top-1 beside | window counts per state | none |
| 2 | per C with l*: do-nothing on every state, then P2, P3, twin's P3, verdict | do-nothing share per state against 0.10 (rank m−1 and rank 1 both printed) | rank m−1 share > 0.10 on a state: that C's P2/P3 not read (E112's rule) |

Time check: E112's `stage_minutes` 25 of the job's limit; stage 2 exits 3 between Cs when under 4 min are left, and
`fleet/resubmit` continues. Every scored number is printed to the log as well as `results/stage<N>.json`.

## 7. Earlier claims, file:line

ident32 reuse / rand32 new by construction: `experiments/SAVED.md:154`, `:288`. copy-24 reuse: `SAVED.md:160`; E90
table row. E62 copy reuses, causal: `TRIED.md:136`. lowercase reuse: `SAVED.md:173`. text ciphers new: `SAVED.md:173`.
cipher-24 new, elicit open, rank-1 old by assumption: `SAVED.md:160`, `TRIED.md:126`. emoticon "an existing behaviour":
E90 analysis, table row T1. rand1 new: `TRIED.md:30`.
