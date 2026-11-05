# E90 appendix: apparatus, sources, sizes, saved files

*Companion to `reports/plans/2026-11-05_plan_e90-new-vs-reuse-definition.md`. Grepped paths, 2026-11-05, HEAD
5d7bc91f.*

## Readings reused unchanged

| Piece | Code | Source of its settings |
|---|---|---|
| removal of a basis from the residual, h − U Uᵀ h | `src/commute/removal.py:18` `remove` | E70 |
| value-form basis: per-value mean residuals, centred, orthonormal, rank \|Y_C\| − 1 | `removal.py:50` `centred_basis` | E70's softmax form; the scalar case is A1's `pattern_basis` (`removal.py:153`) |
| random bases of the same rank, 20 per block, seeded | `removal.py:57` `random_basis`; band `removal.py:70` `band_of`, `max_band` | E70, 95th percentile of \|drop\| |
| share A_lo = (state drop − W₀ drop) / state log-odds gain, nan under floor | `src/commute/readout.py:190` `shares` | E64, amended pre-run 2026-11-02 (review H1, M1); E70 scored it |
| g's table form, parity split, unseen = wrong, floor = fit half's mode | `removal.py:132` `lookup_accuracy` | E70's "g ∘ f" lookup |
| readout d → \|Y_C\| per block, Jain's schedule, early checks against the model's loss | `readout.py:394` `train_readouts64`, `early_miss64` | E64 (Jain et al. 2024 §3, 4K AdamW steps, lr 1e-3 decayed; `src/commute/probe.py:5`) |
| relative possession: 3/4 of W₀'s own possessed set | `readout.py:458` `relative_possession` | E64; decision report item 5 |
| mechanism form: a head's output replaced by its Pile mean | `src/commute/copying.py:160` `ablated`, means `head_means` (`:135`) | E62 (Olsson et al.'s prefix score picked the head; mean ablation as E62) |
| hand write X = U pinv(A) into `h.1.mlp.down_proj`, α matched to the fine-tune's gain | `src/commute/perturb.py:61` `key_write`, `:33` `match`, `WRITE_KEY` `:20`; `experiments/regating_table/run.py:375` `write_delta` | E46 P0.4; saved rows `ident32/write`, `ident32b/write` in `experiments/regating_table/results/stage0b.json` (α 15.2 and 12.7, relative error 0.16, cond 8.2, status ok) |
| E62's saved shares | `experiments/induction_cue/results/run.json` `.states[*].share`, `.random_median` | copy 0.563 / 0.545 / 0.471, lookup 0.000 / −0.004 / −0.003, random medians 0.011-0.052 |
| E70 A1's saved shares and bands | `experiments/pcfg_anchors/results/e70_a1.json` `.r2.rows`, `.states` | P 0.133-0.884, N 0.032-0.493 at blocks 1-5; block 6 N 0.174 > P 0.070, band 0.009 |

## The capabilities, as filed

| C | f_C, a text rule | Y_C | X_C (natural, Pile) | Where read | Admissible for |
|---|---|---|---|---|---|
| token identity | the current token | the task's 32 keys | natural occurrences of the keys in Pile rows 32768-61439 (E64's rows, disjoint from the task windows) | block 0 by construction; removal at b* = 1 | copy tasks (A2); not the ciphers (A1) |
| W₀'s next-word guess | W₀'s argmax next token, restricted to the rank-1 table's 18 values | 18 | positions where W₀'s argmax is one of the 18, capped per value as E64 (2,000) | earliest block in {1, 2} with ≥ 3/4 of W₀'s set (E64: blocks 0-3) | rank-1 (A4); not elicit (A6) |
| in-context copying | the token that followed the earlier occurrence of the current token | E62's B pool | repeated random sequences (E62's prefix-score inputs, `copying.repeated_random`) | head (2,4), block 2 | E62 copy (A8); not lookup (A7) |
| embedding class (second pass) | sign of the key embedding's top principal component pair, `src/commute/open_pair.py` class map | 4 | the keys' natural occurrences | block 0 by construction | class map; flagged: model-defined, not a text rule |

"W₀'s reading of C" in the admissibility rule is the readout's value on the task's inputs (stage 2); stage 0 uses
f_C itself as the upper bound. The departure from Jain's single domain stands as the decision report's item 6: C is
possessed and its basis fitted on natural inputs, and removed on the task's marker windows at the answer position.

## The verdict rule as code will see it

Per (task, C, block b*): `teeth = W0_drop > band_W0`; `s = A_lo(state)`, `c = A_lo(control)`; `above(x) = x >
band_states` (strict; nan is not above and is counted). Verdict: not teeth → "unreadable: no teeth"; isnan(s) →
"unreadable: no gain"; isnan(c) → "no call"; above(s) and not above(c) → "reuses C"; not above(s) and above(c) →
"not through C"; both above → "unreadable: not specific"; neither → "unreadable: does not reach the states". Seeds
are read one by one; the task's verdict is the one every seed gives, else "mixed", listed per seed. Tests in
`tests/test_removal.py`'s style run the rule on the edge cases A9, R1, R6 and on E70's and E62's saved numbers (R4,
R5) with no model.

The control for the value form is the hand write of the same table at matched gain (`match` with `ALPHA_MAX`,
`write_rel_tol` as `regating_table`'s filed threshold); its α and `rel_err_mean` are reported, and a hand write
whose match fails ("unreachable", "no_match") makes the row "no call". The control for the mechanism form is the
other task on the same windows (E62's lookup for its copy map).

## States read, by stage

| Stage | States (all `~/commute-data/results-large/`, fp32 dW dicts, 113 MB each) | Windows |
|---|---|---|
| 1, R2 | `regating_table/ident32_s0.pt`, `regating_table/rand32_s0.pt` (same 32 keys, offset 100, `experiments/new_vs_existing/run.py:146-147`), ident32 hand write rebuilt | `regating_table/windows_ident32.pt`, the rand32 windows |
| 1, R3 | `open_pair/amend1/rank1_s0.pt`, `open_pair/cipher_s0.pt` (24 keys), rank-1 hand write rebuilt on E54's windows | E54 windows, 403 train / 97 held |
| 2, row 1 | `induction_cue/{copy,lookup}_s{0,1,2}.pt`, `heads.pt`, `windows.pt` | E62's read windows |
| 2, rows 2-4 | rank-1, cipher, elicit s0-2 (24 keys); ident32, rand32 s0-1; `open_pair/amend1/copy_s{0,1,2}.pt` | as above |

Every state's sha256 is checked against the saved record before it is read (E59's stage-0 rule); a mismatch stops.

## Job sizes, MATS `debug`, one L40, `--mem=24G` (`cluster/readout64.sbatch:27`), `--time=00:30:00`

| Stage | Work | Reference timing | Estimate |
|---|---|---|---|
| 0 | labels and tables on CPU, `cpu-run`, no model | — | minutes |
| 1 (i) | one readout over 18 values at 4 blocks on cached residuals | E64 job 48611, twelve states and the readout in 1 min (`TRIED.md:135`) | 1 min |
| 1 (ii) | W₀ under 3 bases × 3 blocks × 21 removals on ≤ 2,000 X_C positions | E70 A1 control stage 45 s (`cluster/logs/commute-e70-a1-50499.out`) | 1 min |
| 1 (iii) | 2 hand writes (seconds each; `regating_table` stage 0b built four) + 4 states × 2 bases × 3 blocks × 21 removals on 97 windows | E70 A1 states stage 859 s for 20 states × 6 blocks × 21 (3M model) | 3 min |
| 2 | ~15 states as above + E62's six under the head (E62 job 48541 ran six fine-tunes and the ablations in 3.5 min, `TRIED.md:136`) | | 10 min, resume from JSON at a 22-min deadline as `e70_a1.py` |
| second pass | class map rebuild 10 fine-tunes 11 min (job 46673, `TRIED.md:132`); lowercase pair 107 s + 217 s (`TRIED.md:112`); emoticon and a ladder rung seconds (`TRIED.md:46`) | | one job each, ≤ 20 min |

## Saved for audit

`experiments/reuse_definition/results/`: `admissibility.json` (stage 0, every pair, g's held-out accuracy and floor,
the held-out count), `edge.json` (stage 1: every drop, band, share and variance share per block, the hand writes'
α and fit), `table.json` (stage 2, per seed), `stop.json` on a stop. `results/large/`: the readout weights (768 ×
18 × 4 blocks, fp32), the removal bases (768 × rank per C and block), the two hand-write X (768 × 3072 fp32, 9 MB
each), each state's per-window log-odds before and after removal (97 × 21 per block, fp16). Each stage writes as
it ends; every scored number goes to the log with its band.

## Departures from the literature

- Jain et al. 2024 Def 1 reads C at one domain; here C is possessed on natural Pile inputs and removed on marker
  windows W₀ never saw (decision report item 6).
- Jain's Def 2 puts no bound on g; the g class here is {constant, identity, named text rule, table} scored on a
  held-out half (row 87's departure, kept as the admissibility rule of row 116).
- Jain's causal evidence is pruning and reverse fine-tuning; here the label is removal (row 115), and those are
  reported beside it from E55 C3 where they exist.
- Jain's possession is "for all x"; here 3/4 of W₀'s own output accuracy on held-out rows (E64's relative form).
