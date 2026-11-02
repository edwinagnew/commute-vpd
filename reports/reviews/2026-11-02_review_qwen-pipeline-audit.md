# What else is hiding in the Qwen3-1.7B pipeline?
*Review, 2026-11-02. Independent audit of code, result files and reports; no GPU, no training, no model loaded. Scripts and outputs: `~/commute-data/scratch/qwen-decomp-lead/pipeline-audit/`.*

We looked for errors of the same shape as the rounded fine-tune: two versions of one thing that were never compared, numbers that disagree on file, and controls that cannot fail. The most important finding is that the one reading still said to beat noise is described as keeping the original masks, while its code moves those masks, and the size it is confounded with depends on which of two measures of size you pick. We also settled one open surprise from saved files: the switched-on sets are mostly subcomponents sitting just under the gating line, which explains why every change, random ones included, seems to overlap every reference set.

## Ranked findings
| # | finding | claim on the log it touches | how known | cheapest $0 check, one L40 |
|---|---|---|---|---|
| 1 | The direct share's "original masks" are not original: K10's masks read moved vectors, and those are rounded to bf16 only in the mask path | "the one reading that beats noise"; E61 Part 2's filed line; E44 reading B's kept shares | code verified; rounding measured on 5 sites; size of effect on the share inferred | K10 on 16 rows with masks from the original V beside E57's: about 20 min per state (gold, E40, anchor 2, Gaussian; the split is 15-19 of it) |
| 2 | "Size" has two axes that disagree 2.5-fold across kinds of change; the "rises with size" reading holds on only one | "maths fine-tunes are the largest"; D3; E61's matched-size line | verified from files | Both KL directions, unmasked, every state on the 128 rows of 256: 15 min, no Γ, no split |
| 3 | Switched-on sets are threshold-crossers; overlap's filed null cannot fail | "three to four times chance"; anchor 2's 6.4 times; the overlap's seed-pair control; Stage D's decisive rows | **settled from files, $0, done** | Done: `near_threshold.txt`. Refile the null before Stage D |
| 4 | Γ in bf16 was never compared with fp32 Γ at a changed model, though the port report set that as a precondition; every JAX comparison is at W0 | K01, K11, S at any state; matters most at E61's small states | absence verified by grep; effect unknown | Floor: E40 × 1e-3 through `cells`, 128 rows, counts on: 4 min. fp32 Γ on 4 rows with 14 sites' heads: about 25 min |
| 5 | Maths rows are packed with no resets; the fit used per-document attention and position resets; K0 has no reference on the fit's data | K0 of 0.07 as "the decomposition's faithfulness" | format verified; effect unknown | K0 on E57's unpacked rows beside the packed ones, plus one row at 256 against 512: 5 min |
| 6 | The own run's accuracy gain is less separated than reported, and no-op W0 pairs differ by 0.011 | "raised GSM8K accuracy by five points" | verified from answer files; cause of the gain unknown | fp32 greedy at W0 and own_s0 step 100, all 1319: 20 min |
| 7 | Placement mixes rows: maths Σ with on-sets from other rows | Stage D's placement row, anchor comparison | code verified; effect unknown | Fix in Stage D's code; saves piece energies for each state (none are on disk but E40's): adds under 1 min |

**Re-gating near 1, invented facts below random, and K11 above K0 are in "Numbers" below**: the first has an algebraic reason, the other two have no error I could show.

## Evidence

### 1. K10's masks move
`lab.py:353` applies the split to `comps`, then `lab.py:359` calls `importances(model, t, gamma, comps)` in the W0 arm. `readout` (`qwen_vpd.py:249`) reads x·V′, so K10's g0 is not K00's g0. The filing knows this (`qwen_reading/results/predictions.json` definitions.cells: "V' in the readouts"; `decisions.md:86`). The prose does not: "Γ's masks are kept as at the original model" (`2026-10-30_analysis_e57-reading-the-update.md:18,34`), and the request as restated for the user, "with the original masks kept" (`2026-11-02_plan_e61-direct-share-size.md:9`). The user reads the prose.

No run read K10 with true g0. Every check sits on the side V′ does not touch: `C_g0` (`run.py:284`) tests the K00 pass only, and the K10 pass collects no counts (`run.py:296`). Do-nothing moves no column, so its direct share of 0 is automatic.

Why it may matter (inferred, no number here reads it):
- Readouts alone are strong. On E40's first 16 rows, fine-tuned readouts with original taps moved K by 0.31 nats (`2026-10-29_analysis_e44-amend1-layer-cut.md`, 1.4957 to 1.1845). E40's whole direct numerator, K00 − K10, is 0.080 nats on 128 rows (`C_e40.json`). Those readouts read x_FT·V, not x₀·V′, so this is an analogy, not a measure.
- By hand, moving a read vector changes a subcomponent's output by (x·dV) g U directly and by (x·V) s (x·dV) U through its mask, with s the readout's slope. The ratio is (x·V) s / g. Median s is 0.16-3.1 on 5 sites, g of the moved energy averages 0.24, so the mask path is not obviously smaller.

**The same failure shape inside it.** Γ's readout casts V′ to bf16 (`qwen_vpd.py:249`); the masked forward keeps V′ in fp32. Measured on 5 sites from saved δW, `pieces.txt`, as cos(bf16(V+dV) − bf16(V), dV) and the norm ratio:

| δW | entries changed | cos | norm ratio | dV / V |
|---|---|---|---|---|
| E40 step 10 | 0.21-0.50 | 0.38-0.71 | 2.6-1.4 | 0.0005-0.0017 |
| gold step 100 | 0.31-0.63 | 0.51-0.84 | 1.9-1.2 | 0.0009-0.0029 |
| anchor 2 step 100 | 0.46-0.68 | 0.69-0.87 | 1.4-1.1 | 0.0018-0.0035 |

V is fp32 and off the grid (`convert.py:5`), so unlike W0 this does not hide dV: it adds noise, and the noise share falls as the change grows. A size-dependent path into a share whose next test is against size deserves one measurement before E61's Job S. A share computed with true g0 is what the prose already says it is.

### 2. Two sizes
| state | K00 − K0 (E57's size) | KL(W0 ‖ state), 128 rows of 512 | same, 16 rows of 256 | direct share |
|---|---|---|---|---|
| E40 step 10 | 1.304 | 0.521 | 0.618 | 0.063 |
| gold step 100 | 1.444 | 0.697 | | 0.092 |
| anchor 2, s0 / s1 | 0.638 / 0.628 | 0.629 / 0.607 | | 0.012 / 0.012 |
| Gaussian | 0.857 | not read | 0.532 | 0.022 |
| alive pieces | 0.625 | not read | 0.502 | 0.029 |

Sources: `C_*.json`, `C0.json`, `regen.json` `.kl`, `E_*.json`, `predictions_amend2.json`.
- K00 − K0 per unit of KL to start is 2.1-2.5 for the maths fine-tunes, about 1.0 for anchor 2. On the filed matching axis (E45 Amendment 2), anchor 2 is larger than E40, yet its direct share is a fifth of E40's. So "the direct share rises with size among the real updates" (E57 report) holds on K00 − K0 and fails on KL to start.
- On 16 rows E40 and the Gaussian sit within 1.16 on KL to start, with a direct share 2.9 times higher. The report's "E40 is 1.52 times the Gaussian" is the other axis.
- K00 is not a distance from the start. E44 Amendment 1 has KL(FT ‖ W0) 1.817 against K00 1.496 on the same rows: the masked model sits closer to the fine-tuned model than W0 does. K00 − K0 mixes the update with how masking softens outputs.

Neither axis is shown to be right. E61 fixes K00 − K0 for its band and its verdict; it should report both, and say beforehand which decides.

### 3. Switched-on sets: settled
From saved gate counts (`large/C/*`, `large/A`, `large/B2`), binned by the share of positions at W0 with g > 0.01, the line being 0.01:

| state | share of S at 0.005-0.01 | P: fold flat, within bins | T | C |
|---|---|---|---|---|
| E40 | 0.90 | 10.1, 1.39 | 4.5, 0.99 | 4.5, 1.31 |
| gold | 0.85 | 8.9, 1.30 | 4.2, 0.97 | 4.3, 1.28 |
| anchor 2, s0 / s1 | 0.95 / 0.96 | 14.3, 1.84 / 13.5, 1.72 | 5.9, 1.26 / 6.2, 1.30 | 4.2, 1.17 / 4.3, 1.18 |
| Gaussian | 0.85 | 9.8, 1.41 | 5.2, 1.18 | 4.5, 1.36 |
| alive pieces | 0.91 | 11.9, 1.60 | 5.5, 1.21 | 3.9, 1.13 |

- Almost every member of S was already just under the line. Matched on that, overlap with the thinking set is 1.0-1.3 times chance: E57's open "why three to four times" is answered, and its guess was right.
- Anchor 2's exception on the format prompt shrinks to 1.7-1.8, against 1.3-1.6 for the rest. Small, and read off 135-165 members.
- The seed pair, the overlap's control, falls from 72 to 4.8. Any two states read 2.0-3.9. It passes for any near-threshold sets.
- Churn runs both ways, unreported: on at W0 and off at the state is 6,152 for E40 against 5,730 switched on (`near_threshold.txt`). So S is one side of a jitter, not a set Γ recruits.
- My flat folds drop E57's site stratification, so they read higher than its 3-4. Bins and sites were not combined.

Stage D's decisive `prompt_overlap` and `think_overlap` use the filed null, and every change clears its q99 there, so anchor 2 would miss them whatever it learned.

### 4. Γ at a changed model
`qwen_vpd.py` casts taps to bf16 in `backbone`, so Γ sees an activation change only past bf16's spacing, about 0.4-0.8% of a value. Port checks 0-4 are all at W0, on 4 web rows and 4 prompt-only rows; no answer tokens, no δ. `2026-10-27_analysis_qwen-port.md:222` asks for "an fp32 Γ against the bf16 one, once, before Γ is read at W0 + δW". Grep finds it nowhere since. This is the prompting error's shape: the comparison passed at the one point where bf16 is exact. Large states are probably fine, since a rounding floor is small beside K00. Unknown at E61's small states, and for S, whose members sit at the line and need only a small push.

### 5. Row format
The fit read "packed documents with per-document attention masks and position resets, in both the target and Γ" (`2026-10-27_analysis_qwen-loader.md:270-271`). `qwen_sft.pack` joins sequences with neither, and rows after the first start mid-answer. `qwen_vpd.py`'s docstring still says rows hold "one document starting at position 0". E57's own per-problem rows exist but were only counted, not read for KL.

K0 itself is consistent: 0.0589 (`C.json`), 0.0589 (`G.json`, 512 tokens), 0.0696 (first 256), 0.0656 (first 16 rows); the second half is then 0.048, unverified per position. But nothing relates it to the fit. FineWeb reads 0.142 and is not the fit's data; the eval split `firestock3_qwen3_1_7b_512_eval` is not on disk, and no sharer's number is on file (`qwen-loader.md:278`). KL also falls with the model's confidence, so 0.059 against 0.142 on its own greedy answers is not evidence of better faithfulness. Inferred, not measured.

### 6. Own run's accuracy
From answer files (`w0_pairs.txt`), paired, drop rule:

| run, step 100 | against W0 | se |
|---|---|---|
| own_s0, correct answers only, rate 1e-5 | +0.0525 | 0.0102 |
| own2_s0, same data, rate 2e-5 | +0.0235 | 0.0115 |
| null_s0, no correctness filter | +0.0251 | 0.0107 |
| own_s0 minus null_s0 | +0.0253 | 0.0097 |
| own_s0 minus own2_s0 | +0.0319 | 0.0099 |
| **W0 on H100 (E40) against W0 on L40 (E45), same weights** | +0.0111 (all rule +0.0106) | about 0.007 |

- Same weights, two runs: 713 of 1319 responses differ, 84 grades flip. Baseline against E45's W0, same card, same batch: 0 differ. So a card change alone is about a fifth of the reported gain and half the null's. E40 also used batches of 128 (`qwen_sft/run.py:118`), so the card is the likely cause, inferred.
- The filed rule's `min` of 0.01 is also passed by the null and own2, so it does not read the data's filter.
- Twice the rate on the same data gives half the gain, unexplained. All are read on bf16-rounded weights with cos 0.74 to the trained state (recipe review), and no W0 or own state has been generated in fp32 over all problems.
- E61's rules use fp32 generation with absolute thresholds set against bf16 W0 readings. fp32 W0 should sit beside them.

### 7. Placement
`run.py:310` uses own-row on-sets with Σ from maths rows (`lab.py` `sigma`); E40 uses maths on-sets. So E40's excess and the others' are on different rows, and anchor 2's energy is weighted by maths inputs. Gold's excess (0.029) is below anchor 2's (0.058-0.091) while gold's direct share is six times higher, so placement does not explain direct.

## Numbers on file checked
- **Re-gating near 1.** For any K11 at or above K0, share ≥ 1 − (K01 − K0)/(K00 − K0): 0.973, 0.951, 0.947, 0.959, 0.931, 0.924 across the six states. So K01 alone puts it past the 0.8 line, whatever the split does. K11 near K01 is measured, not forced; K01 near K0 is Γ following its 29 taps (E44 Amendment 1b). I re-read the hooks: no weight enters a Δ-off masked forward, and norms, head and embedding stay at W0. No leak. "No teeth" stands, and E45's filed re-gating prediction (anchor 1 ≥ 0.5) was vacuous from the start.
- **K11 above K0.** K11 is within 0.005 of K01 in every state, so this is K01 above K0: about 0.03-0.07, as Amendment 1b's hybrids also show (0.066 to 0.102 on 16 rows). No error shown; cause not shown. Split gains at g1 are 0.0007-0.0048 nats.
- **Invented facts' direct share below random.** Real as a number on both seeds and on own rows (0.016, 0.014). Not an off-domain artefact, as the report half-suggests. Energy-weighted mean g is the same for E40's split and its null (0.235, 0.224; `pieces.txt`), so "the pieces sit on quiet subcomponents" does not separate kinds either. Unexplained; check 1 would say whether masks carry the gap.
- **Masked K0 of 0.07.** See 5.
- **Placement's control at half.** The expected value is mis-specified, not the code. It assumes the often-on half is all gated on: gated-on is 89,144 of about 247,000 (`C_e40.json`), yet the half places 0.79 of its energy there. So it did see placement, and `sees` passes narrowly (0.2105 against 0.2082) on a wrong line. The null's 0.58 sets the ceiling. Inferred from counts.
- **Own accuracy.** See 6.
- **Template from step 19.** Same in both forwards: fp32 0.953 and bf16 0.953 on 64 problems at 19, 0 at 8 in both (`qwen_recipe/results/A.json`). No fault. Strict accuracy at step 8 is 0.15 in fp32 and 0.70 in bf16: so the bf16 reading flatters that state, which supports not reading it.

## Controls that cannot fail
| control | why it passes | case it misses |
|---|---|---|
| `A_W0` | same inputs give the same numbers; zero split moves nothing | any error at a nonzero change |
| `A_do_nothing`, direct share 0 | no column moves, so K10 ≡ K00 and K11 ≡ K01 | V′ in readouts (finding 1) |
| `C_g0` | re-reads g0 of the original-V pass | K10's masks |
| `A_E40`, `C_E40`, `C0`, `C_E45`, E61's Job S control | same code reproduces its own earlier number | shared errors; only catches file swaps, drift. `C_E45` compared fp32 with fp32: the fine-tune's error passed it |
| Port checks 0-4 | all at W0, where bf16 is exact | any δ, any answer row |
| E45 `own_accuracy` min 0.01 | null and own2 pass too | whether the filter did it |
| Overlap seed pair `above_99` | any near-threshold pair clears q99 | shared content vs shared closeness to the line |
| `A_gate_counts` | real, but one row, W0 | S at a state |
| Placement `sees` | line set from a wrong expected value | can pass or fail for count reasons |
| `same_as_baseline` | job wrote null: `if base else None` (`qwen_updates/run.py:117`); file presumably absent on the job | silently no check; I compared, 0 of 1319 differ |
| E40 `P0_zero`; E61's W0 `rounded = fp32` | δ = 0; W0 on the grid | nothing beyond arithmetic |

## Cleared
- States are what they say: sha256 of E40 regen step 10, gold step 100 and anchor 2 steps 100 match `regen_SHA256SUMS` and `dW_SHA256SUMS`, re-hashed today. `dW_SHA256SUMS` is trust-on-first-read; `C_E45` and E61's early checks close it.
- E40 regenerated against original: KL 0.1064 / 0.1062, 0.5211 / 0.5204, 0.6679 / 0.6637; loss within 0.1% at steps 1-100 (`regen.json`, `F.json`). Close, different cards. Files 2, 3, 5, 20, 50 are the original run and 1, 10, 100 the regen: never treat them as one path. Original step 10 no longer exists to diff.
- Σ never enters the least-squares split (`split.least_squares_x`): only energies, the Gaussian's scale and the stored-error check. 512 against 256 in Σ cannot touch K10.
- The split is first order in practice: second over first term 0.0003-0.0068, dV/V under 0.4%, 5 sites, 3 updates. dU/U reaches 1.3% on one gate site.
- `qr.own_rows` re-implements E45's rows; `C_E45` closes it. `gate_counts` equals `importances` exactly on row 0.
- Γ has no cross-batch operation; right padding is causal and Γ never reads padded rows (read, not run). JAX against torch at W0 stands, by the port's numbers.
- KL direction in `kl_per_position`; cell order in `cells`; `applied` restores columns. E44's piece energies join to E57's on-sets: 0.6367 and 0.5835 reproduce `C_e40.json`.

## Not checked
- Qwen teacher training (`qwen_teacher.py`): rehearsal only, no scored run. `pgd_kl` on maths rows beyond the reports.
- Size of finding 1 and of finding 4 on any share: needs the GPU checks above.
- Rounding of V′ beyond 5 sites, or Σ-weighted; x-weighted readout changes: no activations on disk.
- Overlap with bins and site strata together; anchor 2's within-bin excess on the prompt set against a proper null.
- Contents of firestock3; LLaMA-Factory's setting; W&B project of the authors.
- Stage D (not run); E46-E56 and the 67M; `qwen_prefix_test`, `qwen_fresh_*`; per-position K0; whether H100 against L40 is the cause in finding 6.
