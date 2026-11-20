# Plan: E217's top-activating-neuron featurization re-read with the upper tail only (cell P2.45; was P2.42 run 2)

**In plain words.** E217 read the definition with a neuron counted as a feature when it fired above its 97th
percentile or below its 3rd. The small model's MLP is a plain GELU, whose output never goes below about -0.17, so the
"below the 3rd percentile" side is every neuron sitting on that floor: E217's saved percentiles put every neuron's p3 at
-0.1699 to -0.17. Such near-zero firings counted as features, and a tiny shift in a fine-tune flips them in or out.
This run counts only firings above the upper percentile and reports every number beside E217's.

**Restated in the Terms (user, 2026-11-20 dialog: "Yes, upper tail only").** The featurized model is still the base
with weak activations pruned into one exact remainder per MLP site (τ₀∘σ₀ = id). At site l on a token, neuron i is a
feature when its post-GELU activation (`commute.rho` line 81, `L.acts`) is above its p97 (main band); p99 and p90
beside. Everything else at the site, GELU's negative floor included, is the remainder. The percentiles are E217's,
fixed from M on 400 Pile val rows and reused (`experiments/topk_neurons/results/large/percentiles.pt`, sha256
`e70d0a45...`, its q holds 0.90, 0.97, 0.99). The rule reads F unchanged. F is read by the definition's construction
exactly as in E217: s_u per variable, K* by mean ablation, R, f, remainder share and s, |K*|, strong-set overlap.

**Departure from the source, now removed.** E217 departure 2 (Goodfire's p97/3 is two-sided for SwiGLU, which fires
both ways; the 67M's NewGELU does not) is replaced by: one-sided upper tail, because the GELU's lower tail is its floor.
The other E217 departures stand (weak part kept as a remainder, not ablated; percentiles from 204,800 tokens; percentile
rule, not top-k by magnitude; 8-token read windows).

## Design check (`/design`)
- **Goal.** `GOALS.md` part 2: a featurization below the width for §4.2.3, now one that suits the model's activation.
- **Nearest tried.** E217 (TRIED row 313): two-sided p97/3, R_read 0.9598-0.9896, overlap 0.231-0.361, lowest remainder
  s 0.832-0.942 (two filed leans reversed). This run asks whether those readings came from GELU's floor tail.
- **Patch or step.** A patch on E217 (its source departure), the cell's second and last run, the user's call.
- **Decisions relied on.** As E217: exact featurization with a mandatory remainder; K* by mean ablation; read grade
  main. No departure.
- **Adopted definition, unchanged?** Yes; descriptive, no reuse/new verdict.
- **What would change next steps.** Rival B (overlap up ≥ 0.1 and R_read up ≥ 0.01 over E217 on most fine-tunes): the
  §4.2.3 reading is replaced by this one and E217's reversals are reported as a floor-tail artefact. Rival A (within
  those margins): E217's reading stands and the cell closes.
- **Control first.** Stage 0, its own job, exit 1 on any miss: the percentile file's sha and quantiles; the upper-only
  Pile share; W0 at p97, p99, p90 and the width; the width control on all 20 items (R over every part and over E208's
  circuits equal E213 within max(10 × E217's floor 9.04e-9, 1e-8) = 9.04e-8; E217 passed it at ≤ 7.9e-8). It can fail
  on a changed percentile file, the upper-only Feat path, the loaders or the grade code.
- **Numbers and expected values** (`experiments/topk_upper/results/predictions.json`):

  | number | W0 | width control | p97 upper (lean) | E217 p97/3 |
  |---|---|---|---|---|
  | Pile feature share | — | — | 0.03 ± 10% (p99 0.01, p90 0.10) | 0.0599 |
  | changed vars, s, gates, overlap | 0, 1, ≤ 1e-3, 1 | rem s = 1 | — | same |
  | R_read | not searched | = E213 within 9.04e-8 | 0.95-0.9995, ≥ E217's, below width | 0.9598-0.9896 |
  | overlap | 1 | 1 | 0.25-0.95, above E217's | 0.231-0.361 |
  | lowest remainder s | 1 | 1 | 0.8-1, no direction | 0.832-0.942 |
  | remainder share, f, size | — | — | < 0.01, ≥ 0.99, 300-9000 | 0.0006-0.0014, ≥ 0.9993, 2826-6270 |
  | built-pair gaps | — | — | reported per seed, no lean | case 0.0112/0.0108 |

  Descriptive rules: each lean is a range checked on all 20 trained fine-tunes (P1-P8); P2 and P4 are the per-item
  differences from E217 (0 when nothing changes, which the CPU dry run reproduced by scoring E217's items as run 2's);
  p99 and p90 and the gaps are reported beside E217's with no range (P9, P10). The rivals split at 0.1 overlap and
  0.01 R; the seed spread of R_read is about 0.0006.
- **Blind to.** As E217: the overlap counts a neuron just across its threshold as lost; remainder s cannot tell many
  small turns from one large one; K* from one greedy order.
- **Early guards.** Pile share within 10% relative of 1 − q (E217 read 0.0008 relative); gate "all" ≤ 1e-3 per item.
- **Saved.** Item JSONs `experiments/topk_upper/results/items/<band>/`, per-variable s and K* in
  `results/large/items/<band>/*.npz`, `results/stage0.json`, `results/summary.json` (each item beside E217's).

## Run
Cluster `debug` (L40), account c11, $0; `cluster/topk_upper.sbatch`. Stage 0 alone (~3-5 min; E217's took 2.3 min
with the percentile pass and floor it no longer does), then stage 1 afterok as an array of 6 (p97, p99, p90 × seeds),
10 items each, resumable per item, `--time=01:45:00`, at most 4 at once. E217's measured per-item times at its bands
were 82-612 s (`experiments/topk_neurons/results/items/*/*.json` → item_seconds; mean 441 s at p97/3, 297 s at p99/1,
450 s at p90/10), so a task is 10 × 300-600 s = 50-100 min, the whole run ~2-3.5 h with two waves. Then
`cpu-run python experiments/topk_upper/run.py --stage score`.

**Changed shared code.** `commute.kstar.Feat.lo` is optional (`None` = upper tail only); two-sided behaviour unchanged.
Callers: `experiments/topk_neurons/run.py` (E217, frozen, passes both bounds) and `kstar.split`; tests/test_kstar.py
passes. Run 2 imports E217's functions (`load_experiment("topk_neurons", "run")`) and adds only the upper-tail Feat,
the Pile share, its own item writer and the side-by-side score.
