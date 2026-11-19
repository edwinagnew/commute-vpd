# Plan: the definition read at a top-activating-neuron featurization (E217, cell P2.42 run 1)

**In plain words.** The paper's §4.2.3 says the definition can be read with the base model's strongest-firing neurons
as its features, but no reading below the full width has been made. This run makes it: on the 20 trained 67M
fine-tunes E213 read, each MLP neuron counts as a feature on a token only when it fires strongly by the base model's
own yardstick, and everything weaker goes into one remainder per layer. It reports how much of the circuit each
fine-tune needs still computes what the base computed (R), beside the plain overlap of strong neurons.

**Restated in the Terms (user, 2026-11-19, settled).** No sparsification training: the featurized model is the base
with weak activations pruned. Features at MLP site l on a token are the neurons whose activation is above that neuron's
97th percentile or below its 3rd, percentiles from M on Pile text; the remainder is e = h − strong(h), so τ₀∘σ₀ = id
and the transformation is exact. The rule is fixed from M and applied to F's activations. F is read by the
definition's construction: s_u per variable, K* by mean ablation, R = mean s_u over K*, the changed share f and the
remainder's share of K* (definition-lead's `commute.kstar`, E214), with p99/1 and p90/10 beside.

**Source rule, quoted** (Goodfire, "Neurons as an interpretable basis"): "p97/3 refers to both positive and negative
extremes, p3 ∪ p97"; "we compute neuron activation strength percentile bins and apply mean ablation to everything
between the positive and negative extremes". Departures: (1) the weak activations are not ablated but kept exactly as
the remainder (the user's design); (2) the 67M's NewGELU is not bipolar like SwiGLU, so its p3 side is the most
negative GELU outputs, which are small in size; the rule is applied as written; (3) percentiles from 204,800 Pile
tokens (400 val rows), not 10B; (4) the paper's §4.2.3 text says top-k by magnitude; this run uses the user's
percentile rule; (5) the read windows are 8 tokens long, so the share of activations that are features there can
differ from the Pile's; it is reported at W0.

## Design check (`/design`)
- **Goal.** `GOALS.md` part 2: the definition read at a featurization other than the width, so §4.2.3 has a reading.
- **Nearest tried.** E213 (width: R ≥ 0.99 on all 20, saturates); E190 (top-K masking, a baseline, not a reading;
  masking cost 0.89-5.76 nats, which is why the weak part is a remainder here, not dropped). New: the first reading
  below the width.
- **Patch or step.** A step: a named open cell (P2.42), user's design.
- **Decisions relied on.** "Exact featurization with a mandatory remainder", "Gain and graft declarations for R",
  K* by mean ablation, read grade main (all 2026-11-18/19). No departure.
- **Adopted definition, unchanged?** Yes; descriptive, no reuse/new verdict.
- **What would change next steps.** R at p97/3 falling below 0.99 or opening a counted gap ≥ 0.01 gives §4.2.3 a level
  that does not saturate; holding at ≥ 0.99 closes neuron-level featurizations (card stop).
- **Control first.** Stage 0, its own job: W0 (F = M: no change, every s = 1, exact gates, overlap 1) at every band,
  then the width control: R over every part and over E208's circuits must equal E213 per item within
  max(10 × floor, 1e-8), the floor measured in the same job by rerunning E208's own `rho.profile` on copy24_s0 against
  its saved scores. A miss exits 1 and stage 1 (afterok) never starts. It can fail on the loaders, the windows, the
  grade code or the Feat path at the width.
- **Numbers and expected values.** In `results/predictions.json` → "expected" and P1-P6: W0 as above; width R_read ≥
  0.99; p97/3 R_read 0.95-0.999 and below the width's; site remainders all in K* (share < 0.01); remainder s ≥ 0.99;
  overlap 0.7-0.99 (1 at W0 and width); f ≥ 0.99; reuse-minus-twin gaps reported per seed, no lean.
- **Early guards.** Pile feature share within 0.02 of its band (broken percentiles); gate "all" ≤ 1e-3 on every item
  (exactness); each logged when it passes.
- **Saved for audit.** `results/large/percentiles.pt` (sha in stage0.json), per item and band the s of every variable
  and K* indices (`results/large/items/<band>/<item>.npz`), item JSONs in `results/items/`.

## Run
Cluster `debug`, account c11, $0. Stage 0 (~10-15 min), then stage 1 as an array of 8 (band × seed, 10 items each,
35-75 min each, at most 4 at once; resumable per item). Commands in `cluster/topk_neurons.sbatch`; then
`cpu-run python experiments/topk_neurons/run.py --stage score` writes `results/summary.json`.

**Changed shared code.** `commute.kstar` gained an optional `feat` (Feat) on `prepare`, `forward`, `grades`, `rem` on
`level` and `summary`; defaults unchanged (tests/test_kstar.py passes). Frozen callers: `experiments/kstar_levels/run.py`
(E214) only.

**Run 2 (not planned here).** Whatever run 1 leaves open; the cell's second run.
