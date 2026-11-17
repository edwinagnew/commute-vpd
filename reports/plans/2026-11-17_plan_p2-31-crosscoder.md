# P2.31 (E198): a crosscoder baseline on the part 2 scorecard

Card: `reports/plans/2026-11-17_plan_p2-31-card.md` (passes the bar: 5.8 SE at n = 3, with a proxy spread from the share).
Code: `experiments/crosscoder/run.py`, `plot.py`; shared steps `src/commute/crosscoder.py`. Rules and expected values:
`experiments/crosscoder/results/predictions.json`. Slurm: `cluster/crosscoder.sbatch`. Descriptive: the sides keep the
names they were built with, and no side is called reused or new.

**The request, restated in the Terms' words.** For each saved fine-tune that the re-gating share read on the nine 67M
built pairs (E136's and E147's states, 2 to 8 seeds a side), and again for each pair's larger side scaled by E187's c,
train a BatchTopK crosscoder between the base model's and the fine-tuned model's activations at one site. Classify its
latents by relative decoder norm, and use latent scaling to drop fine-tune-only latents that are sparsity artefacts.
Then report how much of the activation change the shared latents carry, against how much the fine-tune-specific latents
carry. The result goes into the scorecard as columns beside the share, on the same colour scale and under the same
seed-range rule.

## Design check

- **Goal served.** This is part 2 of the 2026-11-09 roadmap: methods compared against how the pairs were built. It is
  one more baseline from model diffing, the literature's standard tool for "what did fine-tuning add". It fills open
  cell P2.31.
- **Nearest tried.** E155 (top-240 neuron overlap), E189 and E190 (neuron jobs, gain kept under top-K), E187 (size
  matching). All of those read neurons or weights. None reads a learned dictionary of the change, so this run learns
  something they could not. E190's lesson applies here: a reading needs a known-change check that passes, so this run
  has a planted-direction control.
- **Patch or step.** A step: a new method column. It patches nothing.
- **Decisions relied on.** The scorecard's seed-range rule (`plot_final.score`), E187's c and larger side, and the
  share's state set (`size_matched.run.pairs`). No departure from `decisions.md`.
- **Definition.** No verdict and no new rule. The reading is descriptive.
- **What would change what we do.** If the reuse side reads higher on most pairs, as built and size-matched, the
  crosscoder joins the share as a cheap reading worth carrying to Qwen. If it reads inside the seed ranges, it goes in
  the baseline column as not telling the built sides apart (the card's stop).
- **Apparatus check first.** Stage 0 runs alone, and a miss stops the job. (a) Base against itself: the "fine-tune"
  is a zero update loaded through the state path. Because the init is paired, both halves see the same data and must
  stay identical, so this control can fail only on a code fault: a side mix-up, or a path that does not reproduce W0.
  It cannot fail on training noise, and that is said in its name. (b) Planted: the fine-tune side is the base plus one
  random direction at each task window's answer position. This is a change known to be new, so it tests whether the
  apparatus can find a fine-tune-specific latent and credit the change to it.
- **Saved for audit.** Per side, `results/large/<side>.pt` holds the relative norm, the alive mask, each latent's
  carried part s_j, the latent-scaling betas and ratios, and the group masks (about 0.5 MB). Decoders are saved in fp16
  for the 6 crosscoders of stages 0 and 1 (75 MB each).

## The reading

At block 2's input (residual after 2 of 4 blocks), on the task's score-half windows (odd indices), every position
except 0, the change is D(x) = h_ft(x) − h_base(x). The crosscoder writes this change exactly as the latents' decoder
differences, f_j(x)(d_j^ft − d_j^base), plus a bias difference and an error. A latent's carried part is
s_j = Σ_x f_j(x)⟨d_j^ft − d_j^base, D(x)⟩ / Σ_x |D(x)|². The parts of the groups, the bias and the error sum to one.

The groups are taken over alive latents (at least 10 firings on the held-out mix) and set by the relative norm Δ:
shared (Δ in 0.4 to 0.6), fine-tune-only (Δ ≥ 0.9), base-only (Δ ≤ 0.1), and other. Fine-tune-only latents are then
split by latent scaling. They are fine-tune-specific when ν_r < 0.5, ν_ε < 0.2 and every β is in range. Otherwise they
are artefacts: they are dropped, reported apart and counted on neither side of the reading. The scorecard number is
**d = shared part − fine-tune-specific part**. The gap is the reuse side's mean d minus the new side's mean d.

| Number | Self control | Planted | Random update (do-nothing reference) | e54_matched seed 0 |
|---|---|---|---|---|
| fine-tune-only latents | 0 (stop if any) | 1 to 3 specific (stop if 0) | reported | reported |
| max \|Δ − 0.5\|, alive | 0 (stop over 1e-3) | — | — | — |
| FVE, both models | ~0.85 (stop under 0.5) | — | within 0.05 of self (stop) | within 0.05 of self (stop) |
| specific part | nan (no change) | 0.7 to 0.9 (stop under 0.5) | ~0 | reported |
| d | nan | ~ −0.8 | near 0, error part ≥ 0.5 | A: reuse ~+0.3, new ~0; B: within 0.1 |
| reseed \|d₀ − d₁\| (reuse state) | — | — | — | ≤ 0.05 (stop) |
| 1 − error part | — | — | — | stop if under 0.1 on both states |

Predictions (filed): P1, as built, gap > 0 on at least 7 of 9 pairs (the user's expectation). P2, the same for
size-matched. The apart count is reported beside each. Edge cases are in `predictions.json` ("edge_cases"). If both
sides read like the random update, P1 still passes by chance about 9% of the time; the apart count is the stricter read.

## Order, card and cost

1. **Stage 0, alone** (job 1): the self control, then the planted control, each with its stop. That is 2 crosscoders.
2. **Stage 1** (same job, after stage 0 passes): e54_matched seed 0, both sides; the random update at the reuse state's
   per-matrix norms; the reuse state with crosscoder seed 1. That is 4 crosscoders. Stops: FVE drop, reseed spread,
   change seen.
3. **Stage 2** (after stage 1 passes): the other 66 unique states, then 40 size-matched states. That is 106
   crosscoders, one file each in `results/sides/`, resumable, split with `run.py 2 i/n`. Each shard stops itself if its
   projected time passes 110 min.
4. `plot.py` on CPU: `figures/scorecard_crosscoder.png`, which is the scorecard's 67M panel (`gate_plot_1116.cells_67m`
   and `draw`) with three columns inserted after the share: crosscoder, share size-matched (E187) and crosscoder
   size-matched.

**Per crosscoder:** 2 epochs × (16.7M Pile tokens + 1.9M task tokens) ÷ 4,096 = **9,086 steps**. At 4096 × 24,576
latents, one step is about 1.9 TFLOP (encoder, dense decoder, backward), in TF32. The activations need a forward pass
through only blocks 0 and 1, which the hook cuts short. This is an estimate, not a measurement: about **4 min on an
H100** (~25 ms/step) and **about 9 min on an L40** (~60 ms/step). The first side logs its steps, so it is measured
within minutes. **Stages 0 and 1** (6 crosscoders): about 55 min on a cluster debug L40, under the 2 h QoS limit, so
they go there. **Stage 2** (106): about 16 h of L40 time or 7 h of H100 time. For the earliest result it goes to 4
H100 pods in parallel, about 27 crosscoders each, about 1.8 h, about $25 in total at $3/h. Each pod gets its inputs by
`sync()`: 7.7 GB of states (68 × 113 MB) + 3.0 GB `artifacts/vpd` + 0.14 GB Pile sets = 10.8 GB, so 10.8 GB ÷ 10 MB/s
≈ 18 min per pod. A shard needs only its own states, about 1.9 GB, which brings it to about 8 min. The alternative is
10 debug shards on the cluster at ~11 crosscoders each, if 10 L40 slots open at once (8 cards, 2 in use now). Under
$60 and 6 h of wall time, so the run is not gated.

## Departures from the literature

1. **Model and scale.** The 67M model (d = 768) and 32× = 24,576 latents, not Gemma 2 2B with 73,728. The expansion
   factor, k = 100, the learning rate of 1e-4, 2 epochs, the BatchTopK init norm of 1.0 with paired init, k_aux 512 and
   α 1/32 are all Minder's. k = 100 is a larger share of a 768-wide residual than of a 2304-wide one; it was kept as
   copied.
2. **Tokens.** 16.7M Pile train tokens (the cached rows 0 to 32,767), plus the task, for 2 epochs. Minder used 100M web
   + 100M chat. The reason is 112 crosscoders; the reseed check and FVE bars guard the smaller budget.
3. **Data mix.** The task's fit-half read windows, drawn with replacement, are 10% of tokens. Minder used 50% chat. The
   task windows number only 38 to 250, and at 50% each would repeat about 10,000 times. The reading uses the held-out
   score half.
4. **Site.** Block 2's input, the middle of 4 blocks. This matches "the middle layer" in both sources. A change that
   happens only in blocks 2 and 3 is invisible here; "1 − error part" and the change energy are reported per side.
5. **Activation scale.** One scalar for both models sets the base's mean norm to √d. This is from Anthropic's SAE
   training notes (April 2024); Minder does not state a scale, and tuning the init norm is their advice. On toy data
   the unscaled init converged slowly.
6. **Position 0 dropped** in training and reading. This model has no BOS, and first-position residuals are often
   outliers; that was not measured here.
7. **Batch 4,096 tokens, dead after 2M tokens, alive at 10 firings, θ from the last 10% of steps, constant learning
   rate.** None of these is given in the paper.
8. **Latent scaling on a small held-out mix:** 131k Pile val tokens + the task's score half, against Minder's 10M.
   Fine-tune-only latents with negative β_base or non-positive β_ft are counted as artefacts. Minder filters negative
   β_base out of the analysis.
9. **Artefacts are dropped from both sides of d.** Minder reports them; this run lists their part separately.
10. **Precision.** Model activations are computed in fp32 with TF32 off; the crosscoder's matmuls use TF32.
11. **Loss.** Squared errors. Minder's eq. 12 prints |ε| unsquared; this run reads it as the squared error of the
    BatchTopK loss they cite (Bussmann et al. 2024).

## Not run

Crosscoders on Qwen3-1.7B; any layer but block 2's input; the L1 crosscoder for comparison; the twin pairs (their
cells stay ×); more than one crosscoder seed per state, except the one reseed; a token-budget sweep.

Hand-off: a pod-runner brief from this plan; input files in `cluster/crosscoder.sbatch`'s header.
