# Review (E55 C3 code): commit fd8be40, adversarial, independent of writer and planner

**Reviewer: a fresh Opus agent, 2026-10-31.** I did not write the code or the plan. Scope: `src/commute/revival_kv.py`,
the additions to `src/commute/finetune.py`, `tests/test_revival_kv.py`, `experiments/open_pair_instruments/*.py`,
`cluster/open_pair_instruments.sbatch` and filing A in `results/predictions.json`, read against
`reports/plans/2026-10-31_plan_e55-stage-c-open-pair.md`. Job 1's output (job 46774, `results/measure.json`) is
included, because it arrived during the review. The CPU tests pass (32 of 32, run through `fleet/cpu-run`).

## Verdict in plain words

I found no code bug that gives a wrong number and no crash on the GPU paths I could trace. κ, D, the prior, the
pruning scores, the reverse loop, the half-life, censoring and rules 1-7 all match the plan. Job 1 brought out two
problems in the design, not the code. Both must be settled before filing B. (1) The planted control cannot pass as
specified: the unit rule picks nearly silent units, and the off-answer bound caps the planted D at 0.18 nats against a
target of 7.0. (2) The measured baselines read p_D = 1.47 and p_prior = 0.92, both firmly "VPD". Under filed rules 5
and 6, a "VPD" reading from any instrument therefore becomes "distance and prior". That holds for r and the stand-in
ratio too. The verdicts "shared" and "VPD alone" can no longer be reached.

## Findings

| # | where | severity | finding | fix |
|---|---|---|---|---|
| 1 | `revival_kv.py:199-208` (`planted_units`), `:218-224` (`planted_dw`), `measure.py:88-103`; plan § 5 item 5 | **design, blocks C0b** (the code follows the plan) | The rule "above-median activation, lowest single-unit zeroing cost" picks the least active units that qualify. The block-3 median activation is negative, so 1,160 units qualify, and the 13 cheapest have ā between 0.0024 and 0.097, with zeroing costs of 4-8e-5 nats. The edit is −α·w/ā_j on the down_proj column, so it is scaled up by as much as 400×. It acts at every position where the unit fires, and a nearly silent unit at the answer position can fire more strongly elsewhere. The off-answer KL reaches the bound (0.577, the cipher's) at α = 0.0067, where κ is only 0.184 against a target of 7.0. For comparison, the real states have off-answer KL 0.12-0.58 at D 4.4-10.6: they are 20-80× more answer-specific than this plant. The bisection worked as written: off = 0.57670470 against a bound of 0.57670472, and the fallback was recorded. | A dated amendment to the planted rule, then a rerun of job 1 items 5-6. Choose units by **answer selectivity**: the fit-half mean activation at the answer position divided by the mean of \|activation\| at other positions, keeping only units whose ā is above, say, the block's 90th percentile. A single-unit cost cap can stay as a tie-break. Also consider dropping the 1/ā_j factor (equal α per unit), because it is what amplifies the near-silent units. If 13 units still cannot reach a pole-sized D within the bound, file a planted D floor (below). |
| 2 | `measure.py:106-125`, `instruments.py:44-66`; plan § 5 job 3 item 3 | **design, makes the pruning control uninformative** | The planted pass is relative to the ceiling, (D − κ_zeroed)/D = 0.99, so R_min = 0.79. Noise is not relative: at the planted D of 0.184, the random 13-unit removals have an sd of 0.13 of D, against 0.003-0.008 on the real pole states. The result, 4 of 13 found and R 0.39, comes from a plant that is close to noise. The planted state's D is also below `D_void` (0.98), so as a real state it would be void. Nothing in the code checks this. | With finding 1, require planted D ≥ D_void (or ≥ some fraction of the median pole D) for C0b to count. A planted state under the floor reads "control not built", not "pruning failed". |
| 3 | `revival_kv.py:381-394` (`tracks`, `verdict`); rules 5-6 as filed; `measure.json` `filing_B_draft` | **design, decides the verdict before any instrument runs** | Job 1 measured mean D at rank-1 8.72, cipher 4.89, elicit 10.53 and copy 7.84, giving p_D = 1.47 (90% interval 1.43-1.53, firm "VPD"). The prior means are −5.00, −14.32, −5.78 and −8.85, giving p_prior = 0.92 (constant per map, so the bootstrap interval has zero width, firm "VPD"). Rules 5 and 6 fire when an instrument's firm side *equals* the baseline's. So any instrument that reads "VPD" becomes "distance and prior". The same happens to r (p 0.876) and the ratio (0.787). Rule 7 can no longer reach "shared" or "VPD alone". The best "shared"-like outcome now reads "not settled". P2/P3's 25% "VPD" leans and V's 25% "shared" lean are therefore no longer reachable. The code implements the filed rule correctly; the filed rule no longer separates anything on this axis. | The lead must decide before filing B, and possibly with the user, since it bears on the line's claim. Options: (a) keep the rules and revise the leans in filing B: P2/P3 VPD → 0, V shared → 0, and say plainly that on this pair the decomposition-free baselines already place elicit at the old end; (b) a dated amendment that keeps only the 0.1 clause (\|p − p_base\| ≤ tol) and drops "same side", because "same side" is far too coarse when the baseline sits outside [0, 1]. Filing A cannot be edited in place either way. |
| 4 | the plan's § 4 expected values (not the code) | **expected-value miss, not a bug** | D and the prior are computed as the plan defines them. D is KL(p_W0 ‖ p_m) (`kl_rows(lp0, lp)` = Σ p0 (log p0 − log p)), at the answer position (`ref.half`, the logit at the key token), on the score half (odd held-out windows of the trained keys), in fp32 with float64 sums. The prior is `Task.lp0` = log P_W0(value) at the same position, on the same score half. Rank-1's prior of −5.00 agrees with Amendment 1's recorded W0 geometric mean of 0.00567 on the trained keys (log −5.17). The plan's guess of −2 to −3 was wrong: a rank-1 value is W0's argmax for the key in another context, not at these windows. The D guesses (1.5-3) were wrong because KL(W0 ‖ m) is dominated by the W0 mass on the tokens a sharpened state suppresses. It reads how peaked the state is, which explains elicit > rank-1 > copy > cipher (the cipher is read at an early, norm-matched step). | Nothing to fix in code. Filing B should say that D measures how peaked the state is, and should replace § 4's guesses with the measured means. |
| 5 | `measure.py:76-85` with `revival_kv.py:261-271` | minor (cosmetic) | `spread` calls `prune_readings(..., scores=None)`, so job 1 never runs a top-K removal on a scored state, as the plan requires. Without a "top" reading, `revival()` returns Δ and R as nan, which is expected. The random-removal sd, which is the only thing used (R's floor = 0.00825, from K = 40), is computed correctly. | Leave it, or store only `spread` in the spread record so nobody reads the nan R as a failure. Job 1 does not need a rerun for this. |
| 6 | `predictions.json` `constants.planted.block` = 3 read as site `mlp3`; plan § 3 numbers the probe's blocks 1-4 (1-based) | minor (convention) | "Block 3" in plan § 5 is ambiguous. The plan's own probe numbering is 1-based, which would make it `mlp2`. The writer filed `mlp3`, 0-based, in filing A's implementation details, so it is fixed and stated. The last block is the easiest place to plant a control, because the edit reaches the unembedding directly. | No change needed. If finding 1's amendment re-specifies the units, state the block again in the same 0-based words. |
| 7 | `revival_kv.py:145-170`, `finetune.py` `finetune_to_reference` | judged against the plan: right | The reverse fine-tune uses a frozen deep copy of the target at W0 as its reference, with the loss KL(p_ref ‖ p_model) (`kl_per_position(ref_logits, logits)`). That is the plan's loss. The plan's `ref_logprobs` argument would not have fit in memory, and the departure is filed. `_run` treats its starting weights (W0 + δW) as "W0" and restores them on exit. `net.at` then restores the true W0. Job 1's planted reverse ended with the weights exactly at W0. The loss runs under bf16 autocast, as E54's forward recipe does, while κ is read in fp32. | None. |
| 8 | `revival_kv.py:336-353` (`bootstrap_p`, `percentiles`) | judged against the plan: right | Draws with a zero denominator go to −inf for the 5th percentile and +inf for the 95th. A VPD side needs lo ≥ 2/3 and a label side needs hi ≤ 1/3, so a nan draw can only count against a side. That is the plan's "outside every band". | None. |

## Paths checked and judged right

- **κ and the halves.** The fit half (even held-out windows) only ranks units and selects the planted units. D, R,
  t½ and the planted α are all read on the score half. The reverse fine-tune trains on the separate `train` windows.
  No score-half window enters a gradient.
- **Pruning.** In `unit_scores`, the score Σ grad·w over row j of `c_fc` (the model is `llama_simple_mlp`: `c_fc`, NewGELU, `down_proj`, no
  biases) is the first-order drop in κ from zeroing unit j's pre-activation. That equals zeroing at site `mlp<l>`,
  because NewGELU(0) = 0. The sign is right: the largest scores are removed. The random draws use `default_rng([seed, K])`, the same for every
  state. `zero_units` maps flat ids to blocks correctly (and is tested). Δ_K, R = max_K Δ_K/D and the spread in
  units of D match the plan.
- **Half-life.** `curve[i]` is κ after optimizer step i (`on_step` runs after `opt.step()`). The interpolation
  `i − 1 + (a − t)/(a − b)` is right. A cap exit becomes "censored", and a censored pole voids t½. A censored elicit or copy
  seed is dropped. A deadline exit records nothing for the state, which reruns on resubmission.
- **The probe.** `in<l>` for l = 1-3 is the residual after blocks 1-3, counted from 1. `out3` is the last block's
  output: the Block returns a tensor, so the capture works. The lens check compares the block-4 lens with the real
  logits.
- **Rules 1-7** match § 6. The tests cover the plan's worked examples. The undirected columns use |gap|, a gap of
  exactly 0 reads "no gap", and the joint-verdict precedence is VPD, then label, then conflict.
- **Job 2** calls `elicit_ft` and `cipher_ft` exactly as E58 and Amendment 1 do: the same windows files, alive
  path, thresholds and bases per map. It trains elicit before the cipher in every seed, and passes the cipher the same seed's elicit norm.
  Resume skips a state only when the file's sha256 matches.
- **Job 3** resumes by state, reruns stage 0 and checks every state file against its recorded sha256. It also checks
  that the weights are back at W0 after every reverse run.
- **Frozen callers of `finetune.py`.** Only new code was added; `_run` and `finetune_arm_a` are unchanged.

## Job 1: what can be trusted

- **Trust these, no rerun:** the stop rows (sha256, identical tokens, every G within 1e-4, the zero update's D
  exactly 0); D, the off-answer KL and the MLP share per state; the prior per map; the random-removal spread and R's
  floor (0.00825); `D_void` (0.978).
- **Rerun after an amendment:** items 5-6, the planted wrapper and its pruning check (findings 1-2). `measure.py`
  refuses to run while `measure.json` exists, and it recomputes everything. The simplest rerun is therefore to move
  `measure.json` aside and run job 1 again (55 s).

## Not run

- No GPU execution of my own. Findings 1-4 rest on job 46774's numbers and on reading the code.
- Job 2 and job 3's per-state paths have never run on a GPU. Stage 0's lens check and `score()` were traced by
  reading only.
- I did not check whether 13 selective units can reach a pole-sized D within the off-answer bound. That needs the
  amended job 1.
