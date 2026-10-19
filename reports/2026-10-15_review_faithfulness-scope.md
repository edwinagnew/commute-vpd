# Review: what round 2's "faithful" claims can and cannot see

*2026-10-15, coding agent, after the user's objections to the round-2 reading ("lots of red flags"). Applies to
`reports/2026-10-14_analysis_lift-test.md` (P2′, P3′, "What this does and does not say"), `experiments/TRIED.md` row 2
and `reports/2026-10-14_plan_new-skill-test.md`. Everything below was checked against the code
(`experiments/lift_test/run.py`, `src/commute/pullback.py`, `src/commute/faithfulness.py`) and the cluster's
`results/large/` listing, not recalled from the reports.*

## 1. The faithfulness harness cannot see the trained behaviour

**What the four ratios are computed on.** Every faithfulness number in rounds 1 and 2 (stochastic, PGD, CI-masked,
unmasked, and the "gap recovered") is a mean KL over Pile val rows 0..255: 256 rows × 512 tokens, about 131 000
positions (`_faith_blocks`, `faithfulness.py`). None is computed on the emoticon windows.

**How much of that is the trained behaviour.** Round 1 found 438 trigger positions in 32 768 Pile train rows, one per
75 rows. So the 256 evaluation rows hold about 3 trigger positions out of 131 000. Even if the component model got the
`→ o` behaviour completely wrong at every one of them (say 5 nats each), the mean would move by about 1e-4 nats against
a baseline of 0.25 (stochastic) or 0.7 (PGD). That is invisible at any threshold we have used.

**What the harness does see.** Two things: the fine-tune's generic side-effects on Pile text (the fine-tune moves Pile
KL from W₀ by 0.02 nats), and global fragility under the adversary (the lift's PGD rise to 5.4). Both are real and worth
measuring. Neither is "the components explain the new behaviour".

**Consequences for the filed claims.**
- P3′ (pull-back "stays faithful on every metric") means: the pull-back did not damage the decomposition's account of
  the *old* model and its side-effects. Whether the `→ o` behaviour at the trigger is carried by components Γ gates on
  there, under CI masking, is **unmeasured**. Do-nothing also scores 0.99 on the stochastic ratio, which shows how
  little the bulk metric asks.
- P2′ (lift "carries the update", gap recovered 0.88–0.93) is an unmasked number on Pile. It says the lifted components
  reproduce the fine-tuned model's generic effect. It does not say the behaviour survives masking.
- Round 3's falsifier as planned (CI-masked ratio on Pile) has no power for the same reason. It would pass whatever
  happened at the trigger. **Round 3 cannot be run with this harness.**

**Fix (measurement, before any new round).** Run the same four metrics on the held-out emoticon windows, with the
trigger position reported separately from the other 40 positions, and report P(o) of the masked component model at the
trigger. A positive control is needed to show the fixed harness has power: the unchanged decomposition on W₀ + δW
(do-nothing) must *fail* at the trigger, or the trigger-local metric is also blind.

## 2. A noise floor has never been measured for the ratios

The PGD adversary is itself a stochastic optimisation and the paper's own run logged 0.59–2.24 across 100 random
batches. Our ratios use the same rows and seed in one job, which removes batch variance, but the seed-to-seed spread of
the ratio on W₀ against itself has not been measured. The 1.5 threshold is therefore unanchored. Fix: repeat each metric
on W₀ with three adversary seeds before setting any threshold, and file the spread with the predictions.

## 3. The pull-back was never run on a LoRA

Checked: the pull-back stages (`stage_pullback`, `stage_pullback2`, `stage_pullback_early`) all take arm A's full-weight
recipe and train the alive (U, V) offsets of all 24 matrices. The LoRA stage (`stage_lora`) trains a normal LoRA on
`h.2.mlp.down_proj` and only measures its obstruction ω. So the sentence "cheap continuous updating during
post-training, at least with LoRA" has **no evidence behind it** in either direction.

The LoRA obstruction result is also uninformative: a LoRA starts with B = 0 and random A, so the read side is at its
random init (cosine 1.00 at step 1, 0.92 at step 10), and ω ≈ null follows by construction. Only the write side was
trained from zero, and it was no better aligned than random (0.547 against null 0.534).

What "LoRA through the pull-back" should mean has to be fixed by `/design`; candidates are (i) the pull-back restricted
to the LoRA's matrix, (ii) a rank-r restriction of (dU, dV), and (iii) appending the LoRA as a new rank-1 component with
its own gate, since a rank-1 LoRA b aᵀ already has the form of a VPD subcomponent.

## 4. Which components the pull-back changed, and whether Γ gates them, was not measured

Round 2 scored faithfulness and nothing else about the pull-back's solution. The per-component (U, V) states at steps 10
and 30 were held in memory for the faithfulness evaluation and **not saved** (`_pullback_runs` returns them; only the W
fine-tune's δW was written to `results/large/A_rerun`). The cluster's `results/large/` holds `A_rerun`, `LoRA_r1`,
`LoRA_r8` and three `SGD_*` directories, no pull-back states. The identity test (`reports/2026-10-15_plan_identity-test.md`,
running on 2026-10-15) regenerates them and measures whether the carrying subcomponents stay the same subcomponents.

What it does not score, and what the user asked for on 2026-10-15: **compare the update to Γ's values.** At the
held-out trigger positions, read the frozen Γ's g_c(x) for every alive subcomponent and set it against each
subcomponent's share of the update (gauge-free: functional drop on revert, as the identity test selects, or
‖Δ(u_c v_cᵀ)‖). Two questions: is the update concentrated on subcomponents with high g at the trigger, and are the
subcomponents that carry it gated on (g_c > 0.5) there at all. If the behaviour sits in subcomponents Γ never gates on
at the trigger, the CI-masked model cannot express it, which is exactly what the trigger-local metric in §1 would
show from the other side. No Γ re-training is involved; this is reading its values.

## 5. What this does to the round-2 reading

- "Inside works" shrinks to: training the alive (U, V) with Δ and Γ fixed learns the emoticon behaviour with less
  off-target damage than the W fine-tune and without breaking the decomposition's account of the old model on Pile.
- "Outside fails adversarially" stands as a Pile-level fragility result, cause unknown.
- Nothing yet says the new behaviour is explained by the components under Γ, for either route. That is the question
  the project is about, and it needs the §1 and §4 measurements before anything else is planned.

## 6. Why these were missed

Each round filed predictions on the metrics that existed rather than asking first what each metric is blind to. A
"blind to" line per metric, and a positive control showing the metric moves when the claim is false, belong in every
plan before predictions are filed. Rule proposed in `backlog.md` and recorded in the agent's memory.
