# E127: why the least-squares start reads worse than do-nothing under the adversarial check

*2026-11-06, regating-lead's writer, from the lead's brief (run down an anomaly before readings build on it). Code
`experiments/ls_start_faith/run.py`, filing `results/predictions.json`, job `cluster/ls_start_faith.sbatch` (MATS `debug`,
L40, $0, about 30 min as two jobs). Not gated: $0 and under 6 h.*

## The anomaly, in the Terms' words

Placing a fine-tune's update δW in the released decomposition's alive subcomponents by the least-squares split (the
"least-squares start") reads worse under E34's adversarial check than the do-nothing state (δW left in Δ):
lowercase Adam (E122 cell c3, Pile set A) do-nothing 4.79, start 8.33; emoticon Adam (E121 Amendment 1) 1.39 against
1.57. Copies trained from that start keep the excess (9.40, 1.55-1.57); standard copies do not (2.17, 0.97-1.00).

**Two premises of the brief that the code does not bear out.** (a) *"Same function at full masks."* Both saved starts
were read with the target at W₀ + S, where S = (U + dU)(V + dV)ᵀ − UVᵀ is the split's own change (E121
`amend1.make`: `ls[1]`; E122 `stage_c3` → `offsets_state`), not W₀ + δW. S drops the split's remainder (the part outside
the alive subcomponents' reach, plus dU dVᵀ): E16 put the emoticon least-squares remainder at 0.36-0.38 of δW's
energy, and E121 read the start's P(target) at 0.987 against do-nothing's 0.999. So start and do-nothing are different
functions. (b) *"In Δ the update is always on."* Δ's mask is adversarial too (`use_delta_component=True`: one source per
matrix, in [0, 1]); the difference is that in Δ the update shares one knob per matrix with W₀'s own Δ, while on the
subcomponents it rides 9,881 per-subcomponent masks, each ci + (1 − ci)a, so in [ci, 1].

## Design check (`/design`)

Approved class 1 (GPU passes of the existing harness on saved states) and 2 (controls: the reproductions). It reuses
E17's cancellation reading and E20's finding that the step-10 split state's excess sat in the dead and Δ blocks'
interplay with the alive removal, and asks the question at step 300/1000 on full updates, which neither read. No
definition is touched; the reuse definition is not read.

## What runs

Per task (lowercase L first: its excess is 3.5 ratio points, beyond the floor; emoticon E: 0.18, under it), with the
W₀ floor f = 0.209 (L40, seeds 0-2) as the scale:

| State | Target | Where the change sits | Read |
|---|---|---|---|
| nothing | W₀ + δW | Δ | free (control) |
| start | W₀ + S | alive subcomponents | free (control), held |
| nothing_start | W₀ + S | Δ | free, held |
| W₀ | W₀ | none | free (control), held |

"Held" = every alive subcomponent's mask fixed at 1 (the adversary cannot move them; dead subcomponents and Δ stay
free); its ratio is over W₀ read under the same hold. The split changes every alive subcomponent (measured on the saved
offsets: 9,881 of 9,881 non-zero, 79% needed for 90% of the energy), so the changed set is the alive set.
Random masks (`faithfulness` with 0 PGD steps: stochastic, CI-masked, unmasked) at W₀, nothing_start and start. Weight
side: S against δW (input-weighted on the check's rows, and Frobenius), and the coherence κ of start's, the standard
copy's and the least-squares-start copy's per-subcomponent changes (E17's κ, plus input-weighted).

## The readings, each able to fail (rules in `predictions.json`)

1. **Function or placement.** The excess start − nothing splits exactly into Xf = nothing_start − nothing (S is not δW)
   and Xp = start − nothing_start (same function, change moved from Δ to the subcomponents). Each carries if at least
   f times its reference. Lean (L): placement 0.5, both 0.25, function 0.15. Expected E: within the floor.
2. **Exposure of the alive masks** (if placement carries). H = share of Xp removed under the hold. H ≥ 0.75: the
   adversary's freedom over the alive masks carries it; ≤ 0.25: it does not (it sits in the dead or Δ masks). Positive
   control: W₀|hold against W₀ (the hold must change the adversary; a no-op hold reads H = 0 and W₀|hold = W₀).
   Expected W₀|hold 0.5-1.2 nats (E20: alive kept whole, dead and Δ fractions removed, 1.08 against 0.66).
3. **Random masks** (if placement carries). start over nothing_start under random masks, minus 1, against f. Lean
   "adversary only" 0.8 (E15: the step-10 split state read 0.92-1.07 stochastic at PGD 6.96).
4. **Cancellation** (weight side). κ < 0.5 on start and the least-squares-start copy and not on the standard copy:
   "tracks the failing states". Expected start 0.3-0.6, standard copy 2-4 (E17). Descriptive: E17 found the step-10
   split's cancellation did not carry its excess (≤ 7%).

## Controls first (each stops the job on a miss)

W₀ mean block KL 0.69174, then E do-nothing 1.39002, E start 1.57328, L do-nothing 4.79423, L start 8.33289, each within
1e-3 relative of its L40 reading (same rows, seed, code path and saved files); L40 card; a resubmitted job rereads W₀'s
first block within 1e-6; the alive U, V equal E121's.

## Apparatus

- `faithfulness.harness_pgd(..., hold=None)`: new keyword, default unchanged. It wraps the authors'
  `interpolate_pgd_mask` so held entries read 1 (zero gradient, so their sources never move). Frozen callers (no
  `hold`): `experiments/{broad_fragility/run, copy_gap_ladder/amend1, damage_crossed/run, row_blocks/run,
  teacher_training/amend5}.py`, `faithfulness.faithfulness`, `start_kl`, `gamma.py`. CPU check: held entries 1 with zero
  gradient, others unchanged, restored after an exception, unknown matrix refused.
- States are rebuilt exactly as their source runs read them: E start by `copy_gap_ladder/run.ls_start` from A8c's
  saved δW (fp32), L start from `tt_c3_start.pt` through `offsets_state`.
- Order: the five control reads, the weight side, then L's held and nothing_start reads, random masks, then E's.
  Each unit is written as it ends; one that might not end inside 28 min exits 3 (`fleet/resubmit --code 3`).

## Time and cost

12 PGD reads at 118-132 s each (`damage_crossed.json` and `copy_gap_ladder_amend1.json` `.seconds`) ≈ 25 min; start-up
≈ 1 min (E121 A1 log: bases and moments 9 s; E122 start-up 34 s); 5 random-mask reads and the weight side not measured
(bounded at 60 s and 120 s). About 30 min: the first job exits 3 near 28 min, a second job of ~3-5 min finishes. $0.

## Not run

Adversary seeds 1-2 (emoticon's excess is under the seed floor and is not readable from one seed); a hold at the CI
values (the adversary's minimum: a different question, switching the update off); a hold of a subset of the alive
subcomponents; Pile set B; the copies' PGD (read in E121 A1 and E122); the start with target W₀ + δW and the remainder in
Δ (`split.split_state` as written; nothing_start covers the function difference).
