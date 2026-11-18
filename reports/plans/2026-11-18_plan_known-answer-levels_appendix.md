# Appendix: known-answer test across levels (plan `2026-11-18_plan_known-answer-levels.md`)

*2026-11-18, definition-lead. Detail for the main plan: design check, the edits' recipes, expected values with their
derivations, code, states, pre-mortem. Nothing run.*

## 1. Design check (`/design`)

- **Goal.** `GOALS.md` → working definition of reuse (2026-11-18, without ρ). The paper's part 1 needs a definition
  that tells reuse from new. E208 showed the neuron reading saturates. This test asks whether a declared level reads
  the right label on items whose label is fixed by construction, before the trained pairs are read at it.
- **Nearest tried.** E193/E195 (the circuit turned up, the rebuilt circuit: neuron and head/MLP-block levels, the cos²
  or size-aware r), E206/E208 (grafts written into M's slots, read as new by r ≈ 0), and E196/E208 (the trained pairs,
  saturating, with gaps that follow update size). What is new here: (a) every edit has a known label, including a
  reuse edit that is re-triggered rather than turned up; (b) a new item spread thinly, the case the neuron level is
  predicted to miss; (c) the user's interchange retention, a behavioural r, in place of the cos² r; (d) coarser levels
  than heads and MLP blocks.
- **Patch or step.** A step. It checks the reading's premise on known answers before more trained items.
- **Decisions relied on.** The rows "Working definition of reuse, stated in causal-abstraction terms" and "Working
  definition of reuse without ρ" (2026-11-18), and the theory note's starred choices, except where the user's spec
  replaces clause 4's distance by interchange for multi-unit parts. No departure. Points 1 and 2 of the main plan are
  definition questions put to the user; nothing is picked between them.
- **Under an adopted definition, unchanged?** Yes, with the user's interchange for clause 4. R (mean r_u over K) and
  the band are this test's summary and reading rule. They are not added to the definition. No verdict is given on any
  trained pair.
- **What result changes what we do next.** If a level passes under reading A, run 2 reads the trained pairs there. If
  none passes, as P1-P3 derive, point 1 (which parts F needs) goes to the user before any more readings. If reuse
  small reads above new few at some level, the derivation in §3 is wrong and the code is checked first.
- **Apparatus control, first and alone.** Stage 0: the do-nothing copy at four levels under A and B; the hybrid gate;
  M's induction reproduced (E62 a_heads: `copying.induction` top-1 within 0.01 of 0.8247, and under 0.3 with (2,4)
  mean-ablated). It can fail on the swap code (do-nothing showing changed parts, a gate over 1e-3) or on the assumed
  circuit (no copy in M). Any miss stops the job.
- **Numbers.** R per edit per level per reading (4 × 4 × 2 = 32), the per-level band, and the rebuilt circuit's R per
  level (reported). Expected values are in §3. Nothing else is scored.
- **Early guards.** Stage 1 logs, for each edit, the behaviour reached (top-1 on D, KL(F ‖ M) on D, KL on Pile
  held-out as a side effect), the solved scale or gain, ‖ΔW‖_F, and the all-swapped gate at each level, even when
  they pass. The neuron-level search of new thin logs its first order's time and projects the five orders.
- **Saved for audit.** Per edit and level: every r_u, K per order, R per order, and the split-half R. Per edit, the
  changed weights only (8 slots' rows and columns; layer-0 `down_proj` for thin; (2,4)'s o_proj columns) in
  `results/large/`, with sha256. D's token ids and the cue table.

## 2. The edits

Notation: x_1(t) is the mean over D's contexts of the residual entering block 1 at a position holding token t (M's
run). ĥ_0(t) is the same for the layer-0 MLP input after rms_2. e(y) = lm_head[y] − mean_v lm_head[v].

- **Cues.** Eight A_i are drawn from E62's key pool (`copying`, the keys whose copy M already does). Eight A′_i are
  drawn from frequent single tokens outside that pool, with M's top-1 copy rate after A′_i (in place of the second
  A_i) under 0.05, checked in stage 1. Eight y_i are random frequent tokens, disjoint from both. All are seeded and
  saved.
- **D.** 512 Pile spans of 41 tokens, built like E62's `cue_windows` (the pair A_i B written earlier), with A′_i at
  the last position instead of "A M". 64 windows per cue. Read at the last position. For reuse items the target is B,
  for new items y_i. The same windows serve all four edits.
- **Slots.** The eight layer-0 neurons with the smallest mean activation² × ‖output column‖² on Pile held-out. This is
  E206's slot rule (`rho_saved_items/run.py: slotted`), at layer 0.
- **Detector rows (shared by reuse small and new few).** Slot i's input row is β (ĥ_0(A′_i) − mean ĥ_0) / ‖·‖. β is
  the smallest power of 2 at which the slot's activation on A′_i is at least 10 × its 99th percentile on Pile
  held-out. The rule is measured, not assumed.
- **Reuse, small.** Slot i's output column is γ_T (x_1(A_i) − x_1(A′_i)) / act_i, where act_i is the slot's mean
  activation on A′_i. γ_T starts at 1, and if needed doubles until the copy top-1 on D is at least 0.8 × M's own on
  E62's windows. A γ_T above 8 stops the job, as the circuit is not re-triggered.
- **New, few units.** Slot i's output column is γ_N e(y_i) / act_i. γ_N is bisected to the smallest value with
  lookup top-1 on D at least 0.9 (E206's graft reached 0.924).
- **New, thin.** ΔW_out(layer 0) = W A⁺, where A (3072 × 8) holds M's mean layer-0 activations on each A′_i and W
  (768 × 8) holds the columns γ e(y_i), with γ bisected as for new few. This is the least-norm change. Every neuron
  active on a cue moves a little; none is rewritten.
- **Reuse, large.** Reuse small, plus head (2,4)'s 128 o_proj columns × g. g solves ‖ΔW‖_F = max(‖ΔW_few‖_F,
  ‖ΔW_thin‖_F), the root of the summed squares over every changed weight, in absolute units. If no g > 1 solves it
  (the new edits are not larger than reuse small), g is set so that ‖ΔW‖_F = 2.67 × ‖ΔW_small‖_F. That is E196's
  largest twin-over-reuse MLP update ratio (case, `rule_retention_pairs/results/update_size.json`: 0.0200 / 0.0075),
  and the match is then reported as not met. The relative norms beside E196's (MLP 0.0052-0.0200, attention
  0.0042-0.0211) are reported too.

Why these labels hold. In reuse small, the eight slots only translate a new cue into an old one. B is found and copied
by heads (1,1) and (2,4) exactly as in M, since a token-local write cannot know B. In new few, the same slots write a
fixed answer that M never gives after A′_i. In new thin, the same answer is written by small changes to every active
neuron. Reuse large is reuse small with M's copying head turned up.

## 3. Expected values (reading A, clause 5 as filed)

K holds only changed parts (theory note, (b)). F with every changed part swapped is M. So for any part u that holds
the whole edit, r_u = 1 − KL(F ‖ M) / KL(F ‖ M) = 0 exactly.

| edit | neuron | head | MLP layer | block |
|---|---|---|---|---|
| do-nothing | no change | no change | no change | no change |
| reuse, small | ≈ 1 − 1/8 (K = the 8 slots; each carries one cue) | 0 (K = {MLP 0}) | 0 | 0 (K = {block 0}) |
| reuse, large | ≈ (8 (1 − 1/8) + r_(2,4)) / 9 | r_(2,4) / 2 | r_attn2 / 2 | r_block2 / 2 |
| new, few | ≈ 1 − 1/8 | 0 | 0 | 0 |
| new, thin | ≥ 0.99 (each neuron carries ~1/|active|) | 0 | 0 | 0 |
| rebuilt (E195) | reported | reported | reported | reported |

Here r_(2,4) = 1 − KL(F ‖ F with (2,4) at gain 1) / KL(F ‖ M) lies strictly between 0 and 1. "≈ 1 − 1/8" is exactly
1 − (mean over K of each slot's share of KL), so reuse small against new few measures how KL splits across the eight
cues, not reuse. If a slot carries under 0.1 of KL it can drop out of K and R moves; this is logged.

So under reading A no level passes. The coarse levels tie reuse small with both new edits at 0. The neuron level puts
new thin on top. Reuse large reads above 0 only through its extra changed part.

**Reading B (patching).** Swapping a part outside K to M's own-run value moves the unchanged downstream parts too: the
copy is lost at (2,4) because (2,4) did not copy in M. So (2,4) and the layer-2/3 parts carrying the copy enter K
with r near 0, and reuse small reads no higher than under A. For new items, the direct write passes through later
parts, whose values also differ from M's, so they enter K as well. P4 files only the reuse direction and "no level
passes".

**Reading 2 of point 1 (not built), for the user's choice.** If unchanged parts the behaviour runs through count as
needed and keep their rule at r = 1, reuse small's block-level reading rises to about 2/3 (block 0 at 0, blocks 1 and
2 at 1). New few and new thin stay near 0, as their write needs no later part. This is the reading under which the
pass rule could pass. It needs a rule for what an unneeded part is replaced by (a mean over D, a resample), and that
rule is the user's.

## 4. Code

- `src/commute/levels.py` (new, about 120 lines). A level maps each named part to a list of `rho` parts (neuron,
  head, embed, unembed); swapping a part swaps them all. Functions: `parts(level, pair)`, `changed(level, pair)` (a
  group is changed if any member is), `interchange(pair, batches, group, patch)` → r_u, and `search(...)` → K per
  order, which calls `rule_retention_fixed.eliminate` with a closure over groups. `rho.forward`, `swap`, `m_run` and
  `Batches.kl` are called unchanged. No finished experiment's frozen script calls this new module.
- `src/commute/known_edits.py` (new, about 150 lines). Cue and window build (adapting `copying.cue_windows`, not
  changing it), slots, detector rows, the four edits, the γ bisections, g's solve, and ‖ΔW‖_F.
- `experiments/known_answer_levels/run.py` (about 250 lines). Stages 0-3, plus `--pairs` for run 2. It reuses E196's
  pair loading (`old_tasks`, `variable_read.load_state`). The rebuilt circuit is loaded as E208 loads it.

## 5. States

- M: `artifacts/vpd` (the 67M), as every 67M run.
- Rebuilt circuit: `experiments/rule_retention_fixed/results/large/rebuilt_mlp.pt` (37.8 MB, on disk), read on
  copy24's windows as E195/E208 did.
- Run 2: E196's `PAIRS` at seeds 0 and 1, all on disk 2026-11-18: `open_pair/results/large/amend1/copy_s{0,1}.pt`,
  `open_pair/results/large/cipher_s{0,1}.pt`, `regating_table/results/large/{ident32,rand32}_s{0,1}.pt`,
  `case_label/results/large/dW_{real,twin}_s{0,1}.pt`, `fresh_reuse_pair/results/large/dW_{real,twin}_s{0,1}.pt`,
  `memorised_fact/results/large/fact_s{0,1}.pt`, `built_control_sum/results/large/amend1_control_s{0,1}.pt`. Seeds 2
  and 3 exist only for copy and cipher (copy_s2, cipher_s2), so run 2 reads seeds 0 and 1. Each pair is read on its
  E196 read windows (512, `read_seed` 195200).

## 6. Pre-mortem (ungated: under $60 and 6 h)

- **Recipe.** No training. The risks are in the hand-built writes. A detector that fires off-cue gives a side effect
  on Pile (logged as KL on Pile held-out, not scored). A translation that fails to re-trigger is caught by the copy
  top-1 bar and γ_T's cap.
- **Data.** A′_i must not already trigger copying in M (the 0.05 check). A_i must occur only at the planted position
  (as `cue_windows` filters).
- **Precision.** fp32 throughout, as E208. The gates are at 1e-3.
- **Memory and time.** The 67M with 512 × 41 windows fits any card. The only long search is new thin at the neuron
  level (3,072 candidates against E208's ~12,000), projected from its first order.
- **Saved.** Changed weights only (about 2 MB) plus the JSONs. Downloaded at once if run on a pod.
- **Controls.** Stage 0 alone. The all-swapped gate per edit and level.
- **Predictions.** Each rule was run against its edge cases (§7).
- **Code never run on a GPU.** `levels.py` and `known_edits.py`. The scored run is its own smoke (stage 0 first).

## 7. Edge cases of the filed rules

- **Do-nothing.** No changed part at any level, so R is undefined and the item reads "no change". The pass rule is not
  applied to it, and it cannot pass.
- **Ties.** "Exceeds by more than the band" is strict. A band of 0, as at coarse levels where R is exact, needs a
  strictly positive gap, so reuse small at 0 against new at 0 is not above.
- **Empty K.** Impossible while KL(F ‖ M) > kl_floor, since F with every changed part swapped is M. An empty K stops
  the job (a code error).
- **One-part K.** R = r_u. This is the coarse-level case, and it is exact.
- **R below 0.** If a swap overshoots, KL(F ‖ F_swap) can exceed KL(F ‖ M). R is kept signed, not clipped, and the
  rule compares signed values.
- **The g solve has no solution above 1.** The fallback is stated in §2 and reported.
- **A slot drops out of K.** R is computed over the K found. This is logged, not a stop.
