# Appendix: P1.35 (T2), R on built items at four levels

*2026-11-18, definition-lead. Detail for `2026-11-18_plan_p1-35-t2.md`. Nothing run. Notation: M the 67M (4 layers,
6 heads of 128, 3,072 MLP neurons per layer, width 768), F the edited model, D an item's windows, s_u a variable's
mechanism grade, K\* the minimal circuit, C the changed variables, f = |K\* ∩ C| / |K\*|, ρ′ = mean s_u over K\* ∩ C,
R = mean s_u over K\* = 1 − f(1 − ρ′).*

## 1. Levels (unit featurizations fixed from M; exact, no remainder needed)

| level | variables | count |
|---|---|---|
| L1 neuron + head | each MLP neuron, each head, embedding, unembedding (with ln_f) | 12,314 |
| L2 head + MLP layer | each head, each whole MLP layer, embedding, unembedding | 30 |
| L3 MLP layer + attention layer | each attention layer, each MLP layer, embedding, unembedding | 10 |
| L4 block | each block, embedding, unembedding | 6 |

Every weight belongs to one variable: the rms gains go with their head, neuron or layer, and ln_f goes with the
unembedding. So the coverage condition holds with no remainder, and the remainder's share of K\* is 0 on every M-shaped
item. A grafted head or neuron is its own variable where its size is a level's unit: L1 and L2 for a head, L1 for a
neuron. A coarser level holds it inside the MLP layer, attention layer or block it was added to (choice §4e). E137's
pieces write at block 2's input and belong to no unit of M. They enter at every level as the rest-of-network remainder
variable of the coverage condition, and M's mechanism for it writes zero.

## 2. Items

- **P1.31's four edits.** These are built exactly as `2026-11-18_plan_known-answer-levels_appendix.md` §2 gives them:
  the eight cue pairs (A_i, A′_i), D of 512 41-token Pile windows with A′_i last, the eight quietest layer-0 slots,
  the detector rows, γ_T, γ_N, the least-norm thin change, and g matched to the larger new edit's ‖ΔW‖_F with the
  2.67 fallback. The build checks and their stops are kept: copy top-1 ≥ 0.8 × M's, lookup top-1 ≥ 0.9, M's copy
  after A′ ≤ 0.05, and γ_T ≤ 8. Read at the last position.
- **E195**, on copy24's 75 held-out windows at every position (E195's D). ×2 and ×10 scale the output columns of
  layer 1's eight copy neurons C. The moved copy zeroes C's columns and copies C exactly onto eight appended neurons.
  Rebuilt zeroes C and loads layers 2 and 3's MLP from `rule_retention_fixed/results/large/rebuilt_mlp.pt`
  (37.8 MB). The builders are E208's (`rho_saved_items/run.py: rebuilt`, and E195's `moved`), loaded with
  `commute.jobs.load_experiment`.
- **Grafts**, on E206's read windows (val rows, window 32, 512 windows, seed 201005), read where E208 read them.
  They are E201's head (`graft_negative/results/large/graft.pt`), E206's `last_lowest`, `last_rank5`, `last_rank9`
  and `last_rank13` heads, and its 256 appended `neurons` (`graft_sites/results/large/*.pt`, 1.6 MB each).
  `early_lowest` failed its checks in E206 and is left out.
- **E137's pieces**, on each pair's read half, at the last position: Always on, Cue, XOR and Table, from
  `reuse_variable_test/results/large/pieces.pt` (2.4 MB). Each is a one-line write at block 2's input
  (`pieces.piece_edit`). Their E137 labels (reuse, reuse, built, rewired) were set under the superseded second wording,
  so they are reported and not counted.
- **Do-nothing**: F = M, read on the edits' D.

## 3. Expected K\* and R by hand

The assumptions are made up and stated so the arithmetic can be checked: q = 1/41 = 0.0244 is the share of settings
holding the cue, the cosine at the cue is 0.5 for the reuse write and 0.2 for the new write, and the thin edit's leak
off the cue gives 1 − s = 0.04. The gain on (2,4) gives 1 − s = 0.15 for attention layer 2 and 0.08 for block 2.
Slots at L1 have s ≈ 0, because a rewritten column has cos² ≈ 0 against M's: E208's slotted item read 0.002. Under
these, 1 − s is q × 0.5 = 0.0122 for the reuse slots and q × 0.8 = 0.0195 for the new ones.

| item | L1 (K\* assumed) | L2 | L3 | L4 |
|---|---|---|---|---|
| do-nothing | 1, f = 0 | 1 | 1 | 1 |
| reuse small | 1 − 8/300 = 0.973 (K\* ~300: induction path + slots) | 1 − 0.0122/10 = 0.9988 | 1 − 0.0122/8 = 0.9985 | 1 − 0.0122/6 = 0.9980 |
| reuse large | 0.973 ((2,4) in K\*, s = 1) | 0.9988 ((2,4) s = 1) | 1 − (0.0122 + 0.15)/8 = 0.9797 | 1 − (0.0122 + 0.08)/6 = 0.9846 |
| new few | 1 − 8/30 = 0.733 (K\* ~30) | 1 − 0.0195/4 = 0.9951 | 1 − 0.0195/3 = 0.9935 | 0.9935 (K\* = {emb, b0, unemb}) |
| new thin | 1 − (200/400)(0.02) = 0.99 | 1 − 0.04/4 = 0.990 | 1 − 0.04/3 = 0.9867 | 0.9867 |
| min reuse − max new | −0.017 (no) | +0.0037 (pass if band < 0.0037) | −0.014 (no) | −0.009 (no) |
| ×2, ×10 | 1 exactly (gain) | 1 − (1 − s_mlp1)/\|K\*\|, ×10 lower | same | same, block 1 |
| moved copy | 1 − 8/\|K\*\| (appended C in K\*, s = 0) | 1 (MLP1's write is M's exactly) | 1 | 1 |
| rebuilt | ≥ 0.995 (E208: ρ′ ≈ 0.995 on its 6,144 changed neurons) | MLP1, 2, 3 graded | graded | blocks 1 to 3 graded |
| graft heads | 1 − 1/\|K\*\|; 2/3 if K\* = {emb, graft, unemb} | 1 − 1/\|K\*\| ∈ [0.67, 0.88] | 1 − (1 − s_attn3)/\|K\*\|, about 0.8 to 0.93 | block 3 graded |
| graft neurons | 1 − g/\|K\*\|, e.g. 1 − 200/252 = 0.21 | MLP3 graded | graded | graded |
| Always on | 1, f = 0 (a constant is its own position-wise mean, so never in K\*) | 1 | 1 | 1 |
| Cue, XOR, Table | 1 − 1/\|K\*\| if the piece is in K\*, else 1 | same | same | e.g. 6/7 = 0.857 |

What the table shows: at L2 the gain on (2,4) is a kept mechanism, so reuse large ties reuse small, and the new
edits lose by their small K\*. At L3 and L4 the same gain changes the whole layer's or block's write at every position,
so reuse large drops below the new edits. At L1 the thin edit's many slightly-moved neurons pull R up. The numbers
depend on the made-up cosines. The directions in P1 to P3 depend only on two things: the head-level gain grading 1,
and K\*(new few) being smaller than K\*(reuse small).

## 4. Measurement choices for the lead (not definition changes)

a. **Mean ablation.** A variable outside K takes its position-wise mean write over D, from F's unablated run.
   Consequence: a constant write (E137 Always on) is never needed. A mean over all positions would put it in K\*.
b. **Settings for s_u.** Main reading: every position of D, in both M's run and F's run, as E208 did. Side reading:
   the read positions only. The cue edits act at 1 of 41 positions, so the side reading's gaps are about 40 times
   larger (§8, question 1).
c. **Coarse s_u.** A layer's or block's s_u is 1 − mean over settings of (1 − cos²) between F's and M's total write
   of that variable on the same setting. A block's write is its attention write plus its MLP's write on the
   post-attention residual. Both-zero counts 1 and one-zero counts 0, as in E208. This is the squared-cosine
   declaration applied to the variable's own value. I did not use an energy-weighted mean of member grades, because
   that grades the members' mechanisms, not the variable's. Grafts and pieces score 0 by ask 51.
d. **Behaviour and its bar.** B is F's full next-token distribution at the read positions. K\* is the smallest K with
   KL(F ‖ F_K) ≤ (1 − α) KL(F ‖ F_∅), where F_∅ has every variable mean-ablated (T2's form). F_∅'s logits are the
   unembedding's mean, so KL(F ‖ F_∅) is the same at every level.
e. **Changed set C and grafts at coarse levels.** C holds the variables any of whose weights differ from M's, plus
   every graft or piece variable. A graft inside a coarse variable makes that variable changed, graded by (c).
f. **Band.** 3 × the largest, over the four edits, of three quantities: the sd of R over search orders (0 where
   exhaustive), |R on half 1 − R on half 2| (a seeded split, K\* searched per half), and R's range over tied minimal
   K\*. Pass also needs f > 0 for both reuse edits.
g. **Search** (§5). L2 is greedy plus a polish, not proved minimal. L1's first order is by attribution.

## 5. The search

- **The KL table.** For a set K, take the per-window KL(F ‖ F_K) at the read positions. One pass serves α ∈ {0.8,
  0.9, 0.95}, full D, the restricted D (the half of D's windows with the largest KL(F ‖ M)) and the halves, wherever
  the search is a table.
- **L4 and L3, exhaustive.** Every subset: 64 and 1,024, or 128 and 2,048 with a piece. K\* is the smallest size that
  meets the bar. Ties are all listed, the lowest-KL tie is reported, and the range enters the band.
- **L2, greedy plus polish.** Backward elimination (`rule_retention_fixed.eliminate`, chunk 1) along five seeded
  orders. Then polish: drop any one variable, then swap any two for one outside, repeating until no move applies.
  Each variant (α, D, half) is searched separately. The edits and the E201 graft get every variant; the other items
  get one search at α = 0.9 on all of D.
- **L1, greedy in chunks.** `eliminate`, chunk 256, as E208. Order 0 is the ascending attribution score
  |⟨mean − actual write, ∂KL/∂write⟩| summed over D, from one backward pass. Orders 1 to 4 are seeded random. The edits
  get 5 orders plus α 0.8 and 0.95, restricted D and the two halves: 10 searches each. The other items get order 0 at
  α = 0.9 only. That is about 55 searches at about 2 min each (E208: 110 s per order on an H100).

## 6. Code spec for the writer

- **`src/commute/kstar.py`** (new, about 220 lines). It is built on `commute.rho`'s `Pair`, `Net`, `Layer` and
  `changed`, which are called and not changed.
  - `Level`: a name, plus a list of variables, each a list of rho parts or the tag `"piece"`.
  - `levels(p, piece: bool) -> dict[str, Level]`.
  - `means(p, toks, level, piece) -> Means`: position-wise mean writes in F's run.
  - `forward(p, toks, level, keep: set[int], mu, piece) -> logits`: variables outside `keep` write their means. The
    function adds the piece's write at block 2's input when the piece is kept, and its mean otherwise.
  - `kl_rows(...)`: per-window KL.
  - `exhaustive(...)`, `greedy(...)` (calls `rule_retention_fixed.eliminate`) and `polish(...)`.
  - `grades(p, batches, level, at) -> dict[int, float]`: s_u per §4c, both runs, double precision, and 0 for lacked
    or piece variables.
  - `summary(K, s, C) -> {R, f, rho_prime, remainder_share}`, which asserts R = 1 − f(1 − ρ′) to 1e-9 when f > 0.
  - `Piece = Callable[[list[Tensor], Tensor], Tensor]`, taking the residuals so far and the tokens.
- **`src/commute/known_edits.py`** (new, about 150 lines): P1.31 appendix §2 and §4, unchanged in content.
- **`experiments/kstar_levels/run.py`** (about 300 lines).
  - It has a frozen `Config` dataclass whose `__post_init__` asserts the sizes agree: the levels known, the αs sorted
    with 0.9 in them, eight cues, and the graft roles known. It is built with
    `commute.jobs.config_from_settings(Config, settings)`.
  - Items come from E208's and E195's builders through `load_experiment`. E137's four pieces are rebuilt as one-line
    `Piece`s from `pieces.pt`, and a stage 0 gate checks each against E137's `piece_edit` through `steering.Net`
    within 1e-4.
  - CLI: `--levels`, `--items`. Each item and level writes `results/items/<item>_<level>.json`. The per-variable s
    and K\* go to `results/large/<item>_<level>.npz` (L1 only, about 0.3 MB each). A rerun skips what is written, and
    the script ends with `commute.jobs.hard_exit`.
- Lint with `make lint FILES=...`. Each edited function is re-read, and none is done until it has run once on a GPU.

## 7. Edge cases, run against each filed rule before filing

- **Empty K\*.** K = ∅ gives KL = KL_∅ > (1 − α) KL_∅ whenever KL_∅ > 0, so it cannot meet the bar. KL_∅ ≤ kl_floor
  means there is no behaviour: a stop for the edits, and "no reading" for the other items.
- **One-variable K\*.** R = s_u, and f is 0 or 1. The rules compare R values, so they still apply.
- **f = 0.** ρ′ is null, R = 1, and the identity check is skipped. The pass rule needs f > 0 for both reuse edits, so
  a reuse edit whose changed variable leaves K\* cannot pass by reading 1.
- **Zero denominators in s_u.** Both writes zero counts 1, and one zero counts 0, by E208's convention. A band of 0 is
  possible only if every spread is exactly 0, and then "greater than the band" needs a strictly positive gap.
- **Ties.** Tied minimal K\* go into the band through their R range. The pass rule and P3 are strict (> band). P4's
  "below by more than the band" is strict.
- **Do-nothing against the bar.** It is not in the pass rule. If a new edit silently built as do-nothing, it would read
  R = 1, the top value, so the pass would fail rather than falsely pass, and stage 1's build checks stop it anyway.
  All four edits equal gives a gap of 0, which is not a pass.
- **P1.** If thin's changed neurons all leave K\*, R(thin) = 1 ≥ 0.9, and the comparison with few is still defined.
- **P4.** If C leaves K\* at L1, R = 1 still holds. The s_C = 1 check reads C directly in stage 0, not through K\*.
- **P5 and P6.** A graft or appended copy outside K\* gives R = 1. That breaks the prediction, which is written to fail
  in that case, not to fall through.
- **P7.** Not read if P2 fails. The restricted D is not defined for do-nothing, where KL(F ‖ M) = 0, and is not
  applied there.
- **Signs.** cos² is in [0, 1], so s_u and R are in [0, 1]. No clipping is needed.

## 8. Possible definition questions (for the user, not decided here)

1. **Which input settings a mechanism is compared on.** Is it every position of every window in D (read here, as
   E208 did), or only the positions where the behaviour is read? On these edits that choice changes the reuse and new
   gaps about 40-fold. The side reading reports the read-position form.
2. **A graft inside one of M's coarse variables.** Under ask 51 a variable M lacks scores 0. At a level where the
   graft sits inside one of M's variables, is that variable's mechanism changed and graded (planned here)? Or does the
   graft still count as a lacked part at s = 0? And are E137's pieces, outside every unit, the coverage condition's
   rest-of-network remainder? That is assumed here, and it gives s = 0 either way.

## 9. Pre-mortem (ungated: about $10, under 6 h)

- **Recipe.** Nothing is trained. The risks are P1.31's build risks, and its checks are kept.
- **Precision.** fp32 forwards and fp64 grades, as E208. Gates: K = all gives F within 1e-3, and K = ∅ gives the mean
  logits exactly.
- **Memory.** The 67M with 512 windows of 41 tokens fits an L40. The per-window KL table at L3 is 2,048 × 512 floats.
- **Time.** The L2 polish is capped at 2,000 evaluations per search, and a cap that is hit is logged. L1 estimates its total from
  its first search, and an estimate over 3 h stops it.
- **Saved and downloads.** Each item's file is downloaded from the pod as it is written. The pod is terminated once
  the last file's sha256 matches.
- **Code never run on a GPU:** `kstar.py`, `known_edits.py` and `run.py`. Stage 0 is the smoke.

## 10. Not in this plan

T3 (the trained pairs), T5 (VPD), a weighting of R, any band or cut-off added to the definition, and E187's
size-matched pairs.
