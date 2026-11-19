# How far is the working definition of reuse from what the experiments measured?

*2026-11-18, audit for definition-lead. Read-only: nothing was run or trained, and no wording in the definition, the
log, `GOALS.md` or `decisions.md` was changed. The definition audited is the one in `decisions.md`'s last two rows
("Exact featurization with a mandatory remainder" and "The variables F needs are every variable the behavior runs
through") and in Part 1 of `reports/drafts/2026-11-18_draft_sec4-bullets-ask48b.md`. The only numbers computed here
are summaries of E208's saved per-variable scores, taken with `jq` from
`experiments/rho_saved_items/results/items/*.json`.*

No experiment so far has measured the reuse score the definition now names. Every past measurement either asks how
much of the fine-tune's change from the original survives some edit, or finds the smallest set of parts that carries
that change. The definition instead finds the smallest circuit that carries the whole behaviour, whether or not its
parts changed. It then averages over that circuit whether each part still has the original's mechanism. One reading
transfers almost exactly: at the neuron level, the saved per-neuron scores bound the definition's score between 0.88
and 1 on every trained fine-tune, whatever circuit is found. Nothing on file says whether the definition puts the side
built to reuse above the side built to be new, on the built items or on the trained pairs. Five tests on saved states,
none costing more than about $15 of GPU time, would settle that.

## The definition, in the form used below

Write N₀ for the original model, N₁ for the fine-tune and U for the variables of one fixed featurization taken from N₀.
The units case uses neurons, heads, MLP layers or blocks, and has no remainder. The VPD case uses component contributions
plus a remainder Δx per matrix.

- K* is a smallest K ⊆ U such that mean-ablating every variable outside K in Ñ₁ keeps
  KL(N₁ ‖ Ñ₁,ᵢK) ≤ (1 − α) · KL(N₁ ‖ Ñ₁,ᵢU). The reference on the right is **every variable mean-ablated**, not the
  original model. So K* is the circuit for the fine-tune's whole output on D, not the circuit for its change.
- R = (1/|K*|) Σ_{u∈K*} s_u, with s_u grading F̃⁽¹⁾_u against F̃⁽⁰⁾_u on the parent settings that occur in either
  network's run on D.

One identity follows, and it organises everything below. Let C be the variables whose mechanism changed. Let
f = |K* ∩ C| / |K*| be the changed fraction of the circuit, and ρ′ the mean s_u over K* ∩ C. Then

  **R = 1 − f · (1 − ρ′).**

Every earlier reading estimates something like ρ′, a grade of the changed parts, or a behavioural share of the change.
**No earlier reading estimates f.** Yet f is what separates a small new mechanism plugged into a large old circuit
(f small, so R stays high) from a broad rewrite (f near 1).

Three consequences:
- For full-weight fine-tunes at any unit level every unit changes, so C = U and f = 1. R is then the plain mean of
  s_u over K*.
- For hand-built edits and grafts, C is a handful of units, so R is set mostly by |K*|.
- In every earlier search, K ranged over changed parts only, scored against KL(F ‖ M). That is the change's
  circuit, which is a different set from K*.

## The measurements against the definition

"Value" means a quantity read from activations or outputs. "Mechanism" means a function of a variable's parents,
compared on the same parent settings.

| measurement | quantity | relation to R | agrees with R when | saved states usable for R? |
|---|---|---|---|---|
| Second wording, re-weighting share G and rewiring share H (E137, E139, E142) | At one site a 4-dimensional subspace z is found from both networks. G = share of the output gap to F closed by M with z replaced by an affine map of z's own value; H = the same with a map from all earlier activations | **Unrelated in what it separates.** It acts on values, uses a subspace fitted with F (not fixed from N₀), and has no K. Under R, re-weighting and rewiring are both one variable with a changed mechanism; R tells them apart only through s_u and K* | The change sits in one variable, and G and s_u happen to rank alike. **Disagrees on** E137's Always on (a constant push, labelled reuse), whose K* may be tiny because a push needs no upstream input. f is then large and R low, below a lookup that runs through the embeddings | Only E137's hand-built pieces, `experiments/reuse_variable_test/results/large/pieces.pt` (2.4 MB), and E139's imitations (268 MB each). E142's old-task states are in the E136 inventory. No variable-level scores were saved |
| Fourth wording, closure ranking (E180, E182, E183, E184; curves E191, E192) | closure(I) = 1 − KL(F ‖ M_I) / KL(F ‖ M) for interventions I on M's neurons and principal directions; rung = the simplest class (0 to 4) whose best I closes 0.9 | **Unrelated, opposite direction.** It edits M's *values* until M matches F; R grades F's *mechanisms* against M's. Its K is the smallest set of edits that explains the change, inside C. Nothing in it estimates f | Changes are constant shifts of a few units, and K* has a fixed size across items. **Disagrees on** a one-unit lookup: one changed variable (R near 1 when K* is large) but rung 4, since no list intervention closes it | E184's items are rebuilt in code from `simplest_intervention/results/large/pcs.pt` (1 MB). Its five trained pairs are E196's states (below) |
| Causal-rule retention ρ, level A (E193, E194, E195, E196) | r_u = aggregate cos² of F's unit write against its *assigned* M partner (linear_sum_assignment) over M's inputs only. K = smallest set of changed units kept at F's rule, all others reverted to the partner's rule, closing τ = 0.9 of KL(F ‖ M). ρ = mean r_u over K | **A relative of ρ′, not R.** Its s_u is a mechanism comparison, but the re-pairing breaks "the same variable in both", settings come from M only, the K is the change's circuit under base-mechanism swaps rather than mean ablation, and f is never read | f = 1 (full fine-tunes) and the assignment is the identity. Then R ≈ ρ up to K choice. **Disagrees on** E195's x2/x10 (ρ = 1 by the gain-blind cos²) whenever a grade is gain-aware, and on any built edit where f is small | Only per-item summaries were saved. The states are the same as E196's (below) and E195's `rule_retention_fixed/results/large/rebuilt_mlp.pt` (37.8 MB) |
| ρ as the theory note wrote it, neurons and heads (E208) | q_u = 1 − mean_v (1 − cos²(f_F(v), f_M(v))) over settings from both runs (identity pairing). K = smallest changed set kept at F's mechanism, every other changed part at M's mechanism, closing 0.9 of KL(F ‖ M); ρ = mean q_u over K | **The s_u is R's s_u at the neuron-and-head featurization** with d = 1 − cos² declared; only K differs (change's circuit, base-mechanism swaps, against KL(F‖M); K* is the behaviour's circuit under mean ablation). For full fine-tunes f = 1, so R = mean q_u over K*, **bounded by the saved scores** | Always for the level of R at neurons: R lies between the smallest and largest q_u. Its *ordering* of pairs (gaps of 6.5e-5) does not carry over, since swapping K moves R by more. **Disagrees on** built edits and grafts, where q_u = 0 on a few units and R depends on |K*| | **Yes, directly.** `rho_saved_items/results/large/parts/<item>.npz` (24 files, 8.4 MB) hold q_u for all 12,312 changed parts per trained fine-tune, under four grades, plus the K of each search order. Once a K* is found, R is a CPU average |
| Held-model survival S (E197, E201, E205, E206) | H = M with each site's output refitted (per-site least-squares map, or one gain per unit) to F's write on M's inputs; S = 1 − KL(F ‖ H) / KL(F ‖ M) | **Unrelated as a formula; a behavioural proxy for 1 − f(1−ρ′) only in spirit.** It gives *every* variable M's mechanism at once (K = ∅ for F's mechanisms), lets a fitted linear map re-coordinatise (R does not allow this), and weights by behaviour, not by variable count | The change is spread evenly, so behaviour lost tracks the share of variables changed. **Disagrees on** grafts. S reads a last-layer head graft 0.13. Under R, if the graft's K* is {embedding, graft, unembedding}, R = 2/3 (graft s = 0). If K* also holds the 3,000 or more units E208 needed for ordinary fine-tunes, R is about 0.9997, above most trained fine-tunes. It also disagrees on the memorised fact: S 0.65, while the draft itself says the fact reads mostly reused under R | Grafts: `graft_negative/results/large/graft.pt` and `graft_sites/results/large/*.pt` (1.2 to 4.7 MB each). Fitted maps are saved under `held_model_retention/results/large/maps` and `units`, but are not needed for R |
| P1.31 known-answer test, planned (`reports/plans/2026-11-18_plan_known-answer-levels.md`) | r_u = 1 − KL(F ‖ F with u at M's mechanism) / KL(F ‖ M); K from E208's search over changed parts; R = mean r_u over K, at neuron, head, MLP-layer and block levels | **Written for the superseded clause** (reading 1: changed parts only). Ask 48 adopted reading 2 the same day, and `decisions.md` says the plan "must be redone". Its r_u is behavioural (a share of the change), not a mechanism grade s_u. Its predictions P1 to P3 follow from reading 1 only | Under the current K*, the edits *should* separate. Reuse-small's K* holds the unchanged induction heads (s = 1), so f is small. New-few's lookup path writes straight to the output, so K* is small and f larger | Nothing run. The edits are about 2 MB, built in code. The E196 states for run 2 are on disk |
| Re-gating share r (compare-lead; E133, E187, E200, E211) | r = (L(C₀,G₁) − L(C₀,G₀)) / (T₁ − T₀): share of the trained answer's log-odds gain reproduced by VPD's masked model with the original components and the gates computed on the fine-tuned activations | **Unrelated.** The definition's VPD featurized model has no gate: F̃_c = u_c v_cᵀx always on. Γ's switching is not a mechanism of any variable in Ñ. r is a behavioural attribution inside VPD's sparse stand-in. Under a least-squares split of a dense δW nearly every component's mechanism changes, so f ≈ 1 and R at VPD is a mean grade, much like the neuron case | Possibly when the change enters through a few early components and later components are unchanged and switch on downstream. **Disagrees on** E200's built kinds (r reads built rewiring like built re-weighting) and on Qwen (r is 0.99 on every state) | δW per state is saved (E136 inventory; regating_table etc., 113 MB each), and so is the decomposition (`artifacts/vpd`, 3.0 GB). R at VPD needs a declared split and a new K* search |
| Outside share (Terms, E2 to E9 and later) | E‖(I − P_U) δW a₂‖² / E‖δW a‖², the input-weighted energy outside the components | **Part of the remainder's grade.** It measures δW_out x against δW x, which bounds how far the remainder's mechanism moved. It is not R and says nothing about K* | It describes the remainder's s_e when e ∈ K* | δW saved, as above |
| Steering (E135, E143, E146) | Closure of M with the fine-tune's mean activation change added at one block input, strength searched | **Unrelated (values).** A one-site value shift. Under R that is one variable's mechanism change, weighted 1/|K*| | One-site constant shifts. **Disagrees on** input-dependent changes | Saved δW |
| Neuron share, whole width and half (E98, E143) | Gain share kept when M's MLP output weights (whole width) or M's whole MLP (half) run on F's neuron firing or MLP input | **Closest baseline to a mechanism reading.** "Half" is the behavioural share kept under a soft intervention that gives every MLP layer M's mechanism on F's parent settings. That is s at the MLP-layer level, weighted by behaviour, with K = all MLP layers | MLP layers are in K* and attention changes are small. **Disagrees on** attention-carried changes (E196: reuse sides carry more of their change in attention) | Saved δW |
| Top-set swap, top-neuron overlap (E143, Goodfire readings) | Gain share from swapping top-firing neuron sets; Jaccard overlap of top-240 firing sets before and after | **Unrelated (values).** They compare which neurons fire, not what a neuron computes from its parents | Never in general. Shifted inputs change top sets with unchanged mechanisms, the case R is built to count as reuse | Saved δW |
| Prior, effective rank, update size (E133, E141, E171, E187) | log₁₀ p_M(answer); Roy–Vetterli rank of δW; ‖δW‖ / ‖W‖ | Prior and rank: **unrelated**. Update size: **a likely proxy for R at unit levels.** With f = 1, the deficit 1 − s_u grows with each unit's weight change (E196's gaps followed MLP update size) | Unit-level R on full fine-tunes, small updates. Note that E133 counted update size as "never agrees" because the new side is bigger; under R a bigger update means lower R, i.e. it *agrees* with the construction labels | Saved δW |
| Crosscoder, BatchTopK with latent scaling (E198, E203) | Shared minus fine-tune-specific latent share of a dictionary trained jointly on both models' block-2 inputs | **Unrelated, and breaks "one fixed featurizer".** The dictionary is fitted on F too, and it compares representations (decoder norms), not mechanisms. Closest analogue: the remainder's share of activation energy under an SAE featurization fixed from N₀, which is still not f | Never shown | Crosscoders in `~/commute-data/results-large` per E203 (not opened here) |
| Labels: "made to reuse" and "made to be new" matched pairs (E133, E142, E187, E196, E211) | Construction labels: the reuse side's task is something M half does (copy, case), the new side's something it lacks (cipher, random map, fact) | **A hypothesis about R, never checked.** For full fine-tunes at unit levels the labels predict R(reuse) > R(new), which holds only if the reuse side's per-unit mechanism changes are smaller. That is update size again | Reuse sides change their units less | Five pairs × seeds 0, 1 on disk (E196 list, 57 to 268 MB per state); nine E187 pairs in the E136 inventory |
| Labels: built items (E137 pieces, E184 planted items, E195 x2/x10/copy/rebuilt, E201/E206 grafts, P1.31 edits) | The changed variables are known exactly | **The only items where R can be computed against a known answer**, provided the label is stated in R's terms (f and s_u). Two cases need a declaration first. The x2/x10 edits read s = 1 under cos² and below 1 under a gain-aware grade. Grafts add a variable N₀ lacks, and the units featurization has no remainder to hold it | — | All saved or rebuilt in code (sizes above) |

## Which log claims survive a re-reading under R

**Untouched, because they are not about reuse.** Every Part 1 ("Commute") row of the scoreboard (`sections/claims.html`)
is about splits, faithfulness and Δ. So are the Qwen decomposition rows of Part 4.

**Stand as facts about the measurement, with no evidence about R.**
- Every "the reuse side reads higher" row: G (E142), the re-gating share (E133, E187, E211), the baselines' counts
  (E133), ρ's 8 of 8 (E196), S's 8 of 9 (E205), and the crosscoder (E203).
- Each is true of its own quantity against construction labels. None was read against R, and none estimates f.
- The claims that the share "adds" over baselines are comparisons against those labels, not against R.

**Carry over to R, with evidence from saved states.**
- Saturation at the neuron level. On all 20 trained pair fine-tunes, the per-unit grade at the neuron-and-head
  featurization (d = 1 − cos², settings from both runs) has a minimum of 0.881 to 0.975 and a mean of 0.9994 to
  0.99995. Every unit is changed, so R lies in [0.88, 1] for any K* at all.
- R stays above 0.99 unless K* is made almost entirely of the few worst units.
- E208's own K (3,000 to 8,600 parts) gave ρ within 2e-5 of the all-unit mean. So R at this level is about 0.9994 or
  more for any K* of that size.
- The draft's line "neurons saturate" therefore holds under the current definition, not just under ρ.

**Do not carry over.**
- E208's and E196's orderings of pairs (gaps of 4e-5 to 3e-3) depend on which K is used and are below what a K
  change can move.
- The graft readings (S 0.13 to 0.22 on heads) have no counterpart, since R was never read on a graft. Under the
  current K* a graft may read as mostly reused.
- P1.31's predictions P1 to P3 were derived under the superseded clause.

**Two statements in the draft that do not match the code.** These are flagged here, not edited.
- §H says "The re-gating share is a norm share of δW". By `GOALS.md` → Terms, r is a share of the log-odds gain in
  VPD's masked model. The norm share is the outside share.
- §F writes ρ as a mean "over K*". E208's K was the change's circuit under base-mechanism swaps against KL(F ‖ M),
  not K* as now defined.

## Ranked tests that would connect the experiments to the definition

The tests are ordered by information per dollar. T1 runs on the dev node. T2 to T4 need one new piece of code: a
mean-ablation K* search over the variables of a level, plus a mechanism grade s_u for coarse variables. T2 and T3 share
it. Pod costs assume an H100 SXM at $4.40/h, the rate P1.31's plan quotes. Uploads are priced at 10 MB/s.

1. **T1. Bound R from E208's saved scores** (CPU, `cpu-run`, a few minutes, $0, no GPU).
   - *Computes:* for each of the 24 E208 items, R's range over every possible K*, and the worst R for |K*| = k (the
     mean of the k smallest q_u) as k varies. Also R with K* set to E208's own five K's, read from the npz `K_replace_*`
     arrays. Done for all four saved grades (main, i, iii, b).
   - *Inputs:* `experiments/rho_saved_items/results/large/parts/*.npz` (8.4 MB).
   - *If the old reading tracks R:* R at neurons is at least 0.99 for every trained fine-tune unless k < roughly 50,
     and the five E208 K's give R within 1e-4 of ρ. *If not:* some grade (b, the size-aware one) opens gaps of 0.01 or
     more at plausible k, so the saturation is a property of cos², not of the level.
   - This is the only test that needs no new search.

2. **T2. R on the known-answer built items at four unit levels, under the current K\*.** This redoes P1.31 under
   reading 2.
   - *Computes:* K* by mean ablation over D (α declared, full-ablation reference), then s_u, R, and separately f and
     ρ′. Levels: neuron + head, head + MLP layer, MLP layer + attention layer, block.
   - *Items:* P1.31's four edits (rebuilt in code); E195's x2, x10, copy and rebuilt (`rebuilt_mlp.pt`, 37.8 MB);
     the E201 and E206 grafts (about 10 MB in all); E137's four pieces (`pieces.pt`, 2.4 MB).
   - *Expected if R reads as the user intends:* at every level, reuse-small reads above new-few, because K* holds the
     unchanged induction heads (1,1) and (2,4). Grafts and the rebuilt circuit read below x2/x10. *If not:* |K\*|
     swamps the changed units, all built items read R > 0.99, and only f separates them. That outcome would mean the
     average over K* needs a weighting the definition does not yet have.
   - *Needs first:* a declaration of how a graft, a variable N₀ lacks, enters the units featurization. That is a
     definition question for the user, not a measurement choice.
   - *Cost:* coarse levels are exhaustive and take seconds. The neuron level is a greedy elimination, about 10 min per
     item per order on an H100, after E208's timing. About 2 h, roughly $10 on a pod (inputs about 50 MB). The coarse
     levels alone fit a `debug` L40 at $0.

3. **T3. R on the trained matched pairs.**
   - *Computes:* K* by mean ablation at the same four levels on E196's five pairs (seeds 0 and 1, 20 states). At the
     neuron + head level s_u is already saved, so only K* is new. Adding E187's nine built pairs is optional.
   - *Inputs:* the E196 state list in the P1.31 appendix §5, about 2.6 GB. That is about 4.5 min to upload.
   - *Expected if the construction labels track R:* reuse side above new side on every pair, beyond the seed spread,
     at the block level at least. *If they track only update size:* gaps at unit levels follow ‖δW‖, as E196 found,
     and vanish after E187's size matching.
   - Also reports the overlap between K* and E208's K. That separates "K choice matters" from "grade matters".
   - *Cost:* about 2 to 3 h on one H100, roughly $12, or two pods in parallel.

4. **T4. R against each old reading on the same items** (CPU after T2 and T3, $0).
   - *Computes:* per item set, sign agreement and Spearman correlation of R (each level) with ρ (E196, E208), S
     (E197, E205, E206), the re-gating share (E187), the fourth-wording rung and closure (E184 and E192 on the same five
     pairs), G (E142, where states coincide), the crosscoder (E203), steering and the neuron readings (E133), and update
     size.
   - *Expected if a reading tracks R:* the same sign on all counted pairs and a rank correlation of 0.7 or more. Nine
     pairs reach about 0.7 by chance one time in twenty, so only the five-pair-by-two-seed and built-item sets give
     power. *If not:* correlations near zero, as the share against S already showed (0.13 and −0.27,
     `2026-11-18_analysis_regating-vs-definition.md`).
   - Report f beside every point. A reading that tracks f but not R, or the reverse, is the finding.

5. **T5. R at the VPD featurization against the re-gating share** (E187's nine pairs).
   - *Computes:* the least-squares split declared as the split rule. Variables are the component contributions per
     matrix plus one remainder per matrix. s_c is graded on the settings that occur, and K* is found by mean ablation
     over those variables. Then R and the remainder's share of K*, set beside r.
   - *Inputs:* `artifacts/vpd` (3.0 GB, already on any pod image that ran E187) plus the E136 δW inventory.
   - *Expected if r tracks R:* the same order on the nine pairs. *If not:* R saturates as at neurons, because a dense
     δW gives every component a small δW_c (f ≈ 1), and r's separation has no counterpart in R.
   - This is the test that decides whether §5 can ever say the share reads the definition.
   - *Cost:* the search runs over thousands of components. Expect 3 to 6 h on an H100, $15 to $30. Under the $60 and
     6 h gate if the first-order timing holds, so time it on a `debug` L40 first.

6. **T6. K* sensitivity to α and D** (folded into T2 and T3; no extra GPU).
   - *Computes:* R at α ∈ {0.8, 0.9, 0.95}, and with D restricted to the windows where the behaviour departs most.
   - *Expected:* if R moves by more than the reuse-against-new gap, the score is set by declared choices, not by the
     fine-tune.

7. **T7. The graft and gain declarations** ($0, a question to the user before T2).
   - A graft adds a variable that N₀ lacks, and the units case has no remainder to absorb it.
   - A pure gain change reads s = 1 under cos².
   - Both decide the built items' expected R, so both must be fixed before T2 files predictions.

## Not checked

- E211's description of the share ("the fraction of the gain still present when the change is passed through the
  released decomposition's subcomponents") reads like the stand-in ratio or a split, not the re-gating share r. I did
  not open its code.
- The E136 inventory's paths and sizes for E187's nine pairs, and where E203's crosscoders are stored.
- Whether E142's old-task states coincide with any E196 or E187 state, which T4 needs for G.
- Whether E208's "embed" and "unembed" were unchanged on the 12,312-part items. 12,312 is 12,288 neurons plus 24 heads,
  which suggests so, and the fact and twin items have 12,314.
- The mechanism grade at coarse levels (a block's s_u needs a distance on a residual write), and whether the
  definition's 𝒱_u for a coarse variable includes settings the hybrid alone reaches.
- The arithmetic of the neuron-level search time under mean ablation. It is projected from E208's 110 s per order,
  which ran over changed parts against KL(F ‖ M), not measured.
- The Qwen readings (E132, E144, the Qwen half of E187). The same identity R = 1 − f(1 − ρ′) applies, but no
  Qwen state was inspected.
- No independent check of this audit.
