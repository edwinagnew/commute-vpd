# Review: why the LoRA commute test failed, and which fixes are worth running

*2026-10-19, coding agent, for the user's question "is the read-side arm a possible way of fixing it; if not, consider
alternatives". Status: design checks awaiting the user's confirmation. Nothing here has been run.*

The LoRA commute test put a LoRA on the paper's matrix and found that lifting it onto the released subcomponents left
a third to two fifths of the behaviour in the residual. We traced that to the matrix rather than to the LoRA: the
decomposition reads only a fifth of that matrix's input energy, so even the full fine-tune's own update there is a
third inexpressible, and the LoRA's random read side adds only a little on top. For the goal this means the question
"is a LoRA easier to commute" was asked at the one place where almost nothing commutes, and the fix is where the LoRA
sits, not its rank or its initialisation.

## 1. Two questions from the user

**"Can't you just take the final BA and decompose it?"** Yes, and both ways of doing so are already settled.

- *As a change to the existing subcomponents.* That is the lift (GOALS's Φ for a finished weight change), and it ran
  on every LoRA arm. `lift.py` writes BA as the smallest first-order change (dU, dV) of the alive (U_A, V_A); what it
  cannot write is the obstruction O(BA) = P_U^⊥ BA P_V^⊥, which vanishes only when A's rows lie in the span of the
  alive read vectors or B's columns in the span of the alive write vectors. At `h.2.mlp.down_proj` the alive read
  vectors cover 21% of the input energy and the alive write vectors span about half the output space, so a random
  BA is 42% inexpressible there, and the lift sends that part to Δ. The decomposition of BA exists; it fails reading
  A's clause "Δ does not absorb δW".
- *As new subcomponents.* Appending BA as r new rank-1 pieces with their own gate is definition (iii) in
  `experiments/TRIED.md` (line 29), which the user dropped at `/design` on 2026-10-15: it assumes every update is a
  new capability (the thing under test) and its gate is either a trained net (not simple) or always on (Δ absorbing
  δW under another name). Nothing in round 4 changes that reasoning.

**"Does this contradict the per-step versus full-run decision?"** No. The Euclidean lift with the W₀ bases held fixed
is linear in the update, so lifting each step and summing gives exactly the one-shot lift of the sum: the same (U, V),
the same Δ. The LoRA's inexpressible share is a property of each step's read direction and is the same per step as
for the run. The only per-step variant that differs recomputes the bases from the lifted (U, V) after each step
(section 3.3 below); that variant can change the LoRA numbers and is the one the addendum's recommendation should
mean. For a LoRA handed over as a finished BA, the run is the only object there is (GOALS: "for a LoRA handed over as
BA, only the lift applies"); the step framing applies when the path is available, which it is for our own LoRAs.

## 2. Why the LoRA arms failed

Per-matrix inexpressible shares of the full-weight fine-tune at step 10 (`lora_commute.json`, `ft.steps.10`). E% is
the matrix's share of the fine-tune's input-weighted update energy; read_out the share of that matrix's input energy
outside the alive read span; null the share of an isotropic update that is inexpressible; ω the fine-tune's own share.

| matrix | E% | read_out | null | ω (ft) |
|---|---|---|---|---|
| h.3.mlp.c_fc | 22.0 | 0.000 | 0.000 | 0.000 |
| h.2.mlp.c_fc | 18.5 | 0.530 | 0.479 | 0.096 |
| h.1.mlp.c_fc | 13.4 | 0.586 | 0.538 | 0.177 |
| h.0.mlp.c_fc | 7.7 | 0.000 | 0.000 | 0.000 |
| h.3.mlp.down_proj | 5.5 | 0.463 | 0.000 | 0.000 |
| h.3.attn.q_proj | 4.2 | 0.882 | 0.841 | 0.551 |
| **h.2.mlp.down_proj** (the LoRA site) | **3.3** | **0.788** | **0.421** | **0.362** |
| h.3.attn.v_proj | 3.3 | 0.680 | 0.460 | 0.259 |
| h.3.attn.k_proj | 3.1 | 0.837 | 0.771 | 0.587 |
| h.0.mlp.down_proj | 2.9 | 0.621 | 0.000 | 0.000 |
| h.2.attn.q_proj | 2.7 | 0.780 | 0.687 | 0.383 |
| h.1.mlp.down_proj | 2.5 | 0.330 | 0.239 | 0.564 |
| h.2.attn.k_proj | 2.2 | 0.711 | 0.556 | 0.307 |
| h.2.attn.v_proj | 1.8 | 0.377 | 0.128 | 0.045 |
| remaining 10 attention matrices | 7.0 | 0.26–0.93 | 0.08–0.91 | 0.02–0.80 |

Four matrices (null 0) are saturated: every update to them is expressible by a first-order change of the alive
subcomponents, random or not. They carry 38% of the fine-tune's energy; two more (h.2 and h.1 `c_fc`, ω 0.10 and
0.18) carry another 32%. That is why the fine-tune's pooled share is 0.27 and its lift leaves e_Δ 0.085.

The paper's LoRA site carries 3.3% of the fine-tune's energy and is one of the worst-covered matrices. Three facts
follow, all from the round-4 numbers:

1. **At the paper's site nothing commutes, random or data-shaped.** There the isotropic null is 0.42, the LoRA
   reads 0.41 (step 10), and the fine-tune's own gradient-shaped update reads 0.36: the random read side costs
   about 0.05. By step 100, after A has moved (|cos| with its init 0.89), the LoRA reads 0.30, below the
   fine-tune's 0.32 at the same step. On the matrices where the fine-tune's energy actually sits it is the
   reverse: `h.2.mlp.c_fc` reads 0.10 against a null of 0.48 and `h.1.mlp.c_fc` 0.18 against 0.54, so there a
   data-shaped read side is worth about 0.4 and a random A would land near the null. The report's "A stays near
   its random init" is the minor term at the paper's site and would be the whole story on a well-read matrix.
2. **Rank is irrelevant** (the report's own finding): 0.38–0.43 at ranks 1 to 32, because a random read direction at
   this site is inexpressible at any rank.
3. **Expressible is not faithful.** The one-matrix pull-backs `pb_r1`–`pb_r32` are exactly expressible by construction
   (e_Δ 0.004–0.011) and still read Pile PGD 2.1–4.1 against the floor of 1.11; `pb_full` 1.39; only `pb_all` (all
   24 matrices) sits at 1.10. The fine-tune's lift, 73% expressible, reads 5.4–7.3. So making the LoRA expressible
   would at most move e_Δ, and the second half of reading A (VPD's adversarial check) is failed by every one-matrix
   state we have, lifted or trained.

So the round-4 comparison C1/C2 (one-site LoRA against a 24-matrix fine-tune) confounded the method with the site.
The user's instinct was not tested on a fair footing.

## 3. Candidate fixes, each through the design check

Three forks ran `/design` on one candidate each (checks only; nothing built). Summaries; the forks' full answers are
condensed here.

### 3.1 Read-side LoRA (the report's and backlog's "obvious next arm"): nothing new

A initialised in the span of the site's alive read vectors (frozen, trained, or trained first), B from zero, lifted.

- **Goal.** Weak version, indirectly: it asks whether a decomposition-aware LoRA commutes, not whether a
  post-training update does. Departs from decision 16 (LoRA as the canonical update: nobody ships a LoRA initialised
  from a decomposition).
- **Nearest tried.** `pb_r` (TRIED row 7). With A frozen in span(V_A), δW = (B Cᵀ) V_Aᵀ is the write-only half of
  `pullback.Offsets`' rank-r space; `pb_r1`–`pb_r32` are exactly expressible (e_Δ 0.004–0.011), learned, and read
  Pile PGD 2.1–4.1 on this matrix. Expressible-but-fragile on one matrix is already measured with a strictly larger
  space. With A trainable it starts there and drifts; the only new quantity is how fast (ω and |cos| per checkpoint),
  a one-minute trajectory measurement, not an arm, and section 2 already bounds it (the fine-tune's own read side
  at this site ends 0.36 outside the span).
- **Expected numbers.** ω at the site 0 at init, ≤ 0.1 at the scored step (do-nothing 0.41, `pb_r8` 0); e_Δ ≤ 0.05
  (do-nothing 1.0, `pb_r8` 0.007); Pile PGD 1.4–4 (W₀ + 3 floors 1.11, `pb_r8` 4.06). Only PGD ≤ 1.11 with e_Δ ≤ 0.02
  would change anything, and `pb_r1` at 2.13 with the superset space makes that unlikely.
- **Verdict.** Patch; nothing new as an arm. Cost would be ≈ 10 min of GPU.

### 3.2 Matched-matrix LoRA (rank 8 on all 24 decomposed matrices): run

The standard "all linear layers" placement, rank 8, arm A's loss, lifted with round 2's 24-matrix lift at the first
checkpoint with P(o) ≥ 0.9; do-nothing and lifted through the same harness (route split, pooled ω, Pile ratios
against the 3-seed W₀ floor); seed-1 repeat for the floor m; the fine-tune's lift rerun in the same job as the
positive control.

- **Goal.** Weak version with LoRA as the canonical update (decision 16): does a LoRA placed the way LoRAs are placed
  commute as well as a full-weight update? Round 4 could not say, because the site differed between its arms.
- **Nearest tried.** Rows 2 (ft lift) and 7 (one-site LoRA). Neither separates method from site; this is the first
  fair C1/C2. The one-site data-shaped read side (higher lr on A, or scoring at the plateau) is bounded by the
  fine-tune's own 0.32–0.36 at that site and is not worth an arm.
- **Patch or step.** A patch on round 4's confound, but it is the comparison round 4 was for.
- **Decisions.** Relies on 16 and 22; same lift, same harness; no departure.
- **What changes next.** Lift PGD ≥ 5 (fine-tune-like): the lift's fragility is a property of expressible
  24-matrix updates and the LoRA framing closes; the per-step lift (3.3) is the last lift candidate. Lift PGD ≤ 2
  with e_Δ ≈ 0.1: a LoRA update is expressible and survives where the fine-tune's does not, which reopens C2 and
  asks what differs (rank limit per matrix). Either outcome is read against the fine-tune's lift, so the L40 rerun
  that closes the G4 miss (5.4 vs 7.3) must be in the same job.
- **Numbers and expectations** (ratios to W₀; do-nothing = δW in Δ; control = ft lift step 10):
  P(o) ≥ 0.9 at the scored step (gate); pooled ω bracketed, 0.27 if A becomes data-shaped (the fine-tune's value)
  to about 0.5 if A stays random (0 on the four saturated matrices, near the null on the rest); lift e_Δ 0.1 to
  about 0.4 by the same bracket (ft 0.085, one-site 0.33–0.43, do-nothing 1.0); lift Pile PGD the open number,
  1.4–2 or 5–7 (floor 1.11); do-nothing PGD 1.4–2.5 (one-site LoRA 1.39, ft 2.49); A's |cos| with its init per
  matrix, reported. The ω bracket is what separates init from site, which round 4 could not do: a result near 0.5
  would say the random read side is the obstacle on the well-read matrices, and on those matrices an "A trained
  first" LoRA is no longer bounded by `pb_r` (section 3.1's verdict is about the paper's site only).
- **Cost.** Two LoRA trainings, the ft rerun, three W₀ seeds, the Pile harness on four states: about 15 min on an
  L40, free on the cluster (8 L40s, all in use by three jobs at the time of writing; queue has pending jobs, so
  RunPod at under $1 is the fallback). `LoraConfig.modules` already takes a tuple; a short new run script, not an
  edit of the finished `run.py`.
- **Verdict.** Run, with the ft-lift rerun folded in as the control.

### 3.3 Per-step lift (the addendum's recommendation): run with one change

Apply the lift to each training step with the state carried forward, on the LoRA r8 path (100 steps) and the
fine-tune path (steps 1–10), and watch the error grow.

- **Resolved first.** `euclidean_lift` is X = K⁺M with K built once from W₀'s bases (`lift.bases` in `load_all`),
  and K⁺ and the obstruction are linear in M, so with the bases held fixed the sum of per-step lifts is the one-shot
  lift of the sum: identical (U, V), Δ and every number. The variants review's (b) is written with K at W₀, i.e.
  this fixed-bases variant, which is why it read as "the lift with a worse preimage"; the addendum does not say
  which bases it means. Only the variant that recomputes the bases from the lifted (U, V) after each step is a
  different map. For the LoRA, step 1's lifted dV picks up components along A₀, so the alive read span rotates
  towards A and the later dB·A steps become expressible, while the first steps' obstruction stays in Δ.
- **Goal.** Direct: GOALS's "small gradient updates" and "interpretable continuously along training", the user's own
  framing of Φ as acting on a step (addendum). A step towards the goal, not a patch; it applies to both paths.
- **Nearest tried.** Row 2's one-shot lift at steps 1–10 (PGD 1.14 → 5.41) and row 7's one-shot LoRA lifts. What it
  learns that they could not: whether the lift's fragility and the LoRA's inexpressible share belong to the one-shot
  framing (bases left at W₀, second-order term in Δ) or to the lift family.
- **Decisions.** Inside reading A (Δ takes only each step's obstruction, so "Δ not absorbing δW" is measured),
  inside "simple" and "Not the goal" (no refit, Γ frozen). Departs from the variants review's "drop (b)", which the
  addendum already said does not carry over.
- **What changes next.** Fine-tune path: accumulated-state Pile PGD at step 10 ≤ 1.5 (one-shot 5.41) means the
  fragility was the framing and the lift is a live Φ beside the pull-back; still > 1.5 means it is intrinsic to the
  lift family and the pull-back is the only map. LoRA path: per-step ω falling below 0.2 within ~10 steps with e_Δ
  < 0.15 at step 100 means round 4's LoRA result was a framing artefact; ω staying near 0.42 means the site's
  coverage is the obstacle in any framing.
- **Numbers and expectations.** ω of each step against the current bases (fixed-bases reference 0.42 LoRA, 0.36 ft
  at the site; expect 0.42 at step 1 then falling); accumulated Δ change and e_Δ at the scored step (one-shot 0.40
  LoRA r8, 0.085 ft; do-nothing 1.0); Pile PGD of the accumulated state at steps 1, 3, 10 (ft) and 10, 30, 100 (LoRA)
  against one-shot 1.14/5.41 and 2.02, do-nothing 2.2–2.5 and 1.39, floor 1.11; G2 (P(o) equals the fine-tune's).
  Control: the per-step lift of the pull-back's own path must reproduce the pull-back state with Δ change ≈ 0.
- **The number that decides whether the variant is real.** After step 1 the lifted dV holds an outside-span
  component of learning-rate scale. With the current tol 1e-6 that direction counts as expressible with a singular
  value near 1e-3, and `euclidean_lift` divides by s² + t², so the next step's lift along it is amplified by about
  1e6 and the pieces move by orders of magnitude more than the update. Raise tol and the direction does not count,
  the span never rotates, and the variant is the one-shot lift again. The design must therefore state the tol rule
  (candidate: tol relative to the step's own scale, or a floor on t equal to the smallest alive singular value) and
  report, per step, the max relative change of any piece and the share of the step's Δ change that is the
  second-order dU dVᵀ term, both expected well below that step's ω; a run where e_Δ is met by blowing the pieces up
  fails the check.
- **Cost.** Retrain the fine-tune (10 steps) and LoRA r8 (100 steps) with every-step checkpoints (nothing per-step
  is on disk), per-step SVD bases, about 8 Pile evaluations: 30–40 min on one L40, under $1 on a 4090.
- **Verdict.** Run with changes: drop the fixed-bases variant (provably the one-shot lift), fix the tol rule and
  the per-step blow-up numbers above, run the updated-bases variant on both paths with the pull-back-path control,
  after or alongside the L40 rerun that closes G4.

## 4. Recommendation (for the user's confirmation; nothing runs before it)

1. **Do not run the read-side arm.** It is the write-only half of `pb_r`, which is already measured: expressible
   and still fragile. Record it in `backlog.md`/`TRIED.md` as checked and dropped once the user agrees.
2. **Run the matched-matrix LoRA (3.2) first.** It is the direct repair of the LoRA experiment: round 4 compared a
   LoRA on a matrix the decomposition barely reads against a fine-tune whose energy sits on matrices it reads well.
   Fold in the L40 rerun of the fine-tune's lift as the control, which also closes the G4 miss. About 15 min on an
   L40; RunPod under $1 if the cluster queue blocks.
3. **Then the updated-bases per-step lift (3.3)**, as the answer to the per-step question, on both paths. About
   30–40 min. Its fine-tune-path number (PGD at step 10 against 5.41) is the one that decides whether the lift
   stays a candidate Φ at all.
4. Both need predictions filed first, a smoke run, and one Codex adversarial review of the new run script. Round 4
   is still uncommitted (72 paths); committing it first needs the user's go.

Order 2 before 3 because 2 is cheaper and its outcome (fine-tune-like or LoRA-like PGD for an expressible LoRA)
tells how to read 3's LoRA path.

## Not run

Nothing in this report was run. Not measured: ω and |cos| trajectory of a free A started in span(V_A) (section 3.1,
one minute, only if a drift number is wanted); any LoRA on a matrix other than `h.2.mlp.down_proj`; any per-step
lift; the L40 rerun closing G4; VPD's sparsity loss on any lifted state. The per-matrix table in section 2 is the
fine-tune's own update at step 10 and says nothing about how a LoRA placed on those matrices would train.

## Hand-off

A new session should start from this report, `reports/2026-10-19_analysis_lora-commute.md` and
`reports/2026-10-19_decision_success-criterion.md`. If the user confirms 3.2: write `experiments/lora_matched/run.py`
(new script; reuse `finetune.finetune_lora_arm_a` with all 24 modules, `lift.lift_state`, `faithfulness` as in
`lora_commute/run.py`'s `measure`), file predictions, smoke, Codex review, submit to the cluster (account `c11`,
partition `compute`, caches in `/ephemeral/$USER`), record the job ID.
