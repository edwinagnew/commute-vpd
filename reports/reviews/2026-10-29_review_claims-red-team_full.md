# Red team of five claims: the reviewers' full reports (2026-10-28)

Appendix to `2026-10-29_review_claims-red-team.md`. Each section is one independent Opus reviewer's report, unedited.

---

# Claim 1

## Red-team review, claim (i): teacher training keeps behaviour and passes with Δ fixed

Paths relative to the repo root; `R/` = `experiments/teacher_training/results/`; `TRIED` = `experiments/TRIED.md`.

## 1. Claim as understood
Teacher training (Adam component training of the alive (U, V) on KL(fine-tuned teacher ‖ student), Γ and Δ frozen) gives a state that keeps the teacher's held-out P(target) and passes the adversarial-mask check on Pile validation rows, where leaving δW in Δ (do-nothing) does not.

## 2. Support
- E34: 3 trigger tasks (r1, r0, emoticon) × 2 teacher optimisers (SGD, Adam) × 2 arms (task text A, Pile text B) at batch order 0 (`R/teacher_training_{r1_t534_v369,r0_t352_v310,emoticon}.json`), repeated at batch order 1 (Amendment 4, `R/teacher_training_amend4.json`): 24 copies. LoRA teacher ×3 tasks ×2 arms, 1 order (A3).
- E41: emoticon + " which", checkpoints T10/T100/T1000, 1 order (emoticon T10 task text: 3 orders); chain of 100-step links (A2), emoticon only.
- E42: 3 cipher tasks × 2 arms, 1 order, L40.
- Every teacher is fine-tune seed 0; one model (67M); every harness reading is one adversary start (seed 0), 20 steps, 256 rows. Only A4 adds a second start type (budget code, 128 rows, 20 and 160 steps) on the 12 seed-1 copies.
- "Keeps behaviour" is read on held-out task windows (plan `reports/plans/2026-10-23_plan_teacher-training.md` l.51; E41 analysis l.52). In E42 the held-out windows share the training keys (rand1: one pair), so it tests new contexts, not new mappings (`reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md` l.492).

## 3. Attacks
1. **Under the harness, doing nothing never failed in E34: the check had no shown power where the copies "pass" (weakens, close to fatal for E34 alone).** Do-nothing harness ratios: r1 SGD 1.588, r1 Adam 1.522, emoticon Adam 1.499 (inconclusive); r0 Adam 1.353, r0 SGD 1.121, emoticon SGD 0.984 (pass) (`.teachers.*.nothing.pgd` in the three E34 JSONs; lines 1.40 / 1.61, `R/predictions.json` `.measurements.verdict`). The filed positive control (r0 Adam do-nothing should fail, `.measurements.pgd`) passed at 1.35. On r0 the copies are no better than doing nothing: r0 SGD arm A 1.296 vs do-nothing 1.121; r0 Adam arm B 1.366 vs 1.353. E42: all three do-nothing states inconclusive (1.50 / 1.56 / 1.46, L40 lines 1.24 / 1.81; TRIED E42); only least-squares (1.83-1.98) fails, power "thin on rand1 and ident32".
2. **In every harness cell where doing nothing does fail, the copy sits at or over the line (weakens).** E41 T10 emoticon (do-nothing 2.516): copies 1.547 / 1.280 / 1.505 at three orders, chain link 1.325; T10 " which" (2.164): 1.376 against a 1.396 line, 1.4% inside (TRIED E41, E41 A1, A2; `reports/reviews/2026-10-28_review_e41-check.md` row 2-3). r2 rare target (do-nothing 11.34): copy A 1.451 inconclusive with P 0.808 (`R/teacher_training_amend6.json` `.states["r2_t275_v2667/sgd/A0@1000"]`). So 3 of 6 powered cells pass, each within the 7.5% adversary-seed floor of the pass line; none passes clearly.
3. **The power that exists comes from one reading (weakens the noise margin).** A4's budget start reads the do-nothing states at 1.68-10.45 (160 steps: 2.41-10.45; raw KLs `R/teacher_training_amend4.json` `.tasks.*.teachers.*.nothing.budget`, W₀ 0.6634 / 0.7076) and the copies at 0.95-1.15. That is the strongest support. But W₀ alone at adversary seed 1 reads 1.231 at 160 steps (`.C1b.seed1_over_seed0["160"]`) against the pass line 1.241: the adversary's own spread nearly fills the pass band. Each copy got one adversary seed, and the denominator is one W₀ draw (0.708 vs 0.871 at seed 1).
4. **One-start readings may be lenient; E41 and E42 never got the second start (weakens).** Amendment 5: the same do-nothing state reads 1.588 from the harness's start and 9.22-10.45 from the budget code's (`R/teacher_training_amend5.json`, TRIED E34 A5); 5 of 6 A4 do-nothing states read higher there, including r0 SGD 1.12 → 2.09. E34's copies survive both starts; E41's step-10 copies (already at the line) and E42's copies were read only from the harness's start.
5. **"Keeps behaviour" fails outside its best cases (weakens scope).** Pile text: E42 arm B P 0.019 / 0.0013 on two of three ciphers; LoRA arm B P 0.567 / 0.875 / 0.521 at 1000 (A3). Rare target r2: arm A P 0.808 under the 0.813 kept line and sliding from 0.871 (A6 curves). SGD as student: arm S fails on r0 and r1 (3.39, 3.14), plain-SGD copies fail on Adam and LoRA teachers (A6). The claim holds for Adam on task text with frequent targets.
6. **Sparsity loss and other adversary families not read (cosmetic for this claim, but "faithful" in GOALS lists five checks).** `reports/reviews/2026-10-28_review_e42-check.md` row 1.
7. **"Δ fixed" is true by construction** (Δ frozen; share ≤ 1e-8 everywhere). Not evidence (cosmetic).
8. **The per-seed spread of a copy's harness reading is large next to its margin.** A4 P2: up to 24.6% (r0 Adam B 1.37 → 1.03; `.score.P2.largest.ratio`); E41 A1 T10: 21% span. Filed thresholds were all set before the runs (predictions files present per amendment); I found no rule read differently from its filing.

## 4. Verdict
Weakened: behaviour on held-out windows is kept (P 0.92-1.00, seed spread ≤ 0.005 on task text) and copies pass both start types in E34/A4 (0.95-1.15 against do-nothing 1.68-10.45 under the budget start), but under the harness used for E34, E41 and E42 do-nothing never fails on the six E34 teachers or three E42 ciphers, and in the six harness cells where it does fail the copies read 1.28-1.55 (3 of 6 pass, all within 7.5% of the 1.40 line).

## 5. Cheapest settling test
Cluster `debug` job (L40, ≤ 30 min, $0): read E42's saved arm-A copies (`R/large/e42/{rand32,rand1,ident32}/A_step1000.pt`) and the r1 SGD copies (`R/large/r1_t534_v369/amend5_{A,B}_sgd_seed{0,1}_step1000.pt`), beside W₀ and the regenerated do-nothing states (r1 SGD fine-tune, 1000 SGD steps; rand32), under 4 adversary starts each (harness seeds 0-2 plus the budget code's start), 20 steps, 256 rows; score max over starts. Stands: every copy's max ≤ W₀'s own max-over-starts × 1.40 and every do-nothing max ≥ 1.61. Falls: any copy's max ≥ 1.61, or a copy's max within W₀'s spread of its do-nothing's. States exist for E42 and A5; the E34/A4 main-grid copies and every E41 copy are not saved (`experiments/SAVED.md` l.217), so E41's step-10 line cell needs a re-train (~$1 H100, or cluster).

## 6. Not checked
Notebooks were not read; E34's long analysis was read by grep only; the E41 T10 numbers come from TRIED and the check reports (my jq path missed them in `R/teacher_training_e41*.json`); the stochastic, CI-masked and unmasked ratios were not re-checked per state; the E43 remainder and the A7 files were not read.

---

# Claim 2

## Red-team review, claim (ii): the rank-8 LoRA's teacher-trained decomposition passes, and not because of the rank cap

Paths are relative to the repo root. BF = `experiments/broad_fragility/results/`.

## 1. Claim as understood
On E36's broad lowercase task, teacher training on a rank-8 Adam LoRA fine-tune leaves a decomposition that passes the 20-step harness check (≤ 1.40). The same teacher training on the Adam and plain-SGD full-weight fine-tunes leaves decompositions that fail it (≥ 1.61). The claim adds that this pass is a real result, not something the rank-8 cap produces.

## 2. Support
- **LoRA teacher-trained decompositions**: 2 (LoRA seed 0 with batch order 0; seed 1 with order 1, changed together). Checks 1.2726 / 1.2688 on the first 256 rows and 1.2066 / 1.2425 on rows 256-511 (`BF/amend2.json .reads.tt_lora_s0.check`, `.tt_lora_s1.check`, `..._rows2.check`). E39 gave 1.2726 too (`BF/broad_fragility.json .reads.tt_lora.check`).
- **Adversary**: one start, seed 0, for every state. `faithfulness.harness_pgd` calls `torch.manual_seed(seed)` (`src/commute/faithfulness.py:128`), so every state gets the same start mask.
- **Tasks and models**: one broad task, one 67M model, one card (H100).
- **Full-weight fails**:
  - Adam: 2.201 / 1.883 on the two row sets (`.reads.trained`, `.trained_rows2`); E37 order 1 read 1.88.
  - Plain SGD: 2.067 (`BF/amend1.json .reads.tt_sgd.check`).
  - SGD held to rank 8: 2.122 (`.reads.tt_sgd_r8.check`).
  - None of the SGD states was read on the second rows.
- **Predictions**: P12 and P13 were filed before the run (`BF/predictions_amend2.json`, committed 14:10:49) and scored as filed; both hold.

## 3. Attacks

1. **The rank-cap explanation was tested with one confounded arm. "Not an artefact of the cap" is not shown. (weakens, the main gap)**
   - The only test is SGD held to rank 8 per matrix, at one rate (0.03) and one seed. Its check is 2.122.
   - That shows a rank-8 cap is not sufficient *under plain SGD*. The LoRA was trained with Adam, and the analysis lists the optimiser, the factorisation, the subspace and the gain as still differing (`reports/analyses/2026-10-28_analysis_e39-amend2.md` §2).
   - None of the following was run:
     - a LoRA rank sweep ("a LoRA at higher rank" is under Not run);
     - Adam held to rank 8;
     - a truncated SVD of the Adam δW;
     - a rank-matched full-weight Adam fine-tune.
   - So nothing tests whether the cap is *necessary* to the LoRA's pass, which is what "artefact of the rank-8 cap" asks.
   - The approved next run, a rank-8 LoRA trained with SGD, separates the optimiser from the factorisation. It does not vary the rank either.

2. **The LoRA-against-SGD contrast rests on one block of 128 rows. The LoRA's own pass rests on averaging two blocks. (weakens)**
   - Per-block checks (analysis §2 table; `reports/reviews/2026-10-28_review_e39-amend2-check.md:40-41`):

     | state | block 1 | block 2 |
     |---|---|---|
     | plain SGD, teacher-trained | 1.16 | 2.93 |
     | SGD held to rank 8, teacher-trained | 1.25 | 2.95 |
     | LoRA seed 0, teacher-trained | 1.11 | 1.42 |
     | LoRA seed 1, teacher-trained | 1.09 | 1.44 |

   - On block 1, both SGD states pass. On block 2, both LoRA states sit in the zone between the lines.
   - The SGD states were never read on the second rows. The Adam teacher-trained decomposition was, and its check fell there from 2.20 to 1.88.

3. **The adversary's start was never varied, and the margin is about 10%. (weakens)**
   - The LoRA's distance to the pass line is 1.40 / 1.27 = 1.10.
   - The unchanged model's KL over adversary seeds 0-2 is 0.699 / 0.751 / 0.707, so the denominator alone moves by up to 7.5% (`BF/amend2.json .stage0.w0_pile`).
   - E34 Amendment 5 read one do-nothing state at 1.6 or 9.2 depending only on the start (`experiments/TRIED.md:76`). E34 Amendment 4 found a trained copy's check moving up to 25% between batch orders (`TRIED.md:65`).
   - Both LoRA repeats share start seed 0, so the "replication" samples LoRA seed, batch order and row set, but not the adversary.
   - No 160-step or budget-adversary read of a LoRA teacher-trained state exists. E37 has one for Adam only.

4. **The size match is loose, and the LoRA starts much less fragile than Adam. (weakens the Adam contrast, not the SGD one)**
   - Steps (300) and data are matched (`experiments/broad_fragility/run.py:60`).
   - The LoRA's rate is the lowest on its grid (1e-4), and its gain is still 14% above Adam's: 0.148 against 0.130 (`BF/broad_fragility.json .shape.lora.grid`). No smaller-gain LoRA exists.
   - Do-nothing checks: LoRA 2.653 against Adam 4.868. Teacher-trained over do-nothing is 0.48 for the LoRA and 0.45 for Adam. So against Adam, "teacher training halves the check, and the LoRA starts lower" fits the numbers.
   - Only the SGD comparison breaks this: do-nothing 2.623, teacher-trained 2.067 (`BF/amend1.json`). That comparison sits under attack 2.
   - The capped SGD fine-tune's do-nothing check was not read.

5. **The second seed may be nearly the same update. (cosmetic to weakens)**
   - The two LoRAs reach the same gain (0.1479 against 0.1481), and their checks agree to 0.004.
   - No cosine between the two LoRA δWs, or between their teacher-trained offsets, is reported. Seed and batch order are changed together.

6. **The pass depends on the recipe. (cosmetic for this claim)**
   - In E34 Amendment 6, a plain-SGD map run on a narrow-task LoRA teacher failed at 2.86 (`TRIED.md:91`, P4).
   - So "a LoRA passes" is a property of the update together with Adam teacher training, not of the update alone.

7. **Blind spots.**
   - One broad task, 67M only, the mean over 256 rows, a check that is a lower bound.
   - The check is blind to which rows fail when the mean passes (see attack 2).

## 4. Verdict
**Weakened.** The LoRA's teacher-trained decomposition passes four times out of four (1.27, 1.27, 1.21, 1.24), but all four use one adversary start, with a 10% margin. The contrast with SGD rests on one block of 128 rows (2.93 against 1.42). "Not an artefact of the rank cap" rests on a single SGD rank-8 arm (2.12) that cannot test whether the cap is necessary under Adam.

## 5. Cheapest settling test
**What**: an Adam LoRA rank sweep, r ∈ {8, 32, 128}, at rate 1e-4. Same rows, 300 steps, seed 0. Teacher-train each for 1,000 steps with E39's recipe. Read the harness at adversary seeds 0, 1 and 2 on both row sets, with the unchanged model's KL at the same seed as each denominator. Prelude at $0 in the same job: re-read the saved states `BF/large/a2_tt_lora_s0.pt`, `a2_tt_lora_s1.pt`, `a2_tt_sgd_r8.pt`, `tt_sgd_step1000.pt` and `trained_step1000.pt` at adversary seeds 1-2 and on the second rows.

**Where and cost**: the MATS cluster's `debug` QoS. Three fine-tunes plus teacher training on the 67M take about 25 min (E39 Amendment 2 ran nine stages in 38 min on an H100); $0.

**Read rule**:
- **Stands** if:
  - every LoRA teacher-trained decomposition reads ≤ 1.40 as a mean over starts on both row sets;
  - the r = 32 and r = 128 ones also pass;
  - both SGD teacher-trained states read ≥ 1.61 on the second rows.
- **Falls** (cap artefact) if the check rises past 1.61 as the rank grows.
- **Falls** (not real) if a LoRA's mean over starts exceeds 1.40, or the SGD states pass on the second rows.

**Saved states**: the five offset files above exist in `BF/large/` (each 60 MB, bf16). The higher-rank LoRAs need new training. The repo has no reader script for adversary seeds other than 0; `pile_faithfulness(seed=...)` takes the argument.

## 6. Not checked
- Did not open the notebook `E39_broad_fragility.ipynb` or E37's analysis/notebook in full for overstatement.
- Did not read the per-block log `BF/large/amend2.log` myself; per-block numbers are from the analysis and its review.
- Did not check `experiments/SAVED.md` (grep found no broad_fragility entry; the files exist on disk).
- Did not verify whether the LoRA's damage profile (where on the rows) differs from capped SGD's.

---

# Claim 3

## Red-team review: claim (iii), "splits pass only while the update is small"

## 1. Claim as understood
The least-squares, adjoint (scaled, c fitted per matrix) and rescaling splits of a finished fine-tune update meet the weak version only while the update is small (largest relative matrix change, "size"); past some size they fail (E16, E22, E31 + amendments 1-3). Source of the wording: `decisions.md:31` and E31 Amendment 1, `reports/analyses/2026-10-22_analysis_adjoint-sgd.md:217-218, 289`.

"Fail" means two different things in the evidence: (a) the adversarial check (least-squares and do-nothing, E16/E22); (b) keep = adjoint P(target) / fine-tune P(target) (scaled adjoint split, E31). The adjoint split never fails (a).

## 2. Support
- E16: emoticon, 1 seed, Adam 3e-5 fine-tune + E9 LoRA, steps 10/100/1000, 1 H100. Gate G1 **failed** (ft step-10 least-squares 6.96 vs E14 5.24, `2026-10-21_analysis_split-comparison.md:10,215`), so P1-P3 are unscored.
- E22: emoticon, seed 0 only, 3 fine-tunes (SGD 0.01, Adam 3e-6, Adam 3e-5), 1 H100 (`2026-10-22_analysis_small-finetune-splits.md`, "One task, one training seed").
- E31 + Amend 1-2: 3 tasks (emoticon, r0_t352_v310, r1_t534_v369, all single-token), SGD 0.01 and 0.03, 2 batch-order seeds per cell, 3 steps each; Adam only on emoticon (2 seeds) and Adam 3e-5 on r0/r1 (1 seed). 1 model (67M), 1 decomposition. Cards mixed: L40 (floor 0.209) and L40S (floor 0.046) (`adjoint_sgd.json`, `adjoint_sgd_amend{1,2}.json` `.rows[].floor`).
- Amend 3: emoticon, seed 0, 3 fine-tunes, unscaled vs scaled, 1 H100.
- Notebooks: only `notebooks/E16_split_comparison.ipynb`; **E22 and E31 have no notebook**. E22's cited log `results/large/full.log` is absent (`experiments/SAVED.md`: "small_finetune_splits — Empty").
- No δW tensors saved for E16/E22/E31 (`experiments/SAVED.md` → adjoint_sgd: "Logs only").

## 3. Attacks
1. **For the adversarial sense (a), the data contradict "fails past some size". Fatal for that half.** E22: SGD 0.01 at step 1000, size 0.0122, do-nothing 0.98, least-squares 1.01 (pass), while Adam 3e-5 at size 0.0113 fails (2.52 / 6.96) (`small-finetune-splits.md` table). Within runs, adversarial failures sit at the *smallest* size: r1 SGD 0.01 step 10, size 0.0034, do-nothing 2.56 / 2.70, passing at 0.014 (0.96 / 0.95); large Adam do-nothing 2.18 (0.011) then 1.12 (0.021) (`adjoint_sgd*.json .rows[] | nothing_pgd`). E16 least-squares is non-monotone, 6.96 / 1.48 / 2.11. E22's own conclusion is "size alone does not decide the verdict". The failures go with step 10, not with size.
2. **Size is confounded with step count, rate and optimiser; the one size-matched contrast reads against size. Weakens.** "B2_size" compares 0.01 vs 0.03 at the *same* step 1000, which is a rate test. Its filing says it is "blind to... size against path (no size-matched pair of rates)" (`predictions_amend2.json .predictions[1].blind_to`). The unscored matched comparison (`adjoint-sgd.md:441-445`): the faster run at step 100 keeps more than the slower run at step 1000 at nearly equal size. r1: 0.27 / 0.24 at 0.012 vs 0.13 / 0.13 at 0.014. r0: 0.49 / 0.58 at 0.013 vs 0.39 / 0.40 at 0.015. Keep is also not monotone in size within a task. r1 keep is 0.37 / 0.39 at 0.0034 (0.01, step 10), 0.56 / 0.68 at 0.0057 (0.03, step 10) and 0.42 / 0.45 at 0.008 (0.01, step 100) (rows above). Across optimisers, Adam 3e-5 keeps 0.93 at 0.011 and SGD 0.01 keeps 0.74 at 0.0024, while Adam 3e-6 keeps 0.24 at 0.008 against SGD 0.60 at 0.012.
3. **No "passes while small" region on r1. Weakens.** With E31's own "keeps" bar of P ≥ 0.5 (P1/P2), r1 at SGD 0.01 never passes: keep is 0.37 / 0.42 / 0.13 on seed 0 and 0.39 / 0.45 / 0.13 on seed 1, so it fails already at size 0.003. r0 crosses 0.5 at 0.03 step 100, at 0.49 and 0.58 on the two seeds. That boundary lies inside the step-100 seed spread of up to 0.11 (`adjoint-sgd.md:416`). No size threshold was ever filed. The "size" reading was adopted after both Amendment 1 predictions missed (A1 "mixed", A2 "rate").
4. **The fall in keep is mostly the fitted scale, not size. Weakens: the fall is partly a property of the measurement, not of size.** Amend 3: the unscaled split (c = 1) keeps 1.00-1.03 in 8 of 9 states, including SGD at size 0.012 and Adam 3e-6. The scaled split keeps 0.10-0.93 (`adjoint-sgd.md:571-581`). Median fitted c falls with the step: emoticon SGD 0.03 goes 0.62 → 0.51 → 0.33, Adam 3e-5 goes 0.43 → 0.29 → 0.20 (`.arms[].steps[].per_matrix[].c`, median, `adjoint_sgd*.json`). c is fitted by least squares, cK(δW) ≈ δW, and K is linear, so c does not change when δW is scaled. c can fall only because δW's direction changes along training, not because it grows. At matched size, c does not separate the tasks either: SGD 0.01 at step 1000 has median c 0.57 / 0.48 / 0.52 (emoticon / r0 / r1) against keep 0.60 / 0.39 / 0.13. No random-update null exists for keep (it cannot have one), and no rescaled-δW control was run.
5. **References.** The do-nothing reference was read on the same rows for the adversarial check (every row has `nothing_pgd`). For keep, do-nothing is 1 by construction, so it is not a reference. The pass lines differ by card: L40 pass ≤ 1.24 with floor 0.209, L40S ≤ 1.43 with floor 0.046, H100 ≤ 1.40 with floor 0.075. r1's step-10 adjoint state reads 1.33 on the L40S: a pass there, inconclusive on the L40 (`adjoint-sgd.md:309-310`). Cosmetic for (iii).
6. **Rescaling split.** It was read as a state only in E16 (large Adam and LoRA). It never kept behaviour at any size (P 0.012-0.016, W₀ 0.012). It gives no evidence on "passes while small". The claim should not name it. Cosmetic.
7. **E16's large-update failure has an unexplained card dependence.** The step-10 least-squares state reads 5.41, 5.24 and 6.96 across cards. G1 failed, and its cause is "not established" (TRIED E16 row). Weakens the E16 leg.

## 4. Verdict
**Weakened, and falsified in part.** The adversarial half is falsified by E22: SGD at size 0.0122 passes at 0.98 / 1.01, where Adam fails at 0.0113, and failures cluster at step 10 down to size 0.003. The behaviour half holds only as "the scaled adjoint split's keep falls along training for plain SGD within a task" (2 seeds, 3 tasks, step-1000 spread ≤ 0.016). It does not show that size is the variable. At matched size the run with fewer steps keeps more (r1 0.25 vs 0.13). The unscaled split keeps 1.00 at the same sizes. And r1 never keeps 0.5 at any size.

## 5. Cheapest settling test
One MATS `debug` job, L40, about 20-25 min, $0 (no tensors are saved, so the fine-tunes are regenerated). Run E31's `arm()` recipe on r1_t534_v369, seed 0:
- (i) SGD 0.03, with checkpoints at the steps where size equals the 0.01 run's step-100 and step-1000 sizes (0.008 and 0.014; about step 50 and about step 330 from the existing curve);
- (ii) the step-1000 SGD 0.01 δW multiplied by α, so that its size is 0.0034 and 0.008. Its scaled adjoint keep is read with c unchanged, as c does not change under scaling.

Read rules:
- **Size stands:** (i) agrees within 0.05 of the 0.01 run's keep at each matched size (step-10 seed spread to 0.11, so tighten with seed 1), and (ii) restores keep to ≥ 0.37 / 0.42.
- **Size falls (path or direction):** (i) differs by more than 0.1, as the existing 0.25-vs-0.13 hints, or (ii) stays near 0.13 at small α.

Re-reading the existing JSONs already settles the adversarial half; no run is needed for it. Saved states: none (`experiments/SAVED.md`).

## 6. Not checked
- The E16 notebook's text against its JSON; the LoRA splits' sizes in E16.
- `2026-10-22_analysis_step-split-small.md` and `sgd-vs-adam.md` in detail.
- The budget sweep beyond the analysis's summary.
- Whether keep's held-out set (73 windows) changes between tasks.
- Whether the "plain SGD" step is gradient-clipped (the review notes clipping at 1.0 is always active), which would make size depend on rate × steps nonlinearly.

---

# Claim 4

## Red-team review, claim (iv), E43: teacher training plus remainder is worse than do-nothing

Paths are relative to the repo root. R = `experiments/teacher_training/results/teacher_training_e43.json`, L = `experiments/teacher_training/results/large/run_e43.log`, A = `reports/analyses/2026-10-28_analysis_e43-teacher-remainder.md`.

## 1. Claim
The state "teacher training plus remainder" has the subcomponents after teacher training and Δ′ = Δ + W_ft − W_tt, so its weights equal the fine-tuned model's. On the emoticon fine-tune at steps 10 and 1,000, this state reads a higher Pile-harness PGD ratio than do-nothing (the whole δW in Δ).

## 2. Support
- 1 model (67M) × 1 task (emoticon, Adam 3e-5 arm A) × 1 fine-tune seed (0) × 1 teacher-training batch order (0) × 1 adversary start (seed 0, 20 steps, 256 rows = 2 blocks of 128) × 2 steps (10, 1000). One H100 card. A → "How compelling", TRIED row E43.
- Do-nothing was **run fresh in the same job**, on the same rows with the same adversary seed and budget. It matched E41 exactly (R `.gates."PC T10".pgd` 2.5159000, `."PC T1000".pgd` 1.4993958). So the reference was not just quoted from E41.
- "Worse than do-nothing" was **not a filed prediction**. P3 (`results/predictions_e43_v2.json`) compares plus remainder with teacher training's own state, at 1.25×. The comparison with do-nothing appears only as the reported "excess removed" quantity (R `.score.P3_check_paired.per.*.excess_removed` −0.927 / −0.732).

## 3. Attacks
1. **"Worse" holds on the PGD ratio only. The stochastic half of the check shows no difference (weakens).** Stochastic ratios: plus remainder 1.0033 / 0.9174, do-nothing 1.0026 / 0.9162 (L lines 129, 149, 161, 181). The claim should say "by the PGD adversary". The removal shares say nothing about the comparison: do-nothing's share is 1 by construction, and plus remainder's removal state is teacher training's own state (A, "How much...").
2. **At step 1,000 the margin rests on one 128-row block (weakens).** Ratio 1.865 vs 1.499, or 1.24×. Per-block PGD KL, plus remainder vs do-nothing: block 1 1.040 vs 1.081, where plus remainder is *lower*; block 2 1.567 vs 1.015 (L 158-159, 178-179). The W0 block-2 KL across adversary seeds 0/1/2 is 0.718 / 0.828 / 0.690, a 20% range (L 8, 11, 14), against the whole-set floor of 7.5% (R `.W0.pgd_floor` 0.0748). The do-nothing and plus-remainder states have never been run at more than one adversary seed. The noise floors quoted in A (7.5% W0 adversary spread; 3.2% batch-order spread of teacher training's own state at T1000) are not this comparison's noise. A 24% margin carried by one block is above the measured floors but has no matching floor of its own.
3. **The PGD ratio is spiky along steps even for one fixed fine-tune (weakens step 10).** Do-nothing reads 1.049, 1.131, **2.516**, 1.052, 1.136, 1.243, 1.499 at steps 1, 3, 10, 30, 100, 300, 1000 (`results/teacher_training_e41_amend2.json` `.teachers.*.nothing.pgd`). Step 10 is an outlier, and its excess sits in block 2 (3.79× vs 1.17×, `teacher_training_e41_amend1.json` `.teachers."10".nothing.pgd_blocks`). One start on 256 rows is therefore a heavy-tailed reading. Plus remainder at T10 (3.92) has its excess in the same block 2 (4.49 nats, L 147).
4. **At step 10 the result is close to expected from size alone (weakens step 10 as evidence).** The remainder's squared size is 5.12× the fine-tune's (R `.steps."10".plus_remainder.delta_share` 5.115), and cos(R, δW_tt) = −0.903. So Δ′ mostly cancels teacher training's own weight change, and an adversary that masks subcomponents unmasks that cancellation. A residual 2.3× do-nothing's in norm scoring worse than do-nothing is unsurprising. At T1000 the sizes are comparable (1.121), so the size argument does not apply there.
5. **Not specific to teacher training (cosmetic, but it matters for the reading).** The least-squares split, which also gives exact weights and moves subcomponents, reads even worse than do-nothing: 6.957 / 2.106 (R `.steps.*.least_squares.pgd`). Data so far show "states that move subcomponents and keep exact weights read worse than do-nothing", not something peculiar to the remainder of teacher training.
6. **"The comparison is paired ... same adversary seed, so their noise is shared"** (notebook line ~111; A "Large miss, paired comparison"). This is an overstatement. A shared PGD seed on two different models does not make their adversary noise shared. The pairing controls rows and budget, not seed-to-seed spread (cosmetic).
7. **Card dependence.** The W0 PGD floor is 0.0748 on the H100 and 0.209 on the L40 (`src/commute/faith_score.py:16`). The absolute ratios are not portable across cards, but the paired ordering may be. RUN_BRIEF_e43.md:40 expects an L40 to fail C1/PC (cosmetic for the claim, relevant to the test below).
8. Filed-rule fidelity is fine. P3 "does not hold" is scored correctly at 2.536× / 1.945× against 1.25× (R `.score`). The independent check review (`reports/reviews/2026-10-28_review_e43-check.md`) found no wrong number. No state was saved (A, "Not run"; `experiments/SAVED.md` has no E43 entry).

## 4. Verdict
**Weakened.** Plus remainder reads above do-nothing on the PGD ratio at both steps (3.92 vs 2.52; 1.87 vs 1.50) with a fresh, same-row, same-seed do-nothing. However, it rests on one adversary start, one task and one seed. The stochastic ratio is identical (1.003 vs 1.003; 0.917 vs 0.916). At T1000 the 24% margin comes entirely from one 128-row block whose W0 adversary-seed range alone is 20%. At T10, Δ′ is 5× the fine-tune's squared size.

## 5. Cheapest settling test
No saved states exist, so a CPU re-read is impossible. Instead, run a cluster `debug` job (≤ 30 min, $0). The fine-tune and teacher training are regenerated in-job and deterministic: E43 reproduced E41 to 4+ digits, and the whole job took 12 min on an H100. Run `remainder.py`'s T1000 (and T10) do-nothing and plus-remainder states with adversary seeds 0, 1 and 2 on the same 256 rows, all in one job and on one card. Compare within the job only, since the card changes the absolute numbers.
- **Read rule.** For each step, compare the minimum over seeds of plus remainder's PGD KL with the maximum over seeds of do-nothing's.
- **Stands:** at T1000 the plus-remainder minimum is above the do-nothing maximum, and the per-seed ratio is ≥ 1.1 at all 3 seeds.
- **Falls:** at T1000 the ranges overlap, or plus remainder is ≤ do-nothing at any seed.
- **Better extension, same job:** add teacher-training batch orders 1 and 2 (E41 A1 recipe) to plus remainder at T1000.

## 6. Not checked
- I did not read `remainder.py` line by line (I relied on the code review's verdict on the Δ′ construction).
- I did not check the plan's decision tree or the figures.
- I did not check whether a "budget adversary" reading exists for any E41 do-nothing state.
- I did not look at the r1 task's E41 do-nothing seeds for a cross-task noise estimate.

---

# Claim 5

## Red-team review, claim (v): "teacher training needs Adam; with plain SGD it fails" (E34 Amendment 6)

Paths are relative to the repo root. TT = `experiments/teacher_training/results/`.

## 1. Claim as understood
Teacher training on the alive subcomponents' read/write vectors (Γ, Δ frozen) passes the adversarial check with Adam at 1e-5 but fails it with plain SGD (no momentum, `src/commute/component_training.py:116,143`), on the Adam and LoRA teachers as well as the SGD teacher. "Fails" means the adversarial check, not the behaviour: every plain-SGD copy keeps P ≥ 0.985.

## 2. Support
- Amendment 6 stage c: **one task (r1) × three teachers (SGD 0.03, Adam 3e-5, LoRA rank 8) × one SGD rate (0.03) × one step read (1,000) × one batch order × one 20-step adversary start**. `TT/teacher_training_amend6.json` `.states["r1_t534_v369/{sgd,adam,lora}/S0@1000"].pgd` = 3.137, 3.138, 2.856, all `verdict:"fail"`. W₀ floor 0.0748 (`.W0.pgd_floor`), so the pass line is about 1.40 and the fail line about 1.61 (`src/commute/faith_score.py` `verdict`).
- The Adam references at step 1,000 come from earlier same-card jobs, not this run: 1.004 (`TT/teacher_training_r1_t534_v369.json .teachers.sgd.A["1000"].pgd`), 0.968 (`.teachers.adam.A["1000"]`), 0.967 (Amendment 3).
- Filed before the run: P4 rule "S at step 1000 on the Adam and LoRA teachers: fail → map" (`TT/predictions_amend6.json .predictions[] | select(.id=="P4_sgd_map")`). Scored as filed, and it passes.

## 3. Attacks
1. **Plain SGD already passes in the project's own data, so "needs Adam" is contradicted. Fatal to the general wording.**
   - On emoticon, E34's plain-SGD copy (lr 0.01) **passes at both reads**: 1.112 at step 100 and 1.111 at step 1,000, P 0.994 / 0.998 (`TT/teacher_training_emoticon.json .teachers.sgd.S`). The 2026-10-23 analysis says so (`reports/analyses/2026-10-23_analysis_teacher-training.md:101`, "passes throughout on emoticon").
   - On the r1 SGD teacher, the same rate 0.03 **passes at step 100**: 0.9998, P 0.986 (`TT/teacher_training_r1_t534_v369.json .teachers.sgd.S["100"]`).
   - The Amendment 6 analysis, the log card (`docs/research_log/parts/experiments/e34.html:41`, "so the map needs Adam"), the notebook ("the map needs Adam") and `decisions.md:50` ("Teacher training uses Adam") never mention the emoticon pass. The decisions row cites E34 as support.
2. **It is a 1,000-step overshoot, not an optimiser verdict. Weakens.**
   - Only step 1,000 was read under the adversary on the Adam and LoRA teachers. The plan's own expectation was "pass at 100" (`TT/predictions_amend6.json .expected.S`). Step 100 under the adversary is listed under "Not run" (analysis, l. 402).
   - The behaviour is already kept at step 100: P 0.992 on the Adam teacher and 0.990 on the LoRA teacher (`.curves["r1_t534_v369/{adam,lora}/S0"][0].p_target`).
   - An early-stopped SGD copy may well pass, as it did on the SGD teacher. Under that reading the result is "plain SGD at 0.03 for 1,000 steps", not "plain SGD".
   - Window damage does not settle step 100 either way. On the SGD teacher it was flat (0.063 at step 100, 0.067 at step 1,000) while the ratio went from 1.00 to 3.14. On the LoRA teacher the SGD copy's window damage at step 100 is already 0.053, against the teacher's 0.0148 (3.6×), which hints that it may fail early there.
3. **One rate, never matched to Adam. Weakens.**
   - 0.03 is E21's component-training rate (`amend6.py:73`), not tuned for teacher training. Rates 0.003 to 0.01 were never tried on r1; the emoticon pass used 0.01. The memory's "scope a grid null" rule applies directly: this is a one-point grid.
   - Step 1 was never measured. Size at step 100 is 0.0079 for SGD (`.teachers.sgd.S["100"].size`, E34 r1) against 0.0129 / 0.0133 for Adam (`.teachers.{sgd,adam}.A["100"].size`). By step 1,000 the sizes match: SGD 0.017-0.021, Adam 0.017-0.021. So speed of movement does not obviously explain the gap, but nothing was matched.
4. **Task-dependent even among the existing reads. Weakens.** On r0 (rate 0.03) SGD fails already at step 100 (1.888) and at step 1,000 (3.385) (`TT/teacher_training_r0_t352_v310.json .teachers.sgd.S`). Across tasks, plain SGD gives pass/pass (emoticon), pass/fail (r1) and fail/fail (r0). That is not one optimiser verdict.
5. **The noise floor is not the issue. Cosmetic, and in the claim's favour.**
   - The fails sit 1.77 to 1.95× above the fail line.
   - The adversary spread on W₀ is 0.699 / 0.751 / 0.707 (`.W0.pile[].pgd_kl`), about 7%.
   - A stronger adversary only raises a fail.
   - The two ratios 3.1368 and 3.1383 on different teachers agree to 5e-4. Most likely a coincidence; I did not check it further.
6. **Where the Adam references come from. Cosmetic.** The Adam pass readings are from E34 and Amendment 3 jobs (same card, same floor 0.0748, asserted in `plot_amend6.py`), not re-read in this job. The analysis states this.

## 4. Verdict
**Weakened.** The scored fact stands: SGD at 0.03 read at step 1,000 fails on the r1 Adam and LoRA teachers, at 3.14 and 2.86 against a fail line of 1.61, with the behaviour kept. But "teacher training needs Adam" is contradicted by plain SGD passing on emoticon (1.11 at steps 100 and 1,000) and on the r1 SGD teacher at step 100 (1.00). The supported claim is "plain SGD at 0.03, run for 1,000 steps on r1, fails". The log card and `decisions.md` read it as an optimiser property.

## 5. Cheapest settling test
- **What.** Regenerate the r1 Adam and LoRA teachers (seconds each). Train arm S at plain-SGD rates {0.003, 0.01, 0.03}. Read the adversary at steps 100 and 1,000. Six arms, about 190 s each by the plan's estimate (`reports/plans/2026-10-23_plan_teacher-training.md:249`).
- **Where.** MATS cluster `debug` QoS (≤30 min, $0), using `amend6.py`'s `student(…, "S", …)` with `s_lr` varied and steps (100, 1000).
- **Read rule.**
  - **Stands** (SGD fails as an optimiser on r1): no (rate, step) with P ≥ 0.9 reads "pass" on both the Adam and the LoRA teacher.
  - **Falls** (the result is about rate and step count): some cell passes on both teachers.
  - **Mixed**: a cell passes on one teacher only. Report it per teacher.
- **Saved states.** None are saved. No S offsets exist (`experiments/SAVED.md` → teacher_training; `TT/large/amend6/` holds only r2 A/B and the LoRA B copies), so a CPU re-read is impossible and this has to be a GPU run.
- **$0 alternative.** Rewrite the claim's scope now. Add E34's emoticon pass and r1 step-100 pass to the log card, the notebook and the `decisions.md:50` row.

## 6. Not checked
- Whether Adam at higher rates also fails late, which would make the issue step size rather than optimiser.
- E21/E26 component-training SGD rate scans and how they transfer here.
- The 2026-10-22 sgd-vs-adam analysis text.
- The run log `run_amend6.log`.
- The figure `amend6_sgd.png`.
- E39's labelled SGD pass at step 100 on the broad task (seen only in a log-card line), which is further evidence that SGD can pass early.
