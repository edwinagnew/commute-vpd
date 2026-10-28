# E34 Amendment 6: independent check of the analysis and notebook section (review)

*2026-10-28, independent agent that did not write the code, the run or the analysis; read-only apart from this file.
Checked from the raw files: `experiments/teacher_training/results/teacher_training_amend6.json` (sha256 `2b6a7338…`,
matches the analysis), `results/large/amend6/run_amend6.log` (`9068de04…`, matches; 685 lines, exit 0, 12:52:33 to
13:57:13), `results/predictions_amend6.json` (`cab3ba1b…`, committed once in d92eebb at 12:07, unchanged at HEAD),
`amend6.py` (`1d1baed5…`; `summary`, `long_category`, `r2_category`, `score`), E34's
`teacher_training_r1_t534_v369.json`, Amendment 3's `teacher_training_amend3.json`, Amendment 4's
`scorecard_amend4.json`, the plan's "Amendment 6", `reports/analyses/2026-10-28_analysis_e34-amend6.md` and the
notebook cells "Amendment 6" of `notebooks/E34_teacher_training.ipynb`. Arithmetic with `jq` on the stored values.*

Every number that carries a claim matches the raw files, and all six predictions and the three controls are scored by
their filed rules; the run's own `score` block and the analysis's scorecard agree. The errors are in wording, not
numbers. Five sentences claim more than one task, one rate and one seed show: "at most 1.7 times" for all copies (two
unscored copies read 2.0 and 2.6), "every kind of fine-tune" and "fails late whatever the teacher" (one task, one rate,
and no early adversary read on the two new teachers), "the largest update here" (the LoRA teachers are larger), "none
moves towards the LoRA" (emoticon's copy closes a quarter of the distance), and "well inside the batch-order spread"
for the rare target (that spread was measured on Pile-text copies; task-text copies moved at most 0.005, the size of
the miss). The decision-tree branches named are the filed ones, with two gaps: P1's own branch (plain-text
harmlessness onto the card and matrix) is not listed, and the rare-target "cheap checks" are new GPU runs outside the
tree.

## 1. Numbers

Pass and fail lines at this job's floor 0.0748164: 1.5 / 1.0748 = 1.3956 and 1.5 × 1.0748 = 1.6122. Match.

| Quantity (analysis) | Raw | Check |
|---|---|---|
| C1 floor 0.0748, stochastic floor 0.0011; W₀ plain-text damage 0 | 0.074816, 0.001094; 0 | match |
| G1: 0.98343, 0.02875, 1.5884; plain-text damage 0.0430, gap 0 | same | match |
| G3: fail 3.137, P 0.98566 | 3.13684, 0.985656 | match |
| Teachers vs same-card references (Adam r1, three LoRAs) | all inside tolerance, `matches: true`; r2 has no reference | match |
| Table § 1: every P(target), plain-text damage, × fine-tune | recomputed from `states.*.pile_damage / .../nothing.pile_damage` | match, all 22 rows |
| Table § 1: "Window damage, × the fine-tune's" | the denominator is the teacher's own fine-tune eval (`teachers.*.reading.damage`), not the do-nothing state's harness damage; for LoRA teachers these differ by 1-2%: r1 LoRA task-text copy 2.12 vs 2.09, plain-SGD copy 3.46 vs 3.41, r0 Pile-text 1.99 vs 1.95 | match, mixed denominators; say which |
| Teacher plain-text vs window damage (0.043/0.056, 0.090/0.076, 0.013/0.015) | 0.04296/0.05564, 0.09028/0.07620, 0.01295/0.01476 | match |
| § 2 table: E34/A3 ratios 1.522, 1.004, 0.968, 1.003, 0.967 | 1.52247, 1.00376, 0.96759, 1.00311, 0.96663 | match |
| § 2: S ratios 3.137, 3.138, 2.855; stochastic 0.968, 0.934, 1.042 | 3.13684, 3.13828, **2.85590**; 0.96899, 0.93445, 1.04241 | 2.855 should be 2.856 (truncated) |
| § 2: P at 100 / 1,000; sizes 0.0212, 0.0191, 0.0170 | curves 0.9859/0.9852 (state 0.98566), 0.9923/0.9874, 0.9904/0.9974; sizes match | match |
| § 2: "the same deterministic states E34 read" | E34 A copies at 1,000: P 0.996295 (SGD), 0.994434 (Adam); size 0.017213, 0.020946; A3's r1 LoRA A: P 0.995341, size 0.018259 — identical to this run to every digit | match |
| § 3: teacher P 0.8634, size 0.0849, do-nothing 11.34 / 1.079; kept line 0.8134 | same | match |
| § 3: task-text copy 0.8079, 1.451, stochastic 1.060 | 0.80791, 1.45143, **1.06069** | 1.060 should be 1.061 (truncated) |
| § 3: misses "0.0055 in P, 4% above the pass line" | 0.8134 − 0.8079 = 0.0055; 1.4514 / 1.3956 = 1.040 | match |
| § 3: curve 0.871, 0.840, 0.814, 0.806, 0.806; Pile-text copy 0.69-0.74 | 0.8707, 0.8399, 0.8141, 0.8057, 0.8062; 0.688-0.739 | match |
| § 4 table: P, ratios, sizes, 50-step means | all P and 50-step means match; several third decimals are truncated rather than rounded: emoticon 2,500 stochastic 0.9876 (0.988, not 0.987); order 1 ratio 0.8809 (0.881), stochastic 0.9926 (0.993); r1 2,500 ratio 0.8909 (0.891), stochastic 0.9937 (0.994); r1 order 1 ratio 0.9929 (0.993); r0 1,000 size 0.01590 (0.0159); r0 2,500 ratio 1.0097 (1.010); r0 5,000 ratio 0.9555 (0.956), stochastic 1.0036 (1.004), size 0.02359 (0.0236); r0 order 1 ratio 1.0898 (1.090), stochastic 0.9938 (0.994), size 0.01947 (0.0195) | no claim changes |
| § 4: plateau rises 0.019, 0.176, 0.262 | 0.91316 − 0.89421 = 0.0190; 0.79293 − 0.61730 = 0.1756; 0.88264 − 0.62114 = 0.2615 | match |
| § 4: r1 held-out curve 1,500 to 5,000 ranges 0.852-0.933 | 0.8524 (4,500) to 0.9330 (3,000) | match |
| § 4: step-1,000 reproduction of A3 to four decimals | 0.56672, 0.87530, 0.52098 vs 0.5667, 0.8753, 0.521 | match |
| § 4: second batch order −0.042/+1.8%, −0.002/+11.4%, −0.032/+7.9% | −0.04162/+0.01783, −0.00242/+0.11448, −0.03198/+0.07937 | match |
| § 5: every gap, matching and filed ratio (16 rows) | recomputed `pile_gap / teachers.*.pile_gap_w0` and `pile_gap / nothing.pile_damage` | match |
| § 5: r1 training loss 0.0085-0.0124 from step 250; W₀'s KL(teacher ‖ W₀) 0.0091 | 0.00851-0.01237; 0.009078 | match |
| Provenance: run times, hashes, eight saved files of ~60 MB | log 12:52:33-13:57:13; hashes match; eight files of 60,299,0xx bytes | match |

One further reading: the "Not run" list says W₀'s P(target) on r2 was not read in this job. The log's harness line for
the r2 do-nothing state (line 469) prints P with the components alone, Δ excluded, at 0.004, against the plan's 0.0034.
That is close to, but not exactly, W₀'s P; the item can say so.

The G1 row in § 6 lists "plain-text damage above 1e-3 with gap at most 1e-4" as part of the rule. That condition is
not in the filed G1 text; it was added to the code in c7ff923 (12:14, after filing, before the run). It only makes G1
stricter, so nothing changes, but the table should say "added in code before the run".

## 2. Each prediction against its filed rule

| id | Filed rule, applied to the raw file | Verdict in the analysis | Check |
|---|---|---|---|
| P1 | five r1 copies at 1,000, `pile_damage ≤ 2 × do-nothing pile_damage` or ≤ 0.002; all five present | holds (0.845, 0.997, 0.418, 0.945, 1.739) | correct |
| P2 | r2 A: kept = 0.80791 ≥ 0.86337 − 0.05 = 0.81337? no; faith inconclusive ≠ pass → "neither" | fails, neither | correct |
| P3 | as P2 for B: 0.73939 < 0.81337; faith fail → "neither" | neither, against the lean | correct |
| P4 | G3 pass; Adam S verdict "fail" → map, LoRA S "fail" → map; both map → "map" | map, on the lean | correct |
| P5 | step-5,000 single held-out P and verdict; else the 50-step mean of the per-step held-out P at 5,000 minus at 2,500, against 0.03 | emoticon learnt (0.9083, pass); r1 plateau (0.8870; rise 0.0190); r0 rising (0.8193; 0.1756); fails | correct |
| P6 | per task at 2,500: \|ΔP\| ≤ 0.05 and \|ratio₁/ratio₀ − 1\| ≤ 0.25 | holds on all three | correct |

**P5, the filed plateau rule against the matching-denominator reading.** These are two different things, and the
analysis keeps them apart correctly. The filed plateau rule reads P only: the 50-step mean of the per-step free
held-out P (`summary` in `amend6.py`), at 5,000 minus at 2,500. It puts r1 at a plateau (0.019 < 0.03) and r0 rising.
The matching-denominator reading is not part of P5 at all; it is a second way of computing the plain-text gap that the
decision tree, not the scorer, uses. Under the filed denominator (KL(W₀ ‖ teacher), as the plan's "teacher's plain-text
damage" says) the step-5,000 gaps are 0.807, 0.787, 1.482; under the matching one (KL(teacher ‖ W₀)) 0.753, 1.123,
1.521. Both are far above the tree's 0.1 on every task, so the branch is the same either way. The analysis states this.
The only caution: the analysis's "The two readings disagree most on r1's LoRA … because the LoRA's two KLs differ by
30%" is right, but r2's two KLs differ more (0.170 against 0.240); the r1 disagreement is largest only because its
ratios are near 1.

**Is "emoticon learnt" against "r1 plateau" separated by checkpoint noise?** Yes, as worded. The single readings sit
0.008 above and 0.013 below the bar; r1's held-out readings every 250 steps from 1,500 on span 0.852-0.933, a range six to ten
times either margin; and the 50-step means swap the two (0.883 and 0.913). The filed categories stand, and P5 fails by
its rule. Two qualifications the text should carry: emoticon's mean sits below its single reading partly because it is
still climbing (0.868 at 4,750, 0.908 at 5,000), not only from noise; and the plan's clause marking categories within
0.05 of a bar "unsettled" fires only if P6 fails, which it did not, so r1's "plateau" is formally settled even though
it is within 0.013 of the bar.

## 3. Claims, verdict and corrected wording

| Where | Claim | Verdict | Corrected wording |
|---|---|---|---|
| Opening paragraph | "The copies it makes harm ordinary text about as much as the fine-tunes they copy" | too broad | "The five scored copies, of three fine-tunes of one task, harm ordinary text 0.4 to 1.7 times as much as the fine-tune they copy." |
| Opening paragraph | "plain SGD fails as its optimiser on every kind of fine-tune" | overreaches: one task (r1), one rate (0.03), one batch order | "plain SGD at the one rate tried fails as its optimiser on all three fine-tunes of one task: SGD, Adam and LoRA." |
| Opening paragraph | "the rare target is still placed neither faithfully nor fully" | too strong: the task-text copy is inconclusive, not unfaithful | "the rare target is not yet placed: the closer copy missed both bars narrowly, on one batch order of a fine-tune that itself reached only 0.86." |
| The answer (analysis and notebook) | "Its copies change ordinary Pile text at most 1.7 times as much as the fine-tunes they copy." | false as a general statement: the unscored r0 Pile-text LoRA copy reads 2.00 (1.95 at 1,000) and the plain-SGD copy of the LoRA 2.61 | "The five copies the prediction scored change ordinary Pile text 0.42 to 1.74 times as much as their fine-tune; two unscored LoRA copies read 2.0 and 2.6." |
| The answer | "it failed the adversarial check on every fine-tune tried" | literally true; scope missing | add "all three on r1, at SGD's rate 0.03". |
| The answer, notebook § rare target | "both sat far closer to the unchanged model than the fine-tune did" | ambiguous: on plain text the Pile-text copy is barely closer (0.153 vs 0.170) | "both read far lower adversarial ratios than the fine-tune left in Δ: 1.45 and 2.48 against 11.3." |
| § 1 | "The Pile-text copies sit at the fine-tune's level" | true of the full-weight ones only (0.997, 0.945); the LoRA Pile-text copies read 1.17-2.00 | "The full-weight Pile-text copies sit at the fine-tune's level." |
| § 2 | "The plain-SGD copies' sizes … are all above the 0.009 above which E21 found SGD component training fails" | true but misleading as an explanation: the Adam task-text copies of the same teachers are the same size (0.017-0.021) and pass | add "The Adam copies are the same size and pass, so size does not separate them here." |
| § 2 heading, What we learned, notebook | "Plain SGD … fails late whatever the teacher"; "the late failure belongs to the optimiser" | "late" is shown only for the SGD teacher (E34: pass at 100); on the Adam and LoRA teachers the adversary read step 1,000 only. "Whatever the teacher" is three teachers of one task | "fails at step 1,000 on all three r1 teachers, as it failed late on the SGD teacher in E34; so the failure follows the optimiser at this rate, not the SGD teacher." |
| Notebook § 2 | "The plain-SGD copies keep the behaviour, at 0.99 or more" | the SGD teacher's copy reads 0.986 | "at 0.986 or more". |
| § 3, notebook Surprises | "is the largest update here at size 0.085"; "It is also the largest update read here" | wrong: the LoRA teachers' sizes are 0.120 (emoticon), 0.107 (r1), 0.084 (r0) | "the largest full-weight fine-tune here, 0.085 against 0.029 and 0.033." |
| § 3, How compelling, notebook | "each inside Amendment 4's batch-order spread of 0.032 in P and 25% in ratio"; notebook "well inside" | uses the Pile-text copies' spread for a task-text copy. A4's task-text copies moved at most 0.005 in P and 11% in ratio, at P near 1; no spread exists at P 0.8 | "The P miss, 0.0055, is about the largest batch-order shift Amendment 4 saw for task-text copies (0.005, at P near 1); the ratio miss, 4%, is inside their 11%. With no spread measured at P 0.8, the category is not settled by one batch order." |
| What we learned, notebook | "The rare target remains out of reach of every map tried with Γ frozen" | stronger than the evidence, and contradicted by the next clause ("missing inside the noise") | "No Γ-frozen map tried has placed the rare target by the filed rule. Teacher training's task-text copy came closest, missing both bars narrowly on one batch order, so whether it is out of reach is not settled." The tree's card wording ("no Γ-frozen map tried places a rare target") can stand as the filed branch text, with this qualifier. |
| What we learned, notebook | "Plain text carries a rank-8 LoRA's trigger behaviour slowly and to about the bar" | fair for emoticon and r1; r0 is at 0.82 and rising | add "on two tasks of three; on r0 it reached 0.82 by 5,000 steps, still rising." |
| What we learned | "It does not carry the LoRA's own change to plain text" | true on r1 and r0; on emoticon the copy closes a quarter (0.75 of W₀'s distance) | "On r1 and r0 it does not carry the LoRA's change to plain text, ending farther from the LoRA than W₀ is; on emoticon it closes a quarter of the distance." |
| What each stage adds (d), Surprises | "none moves towards the LoRA on plain text"; "The LoRA copies do not move towards the LoRA on plain text" | wrong for emoticon (0.873 → 0.753) and, in absolute terms, every gap fell from 1,000 to 5,000 | as the notebook's Surprises already says: "close a quarter at best (emoticon), and on r1 and r0 end no closer than W₀." |
| § 5, notebook | "the LoRA copies move as far from W₀ on plain text as the LoRA does, 1.23 and 2.0 times its damage" | "as far" understates | "at least as far … 1.23 and 2.0 times". |
| Notebook Surprises | "the plan had expected the distance to shrink as the trigger behaviour was learnt" | the plan filed no such expectation; its tree only had a branch for a small gap | drop the clause, or use the analysis's wording: "Not filed; the plan's tree treated a large gap as one possible branch." |
| No filed prediction reversed | — | correct. P2 is a miss of a medium-low prediction, P3 landed against a two-sided lean, P5 a narrow miss on r1; all are listed under Surprises | stands |

Claims checked and fair as written: P1 holds on all five; the scaled adjoint states damage less because they lose the
behaviour; P4 reads "map"; the plain-SGD failures are large (1.77 to 1.95 times the fail line) and, since a fail under a
weak adversary is a lower bound, they do not depend on the one-start adversary being strong; every adversarial read
of a Pile-text LoRA copy passed (0.87-1.09); a second batch order agrees on all three LoRA tasks; the full-weight
copies close two thirds or more of the plain-text distance (matching ratios 0.08-0.33); the r2 do-nothing state reads
11.3; the "How compelling" list names one batch order, one start, one rate and the r2 teacher's own P of 0.863.

**The r2 teacher below 0.9.** The analysis says it, once, in "How compelling". It matters more than a caveat: every r2
state's full verdict reads "below P 0.9", so P2 and P3 could only be scored through faith alone and the relative
"kept" line, and the teacher is the most damaging update in the run (plain-text damage 0.170, window damage 0.133,
ratio 11.3). A copy of a weak, damaging fine-tune is a different test from E34's copies of 0.98+ teachers. The opening
and the card should say "of a fine-tune that itself reached only 0.86".

## 4. Decision-tree branches

| Branch the analysis names | Plan's filed tree | Check |
|---|---|---|
| r1 and r0 Pile-text LoRA copies to 10,000 steps, ~$1.5 each | "Plateau with a larger gap, or rising: that task's student B to 10,000 steps (approved branch, about $1.5)". r1 is plateau with gap 0.79 (filed) / 1.12 (matching) > 0.1; r0 rising | **matches**. Two notes: the run needs its own predictions filed first (Rules), and since r1 sits at the bar, its scoring rule should be fixed in advance (the analysis's "a mean over checkpoints decides more than steps do" belongs in that filing, not after). Emoticon "learnt" triggers nothing, correctly; "learnt, not passed" did not occur |
| A pending-decision card for the rare target | P2 "kept but not faithful, or neither: … a pending-decision card, since it bears on the 'Create' reading and on updating Γ" | **matches**; "faithful but not kept" (student A to 3,000 steps) did not fire, since faith is inconclusive, not pass. The curve (P falling from 0.871 to 0.806) argues against that branch anyway |
| "A cheap check first: the task-text copy at step 100 under the adversary … then a second batch order of it at step 1,000" | not in the tree | **outside the tree**: both are new GPU runs (the step-100 state was not saved; only step 1,000 was), so they need `/design` and the user, or go on the card as options. They are not covered by the approved branch |
| `decisions.md` row "teacher training uses Adam", and the optimiser subgoal's reading | P4 "map": "Adam is the only optimiser for teacher training; the lead adds a `decisions.md` row and the optimiser subgoal's reading" | **matches**. The row should carry the scope: "Teacher training uses Adam at 1e-5. Plain SGD at 0.03 failed the adversarial check at step 1,000 on r1's SGD, Adam and LoRA teachers (E34, A6); revisit if another SGD rate is tried." |
| (not named) | P1 holds: "Plain-text harmlessness joins teacher training's card and the matrix" | **missing from What next** in both the analysis and the notebook; add it as item 0 (a log and matrix edit, no run) |
| (not named) | P6 holds | nothing fires, correctly |
| Notebook What next | drops the `decisions.md` item | fine for the notebook, since it is the lead's, but the analysis must keep it |

## 5. Wording allowed above the log's cards

- **Harmless on plain text.** On ordinary Pile text, the five copies scored change the model 0.4 to 1.7 times as much
  as the fine-tune they copy, all three fine-tunes being on one task. The one copy above its fine-tune is a LoRA's
  task-text copy, as on the task's own windows.
- **The map needs Adam.** Teacher training with plain SGD at 0.03 failed the adversarial check on all three fine-tunes
  of one task tried (SGD, Adam and a LoRA) while keeping the behaviour; with Adam the same copies pass.
- **The rare target is not yet placed.** On " in" → " Fig", with a fine-tune that itself reached only 0.86, neither
  copy both kept the behaviour and passed; the closer one missed both bars narrowly, on one batch order.
- **A LoRA from plain text alone.** Given 5,000 steps, plain text carried the LoRA's trigger behaviour to about the bar
  on two tasks of three (one just above, one just below, apart by less than checkpoint noise) and to 0.82, still
  rising, on the third. On two of the three it did not carry the LoRA's change to ordinary text at all.

Not allowed above the cards: "harmless at most 1.7 times" without "the five scored copies"; "fails on every kind of
fine-tune" or "whatever the teacher"; "out of reach" for the rare target; "the largest update"; "none moves towards the
LoRA".

## 6. Not checked

- The figures `amend6_*.png` and `plot_amend6.py` were not opened; only the numbers they are said to show.
- Whether `train_student_seeded`'s per-step free held-out P uses the same rows as `read_state`'s held-out P: they differ
  by up to 0.002 at the same step (r1 at 5,000: 0.8887 curve, 0.8870 state), which does not change any category.
- The cost of $6.9; no billing record was read.
- The upstream E21, E26 and E34 claims the analysis cites, beyond the E34, A3 and A4 numbers in § 1.
- `check.py` was not run on this file.
