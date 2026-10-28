# E39 Amendment 2: independent check of the analysis, notebook section and TRIED row (review)

*2026-10-28, independent agent, read-only; no python run, no model loaded. Scope: commit 63aeb7a, i.e.
`reports/analyses/2026-10-28_analysis_e39-amend2.md`, `notebooks/E39_broad_fragility.ipynb` → "Amendment 2", the
"E39 Amendment 2" row of `experiments/TRIED.md`, and `figures/amend2_{labels,lora,size}.png` with `plot_amend2.py`'s
state list. Numbers recomputed with `jq` from `experiments/broad_fragility/results/amend2.json`, `amend1.json`,
`broad_fragility.json` and `experiments/teacher_spread/results/teacher_spread.json`. Per-block ratios recomputed from
the `[faith]` lines of `results/large/amend2.log`, `amend1.log` and `run.log`: each 128-row block's PGD KL over the
unchanged decomposition's same block in the same job. Filed rules: `results/predictions_amend2.json`. Plan:
`reports/plans/2026-10-27_plan_broad-fragility.md` → "Amendment 2 (2026-10-28)". Approval: `decisions.md` line 47.*

I recomputed every number the write-ups quote and scored every prediction by its filed rule. All of them match the
files, and every tree branch fired as filed. The run of next work the analysis names, a rank-8 LoRA trained with plain
SGD (~$1.5), is the branch the plan approves for this outcome. The "size against form" arithmetic is right: every
state sits on the same rows with the same unchanged-model reading, and the damage band 0.09-0.13 spans 1.269 to 2.700.
Its reading goes too far in two places, though. At the Adam teacher-trained decomposition's own damage every state
read fails, so the filed D reading still stands as a possible account of the Adam fail. It fails as an account of the
SGD teacher-trained fails and of the middle band. Four phrases claim a cause that no test isolated: "how the update was
made decides it", "rank is not why", "size explains the ends" and "the optimiser matters". Five small number or wording
slips are listed in section 1.

## 1. Numbers

**Matched exactly** (to the digits quoted). These are the state values and every derived ratio below:

- Controls: spread 0.0748, blocks 0.6794 and 0.7182, gain 0.1300, check 2.2011, kept 0.9978, do-nothing 4.8678, LoRA
  gain 0.1479 with check 1.2726, second rows' spread 0.0621 with Adam 1.883, and the cap's largest share 2.29e-9
  (pooled 1.04e-10).
- All 15 labelled states (gain, damage, gap and the divergence flag) and the two chosen states. The rule picked
  correctly: Adam 0.1343 is 0.0044 from 0.12995 (next nearest 0.1431, 0.0131 off), and SGD 0.1253 is 0.0046 off (next
  0.1217, 0.0082 off).
- Every `reads` entry quoted: check, damage, gap, kept, size, stochastic ratio and teacher-training KL end. Gain ratios
  1.034, 0.964, 1.011 and 1.139.
- The capped rate rule: 0.03 at gain 0.1314, against 0.1237 and 0.0917. The plain SGD rerun at 0.01 gave 0.1367, with
  shares beyond rank 8 of 0.395 pooled and 0.551 at most.
- The damage search: seed 0 used 6 reads and stopped at x 4.967, damage 0.2436, 4.1% over. Seed 1 used 4 reads and
  stopped at x 4.757, damage 0.2405, 2.9% over. Excess multiples (16.461 − 1)/1.2011 = 12.9 and 21.85, so "13 and
  22". Weight size 0.181 and 0.174 against 0.036, so "about five times".
- Per-block ratios. Adam teacher-trained decomposition 1.69 and 2.68, on the second rows 1.80 and 1.97. Plain SGD
  (Amendment 1) 1.16 and 2.93. Rank-8 SGD 1.25 and 2.95. LoRA seed 0 1.11 and 1.42, on the second rows 1.17 and 1.24.
  LoRA seed 1 1.09 and 1.44 (1.4357), on the second rows 1.22 and 1.26. Labelled Adam 1.95 and 12.2. Random moves
  17.3/15.7 and 23.3/31.0. The check is the ratio of block means, and each one reproduces from its blocks (for example
  (0.8463 + 2.1192)/2 / 0.6988 = 2.1218).
- Provenance: the sha256 prefixes of `amend2.json`, `predictions_amend2.json`, both logs and the five `.pt` files all
  match. The predictions were committed once, in 1626f27 at 14:10:49, before the log's first line at 14:33:50, and
  never changed. The stdout log ends "FINISHED (exit 0)".

**The size-against-form reading, state by state.** The states are comparable. The unchanged decomposition's per-block
PGD KL is 0.6794 and 0.7182 in the E39, Amendment 1 and Amendment 2 jobs alike. The Adam teacher-trained
decomposition's damage is 0.23386794626119772 in all three jobs and in E37's `lower_s0`, so damage is read on the same
64 rows. E39's LoRA teacher-trained state reads the same damage and check as this run's seed 0, to every digit. Every
check is over its own job's unchanged-decomposition reading, and that reading is identical across the jobs.

| damage | check | verdict | state | file |
|---|---|---|---|---|
| under 0.06 | 0.946-1.069 | all pass | W₀; E39's 3 weight-matched moves (0.003-0.010); scaled adjoint splits Adam 0.050 (E37), LoRA 0.024, SGD 0.026; reverts common 0.021, MLP 0.025, random 0.054 | E37, E39, A1 |
| 0.086 | 2.623 | fail | SGD do-nothing | A1 |
| 0.090 | 2.067 | fail | plain SGD teacher-trained | A1 |
| 0.098 | 2.072 | fail | E39 revert layer 3 | E39 |
| 0.111 | 1.445 | between | E39 revert, rarely important half | E39 |
| 0.121 | 2.653 | fail | LoRA do-nothing | E39 |
| 0.124 | 1.273 | pass | LoRA teacher-trained, seed 0 | A2 |
| 0.125 | 1.275 | pass | labelled SGD, 0.03, step 100 | A2 |
| 0.126 | 2.122 | fail | rank-8 SGD teacher-trained | A2 |
| 0.127 | 1.269 | pass | LoRA teacher-trained, seed 1 | A2 |
| 0.128 | 1.897 | fail | E39 revert layer 2 | E39 |
| 0.133 | 2.700 | fail | E39 revert attention | E39 |
| 0.168 | 1.355 | pass | E39 revert layer 0 | E39 |
| 0.175 | 1.747 | fail | E39 revert layer 1 | E39 |
| over 0.2 | 1.736-27.24 | all fail | Adam teacher-trained, orders 0 and 1 (2.201, 1.884); E37 `plain_s0` (0.224, 1.736, not plotted); Adam do-nothing 4.868; damage-matched moves 16.46, 27.24; labelled Adam 7.242 | A2, E37 |

So "under 0.06 all pass, over 0.2 all fail, 0.09-0.13 spans 1.27-2.70" is correct. The upper edge holds only after
rounding: the attention revert sits at 0.1326, and the figure shades 0.088-0.136. Two limits on what this table can
show. First, the high end holds only states made from the Adam fine-tune and random moves scaled to its damage. No
state from another recipe reaches 0.2, so the high end cannot tell size from origin. Second, the low end is reverts,
adjoint splits and tiny moves, which remove most of the update or move very little. The ends are post-hoc groupings
across four jobs, not a test.

**Slips:**

1. Analysis §3: "at damage near 0.125, three states pass and one fails". In fact, between 0.121 and 0.128 three pass
   (1.269, 1.273, 1.275) and three fail: the LoRA do-nothing at 2.653, the rank-8 SGD state at 2.122 and the layer-2
   revert at 1.897.
2. Analysis §1: "Its later states here reach gains of 0.15 to 0.17 with damage up to 0.23". This holds only at rates
   0.01 and 0.03 (gains 0.146-0.166, damage 0.10-0.23). At 0.1 the later states have gains of 0.057-0.082 and damage
   of 0.29-0.37.
3. Notebook: "a gain 14% higher for the LoRA" than the capped SGD. 0.1479/0.1314 = 1.126, so 13%. The 14% is the LoRA's
   gain over the Adam fine-tune's.
4. Analysis §3, "which is where the LoRA, SGD and labelled states all sit", and the notebook's matching line: only the
   labelled SGD state sits in the middle. The labelled Adam state is at 0.270.
5. Noise floor (§ "How compelling"): "The labelled SGD state, at 1.28, sits 0.12 under the pass line, more than the
   seed spread". That is true of the adversary-seed spread, 0.075. But the labelled state was read at one batch order,
   and E37's batch-order spread of 14% is ±0.18 at 1.28, which would reach the zone between the lines. Say so.

Minor items. Readings that missed their filed expected ranges, as distinct from the rules, are given as numbers without
saying they missed. The random moves' behaviour kept was filed at about −0.1 to −0.5 and read −1.86 and −1.88.
Labelled Adam's damage was filed at 0.15-0.25 and read 0.270. In `amend2_labels.png` the two panel titles overlap
("teacher trainithe SGD one…").

## 2. Predictions against the filed rules

| id | filed rule (gates) | inputs | my verdict | run's | analysis |
|---|---|---|---|---|---|
| controls i-iii | spread ≤ 0.1, blocks within 1%; gain ±0.005, check ±3%, kept ±0.01; do-nothing ≥ 1.61 and ±3% of 4.8678 | 0.0748, 0.6794/0.7182; 0.12995, 2.2011, 0.99779; 4.8678 | pass | pass | pass |
| G_C | gain ±0.005 of 0.1479, check ±3% of 1.2726 | 0.14792, 1.27260 | holds | true | pass |
| G_rows2 | spread ≤ 0.1 and Adam ≥ 1.61 | 0.0621, 1.8826 | holds | true | pass |
| cap | largest share ≤ 1e-4 | 2.29e-9 | holds | — | pass |
| P9 | check ≥ 1.61; gain within 30% | 7.2424; 1.034 | holds | holds | holds |
| P10 | check ≥ 1.61; gain within 30% | 1.2754 (≤ 1.40, so pass); 0.964 | misses (the opposite verdict) | misses | misses, reversed |
| P11 | check ≥ 1.61; gain within 30%; cap held | 2.1218; 1.011; 2.29e-9 | holds | holds | holds |
| P12 | check ≤ 1.40; gain within 30% | 1.2688; 1.139 | holds | holds | holds |
| P13 | both ≤ 1.40; scored only if G_rows2 | 1.2066, 1.2425 | holds | holds | holds |
| P14 | mean over seeds within 5% ≥ 1.61 | both matched (4.1%, 2.9%); mean 21.851 | holds | holds | holds, far above 2-5 |

Every gate was evaluated, none was degenerate, and no verdict depends on a tie at a line.

## 3. Claims and corrected wording

What holds as written:

- **The A reading.** The labelled SGD pass is at step 100, the step the gain rule chose. Steps 300 and 1,000 were not
  read. The analysis says so in §1, in What next and in Not run, and the notebook says so too.
- **The causes of the SGD pass.** They are "named, not asserted" in the Surprises.
- **The LoRA repeat (C).** It is correct as stated, with seed and batch order changed together.
- **The cap was a real change.** 39% of the update lay beyond rank 8, and the cap left at most 2.3e-9.
- **Scope lines.** The analysis gives one task, one batch order and one adversary start in §1, Coverage and Not run.

Corrections. Each item gives the claim, where it appears, what is wrong, and the wording to use instead.

1. **"so rank is not why"** (The answer; notebook answer).
   - What is wrong: one way of capping was tested, by cutting after each step, at one rate and one seed. The capped
     run's rate (0.03) also differs from plain SGD's (0.01).
   - Use: "Cutting a plain SGD fine-tune back to rank 8 after every step does not make its teacher-trained
     decomposition pass (2.12). A rank-8 cap alone does not explain the LoRA's pass, for this way of capping at one
     rate and one seed."
2. **"the decomposition can hold this behaviour in at least one way"** (The answer; notebook).
   - What is wrong: the state holds the mean gain, not the fine-tune's outputs. Its gap is 0.135, half the unchanged
     model's 0.272.
   - Use: "Trained on the labels, a frozen-Γ decomposition reached 96% of the fine-tune's mean gain and passed (one
     state, SGD at step 100). It does not reproduce the fine-tune's outputs."
   - The same applies to §1's "'no state of this decomposition can carry the broad behaviour faithfully' is false".
     Read it as "no frozen-Γ state reaches this task's mean gain and passes" is false.
3. **"how the change/update was made decides it"** (The answer, §3, What we learned, plain-words paragraph, notebook,
   TRIED row).
   - What is wrong: no test isolated this. States at matched damage differ in recipe, and also in gain, weight size,
     which matrices move, subspace and batch order.
   - Use: "At the same mean damage on plain text, states made in different ways land on both sides of the lines, so
     damage alone does not set the verdict there. Which difference between the recipes matters is not tested."
   - Plain words: "nor a matter of how much the outputs move" becomes "nor set by how much plain-text outputs move
     alone".
4. **"Its conclusion does not [stand] … contradicted twice"** (§3, Surprises, notebook, TRIED "does not stand against
   the other states"). The second contradiction does not hold. The filed reading says any move this large fails, and
   the random moves do fail; their larger excess shows the check's size depends on direction, not that the verdict at
   that damage does. The first contradiction holds only for the SGD teacher-trained fails and the middle band. At about
   0.23, every state read fails, so "size can explain the broad fail" is still possible for the Adam fail. Use: "D's
   fail branch fired as filed. Its reading survives as a possible account of the Adam teacher-trained decomposition's
   fail: every state read over 0.2 nats fails, though all of them come from the Adam fine-tune or from random moves
   scaled to it. It fails as an account of the broad fails in general, because the SGD teacher-trained decompositions
   fail at damage where three other states pass. Of its parenthesis, 'smaller' fails (the LoRA's state is at 0.124,
   the failing rank-8 SGD state at 0.126) and 'better placed' is untested."
5. **"Size explains the ends" / "sets the ends" / "What size does explain"** (§3, What we learned, notebook, commit
   title). The table in section 1 is a co-occurrence across four jobs, and the high end holds only Adam-derived states
   and random moves. Use: "Damage and verdict line up only at the ends: every state read under 0.06 nats passes and
   every one over 0.2 fails. Between 0.09 and 0.13 the check runs from 1.27 to 2.70."
6. **"training's directions are far less fragile than random ones"** (§3 second paragraph, Surprises, notebook
   Surprises). §3's later paragraph gives the caveats, but the Surprises lines drop them. Use: "At the same mean damage,
   the Adam teacher-trained decomposition reads far below random moves of the alive U and V (2.20 against 16.5 and
   27.2). Those moves needed about five times its weight change and lost the lowercase gain (behaviour kept −1.86 and
   −1.88)."
7. **"the update's form, and the optimiser, matter"** (the tree's A-with-C bullet). The labelled Adam and SGD states
   differ in optimiser, rate, step (1,000 against 100) and damage (0.270 against 0.125). Use: "With labels, SGD at step
   100 passed and Adam at step 1,000 failed. Optimiser, step and damage all differ between the two, so the optimiser is
   not isolated."
8. **"Three simple reasons … ruled out"** (What we learned, notebook) and **"ruled out as its cause"** (TRIED). Use:
   "Stable rank, a rank-8 cap and a smaller mean damage each fail to explain the LoRA's pass on their own, at one rate
   and seed each, on this task."
9. **What next 1: "If it fails like the SGD teacher-trained decompositions, the Adam optimiser does."** Use: "A pass
   would point to the factorisation and a fail to the fine-tune's optimiser, among the differences listed in §2, which
   also include the subspace and the gain."

## 4. Tree branches

| branch | filed condition | inputs | fired (code) | fired (mine) | quoted correctly |
|---|---|---|---|---|---|
| Stage 0 | stop on a miss | all pass | — | no stop | yes |
| G_C | report a miss at once | holds | — | no miss | yes |
| A, Adam | P9 verdict | fail | fail | fail | yes |
| A, SGD | P10 verdict | pass | pass | pass | yes |
| B | P11 verdict | fail | fail | fail | yes |
| C | seed 1 and the second rows | 1.269; 1.207/1.243 with G_rows2 | stands | stands | yes |
| D | P14 verdict | fail | fail | fail | yes; reading discussed in §3 item 4 |
| A with C | A's verdict and C's | Adam fail, C stands | fail/stands | fail/stands | yes, with a qualifier |

Two more points about the tree:

- **A with C keys on Adam's half.** That follows the plan, where "A" is P9 and SGD "is read beside Adam's". The tree
  has no wording for the two optimisers disagreeing. Had SGD's half been used, "pass/stands" ("teacher training's copy
  of full-matrix fine-tunes is what fails") would also fit B and C. The analysis is right to qualify the fired text and
  not lean on either.
- **The plan's tree text names no combined reading for "Adam fails, SGD passes".** The analysis says so correctly. The
  filed `readings.A_sgd.pass` is a one-line verdict only.

**The named next run is the approved branch.** The plan's B bullet reads: "*Fail*: the cap does not explain it. If C
stands, this plan's next branch is a rank-8 LoRA trained with plain SGD on the lowercase task, teacher-trained and
read, to tell the factorisation from the Adam optimiser: about 15 min, about $1.5, RunPod H100, approved with this
plan." Both conditions held: B failed and C stood. The user approved Amendment 2 (`decisions.md` line 47), and the
branch is written into it. At ~$1.5 and ~15 min it is not gated. RunPod H100 rather than the cluster's debug QoS is
justified: stage 0's per-block controls are that card's, and the plan says an L40 would stop there. Before it runs:

- It needs filed predictions; `predictions_amend2.json` holds none for it.
- It needs a rate grid with a source. The plan gives no grid; E27's plain-SGD LoRA on emoticon and E39's gain rule are
  the nearest sources.
- It needs the one code review.
- It needs a plain what and why in chat first (memory "Explain tree branches before running").
- Pre-mortem: plain SGD on a LoRA whose second factor starts at zero may learn slowly. The grid must be able to reach a
  gain within 30% of 0.130, or the prediction goes unscored.

## 5. Wording allowed above the log's cards

Allowed:

- "The rank-8 LoRA's teacher-trained decomposition passed the check again at a second seed and batch order (1.27) and
  on 256 new Pile rows (1.21, 1.24), where the Adam one still fails (1.88)."
- "Cutting a plain SGD fine-tune back to rank 8 after every step did not make its teacher-trained decomposition pass
  (2.12), so a rank-8 cap alone does not explain the LoRA's pass (one rate, one seed)."
- "Trained directly on the task's labels, the subcomponents passed with plain SGD at step 100 (1.28, 96% of the
  fine-tune's gain), the state the gain rule chose; later steps were not read. With Adam, the chosen state (step 1,000)
  failed at 7.24."
- "So on this task the broad fail is not a fixed limit of the decomposition: a frozen-Γ state can reach the task's gain
  and pass."
- "Random moves matched to the Adam teacher-trained decomposition's damage on plain text failed far worse (16.5, 27.2)."
- "Damage on plain text lines up with the verdict only at the ends (under 0.06 every state passes, over 0.2 every one
  fails). Between 0.09 and 0.13 the check runs from 1.27 to 2.70, so damage alone does not set the verdict there."
- "Next, approved with the plan: a rank-8 LoRA trained with plain SGD, teacher-trained and read (~$1.5)."

Not allowed:

- "how the update was made decides it"
- "rank is not why"
- "size explains / sets the ends"
- "the optimiser matters"
- "the decomposition can hold this behaviour" (say "the task's gain")
- "D's reading is contradicted" without the Adam-fail qualifier
- "training's directions are less fragile than random ones" without the weight-size and lost-behaviour caveat
- "ruled out as its cause"
- "a passing broad state exists" without "at step 100, one batch order"
- "the copy" for any state

## 6. Not checked

- The run's code (`amend2.py`, `src/commute/rank_cap.py`, `fragility.random_move`). I read only `plot_amend2.py`'s
  state list and band limits.
- The contents of the saved `.pt` files (sha256 only).
- The cost ($3.65) and the pod's up-time beyond the log timestamps.
- `check.py` on the analysis and notebook (no python on the dev node).
- The notebook's code cells and outputs.
- The plan's claim that the rules were tested on 314,928 combinations.
- `docs/research_log/PENDING.md`.
- That the figure's plotted positions match the data, beyond reading the images by eye.
