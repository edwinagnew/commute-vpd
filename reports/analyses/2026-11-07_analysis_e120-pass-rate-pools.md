# E120: does fine-tuning on problems the model never solved teach it to solve unseen ones?

*2026-11-07, qwen-rl-lead's writer. Provisional throughout: the pools are 77 a side, under the 100 the plan needed, and
the run had no independent check, as the fast lane allows below the review threshold. Plan with Amendments 1-2
`reports/plans/2026-11-06_plan_e120-pass-rate-pools.md`; filed rules `experiments/qwen_pass_pools/results/predictions.json`;
decision `decisions.md`, last row (ask 13: the four fine-tunes ran on the 77-a-side pools as a provisional row, and the
backup E128 goes ahead). Sampling and pools on RunPod, $15.72 in all; the four fine-tunes on the MATS cluster, `debug`
L40, jobs 54969 (known, seed 0), 55011 (known, seed 1), 55012 (new, seed 0), 55013 (new, seed 1), $0. Grading of the
stored answers on CPU, $0, by `experiments/qwen_pass_pools/grade_arms.py` into `results/grades.json`. Figures from
`experiments/qwen_pass_pools/plot.py`; notebook `notebooks/E120_qwen_pass_pools.ipynb`. The decomposition readings of
these updates are qwen-decomp-lead's and are not in this report.*

We split real maths problems by whether the unchanged Qwen3-1.7B could already solve them, fine-tuned it on the human
solutions of 62 problems from each side, and read the result on problems held out from training. Both fine-tunes
reproduced every training problem, the fine-tune on never-solved problems solved almost none of the held-out
never-solved problems, and both cut GSM8K test accuracy from 0.78 to between 0.42 and 0.55, while the never-solved pool
came out at 77 problems rather than the 100 the plan needed. For `GOALS.md` this hands qwen-decomp-lead a provisional
pair of updates of known kind, but both are heavily overfit updates on a small set, so a reading that tells them apart
may be reading the pool's difficulty or solution length rather than new skill against existing skill.

## The answer

No. Fine-tuning on the human solutions of 62 problems the model never solved in 64 samples did not teach it to solve
held-out problems of that kind. At step 100 it got 5% and 2.5% of held-out samples right in the two seeds, all on one
problem, against a filed bar of 10%; it reproduced all 62 training problems. Both fine-tunes damaged the model: GSM8K
accuracy fell from 0.78 to between 0.42 and 0.55. Labelled provisional: 77 problems a side, under the planned 100.

"New" in this report always means "never solved in 64 samples of at most 2,048 tokens". It does not mean the skill is
absent from the weights: with thinking on, the model solved a quarter of these problems at least once in 4 tries.

## What we were trying to test

D3 needs two updates of known kind on one real task: one that strengthens a skill the model has, and one that teaches
something it lacks. We built both from GSM8K train and MATH train levels 3-5. The unchanged model answered every
problem 16 times without thinking, and 48 more times for every problem it never solved and for up to three
"known" candidates per never-solved problem in the same stratum. A problem is **known** if it was solved in at least
36 of the 48 fresh samples, and **new** if it was solved in none of 64. Guards then removed problems whose gold label
was likely wrong, problems with 16 or more samples cut off at the length cap, and problems too long to train on.
Known and new problems were matched one for one by source, level and subject, and 20% of each pool held out.

Each pool's 62 training problems were then used for supervised fine-tuning on their human-written gold solutions, by
the recipe of the earlier GSM8K gold run unchanged: peak rate 1e-5, warm-up 10 steps, cosine over 100 steps,
64 sequences a step, fp32 forward. Two seeds per pool change only the order. 100 steps of 64 sequences over
62 problems is 103 passes over each problem (`results/train_<arm>.json` → `data.passes`). The weight change δW
was saved in fp32 at the steps where its Frobenius norm crosses 0.35 and 0.49, at step 10 and at step 100, for
qwen-decomp-lead. 0.35 is the size of the RL update and 0.49 that of the template update, the two D3 anchors already
read.

At step 100 each arm was read four ways: loss on the held-out gold solutions of both pools; 8 samples per held-out
problem of both pools at the model card's no-think settings, capped at 1,024 new tokens; one greedy answer per
training problem of its own pool; and greedy GSM8K test 0-255, strict, against the unchanged model's 0.784.

**Pool sizes and makeup.** `results/pools_amend2.json` → `meta`: n = 77 a side, 62 train and 15 held. The new pool is
43 GSM8K and 34 MATH problems; the MATH part is 5 at level 4 and 29 at level 5, 14 of them counting and probability.
The held-out part of each pool is 9 GSM8K and 6 MATH problems. Of 12,973 problems sampled, 10,413 stopped at 16
samples. The guards removed 246 never-solved problems as likely gold errors and 107 for truncation, and 16 known
problems for truncation (`meta.guard_cuts`).

## What we expected, and why

- **Pool size.** At least 100 problems a side after the guards, from a lean of 200-600: MATH level 5 has many hard
  problems.
- **Novelty check.** With thinking on, at most half the new pool solved at least once in 4 samples. Lean 0.5: thinking
  is known to recover many no-think failures.
- **Does the new arm learn?** Held-out pass@1 of at least 0.10 in both seeds counts as learning. Lean against, at 0.35:
  62 problems seen 103 times each should be memorised and carry little, as the invented-facts fine-tunes did.
- **Does the known arm leave GSM8K alone?** Within 0.05 of 0.784 in both seeds. Uncertain: the same recipe on all of
  GSM8K read 0.598 at step 100 in the filed reference run.
- **Size.** The new arm's step-100 update larger than the known arm's in both seeds: fitting unfamiliar solutions
  should take a larger change.

## Did the new arm learn to solve unseen never-solved problems?

![pass](../experiments/qwen_pass_pools/figures/pass.png)

**Reading it.** No. On the 15 held-out new problems the new arm's pass@1 is 0.050 at seed 0 and 0.025 at seed 1,
under the 0.10 bar; the unchanged model reads 0 there by construction. Every one of those correct samples is on a
single GSM8K problem, `gsm8k/train/4668`, and read by eye they reach the gold's answer of 3 by a different and wrong
calculation, 9 hours over 3 nights, where the gold has 15 over 5. The new arm answers all 62 of its training problems
correctly by greedy decoding, so by E84's filed label it is "memorised, carrying little". The known arm reads 0.008
and 0 on the same problems.

The right-hand group is the surprise. On the 15 held-out known problems, which the unchanged model solves at 0.971
over 64 samples, every arm falls to between 0.475 and 0.542. Fine-tuning on known problems roughly halved the model's
success on other known problems. The unchanged model's number comes from stage 1, through vLLM, with caps of 512
tokens for GSM8K and 2,048 for MATH; the arms' come from the training code's sampler at a 1,024-token cap. Truncation
in the arms' samples is at most 0.042, so the cap does not explain a drop of this size.

| arm | new held, pass@1 | known held, pass@1 | own training problems, greedy | held samples truncated |
|---|---|---|---|---|
| unchanged model | 0.000 | 0.971 | not read | stage 1 |
| known, seed 0 | 0.008 | 0.542 | 62 of 62 | 0.025 / 0.008 |
| known, seed 1 | 0.000 | 0.475 | 62 of 62 | 0.017 / 0.017 |
| new, seed 0 | 0.050 | 0.542 | 62 of 62 | 0.017 / 0.017 |
| new, seed 1 | 0.025 | 0.525 | 62 of 62 | 0.042 / 0.042 |

Source: `results/grades.json` → `arms.<arm>.{new_held_samples,known_held_samples,<pool>_train_greedy}.pass1` and
`.truncated` (new held / known held); `w0.{new_held,known_held}.pass1`. Per-problem counts are under `k_of_n`. The
grader is `grade.py` as amended in Amendment 2. Every held file has 15 problems with 8 samples each, as P3's rule
asks.

## Did either arm leave the rest of the model's maths intact?

![damage](../experiments/qwen_pass_pools/figures/damage.png)

**Reading it.** No. GSM8K test accuracy fell in all four arms: the known arm to 0.424 and 0.438, which misses the filed
no-damage rule by about 0.35 in both seeds, and the new arm to 0.477 and 0.549. The unchanged model reads 0.784. The
same recipe on 7,217 GSM8K problems read 0.621 at step 100, and 0.598 in the filed reference run. Loss on held-out gold
solutions rose in every arm, from about 2.0 nats per token to between 2.75 and 3.42. In the 7,217-problem run it fell
from 2.50 to 0.51 on its own held-out GSM8K set.

| arm | GSM8K strict, step 100 | minus 0.784 | held loss, own pool: W0 → 100 | held loss, other pool: W0 → 100 |
|---|---|---|---|---|
| known, seed 0 | 0.4235 | −0.361 | 2.007 → 2.898 | 1.960 → 3.215 |
| known, seed 1 | 0.4375 | −0.347 | 2.007 → 2.955 | 1.960 → 3.416 |
| new, seed 0 | 0.4766 | −0.308 | 1.960 → 3.001 | 2.007 → 2.796 |
| new, seed 1 | 0.5490 | −0.235 | 1.960 → 3.059 | 2.007 → 2.753 |

Source: `results/read_<arm>.json` → `step_100.gsm8k.strict_dropped` with `k_fits` 255 or 256, and
`step_100.held_loss`, `w0.held_loss`. The unchanged model's 0.7842 is `step_100.gsm8k.w0_strict_dropped`, from
`experiments/qwen_recipe/results/train_W.json` → `states.w0.unprompted.strict_dropped`. The 7,217-problem run:
`experiments/qwen_recipe/results/train_gold_1e-5.json` → `states.step_100.unprompted.strict_dropped` 0.621 and
`states.step_100.held_gold.loss` 0.510; its starting held loss 2.495 is `train_W.json` → `states.w0.held_gold.loss`;
the filed reference 0.598 is `train_W.json` → `states.gold_s0_100.unprompted.strict_dropped`.

## How closely did the arms fit their 62 problems?

![fit](../experiments/qwen_pass_pools/figures/fit.png)

**Reading it.** Almost exactly. Training loss first falls below 1e-3 between steps 27 and 37, with brief spikes above
it until step 53 in one arm, and ends between 1.7e-5 and 4.3e-5 nats per token in every arm. The same recipe on 7,217 problems, which sees each problem less than once, levels off
near 0.5. The known pool's solutions are no easier to fit at the start: step-1 loss is 2.07-2.10 in all four arms.

Source: `results/train_<arm>.json` → `steps[].loss`; step 100 reads known 1.81e-5 and 1.67e-5, new 3.96e-5 and
4.34e-5; the 7,217-problem run's step 100 reads 0.445 (`train_gold_1e-5.json` → `steps[99].loss`).

**What this is consistent with, not a cause.** The damage in the previous section, the rising held-out loss and the
near-zero training loss are all consistent with 103 passes over 62 problems pulling the model towards reciting those
62 solutions. Nothing run here separates that from other explanations, such as a property of these particular
solutions. The test is in "Not run": the same pools trained for one or two passes, or 62 random GSM8K problems trained
for 103 passes.

## Is the new pool's update larger than the known pool's?

![size](../experiments/qwen_pass_pools/figures/size.png)

**Reading it.** Yes, in both seeds, by about 5%. At step 100 the new arm's update has norm 3.612 and 3.634, the known
arm's 3.442 and 3.430. The four curves lie on top of each other until about step 20 and separate only as the fit
completes. The new pool's training sequences are also longer, 308 tokens on average against 232
(`train_<arm>.json` → `data.train_len_mean`), so the difference need not come from what the solutions contain.

All four arms cross the two D3 sizes at the same steps: 0.35 between steps 5 and 6, 0.49 between steps 6 and 7.

| arm | step 5 | step 6 | step 7 | step 10 | step 100 |
|---|---|---|---|---|---|
| known, seed 0 | 0.2999 | 0.4190 | 0.5492 | 1.0072 | 3.4420 |
| known, seed 1 | 0.3001 | 0.4203 | 0.5496 | 1.0079 | 3.4297 |
| new, seed 0 | 0.2963 | 0.4151 | 0.5446 | 1.0053 | 3.6122 |
| new, seed 1 | 0.2965 | 0.4162 | 0.5454 | 1.0058 | 3.6339 |

Source: `results/read_<arm>.json` → `files.<step>.norm`, the fp64 norm of each saved file over its 196 tensors, each
within 1e-6 of the training record (`files.<step>.holds` true in all 20). The files are
`results/large/dW/<arm>/step_<t>.safetensors`. The 7,217-problem run ends at 2.858.

## The pools: why 77, not 100 or 115

Before Amendment 2 the new pool held 78 problems (`results/pools.json` → `meta.n`). A $0 check of the problems the
gold-error guard removed found that most MATH removals were correct answers the grader failed to read
(`reports/reviews/2026-11-06_review_e120-gold-error-guard.md`). Amendment 2 fixed the grader and projected about 115.
The rebuild gave 77 (`results/pools_amend2.json` → `meta.n`). The fix does not return those problems to the new pool:
a grader miss is a problem the model did solve, so after regrading it leaves "never solved" altogether. Stage 1's
label moves show it: 14 MATH problems went from new to known and 4 from new to neither
(`results/stage1_amend2.json` → `label_moves`), and MATH's never-solved count fell from 176 to 158
(`before_after.math.labels.new`). The one problem lost from the 78 is `math/number_theory/654`, the grader miss the
check named among the survivors. The known pool was rematched: 11 of its 78 problems left and 10 others came in.

The grader controls held on the amended grader: gold against its own solution 1.000 for both sources, MATH gold
against the next problem's gold 0.007 (`results/stage0_amend2.json` → `math_gold`, `gsm8k_gold`, `math_next_gold`).

Two facts the plan did not expect. GSM8K supplies 43 of the 77 new problems, against the plan's "almost nothing";
the guard check found about one in five never-solved GSM8K problems carries a wrong gold label, which the guard may not
catch, so some of the 43 may be "new" only because their label is wrong. And on MATH level 5 the truncation guard,
not the gold-error guard, limits the pool: 74 of the 107 truncation removals are level 5 (`meta.never_solved_removed_by_truncation_guard`).

## Novelty check

With thinking on, 4 samples per problem at the card's thinking settings and an 8,192-token cap, the new pool was
solved at least once on 20 of 78 problems, 0.256 (`results/n1.json` → `new.solved_share`); the known pool, as the
control, on 0.974 (`known.solved_share`), above the 0.8 floor. On the 77 kept problems the share is 20 of 77, 0.260:
the dropped problem was not solved with thinking (`new.k_of_4`). The check ran on the 78-problem pool before
Amendment 2. A caution beside it: 0.446 of the new pool's think-mode samples were cut off at 8,192 tokens
(`new.truncation`), against 0.080 on the known pool, so the solved share is a lower bound on what thinking can do.

## What we learned

The model does not learn to solve unseen never-solved problems from 62 gold solutions at this recipe; it memorises
the 62. But the known arm, which should have been the gentle update, damaged the model about as much as the new arm.
It halved the model's success on other known problems and cut GSM8K accuracy by nearly half. So for D3 these two updates differ
less in kind than the plan intended. Both are overfit updates on a small set, and the clearest difference between
them is a 5% larger norm that longer solutions could explain. A D3 reading that separates them is evidence about
these two updates, not yet about new against existing skill in general.

## How compelling is the evidence

- **Small held-out sets.** 15 problems a pool, 8 samples each. The new arm's 0.05 is 6 correct samples on one problem;
  one more problem solved four times would have crossed the bar. The verdict is clear in direction but rests on few
  problems.
- **Two seeds.** They agree on every verdict. The GSM8K numbers spread by 0.014 between the known seeds and 0.072
  between the new seeds.
- **The reference numbers come from a different path.** The unchanged model's held-out pass@1 is from stage 1, with
  64 samples through vLLM at different caps; the arms are read with 8 samples through the training code's sampler. No
  run reads the unchanged model on the arms' path.
- **Only step 100 was read.** The saved updates at steps 5, 6, 7 and 10, the ones nearest the D3 sizes, have no
  behaviour reading here: we do not know whether they already show the damage.
- **Blind spots.** Pass@1 on held-out problems cannot see a skill the model uses only with thinking; the GSM8K score
  cannot tell overfitting to a style from a lost skill.

## Surprises

- **The pool came out at 77, not 200-600.** The filed lean for pool size missed by a factor of three. Amendment 2's
  estimate of 115 missed because a fixed grader miss leaves the never-solved pool rather than rejoining it. The filed
  prediction on pool size stands as missed.
- **The known arm damaged the model as much as the new arm.** Held-out known problems fell from 0.971 to about 0.5,
  and GSM8K fell further in the known arm than in the new. The plan read the known arm as the update that strengthens
  an existing skill; at 103 passes it did not behave like one.
- **Held-out gold loss rose in every arm**, where the same recipe on all of GSM8K lowered it about fivefold.

## What next

- qwen-decomp-lead reads the 20 saved updates under its filed D3 rules, with this report's caveat attached. If the
  readings separate the arms at steps 5-7, where the damage is unmeasured, that is the more useful comparison.
- A cheap test of the overfitting reading: the same pools trained for one or two passes, with the same reads at the
  end. If GSM8K stays near 0.78 and held-out loss falls, the damage here came from the passes, and the anchor pair
  should be rebuilt at that length.
- E128, post-cutoff facts, is the backup for anchor 2 and goes ahead under ask 13.

## Scorecard

Filed rules as written, values rounded to 9 places, bands including their ends (`predictions.json` → `rounding`).

| id | claim | rule | read | verdict |
|---|---|---|---|---|
| P1 | the new pool has at least 100 problems | `pools.json` → `meta.n` ≥ 100 | 78; 77 after Amendment 2 (`pools_amend2.json` → `meta.n`) | **missed** (stands) |
| P2 | novelty check passes | `n1.json` → `new.solved_share` ≤ 0.5; control ≥ 0.8 | 0.256; control 0.974 | **passes** |
| P3 | the new arm learns | held-out pass@1 ≥ 0.10 both seeds | 0.050, 0.025 (`grades.json` → `arms.new_s{0,1}.new_held_samples.pass1`); trained-part greedy 1.0 both | **no: "memorised, carrying little"** (as leaned) |
| P4 | the known arm leaves GSM8K within 0.05 of 0.784 | per seed, abs difference ≤ 0.05, both seeds | 0.4235, 0.4375 (`read_known_s{0,1}.json` → `step_100.gsm8k.strict_dropped`): −0.361, −0.347 | **missed, both seeds** |
| P5 | the new arm's step-100 update is larger | new s > known s, both seeds | 3.612 > 3.442; 3.634 > 3.430 (`read_<arm>.json` → `files.100.norm`) | **holds, both seeds** |
| control | stage 0 | greedy in [0.72, 0.84]; sample within 0.05 of 0.753; graders 1.0 | 0.785; 0.756; 1.0 (`stage0.json`), 1.0 on the amended grader (`stage0_amend2.json`) | passed |
| control | novelty check, known pool | ≥ 0.8 solved | 0.974 | passed |

P4's filed rule reads W0 from `train_W.json` → `states.w0.unprompted.strict_dropped`, 0.7842. No prediction was edited.
No D3 reading was predicted here.

## Departures from the literature

- **Length cap.** The Qwen3 model card suggests an output budget of 32,768 tokens; stage 1 capped MATH at 2,048 and
  GSM8K at 512, the training code's reads at 1,024, and the novelty check at 8,192, for run time. Hence "never solved
  in 64 samples of at most 2,048 tokens".
- **Grader.** math-verify 0.9.0 is the source; Amendment 2 compares MATH answers also after stripping presentation,
  which math-verify does not do, because the check found it marking correct answers wrong.
- **pass@1.** The mean over problems of correct samples over 8, the unbiased estimator of Chen et al. 2021 at k = 1;
  no departure.
- **Recipe.** E61's gold recipe is used unchanged, but on 62 problems instead of 7,217, so 103 passes instead of under
  one. That is a departure from the regime the recipe was set for, not from a published source.
- **Adaptive sampling.** 64 samples only for never-solved problems and their candidate matches, not for every problem,
  as stated in the plan.

## Not run

- Any D3 reading of these updates: qwen-decomp-lead's.
- Any behaviour reading at steps 5, 6, 7 and 10: only step 100 was read.
- The unchanged model on the arms' sampling path, 8 samples at a 1,024-token cap through the training code's sampler.
- A run of the same pools for one or two passes, or of 62 random GSM8K problems for 103 passes: the test of whether
  the passes cause the damage.
- Thinking-mode samples at step 100.
- The plan's "known pool held-out loss drop beyond the seed spread": the loss rose in every arm, so there is no drop
  to compare with the seed spread.
- The novelty check on the 77-problem pool itself: it is computed from the 78-problem run's per-problem counts.
- An independent check of this analysis.

## Provenance

- Results: `experiments/qwen_pass_pools/results/` → `stage0.json`, `stage0_amend2.json`, `stage1.json`,
  `stage1_amend2.json`, `pools.json`, `pools_amend2.json` (sha256 `faf4a9f3…` in `train_<arm>.json` →
  `data.pools_sha256`), `n1.json`, `train_<arm>.json`, `read_<arm>.json`, `grades.json`, `sft_<arm>.log`.
- Large files, gitignored: `results/large/dW/<arm>/step_{5,6,7,10,100}.safetensors`,
  `results/large/samples/<arm>/*.json`.
- Fine-tunes: MATS `debug` L40, about 25-30 min of training and 33-41 min per job, peak 32.0 GiB of GPU memory and
  12.4 GiB of host memory (`read_<arm>.json` → `peaks`). Early checks passed in all four: step-1 update exactly 0,
  step-2 size ratio 0.941-0.947 in the [0.85, 1.05] band, loss falling by step 10 (`train_<arm>.json` → `checks`).
- Grading: `PYTHONPATH=~/commute-data/scratch/qwen-rl-lead/e120-a2-writer/mvlib fleet/cpu-run -c 4 --mem 4G python
  experiments/qwen_pass_pools/grade_arms.py --large <clone>/…/results/large --dl <clone>/…/results/pod_dl`, math-verify
  0.9.0 linked from `~/commute-data/venvs/mathverify`.
