# Decision: the shuffled-reward null collapsed, so reinforcement learning on Qwen has not yet been compared with a working null. What should the cell's last run do? (due 11-16)

**Owner:** qwen-rl-lead. Cell P3.4 (reinforcement-learning updates on Qwen3-1.7B that the decomposition can see, each
with nulls), due 11-16. Two of its three runs are spent; this card decides the third. The cell has spent about $226 of
the ~$263 the user approved, so any paid run needs a yes on money as well.

**Question.** Run 2 trained, on two seeds, reinforcement learning on school maths, the same loop with the rewards
shuffled among each problem's answers, and plain training on the model's own correct answers. By the filed rule
reinforcement learning stands apart from the shuffled-reward null at twice the decomposition's error. But the null
broke: it stopped answering correctly, so the comparison says nothing. Should the last run train a null that stays a
null, or should the cell close as read?

**Background.** The **output change** is the KL divergence from the unchanged model to the trained one, per token, on
128 maths rows. The decomposition's own error on those rows is 0.0932 nats per token; "2 times" is twice that, the
smallest size at which an update stands clear of the decomposition's noise. The **pass rate** is greedy accuracy on the
1319 held-out GSM8K test problems; the unchanged model reads 0.782. The shuffled-reward null rules out that any update
of this kind, trained on the model's own answers, raises the pass rate as much as reinforcement learning does.

**What run 2 found** (analysis: `reports/analyses/2026-11-12_analysis_e167-qwen-rl-readable-run2.md`).
- **Reinforcement learning at peak rate 3e-6** reached twice the error at step 75 on both seeds, and five times at step
  250 on seed 70 only. Its pass rate stayed near 0.88 at every size: 0.882 and 0.884 at twice the error.
- **The null collapsed on both seeds.** The share of its training answers that were correct fell from 0.80 to under
  0.2 by step 30. Its answers shrank from about 190 tokens to under 10, its pass rate fell to 0.039 and 0.000, and it
  stopped updating once no problem had a mix of right and wrong answers. Its output change was already about 4 times
  the error at step 75, so it was not at reinforcement learning's size either. The cause was not checked.
- **The earlier runs' null held.** At a third of this rate, 1e-6, the same null ran 100 steps three times, on two
  seeds, without collapsing. The share of its training answers correct never fell below 0.53, and its answers never
  averaged under 125 tokens. At step 100 it passed 0.870 and 0.794 on the two seeds, against reinforcement learning's
  0.881 and 0.877. Run 1's null read 0.78 times the error at step 100. It has never been run past step 100.
- **The fine-tune on its own correct answers** passed 0.814 to 0.832 but barely moved the output, 0.05 to 0.22 times
  the error. It is a reference, not a null of the same size.
- **Every update sits outside the decomposition's subcomponents as much as a random change of the same size does**,
  for all three kinds of training.

**Options.** Costs use run 2's measured times on an H100 at $3.99 an hour: 61 s a step for reinforcement learning with
its reads (400 steps in 6.8 h), about 60 s a step for a working null (run 1's), 40 minutes of set-up and about 20
minutes of downloads per pod. Readings run on the cluster's L40s at no cost, about 16 minutes per state.

- **(a) New null alone, at the rate where it held, matched by output change. Recommended.** Two pods, one per seed,
  train the shuffled-reward null at peak 1e-6 and save every 25 steps. Each stops at the first save where its output
  change reaches that of run 2's reinforcement-learning state at step 75 on the same seed, which is already saved and
  read: 2.09 times the error on seed 0, 3.06 on seed 70. A filed stop ends a pod if, over 5 steps in a row, under half
  its training answers are correct or its answers average under 100 tokens, just below what the three earlier nulls
  ever read. What it tells us: whether reinforcement learning's pass rate at about twice the error is above that of a
  working null that changes the output as much, by more than the seed spread. Run 1's null at this rate read 0.78
  times at step 100, so the match may come near step 250 on seed 0 and near step 400 on seed 70. The cap is step 500,
  about 9.3 hours a pod, so at most about $75, less if it matches sooner. That would take the cell to about $300,
  about $38 over the approval. Next: the cell closes on that reading. If the null collapses again, the cell closes
  saying no working null was found, and the stop keeps that cheap. Two things depart from the cell as written: the
  null trains at a third of reinforcement learning's rate, and it is matched by output change, not by step. Run 2
  showed that a step match never gave a size match: at the same step the null was at twice reinforcement learning's
  size and the fine-tune at under a fortieth.
- **(b) Retrain both reinforcement learning and the null at 1e-6, matched by step as filed.** Four pods, one per arm
  and seed, to step 500. States are saved where reinforcement learning crosses twice the error, as in run 2. What it
  tells us: the same comparison, keeping the cell's step match, with both arms at the earlier runs' rate. At this rate
  reinforcement learning reached only 1.36 times by step 250 on seed 0 in run 1, so twice the error may not arrive by
  step 500 on seed 0, and that size would then be unread. About 9.5 hours a pod, about $150, taking the cell to about
  $376, about $113 over. Next: the cell closes on that reading.
- **(c) Close the cell as read; no spend.** The report states: "apart at twice the error by the filed rule, but
  against a null that collapsed, so no claim at the bar." Five and ten times are unread. What it tells us: nothing more
  about whether the reward matters at a size the decomposition can see. Part 3 then has a reinforcement-learning
  update at two and five times the error, read on the decomposition, with no behaviour comparison behind it. Next:
  part 3's readings of these updates go ahead without that comparison.

**Why it is a question.** Options (a) and (b) both cost more than the money left in the approval, and both run over
6 hours, so each is a gated run: pre-mortem, a free rehearsal, one review and your go-ahead with the rehearsal's
numbers. Option (a) also reads the cell's "matched steps" as "matched output change", which is the user's call.

**Recommendation: (a).** It reuses run 2's reinforcement-learning states, which are already saved and read, and it trains the null
at the only setting where that null has been seen to hold, so it costs half of (b) and is not at risk of the
reinforcement-learning arm falling short of twice the error. Its weak point is that the null has never run past step
100 at that rate; the filed stop on correct answers and answer length ends a repeat collapse within a few steps. A
caution for reading it: the earlier runs' working nulls differed by 0.076 between seeds, so a working null may well
read "together" with reinforcement learning, and that would be the cell's answer.

**Evidence.**
- `reports/analyses/2026-11-12_analysis_e167-qwen-rl-readable-run2.md` and its figure
  `experiments/qwen_rl_readable/figures/run2.png`
- `experiments/qwen_rl_readable/results/run2_reads/verdict.json` (the filed rule's call and every state's reads)
- `experiments/qwen_rl_readable/results/run2_reads/share.json` (outside share against a random change)
- `experiments/qwen_rl_readable/results/run2/N_s0.json` and `N_s70.json` (the null's collapse, step by step)
- `experiments/qwen_rl_readable/results/run2/chosen.json` (matched steps)
- `experiments/qwen_rl/results/lr1e-6w10/N.json`, `lr1e-6w10_s70/N.json`, `experiments/qwen_rl_readable/results/N.json`
  (the three nulls at 1e-6 that held to step 100)
- `experiments/qwen_rl_readable/results/predictions_run2.json`, plan `reports/plans/2026-11-12_plan_qwen-rl-readable-run2.md`

**Not run.** The cause of the null's collapse; any null past step 100 at 1e-6; a null matched by output change; ten
times on any seed and five times on seed 0.
