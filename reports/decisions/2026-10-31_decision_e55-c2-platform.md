# E55 C2: run VPD on the counting truth set, and where? (decision for the user)

Plain words: C2 would decompose the small counting-task model with VPD and score VPD on the same 21 states where
pruning, reverse fine-tuning and steering all scored 21/21 (C1b). A $0 rehearsal on the cluster measured what the
full decomposition costs, and every way of running it is a gated run, so it needs your choice.

## What the rehearsals measured (jobs 46971, 47013, 47071; `experiments/pcfg_anchors/results/c2_rehearsal.json`)
- The recipe's size (38,016 subcomponents) runs at 1.19 s/step and fills 46 of an L40's 47 GiB; it ran out of
  memory after a save. Dropped.
- The 67M decomposition's ratio (14,592 subcomponents, a stated departure from the recipe) runs at 0.44 s/step in
  19 GiB: 400K steps (every language-model recipe in param-decomp uses 400K) is about 50 L40-hours.
- About 70-75% of each state's gain on its task sits in the decomposed matrices (matrix share 0.76 for a P state,
  0.68 for an N state); the rest is in parts VPD does not decompose. Over half the update's norm is outside them.
- A stopped-and-resumed run does not yet match an unbroken one (max parameter difference 0.029): a code fix
  before any long resumed run.

## Why it may not be worth it
On this set the answer-only probe and update size within a recipe also score 21/21 (C1b). VPD can at best tie; a
VPD fail would be the sceptic's result. The open-pair set (C3, ladder) gave the cheap instruments no working
reading at all, so it cannot host the comparison either.

## Options
- **(a) Cluster `normal` QoS, $0.** Three resumed jobs of up to 24 h. Needs the "Qwen line only" rule for `normal`
  relaxed for this run, and an empty queue. Recommended by the C2 planner if the rule is relaxed.
- **(b) RunPod H100, ~$57-140 (central ~$90).** Over the $40 session cap; H100 speed on this job unmeasured (a
  short first stage with a stop would measure it).
- **(c) Fewer steps.** No source supports it for this recipe.
- **(d) Drop C2.** Write up C1b + C3 + ladder as the reuse line's benchmark result: cheap instruments work on the
  counting set (where trivial signals also work) and fail their control on the 67M open-pair maps.

Lead's recommendation: (a) if you will relax the rule; otherwise (d). Not (b): ~$90 for a likely tie.
Whatever you pick, the remaining gates (resume fix, filed predictions, read/score code, two reviews) come first.
