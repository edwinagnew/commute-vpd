# Decision: reinforcement learning on Qwen did not reach five times the decomposition's error with accuracy and answer length held. What should the main run train? (due 11-16)

**Blocking for the cell's main run.** Owner: qwen-rl-lead. Cell P3.4, due 11-16. Run 1 of 3 is spent; this card chooses
run 2.

**Question.** Run 2 trains three arms on Qwen3-1.7B, on seeds 0 and 70: reinforcement learning on school maths, the same
loop with shuffled rewards, and a plain fine-tune on the model's own answers. It saves states where the reinforcement
learning arm's output change is 2, 5 and 10 times the decomposition's own error. No recipe tried in run 1 reached 5
times while keeping both its held-out pass rate and its answers inside the generation window. Which recipe should run 2
train?

**Background.** The **output change** is the KL divergence from the unchanged model to the trained one, per token, on
128 maths rows. The decomposition's own error on those rows is 0.0932 nats per token, so "5 times" is 0.47. A state
counts only if its held-out GSM8K pass rate is at least 0.842 and at least 0.9215 of its answers fit the window. Run 1
trained the reinforcement-learning arm at three peak learning rates, one seed, and kept no states.

**What run 1 found** (analysis: `reports/analyses/2026-11-12_analysis_e163-qwen-rl-readable-run1.md`).
- **Peak 1e-6, the earlier run's rate:** 1.36 times by step 250. Accuracy held; too slow.
- **Peak 3e-6:** 2.65 times at step 150 with pass 0.891, and 2.89 times at step 200 with pass 0.850. Both states
  count. From step 100 to 200 it gained about 0.8 times per 50 steps, but only a quarter of a multiple in the last 50.
- **Peak 1e-5:** 5.23 times by step 25 with pass 0.890, but only 0.829 of its answers fit the window. The run stopped
  there on its filed drift rule. Its step 10, at 1.88 times, had pass 0.826, under the bar.
- **Runs diverge.** The repeat of the earlier run matched its update norm and pass rate at step 100, but its output
  change was 0.048 against 0.064. A check with the same measuring code on the earlier run's saved updates gave 0.064
  again, so the measure is right and the two runs are different updates of the same norm. The cause is not checked.
  With the earlier run's two seeds at 0.064 and 0.098, the output change at a fixed step varies by up to about two
  times between runs.

**Options.** Costs use an H100 at $3.99 an hour, about 40 minutes of set-up per pod, and the run 1 step times: about
66 s per step for the reinforcement-learning arm with its reads, 60 s for the shuffled-reward loop, 40 s for the
fine-tune. Each saved state is 5.64 GB, about 7 minutes to download. Six pods run in parallel, one per arm and seed.

- **(a) Peak 3e-6, to step 400. Recommended.** The only recipe that held both the pass rate and answer length. It
  saves each seed's states where that seed's reinforcement-learning arm first crosses 2 and 5 times, and the other two
  arms at those same steps. What it tells us: whether a reinforcement-learning update of 2 and 5 times the noise floor
  stands apart from its shuffled-reward twin on two seeds. At its run 1 growth, 5 times arrives somewhere around step
  335 to 410; if growth stays as slow as in its last 50 steps, it does not arrive by 400. Ten times is out of reach;
  the report says so. About 8 hours, about $170. Next: if it falls short of 5 times but is still growing, the cell's
  held-back third run can extend it.
- **(b) Peak 5e-6, to step 250.** A rate between the two tried; untested. What it tells us: it may reach 5 times
  sooner, but answers may run past the window as they did at 1e-5, and then nothing at 5 times counts. About 5.5
  hours, about $115. Next: if the window fails, run 2 is spent and only the third run is left.
- **(c) Peak 1e-5 with a longer generation window, to step about 60.** Reached 5 times in 25 steps. Needs a code and
  measurement change: the longer window changes which answers count, so pass rates are no longer comparable with the
  unchanged model's 0.782 unless the unchanged model is read again at the new window, and the bars are filed again.
  What it tells us: whether the fastest recipe reaches 5 and maybe 10 times once long answers are allowed; it also
  trains on longer answers, so it is a different recipe from the earlier run's. About 3 hours, about $55, plus a day of
  code, new predictions and review.
- **(d) Save only the 2 times states.** Peak 3e-6 to about step 150 to 200 on both seeds, all three arms, and report
  that reinforcement learning could not reach 5 or 10 times with accuracy and answer length held. The cell's "done
  when" allows this. What it tells us: only whether the arms separate at 2 times. About 4 hours, about $75 to $95.
  Next: part 3 reads one size only.

**Why it is a question.** Every option above is a gated run, over $60 or 6 hours, except (c) at cost; (c) changes the
measurement. Option (a) also takes the cell past its $200 RunPod allowance: run 1 spent $50.82, so (a) brings the cell
to about $220.

**Recommendation: (a).** It is the only recipe that held both bars, it still gives the 2 times states that (d) gives,
and it may give 5 times as well. States are matched by step to the reinforcement-learning arm of the same seed: the
shuffled-reward and fine-tune arms of seed 70 are saved at the steps where seed 70's own reinforcement-learning arm
crossed 2 and 5 times. Because runs diverge by up to two times at a fixed step, states are never matched by step
across seeds; across seeds they are compared at the same size, with the step each one took. The output change also
dips between reads, so a crossing is the first read at or above the size, read every 5 steps.

**Budget note.** The remote-kernels plugin still shows a $100 session budget, though the settings file says $300. It
would stop (a) once total session spend reaches $100, about $49 from now. A `/mcp` reconnect of remote-kernels, which
only the user can run, applies the $300.

**Evidence.**
- `reports/analyses/2026-11-12_analysis_e163-qwen-rl-readable-run1.md` and its figure
  `experiments/qwen_rl_readable/figures/run1.png`
- `experiments/qwen_rl_readable/results/selection.json` (the filed selection rule's outcome)
- `experiments/qwen_rl_readable/results/{control,N,F,r3e-6,r1e-5}.json` (every read)
- `experiments/qwen_rl_readable/results/kl_check.json` (this run's measure on the earlier run's saved updates)
- `experiments/qwen_rl_readable/results/predictions.json`
- `reports/plans/2026-11-11_plan_qwen-rl-readable-run1.md`

**Not run.** Peak 5e-6; any longer window; seed 70; any saved state; the cause of the divergence between runs.
