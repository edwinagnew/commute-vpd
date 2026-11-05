# Advice: the first neuron experiment on Qwen3-1.7B

*2026-11-05, Fable, for baseline-lead. Read-only; nothing run. Every number carries its file and line; each citation was
re-opened before it was written down.*

## Where the Qwen line stands tonight

- **VPD's reading on Qwen is E94's D3 rows, not r.** The re-gating share reads 0.993-0.9995 for every change, random
  included (`reports/analyses/2026-10-30_analysis_e57-reading-the-update.md:93`), so there is no r to stand beside. E94's
  control passed and anchor 1's two states have landed in `qwen-decomp-lead`'s clone: direct share on maths rows 0.0093
  and 0.0091, outside share 0.123 against a null of 0.212, and the score so far reads "no number counts"
  (`~/commute-agents/qwen-decomp-lead/experiments/qwen_d3_rows/results/run.log:45,61,62`); anchor 2's matched states are
  job 52941, pending on `Resources`. So a neuron reading has D3 numbers to sit beside on anchor 1 today and on anchor 2
  tomorrow, on the same states and rows (`reports/plans/2026-11-05_plan_e94-d3-rows.md:19-24,28`).
- **The labels.** Anchor 1, the template fine-tune: a behaviour W0 already produces under a system turn, 94.1% prompted
  against 0% unprompted (`reports/analyses/2026-11-05_analysis_e90-qwen-check.md:15,25`); the definition writes for it
  (`:72`). Anchor 2, TOFU step 1250: "new facts memorised, carrying to rephrasings for most facts" (`decisions.md:148`),
  with the pull to trained names stated on every claim (`reports/plans/2026-11-05_plan_e94-d3-rows.md:26`). GRPO: no
  capability holds (`e90-qwen-check.md:74`). These are the firmest two labels the line has; GRPO stays description.
- **The 67M route is closed** (`experiments/carrying_jobs/results/score_amend1.json:236`): C1a 0 of 10 on every task, under
  a judge that saw the task's windows. Its failure was vocabulary: a Pile job about punctuation cannot be tied to "repeat
  the key". On Qwen the tasks are prose, and so are the jobs.

## A Qwen "neuron" and its firing

Each of the 28 blocks has a SwiGLU MLP with 6144 units (`src/commute/qwen_vpd.py:9,52`): 172,032 neurons. Neuron i's
firing is a_i = silu(g_i)·u_i with g = W_gate x and u = W_up x; its write direction is column i of W_down. E98's cells
carry over unchanged: the neuron share is W0's gate, up and down applied to the MLP input the state produces; the
carrying ranking is W0's gradient at the readout times the change in a_i. Top windows are ranked on the positive side of
a_i, as the essay does on Qwen3-32B; state that signed firings exist and are not read.

## The options

**(a) E98's carrying sets on Qwen, then E101's writer, maximiser and judge.** Stage A ($0, one `debug` job): per state
the carrying set and k_half, plus a FineWeb-Edu pass (the stand-in corpus, 512-token rows, `src/commute/qwen_data.py:1,12`;
Qwen's pretraining text is not public, `e90-qwen-check.md:9`) keeping each neuron's top 20 windows and a quantile sample.
Stage B: agents. Cost: E92 read four states, with a per-block steering fit, in 431 s at 21.3 GB on an L40
(`experiments/qwen_comparators/results/run.log:74`); E101's stage 0 on the 67M took about a minute at 33 GB
(`reports/analyses/2026-11-05_analysis_e101-carrying-jobs.md:275`). Six states with ~15 forwards each plus a 1M-token
corpus pass: under an hour, well inside `debug`'s 2 h, $0; the RunPod fallback is an H100 at $3.99/h
(`reports/plans/2026-11-03_plan_e69-grpo-loop.md:180`), about $4. Agents: E101 batched 352 neurons into 12 batches and 36
Sonnet calls (`reports/plans/2026-11-05_plan_e101-carrying-jobs.md:90`); about 800 neurons here is ~80 calls. **Not gated.**

**(b) The agent-free reading alone: stage A's numbers.** Same job, no agents. Two numbers per state: the **engaged
share** (E101's P0: the carrying neurons' W0 firing at the task's answer positions as a quantile of their FineWeb firing,
median over the set, random layer-matched sets at ~0.5) and a **prompt overlap** that exists only on Qwen: the share of
anchor 1's carrying set inside the set of neurons whose W0 firing rises most when the system turn is added, read on the
same 64 template rows (`src/commute/qwen_comparators.py:60`). $0, ~40 min, one job.

**(c) E98's neuron share on Qwen as the headline.** Not recommended: its 0.43 relay floor on four blocks
(`reports/analyses/2026-11-05_analysis_e98-neuron-r.md:88`) can only be worse on 28, and a share cannot be read across
anchors 37 times apart in size (`plan_e94-d3-rows.md:24`). Print it as a beside-row only.

## Recommendation: (b) first, in the same job as (a)'s stage A; agents only on what (b) licenses

Run stage A on E94's six stop-rule states (anchor 1 s0, s1 step 100; anchor 2 s42, s142 at steps 29/30 and 1250) and
E94's Gaussian change at anchor 1's output size, on E92's rows (`qwen_comparators.py:60,72`). Then stage B only if the
positive control below passes.

**Positive controls that can pass, and why E101's could not.**
1. *Prompt overlap, template rows.* The prompted W0 is the reference behaviour itself (`e90-qwen-check.md:25`). Its
   contrast set, read on even rows, must overlap itself on odd rows far above a layer-matched random set; the Gaussian
   change's carrying set must not. Both are mechanical; neither needs a judge.
2. *W0 picks on the TOFU rows.* Neurons that fire at biography-answer positions more than on FineWeb and on template
   rows. Their jobs will be about names, dates and book titles in biographical prose; a forced-choice judge over four
   prose descriptions (layout answers, author biographies, maths solutions, general web text; chance 1 in 4) can tie
   those to "answer questions about an author's life". E101's picks were position and word-fragment neurons on
   marker-token windows (`e101-carrying-jobs.md:85-99`); here the answer positions are sentences.

**Predictions in outline** (bands set from the random sets and the Gaussian change in the same job, fast-lane rule).

| number | W0 | do-nothing | Gaussian change | anchor 1 (reuse by label) | anchor 2 full (new facts by label) |
|---|---|---|---|---|---|
| prompt overlap | no set | no set | chance | lean: well above chance, both seeds | not defined (no prompt elicits the facts); print real-author overlap beside, unscored |
| engaged share | no set | no set | ~0.5 | lean: above 0.5 by the band | no lean: biography prose engages W0 too |
| related share (stage B) | no set | no set | judge floor | lean: above the floor, jobs about lists, "Given", answer lines | lean: *also* above the floor, jobs about biographical fields |

The last cell is the point of the experiment, said in advance: if TOFU's carrying neurons have jobs that name
biography fields, the essay's reading scores a labelled-new update as reuse, the granularity limit in neuron form, and
that is reported as what the baseline reads beside VPD's "no number counts". The matched-step TOFU states (not learned)
are the size control: their engaged share should sit with the Gaussian change.

**Blind to.** Relays (E98's 0.43); facts stored as new write directions onto old neurons, which this reads as reuse; the
in-sample ranking; FineWeb as a stand-in for Qwen's corpus. **Noise floor.** Seed spread of each anchor, the random sets'
spread, and E57's finding that overlaps read 3-4 times chance for every change including random
(`experiments/TRIED.md:133`): the Gaussian change's overlap, not chance, is the floor.

## What would change this

- Control 1 fails (the prompted contrast does not reproduce across row halves): the neuron sets are too diffuse at this
  grain; stop, report, no agents.
- E94's anchor 2 rows come back counting a number: run stage B on both anchors at once, since there is then a VPD
  separation to match.
- Stage A exceeds ~90 min projected from its first state: split states across `debug` jobs, never `normal` while the
  queue holds 47 jobs (`squeue -p compute`, 22:30 tonight).
