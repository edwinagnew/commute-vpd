# Plan: which of Qwen's neurons carry the template and TOFU fine-tunes, and were they already doing that job? (E102)

*2026-11-05, baseline-lead E102 writer. Stage A only, coded, not run. Adopts option (b) of
`reports/plans/2026-11-05_advice_qwen-neuron-FABLE.md`. Code `experiments/qwen_carrying/{run,score}.py`,
`src/commute/qwen_neurons.py`; rules `experiments/qwen_carrying/results/predictions.json`; job `cluster/qwen_carrying.sbatch`.
Not gated: two MATS `debug` jobs, about 19 and 10 min, $0.*

## In plain words

On Qwen3-1.7B we find, for each saved fine-tune, the MLP neurons of the unchanged model that carry its gain when they
fire differently with their weights kept, and ask two questions of them that need no agent and no decomposition. Were
they already firing at the task's answers before training, and, for the template fine-tune, are they the neurons a
system turn already moves? Both are read against a random change of the template's output size, so a pass means more
than "any change moves some neurons".

- **Claim.** The template fine-tune (anchor 1, labelled reuse by construction) re-fires neurons that the original
  model's own prompted behaviour already uses.
- **Null.** Its carrying neurons overlap the prompted contrast set, and are engaged at its answers, no more than a random
  change's carrying neurons of the same output size.

## The request in the Terms' words

The user asked for a neuron-level baseline in the spirit of the "already clean" essay
(https://already-clean.goodfire.com/essay.md), on the 67M first, then Qwen, scored on the construction labels and marked
as assumed (`decisions.md`, row "baseline-lead's new task"). Here: a **carrying set** is the 1024 neurons of W0 with the
largest first-order part in the share of the gain carried by W0's own neurons re-firing on the state's MLP input (E98's
r_half cell, `src/commute/neuron_r.py`). No split, outside share or decomposition reading is computed; E94's direct and
outside shares are copied beside as the VPD column. Labels are assumptions: anchor 1 is reuse by construction
(`reports/analyses/2026-11-05_analysis_e90-qwen-check.md:15,25`), anchor 2 is "new facts memorised, carrying to
rephrasings for most facts" (`decisions.md`, E88 row), with its pull to the training's names stated beside it.

## Design check

- **Goal.** `GOALS.md`'s reuse-against-new question, by a decomposition-free reader that VPD's readings must beat to count.
  Class 4 (the Qwen deliverables' named next step: cheap comparators beside every Qwen reading,
  `reports/plans/2026-11-03_plan_qwen-reorient.md`) and inside the user's neuron-baseline task. No sign-off needed.
- **Nearest tried.** E98 (`TRIED.md`, neuron r on the 67M): the share reads 0.43 for output weights alone, so it is
  printed beside only. E101: the agent reading had no power on the 67M because its jobs and tasks shared no vocabulary
  (`reports/analyses/2026-11-05_analysis_e101-carrying-jobs.md:85-99`). E92 read size, rank and steering on the same
  states and rows. New here: Qwen's 172,032 neurons, prose tasks, and a positive control that can pass (the prompt).
- **Patch or step.** A step: the 67M route closed (`experiments/carrying_jobs/results/score_amend1.json`); Fable's
  "What would change this" names Qwen with prose tasks as the next place.
- **Decisions relied on.** The matched TOFU step is a size control only (`decisions.md`, row of ask 11); anchor 2's label
  and pull note (E88 rows). No departure.
- **What changes next.** Control (i) fails: the sets are too diffuse at this grain; stop, no stage B. P1 holds: stage B
  runs, and the forced-choice judge is read against it. P1 misses with C1 passing: the template's carrying neurons are
  not the prompt's, a reading against reuse at the neuron grain, reported beside E94's "no number counts".
- **Control first.** E0 (do-nothing through the cell path), E1 (W0's log P and each state's gain against E92's), then
  C1a (the prompted contrast reproduces across item halves) run before the corpus pass and before any anchor; C1b
  (the Gaussian change's set does not overlap the contrast set) runs before any anchor. Each miss stops the job. C1 can
  fail on the contrast code, on the item halves, and on a carrying ranking that picks the same neurons for any change.

## Stages

**Job 1** (`--states gauss,a1_s0,a1_s1`). W0 stage: E0, E1; the prompted contrast on anchor 1's 256 items (the same
answer ids after the plain chat and after E45's `format` system turn), its top 1024 on even and odd items, C1a; the
FineWeb-Edu pass (2000 rows of 512; each neuron's top 20 rows by peak firing, position and value, saved with the rows in
`large/tops.safetensors`, so stage B needs no GPU); C0; engaged values of every neuron on both row sets. Then E94's
Gaussian direction (`experiments/qwen_d3_rows/run.py:302-307`), scaled to anchor 1's clean KL on E91's 16 maths rows;
its carrying sets on both row sets; C1b; anchor 1 at both seeds. **Job 2** (the four TOFU states) refuses until
`control.json` passed. States are E94's names, files from `experiments/qwen_size_grid/results/size_grid.json`, read
only; all six are on disk. Rows are E92's (`src/commute/qwen_comparators.py:60,72`); the readout is E92's log P.

## The numbers

| number | definition | W0 | do-nothing | controls | blind to | noise floor |
|---|---|---|---|---|---|---|
| prompt overlap (P1) | share of a state's carrying set inside the contrast set (all items) | no set | no set | contrast halves: C1a's overlap; Gaussian: at its draws | relays; neurons the template moves that the prompt does not; in-sample ranking | Gaussian's overlap, 100 layer-matched draws (max, sd), seed spread |
| engaged share (P2, P3, R1) | median over the set of W0's firing at the answer positions as a quantile of the neuron's FineWeb firing (E101) | no set | no set | held-out FineWeb 0.50 (C0); Gaussian near its draws | relays; FineWeb as Qwen's corpus; chat-format neurons read engaged on any chat row | as above |
| printed beside | gain, k_half, neuron share nums[N]/G, top-1024 share of the numerator, halves overlap, E94's direct and outside shares | | | | | |

Fable's leans, re-checked at each cited line: anchor 1 above on both readings; matched TOFU (step 29/30, not learned)
with the Gaussian; anchor 2 full no lean on the engaged share; anchor 2's prompt overlap printed, unscored, since no
prompt elicits the facts. E57 found set overlaps 3-4 times chance for every change including random (`TRIED.md:133`),
so the Gaussian, not chance, is the floor. E92's gains, which E1 reproduces: anchor 1 0.666, 0.666; matched TOFU 2.05,
2.06; full TOFU 3.77, 3.76 nats a token (`experiments/qwen_comparators/results/states/`). The matched TOFU gain is three
times anchor 1's on its own rows, so P3 controls the update's norm, not its output size there.

## Rules (`commute.qwen_neurons.above`, `control_i`; tests `tests/test_qwen_neurons.py`)

`above`: floor = the larger of the Gaussian's value and every random draw; band = the larger of the two seeds' spread and
twice the draws' sd. "Above" iff both seeds exceed the floor and their mean exceeds it by more than the band, strictly.
P1, P2 hold on "above"; P3 holds on "not above"; R1 is reported. Run on its edge cases before filing:

| case | reads |
|---|---|
| W0, do-nothing, any state that moves no firing | no positive score, no set: unread |
| a seed or the Gaussian missing, non-finite, or under two draws | unread ("no call") |
| a seed exactly at the floor; the gap exactly at the band | not above (strict) |
| seed spread 0 | the gap must beat twice the draws' sd |
| the Gaussian as both anchors | not above: cannot pass |

C1a: halves overlap minus the largest of 100 draws at least 0.2 (E51/E52's gap). C1b: halves overlap minus the Gaussian's
overlap at least 0.2. A missing number fails its half. **A reading to flag:** Fable's "the Gaussian change's carrying set
must not" is read as "must not overlap the contrast set", from its table (Gaussian at chance). The other reading, the
Gaussian's set not reproducing across row halves, is printed per state as `halves_overlap`, unscored.

## Early checks and stop rules

| when | check | stops on |
|---|---|---|
| minute 1-2 | E0: do-nothing's max \|T1 - T0\| and nums[N] | over 1e-5 nats |
| minute 2 | E1: W0's mean log P on a1 and TOFU rows against E92 (-1.5376, -4.2481) | over 1e-3 relative |
| minute 3-4 | C1a | under the bar |
| minute 8-10 | C0: held-out FineWeb median engaged | outside [0.45, 0.55] |
| each state, first pass (30-80 s in) | E1: the state's gain against E92's | over 1e-3 relative |
| after the Gaussian | C1b | under the bar |
| before every unit | elapsed plus estimate (measured per row after the first state) over 27 min | exit 3: resubmit, it resumes |

Every check is logged when it passes. A stop writes `results/stopped.json`, and every run refuses while it exists.

## Cost and time

On one L40, $0. A forward of 32,768 tokens took 5.8 s in E92 (`experiments/qwen_comparators/results/run.log:63-64`),
a 5.64 GB state 24 s to load (`experiments/qwen_size_grid/results/run.log:5-6`), Σ 181 s
(`experiments/qwen_d3_rows/results/run.log:17`). A state costs about 13.5 forwards a row (the gradient pass about 3, the
state and W0 forwards 2, seven patched forwards): about 40 s on anchor 1's rows, 80 s on TOFU's, plus the load. The
FineWeb pass is 1.02M tokens without the head, about 3 min; the fetch is not measured (cached after the first job).
**Job 1 about 19 min, job 2 about 10 min**, under the 30-min `debug` rule; host memory peaks near 30 GB with Σ
(`--mem=48G`). Not gated (under $60 and 6 h).

## Stage B (coded 2026-11-05 after stage A; C1 passed)

*Rules `experiments/qwen_carrying/results/predictions_stageB.json`, filed before any agent; code
`experiments/qwen_carrying/{stage_b,maximise_b}.py`, briefs `experiments/qwen_carrying/briefs/{writer,maximiser,judge}.md`,
job `cluster/qwen_carrying_b.sbatch`; tests `tests/test_qwen_carrying_b.py`. Replaces the sketch this section held
before stage A ran (24-neuron groups, a "general web text" option, the Gaussian's group), after the stage A analysis
(`reports/analyses/2026-11-05_analysis_e102-qwen-carrying.md`, What next 1). Not gated: $0, one `debug` job under 10 min,
78 Sonnet calls.*

**What runs.** E101's writer → maximiser → judge, on stage A's saved FineWeb-Edu windows (`large/tops.safetensors`, 20
per neuron), with one change: a **forced-choice judge**. Each job is shown with four tasks Qwen was taught in this
project, each phrased by what the model reads and writes, in a fresh letter order per batch; the judge picks exactly one
and marks the fit `strong` or `weak`. Chance is 1 in 4. No kind of token is excluded (E101's first judge excluded
punctuation, which sank its control). The four: `fmt`, anchor 1's template layout ("Given:", "Steps:", "- " lines, "The
answer is N."); `tofu`, anchor 2's author facts; `grpo`, E69's right final number; `gold`, E45's answer-key solutions.
Exact wording and sources in the rules file.

**Groups** (770 distinct neurons, 26 mixed batches of 30 under hashed ids; the key in `results/large/key_b.json`):

| group | what | n |
|---|---|---|
| a1 | anchor 1's carrying neurons, consensus of seeds 0 and 1 | 200 |
| a2m | anchor 2 matched (steps 29/30), consensus of seeds 42, 142: the main comparison | 200 |
| a2f | anchor 2 full (step 1250), both seeds | 200 |
| random | a block-matched partner for 70 of each group, none in any stage-A carrying set | 210 |
| control (ii) | W0 picks at biography answers, by contrast | 90 |

a1 and a2m share 36 neurons, a2m and a2f 90; a shared neuron counts in each group. Control (ii) is stage A's filed
`w0_picks`: W0 engaged at TOFU answers minus the larger of its engaged value at template answers and 0.5 (web text).
That is a contrast against the other saved task and web text, not a raw top. The mean contrast of E101's Amendment 1,
against the one other saved task, picked neurons silent on maths instead (0.70 at biography answers against 0.80+ here),
so it is not used. No GPU control script was needed.

**Rules** (all `commute.carrying_jobs`; min_gap 0.2 and 2 binomial se, as E101; a side under 8 judged neurons or over
10% unjudged is unreadable):

| id | reads | rule | lean |
|---|---|---|---|
| CB1 (gate) | control (ii) picking `tofu` | `forced_gate`: above the random partners' `tofu` rate (`gap_rule`) and above 0.25 (`chance_rule`); if it does not pass, nothing else is read | pass |
| CB2 | random partners marked `strong` | ok iff ≤ 0.5; else PB1, PB2 reported, not read | 0.1-0.4 (E101 Am. 1: 0.117) |
| PB1 | a1 picking `fmt` against a2m picking `fmt` | `gap_rule`: above holds | holds |
| PB2 | a2f picking `tofu` against random | `gap_rule`: above = related | related (Fable) |
| R | explained share per group (maximiser wins), beside E101's 0.82; pick rates; strong-fit rates | reported; explained read only if C2 passes | |

Edge cases run before filing (`tests/test_qwen_carrying_b.py`): a judge picking at random or always `tofu` fails CB1;
always `strong` fails CB2; a control group mostly unjudged is unreadable; se 0 at or below chance reads level; a gap
over 0.2 but under 2 se is no call; PB1 misses when a2m reads like a1.

**Blind to.** The rows: a1 is ranked on template rows and a2m, a2f on TOFU rows, so a judge that ties maths-row neurons
to maths tasks passes PB1 with no reuse; the random change's sets on each row set would control this and are not in
stage B. `tofu` is the only task not about maths, so prose jobs lean to it; the random rate, not chance, is the floor.
PB2 under the elicitation-granularity limit: a new circuit from old pieces reads "related" here as it does to VPD.

**Early checks** (`maximise_b.py`, before any agent text is scored): E0, 16 pool neurons' rank-1 rows read again
against stage A's saved peaks (1e-3 relative, 1e-4 absolute); C2, E101's self test (own windows 2-6 against other
neurons' windows; ≥ 0.8). Either miss exits 1.

**Order.** 1 `stage_b.py batches` (CPU) → 2 writer, one call per batch → 3 maximiser, one per batch → 4 `stage_b.py
judge-inputs` (CPU) → 5 judge, one per batch → 6 `sbatch cluster/qwen_carrying_b.sbatch` → 7 `stage_b.py score` →
`results/score_stageB.json`. Steps 4-5 may run beside 3 and 6. Cost: 78 Sonnet calls; one L40 `debug` job of a few
minutes, $0.

**Finished callers.** E101's `maximise.py` now calls the same functions moved to `commute.carrying_jobs` (`peaks`,
`own_windows`, `self_test_run`, `text_scores`), same arithmetic; it is not re-run on a GPU. `commute.jobs.require_filed`
is new.

## Departures

- E94 scales its Gaussian to the masked K00 - K0; here to clean KL on E91's rows, so no decomposition is loaded. The b
  and both sizes are written to `results/gauss.json`.
- One set size, 1024, for every state, in place of E101's top k_half (capped): chance is then the same across states.
- The contrast is read on unpacked items, where E92's rows are packed: each answer token is compared with itself.
- E98 read log-odds at one position; here log P at every answer position, E92's unit.
- Firing is signed and read signed; the essay reads the positive side, as the corpus tops do here.

## Code, and finished callers that might break

`neuron_r.forward`, `w0_grads` and `top_masks` now go through `hooked`, `firing_grads` and width arguments, same
signatures and defaults: E98's and E101's `run.py` call them, not re-run on a GPU. `carrying_jobs.blocks_of` and
`block_matched` take `width` (default 3072); `tests/test_carrying_jobs.py` passes.

## Not run

- Steering or ablating any carrying neuron.
- Stage B on the random change's carrying sets (the row control for PB1), and W0's engaged values at the GRPO and
  gold-SFT answer positions (control (ii) is contrasted against template answers and web text only).
- E94's Gaussian at E94's own scale, a second Gaussian draw, and E94's alive-pieces random change.
- Anchor 2's past-line states (steps 550, 500), the GRPO states, and the real-author prompt overlap Fable named.
- A relay-free neuron share; the negative side of each neuron's firing.
- E94's VPD column for anchor 2: its reads of the matched states are job 52941, pending; copied in when
  `experiments/qwen_d3_rows/results/table.json` holds them.
