# P1.4 run 2 (E___): fine-tunes that imitate a reuse edit and a new edit, read with the write share

Cell P1.4, run 2 of 3 (reopened 2026-11-23 by whip-lead under the user's standing authority; the task file's item 2:
"trained reuse positive control as P1.4 run 2 after E294: closeness bar first from E139 imitation leftover; debug jobs
<2 h each; VPD seed-0 H100 (~$6) via pod-runner"). Card `experiments/trained_reuse_control/card.md`; filed settings and
rules `experiments/trained_reuse_control/results/predictions.json`; code `run.py`, `cluster/trained_reuse_control.sbatch`.

We have read built edits whose label is known (x2 reads 0.9998 at neurons, new_few 0.0047) and trained fine-tunes whose
label is not (0.014-0.038). This run joins the two: ordinary fine-tunes trained to copy x2's outputs and new_few's
outputs. If the definition reads what an update does inside, a fine-tune that copies a reuse edit closely should read
higher than one that copies a new edit (rival A); if every trained update reads low whatever it does, the two read alike
(rival B, P1.59's finding).

## The closeness bar, first

The definition can only label an imitator that lands near its target. E139 measured the **leftover**: TV(fine-tune,
target) over TV(fine-tune, M), each a mean over the read windows of the total variation between next-token
distributions (`experiments/reuse_variable_finetune/run.py` `distances`, line 239; logged as "leftover share" by
`amend2.py` line 151). At the end of amendment 2's recipe (3,000 steps, peak 1e-5) it read 0.19216, 0.19185 and 0.19243
(`results/amend2_always_on_s*.json`, `steps.3000.dist`, tv_to_F / gap). **Filed bar: an imitator counts only if its
leftover at step 3,000 is at most 0.1924**, both gaps at least E139's 0.02 floor; otherwise "no reading". The bar is
not a guess: it is the closeness the same recipe actually reached on this model, the closest any ordinary fine-tune
here has been measured to a built target. The leftover is measured on both targets, at steps 300, 1,000 and 3,000.

## What runs, in order

| stage | what | where, time (measured per item x count) |
|---|---|---|
| train (3 jobs) | R0b (an untrained copy reads 0.0 from M; target gap >= 0.02), R0d (new_few's rebuilt edit equals the committed record), then one new_few and one x2 seed each: 3,000 steps with R1's early stops | debug L40. new_few ~1.4 s/step (E139's 0.27 s/step at 1,024 windows of 8 tokens, x 41/8) x 3,000 = ~70 min; x2 ~0.04 s/step (E147: 1.1 s per 30 steps at 32 x 41) = ~2 min; ~75 min per job, $0 |
| check (1 job) | E277's stage 0 (identity 0 flips, do-nothing E_b 1, dd 0), then R0c: x2 and new_few read again at neurons, within 2.03e-5 of E277 | debug: 303 s + 1,213 s + 308 s (E277's stage 0, x2, new_few on a cluster L40) = ~31 min, $0 |
| neurons (3 jobs) | after check: E277's stage 0, then 4 of the 12 items each (2 on x2's D, 2 on new_few's) | debug: 303 + 2 x 1,213 + 2 x 308 = ~56 min each, $0 |
| vpd rehearsal (1 job) | E290's VPD stage 0, then R0c at VPD: x2, new_few within 3.3e-5 of E268 | debug, 40-min limit: 16 + 1,445 + 515 s (E268 on a cluster L40S) = ~34 min, $0 |
| vpd (1 pod) | the 4 seed-0 items (imitator and random per target), descriptive | one H100: 2 x 1,445 + 2 x 515 = 66 min (fp32 without TF32 is no faster on an H100 than an L40S, so E268's times stand) + ~10 min set-up (two 268 MB deltas: 54 s at 10 MB/s; the decomposition from W&B ~65 s); ~1.3 h x $3/h = ~$4, ≤ $6 |

The train stage needs nothing from the check, so they run side by side; the neuron reads gate on both (a read job stops
at once if `stage0_repeat_neurons.json` is not done or its item's training is missing). A stop in train stops only that
(target, seed); a stop in check stops every read. The run is not gated (under $60 and 6 h).

## Expected values

| reading | do-nothing | built x2 | built new_few | x2 imitator | new_few imitator | random |
|---|---|---|---|---|---|---|
| E_b over K, neurons | 1.0 | 0.9998 (E277) | 0.0047 | lean 0.02-0.04 (B); A: 0.2-0.8 | 0.02-0.04 | 0.0004-0.013 |
| E_b over K, VPD | 1.0 | 0.761 (E268) | 0.0127 | 0.027-0.040 (E278's trained) | same | 0.013-0.017 |
| leftover at 3,000 | undefined: no reading | - | - | lean 0.1-0.4 (the risk) | well under 0.19 | - |
| KL(M ‖ fine-tune), steps 1/2/3 | 0 | - | - | 0 / ~1e-7 / ~1e-6 | same | - |

W0 (the untrained copy) reads TV and KL to M of exactly 0.0, so it is "no reading" by the gap floor, never a pass.

## Filed predictions and what each is blind to

- **R2 closeness**: blind to how the imitator gets there; it reads outputs on D only (E139: a close copy can be built
  differently inside). That is the point of reading it with E_b.
- **R3 rivals** (neurons, E_b over K as filed): per seed where both imitators count, d = x2 imitator minus new_few
  imitator; A when d >= 0.2 on every such seed, B when under on every one, else split; fewer than 2 seeds, no call. The
  gap is E290's filed rival gap, 18x the seed sd 0.011 of this reading. Blind to the remainders: at neurons they carry
  0.5-0.8 of a trained update's write change (E277), so a low reading may be theirs; E_b without them is reported
  beside. Positive control of the rule: the built pair reads d = 0.995 (A).
- **R4**: randoms at or under 0.05, reported. VPD is descriptive (n = 1).

## Edge cases (run against each rule before filing)

Do-nothing: gap 0 makes the leftover a division by zero; R2 reads the floor first, so "no reading". An empty needed
set reads E_b 1 by convention: |K| < 100 reads "too small", never a pass. A tie d = 0.2 reads A; every d is A or B. No
seed or one seed with both imitators counted: no call, d reported. All x2 imitators missing the bar: no call, whatever
new_few reads.

## Pre-mortem

- Recipe: E139 amendment 2's, sourced; departures below. The step-2 and step-3 KL bands are E139's measured range
  x/÷ 10 on other data: a miss with a sane loss is a band problem, reported, not a reason to retune.
- Data: x2 has only E195's 301 copy24 training windows; 3,000 steps at 32 windows is ~320 passes, so the imitator may
  overfit and miss the bar on D (75 held-out windows). new_few's 100,000 fit windows come from the same Pile rows as D
  at other cuts; exact copies of D are dropped.
- Memory: new_few's batch of 1,024 windows x 41 tokens is accumulated over micro-batches of 256 (2.1 GB of logits each).
- Saved: the step-3,000 delta of each imitator, fp32, 24 matrices (~268 MB each, `results/large/dw_*.pt`); per-item
  num, den, dd, K. The pod needs the two seed-0 deltas and downloads `results/vpd/` and `results/large/vpd_*.npz`.
- Pod loss: the VPD items are written one by one and resume; only the item in flight is lost.

## Departures

1. Trained parameters: the 24 decomposed matrices (decisions.md "Post-training", 2026-10-14; E147's fine-tunes, which
   E277 and E290 read); E139 trained every parameter.
2. Positions per step: x2 is read at all 41 positions, so 32 windows (1,312 positions, arm A's) rather than E139's
   1,024 windows at one position; new_few keeps 1,024 at the last position.
3. Training data: each target's own task windows (E139 used the pair's fit half of 100,000; new_few uses 100,000).

## Design check

- **Goal.** Claim 1 of the paper states the definition and reads it; every trained reading so far is low, and no
  trained update with a known answer has been read. This is that control for imitation, under the frozen definition
  unchanged (`theory/2026-11-22_definition_frozen.md`): no clause, band or score is added; R3 is a test of the
  instrument, not a label.
- **Nearest tried.** E139 (P1.4 run 1) imitated E137's pairs and read the second wording's G and H, never E_b; it set
  the closeness bar used here. E290 (P1.61) trained gain-only fine-tunes (reuse by construction, read 0.18-0.81 at
  neurons against full fine-tunes' 0.012-0.028). Neither trained an ordinary fine-tune towards a known reuse target and
  read it with E_b. A step towards the goal, not a patch.
- **Decisions.** Relies on the frozen definition and E_b over K as its reading (decisions.md, 2026-11-22), E277's and
  E268's featurizations, and the "Post-training" row. No departure.
- **What changes what we do.** A: the instrument tells trained reuse from trained new when the outputs match, so the low
  readings of real fine-tunes describe them. B: closeness of outputs is not enough; with E290 that puts the low readings
  on how ordinary fine-tunes spread a change, and the paper says so. No call (x2 never close): the paper states the
  imitation control could not be built at E139's closeness.
- **Control first.** R0a-R0d run first and alone in each job and stop it on a miss.

## Not run

Seeds 1-2 at VPD; any imitator of x10, rebuilt or new_thin; a longer or larger-data recipe for x2 if it misses the bar
(that needs the cell's last run); an independent check.
