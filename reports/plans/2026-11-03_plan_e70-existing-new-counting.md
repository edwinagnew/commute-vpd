# Plan: E70, the existing/new definition measured in full on the counting model

The user's task (2026-11-03) called this E68; `qwen-decomp-lead` had claimed E68 that morning, so `fleet/ecode` gave E70.
Restated: on LOW, a linear probe for the count of a, trained on pretraining data, reported by block with no possession
threshold. In the fine-tuned task P and task N states of E55 C1b, remove that probe's direction and report the share of
the log-odds gain lost. Random directions and LOW itself are the controls and run first. Any band comes from the controls
in the same job. No VPD, no Qwen. Cluster `debug`, $0.

## Design check
- **Goal.** `GOALS.md`'s second question, new capability or strengthened one, needs a label that does not depend on VPD.
  `decisions.md` (row "Reuse line, existing vs new") defines one and it has never been read whole on a pair whose
  answer is fixed by construction: P (count a + 1) reuses count a, N (count b) does not.
- **Nearest tried.** E55 C1b read pruning, reverse fine-tuning, steering and a probe of each state's answers, none of
  them a removal of a pretraining-trained direction. E59 and E64 ran the removal on the Pile-trained readouts, where the
  labels were themselves in question and the last-block check missed. New here: the removal on a pair with known labels.
- **Patch or step.** A step: it is the positive control the definition itself has lacked.
- **Decisions.** Relies on the reuse-line row, the mechanistic row of 2026-11-02 and the held-out demotion. Departure, by
  the user's instruction: no possession threshold, against "filed before any run" in that row.
- **What changes next.** If P and N separate, the definition's causal test has teeth on known labels and can score
  cheap methods on Qwen. If they do not, the removal cannot serve as ground truth for the method table.
- **Control first.** LOW, then random directions, then the probe on LOW, each with a stop; no fine-tune starts before.
- **Differences from the reference.** The C1b states are lost (below), so this retrains from the same code and seeds on
  the same card type. The controls compare LOW and every state's task accuracy with the committed `a1_LOW.json`, `c1b.json`.

## States are gone
`find` over `~/commute-data` and `~/commute-agents`, the worker's `/ephemeral` and `commute-old-leftovers.tgz` finds no
`pt_LOW` and no `P_1e-*`: they lived in the removed `reuse-lead` clone. LOW is retrained with `run.pretrain("LOW", A1, ...)`
(50K steps, ~10 min), then all 20 C1b runs with `c1b.finetune` (2 tasks, 2 rates, 5 seeds; ~2 min each), saved at 1K and 10K.

## Measurement
Every read is at `=`, in fp32, on held-out rows.
- **Declared inputs.** The definition's sub-domain X_C: 'C a 40' rows of the pretraining distribution, answer count(a).
  The fine-tune's inputs, 'C b 40' natural rows, are read second, as whether the same probe carries to where it is removed.
- **Probe, one direction (the headline).** Ridge regression of count(a) on LOW's residual per block, trained on 'C a 40'
  rows; accuracy = rounded prediction. Its unit weight vector is the direction.
- **Probe, subspace (reported beside it).** The softmax probe of `commute.probe.train_probes`, op a; centred columns of
  the count values seen, spanning rank r. It answers whether one direction misses a count held in several, and carries
  the shuffled-label floor (Hewitt & Liang).
- **Removal.** At block b, `h - (h·u)u` at `=`, through `forward_at_art`'s patch; later blocks still read the text, so
  early blocks can understate (blind spot, read off LOW's control by block).
- **Share.** `revival`-style means over rows, numbers from `readout.shares`: A_lo = (the state's log-odds drop under the
  removal, minus LOW's, same rows and direction) / the state's log-odds gain over LOW, for its own target token. A_lo_raw beside it.
- **g∘f (Def 2).** A table from the LOW probe's rounded reading to the state's answer, fit on half the rows, read on the
  other half. Expected near the probe's accuracy for P, far lower for N. Carrying to unseen values is not read: no held-out rule.

## Run order and stops
1. LOW trained; count_a within 0.01 of `a1_LOW.json` (0.99975), C1b's `reproduce_tol`. Miss: stop.
2. Both probes by block on the declared inputs.
3. On LOW, 'C a 40': log-odds drop of count(a) under 20 random unit directions per block (rank-r random subspaces for the
   subspace), then under the probe. **Band for LOW** = 95th percentile of |random drop| at that block. If the probe's drop
   is within the band at every block for both probes: stop, no fine-tune, report (the removal has no teeth).
4. Fine-tunes. Task accuracy against `c1b.json`, logged and reported, stops nothing: a seed may differ.
5. Per state, per block: random shares first (20 draws), then the probe's, then g∘f. **Band for a state's share at a block**
   = max over 10K states of the 95th percentile of |random share|, computed in this job.

## Numbers and what each should read
| Number | Expected |
|---|---|
| LOW count_a | 0.99975 ± 0.01 |
| Softmax accuracy, 'C a 40', blocks 1-6 | ~ .34 .87 .92 .98 1 1 (`a1c_probe.json`); shuffled ~ .33 |
| Ridge accuracy, 'C a 40' | no reference; above majority (~.33) from block 2, high by block 6 |
| Both on 'C b 40', LOW | near their 'C a 40' values at late blocks, since LOW answers count(a) there; lower early |
| LOW's drop under the probe | above band at late blocks, at least at block 6 |
| Random share on any state | within band, by construction; band itself expected ≪ 0.1 |
| Share, P, 10K | high at late blocks (lean > 0.5 at block 6) |
| Share, N, 10K | near the band at blocks 1-5; may rise at block 6, where the count of a shares the output head with every answer (E64's last block read 0.34-0.53 on all kinds) |
| g∘f, 10K | P near the probe's accuracy; N near its floor, about .35 |

## Filed rules (`results/predictions_e70.json`)
- **R1, control.** Steps 1 and 3 above.
- **R2, separation (lean: holds).** At a block, per probe: min share over the 10 P 10K states minus max over the 10 N 10K
  states exceeds that block's band. Read at every block and reported as a table. The definition's causal test separates
  the pair if R2 holds at any block from 1 to 5; holding only at block 6 is reported as output-side, no call.
  A reversal (N above P by the band at any block) gets the announcement a reversal gets.
- Edge cases: gain below `floor_min` gives nan, state left out and counted; all-nan side gives "no call"; a state under
  task accuracy 0.9 is "not yet learned" and left out of R2 (only 1K states are expected to be, as in C1b).
- Blind to: count held non-linearly; information re-read from the text after the removal block; a direction shared
  between count a and count b, which costs N as well (hence the count a–count b correlation on these rows is printed).
- Noise floor: the bands; seed spread across the five seeds is in the table.

## Not in this run
Pruning and reverse fine-tuning of the definition's causal clause (C1b's readings stand and are quoted). VPD. Qwen.

## Cost and gates
$0, cluster `debug`, ~3 jobs of ≤ 30 min with resume (LOW and controls; fine-tunes). Under every gate: no review, the lead reads it.

## Amendment 1 (2026-11-03): remove the direction the count moves the residual along

Amendment 1 of 2 for E70. Rules `experiments/pcfg_anchors/results/predictions_e70_a1.json`. Code `experiments/pcfg_anchors/e70_a1.py`,
`src/commute/removal.py`, `cluster/e70_a1.sbatch`. $0, `debug`, about 17 minutes, or 2 if LOW's control stops it.

**Taken by default.** Ask 15 (card `decision-counting-one-direction`) was taken by default at 15:28 on 2026-11-03 after two
hours without an answer, under the fast lane's rule for an ask under the gate and inside the plan. The user has not chosen it.

**Restated.** On LOW, per block, at `=`: the unit covariance of the residual with count(a), on the rows E70's ridge probe was
fit on, is removed as E70 removes a direction. LOW's own log-odds drop for count(a) is read first against 20 random single
directions from the same job. Only if LOW loses log-odds are task P and task N read at 10,000 steps, as the share of the
log-odds gain lost, on E70's definition. Nothing is retrained. No VPD, no Qwen.

**Why this should differ, as reasoning to test.** Regression weights are about the inverse covariance times the
cross-covariance, so they favour low-variance directions and cancel noise; the covariance with the count is where the count
moves the residual (Haufe et al. 2014). Zero cross-covariance is the condition for no linear readout (Belrose et al. 2023,
LEACE). Both are cited from memory and not re-read. The job tests this reasoning: the variance shares, the cosine and the
refits read it directly.

**Two limits found while coding, so the reasoning is narrower than the ask's.**
- The zero covariance is with the scalar count. LOW's head is a softmax over count tokens, and LEACE's condition for a class
  is on the one-hot label, up to six directions here. So after this removal a linear readout of the scalar fails, while a
  readout of each count class need not. LOW can keep its answers with R3 passing, and a closed test then says this of one
  scalar direction, not of the six.
- The low-variance story may not cover late blocks. Since the ridge reads the count perfectly, the variance along w is
  var(count)/|w|^2 = 0.014, 0.059, 0.39, 0.99, 4.1 at blocks 2-6 (`probes.pt`), growing 300-fold, and E70's removal did nothing
  at those blocks too. The trace is not saved, so the share is read in this job. w is also mostly outside the softmax span: the
  part of unit w in that span has size 0.12 to 0.35.

**Departures.** An orthogonal removal, h minus its component along a_b, not LEACE's oblique map (subtract along the covariance, read with the filter), and on
a scalar count, not a one-hot concept. At `=` only, as E70, so later blocks still read the text. The covariance is with the
label, not with the probe's output (Haufe's form), which agree where the ridge reads the count.

### Design check
- **Goal.** `GOALS.md`'s question of new against strengthened capabilities; it needs a causal test with labels known by construction.
- **Nearest tried.** E70: the ridge direction does nothing on LOW, six softmax directions do. New here: the rank-1 direction
  chosen by covariance, and accuracy after removal.
- **Patch or step.** A patch on E70's control, the only route to the one-direction reading the task named. Class 1 (GPU passes on
  saved states), and taken by default under the fast lane.
- **Decisions.** Relies on the reuse-line and mechanistic rows and the fast lane's default. No departure from `decisions.md`.
- **What changes next.** The tree below: closes the one-direction test, gives the reading, or drops one direction as ground truth.
- **Control first.** R0, then random draws, then a_b on LOW, alone. R3 and R1 stop before any state is read. Nothing is
  reproduced that differs from E70's control except fp32 rounding, which R0 tests.
- **Early guards.** R0 stops in the first minute. Every number goes to the log. **Saved.** a_b per block in `results/e70_a1.json`.

### Run order and stops
1. **R0.** Ridge accuracy and floor on the held rows against `e70.json`, within one row. Then the 20 random drops against E70's
   same draws, within 1% of its band: same code, rows and card type, so only rounding can differ. A miss stops.
2. **Control.** a_b, then LOW's drop and count accuracy under it, beside the band, and under w again.
3. **Why, on LOW only.** A ridge refit on fresh rows, read on the held rows, after a_b and after w. Variance share along a_b,
   w and random. cos(a_b, w). R3: at blocks 2-6 the refit after a_b is below the midpoint of the floor and the reading
   before removal; a miss stops. Then R1 with no teeth stops.
4. **States.** The 20 at 10,000 steps: 20 random shares, then a_b's scored share, raw share, share in nats, accuracy.
5. **Verdict.** R2 by block, E70's code, then the tree.

### The tree (first match wins, so each case has one branch)
R2 is read only at blocks where LOW's control has teeth; all blocks are printed. E70's half-percent reversal at block 1, on a
removal with no effect on LOW, is why.
- R0 or R3 misses: stop, no branch read, apparatus to the lead.
- No teeth at any block: **closed**. The one-direction test closes, P and N not read, six directions remain the only reading.
- A reversal at a read block: **cannot serve**, announced.
- R2 holds at a read block in 1-5: **separates**. The one-direction reading the task named.
- A read block in 1-5 unreadable, or every read block unreadable: **no call**, reported, no amendment.
- Otherwise, including R2 holding at block 6 alone: **cannot serve**. P and N lose alike, so one direction cannot serve as ground truth here.

### Numbers and what each should read
| Number | At a_b | Random | Do nothing or w |
|---|---|---|---|
| LOW's drop | above band at some of blocks 3-6, about 0.3 to 5; low confidence | inside band; band 0.048, 0.132, 0.215, 0.122, 0.117, 0.077 | 0; w: 0.042, 0.027, -0.035, -0.104, -0.122, -0.020 |
| LOW's count accuracy | at or above about 0.7, lower than clean where there are teeth | at or above 0.99 | about 0.9998 |
| Refit accuracy | 0.33 to 0.40 (floor 0.3575) | not run | none: 0.38, 0.9975, 1, 1, 1, 1; w: above 0.9 at blocks 3-6 |
| Variance share | above w and 1/192 at blocks 2-6 | 0.0052 | w: under 1/192 at blocks 2-4, no lean at 5, 6 |
| cos(a_b, w) | size under 0.5 at blocks 2-6 | | |
| State band | | E70's ridge band: 0.0042, 0.024, 0.012, 0.019, 0.011, 0.009 | |

**Existing/new shares, filed.** Task P: above the band and below E70's six-direction means of 0.125, 0.238, 0.553, 0.720, 0.564 at
blocks 2-6, highest at blocks 4-5, since its answer is a function of count(a) and six directions cost it most there. The scored
share sits above the raw one by LOW's negative drop over the gain, as in E70. Task N: under P at blocks 4-5 and above the band at
blocks 4-6, up to about E70's 0.20, 0.34, 0.49. N may rise because count(a) and count(b) correlate at -0.50 on those rows; a_b
is fit on 'C a 40' rows, where the correlation is also -0.50, so it can carry count(b). Weak lean: R2 holds at block 4 or 5.
Share in nats at most about E70's, P 0.155, 0.308 and N 0.002, 0.013 at blocks 4, 5.

**Dry check, no model.** E70's own 10K ridge shares through `stage_verdict` gave E70's calls, gaps and bands exactly at all six
blocks. `e70.py` is unchanged and still runs: `removal.py` only gained functions. No code path of this run has met a GPU; its first
minutes do that.

### Not run
- LEACE's oblique map, and an erasure of the one-hot count of rank up to 6.
- The six-direction removal on rows where the two counts are uncorrelated.
- A refit on 'C b 40' rows.
- The 1K states, a second LOW or probe seed, any retraining.
- VPD and Qwen.

### Cost and gates
$0, one `debug` job. Under every gate: no review, the lead reads it. Amendment 1 of 2.
