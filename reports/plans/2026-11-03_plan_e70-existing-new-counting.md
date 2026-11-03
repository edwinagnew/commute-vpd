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
