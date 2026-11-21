# Appendix to the P2.49 run-2 plan: settings, pre-mortem, cost arithmetic, rehearsal, code

Plan: `reports/plans/2026-11-21_plan_p2-49-run2-size-ladder.md`. Run 1's plan and appendix:
`reports/plans/2026-11-20_plan_p2-49-crosscoder-proper*.md`.

## A. Settings (gate 2) and what differs from run 1

Run 1's crosscoder settings (`results/predictions.json` `settings.cc.full`) and its bars (`thresholds.full`) are
unchanged. `run2.py` reads them from that filed file. The run-2 settings below are in `results/run2/predictions.json`,
each with a `_source`.

| Setting | Value | Source |
|---|---|---|
| site | A, block 2's input | run 1; where its planted control failed |
| rungs | 0, 2.11, 21.1, 210.99, 2,110, 21,099 per token | run 1's head energy (`run1/items/head.A.c0.json`) × 10^(i−3); lead: geometric, 4-6 rungs, from fine-tune size up |
| rung 3 | run 1's `head.A.c{0,1,2}`, not rerun | the same path at scale 1; `sizes` checks the energy within 1e-4 |
| planted bar | specific ≥ 0.5 on every seed, lead clause off | run 1's S0b |
| size match | within 5%, factor ≤ 1e4 | run 1's Gaussian (`gauss_match_tol`, `run.py gauss`) |
| tasks | copy, rank1, elicit, cipher, prior_cipher; seeds 0-2 | E237's five (E147 `dW_<task>_s{0,1,2}.pt`) |
| random | 3 draws of `finetune.random_direction` at e58 elicit s0's per-matrix norms, seed 249201 + d | run 1's Gaussian construction; 3 draws (user 2026-11-20, E237) |
| pairs at X | rank1 vs cipher, copy vs prior_cipher | E147 inventory's built pairs |

The head is scaled by multiplying the graft's output weight (`graft.out_param`; zeroing it gives back the base,
`graft.py:135`). A state or draw is scaled as dW × c. Each item has its own factor, found by doubling and then log
bisection on its own read rows. **Departures from run 1:** the head at scales other than 1; the self control runs
through the graft at scale 0 on Pile windows, where run 1's was the e54 elicit stream; random draws instead of one
Gaussian with 3 crosscoder seeds. **From E237:** the size is activation change, not weight norm; one factor per item,
not per task.

## B. Pre-mortem (gate 1)

- **Recipe.** The crosscoder is unchanged. At large factors the head's write dominates the fine-tune side's
  activations, and the per-model normaliser (RMS 100) then shrinks the base part on that side. Guard: FVE is checked
  at 12.5% (≥ 0.5) and at the end (≥ 0.8), and the item stops on a miss.
- **Size search.**
  - The size may not grow monotonically with the factor, or may not reach the top rung. Guard: the `sizes` job (before
    any paid round) logs every head rung's factor and its match. An unreachable rung reads "unbuilt" and is skipped.
    Stage-2 items match at their own start and stop with a record.
  - The rehearsal runs the top rung on a state and a draw.
- **Data.**
  - The read rows are each task's held-out windows. The stream's task half is fresh key-value windows from pool rows
    131,072+ (run 1's kv path).
  - The head's stream and read are run 1's: Pile train rows 0-16,383 for the stream, Pile val 1,000-1,999 for the read.
  - The e58 tasks never entered a run-1 stream. They go through the same kv code as e54, and elicit's windows file is
    loaded in the rehearsal.
  - No chat template; the model's own tokeniser.
- **Precision.** Forwards are fp32 with TF32 off; the crosscoder uses TF32 (run 1). The scaled dW is fp32: a factor up
  to 1e4 on bf16-saved deltas stays well inside fp32's range.
- **Memory.**
  - GPU: run 1 peaked at 34.7 GB of 80; an item adds two 24-matrix copies (2 × 113 MB in fp32).
  - Host: as run 1, plus 15 states at 57 MB each if all are kept on disk.
- **Time and cost.** Section C. The projected-time check at 12.5% is run 1's 90 min.
- **Saved.** Per item, the side JSON (≤ 0.2 MB) and the `.pt` (encoder and decoder in bf16, 151 MB in run 1); 26
  items is 3.9 GB. Each item is tarred as it lands and downloaded (15 s at 10 MB/s), with sha256 checked on both
  sides, into `results/run2_dl/`. It is moved to `large/` after the pods are gone (root CLAUDE.md, E36).
  `sizes.json`, `ladder.json` and `summary.json` are small and committed.
- **Pod loss.** There is a checkpoint every 10,000 steps, so a lost pod costs at most one item (28 min, $2). The
  ladder is driven from downloaded JSONs, so a lost pod loses no decision state.
- **Controls.** The self control runs first and alone. Rung 3 comes from the same card class (H100 80GB HBM3) and the
  same code. The random draws are stage 2's first items.
- **Predictions.** The edge cases are in predictions.json. `score2.py ladder` and `read` were run on CPU on fake
  items (2026-11-20). The cases checked were:
  - the round-1 list;
  - the candidate branch (rung 4 seeds 1-2 plus 2 stage-2 keys);
  - X found with a 0.5 tie;
  - G1 failing on a missing draw.
- **Code never run on a GPU.** `Scaled.apply`, `Job.match`, `zero_diff`, `before`, `sizes`, the random draw on an E147
  profile, and e58 elicit's windows. All are in the rehearsal.
- **Reading risk.** At X the planted model may be broken (run 1 at rung 3: Pile loss 2.71 → 7.45). This is reported
  beside the range, not a stop. Run 1's spliced-loss bar may leave stage-2 items unread at X (choice 2 for the lead).

## C. Cost arithmetic (H100 SXM, $4.40/h)

**Measured inputs.**
- Site-A crosscoder: self 27.4 min, heads 27.7, 27.6, 27.7, k sweep 27.9 and 29.6 (`results/run1/pod0_*.log`,
  `pod1_1{2,3}.log`). Taken as 27.7.
- Size search: + 0.4 min, from the B Gaussian items (34.8-35.3 min) against the B pair items (34.3-35.1).
- Per item: **28.2 min**.
- Set-up: from start() (`fleet/cell show P2.49`, PST) to each pod log's first line (UTC).
  - Stage 0: 16:52-16:55 against 01:21-01:24, so 29 min.
  - Stage 1: 18:03 against 02:32 (29 min) and 18:15 against 02:39:57 (25 min).
  - Run 2 uploads 0.7 GB more per pod. All 15 E147 states are 850 MB (57 MB each, `ls`), against run 1's two 113 MB
    states, so 70 s more at 10 MB/s.
  - Taken as **30 min**.
- Hand-back: last item done to last tar on the dev node, 3 s (stage 0) and 2.5 min (stage 1). Taken as **5 min**.

**Schedule.**
- Pod 1 starts at t = 0 and spends 30 min on set-up, 3 min on `sizes` (estimated: about 120 forwards of 16k tokens,
  each about 0.04 s from run 1's 2.6 s per 1.05M-token chunk, plus model load) and 28.2 min on self. It is free at
  t = 61.
- Pods 2-4 start when self starts training (t = 33) and are set up by t = 63.
- Round 1 runs t = 63-91 and round 2 t = 91-119.
- Stage 2 has 16 items left on 4 pods: 4 rounds, t = 119-232. The hand-back ends at t = 237.

**Pod-minutes.**
- Pod 1: 237. Pods 2-4: 3 × 204 = 612. Total **848 pod-min = 14.1 h = $62**.
- By stage: stage 0 is 61 + 3 × 30 = 151 ($11.1). Each round is 4 × 28.2 = 113 ($8.3). Stage 2's rest is 4 × 113 = 451,
  plus the hand-back 4 × 5 = 20, so 471 ($34.5).

**Branches.**
- A failed candidate adds a round: +113 pod-min, +$8.3, +28 min. Rungs 4 and 5 are the likely candidates, so at most
  one extra round is likely: $70, 4.4 h.
- No rung passes in round 1: pod 1 runs 61 + 28 + 5 = 94 min and pods 2-4 run 61 min each, so 277 pod-min ($20) and
  1.6 h.

**Lever (lead's call).** Drop rungs 1-2 from round 1 and run seed 1 of rungs 4 and 5 in their place. This saves one
round ($8, 28 min) when rung 4 or 5 passes, but loses the readings near a fine-tune's size.

## D. Rehearsal (gate 3)

`cluster/crosscoder_proper_run2_rehearsal.sbatch` runs on one L40 with the `debug` QoS, from the clone root after the
E-number is filed. It uses run 1's rehearsal sizes: 3,072 latents, 4.1M tokens, full read and latent-scaling sizes,
and run 1's rehearsal bars (planted bar relaxed to −1, so every rung passes and the X branch runs). In order:
1. `sizes`;
2. self at scale 0, then `check0`;
3. round 1 (rungs 1, 2, 4, 5), then `ladder`;
4. stage 2 at rung 5 (random d0, e58 elicit s0, e58 cipher s0), then `read --x 5`;
5. one item tarred, with its sha256.

Time: about 20 min. Run 1's tiny items took 0.69-0.75 min each on the L40 (`results/rehearse/items`); here 8 items,
plus 4 model loads and `sizes`. It writes into the plan the measured factor per rung, the energy per item, the time
per item and the peak GPU memory. It runs before `fleet/rehearsed`. Rung 3 in the rehearsal is run 1's rehearsal
items (`results/rehearse/`).

## E. Run commands (for the pod-runner brief)

```
PRED_SHA=<sha> python experiments/crosscoder_proper/run2.py sizes                       # pod 1, stage 0
PRED_SHA=<sha> python experiments/crosscoder_proper/run2.py item self.A.c0               # pod 1, stage 0
~/commute-agents/fleet/cpu-run python3 experiments/crosscoder_proper/score2.py check0    # dev node, after download
PRED_SHA=<sha> python experiments/crosscoder_proper/run2.py item head-r1.A.c0            # round 1, one key a pod
~/commute-agents/fleet/cpu-run python3 experiments/crosscoder_proper/score2.py ladder    # after each round: next_ladder, next_stage2
~/commute-agents/fleet/cpu-run python3 experiments/crosscoder_proper/score2.py read      # after stage 2
```

For `ladder` to see rung 3, run 1's `results/run1/items/head.A.c*.json` must be present; they are committed.
