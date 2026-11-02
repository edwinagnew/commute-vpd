# Review: E59 code and predictions (commit 94605be)

Reviewer: a fresh agent that wrote neither the plan nor the code. Scope: `src/commute/readout.py`,
`experiments/jain_readout/run.py`, `experiments/jain_readout/results/predictions.json` (incl. `writer_decisions`,
`edge_cases`), `cluster/jain_readout.sbatch`, `tests/test_readout.py`, against
`reports/plans/2026-10-30_plan_baseline-next2-draft.md`. Read-only; no model loaded. Tests: 12 passed
(`cpu-run .venv/bin/pytest -q tests/test_readout.py`). Lint: ruff and pyright clean on the three files.

**Verdict.** No code bug found that would make the job measure something other than what the plan and predictions
say. One finding about the filed quantity matters before the run: the ablation share A is read in log P, and the
three kinds of state sit at very different points of the softmax (elicit matched P ≈ 0.93, cipher matched P ≈ 0.04,
cipher own-step P ≈ 0.99), so the same logit share reads as different A. That can fail P2 and its own-step clause
with nothing but the unit. It is the plan's definition, so the fix is the lead's or the user's call (H1).

## Findings by severity

### H1 (high, measurement as filed): A in log P depends on saturation, which differs by kind
`readout.py:127-135` (`ablation`, via `steering.read`'s default `logp_at`); predictions.json `settings.A`.

The plan defines A = (log P_tuned − log P_ablated) / G in log P. Removing the direction lowers the target's logit by
some amount c. In log-odds that costs c, whatever P is. In log P it costs about c only while P is small, and much less
near P = 1. The states differ in exactly this respect. From E54's geometric-mean rates (`stage1.json`) and the filed G:
elicit P0 ≈ 0.0017 and G ≈ 6.3, so P_tuned ≈ 0.93; cipher P0 ≈ 1.1e-6 and G ≈ 10.5, so P_tuned ≈ 0.04; cipher
own-step G ≈ 13.7, so P_tuned ≈ 0.99.

Failure scenario (geometric-mean numbers, per-window spread ignored): both fine-tunes route exactly half of their
log-odds gain through the readout direction.
- Elicit: log-odds gain ≈ 9.0, removing 4.5 takes P from 0.93 to ≈ 0.13, a drop of ≈ 2.0 nats, so A ≈ 0.31.
- Cipher (matched): unsaturated, the drop is ≈ 5.3 nats, so A ≈ 0.50.
- Cipher own-step: log-odds gain ≈ 18.3, removing 9.2 leaves P ≈ 0.01, a drop of ≈ 4.6 nats, so A ≈ 0.34.

So P2 reads "cipher higher by 0.19" and its own-step clause fails (|0.34 − 0.50| = 0.16 > 0.1), on states that use
the direction identically. With elicit at 0.7 of its logit gain and the cipher at 0.5, A reads 0.58 vs 0.50: no
separation. The decision tree then sends "P2 fails: r's reading has no causal footing outside VPD", an artefact of
the unit. This is the mirror of E54's steering note ("the same nats are a larger share of the elicit map's smaller
gain"): here the elicit side is the compressed one. P2 is biased towards fail, and the own-step clause compares a
saturated state with an unsaturated one.

Fix, before the run (cheap: `steering.read` already takes `readout=ST.logodds_at`, the route split's unit, exact near
P = 1):
1. Preferred: read A in log-odds (G and the drop both via `logodds_at`). This changes the plan's filed definition, so
   it needs the user's one-line OK before the job, with a dated note in predictions.json (nothing has run).
2. At least: record, unscored, beside each state's A: A in log-odds, the drop in nats, and the state's P_tuned on the
   scoring half, so the analysis can tell a unit artefact from a result. Without them a P2 fail cannot be read.

### M1 (medium, measurement as filed): A includes W0's own use of the direction
`readout.py:131-135`. The removal takes out the whole projection h·u, not the fine-tune's change of it. The W0
positive control requires that W0 already loses ≥ 0.3 nats (median) on elicit from the same removal, and the cipher's
W0 loss should be about 0 (W0 ranks the cipher values > 100). So part of A_elicit is W0's own use, not the gain's. In
log P this offset is mostly hidden by saturation of the elicit states (H1). In log-odds (H1, fix 1) it is not: a W0
loss of 1-2 logits on a log-odds gain of ≈ 9 adds 0.1-0.2 to A_elicit and nothing to A_cipher, the size of min_gap.
Failure scenario: fine-tunes that leave h·u untouched still give A_elicit ≈ 0.15, A_cipher ≈ 0, and P2 passes.
Fix: report the W0-subtracted share (drop_state − drop_W0, same windows, same task) / G beside A. `controls` already
records W0's mean drop per task on the scoring half (`run.py:141`), so it can be computed from run.json, but put it in
the record explicitly. If H1 fix 1 is taken, the filed A should be the W0-subtracted one.

### M2 (medium, power): no positive control that A can read high on a fine-tuned state
The W0 control (`run.py:144`) shows the direction is causal at W0, where P is small. Nothing shows A can reach ~1 on
a tuned state at l*. Without that, A low for both kinds ("the gain survives the removal") cannot be told from a
metric that cannot read high. Fix: add the ablation at the last block on every state, reported and unscored. The plan's
edge cases note the last-block readout is close to the unembedding, so removing the target's column should take
most of the gain (A ≳ 1 on every state). That is 12 more forwards of 48 windows, seconds.

### L1 (low): the random direction does not test centring
`readout.py:103-106`. u is the raw readout column w_v. Softmax ignores any component common to all columns, so if
the trained columns share a large mean, u is mostly that mean and removing it moves every logit alike. The W0
positive control would catch a direction that does nothing (it stops the job), so this cannot pass silently. Fix
(optional): record cos(w_v, mean column) per target value, so a W0-control failure can be diagnosed in one read.

### L2 (low): a stop inside readout training leaves run.json without `stopped`
`readout.py:73-76` raises `SystemExit` directly. run.json then holds only stage 0's inputs, with no note why the job
ended (the log has it). Fix: let `train_readouts` return a reason or raise an exception that `run` turns into
`stop(out, ...)`.

### L3 (low): host memory request is unmeasured
`cluster/jain_readout.sbatch:26` asks 16G. E54 and E56 ran the same `steering.Net` with 24G and their peak is not
recorded. The estimate (3 GB decomposition loaded via CPU, CUDA context, twelve 113 MB weight changes that load to
the device they were saved on) fits 16G, but with less margin than the precedent. An OOM costs a resubmit only.
Fix: none needed; record MaxRSS from the job for the next request.

### Nit
`readout.py:131-132`: the unablated reading runs all 97 windows and is sliced, while the ablated one runs the 48
scoring windows. Different batch shapes give fp32 differences around 1e-6 nats, far below every threshold. Reading
both on `t.toks[sc]` would make the drop exactly 0 when the edit is a no-op.

## Writer's decisions
predictions.json lists five `writer_decisions`, not seven. The other two are in the code and `departures`: window
length 41 with position 0 left out, and the full 50,277-token vocabulary where the plan's cost line says "a small
vocabulary".
1. Twelve states, not fifteen (cipher_own s2-s4 not saved). Checked in `job2.json`: their sha256 is null. The
   own-step clause on two seeds is a size check, not the separation. Acceptable; not degenerate. H1 applies to it.
2. P1 counts block 0. The plan's table and edge cases say "≤ 2", its P1 line "1 or 2". With 0-indexed blocks and
   block 3 last, "≤ 2" matches l*'s definition. This widens P1 slightly. Acceptable; flag it in the write-up.
3. Fit-half possession as a fraction (21 of 28 covered keys). It only chooses l*, and it keeps the plan's 75%.
   Acceptable.
4. Do-nothing as A_rand on every fine-tuned state, not "on W0". On W0 a share has no denominator, and the plan's table
   puts |A| ≤ 0.05 in the A row. This is the only sensible reading. W0's random drop is recorded beside it.
5. W0 control and A on the scoring half, l* on the fit half. As the plan says.
6. (undeclared) Window 41, position 0 excluded. The task windows' length and the answer position 20 lie inside the
   trained range. Fine.
7. (undeclared) Full vocabulary. This is required: the possession band (rank ≤ 20) and P3 (rank 2-20 and > 100) only
   mean what the construction meant if ranks are over the full vocabulary. Correct, and it costs only the training
   time (below).

None of the seven changes what the plan asked or makes a prediction degenerate. H1 is a property of the plan's own
definition.

## Checked, fine
- **No task leakage.** The readout trains only on `pile_windows` of train rows 32768-40959 (`readout.py:36-40, 56-77`).
  E54's task windows come from train rows 0-32767 (`keydependent_elicit/results/predictions.json`
  `task_windows_rule`; `keydep.py:144`), so the rows are disjoint. Held-out loss uses val rows 0-255. Both `.npy`
  caches exist in `artifacts/sets/`, so the job makes no HF download. Task windows touch only ranks and ablations.
- **Readout sites.** `RESID = in1, in2, in3, out3` is the output of blocks 0-3: pre-hook input of the next block, and
  the last block's output before `ln_f`. `Block.forward` returns a tensor (param_decomp `llama_simple_mlp.py:312-315`),
  so the `out3` forward hook captures and edits a tensor. The embedding is not read, as the plan asks.
- **Direction.** `Probes.w` is (k, d, V) = `nn.Linear.weight.T`, so `w[block][:, tgt].T` is the weight row of each
  target value (the plan's "column" of the d→V map), unit-normalised per window. The bias is untouched and
  correctly so, because it is not in the residual. Shared values share the direction.
- **Removal site and model.** `ablation` reads the state with `net.at(dW)` and applies `remove_at` at
  `RESID[block]`, answer position `t.half` only. With causal attention, only position `t.half`'s logits (which
  predict the value) can change. u comes from W0's readout, applied inside the fine-tuned forward, as the plan says.
  `remove_at` is tested: orthogonal, idempotent, other positions unchanged.
- **Halves.** l* comes from fit-half possession (`run.py:126`). A, G_score, the W0 control and A_rand are on the
  scoring half (`readout.py:130-135`). P1 and P3 use all 97 windows and select nothing that is scored.
- **Ranks.** `target_ranks` = 1 + count strictly above, at `t.half`, per block. The readout at position 20 predicts
  token 21, the value, consistent with `logp_at(logits, half, tgt)`. Key medians are lower medians, and a key with no
  window is excluded from numerator and denominator.
- **Filed rules = code.** possession (inclusive fraction, empty → not possessed), `earliest` (blocks 0-2, earliest
  wins), p1, p3 (band inclusive, cipher > 100 strict), do_nothing (inclusive, nan/missing/empty fail), p2
  (`elicit.separation` with min_gap; own-step strict; None/nan → unreadable; void passes through), and branch all
  match predictions.json's rules and edge cases. The tests cover each named bound.
- **Stage 0 cannot pass vacuously.** sha256 of windows and all twelve states against `stage1.json`, `stage2.json`
  and `job2.json` (all keys present; verified with jq). W0 rates and every G must agree within 1e-6, using the same
  `ST.gains` forwards on the single 8×L40 node. A missing file crashes before scoring. P3 needs real counts. A no-op
  hook would pass the do-nothing but fail the W0 control. When l* is None the hook is unchecked, but P2 is then void
  by rule.
- **Guards.** The step-20 projection is effectively synchronised, because each step's host-to-device copy from
  pageable memory waits for the stream. Non-finite losses are checked at every logged step (every 250). A NaN is
  sticky under AdamW, so it is caught by step 250 at most (~20 s lost).
- **Memory and time on an L40 (46 GB), TF32 off.**
  - Probe logits are 4 × 3,936 × 50,277 fp32 ≈ 3.2 GB. The sliced copy, log-softmax and gradients bring the peak to
    roughly 15-18 GB.
  - Probe weights, gradients and AdamW states take ≈ 2.5 GB. The decomposition plus twelve weight changes take a few
    GB more. That fits easily.
  - Compute is ≈ 2.4 TFLOP per step (forward plus weight gradient; the inputs need no gradient), ≈ 60-80 ms at
    realistic fp32 throughput, so ≈ 4-6 min for 4,000 steps. Held-out loss and all ablations take seconds.
  - The writer's 8-15 min is plausible, and the 15-min training cap leaves the 30-min wall safe.
- **Outputs.** run.json (small, ranks and medians only) goes in `results/`. `readouts.pt` (≈ 618 MB) goes to
  `results/large/`, which is gitignored (`.gitignore:27`) and links to the existing
  `~/commute-data/results-large/jain_readout`. Nothing large is committed. A Slurm job writes there directly, so the
  remote-kernels symlink issue does not arise.
- **No shared function changed.** `git show --stat 94605be` adds only new files plus the PENDING and TRIED lines.
  `probe.py` and `steering.py` were last touched on 2026-10-30 (29bc903). `probe.Probes` and the constants are
  imported unchanged, so `pcfg_anchors/run.py` and the frozen E52/E54/E56 callers are unaffected.
- run.py refuses unless predictions.json is committed unchanged, and it asserts the schedule, window and sites
  against the file.

## Unsettled
- **Last-block gap.** Whether a linear map on the pre-`ln_f` residual comes within 0.5 nats of the model's own loss
  is not known. A miss stops the job at $0, but the bound then needs its own amendment.
- **Key coverage.** That all 32 keys appear among the 97 held-out windows (and 28 in the fit half) is the writer's
  statement. I did not load the windows to check. The rule handles any coverage.
- **Rare value tokens.** The readout columns of value tokens seen rarely in 15M training tokens are trained mostly
  through the softmax denominator. If cipher values are rare, their u is close to a generic "rare token" direction,
  which could make A_cipher small for reasons unrelated to the fine-tune. Cheap to record: each value's count in the
  training windows and its column norm.
- The H1 numbers use geometric-mean rates. Per-window P_tuned varies, so the size of the bias is an estimate, but
  its sign (elicit compressed, cipher matched not, cipher own-step compressed) follows from the filed G and rates.

## Not run
No GPU run, no model load, no loading of the windows or states (keys' coverage and per-window P_tuned unchecked);
no check of the writer's run-time estimate on a card.
