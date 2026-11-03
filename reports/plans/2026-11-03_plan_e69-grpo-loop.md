# Plan: the GRPO loop on Qwen3-1.7B: RL, its shuffled-reward null and own-samples SFT, with δW saved (E69)

*2026-11-03, qwen-rl-lead. Approved work: task file and `reports/plans/2026-11-03_plan_qwen-reorient.md` §D2, class 4.
Recipe: `reports/plans/2026-10-28_plan_qwen-rl.md` §3-5, changed only as listed under Departures. Words as in `GOALS.md` → Terms.*

## The question, in plain words

D2 needs the same table for an SFT update and an RL update of the same model, and no RL update exists yet. This run
trains the unchanged Qwen3-1.7B (thinking off) on GSM8K train by GRPO, by the same loop with rewards shuffled within
each problem (the null), and by SFT on the model's own correct samples, on the same prompts at each step. It saves the
update δW of the 196 decomposed matrices at steps 1, 2, 5, 10, 20, 50, 100 for `qwen-decomp-lead` to map and test.
This run applies no map and reads no decomposition.

## Design check (`/design`)

- **Goal:** the weak version on the larger model: real post-training updates to commute. D2 cannot start without them.
- **Nearest tried:** E40 and E45 (SFT on gold and own samples at 1e-5, warm-up, cosine, clipping, autocast), E61 (the
  rounding; retrained at full precision). None is on-policy. New here: the first RL update, its null, and an SFT arm on
  the same prompts and optimiser.
- **Patch or step:** a step, the plan's D2.
- **Decisions relied on:** non-thinking GSM8K, the baseline's prompt and strict grader (2026-10-27); own samples as the
  canonical SFT (2026-10-28).
- **What changes what we do next:** RL gains under 0.01 → branch R2. Reward falls early → branch R3. Otherwise each δW
  goes to `qwen-decomp-lead`, then anchor 1.
- **Control first:** stage A0, alone, with a stop; then the arm's step-1 checks (below).
- **Class:** 4, on the task file. Treated as gated: on the estimate below it runs over 6 h.

## Departures from the 2026-10-28 recipe, each with its reason

1. **Full-precision forward for rollouts and training, not bf16 autocast.** E61: autocast rounds each weight to the bf16
   grid, half a step about 6e-5 at the median weight, and W0 sits exactly on that grid. A 1e-6 step rounds back to W0,
   so autocast rollouts would keep sampling from W0 for the first ~60 steps and the loop would not be on-policy. The
   cost is a slower forward and twice the cache memory, measured in stage S. Real post-training runs do use autocast;
   this is stated in the write-up as a departure from them.
2. **Own samples replace gold in arm F** (reorient plan §D2). Gold stays the off-policy contrast from E61; no new gold arm.
3. **Stage G is not rerun.** E40's stage G read the same 128 rows with the same decomposition: 4.24 nats, "ambiguous" by
   its rule, and every later D1 claim is already a ratio to it. A rerun would repeat a recorded number and would be the
   only reason to mount the decomposition. FYI to the user.
4. **δW goes to W&B, then `~/commute-data/results-large/qwen_rl/`, not the volume.** 21 states at 5.64 GB is 118 GB; the
   volume has about 54 GB free of its 150 (`experiments/SAVED.md`). Without G the pod needs no volume, so it can start
   wherever an H100 is free.
5. **A hand-written sampler that drops finished rows.** `model.generate` keeps every row in the batch to the longest, and
   about one sample in a hundred runs to the 512-token cap, so most of the batch computes padding: E45 took 130 s per
   344 samples on an L40. Dropping finished rows from the cache is expected to cut this by 2-3×, measured in stage S.
   Control: its output must match `generate`'s at step 1 of the rehearsal in distribution of length and reward (below).
6. **Forward and backward only over samples with non-zero weight.** Same gradient at lower cost (39% of problems are
   mixed, so about 60% of RL's samples have weight 0).

## The loop

Sources: §3 of the 2026-10-28 plan; Dr. GRPO form as filed there. Per step t:

- **Prompts.** 64 problems from a seeded permutation of GSM8K train minus the 128 held-out problems.
- **Rollouts.** 8 samples per problem, temperature 1, `top_k` and `top_p` off explicitly (the reset generation config
  otherwise falls back to HF's `top_k=50`), prompt plus response capped at 512 tokens. Full-precision forward, TF32 off
  (`qwen_vpd.fp32_matmuls`). `torch.manual_seed(f(seed, t))` per step, the same in every arm.
- **Reward.** 1 if the strict grader is right and the response ended inside the window, else 0.
- **Weights.** R: rᵢ − mean of its group. N: the same after shuffling the rewards within each group (seeded). F: 1 for
  the samples `qwen_updates.keep_correct` keeps from W0's rollouts at that step, 0 otherwise.
- **Loss.** Σᵢ wᵢ · (summed token log-loss of response i), divided by the constant 64 · 8 · 512 in every arm (F's divisor,
  left unstated in the old plan). Adam makes the constant nearly irrelevant, and the gradient norm is logged to check it
  stays far above Adam's eps.
- **Update.** One AdamW step per batch, lr 1e-6, betas (0.9, 0.999), no weight decay, no warm-up, no clipping, fused.
  All weights zero (every group all-right or all-wrong): no `opt.step()`, the step still counts.
- **Trainable.** The 196 matrices (`qwen_sft.trainable`); embedding, tied head and norm gains frozen.
- **F's data** is W0's rollouts for the same prompts and seed, sampled at the start of arm F and saved.

## Stages (one H100; each writes `results/<stage>.json`, prints every number, and resumes)

| stage | what | stop |
|---|---|---|
| A0 | Step 0: greedy accuracy on GSM8K test (1,319) and GSM1k (1,197), strict, drop rule and all-answers, full precision; held-out set H: 128 GSM8K train problems × 8 samples from W0, seed fixed, saved | P0 fails |
| S | Full-size slice: two RL steps, nothing saved. Reports sampler seconds per step, train seconds per step, peak GPU and host | projected R + N + F over $50 → N runs 50 steps; still over → stop and report |
| R | RL, 100 steps. δW and readouts at the seven steps; accuracy at 100 | early checks below |
| N | Null, 100 steps (or 50 by S). δW and readouts at the seven steps (those ≤ 50 if cut); accuracy at the last | early checks |
| F | Own-samples SFT, 100 steps. δW and readouts at the seven steps; accuracy at 100 | early checks |

Each δW is written in fp32 to `results/large/dW/<arm>/step_<t>.safetensors` and queued to W&B at once
(`qwen_sft.Uploader`). The rollouts of every step (token ids, rewards) go to `results/large/rollouts/<arm>/`; a per-step
summary (reward rate, mean length, window share, loss, gradient norm, |δW|, seconds) is committed in `results/<arm>.json`.

**Early checks in code; a miss stops the job.** Each value's source is named.

| when | check | must read | source |
|---|---|---|---|
| R step 1, N step 1, F's first batch | rollouts identical token for token across arms | exactly equal | same seed, same W0 |
| first live step | \|δW\| / (lr √numel) over all matrices | 0.85 to 1.05 (Adam's first step is about lr × sign of the gradient) | measured 0.926, rehearsal job 50077 (E45's 0.76 was under warm-up, so not comparable) |
| first live step | KL(W0 ‖ W_1), per token on H | 0.01 to 0.06, scaled by (first rate / lr)² | measured 0.028, sanity job 50310; the guessed 0.0015-0.02 would have stopped the run at step 1 |
| every step | loss and gradient norm finite | | |
| step 20, R and N | mean reward over steps 11-20 minus over steps 1-10 | above −0.05 | branch R3; expected near 0 |
| step 1 | reward rate | 0.60 to 0.88 | E45's T = 0.7 strict rate 0.83; T = 1 is expected lower |

The old P5 ("KL(1) < 1e-4") is withdrawn before filing: E45's own recorded step reads 0.0056, so that band was a guess
and would have failed on the first step.

## Numbers, and what each should read

| number | definition | W0 | N | R | blind to |
|---|---|---|---|---|---|
| ΔR(t) | over H's mixed problems: mean log-prob change (summed over response tokens) of W0's right samples minus that of its wrong ones, relative to step 0 | 0 exactly | ≈ 0, noise | > 0, growing | changes that leave right and wrong equally likely |
| KL(t) | mean per-token KL(π₀ ‖ π_t) over all H samples' response tokens | 0 exactly | small | small, growing | text outside H |
| accuracy gain | greedy strict accuracy, last step minus step 0, GSM8K test and GSM1k, drop rule, paired over problems kept at both | 0 | ≈ 0 | +0.03 to +0.08 | sampled behaviour |
| length | mean response tokens of a step's rollouts, share inside the window | 198, 0.95 | flat | within ±25% | content |
| \|δW\| | Frobenius norm over the 196 matrices | 0 | reported | reported | direction |

Positive control for ΔR: R at step 100, where the accuracy gain must also show. Noise floor for ΔR: N at the same step.

## Predictions (`experiments/qwen_rl/results/predictions.json`, filed before the run)

Thresholds use ≥ and < as written, after rounding to 9 places (`qwen_updates.r9`); a missing value is a miss.

- **P0 (step 0).** GSM8K drop-rule strict accuracy within ±0.015 of the baseline's 0.777, GSM1k within ±0.015 of 0.782;
  ΔR(0) and KL(0) ≤ 1e-6. Fail → stop. An empty readout pool: stop (ΔR has no meaning, and H is cheap to redraw).
- **P1 (RL learns).** RL's GSM8K gain at step 100: "learns" if ≥ 0.02 and ≥ 2 paired SE; "no gain" if < 0.01; ambiguous
  between. Expected: learns.
- **P2 (not memory).** d = RL's GSM1k gain minus its GSM8K gain. "Same" if |d| < 0.02 and |d| ≤ 2 SE of d; "GSM8K-only" if
  −d ≥ 0.03 and −d ≥ 2 SE; ambiguous otherwise. Expected: same.
- **P3 (null).** N's GSM8K gain: |gain| < 0.015. Expected: holds.
- **P4 (early signal).** R's ΔR(t) > 0 at every saved t ≥ 1, and R's ΔR(t) ≥ 3 × |N's ΔR(t)| at t = 10, 50, 100 (N's at
  its last step if cut). Expected: holds at 50 and 100; at 10, uncertain.
- **P6 (length).** R's mean rollout length over steps 91-100 within ±25% of steps 1-10's, and the share inside the window
  over steps 91-100 ≥ 0.90. Expected: holds.
- **P7 (SFT vs RL).** F's KL(100) ≥ 3 × R's KL(100), and F's GSM8K gain ≤ R's. Expected: uncertain on the first part;
  the old expectation assumed gold's terse style, and own samples share the model's style. Reported either way.

## Decision tree (approved with this plan)

- **P0 fails** → stop, fix the cause, rerun A0.
R runs alone first and P1 is read on the pod before N and F start (review finding 2): the review's extrapolation puts
|δW| at step 100 near 0.5, against E45's 1.87 for a +0.037 gain, so "no gain" at 1e-6 is a live outcome. N and F run at
the rate of the R that the tree keeps, so the null stays a null for it.

- **R1, P1 learns or ambiguous at 1e-6** → N and F at 1e-6, same pod. δWs go to `qwen-decomp-lead` (a note in its handoff
  folder and an FYI when each lands), ambiguous labelled weak. Then anchor 1 at E61's rate.
- **R2, P1 no gain at 1e-6** → the pod stays up; R again at 3e-6, its files under their own name, the 1e-6 δWs kept (a
  weak update is still an update to read). P1 learns or ambiguous → N and F at 3e-6. Still no gain → N and F at 3e-6
  anyway, since the δWs are D2's deliverable, and a card: the harder-benchmark search. Amendment 1 of two.
- **R3, reward falls by 0.05 by step 20 at 1e-6** → the pod stays up; R at 3e-7, own name, then N and F at 3e-7, read as in
  R1. Amendment 1 of two. FYI.
- **Worst case** R twice, N, F: about 12 h, about $48 (S's numbers replace this).
- **P3 fails** → FYI; the null's δW is still read; the reading of RL carries the gap as its noise floor.
- **P6 fails** → card: answers lengthen past the window; δWs kept and scored.
- **P2 GSM8K-only** → FYI, an analysis line; the plan goes on.

## Gates (record)

**1. Pre-mortem.**
- *Recipe:* 1e-6 from step 1 with no warm-up: Adam's first step moves every entry by lr. Guard: |δW| and KL early checks;
  branch R3. E40's huge first step was at 1e-5.
- *Data:* prompt format, thinking off, end token: reuse `qwen_eval.prompt`, `sft_seq` conventions; H kept out of training
  by index, asserted. A1 of E45's fix: duplicates dropped from F by `keep_correct`.
- *Precision:* fp32 master, fp32 forward, TF32 off; the identity check above catches a stray autocast only in part, so
  the code has no autocast context in the arm path at all. δW fp32.
- *Memory:* fp32 cache 228 kB per token on E45's 114 kB bf16. S measures it; the sampler's batch is set from the card's
  free memory, as E45's `per_batch`. Host RAM: reused `delta` works one matrix at a time.
- *Time and cost:* see below; S's rule guards it; arms run in order R, N, F, so a cut loses the least-needed last.
- *Saved:* δW fp32 (118 GB) on pod disk and W&B as each is written; pod disk asked at ≥ 250 GB; a cluster CPU job fetches
  W&B artifacts into `~/commute-data/results-large/qwen_rl/` and checks sha256 against the run's `SHA256SUMS` in `results/`.
- *Pod loss and resume:* a checkpoint (17 GB, `qwen_train.checkpoint`) at steps 20 and 50 on pod disk survives a kernel
  death, not a pod loss. A lost pod costs at most the arm in flight (~$10-15), and its δWs so far are on W&B. Accepted.
  The run's `results/<arm>.json` also survives through the runner's downloads after each saved step's `run.log` tail.
- *Controls:* same card for R, N, F; A0 on it too; baseline 0.777 was bf16 on an L40, hence P0's ±0.015.
- *Predictions:* edge cases above; P2's d = 0 and SE = 0 now resolves to "same".
- *Code never run on a GPU:* the sampler, the weighted step, the readouts. All run in the rehearsal.

**2. Settings sourced.** Batch, samples, temperature, cap, lr, betas, no KL, one update per batch: §3 of the 2026-10-28
plan, and scoping report `reports/analyses/2026-10-27_analysis_qwen-rl-gsm8k.md` §3 (Dr. GRPO). Departures above.

**3. Rehearsal for $0** on a cluster L40 (`debug`): 4 problems × 2 samples, 3 steps of each arm with H of 4 problems, 8-problem accuracy, saves, a resume from the checkpoint, an upload and fetch of one small file; then stage S at full size to measure time, memory and the sampler's agreement with `generate` (length and reward rate within 2 SE on one batch of 512). Numbers written back here.

**4. Early checks:** the table above.

**5. Review:** one fresh Opus agent, this plan and the code together, after the rehearsal's numbers are in.

**6. The user's go-ahead** on the cost line, by `fleet/ask`, with the rehearsal's numbers.

## Time and cost (estimate; S replaces it)

Sampler on an H100: ~60 s per 512 samples. Train: ~25 s per step for R and N, ~50 s for F. R and N about 2.4 h each,
F about 3 h with its sampling, accuracy passes and set-up 1 h: about 9 h at $3.99/h, about $36, under the gate's $60 and
inside the session's $100, over its 6 h. Rollouts drive the cost, so a slower sampler than estimated is what S watches.

## Not in this plan

Any map; any read of the decomposition; a second seed (11-08); a gold arm; anchor 2; thinking mode; a harder benchmark
unless R2 says so.

## Rehearsal record

Jobs 50067-50076 (tiny, all stages, exit codes as designed), 50077 (S full, L40). Rollouts identical across R, N, F. Fetch done.
The resume ran: R's and N's step 3 had all weights zero, F's was live (|δW| 0.0569 to 0.0681). The review ran four steps
through a resume on CPU, bit for bit equal to four straight.

S full on an L40: sampler 112 s per 512 samples against `generate`'s 223 s (2.0×, less than the 2-3× expected). Length
191.2 against 192.3 (se 2.3), reward 0.818 against 0.815 (se 0.017). Train 42 s and 31 s per step. Then out of memory in
the readout at 8192 tokens per batch, beside Adam's state: lowered to 2048.

S full again at 2048 (job 50208, L40, exit 0, 677 s), the numbers that count: per RL step 112.9 s sampling and 34.2 s
training; per F step 67.9 s training; 121.8 s per readout on H. Sampler 106 s against `generate`'s 209 s. Peak GPU 49.5 GB
(sampler 42.6 at the L40's row count; about 61 GB scaled to the H100's 512 rows, under 80), host 11 GB. KL after two
steps 0.032 on the rollouts' own samples. |δW| 0.0348, 0.0549; gradient rms 2.9e-6, 2.5e-6; reward 0.830, 0.832.

Upload and `verify` (jobs 50234-50266, tiny, cap raised): two 5.64 GB δW went up in about 4 min each, no W&B error,
`verify` read 8 rows, all ok; `next` printed the right branch. 21 δW is about 84 min of background upload.

**Cost line for the go-ahead.** About 15 h on the L40 by the numbers above. The H100's speed against the L40 is not
measured: at 1.5-2× the run is 8-10 h, $32-40 at $3.99/h. Worst case by the tree (R twice, then N and F), about 11-13 h,
$44-55. S on the pod measures the speed first; over $50 projected, N runs 50 steps.

**Band changed before filing took effect.** First live step |δW| 0.03477, ratio 0.926 to lr √numel (numel 1,409,286,144),
outside the guessed 0.95-1.05. Adam moves an entry by lr · |g| / (|g| + 1e-8), and the gradient (rms 3e-6) is
heavy-tailed, so some entries sit near eps. Band now 0.85-1.05, from this reading; no scored run had started. So the
divisor is not quite irrelevant: an arm with a smaller gradient moves fewer entries. Reported per arm as gradient rms.
