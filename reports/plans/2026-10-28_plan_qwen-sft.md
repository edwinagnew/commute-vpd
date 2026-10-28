# Plan: E40, the first Qwen3-1.7B update: SFT on GSM8K, the existing decomposition, and teacher training

*2026-10-28, qwen-decomp-lead. Replaces the RL plan `reports/plans/2026-10-28_plan_qwen-rl.md` (on hold; RL becomes a
later experiment). Words as in `GOALS.md` → Terms.*

## The request, restated (user, 2026-10-28, option 2)

"Test fine-tuning and the existing decomposition first", then teacher training in the same run, and "keep in mind what
future experiments we might want to run based on it and try and save the necessary results then and there".

In the Terms: (1) the control: the unchanged decomposition read on the model's own GSM8K answers against web text;
(2) a full-weight SFT fine-tune of Qwen3-1.7B on GSM8K's human-written solutions, δW of the 196 decomposed matrices
saved at small step counts; (3) the do-nothing state (δW into Δ, subcomponents and Γ untouched) scored by the paper's
adversary at those steps; (4) teacher training (the decomposition's alive subcomponents trained on KL(fine-tuned model ‖
student), Γ and Δ frozen) at the same steps, scored the same way; (5) everything a later experiment on this fine-tune
would need, saved to the volume as it is made (section 5).

## 1. Design check (`/design`)

- **Goal served:** the weak version on the larger model: is the decomposition after teacher training faithful to
  W₀ + δW for a real whole-response update, where Δ does not absorb δW? The do-nothing state is the reference that
  shows whether the pass test can see this update at all.
- **Nearest tried:** E34 (teacher training passes for narrow trigger fine-tunes on the 67M) and E37/E39 (it fails the
  check for a broad lowercase fine-tune; a rank-8 LoRA passes). New: the first update on Qwen, the first whole-response
  update, and the first reading of the Qwen decomposition on task text.
- **Patch or step:** a step: the first Qwen experiment.
- **Decisions relied on:** GSM8K train, scored on GSM8K test and GSM1k, answers past 512 tokens dropped (2026-10-27,
  2026-10-28); the adversary at one start, 20 steps of 0.1, as the pass test, a ratio to the unchanged decomposition's
  reading on the same 128 FineWeb-Edu rows (2026-10-27); teacher training as the map (2026-10-26).
- **Departures:** none. Low-level choices fixed here: fp32 master weights, embedding, tied head and norm gains frozen
  (δW only in the 196 matrices), Adam, no warm-up.
- **What would change what we do next:** G fails → the decomposition does not explain maths answers, card. Do-nothing
  passes at step 100 → the pass test is blind to this update; task rows or a behaviour test must become the pass test
  (card). Teacher training passes with the behaviour kept → the first working map on Qwen; next, RL's update (held
  plan). Teacher training fails → the E37/E39 question (why broad updates fail) moves to Qwen, with this run's saved
  states.
- **Control first:** stage G, then T0 (teacher training with δW = 0 must leave the adversary where it was), both before
  any fine-tune step, each with a stop rule.
- **Class:** confirmed by the user after the restatement (2026-10-28, "i think 2").

## 2. Stages (one RunPod H100 80 GB in US-MO-1, volume ydavm7ct4x mounted; ~6 h, ~$18-25)

Each stage writes its results JSON and prints every number when it finishes; the runner downloads each before the
next ends. Large files go to `/workspace/e40/` (section 5), never downloaded.

| stage | what | stop rule |
|---|---|---|
| G | Unchanged decomposition (fp32, Γ in bf16, as in the adversary run). Rows: greedy answers of the unchanged model to GSM8K test prompts in index order, chat-formatted as generated (template, thinking off, response, end token), packed end to end into 128 rows of 512 tokens ("maths rows"). On maths rows and on the adversary run's 128 FineWeb-Edu rows: importance-masked KL and the adversary (seed 0). Alive set: subcomponents with g > 0.01 at any position of either block | G-fail → stop, card |
| T0 | Teacher training with δW = 0 (teacher = the unchanged model), full recipe below; then the adversary on FineWeb rows | ratio to G's FineWeb reading outside [0.85, 1.15] → stop |
| A0 | Step-0 readings: greedy accuracy on GSM8K test (1,319) and GSM1k (1,197), strict grader, drop rule and all answers; held-out gold log-loss; response lengths | P0 fails → stop |
| F | SFT, 100 steps. Batch: 64 GSM8K train problems from a seeded permutation (minus 256 held out), sequence = prompt (the baseline's chat template and instruction, thinking off) + `gold_as_response` + end token, cut at 512. Loss: mean token log-loss over response tokens. AdamW lr 1e-5, betas (0.9, 0.999), no weight decay, constant rate, no clipping; bf16 autocast forward, fp32 master. Readings at the saved steps (section 3); accuracy and lengths at step 100 | — |
| D | Do-nothing at steps 1, 10, 100: adversary on FineWeb rows; at step 100 also on maths rows | — |
| T | Teacher training at steps 1, 10, 100: adversary on FineWeb rows and behaviour kept; at step 100 also the adversary on maths rows and greedy GSM8K accuracy of the student's weights | — |

**Teacher training recipe (T0 and T).** Student: alive columns of U, V with zero-initialised offsets (dU, dV), weights
(U + dU)(V + dV)ᵀ + Δ, Γ and Δ frozen, no masks in the loss (as E34). Loss: mean over every position of
KL(teacher ‖ student) with `teacher.kl_exact_zero`'s gradient. Rows: each batch 4 packed rows of F's training sequences
and 4 FineWeb-Edu rows (not the 128 evaluation rows), 512 tokens. Adam lr 1e-5 (E34's Pile arm), 1,000 steps, fp32
offsets. If the alive offsets with their Adam state exceed ~45 GB, the benchmark check below stops the job and reports the count.

**Benchmark check (inside G, before T0).** Time one adversary step, one teacher-training step and one evaluation
batch; project stages T0-T. Projected spend over $25 → drop step 10 from D and T; still over → stop and report.

**Three starts near the line.** A FineWeb ratio between 1.2 and 1.8 is re-read with seeds 1 and 2 and decided on the
mean of three; outside that band one start decides.

## 3. Numbers, and what each should read

| number | definition | W₀ | expected | blind to |
|---|---|---|---|---|
| G: maths adversary | adversary KL on maths rows | — | 1-3 nats (FineWeb 1.62) | damage a 20-step search misses (lower bound) |
| G: maths masked KL | importance-masked KL on maths rows | — | 0.07-0.28 (FineWeb 0.142) | parts the importance values keep on |
| ratio | a state's FineWeb adversary reading / G's FineWeb reading, same rows and seed; pass ≤ 1.5 | 1 | see P | damage only on other text |
| gold loss drop | held-out (256 problems) mean token log-loss of the gold solution, step 0 minus step t | 0 | > 0, growing | everything except the gold text's likelihood |
| KL to start | mean per-token KL(π₀ ‖ π_t) on maths rows, fp32 | 0 | small, growing | text outside maths rows |
| behaviour kept | 1 − KL(teacher ‖ student) / KL(teacher ‖ W₀), maths rows (test prompts, so not trained on), fp32 | 0 at W₀, 1 for the teacher | ≥ 0.8 | what the unmasked weights hide under masks (the adversary reads that) |
| accuracy | greedy strict, drop rule, GSM8K test and GSM1k; reported with all-answers | 0.777, 0.782 | step 100 within ±0.05; reported | sampled behaviour |
| length | mean greedy response tokens on GSM8K test | 198 | shorter (gold solutions are terser) | content |

## 4. Predictions (filed in `experiments/qwen_sft/results/predictions.json` before the run)

- **G.** Pass if maths adversary ≤ 2 × the FineWeb reading and maths masked KL ≤ 2 × the FineWeb masked KL; fail if
  either exceeds 3×; ambiguous otherwise (the run continues; a card asks how to read the maths-row results). Expected: pass.
- **T0.** Ratio in [0.85, 1.15]. Expected: holds (the gradient is exactly zero at δW = 0).
- **P0.** Step-0 GSM8K drop-rule accuracy within ±0.015 of 0.777 and GSM1k within ±0.015 of 0.782; gold loss drop and
  KL to start ≤ 1e-6. Fail → stop.
- **P1 (the fine-tune does something).** At step 100, gold loss drop ≥ 0.05 nats per token and KL to start ≥ 1e-3.
  Expected: holds. Mean greedy length at step 100 ≤ 0.8 × step 0's. Expected: holds.
- **P2 (do-nothing).** Ratio at step 1 ≤ 1.2. At step 100, ratio > 1.5 is "the test sees the update"; ≤ 1.5 is "blind
  on web text" (decision tree). Expected: step 1 holds; step 100 uncertain, leaning "blind on web text" since
  an SFT on maths may move web-text predictions little; the maths-row reading beside it is then the informative one.
- **P3 (teacher training keeps the behaviour).** Behaviour kept ≥ 0.8 at steps 10 and 100. Expected: holds (E34, E37
  kept behaviour). Step 1's KL(teacher ‖ W₀) may sit near float noise: if it is below 1e-6, step 1's kept is unscored.
- **P4 (teacher training passes).** Ratio ≤ 1.5 at steps 1, 10, 100. Expected: holds at 1 and 10; at 100 uncertain
  (E37's broad update failed at 2.20).
- **P5 (teacher training beats do-nothing).** At step 100, teacher training's maths-row adversary reading < do-nothing's.
  Expected: holds.

Edge cases the code writer runs each rule against before filing: a zero denominator in behaviour kept (step 1), ties at
every threshold (≥, ≤ as written), a ratio band read with three starts where the three straddle the line (the mean
decides), the drop rule leaving different problem sets at step 0 and 100 (pair on the problems kept at both).

## 5. Saved for later experiments (to `/workspace/e40/`, bf16 unless stated)

Chosen for what a later experiment would otherwise need the fine-tune rerun for.

| item | size | for |
|---|---|---|
| δW at steps 1, 2, 3, 5, 10, 20, 50, 100 (fp32 difference stored as bf16) | 8 × 2.8 GB | other maps (splits, step split), linearity in step count, LoRA comparisons |
| Gradient at W₀ on the first batch (fp32 → bf16) | 2.8 GB | one-step maps (adjoint, component training's first step), Adam's step 1 ≈ −lr·sign(G) |
| Adam moments and step count at step 100 (fp32) | 11.3 GB | continuing the fine-tune past step 100 without rerunning it |
| Teacher-training offsets (dU, dV) at T0, steps 1, 10, 100, with the alive index | ~4 × 3 GB (alive-dependent) | re-reading the decomposition after teacher training (E39 lost such a state with its pod) |
| Per-subcomponent mean g and active share (g > 0.01) on maths and FineWeb rows: W₀, do-nothing at 100, teacher training at 100 | ~3 MB each, committed | re-gating on Qwen: what Γ switches on after the update |
| The adversary's final sources per state | ~3 MB each, committed | where the damage sits |
| Per step: loss, ‖δW‖ per matrix, batch indices; held-out readings | small, committed | step-size analyses |
| Step-0 and step-100 greedy responses on GSM8K test and GSM1k | small, committed | answer-level comparisons |

Space: about 48 GB for items 1-4 against ~46 GB free on the 100 GB volume. If the volume is not enlarged, the runner
drops, in order: Adam moments, δW step 3, δW step 20.

## 6. Decision tree (each branch approved with this plan)

- **G fails** → stop; card. **T0 or P0 fails** → stop; fix, rerun the stage (class 2).
- **P2 blind on web text** → card: make the maths rows part of the pass test (option recommended) or add a behaviour
  test; the run's maths-row readings already answer it for step 100.
- **P4 passes, P3 holds** → FYI, analysis; next plan: RL's update through the same table (held plan, rewritten).
- **P4 fails at 100** → analysis; next plan through `/design`: E39's tests on this saved state (rank-limited update,
  revert by layer and by importance).

## 7. Budget and code

~6 h of H100 at ~$3/h, $18-25, inside the session's $40 with the 15% margin. Code: `experiments/qwen_sft/run.py`,
shared pieces in `src/commute/qwen_sft.py` (SFT step, δW and state saving) and `src/commute/qwen_teacher.py` (teacher
training on Qwen), reusing `qwen_vpd`, `qwen_eval`, `gsm8k`, `teacher.kl_exact_zero`. One review by a fresh Opus agent
before the paid run; runner via `/pod-runner`.

## Not in this plan

RL (held), a shuffled-label null, SGD, a LoRA arm, a second seed, other maps (splits) on these δWs (their inputs are
saved), thinking mode.
