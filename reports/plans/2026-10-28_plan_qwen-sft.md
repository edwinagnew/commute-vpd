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

Space: about 48 GB for items 1-4; the user enlarged the volume to 150 GB (2026-10-28), so ~96 GB is free. If space still runs short, the runner
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

## Amendment 1 (2026-10-28): the corrected learning-rate recipe, teacher training on an H200

**Why.** E40's fine-tune used Adam at a constant 1e-5 with no warm-up. Adam's first step moves every weight by the
full rate, so step 1 alone gave KL to start 0.106 nats per token and 1.57 of the 1.98 nats of gold-loss drop reached
by step 100 (`results/F.json`). Real recipes warm up. The user: "obviously you should run with the correct learning
rate always". Separately, 494,598 subcomponents are alive (73 GB for teacher training), so the user chose option (a)
of card decision-qwen-tt-memory: a larger card, method unchanged. The user also raised this session's pod budget
from $40 to $50 to keep step 10 (chat, 2026-10-28).

**Recipe.** AdamW, peak 1e-5, linear warm-up from 0 over the first 10 steps (rate 1e-5 · t/10 at step t ≤ 10), then
cosine decay to 0 at step 100 (1e-5 · ½(1 + cos(π(t − 10)/90))). This is LLaMA-Factory's documented full-SFT setting
for Qwen models (lr 1e-5, cosine, warm-up ratio 0.1). Everything else as in §2.

**Card.** An H200 141 GB, B200 if none; US-MO-1 with the volume preferred. Elsewhere, fetch the decomposition from
W&B (`qwen_files.fetch`, ~20 min) and upload section 5's saved items to W&B as each is written, because a container
disk dies with the pod. The alive cap rises to 110 GB.

**Stages, in order.**

| stage | what | stop rule |
|---|---|---|
| G1 | The control on this card. FineWeb masked KL and adversary seed 0, maths adversary seed 0, on E40's `maths_rows.json` (not regenerated). Alive set reused from `/workspace/e40_nowarmup/alive.safetensors` (it does not depend on the card) | G_control as filed (0.14167 within 2%, 1.7016 within 15%) |
| L | Step-1 size check. From W₀, one AdamW step on F's first batch at each of 1e-7, 1e-6 and 1e-5, then KL to start on maths rows; seconds per rate | the 1e-5 reading off E40's 0.106 by more than 15% → stop |
| A0 | As in §2, on this card | P0 |
| F1 | SFT with the recipe above, 100 steps, the §5 items saved to `/workspace/e40a1` | — |
| T0 | Teacher training at δW = 0, 20 steps (the gradient is exactly zero there). If the offsets' largest entry is exactly 0, the state is W₀'s and G1's reading stands (ratio 1); otherwise read the adversary | T0 band |
| D1 | Do-nothing at steps 1, 10, 100, as D | — |
| T1 | Teacher training at steps 1, 10, 100, as T (1,000 steps) | — |

**Predictions** (`results/predictions_amend1.json`, filed before the run; E40's filed rules stand unedited for E40):
- **L (control).** KL at 1e-5 within ±15% of 0.106. KL at 1e-6 over KL at 1e-5 between 0.003 and 0.05 (first-order
  scaling gives about 0.01).
- **F1 is small early.** F1's KL to start at step 1 ≤ 0.005 nats per token, and its gold-loss drop at step 1 ≤ 0.3 of
  its drop at step 100.
- **P1-P5** as in §4, read on F1, D1 and T1.

**Budget.** About 7 h on an H200, $28-32 at ~$4.5/h, against ~$31 usable under the raised cap. The benchmark's
projection rule uses the remaining budget, not $25; the re-read guard stops extra starts at the cap less 15%.

### Amendment 1: Gates (`.claude/CLAUDE.md` → Gated runs; first gated run)

A fresh Opus agent reviewed this amendment against the gates (2026-10-28). Its findings are accepted as below; they
supersede the amendment's table above where they differ.

**Changes from the review.** Results go to their own directory (`results/amend1/`, and a stage whose JSON exists
refuses to run): left as it was, the first run's D step 1 and its H100 seed-1 reference would have been read as this
run's. The saves were twice the §5 estimate (offsets are 14.6 GB each in fp32 at 3.65 B alive parameters), so T0's
offsets and the Adam moments are not saved, offsets at the first two read steps are bf16, and a disk pre-flight runs in
minute 1. T0 at δW = 0 could not fail (zero gradient), so it becomes a positive control with a reachable target
(random offsets of matched size). Teacher training's rate is measured on Qwen in the rehearsal (1e-6 against 1e-5),
not carried from the 67M. The SFT schedule is `transformers.get_cosine_schedule_with_warmup` itself, with HF's step
semantics (the first optimiser step runs at rate 0) and HF's default clipping at 1.0. Read steps for D1 and T1 are 10,
100, and the first saved step with KL to start ≥ 0.01: under warm-up step 1's KL is about 0.001 and do-nothing already
read 1.05 at a hundred times that, so a step-1 read could not fail. The benchmark is the first paid stage.

**Pre-mortem.**

| heading | could go wrong | guard |
|---|---|---|
| Recipe | teacher-training rate unmeasured on Qwen; SFT schedule ends at rate 0 | rate check in the rehearsal; the library's schedule |
| Data | separate tokenisation of prompt and response; stale results read as new | ids compared with whole-string tokenisation in the rehearsal; own directory, refuse if exists |
| Precision | bf16 adds at q, k, v, gate, up round away a small change | first-batch fp32 KL against the training forward's, tolerance from the rehearsal's measured gap, stop |
| Memory | teacher training ~90 GB GPU, ~80 GB host, unmeasured | benchmark first: stop on peak > 125 GB or host < 110 GB |
| Time and cost | teacher-training step 1-3 s, unmeasured (±$7) | measured in the benchmark; projection against the remaining budget; smallest read step dropped first |
| Saves | ~100 GB against 50-58 GB free | reduced save list, disk pre-flight |
| Downloads and uploads | off the volume, the container disk dies with the pod; W&B upload never run | upload rehearsed from the cluster; 1 GB timed in the benchmark; volume region preferred |
| Pod loss and resume | F1 not resumable | δW uploaded or on the volume as written; T resumes only against a δW file with a matching hash |
| Controls and references | cross-card reference; T0 degenerate | same-card references only; reachable-target control |
| Predictions | degenerate step-1 rule; P5 one start | read steps from the KL curve; P5 with a margin and three starts near it; NaN is a miss |
| Code never run on a GPU | teacher training, benchmark, resume, the schedule, uploads | all in the L40 rehearsal; full size in the benchmark |

**Settings and sources.** SFT: AdamW (0.9, 0.999), eps 1e-8, weight decay 0, peak 1e-5, cosine with 10% warm-up,
clipping 1.0 (HF Trainer defaults and LLaMA-Factory's full-SFT setting); batch 64 and 100 steps are project choices
(the project studies early steps; one epoch is 117 steps); response-only token-mean loss, bf16 autocast with fp32
master (HF); only the 196 matrices trained (§1). Teacher training: Adam, batch 8 × 512, 1,000 steps, plain mean KL
(E34 on the 67M); its rate is measured in the rehearsal; the alive threshold g > 0.01 follows the port analysis's flat
count between 0.001 and 0.01 (`2026-10-27_analysis_qwen-port.md`). Adversary: the paper's setting (decision
2026-10-27); three-starts band from measured noise (`2026-10-28_analysis_qwen-adversary-128.md`). Evaluation: the
baseline's.

**Open, waiting on the user.** The reading of maths-row adversary numbers (card decision-qwen-maths-adversary): until
it is decided they are reported, not scored, with W₀'s maths reading at three seeds on the same card as reference.

**Still to do before the go-ahead:** the code (writer), the $0 rehearsal on a cluster L40 with its numbers written
here, the code review, then the user's go-ahead with the per-stage cost.

**Gate 5, code review (2026-10-28).** A fresh Opus agent reviewed the Amendment 1 diff. Its critical finding was
already live in the rehearsal (cluster job 40530): teacher training's backward pass ran outside the change hooks
under gradient checkpointing, so every teacher-training stage would have crashed at its first backward on the
paid card. Fixes under way, with the others the review ranked: the disk pre-flight on the real alive count (and
the decomposition fetch onto the new volume 2bun37yawb, US-NC-1), read steps only among files that exist, a
full-size teacher-training rate check in B (positive control, 25 steps at 1e-6 and 1e-5, a filed rule picks the
rate), a mount-point check, host-memory release between read steps, T1's end check report-only, atomic saves and a
stricter resume check, and the maths pass test (ratio to the same-card W₀ maths reading, 1.5 line, user-accepted).
"F1 is small early" is retired as a prediction: under the library's schedule step 1 runs at rate 0, so it is
checked in code at step 2 (KL ≤ 0.005). The rehearsal reruns after the fixes.

**Gate 3, rehearsal record (2026-10-28).** Cluster L40s, tiny sizes (16 alive columns per site, 128-token rows), F1
and L full size; files in `experiments/qwen_sft/results/amend1_rehearse_1/`. Job 40602 (B, G1, L, A0; 330 s): 328,041
alive subcomponents, 2.39 billion offset entries, so 47.9 GB of offsets, gradients, Adam and copies; disk need 52.2 GB
at that count; peaks 39.0 GB GPU, 38.0 GB host; W&B uplink 2.0 MB/s. Job 40609 (F1): 932 s, 36.7 GB GPU, 10.8 GB host.
Jobs 40637, 40647 (T0, D1, T1 killed after its first save and resumed, TR): all ran end to end, peaks 41.8-44.7 GB,
0.85-0.91 s per teacher step. Job 40668: the stale-results guard refused at once. Two checks missed: the first-batch
training KL sat 18-19% above its fp32 value (B, T0) and 44% below (T3), and B's curves did not fall. Job 40673
(`check_teacher.py`, `results/check_teacher.json`) found why. Without autocast the training forward matches fp32 to
2e-6, and the bf16 gap shrinks to 1e-3 at a 16x teacher, so bf16 rounding caused it; TF32 is off by 3.4e-5 and
2.4e-4; 1.04, 1.06, 0.79 s per step for bf16, TF32, fp32 (tiny sizes; no guide to the H200). On 8 fixed rows, 1e-4
diverges (x3.3 by step 25); 1e-5 falls to 0.875, 0.702, 0.590 of the start at steps 25, 100, 300. Gradients flow;
25 steps read on one noisy batch could not show it. Changes, committed with this record:
1. Teacher training runs fp32 with TF32 matmuls, checkpointing kept, no bf16 autocast; readings stay true fp32 (TF32
   on swapped weights read T3's KL as 0.0030 for 0.0317). 2. The first-batch check stays at 2e-3 (8x TF32's 2.4e-4).
3. B tries 1e-5 and 3e-5 (1e-4 diverged), 100 steps each, fp32 KL on 4 maths and 4 FineWeb evaluation rows at steps 0,
   50, 100 (the target sized on the same rows). A rate qualifies if the readings fall strictly and step 100 is at most
   0.9 of step 0; the larger qualifying rate is used, none stops. Zero or NaN starts, ties, both, neither: in the tests.
4. T0 and the step-100 check read that same fixed-row fall. T0 passes at 0.5 or below, an estimate: 300 steps reached
   0.59 on 16 columns, and the full alive set's smaller α should go further; B measures it. T0's 300 steps and end
   fraction 0.7 came from TR's bf16 batch curves, so had no source: T0's steps now come from B's curve (log-linear to
   0.5, at most 300) and the step-100 check uses B's 0.9. TR is deleted. At rehearsal size that extrapolation says 196
   steps while 300 read 0.59, so the curve bends: if B's does too, T0 may stop.
5. Only results JSONs, the log and T1's last offsets go to W&B; δW and other files stay on the volume, so upload time
   leaves the cost estimate (at 2 MB/s it had left no read steps). 6. rehearsal.json lists every miss in checks.json.
Peak GPU memory in teacher training, from the code: ~72 GB at 2.39e9 entries, ~97 at E40's 3.65e9 (limit 125). Not
measured until the paid B: full-size TF32 step time, peak memory at the full alive count, B's rate and T0's steps.

**Gate 5, second review (2026-10-28).** A fresh Opus agent reviewed commit 51afa50 while rehearsal job 40691 ran B on
it. 1. The first-batch check had a 1.8x margin, not 8x: job 40691 read 1.13e-3 relative on B's first batch (both
rates), so a slightly larger gap on full rows would stop B on the paid card. Now 1e-2 (9x the measured gap), dated in
`predictions_amend1.json`. 2. T0 would likely miss and stop after F1 was paid for: the log-linear step count ignores
the bending curve (196 steps extrapolated where check_teacher read 0.59 at 300), and T0 was sized at F1's first-read KL
(0.022 in the rehearsal) while B's curve was measured at 0.01. Now T0 is sized at B's target KL, reads the fixed rows
every 50 steps, ends at the first reading at or under 0.5 of the start, and stops the job if none by step 300; it runs
right after G1 (its FineWeb ratio needs G1's W₀ reading), before A0 and F1. `t0_steps` is deleted, and the rehearsal
no longer excuses a T0 miss. 3. T1's last offsets (9.6-14.6 GB) were still uploaded, 80-120 min of billed idle at the
measured 2 MB/s; now only results JSONs and the log go to W&B. Deferred: minor 4, computing each site's change once per
step instead of twice per micro-batch; its cost is a T1 step about 2x what it could be (3 x 1000 steps), and B's
projection guards the budget either way.
Minor 7: `check_teacher.py` stays frozen as it ran (job 40673), so pyright still flags its `autocast=` calls.

**Gate 5, third review (2026-10-28).** A fresh agent reviewed 0dc2c0a and found no crash or wrong-number bug.
(a) T0's rule (0.5 of the start by step 300) has no full-size measurement; the rehearsal curves bend (0.59 at 300 steps
in check_teacher; 0.72 at step 100 in rehearsal B), and a miss stops the paid job after B, G1 and T0 (~1 h). Guard: the
paid run runs stage B alone first; if B's chosen-rate fixed-row reading at step 100 is above 0.65 of its start, the
runner stops the pod (B's results downloaded) and the lead asks the user before T0. (b) B is now required for every
later stage (fixed in this commit). (c) T0's first 100 steps repeat B's chosen-rate control (~100 teacher steps paid
twice); accepted and counted in the cost. (d) δW and the offsets live only on the network volume 2bun37yawb, not W&B;
the analysis reads the JSONs; the runner brief must pass --volume /workspace.
