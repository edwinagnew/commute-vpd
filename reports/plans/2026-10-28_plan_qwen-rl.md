# Plan: E40, the first Qwen3-1.7B updates: reinforcement learning on GSM8K, with a null and an SFT arm

*2026-10-28, qwen-decomp-lead. The user's task (2026-10-28): "I still don't have a good sense on whether doing RL when
it's already at 75% accuracy is worth doing. If it is, then just go ahead. If not, we need to look for a more difficult
benchmark." Section 1 answers the question; the rest is the run. Words as in `GOALS.md` → Terms.*

## The request, restated

Train the unchanged Qwen3-1.7B (thinking off) by reinforcement learning on GSM8K's training problems, saving the
update δW of the 196 decomposed matrices at small step counts, so that the next plan can apply a map (teacher training,
`decisions.md` 2026-10-26) to those δWs and score the decomposition after it with the paper's adversary. Beside it, the
same loop with rewards shuffled (the null) and with SFT on GSM8K's human-written solutions (the user's SFT-first
preference, 2026-10-27, in the same table). First, as the control, check that the unchanged decomposition explains the
model's own GSM8K answers as well as it explains web text.

## 1. Is RL from 0.75 worth doing? Yes, for this project

1. **The learning signal is there.** RL learns only from problems where the model's own samples disagree; a problem it
   always or never solves gives no gradient. Our baseline: 39% of problems had mixed right and wrong samples. The
   signal per problem peaks at 50% accuracy and is about three quarters of that peak at 75%, so a harder benchmark
   buys at most a little more signal per problem.
2. **We study small steps, not the end score.** The update we want to commute is one to fifty steps. What it needs is
   a behaviour change we can measure, not 25 points of headroom. A continuous readout (the change in the model's
   log-probability of its own right answers against its wrong ones) sees a single step; accuracy is read only at the
   end, to confirm the update did something. Published runs at this size and start gain about 5 to 10 points.
3. **It is the update the goal asks about.** `GOALS.md` asks whether failure to commute comes from post-training
   teaching new capability or from brittle decompositions. RL from 75% is the textbook strengthening update (it raises
   single-sample accuracy, not what the model can solve in many tries: our model solves 91% in 8 samples). If even this
   cannot be commuted, the brittleness reading gains; if it can, we have a working positive case.
4. **A harder benchmark costs more than it buys.** Competition maths needs answers far past the decomposition's
   512-token window (it was fit on 512-token rows), and Qwen has documented memory of MATH-500. A harder benchmark
   would also not change the update's kind: RL from a low start still mostly sharpens. A "teaches something new"
   contrast needs a task the model cannot do at all, which is a separate, later arm.

The case against, kept as a branch: if 100 steps raise accuracy by under one point, the update is too weak to read and
the harder-benchmark search starts (decision tree, branch R2).

## 2. Design check (`/design`)

- **Goal served:** the weak version on the larger model: produce real post-training updates δW whose decomposition
  can then be tested. The control (G) is the first check that the Qwen decomposition explains the task's text at all.
- **Nearest tried:** on the 67M, every update was a narrow Adam or SGD fine-tune to a trigger token (E1-E39); E34/E37
  found teacher training passes for trigger tasks and fails the check for a broad update. The Qwen apparatus rows
  (baseline, fresh numbers, fresh set, adversary at 128 rows) settled the task, the score and the pass test on web text.
  New here: the first whole-response, RL-made update, and the decomposition's reading on maths text.
- **Patch or step:** a step: the first Qwen experiment.
- **Decisions relied on:** non-thinking GSM8K with the baseline's prompt and strict grader, answers past 512 tokens
  dropped from the held-out score (2026-10-27); the adversary with one start, 20 steps of 0.1, as the pass test
  (2026-10-27); teacher training as the map (2026-10-26, next plan). This plan closes card decision-qwen-baseline with
  option (a), on the user's message: keep GSM8K train, score on GSM1k beside GSM8K test.
- **Departures:** none. Low-level choices made here (scoping report `2026-10-27_analysis_qwen-rl-gsm8k.md` §3, its
  recommended options): a hand-written loop in the existing environment, fp32 master weights, embedding and norm gains
  frozen so δW lies only in the decomposed matrices, one update per rollout batch (no clipping, no spurious-reward
  path), no KL penalty, Adam. Rejection-sampling SFT and a LoRA arm go to the backlog; SGD is a later arm (`GOALS.md`
  optimiser subgoal).
- **What would change what we do next:** G fails → the decomposition does not explain the task text and no δW can be
  read through it (card). RL gains < 0.01 → the harder-benchmark search. Answers lengthen past the window → the
  thinking-leak card. Otherwise the next plan is teacher training on these δWs.
- **Control first:** stage G, alone, with a stop rule; then each training arm's step 0 reproduces the baseline before
  any step is taken.
- **Class:** the user's explicit go-ahead in the task file (2026-10-28), conditional on section 1's answer.

## 3. Stages (one RunPod H100 80 GB in US-MO-1 with volume ydavm7ct4x mounted; ~5.5 h, ~$17-20)

Every stage writes its results file and prints every number when it finishes; the runner downloads each before the
next stage ends. δW checkpoints go to `/workspace/qwen_rl/<arm>/step_<t>.safetensors` on the volume (not downloaded).

| stage | what | stop rule |
|---|---|---|
| G | The control. Unchanged decomposition (fp32, Γ in bf16, as in the adversary run). Greedy answers of the unchanged model to GSM8K test prompts in index order, chat-formatted as generated (template, thinking off, response, end token), packed end to end into 128 rows of 512 tokens. Importance-masked KL and the adversary (seed 0, 20 steps of 0.1) on those rows | G-fail (below) → stop the job and card |
| S | Benchmark slice: the RL loop at full batch for 2 steps, nothing saved. Reports step time and peak memory | projected cost of R, N, F over $22 → N runs 50 steps; still over → stop and report |
| A0 | Step 0, shared by every arm: greedy accuracy on GSM8K test (1,319) and GSM1k (1,197), strict grader, drop rule and all-answers; the held-out readout set (below) generated and scored | P0 fails → stop |
| R | RL, 100 steps. δW saved at steps 1, 2, 5, 10, 20, 50, 100; readouts at the same steps; accuracy at step 100 | — |
| N | Null: the same loop, seed and prompt order, rewards shuffled within each group. 100 steps (50 if S says so); δW and readouts at 1, 10, 50, 100; accuracy at the last step | — |
| F | SFT: the same prompts per step, loss = summed token log-loss of each problem's human solution (`gold_as_response`), same optimiser and rate. 100 steps; δW and readouts at 1, 10, 50, 100; accuracy at 100 | — |

**Loop.** 64 problems per step from a seeded permutation of GSM8K train (minus the 128 held-out problems), 8 samples
each at temperature 1, no top-p or top-k, prompt plus response capped at 512 tokens. Reward 1 if the strict grader is
right and the response ended inside the window, else 0. Loss: Σᵢ (rᵢ − mean of its group) · log π(oᵢ), summed over
tokens, divided by the constant 64 · 8 · 512 (Dr. GRPO form). One update per batch, AdamW lr 1e-6, betas (0.9, 0.999),
no weight decay, no warm-up, no gradient clipping. Trainable: the 196 q, k, v, o, gate, up, down matrices; embedding,
tied output head and norm gains frozen. fp32 master weights; rollouts and the training forward under bf16 autocast,
as real post-training runs. Every readout and every saved δW in fp32, TF32 off. Rollouts saved per step as token ids
and rewards (small, committed only as a summary).

**Held-out readout set H.** 128 GSM8K train problems kept out of training, 8 samples each from the unchanged model at
temperature 1 (seed fixed), generated once in A0. The problems with mixed right and wrong samples are the readout pool.

## 4. Numbers, and what each should read

| number | definition | W₀ (step 0) | null N | RL R | blind to |
|---|---|---|---|---|---|
| G: masked KL on maths rows | importance-masked KL(clean ‖ masked), mean over tokens, 128 GSM8K rows | ≈ web text's 0.142 (0.07-0.28) | — | — | parts the importance values keep on |
| G: adversary on maths rows | the pass test's reading on the same rows | 1-3 nats (web text 1.62) | — | — | damage a 20-step search does not find (lower bound) |
| ΔR(t) | over pool problems, mean log-prob change (sum over response tokens, fp32) of the unchanged model's right samples minus that of its wrong ones, relative to step 0 | 0 exactly | ≈ 0, noise | > 0, growing | changes that leave right and wrong answers equally likely |
| KL(t) | mean per-token KL(π₀ ‖ π_t) over all H samples' response tokens, fp32 | 0 exactly | small | small, growing | behaviour on text outside H |
| accuracy gain | greedy strict accuracy at the last step minus step 0, GSM8K test and GSM1k, drop rule, paired over problems kept at both | 0 | ≈ 0 | +0.03 to +0.08 | sampled (non-greedy) behaviour |
| length | mean response tokens of the step's rollouts and the share inside the window | 198 tokens, 0.95 | ≈ flat | within ±25% | content of the answers |
| ‖δW‖ | Frobenius norm over the 196 matrices, fp32 | 0 | reported | reported | direction |

Positive control for ΔR: R itself at step 100, where the accuracy gain must also show. Noise floor for ΔR: N at the
same step.

## 5. Predictions (filed in `experiments/qwen_rl/results/predictions.json` before the run)

- **G (control).** Pass if the adversary on maths rows ≤ 2 × 1.62 = 3.24 nats and masked KL ≤ 2 × 0.142 = 0.284;
  fail if either exceeds 3× (4.86, 0.426); ambiguous otherwise (the run continues; a card asks how to read later δWs).
  Expected: pass.
- **P0 (step 0).** GSM8K drop-rule strict accuracy within ±0.015 of the baseline's 0.777, GSM1k within ±0.015 of
  0.782 (the baseline ran bf16 weights, this runs fp32 master under autocast); ΔR(0) and KL(0) ≤ 1e-6. Fail → stop.
- **P1 (RL learns).** RL's GSM8K gain at step 100: "learns" if ≥ 0.02 and ≥ 2 paired SE; "no gain" if < 0.01;
  ambiguous between. Expected: learns.
- **P2 (not memory).** RL's GSM1k gain minus its GSM8K gain: "same" if |difference| < 2 SE of the difference and
  < 0.02; "GSM8K-only" if GSM8K's gain exceeds GSM1k's by ≥ 0.03 and 2 SE; ambiguous otherwise. Expected: same.
- **P3 (null).** N's GSM8K gain: |gain| < 0.015. Expected: holds.
- **P4 (early signal).** R's ΔR(t) > 0 at every saved t ≥ 1, and R's ΔR(t) ≥ 3 × |N's ΔR(t)| at t = 10, 50, 100.
  Expected: holds at 50 and 100; at 10, uncertain.
- **P5 (small steps are small).** R's KL(1) < 1e-4 nats per token; R's KL(100) between 1e-3 and 0.05.
- **P6 (length).** R's mean rollout length at steps 91-100 within ±25% of steps 1-10's, and the share inside the
  window ≥ 0.90. Expected: holds.
- **P7 (SFT vs RL).** F's KL(100) ≥ 3 × R's KL(100), and F's GSM8K gain ≤ R's. Expected: both hold (human solutions
  are terser than the model's style; SFT moves behaviour more per unit of gain).

Edge cases the code writer must check before filing: an empty readout pool (no mixed problem in H: then ΔR is
undefined and P4 unscored, reported as such), SE of a zero difference, ties at the thresholds (≥ and < as written), a
step with every group all-right or all-wrong (zero gradient: the step still counts and δW is unchanged).

## 6. Decision tree (each branch approved with this plan)

- **G fails** → stop; card "the decomposition does not explain maths answers", options to the user.
- **P0 fails** → stop; fix the cause, rerun A0 (class 2).
- **R1, P1 learns** → the next plan: teacher training of the decomposition on R's δWs at steps 1-100, against N and
  F, scored by the adversary (through `/design`).
- **R2, P1 no gain** → one rerun of R at lr 3e-6 (~$6, approved here). Still no gain → card: the harder-benchmark search.
- **P1 ambiguous** → the next plan proceeds on R's δWs, with the reading labelled weak.
- **P6 fails** (answers lengthen past the window) → card: thinking leakage; the δWs are kept and still scored.
- **P2 GSM8K-only** → FYI and an analysis line; the plan goes on (the update is still an update).

## 7. Budget and hand-off

~5.5 h of H100 at ~$3/h, $17-20, inside the session's $40 with the 15% margin; the benchmark slice S guards it. δWs
stay on the volume (RL 7 × 2.8 GB, N and F 4 × 2.8 GB each, ~42 GB of the ~46 GB free; if space runs short, N and F
drop step 50). Code: `experiments/qwen_rl/run.py`, shared pieces in `src/commute/qwen_rl.py`; G calls `qwen_vpd`'s
existing masked and adversary functions. One review by a fresh Opus agent before the paid run. Runner: `/pod-runner`.

## Not in this plan

Rejection-sampling SFT, a LoRA arm, SGD, a second seed, thinking mode, any map applied to δW (next plan), the outside
share (on Qwen every matrix has at least as many subcomponents as dimensions, so the span is likely full; to be
checked on alive counts in the next plan).
