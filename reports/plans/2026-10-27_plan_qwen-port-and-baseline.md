# Plan: can we run the Qwen3-1.7B decomposition in our harness, and does GSM8K leave room for a small update?

Apparatus, not a finding: no E-code. User go-ahead 2026-10-27 ("start with 1 and 2"), after reading the scoping
reports `reports/analyses/2026-10-27_analysis_qwen-{loader,compute,rl-gsm8k,task-choice,volume-check}.md`.

## The question, in plain words
The Qwen3-1.7B decomposition (Goodfire run p-e9fcffd5) ships with a JAX loader; every measurement we have is PyTorch.
Part 1 converts the checkpoint once to fp32 safetensors and ports the causal-importance function Γ and the masked
forward to PyTorch, then checks the port against the authors' JAX code on the same tokens, so that a later strange
number can be told apart from a port bug. Part 2 measures, on the unchanged model, how well Qwen3-1.7B does GSM8K in
non-thinking mode, how its answers are graded deterministically, and how long its answers are against the
decomposition's 512-token window: the facts the first experiment (a fine-tune on GSM8K's human-written solutions, to
be discussed with the user) is designed around.

## Design check (`/design`)
- **Goal served:** `GOALS.md`'s question on a real post-trained model; decisions.md 2026-10-26 sends the commute line
  to the larger model with teacher training as the map. Neither part tests a claim; both make the first Qwen
  experiment readable.
- **Nearest tried:** none on this model. The 67M harness (`src/commute/vpd.py`, `faithfulness.py`) is the pattern;
  its Γ was global and bidirectional (`reports/reviews/2026-10-27_review_gamma-bidirectional-windows.md`); Qwen's is
  causal, so windows are valid there.
- **Patch or step:** a step (infrastructure for the next phase).
- **Decisions:** relies on fp32 for anything read at W₀ + δW (bf16 hides single steps, RL report §4); the cluster
  `normal` exception for the Qwen line (decisions.md 2026-10-27). No departure.
- **What would change plans:** a port that cannot meet the pass rules → stop and report; GSM8K accuracy near ceiling
  or strict/lenient extraction disagreeing on > 5% of answers → change the prompt or task before the first fine-tune.
- **Control first:** Part 1 stage 0 is the target model alone (HF torch vs JAX logits) before Γ is compared.
- **Class:** user go-ahead in chat for these two items; new model, so the first experiment itself still needs the
  user's confirmation after its plan.

## Part 1: conversion and port

States: target = HF `Qwen/Qwen3-1.7B` snapshot 70d244cc; decomposition = `artifacts/qwen3-1.7b-p-e9fcffd5`
(`runs/p-e9fcffd5`, step 100000). Δ = W − VU (not stored).

Work items:
1. JAX CUDA environment for param-decomp 177d0cce8 from a **separate** clone in `~/commute-data/param-decomp-177d0cce8`
   (the read-only `~/param-decomp` stays untouched), venv beside it; built in a CPU job.
2. `experiments/qwen_port/convert.py` (JAX venv, CPU job via `fleet/cpu-run`, high memory): reads each Orbax array and
   writes float32 safetensors to `~/commute-data/artifacts/qwen3-1.7b-p-e9fcffd5-torch/`, keys = the Orbax tree path
   joined with "." (exactly as listed in the loader report §1), plus `index.json` (key → shape, dtype, sha256) and a
   copy of `launch_config.yaml`. Read-back check: shapes and dtypes equal the Orbax metadata; a few slices compared
   value by value.
3. `experiments/qwen_port/jax_reference.py` (JAX venv, one cluster L40, `debug`, ≤ 30 min, via `/pod-runner`): on fixed
   tokens (4 generic 512-token text rows and 4 GSM8K chat prompts in non-thinking format, written as a token file the
   torch side reads too) writes: target logits; the 29 residual taps (boundary 28 before the final norm); Γ's output g
   for every site on 2 rows and summary statistics on all; masked-forward logits for m = g without Δ, m = 1 without Δ,
   and one fixed source array with Δ on; the 20-step adversary's KL over 3 seeds. Outputs in
   `experiments/qwen_port/results/large/` (gitignored), a small JSON summary in `results/`.
4. `src/commute/qwen_vpd.py` (torch): load the safetensors; Γ; masked forward via hooks on the 196 `nn.Linear` of
   HF `Qwen3ForCausalLM`; stochastic and 20-step adversarial checks. Fp32 by default, TF32 off.
5. `experiments/qwen_port/torch_check.py` (cluster L40, `debug`, via `/pod-runner`): reproduces item 3's outputs and
   scores the pass rules below. Feeding JAX's taps into torch Γ separates a Γ error from an HF-vs-JAX target difference.

Pass rules (fixed here, before any run; loader report §3):

| check | must read | else |
|---|---|---|
| 0. target logits, HF torch vs JAX, same tokens | KL(HF fp32 ‖ JAX) at most 2× the bf16 noise floor KL(HF fp32 ‖ HF bf16) on the same tokens | stop: the target differs; fix before Γ |
| 1. g, torch Γ on JAX taps vs JAX | share of (position, component) pairs with abs diff > 0.1 below 1e-3 | stop, fix the Γ port |
| 2. g, torch end to end vs JAX | same rule | fix the taps |
| 3. masked-forward KL to clean logits, three mask settings | |torch − JAX| ≤ max(5% of JAX, 2× check 0 floor), scored with both sides in bf16 (the floor term: m = 1 sits near the noise floor) | fix the masked forward |
| 4. 20-step adversary KL, 3 seeds per side | ranges overlap | look at the adversary port |

Afterwards: copy the converted files to the RunPod volume (later, with the first paid run), and record
`COMMUTE_QWEN_DIR` in `cluster/env.sh`.

## Part 2: GSM8K baseline (unchanged model)

`experiments/qwen_gsm8k_baseline/run.py`, one cluster L40, `debug`, ≤ 30 min, via `/pod-runner`. Chat template with
`enable_thinking=False`; a fixed instruction asking the model to end with "The answer is N."; fp32 or bf16 weights
(no update is read, so bf16 is fine here; say which).

| number | how measured | expected |
|---|---|---|
| greedy accuracy, test set (1,319) | strict rule (final "The answer is N" number equals the gold `####` integer) | ~0.6-0.75 (one outside non-thinking greedy eval: 65.3%) |
| same, lenient rule | last number in the response | within 0.05 of strict; a larger gap means the prompt needs work |
| response length | tokens of prompt + greedy response | ≥ 95% within 512 |
| sample disagreement | 8 samples at T = 0.7 on 256 test prompts: share of prompts with both correct and incorrect samples | 0.2-0.5 (RL needs this to be well above 0) |
| truncation | share of greedy responses hitting the generation cap (512 new tokens) | < 0.02 |
| positive control | the strict grader on the gold solutions themselves (rewritten to end "The answer is N.") | 1.000 |

Predictions are filed in `experiments/qwen_gsm8k_baseline/results/predictions.json` before the run (the table's
expected column, with these thresholds). The per-example greedy responses are saved (they become the held-out set
for the teacher-forced behaviour score later).

## Cost and routing
All on the MATS cluster (free): CPU jobs for the environment and conversion; three ≤ 30-min `debug` GPU jobs (JAX
reference, torch check, GSM8K baseline). RunPod only if a `debug` job does not start within 10 min (each ≤ $1).

## Order of work
1. Parallel: agent A (env + convert + `jax_reference.py`), agent B (`qwen_vpd.py` + `torch_check.py`, from the JAX
   source; key names as in item 2), agent C (GSM8K baseline + predictions).
2. `/pod-runner`: GSM8K baseline as soon as C is done; JAX reference as soon as A is done; torch check after both A and
   B. Fixes to the port go through the same check.
3. Analysis report for each part (`/writeup`), `PENDING.md` lines; then the first-experiment plan for discussion.

## Not in this plan
Any fine-tune, teacher training or RL; uploading to the RunPod volume; faithfulness of the unchanged decomposition on
GSM8K text (the gate; first stage of the first experiment, after the port passes); Qwen3-8B samples; thinking mode.
