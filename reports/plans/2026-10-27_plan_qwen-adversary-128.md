# Plan: does the paper's adversary read low on the Qwen decomposition at the authors' size?

Apparatus, not a finding: no E-code. Addendum to `reports/plans/2026-10-27_plan_qwen-port-and-baseline.md`. The user
chose option (a) on card decision-qwen-adversary (2026-10-27, decisions.md): run this before the first Qwen
experiment is planned.

## The question, in plain words
On four text rows, the paper's 20-step adversary drove the unchanged Qwen decomposition to about 20 nats of KL from
the model's own outputs, far above the 0.15 nats of the importance-masked comparison, with readings from 7 to 28 across
seeds. The authors evaluate the same adversary with one shared setting per subcomponent over 128 rows of 512 tokens.
A setting shared by four rows can be tailored to them; one shared by 65,536 positions cannot. This run reads the
adversary at the authors' size, three random starts, to decide whether it can serve as the pass/fail test for
decompositions after an update on Qwen.

Restated in the Terms: the decomposition is the unchanged one (no component training, no teacher training, no
update); the measurement is the evaluation adversary of `launch_config.yaml` (`PGDReconLoss`: random start, source
shape c, 20 sign steps of 0.1, Δ on), mean KL(clean ‖ masked) over every position.

## Design check (`/design`)
- **Goal served:** a pass test for the Qwen line (`GOALS.md`: is the decomposition still faithful after an update).
  Without a usable adversarial reading at W₀, no updated Qwen decomposition can be scored the way the 67M ones were.
- **Nearest tried:** the port's check 4 (4 rows, 3 seeds per side, `reports/analyses/2026-10-27_analysis_qwen-port.md`);
  on the 67M, `decisions.md` keeps one start and 20 steps with readings as lower bounds, and the floor rule "max/min − 1
  above 0.25 → read more rows per block".
- **Patch or step:** a step (it settles the pass test before the first experiment is designed around it).
- **Decisions:** fp32 (as every later W₀ + δW reading will be); torch port, Γ in bf16 as in the port check.
  Departure from the authors: they ran in bf16 on their chat-formatted eval set; this runs fp32 on FineWeb-Edu
  (comparable with the 4-row readings, which also used FineWeb-Edu). Named, not removed: the precision gap was
  already present between torch and JAX in check 4, whose ranges overlapped.
- **What would change plans:** "unusable at this size" (below) → a pending-decision card: what Qwen's pass test is
  instead (more steps or rows, the stochastic-mask KL, a behaviour score), before the first experiment's plan.
- **Control first:** stage 1 reproduces the 4-row readings in the same job, with an early stop.
- **Class:** user's explicit choice (a), 2026-10-27.

## Measurements and expected values
Rules are filed in `experiments/qwen_adversary/results/predictions.json`.

| number | expected | blind to |
|---|---|---|
| control: mean of 3 seeds on rows 0-3 | 10-30 nats (port: 18.8 torch, 21.1 JAX); else stop | anything that only shows at 128 rows |
| KL with m = g, Δ off, 128 rows | 0.05-0.3 | adversarial settings |
| ceiling: KL(clean ‖ uniform), 128 rows | 7-11 | nothing it is used for: a fixed reference |
| P1: 128-row mean / control mean | < 0.25 (block size is the cause); > 0.5 is the null | whether 20 steps converge (readings are lower bounds) |
| P2: max/min − 1 over the 3 starts at 128 rows | < 0.25 | spread over more than 3 starts |

Reading: usable with one start if the 128-row mean is below half the ceiling and P2 holds; usable with three starts if
below half the ceiling and the spread is in [0.25, 1); otherwise unusable at this size. Noise floor: the three starts
themselves (P2); the control's own spread on 4 rows was about 0.4 of the mean.

Rows: the first 128 FineWeb-Edu sample-10BT train documents with at least 512 tokens, cut to 512, at the revision the
port used; the first four equal the port's text rows (`experiments/qwen_adversary/make_tokens.py` asserts it).

## Code
`src/commute/qwen_vpd.py`: `pgd_kl_rows` takes a function giving each row's g and clean logits, so the 128-row run
recomputes them per row instead of holding ~180 GB of g; `pgd_kl` keeps its interface (the port check's call).
`src/commute/qwen_data.py`: the FineWeb rows rule. `experiments/qwen_adversary/run.py`: stages control →
references → 128 rows, each written as it finishes. Fresh-Opus review of the diff before the run.

## Cost and routing
~45 min per 128-row start on an H100 (port check: 0.79 s per row pass on an L40 with g held; recomputing g adds
about a quarter), ~2.5 h in all. The run needs an 80 GB card: the target, U, V and Γ held together are ~44 GB before
autograd state. The cluster was full at planning time (8 of 8 L40s allocated, 2026-10-27 ~17:30) and its L40s cannot
hold the recompute layout, so: RunPod H100 in US-MO-1 with the network volume, ~$8-9. The first pod fills the volume
from our W&B artifact of the converted copy (`src/commute/qwen_files.py`), ~20 min, a one-time cost.

## Not in this plan
More adversary steps (the authors' slow PGD reads at 40-1280 steps), the per-step trajectory, chat-formatted or GSM8K
rows, bf16, more than three starts, any updated decomposition.
