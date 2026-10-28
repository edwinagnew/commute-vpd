# Plan: do the five red-teamed claims survive several adversary starts and the missing controls? (E47, `experiments/claim_tests/`)

*2026-10-28. Five $0 MATS-cluster jobs, approved by the user today. Source: `reports/reviews/2026-10-29_review_claims-red-team.md`
("Cheapest test" column) and its `_full.md`. E45 is taken by the Qwen Q2 plan (`2026-10-28_plan_qwen-q2.md`), so this is E47.*

## 0. The request, in the Terms' words, and the design check

One test per claim: (i) teacher training's decompositions pass the adversarial check under several adversary starts
where do-nothing fails; (ii) the rank-8 LoRA's teacher-trained decomposition passes, and the pass does not need the
rank cap; (iii) the scaled adjoint split's keep falls with the update's size rather than its path; (iv) teacher training
plus remainder reads worse than do-nothing at every start; (v) teacher training with plain SGD as its optimiser fails on
r1's Adam and LoRA teachers at every rate and step tried. The adversarial half of (iii) is settled from the JSON already
(E22: SGD at size 0.0122 passes at 0.98 while Adam at 0.0113 fails at 2.52) and is not rerun.

- **Goal.** `GOALS.md` weak version: (i), (ii), (iv), (v) support teacher training as the map with Δ fixed; (iii) the splits.
- **Nearest tried.** E34 Amendments 4-6, E39 Amendment 2, E31 Amendments 1-3, E43. Each read one adversary start; this adds starts and the missing grid points.
- **Patch or step.** Replications and controls of claims already in the log (approved classes 1 and 2), plus the user's go-ahead.
- **Decisions.** Tests `decisions.md` rows 31 (size) and 50 (Adam); departs from none.
- **What changes next.** Each outcome rewrites or keeps its claim's wording in the log card, the notebook and `decisions.md`.
- **Control first.** In every job, W₀ is read first, alone, at every start that job uses, with the stop rules of §3.

## 1. The experiment

E47, `experiments/claim_tests/`, stages T1-T5, each its own `debug` job of 30 min or less. **Card: L40.** Its W₀ floor on
Pile val rows 0-255 is f = 0.2088 (jobs 40166, 40195, 40497 agree to 17 digits), so `faith_score.verdict` gives a
pass line of 1.5/(1+f) = 1.241 and a fail line of 1.5(1+f) = 1.813. On the H100 the lines were 1.40 and 1.61, so no
number here is compared with an H100 number. Every ratio is KL(state, start s) / KL(W₀, start s), with W₀ read in the
same job, on the same 256 rows (both 128-row blocks) and at the same start. Block 0 alone was rejected: it is lenient,
e.g. the Adam teacher-trained decomposition reads 1.15 on block 0 and 1.93 on block 1 (E39).

## 2. Shared reader: `faithfulness.multi_start`

`multi_start(model, blocks, starts, w0=None, stop_at=None, log=print) -> {"kl", "ratio", "mean", "max", "complete"}`

- `"h<k>"` = `pile_faithfulness(model, blocks, seed=k)["pgd_kl"]`, the harness start k.
- `"budget"` = the budget code's start (E34 Amendments 4-5): `gamma.pgd_sources(model, block, 20, step_size=0.1, seed=0)[1]`
  per block, mean over blocks. Amendment 5 showed the start alone takes one do-nothing state from 1.6 to 9.2. It is cheap
  to include: 33-39 s per block on the L40, against 65 s for a harness start. The 160-step budget is dropped (226 s more per block).
- With `w0`, it returns per-start ratios, their mean and their max. With `stop_at` (the fail line), it stops after the first
  start whose ratio reaches the line. That is a final fail under the max, because a max can only rise; `mean` is then None
  and `complete` is False.
- It is additive and imports `gamma` lazily. Unchanged signatures: `faithfulness`, `harness_pgd` and `pile_faithfulness`
  (via `harness.load`/`pile_ratios`, so every `run.py` that uses the harness; `faith_score.w0_check`, `shift_map`,
  `broad_fragility/run.py`, `amend2.py`, `teacher_training/amend5.py`), and `gamma.pgd_sources`
  (cancellation*, gamma_refit, teacher_spread, adjoint_sgd, amend4/5, `excess`).

**Unit times (L40).** u = 135 s per state per harness start on 256 rows (`cluster/logs/commute-teacher-e42-40195.out`, "[pile W0 seed k] 138 s"; 40166: 124-127 s). b = 74 s per budget start on 256 rows (H100 15 s per block in `run_amend5.log`, × 2.6; E31 measured 33 s per block on the L40). Training takes about 1.7× the H100 time: fine-tune 35 s per 1000 steps (40166), student about 100 s (61 s, `run_amend6.log`), broad teacher training about 150 s (90 s, BF `amend2.log`). Loading takes about 1 min (40195).

## 3. Stages

**C0 (every job, first, alone):** W₀ at the job's starts. **Stop** if `w0_check` fails (seed-0 PGD within 10% of 0.69, CI
within 2% of 0.348, f ≤ 0.25), or if the budget start's W₀ KL is outside 0.85-1.15 of h0's (Amendment 5: 2.4%). The
ratio at W₀ is 1 at every start by construction, so W₀ cannot cross a line. The real stop rule is the floor.

**R_max** = max over a state's starts of its ratio. **Verdict:** pass ≤ 1.5/(1+f); fail ≥ 1.5(1+f); inconclusive between.
A tie on a line goes to that line's side. A state with a missing start (error or non-finite value) and no early fail is
*unscored*; an unscored state never counts as a pass.

### T1, claim (i): ≤ 26.5 min; starts h0, h1, budget (budget first)
| State | Source | Expected (W₀ = 1) |
|---|---|---|
| r1 SGD do-nothing, **positive control** | teacher regenerated: `run.Job.teacher`, SGD 0.03, 1000 steps, seed 0 (G: P within 0.02 of 0.9834, size within 5% of 0.02875) | budget ≈ 9 (H100 9.24): **fail**; else stop |
| r1 copy, task text | saved `teacher_training/results/large/r1_t534_v369/amend5_A_sgd_seed0_step1000.pt` | ≈ 1.0-1.15: pass |
| rand32 do-nothing | teacher regenerated: `newskill.py` recipe (Adam 3e-5, batch 32, 1000 steps, seed 0; G1 as filed) | h0 1.50 (L40, E42); budget unknown |
| rand32 copy, task text | saved `large/e42/rand32/A_step1000.pt` | h0 0.96: pass |

Do-nothing states are read with `stop_at`. Time: 1 + 1.5 + W₀ (2u+b) 5.7 + control 1.2 + 3 × 5.7 = 26.5 min.
**Cut:** harness start h2, the arm-B and seed-1 copies, rand1 and ident32.
**Rule.** *Stands*: both copies pass, both do-nothing states fail, and each copy's R_max × (1+f) < its do-nothing's.
*Falls*: either copy fails. *No power on X*: the copies pass but X's do-nothing does not fail; the claim then stands
only for the other teacher. Anything else is *unsettled*.

### T2, claim (ii): ≈ 20.5 min; starts h0, h1
| State | Source | Expected |
|---|---|---|
| Adam full-rank teacher-trained, **positive control** | saved `broad_fragility/results/large/trained_step1000.pt`; δW regenerated (`run.py` Adam 3e-5, 300 steps, seed 0; gain 0.1301 ± 0.005) | h0 ≈ 2.2 (H100 2.20): fail |
| LoRA rank 8, teacher-trained | saved `a2_tt_lora_s0.pt`; δW from `amend2.lora_delta(seed 0)`, 1e-4, 300 steps (gain 0.1479 ± 0.005) | ≈ 1.2-1.3 (H100 1.27): likely inconclusive on the L40 |
| LoRA rank 128, teacher-trained | new: the same LoRA recipe at rank 128, then E37's teacher training (Adam 1e-5, batch 8, 1000 steps, order 0) | real: ≈ rank 8; cap artefact: climbs towards 2.2 |

All teacher-trained states are read from bf16 offsets, the new one after a round trip through bf16, so that every state
has the same precision. The positive control reads h0 only. Time: 1.5 + W₀ (2u) 4.5 + regeneration 0.75 + 2.5 (teacher
training) + 2.25 + 2 × 4.5 = 20.5 min. **Cut:** rank 32 (the first thing to add if rank 128 climbs), the second row set,
LoRA seed 1, the SGD states and the budget start.
**Rule** (statistic: the mean over the starts). *Absolute*: rank 8 passes → stands; rank 8 fails → falls (not real);
in between → unsettled on this card. *Cap*: rank 128 fails while rank 8 does not → cap artefact; rank 128's mean
≤ rank 8's × (1+f) → no cap effect up to rank 128; otherwise unsettled. If the control does not fail, the cap reading is
unscored, the job continues, and the lead is told.

### T3, claim (iii): ≈ 8 min; no adversary
The r1 windows, `adjoint_sgd/run.py`'s `finetune` (plain SGD, arm A) at seeds 0 and 1, all regenerated. Keep = P(target)
of `split.adjoint_state` over the P(target) of the update's own weights. **Control first:** the 0.01 run reproduces E31's
keep at steps 10, 100 and 1000 (seed 0: 0.374, 0.419, 0.130; seed 1: 0.389, 0.446, 0.126) within 0.03, and its sizes
within 2%; stop otherwise. The positive control is α = 1 reading 0.13.
(a) The 0.03 run, checkpointed at the steps whose size is nearest the 0.01 run's step-100 and step-1000 sizes (pass 1
records the size at every step to 400; pass 2 reruns, deterministically, to those steps; matched within 5%).
(b) The 0.01 run's step-1000 δW scaled by α = size₁₀/size₁₀₀₀ and size₁₀₀/size₁₀₀₀ (≈ 0.24, 0.57). Since c ignores
scale, this changes size alone.
**Rule**, per seed. *Size stands*: |keep(a) − keep(0.01, same size)| ≤ 0.05 at both sizes, and keep(b) ≥ the 0.01 run's keep at
the matching step − 0.05 at both α. *Size falls*: (a) differs by more than 0.10 at a size, or keep(b) ≤ keep(0.01, step
1000) + 0.07 at both α. Anything else is mixed. Seeds that disagree make it *mixed*. A step not found within 5%, or a
denominator P below 0.5, leaves that cell unscored.

### T4, claim (iv): ≈ 24 min; starts h0, h1, h2; step 1000 only
`remainder.py`'s recipe: emoticon, Adam 3e-5 fine-tune, seed 0 (G1: P within 0.02 of 0.99961, size within 5% of 0.03209);
teacher training arm A, order 0 (G3: P within 0.02 of 0.99675, Δ share ≤ 1e-6; G2 1e-3 as filed). The states are
do-nothing (the reference; H100 h0 1.499) and plus remainder (H100 h0 1.865), both regenerated. Time: 1.5 + W₀ (3u) 6.75 + 2.3 +
2 × 6.75 = 24 min. **Cut:** T10 (by the reviewer, close to expected from size alone), extra batch orders, the budget start.
**Rule** (q_s = KL_plus(s)/KL_nothing(s)). *Stands*: min over s of ratio_plus > max over s of ratio_nothing, and q_s ≥ 1.1 at all three starts.
*Falls*: q_s ≤ 1.0 at any start. Anything else is *unsettled* (worse at every start but inside the start spread). No
early stop is used here; a missing start makes the stage unscored.

### T5, claim (v): ≤ 27 min; starts budget, h0
Teachers regenerated from `amend6.py` (r1 Adam 3e-5 `ft_teacher`, P 0.9962; `amend3.lora_teacher`, P 0.9992). Arm S:
plain SGD on the task windows, batch 32, order 0. Training: Adam teacher at 0.03 and at 0.01 to step 1000; LoRA teacher at
0.01 to step 1000 and at 0.03 to step 100 (5.2 min). **Positive control**: Adam teacher, 0.03 at step 1000, h0 only (H100 3.14;
fail, else stop). The cells, (0.03, step 100) and (0.01, step 1000), are read on both teachers with `stop_at`; a cell with P < 0.9
is not read. f is taken from the deterministic L40 value 0.2088 when W₀'s h0 KL reproduces 0.69174 within 1e-3;
otherwise h1 is read in the job (2.25 min more). Time: 1 + 1.2 + 5.2 + W₀ (u+b) 3.5 + 2.25 + 4 × 3.5 = 27.2 min.
**Cut:** rate 0.003 (0.003 × 1000 steps equals the 0.03 × 100 product, which is read), (0.03, step 1000) on the LoRA teacher (known: 2.86), h1 and h2.
**Rule.** A cell passes on a teacher if P ≥ 0.9 and its verdict is pass. *Stands*: no cell passes on either teacher.
*Falls*: some cell passes on both teachers. *Mixed*: a cell passes on one teacher only. *Unsettled*: fewer than two
cells per teacher have P ≥ 0.9 (plain SGD did not learn, so it says nothing about the check), or an unscored cell could
still turn the stage into a fall.

## 4. Predictions

These are filed in `experiments/claim_tests/results/predictions.json`, with the rule, thresholds, relative or absolute,
blind-to, positive control, noise floor and lean for each stage. Leans: T1 stands for r1 and has no power on rand32 (low);
T2 is unsettled on the L40 with no cap effect (low); T3 falls, because the path matters (medium); T4 is unsettled (low);
T5 falls (low).

## 5. Pre-mortem (not a gated run)

- **Recipe:** every recipe is copied from its named script and gated on its filed P and size.
- **Data:** the same Pile val rows 0-255 for every state; the windows come from the scripts.
- **Precision:** saved states are bf16; the new ones are round-tripped through bf16 too.
- **Memory:** ask `--mem=32G`, as E42 did; its peak was not recorded, so measure it with `sstat` in this run.
- **Time and cost:** the worst case is 27.2 min; `--time=00:35:00`; $0.
- **Saves:** JSON per state as it finishes; T2's rank-128 offsets (60 MB) go to `results/large/`.
- **Downloads:** none; NFS.
- **Loss and resume:** a rerun skips finished states.
- **Controls:** C0 and one positive control per stage, at the same rows and starts.
- **Predictions:** edges checked: ties, a missing start, early stops, P < 0.9, zero denominators, degenerate learning.
- **Code never run on a GPU:** `multi_start` and its budget branch. The code review checks them, and each job's C0 exercises both.

## 6. Order and runner brief

The jobs are independent; submit T3, T1, T5, T4 and T2 together, or in that order if the queue holds them back. Job
names: `commute-e47-t1` … `t5`; command: `sbatch cluster/claim_tests.sbatch t<k>` → `experiments/claim_tests/run.py t<k>`.
Outputs: `experiments/claim_tests/results/t<k>.json`, `results/large/t<k>.log`, `cluster/logs/commute-e47-t<k>-<id>.out`;
T2 also writes `results/large/t2_lora_r128_tt.pt`. The files are already on NFS; the runner commits the JSON files and
records the job IDs. **Stop and report:** C0 or a positive control misses, or a job has not started 10 min after
submission. No job moves to RunPod without the lead.

## Not run
The adversarial half of (iii) (settled from E22 and E31); every cut listed above; the H100 lines.
