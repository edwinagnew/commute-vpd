# P2.47: VPD subcomponents under a top-k gate in place of the gate function (plan, 2026-11-20)

Cell P2.47 (compare-lead, due 2026-11-23). User, task file 2026-11-20 12:24: "(3) by 11-23: VPD components under a
top-k gate in place of the gate function, forward passes only." Card `experiments/topk_gate/card.md`; rules and
expected values `experiments/topk_gate/results/predictions.json`; code `experiments/topk_gate/run.py`,
`src/commute/topk_gate.py`; job `cluster/topk_gate.sbatch`; inputs outside git `experiments/topk_gate/results/manifest.txt`.

## Design check (/design)
1. Goal: part 2 of the paper plan, the comparison on the 67M. The results skeleton (writeup-lead, 2026-11-20, run 4)
   says the neuron readings differ from the share in the gate as well as the unit. This run changes only the gate.
2. Nearest tried: E143, the TopK neuron share (holes_fill). It uses the same top-k gate on MLP neurons and reads 4 of 9
   pairs the built way and 5 reversed (`method_table/results/final.json`). E136 is the share as saved (9 of 9). Neither
   kept VPD's units with a top-k gate.
3. This is a step toward the goal, not a patch. It fills the gate-versus-unit cell that the results table needs.
   Nothing in `decisions.md` is departed from.
4. Definition: no verdict on reuse. The sides keep the names they were built with. It reads the share's separation and
   adds no clause to the working definition.
5. What would change what we do: "holds above" means section 5.2 can credit the unit. "Falls" means the trained gate
   function carries the separation, and the write-up has to say so.
6. Control first: the zero change reads exactly 0. Then on every state, Gamma's own masks go through the new code path
   and must reproduce the saved share within 0.01 (measured re-read gaps were 1e-4 to 0.002).

## What runs
The run reads the 68 saved fine-tunes of E187's nine built pairs on E136's read rows, each fed exactly as the saved
share fed it (same windows, batch and ci_len). For each fine-tune and each batch there is one clean forward at W0 and
one at the fine-tune. From those forwards five gates each give masks for the released subcomponents (Delta out), and
the masked forward runs once per gate and side:

| gate | mask on subcomponent c of matrix m at a position | role |
|---|---|---|
| gamma | Gamma's causal importance (as saved) | positive control: must reproduce the saved share |
| gamma_bin | 1 if Gamma's importance > 0.5 | reported: binarising alone |
| topk_x1 | 1 if \|x . V_c\| is among the k_m largest of m's alive subcomponents | **scored** |
| topk_x0.5, topk_x2 | the same at round(0.5 k_m) or 2 k_m | reported beside |

Share of a gate = mean(L01 - L00) / mean(T1 - T0), the GOALS.md r with that gate in place of Gamma. The inner
activation x . V_c is param-decomp's `get_component_acts` (pinned 74146b5) on the matrix's cached input, in fp32.
k_m is E136's saved k (Gamma's mean count above 0.5 per position at W0, over 305,423 read positions). It is also the k
of the TopK neuron share's "kmatched", so this gate and the neuron one differ only in the unit.

**Filed rule P1** (topk_x1). Each pair gets the scorecard's call: "built" if every reuse seed lies above every new seed,
"reversed" if every one lies below, "not apart" otherwise, and "not read" if any value is nan. If any pair is not read,
P1 is "no call". Otherwise P1 is "falls to the neuron level" when at most 6 pairs are "built" (neurons for half the gain:
6/0/3, `final.json .tally.built.k_half`), and "holds above" when 7 to 9 are. The prediction filed is **falls**. The
count against the TopK neuron share's 4 is reported beside it. Edge cases are in predictions.json: all nan gives no call;
top-k sets that never move give 0 built, which reads "falls" and is a real outcome; ties count as not apart.

## Stages, time, cost
- Stage 0, alone: the zero change on the first state's rows. The gain, every gate's numerator and every row's
  L01 - L00 must read exactly 0, or the job exits 1.
- Stage 1: the states in an order that puts each pair's first seed on both sides first. Each state is written as it
  ends, and a rerun skips finished states. A gamma-gate miss on the saved share exits 1. The projected time is checked
  after 4 states, with a stop at 45 min. Per-row cells (float32, about 2 MB in total) go to `results/large/cells/`.
- Score (CPU, also run at the job's end): `results/score.json`.
- Queue: MATS debug L40, one job, $0. E136 read its 68 states in about 5 min (job 56944). E143 ran the same kind of
  masked forwards in about 12 min. This run makes about 2.5 times E136's forwards, so the estimate is 12-15 min with the
  model load. The run takes under 30 min, so it goes to debug.
- argv: `sbatch --comment=E### cluster/topk_gate.sbatch read`

## Choices made inside the cell (lead may revise before filing)
- Candidates are round 1's alive subcomponents only (`transport_test/results/large/alive.pt`). A dead subcomponent never
  gets Gamma's gate either.
- The inner activation is |x . V_c|. The output size, |x . V_c| times norm(U_c), is not used: VPD does not normalise U
  and V, so the two rankings can differ. This is listed under Not run.
- One k per matrix, pooled over E136's rows, not one per task. At half, nine matrices with k_m of 1 or 2 stay at k = 1.
- gamma_bin is added at no extra forward cost (one masked forward a side). Without it a fall could not be split into
  "binary masks" and "a different choice of subcomponents".
- The comparator is the skeleton's neuron version: neurons for half the gain, 6. The closest analogue is really the
  TopK neuron share (4 built, 5 reversed). It is reported, not scored.

## Not run
Output-size ranking (|x . V_c| norm(U_c)); per-task matched k; the size-matched states; the four twin pairs; a top-k
gate that reads the whole row (Gamma is bidirectional, top-k is per position); a Qwen version.
