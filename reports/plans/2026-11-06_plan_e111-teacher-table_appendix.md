# E111 appendix: the state, the room, the rates, the pre-mortem

*2026-11-06, Fable for `qwen-decomp-lead`. Apparatus detail for `reports/plans/2026-11-06_plan_e111-teacher-table.md`.
Every number here is quoted from a file; the stage-1 job replaces the forecasts with readings.*

## A. The state E93's table reads, and what teacher training adds to it

A state on E93's table is a set of pieces $(dU_s, dV_s)$ on the alive columns of each of the 196 matrices
(`src/commute/qwen_maps.py:38`, `SiteChange`; `qwen_d3.ls_change`, `src/commute/qwen_d3.py:395`). Its **moved part** is
$U'V'^\top - UV^\top$ (`experiments/qwen_d1_table/results/predictions.json`, `rules.moved_part`). Columns 1, 3 and 4 read the
weights $W_0$ + moved part; columns 2, 5 and 6 read the fine-tuned weights with the remainder $\delta W$ − moved part
in Δ (`rules.2_share`, `rules.5_K_masked`). That second reading is E43's "teacher training plus remainder" on the 67M
(`experiments/TRIED.md:91`), allowed since the Δ decision of 2026-10-29 (`decisions.md:71`).

Teacher training's own state in `GOALS.md:61` is $W_0$ + change with Γ and Δ frozen; `qwen_teacher.Offsets.change`
(`src/commute/qwen_teacher.py:130`) forms the change from offsets $(dV, dU)$ on alive columns, the same objects as the
table's pieces. So a teacher-trained state is read by the table with no new reading code: its offsets are its pieces.

**E111's state.** Pieces = least squares' pieces on every alive column (fixed, `qwen_d3.ls_parts`, `qwen_d3.py:518`)
plus trained offsets $r$ on E76's 200 often-on columns per matrix (`experiments/qwen_masked_teacher/common.py:33`;
39,105 columns, 0.835 of the sum of $g$, `reports/analyses/2026-11-04_analysis_e76-masked-teacher.md` → stage 0).
Moved part = least squares' moved part + $r$'s change (the cross terms are inside $r$'s change). Loss = KL(FT clean ‖
$W_0$ + moved part), E76's arm A term (`experiments/qwen_masked_teacher/arms.py:113`), on rows outside the table's.

## B. Why least squares is column 2's floor, and the room each state leaves

Column 2 is ‖δW − moved‖²/‖δW‖² against its Gaussian null minus the arm's seed spread (`rules.2_share`). Least
squares' moved part is the least-norm solution of $dU\,V^\top + U\,dV^\top \approx \delta W$ (`GOALS.md:56`), so its
remainder is orthogonal to everything the subcomponents can write, up to the second-order term $dU\,dV^\top$. E93 read
the squared size within 0.0007 of one minus the share on every state (`reports/analyses/2026-11-06_analysis_e93-d1-table.md`
→ "Squared size"), which is that orthogonality to four places. Any further change $r$ inside the subcomponents adds
‖$r$‖² to the remainder's squared norm and subtracts nothing:

$$\text{share}(\text{LS} + r) = \text{share}(\text{LS}) + \|r\|^2/\|\delta W\|^2 \quad (\pm\ \text{the second-order term}).$$

So no trained map can pass column 2 where least squares fails it, and where least squares passes, the trained
correction has a **room**: (null − share − spread) × ‖δW‖², in squared Frobenius norm. From E93's share table
(`...e93-d1-table.md` → "The share left in Δ") and its sizes (→ "What was run", fp64 norms):

| state | ‖δW‖² | null − share − spread | room, squared norm | column 2 before training |
|---|---|---|---|---|
| template s0 | 0.2372 | 0.002790 | 0.00066 | holds |
| template s1 | 0.2391 | 0.002829 | 0.00068 | holds |
| TOFU s42 | 327.97 | 0.000459 | 0.151 | holds |
| TOFU s142 | 328.33 | 0.000471 | 0.155 | holds |
| RL s0 | 0.1232 | 0.000948 | 0.00012 | holds |
| RL s70 | 0.1204 | 0.000978 | 0.00012 | holds |
| shuffled s0 | 0.1369 | −0.000421 | none | **misses**; stays a miss |
| shuffled s70 | 0.1122 | −0.000327 | none | **misses**; stays a miss |
| own-samples s0 | 0.0751 | −0.001001 | none | **misses**; stays a miss |
| first maths u1 | 11.42 | 0.000644 | 0.0074 | holds |
| gold maths u2 | 8.18 | 0.000539 | 0.0044 | holds |

For comparison, E76's clean-term arm A, from zero offsets at rate 1e-6, had a squared size of 0.52 × 3.49 = 1.8 after
1200 steps and 0.15 × 3.49 = 0.52 after 100 (`...e76-masked-teacher.md` → "What does the masked fall cost", step table).
One hundred of those steps would overrun every room above except TOFU's by a factor of 1,000 or more.

**The no-room states** (shuffled s0, s70; own-samples) train with RL s70's rate and stop at RL s70's room, 0.00012,
so that the three maths arms get the same correction size; their column 2 is foreseen to miss, by least squares'
margin plus ‖$r$‖²/‖δW‖² ≤ 0.001, and the plan files it so.

## C. Rate, steps and the walk

Adam at a fixed rate moves each offset by about the rate per step while gradient signs agree, and by about rate × √steps
after that (`...e76-masked-teacher.md` → "Two limits apply to the ruler": "under pure noise Adam's offsets reach about
the rate times the root of the steps, and A and C read about that"). A's squared size grew from 0.52 at step 100 to
1.81 at step 1200, that is **1.2e-3 per step at rate 1e-6** on the 200 columns, and the squared size scales with the
rate squared. So the expected walk is

$$\|r\|^2 \approx 1.2\times10^{-3} \times (\text{lr}/10^{-6})^2 \times \text{steps}.$$

The rate per state is set so that 300 steps are expected to use half the room: lr = 1e-6 × √(room / 0.72).

| state | rate | expected ‖$r$‖² at step 25 | at step 300 | room |
|---|---|---|---|---|
| template s0, s1 | 3.0e-8 | 2.8e-5 | 3.3e-4 | 6.6e-4 |
| TOFU s42, s142 | 4.6e-7 | 6.3e-3 | 0.075 | 0.151 |
| RL s0, s70; shuffled s0, s70; own-samples | 1.3e-8 | 4.9e-6 | 5.9e-5 | 1.2e-4 |
| u1 | 1.0e-7 | 3.1e-4 | 3.7e-3 | 7.4e-3 |
| u2 | 7.8e-8 | 1.8e-4 | 2.2e-3 | 4.4e-3 |

These are forecasts from one arm of one run at another start; stage 1 reads the walk at step 25 on RL s70 and the
first-minute band is 0.3 to 3 times the forecast (a factor, not a point: `/design` → early guards). The walk that
matters is the one after the signs stop agreeing; the first 25 steps may run above the line, which is why the band is
wide and the stop rule is the room itself, not the forecast.

Steps: 300 at most, E76's control length (`common.py:63`, arm C). Readings every 25 steps: ‖$r$‖² (no forward) and
KL(FT clean ‖ $W_0$ + moved part) on 8 rows of the arm's table rows. The **kept state** is the last reading with
‖$r$‖² at or under the room; the table reads it. Training ends at 300 steps, or when ‖$r$‖² first exceeds the room,
or when the 8-row K has not fallen over four readings.

Adam's ε is 1e-8 by default; at these rates the gradients' root second moment must stay well above it, or the step
becomes rate × gradient / ε, that is plain SGD at an unknown rate. Stage 1 logs the median √v across sites at step 25
beside ε; under 10 ε is a stop.

## D. Data and precision

- **Training rows.** Two rows of 256 tokens per step, one from the arm's text and one FineWeb row, as E76
  (`common.py:34-37`, `arms.py:246`). Pools of 32 rows each, none a table row: maths arms from
  `experiments/qwen_sft/results/maths_rows.json` past E74's 32 (`common.py:24,104`); the template arm from the
  template fine-tune's own training items (`qwen_sets.load()["train"]`, the table reads "test", `qwen_d3.py:123`); TOFU
  from TOFU rows outside the table's 128 kept rows (`qwen_d3.py:127`); FineWeb from `experiments/qwen_adversary/results/tokens.json`.
  The writer logs the 16-token overlap of the reading rows with the pool (E76 read 14.6%, the prompt template).
- **Teacher logits.** FT = $W_0$ + δW as its own term (`change_hooks`, `qwen_teacher.py:75`), the fp32 δW file where
  one exists (u1: E93 scored the fp32 file, `experiments/qwen_d1_table/run.py:250`).
- **Precision.** Training forwards in TF32 as `qwen_teacher.train` (first-batch KL within 3.4e-5 relative of fp32,
  `qwen_teacher.py:11`); readings with TF32 off. The offsets are fp32; at rate 1e-8 over 300 steps an offset is about
  2e-7, against column entries of order 1e-2, so about 1e-5 relative: 100 times fp32's resolution, and far under bf16's.
  Two consequences, both stated in the plan: the moved part carries $r$ exactly, since the forward adds it as its own
  term; and in column 5 Γ reads $V'$ in the components' dtype (`Offsets.state`, `qwen_teacher.py:140`), where $r$ is below
  resolution, so the masks of the trained state are least squares' masks. The offsets are saved in fp32
  (`Offsets.save` takes a dtype, `qwen_teacher.py:163`; E76 saved bf16), about 0.8 GB per state in `results/large/`.
- **The base must match.** At zero offsets the training forward's K(FT ‖ $W_0$ + least squares' moved part) must equal
  E93's stored `clean.moved` for the state within 1e-3 relative (the TF32 floor), and the table's own read of the same
  state within 1e-5 (E93 reproduced its controls "to within a few parts in a million"). A miss here is a stop: the
  correction would be trained on the wrong base.
- **Note, 2026-11-06 (review f178f4e3).** Now a RunPod H100 or A100 80GB, per the user (`decisions.md`), not the cluster.
  The offsets are saved bf16 (`run.py` `off.save(..., torch.bfloat16, ...)`, 0.44 GB per arm), not fp32 as written above.
  The TF32-vs-fp32 gap at zero offsets is logged against 1e-3, no longer a stop; the TOFU pool is drawn from retain90
  pairs outside the 800 held ones (`teach.tofu_pool`), not from the table's other packed rows.

## E. Time, memory, cost

Quoted from `experiments/qwen_d1_table/results/run.log` and `experiments/qwen_masked_teacher/results/run.log`.

| piece | reading | source |
|---|---|---|
| table read, one maths state | 944-1022 s | `run.log:209,232,255,278,301` (R_s0 1013, R_s70 990, N_s0 1022, N_s70 1022, F_s0 944) |
| table read, template state | 1191-1284 s | `run.log:111,136` |
| table read, TOFU state | 1725-1820 s | `run.log:57,82` |
| table read, u1, u2 | 1226, 952 s | `run.log:159,182` |
| of which the split | 264-311 s | `run.log:15,47,102,152,202` |
| clean-term training step, 200 columns | 0.77 s | E76 `run.log:35` ("A step 50 0.77 s per step") |
| 300 training steps + 12 readings of 8 rows | about 5 min | 300 × 0.77 s plus 12 × 8 forwards |
| E93 peak | 42.66 GiB card, 95.6 GB host, `--mem=110G` | `run.log:111`, `cluster/qwen_d1_table.sbatch:19` |
| E76 arms peak | 26.1 GB card | E76 `run.log:31,85` |

Per state: about 5 min of training plus 16 to 31 min of table, so 21 to 36 min; the training's fixed least-squares
pieces add about 7 GB in fp32 on the card during training (all 196 moved parts, the weights' own size), which the
table's apparatus is not holding at the same time, so the peak stays the table's 42.7 GiB, inside the L40's 48 GB. Host
RAM: E93's 95.6 GB plus the offsets, under the 110G already asked for.

| stage | states | card time | job |
|---|---|---|---|
| 1 | RL s70, in-room arm and 10× arm, both read by the table | about 1 h 10 min | one `debug` job |
| 2 | shuffled s0, s70; own-samples | about 1 h 30 min | one `debug` job |
| 3 | template s0, s1; RL s0 | about 1 h 30 min | one `debug` job |
| 4 | TOFU s42, s142 | about 1 h 15 min | one `debug` job |
| 5 | u1, u2 | about 1 h | one `debug` job |
| all | 11 states, 12 trained states | about 6.5 h | five jobs, $0 |

`debug` allows 2 h and one card per job (`~/commute-agents/fleet/roles/shell-notes.md` → Running things). E93's five RL
states fit one job in 5046 s (`run.log:302`); each E111 job above carries less than that plus 15 min of training.

## F. Pre-mortem, by CLAUDE.md's headings

- **Recipe and hyperparameters.** The rate is a forecast from another start; the walk may run coherent for longer than
  25 steps and overrun the room before the first reading. Guard: readings every 25 steps, the kept state is the last
  one inside the room, and stage 1's first-minute band on the walk (0.3-3× the forecast) stops the job if the forecast is
  wrong by more than that; the user sees stage 1's exchange rate before stage 2.
- **Data.** A training row equal to a table row makes column 1 a memorisation read. Guard: pools are built from row
  indices outside the table's, and the overlap is logged (E76's 14.6% is the template prefix, not whole rows).
- **Precision.** $r$ below bf16 resolution in Γ's readout (section D): column 5 cannot credit $r$ through the masks,
  only through the weights. Stated in the plan as a limit, not a bug; the 10× arm shows what a larger $r$ does.
- **Memory and host RAM.** The fixed least-squares moved parts on the card during training (7 GB fp32) plus E76's 26 GB:
  about 33 GB, under the table's 42.7 GiB. Guard: the stage-1 job logs the training peak at step 25 and stops over 44 GiB
  (E93's `first_minutes.gpu_max_gib`).
- **Time and cost.** Section E; $0 on the cluster. Guard: E93's per-state time forecast at the control
  (`run.py:192`) carries over; a job past 100 min writes its state and exits 3 for `fleet/resubmit`.
- **Saved.** Per state: the offsets in fp32 (0.8 GB, `results/large/`), the readings curve, the table's `state.json`,
  the kept step and ‖$r$‖². Small files committed. Cluster disk, nothing on a pod.
- **Downloads.** None; the cluster writes into the clone.
- **Pod loss and resume.** Cluster only. A state already written is skipped on resubmission (E93's `run.py:328`).
- **Controls and references.** The zero-offset reproduction of E93's stored reads for every state, before its training
  (section D); E93's do-nothing and least-squares columns on the same rows and job (the table recomputes them: `cells_timed`
  base and own, `run.py:263-264`).
- **Predictions.** Section G of the plan; the share column's three foreseen misses are filed as misses, so the
  table-level verdict is not degenerate; the question each column answers is named.
- **Code never run on a GPU.** The least-squares pieces as a fixed base under `Offsets` (new); the per-state rate; the
  room stop; a `change` for the table that sums least squares' pieces and the offsets. All exercised by stage 1 on the
  tightest state before any other state runs.
- **Note, 2026-11-06 (review f178f4e3), replacing the cluster-only lines above on time, saved, downloads and pod loss.**
  Now a RunPod H100/A100 80GB per the user (`decisions.md`): about 5.5 h of card by `RUN_BRIEF.md`'s minutes (fetch 25;
  stages 35, 70, 80, 70, 50), about $15-20. Offsets are saved bf16 (0.44 GB per arm) on the pod and uploaded to W&B by
  `upload.py offsets`; `results/<state>/state.json`, `run.log`, `table.json` and `console.log` are downloaded after each
  state, as `RUN_BRIEF.md` says, and synced back before a resume on a new pod. The step-25 walk band is no longer a stop
  (it is logged with the forecast); the stage-1 stops are the ones in `rules.judge` as amended in `predictions.json`.

## G. Readings the stage-1 report brings to the user

1. The zero-offset reproduction: K, share, size against E93's stored `R_s70/state.json`.
2. ‖$r$‖² at steps 25, 50, ..., kept step, against the forecast and the room.
3. The 8-row clean K along the run, and the table's six columns on the kept state, beside E93's least-squares row.
4. The 10× arm's columns 1, 2, 5: what the clean term buys when the room is ignored.
5. Card and host peaks, seconds per step, seconds per table read.
