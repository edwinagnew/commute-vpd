# Plan: E12 Amendment 3, does the fall on rand32b survive reading in fp32?

*2026-11-02, coding agent, for the lead. Coded, not run. Predictions `experiments/new_vs_existing/results/predictions_amend3.json`; code
`experiments/new_vs_existing/amend3.py`; job `cluster/new_vs_existing_amend3.sbatch`. Class 1 of `/design` (a GPU pass of the existing harness over saved states); no sign-off needed.*

**Amended 2026-11-02, after the code review (`reports/reviews/2026-11-02_review_e12-amend3-code.md`) and before any run.** Changes: every read also returns Γ's gates, and the gap between the precisions is recorded and stops at W₀ if exactly 0; the fine-tune's re-switching term is reported in both precisions (`term_moved`); the stop on the offsets' rounding comes after all six students are read; P1 "gone" needs `tol` under 0.15; a nan on either table stops stage 0; clause wording no longer says bf16 is the source of a fall (it is the authors' setting, and fp32 is not a truer value); the offsets' fp32 rounding is listed as unread.

## The question, in plain words

Amendment 2 found that students of a fine-tune on map table rand32b lose 16% to 20% of the re-switching term, and on rand32 about 2%. The independent review (M2) found that every read of Γ in that job ran under bf16 autocast, on both sides, so its "hook against weights" pair varied only how the change in weights was rounded. We read the same states again in both precisions and ask whether the fall stays. One free debug job of about two minutes; nothing is trained.

**In the Terms' words.** On Amendment 2's six students (teacher-trained on the rand32b and rand32 fine-tunes, Γ and Δ frozen) and their four fine-tunes, read Γ's values on the state's activations and the masked log-odds of the released components at the trigger, Δ out, in two precisions on the same held-out windows, trigger position and map table: under bf16 autocast, as Amendment 2 did, and in fp32. Does the fall in the re-switching term (the fine-tune's L01 minus the student's L01) stay, in fp32, where Amendment 2 read it?

## Design check (`/design`)

**Goal.** `GOALS.md`'s second branch, through E12's map tables: the fall is the one anomaly in Amendment 1's table, and the user's rule of 2026-11-02 says it blocks readings built on it until run down. **Nearest tried.** E12 Amendment 2 (a bf16 read only; its hook pair is not a precision test) and E62's fp32 reads on another read; neither read Γ in fp32 on these states. **Patch or step.** A patch: closes one item on Amendment 2's Not run list. **Decisions.** No departure from `decisions.md`. **What would change what we do.** "Gone" or "mixed" withdraws or limits Amendment 2's reading; "falls" on rand32 withdraws the difference between the tables; "holds" with no move closes the precision question and leaves the cause open. **Control first.** The do-nothing read at W₀ runs first and alone, in both precisions; its bf16 side must match Amendment 2 and its gap is the noise floor. The recipe replays reads, not training. What differs from Amendment 2's end state is below under States, and the control tests it. **Numbers.** Below; each has an expected value.

## States

| state | source | differs from what Amendment 2 scored |
|---|---|---|
| W₀, two tables | the model, with dW zero | nothing |
| four fine-tunes | `regating_table/results/large/{rand32,rand32b}_s{0,1}.pt`, fp32 dW (dtype read from the file) | nothing |
| six students: rand32b s0, s1, s0_o1, s1_o1; rand32 s0, s1 | `new_vs_existing/results/large/amend2/<name>.pt`, rebuilt by `saved_states.offsets_state` | **The saved files are bf16 offsets, not fp32 weights.** Amendment 2 read the fp32 weights in memory after training and did not save them, so each dW here carries the rounding of its offsets. The control reads the size of that effect on L01, for the first time. |

## Stage 0: the control (runs first, alone; early stop)

| check | reads | must read | else |
|---|---|---|---|
| W₀ cells against E46's | T0, U0, L00 (`Job.task`) | within 0.02 | stop |
| do-nothing at W₀, bf16 | L00 on each table, against Amendment 2's stored | within `read_tol` 0.02; expected difference 0 on this card | stop |
| do-nothing at W₀, fp32 | L00 on each table | nothing stops on its size | |
| Γ's gates, fp32 against bf16 | on each table, max over alive subcomponents of the window mean \|g_fp32 − g_bf16\| at the trigger | above 0: exactly 0 means Γ did not run in fp32 (the masked forward alone could differ) | stop |
| the floor | largest of the two gaps, fp32 against bf16 | finite (a nan on either table counts) and above 0: exactly 0 means the fp32 path did not run | stop |

**The tolerance is measured.** A fall is the difference of two reads, each off by up to the floor if it behaves as at W₀, so `tol = 2 × floor + 0.01`. The 0.01 is an absolute guard: Amendment 2's largest gap between its two roundings of dW, 0.0069, rounded up. If the floor is 0.02, then `tol` is 0.05. Too large a floor gives "no power" on a line rather than a guess.

## Stages 1 and 2: the reads

Stage 1 reads the four fine-tunes. Stage 2 reads the students in Amendment 2's order: each seed's rand32b then rand32, then the two second batch orders. Each state is written before the next starts, and every number goes to the log. Each state also records the gate gap (max and mean over alive subcomponents of the window mean \|g_fp32 − g_bf16\| at the trigger), which separates "Γ did not move" from "Γ moved and the mean did not care". Each fine-tune records its term, L01 − L00, in both precisions. **The offsets' stop comes after all six students are read and written:** the worst \|reload_diff\| must be within 0.05, else stop. The fp32 side of the offsets' rounding has no stored value and is **unread**; it sits on the student side only, since E46's fine-tune files are fp32. If a state lands near a line, that student is retrained on the same card (about 115 s, $0; Amendment 2's retrained students matched Amendment 1 exactly) before it is read. Nothing is retrained otherwise.

| number | how measured | at W₀ | do-nothing | control (bf16) | expected in fp32 |
|---|---|---|---|---|---|
| L00 | masked log-odds under W₀'s gates, mean over 97 windows | rand32b −11.98315, rand32 −11.91238 | same | those values, or stop | within a few hundredths, a guess |
| L01, fine-tune | the same with Γ on W₀ + dW | L00 | n/a | −5.97258, −6.21344; −8.16141, −8.41229, within 0.02, or stop | within a few hundredths |
| term, fine-tune | L01 − L00 in that precision | 0 | n/a | rand32b 6.011, 5.770; rand32 3.751, 3.500 (bf16) | within `tol` of bf16; it is the check that the fp32 read can see a change |
| gate gap | see above | above 0 (the paths differ) | n/a | above 0 at W₀, or stop | reported per state; E62 saw 0.832 on one mask entry |
| L01, student | the same on the rebuilt student | L00 | the fine-tune's | Amendment 2's within `repro_tol` 0.05, or stop | within a few hundredths |
| fall | fine-tune's L01 − student's, same precision | 0 by code | 0 by code | rand32b 1.225, 1.023, 1.159, 0.933; rand32 0.076, 0.084 | by the leans |
| `pair_diff` | fall in fp32 − fall in bf16 | 0 by code | 0 by code | n/a | within `tol` |
| `reload_diff` | student's L01 in bf16 − Amendment 2's stored | 0 by code | 0 by code | under 0.05 | under 0.01, a guess: no measured value |

## Predictions

Filed in full in `predictions_amend3.json`. Lines are Amendment 2's, read from its config, not copied. Absolute nats.

- **P1, rand32b.** Fall in fp32 on all four students. Under four states, a number not finite, or a floor not above 0: `unscorable`. `tol` ≥ 0.5: `no power`. All ≥ 0.5: `holds`. All |fall| ≤ 0.15: `gone` if `tol` < 0.15, else `no power`. Otherwise `mixed` (all four shrunk with a third remaining, or a rise, read `mixed`). *Lean: holds, medium-high, none moved*, since E62's forward alone agreed between the precisions to 0.002 (under Γ's bf16 masks; for Γ's own change E62 Amendment 5 gives 0.0008 and 0.0027 on one head's fall), though one mask entry moved by 0.83 there.
- **P2, rand32.** Fall in fp32 on both students. Same unscorable rule. `tol` ≥ 0.15: `no power`. All |fall| ≤ 0.15: `holds`. All ≥ 0.5: `falls`. Otherwise `mixed` (both states between the lines, or a rise on either, read `mixed`). *Lean: holds, medium-high*, since a move of about 0.07 would be needed to cross the line.
- **Beside both:** `moved`, the states with |`pair_diff`| > `tol`; and `term_moved`, the fine-tunes whose fp32 term differs from bf16's by more than `tol`. A tie has not moved, a nan has.
- **Reading.** The analysis quotes the size of any move, not only the `moved` label: `tol` is a size to compare with, not a noise floor (it is measured at W₀ with no dW, on a signed mean that can cancel, with a guard taken from another pair), and `moved` can fire at 0.03 nats on a fall of 1.2 beside a batch-order change of 0.09.
- **Blind to.** Whether a student or fine-tune **trained** in fp32 would fall: both trained under bf16 autocast (`finetune.py:60`), and this varies only the read. Other cards, windows and tables. A rounding that moves every read by the same amount, which a fall cancels (`term_moved` and L01 show it for the fine-tunes). The rounding of the offsets: the 0.05 stop bounds it in bf16 only; in fp32 it is unread and sits on the student side only. Arithmetic puts it well under the lines (noise on dW under 1% of dW; bf16 rounding of W₀ + dW about 14% of dW on rand32b and 25% on rand32, Amendment 2's hook pair put its effect on L01 at 0.0069 or less), but that is not a measurement. rand32 (P2) is where a move is more plausible.
- **Edge cases**, rerun after the review on the rule functions, `score()` and stage 0 with a stub job: 50, all as expected, adding "gone" with `tol` at and under 0.15, `term_moved`, a nan on either table, equal reads and a gate gap of exactly 0 (`~/commute-data/scratch/commute-lead/e12a3-fix/edge.py`). First filing: 32 (`e12a3-code/edge.py`).

## Decision tree

| outcome | means |
|---|---|
| P1 holds, none moved | The fall does not depend on the read's precision: the fall stands; L01 and the term as `term_moved` says. Its cause stays open. |
| P1 holds, some moved | The fall survives but its size depends on the read: quote fp32 numbers, and the user hears the size. |
| P1 mixed | Under the line on some states (all four may have shrunk with a third left, or one rose). Amendment 2's reading does not stand as written; nothing is built on the fall until the states that crossed are run down. |
| P1 gone | **The fall depends on the read's precision.** Read it against `term_moved` and the gate gaps before correcting any report; Amendments 1 and 2's readings of the fall are withdrawn until then; the user hears first. |
| P2 holds | rand32 still has no fall, so with P1 holds the difference between the tables stands. |
| P2 falls | **The difference between the tables depends on the read's precision:** rand32's students lose the term too in fp32, read against `term_moved`. The user hears first. |
| P2 mixed | One state crosses a line, both lie between the lines, or one rose. The table difference is not safe to state until it is run down. |
| no power | The floor is too large for that line. Goes to the lead; the floor's size is the finding. |
| unscorable, or a stop | A bug or a state that does not reproduce. Goes to the lead; nothing is read. |

Nothing further runs automatically in any branch.

## Cost and routing

Free: cluster `debug` QoS, partition `compute`, account `c11`, one L40 (the cluster's only card; the job stops on any other). From Amendment 2's `run_amend2.log`: model load 23 s (16:49:43 to 16:50:06), a fine-tune's two reads about 1 s, W₀ cells under 1 s. Here, 12 pairs of reads at 1 to 2 s, plus 6 rebuilds of offsets at a few seconds each: about 1.5 to 2 min. The fp32 read has not run on these windows, so `--time` is 15 min. Host memory: gates of one state are about 0.6 GB on the CPU, freed after each read; 32 GB is requested, as Amendment 2.

```
cd ~/commute-agents/commute-lead && sbatch cluster/new_vs_existing_amend3.sbatch
```

Writes `results/amend3.json` and `results/large/amend3/run_amend3.log`. Under the "scored run is its own smoke" rule, it has no smoke path. `amend2.py` points `amend1.OUT` at Amendment 2's scored file on import, so `amend3.py` sets it to its own after that import (printed correct in the edge run).

## Departures from the literature

The fp32 side departs from the authors' precision. This decomposition's `artifacts/vpd/s-55ea3f9b/final_config.yaml` has `autocast_bf16: true`, and `param_decomp`'s trainer and evaluator run under it. So Γ and the components were trained in bf16, and the bf16 side is the authors' setting. The fp32 read tests whether a reading depends on precision; it is not a truer value. Carried from Amendment 2: Γ sees 41-token windows where it trained on rows of 512.

## Not in this plan

Training a student or fine-tune in fp32. The fp32 side of the offsets' rounding (students are rebuilt from saved offsets, not retrained; retrain only a student that lands near a line). Mixed reads, fp32 Γ with a bf16 forward and the reverse, which would say which of the two carries a move: the first thing to add if one moves. Sets, restores or the ladder in fp32. Per-window falls. ident32, the copy tables and Amendment 1's `rand32_s0_o1`. A second W₀ pass to test repeatability: Amendment 2 measured differences of exactly 0 on this card.
