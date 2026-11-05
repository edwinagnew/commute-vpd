# Plan: three cheap comparators beside the D3 rows on Qwen3-1.7B (E92)

*2026-11-05, baseline-lead, the confirmed row of `reports/plans/2026-11-05_plan_qwen-reorient-amend1.md`. $0, one cluster L40, under the gate: no review. Rules: `experiments/qwen_comparators/results/predictions.json`.*

## In plain words
D3 asks whether a decomposition can tell an update that reuses what the model has from one that adds something new. This run reads what cheap methods say on the same weights: how large the update is, how highly the unchanged model already ranked what it trains, and how much of the gain one steering vector recovers.

## Design check
- **Goal.** `GOALS.md`'s question of new against existing; plan line "a cheap non-VPD comparator beside every row".
- **Nearest tried.** E52 and E54 (`steering.py`, 67M only, one answer token). E91 for size. New here: Qwen, multi-token answers, the rank read.
- **Patch or step.** A step: a plan row. Approved class 4. Departs from no decision.
- **What changes next.** A D3 row whose separation a comparator matches at the matched step says less than its number suggests; a row none matches is what the write-up leans on.
- **Control first.** Anchor 1's step 1 file (all zeros) and step 100, alone, first job only. Fails on a state left applied, a wrong file, a broken hook, wrong rows.
- **Saved.** One JSON per state (`results/states/`, committed); rank and log P per answer position (`results/large/`, about 80 KB to 130 KB a state). W0's the same.

## The three, as built
`src/commute/qwen_comparators.py`, called by `experiments/qwen_comparators/run.py`.

1. **Update size.** `norm` (root of summed squares), `ratio_norm` (a norm over a norm), `share_squared` (its square), per block and total, on the 196 matrices. Names carry the unit (friction 478). E91's `read_state` has no size function apart from its KL, so the total is checked against its stored read of the same file, within 1e-9.
2. **Rank of the trained answer.** W0 and W0 + δW, rank among the whole vocabulary and log P of the next token at each answer position. Summaries: first answer token, teacher-forced mean over the answer.
3. **One steering vector.** `steering.py` does not carry (`Net`, `N_LAYERS = 4`, one position), so the definition is carried. Fit half the even rows, scoring half the odd. v = mean of (state − W0) block input over the fit half's answer positions, added to W0 at every answer position. **Layer:** measured per state, largest fit-half share of the teacher-forced gain, ties to the lower, as `b3`. **Departure:** every answer position, not one, because a first token alone can carry no gain (TOFU's opening is often "The"); the first-token share is read from the same forward.

**Random control.** Same norm, a Gaussian mix of W0's centred inputs at that layer. **Departure:** not isotropic, which would read 0 by construction. Matched on norm; both variance shares are saved and reported (friction 407).

**Rows.** Anchor 1: E57's 64 x 256, held-out problems (checked identical). Anchor 2: E86's 128 x 256, the only TOFU table rows that exist (checked identical to its selection), which are trained rows, the 13 E84 dropped marked as no answer. **GSM8K arms: a departure.** The D3 maths rows are W0's own greedy answers (`qwen_sft/run.py`, `stage_G`), so W0 ranks them first by construction; gold solutions on the same test problems, 103 x 256.

## Numbers and what each should read
| number | W0 | do-nothing (zero state) | control (anchor 1, step 100) |
|---|---|---|---|
| `ratio_norm`, `share_squared` | not read | 0, 0 | 3.71e-4, 1.38e-7 |
| Median rank, first / all | a read per row set | equal to W0 | first falls to 1, lean high |
| Mean log P, first / all | a read per row set | equal to W0 within 1e-5 | gain positive and defined: **stop if not** |
| Steering share | not read | undefined, no layer | 0.05 to 0.4, lean medium |
| Random share | not read | undefined | about 0, under 0.05, lean medium-high |

Leans for the arms are in `predictions.json`. Anchor 2's full update reads ratio 1.38e-2, about 37 times anchor 1's. Share undefined rule, filed: the gain must be further from 0 than twice its row standard error, with two rows at least. Edge cases: `tests/test_qwen_comparators.py`.

## What each is blind to
- **Size:** direction.
- **Rank:** on anchor 2, trained wording only. On GSM8K, an accuracy gain that does not move gold tokens: expect little there, R no different from N.
- **Steering:** input-dependent change; a low share does not say the mechanism is new. A share at a size-matched anchor 2 state may be undefined because that state learned nothing (E91's second stage leans so).

**Column rule, the amendment's stop rule:** a column counts only if the anchors' gap exceeds the larger same-anchor seed spread. W0's rank does not depend on the step.

## Run
```
sbatch cluster/qwen_comparators.sbatch --states anchor1_s0:100
sbatch cluster/qwen_comparators.sbatch --states anchor1_s1:100,anchor2_s0:1250,anchor2_s142:1250,anchor2_s0:<t>,anchor2_s142:<t>
sbatch cluster/qwen_comparators.sbatch --states R_s0:10,R_s0:100,N_s0:10,N_s0:100,F_s0:10,F_s0:100,R_s70:10,R_s70:100,N_s70:10,N_s70:100
```
`<t>` from E91's second stage. Debug QoS. **Time.** `experiments/qwen_size_grid/results/run.log` lines 7-8: 16,384 tokens in at most 4.8 s (file cached, one row at a time), about 3,400 tokens a second; lines 5-6: 24 s a state with the file over NFS. A state is one pass over all rows, 30 over the fit half (the state, W0, 28 layers) and 2 over the scoring half: 278,000 tokens on anchor 1 (about 1.8 min), 452,000 on gold (2.6), 557,000 on TOFU (3.1). The controls take about 4 min, anchor 1's step 100 among them. The three lines above: 4, 15 and 26 min. **Memory.** `--mem=24G` as E91's sbatch, peak host RSS 10.5 GB (`cluster/logs/commute-qwen_size_grid-52219.out`). Card: under 30 GB by count (model, δW, W0 + δW, logits, inputs).

## States
Present and read by E91 (`reads.jsonl`) with their `file` fields: all listed. Sums: R, N, F and anchor 2's first seed hold records, checked (hand-back gives the result); anchor 1's two seeds and anchor 2's second have no sha record in the repo, so E91's norm is their identity. F's second seed: absent, as the plan knows. Anchor 2's matched states: under `qwen_size_grid/results/large/dW/a2_s<seed>/`, being written by E91's second stage.

## Not run
Nothing is run by this commit. CPU only: the rules' tests, the row builders on the real tokenizer and data, and the read and steer path on a tiny random Qwen3 of 28 layers. Not built: u1 and u2 (not D3 arms), a held-out wording for anchor 2, the D3 maths rows for the rank read, and the 11-12 table.
