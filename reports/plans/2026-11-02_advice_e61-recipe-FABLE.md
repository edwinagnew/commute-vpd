# E61: advice on the recipe fix, the rate grid and option A (Fable, 2026-11-02)

*Advice to the lead, who decides. Sources: `reports/reviews/2026-11-02_review_qwen-finetune-recipes.md`, `experiments/qwen_recipe/results/A.json` (job 48367), `experiments/qwen_recipe/run.py`, `src/commute/qwen_sft.py`, `src/commute/finetune.py`, the E40/E44/E45/E57 log cards, `reports/decisions/2026-10-30_decision_e57-no-teeth.md`. Nothing run.*

## Recommendation
Train with a true fp32 forward (no autocast, no TF32), the same forward the readers use, and say so as a departure from standard bf16 mixed precision. Run a four-arm grid, about 2 h on the cluster at $0: a 9-minute control first (own, E45's recipe, 10 steps), then the template task at peaks 3e-6 and 1e-6, then gold at 1e-5, each saving its states. Start option A from the gold run's ladder, not from gold_s0 step 9: the first question is whether a gold state at the random changes' size (0.63-0.86) still reads three times random. The random arms scaled up to size 1.44 run only if that reading is in between. Invented facts at a lower peak (75 min) waits for A's verdict; if it runs, the programme passes 3 h and is gated.

## 1. The recipe fix
**fp32 forward is the right fix here.** The project's object is the small step, and every decomposition reading (`qwen_reading/lab.py` `put`, E44, E57) is fp32 with TF32 off (`qwen_vpd.py:73-75`). Under bf16 autocast the weights are rounded to a grid whose half-spacing is 6e-5 at the median entry; W0 sits exactly on it (0 of 1.4 B entries change), so an update under that size is invisible to training. Standard practice (HF Trainer `bf16=True`, LLaMA-Factory full SFT) is, from memory and not checked in the repo, the same autocast; it is a throughput choice for runs of thousands of steps, where five blind steps are nothing. Here they are the states we read.

Verified in the repo (A.json): `autocast` and `rounded` agree within 2e-3 at every state, so the effect is weight rounding, not activation precision (R1 confirmed both ways); step-100 states agree under all three readers (R2 excess 0.011, 0.0002, 0.005), so the destinations are usable; an fp32 gradient pass costs 14.7 s at micro 8 (24.8 GiB) and 14.0 s at micro 4 (20.9 GiB), without Adam's 10.5 GiB. Not verified: what the libraries actually do, whether fp32-forward training reaches the same step-100 state (the paths were blind, the destinations happen to agree), TF32's speed.

Alternatives, each with its cost:
- **Keep autocast, read states under `rounded`.** Consistent, $0, but the first 5-10 states then carry nothing readable (rounded δ keeps 0.15-0.4 of the norm, cosine 0.1-0.3). It deletes the object of study. No.
- **TF32 forward** (`qwen_teacher.tf32`). Faster, but TF32 rounds matmul inputs to 10 mantissa bits: half-spacing 7.6e-6 at |w| 0.022, still above own's δ at states 2-3 (RMS 1e-6 to 2.4e-6) and gold's state 3. A smaller dead zone, not none. No.
- **Two-matmul forward** x·W0 + x·δW with δW in bf16 (δ rounds relative to itself; the saved-δW check showed that exact). Near bf16 speed, but a custom Linear inside a checkpointed 1.7 B HF model. More code than the project wants. No.
- **A higher rate so step 1 clears the grid.** E40's mistake. No.
Cost of the fix: 2.6 times the step time (about 15 s against 5.6 s), micro 4 for memory (about 37 GiB with Adam and the fp32 W0 copy; micro 8 would sit near 41 of 44 GiB).

## 2. The smallest grid
Common settings, each sourced: batch 64, sequence 512, warm-up 10, cosine over 100, clip 1.0, AdamW (0.9, 0.999), no decay, as E45 (`qwen_updates/job.py` `Config`); fp32 forward (section 1); micro 4 (measured). Peaks: 1e-5 is the cited LLaMA-Factory value (no copy in the repo; treat as unsourced for 100 steps). 3e-6 and 1e-6 come from the review's measurement: at 1e-5 gold had 0.9 of its loss drop by state 9, where the summed rate was 4.5e-5; a 1e-6 peak sums to 5e-5 over 100 steps (the floor arm, the same total movement), 3e-6 to 1.5e-4 (the middle).

| arm | task, peak, steps | saves (bf16 δW, 2.8 GB each) | readouts | minutes |
|---|---|---|---|---|
| 0 control | own, 1e-5, 10 steps | every state | fp32 held-out (64 rows) and 16-row maths KL at every state, training loss and grad norm every step | 9 |
| 1 | a1, 3e-6, 100 | 10, 20, 50, 100 | held-out distill KL every 10, maths KL every 5, fp32 greedy generation at 20 and 100 on test 0-255 | 39 |
| 2 | a1, 1e-6, 100 | same | same | 39 |
| 3 | gold, 1e-5, 100 | 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100 | held-out gold loss every state to 10 then every 10, maths KL every 5, generation at 100 on 256 | 36 |

Minutes: 15 s a step, 15 s a held-out read, 5 s a maths KL, about 3 min a 256-problem fp32 generation (178 s measured), 3 min to load. Total about 2 h, 42 GB of states (4.3 TB free on `/mnt/nw`). Each arm passes the 30-minute `debug` cap: use `normal` QoS when the queue is empty (allowed for this line), else split an arm at step 50 with Adam's state saved (16 GB) and resumed.

**Cut and why.** a1 at 1e-5 under fp32: its bf16 step 100 is reader-robust and the lower peaks decide the rate; add it only if both fail. gold at a lower peak: the ladder of one 1e-5 run gives "smaller states of the same training" more cheaply. own at 3e-6 and 1e-6 (the review's arms): nothing in A or the user's request reads own; defer to option D's decision. a2long at 3e-6, 300 steps (about 85 min with recall generations): the task learned (recall 0.999) and its rate showed no fault; a lower peak matters only for B's size matching, so it waits for A. Not deferred to dodge the gate: with it the programme is about 3.5 h and gated; without it 2 h, $0, not gated. Write the Gates section anyway (pre-mortem, sourced settings above, the control as the rehearsal, early checks below, one fresh code review); it costs a page and the user then sees the record either way.

**Behaviour-based "learned" rules** (fp32 greedy, read at the arm's last state; 256 problems give a standard error of about 0.03):
- a1: unprompted template rate ≥ 0.74 (0.8 of the teacher's 0.926) and strict accuracy ≥ 0.60 (W0's 0.70 less 0.10). The lowest passing peak is the recipe.
- gold: held-out loss drop ≥ 1.58 (0.8 of the 1e-5 plateau's 1.98) and mean answer length ≤ 0.6 of W0's. No collateral rule: the short human style is the task.
- a2long, if run: invented recall ≥ 0.8; collateral, 4-shot maths accuracy ≥ 0.65 (W0 0.75 less 0.10).
- own, if run: paired strict accuracy gain ≥ 0.03 (se 0.010) and held-out loss at 100 ≤ W0's plus 0.005.

**The control and its early checks.** Arm 0 is E45's own_s0 recipe exactly; the one difference from the reference is the forward's precision, which is the thing it tests. The schedule's step 1 runs at rate 0 (`lr_at`, k = 0), so state 1 is W0. Filed first-minute values: W0 on own's rows 0.0566 and 0 entries changed (as stage A); training loss at step 2 0.055 ± 0.005; at step 3 (state 2, summed rate 1e-6) 0.060-0.075, against the bf16 run's 0.0551 and the fp32 record 0.0647; at step 4 (state 3) ≥ 0.12, the bf16-trained state 3 reading 0.217 in fp32. **Stop if step 4's loss is under 0.08**: training still cannot see its update. Then the fp32 held-out at state 3 within 0.05 of step 4's training loss (same model, other rows; the W0 row gap is 0.0015), and the gradient norm above 5 by step 4 (bf16 run: 3.6 until step 8). Per arm: a1 step-2 loss 0.96-0.98 (W0 on a1 rows 0.974), at 3e-6 under 0.7 by step 6, stop if over 0.9 at step 10; gold step-2 loss 2.41 ± 0.05, step 4 under 1.6 (bf16-trained state 3 read 1.307 fp32), stop if over 2.2. Memory: stop if the first step's peak allocation passes 40 GiB.

## 3. Option A after the grid
**States.** "The biggest" is the fp32 gold run's step 100; gold_s0's bf16 step 100 (size 1.44, direct share 0.092 [0.090, 0.093]) is reader-robust and stays as a second sample of the same recipe, so the two give a free spread. "A smaller state of the same training" is the fp32 gold ladder state whose size lands within 15% of a random change's (0.63 or 0.86 on the card's yardstick, the CI-masked KL less W0's), expected around steps 7-15 given gold_s0's state 9 at maths KL 0.31; the dense saves 5-15 are for this. gold_s0 step 9 is unfit: it was produced by five blind steps and then four seeing ones, a path no recipe will repeat.

**Order.** Nothing of A needs to run before the grid: its cheapest and decisive reading is the in-band gold state against the two random changes already read with bootstraps, one 35-minute read after the gold arm. The scaled-up random arms (one kind, one seed at 1.44, 35 min; a second kind if asked) run only if that reading is in between. They could run now against 1.44 without harm, but they answer the costlier half of the question first.

**Filed line, at matched size (gold state within 15% of a random change's size, the random nearest in size as the reference; floor 0.01 = the 0.007 gap between the two random kinds plus bootstrap half-widths of 0.002-0.003):**
- gold's direct share ≤ random's + 0.01: size explains the direct share → C.
- gold's ≥ random's + 0.03 (three floors; the unmatched gap is 0.06-0.07): training differs beyond size → B.
- In between: run one random change scaled to gold step 100's size and apply the same two lines against gold's step-100 share; still in between → C, with the note that the remainder is under three floors and B's own limit (invented facts at 0.012) would bite.
Edge cases: no ladder state within 15% of either random size → read the two states bracketing it and interpolate linearly in size; the ladder brackets neither (every saved state above 0.86) → go straight to the scaled-up random arm; the in-band state has not learned by gold's rule → say so and use the scaled-up arm instead, since an unlearned state is a few sign steps; the fp32 and bf16 step-100 shares differ by more than 0.01 → that difference replaces the floor; NaN or infinite KL → no reading; a negative direct share reads as 0; a bootstrap interval crossing a line → the point estimate decides and the report says "at the line". The two random kinds disagreeing across a line → the higher random decides (toward C).

## 4. Earlier claims touched
Checked in the code and cards:
- **E40 step 1** (1.57 nats of gold drop, maths KL 0.106). Adam's first step uses the gradient at W0 whatever the forward's precision; the state is what it claims. Its gradient was computed with bf16 activations, a precision difference and not blindness. Stands.
- **E40 step 10** (no warm-up, 1e-5 constant: δ grows by about 1e-5 a step, so the forward saw it from about step 6). E44's no-teeth verdict compares placements of one fixed δW and stands; E57's E40 row (size 1.30, direct share 0.063) is a reading of a real fp32 state and stands as a reading. What changes is the description: "ten steps of fine-tuning" becomes "ten steps, the first five or so blind". Review note, no rerun.
- **E45's matching at low KL.** Size was read in fp32; the "learned" flags at steps 2-12 were bf16 generation on rounded weights (16-21% of entries changed at state 9), and fp32 generation on an early state can differ wildly (a1 step 8: accuracy 0.133 against 0.703). The anchor rows were never scored and B retrains, so a review note is enough. The a2 "step 12, recall 2%" line should carry it.
- **E61 stage A** is the measurement itself; nothing to amend.
- **67M and minGPT lines.** `finetune.py:60` and `pcfg_train.py:76` use the same autocast over fp32 weights, but the 67M target checkpoint is fp32 (`artifacts/vpd/t-9d2b8f02/ckpt/model_step_99999.pt`: `FloatStorage`, 261 MB), so W0 sits off the bf16 grid. Inferred, not measured: rounding W0 + δ then flips each entry with probability about |δ| over the spacing, unbiased on average, so there is no dead zone, only noise, and at `ArmAConfig`'s 3e-5 the first step already moves a quarter of the entries. Stands; one line in the review note. Not checked: the 67M's weight magnitudes, the flip fraction, the minGPT checkpoint's dtype, teacher training's U and V steps under TF32 (`qwen_train.py`).
Needs rerunning: nothing on these grounds. The own and null runs need retraining on the review's grounds (rate), and the grid is where the recipe gets fixed first.

## What would change it
- If the control's step-4 loss stays under 0.08 under an fp32 forward, the hiding is not in the Linear cast and the fix is wrong; stop and look at where the weights are rounded (gradient checkpointing's recompute, the embedding path).
- If TF32 measured at least twice as fast and gold's states 1-3 were not needed, TF32 would do for the grid, not for A's ladder.
- If the fp32 gold ladder has no state in the random changes' band and its scaled-up random arm reads within 0.01 of gold step 100, A is answered toward C without the grid's other arms mattering.
- If the user wants the invented-facts recipe settled with the template's, run a2long at 3e-6 in the same programme and file it as gated from the start.
