# E55 Stage A code review (commit 855c5e9)

Adversarial review of `src/commute/pcfg.py`, `mingpt.py`, `pcfg_train.py`, `experiments/pcfg_anchors/run.py`,
`tests/test_pcfg.py` and `cluster/pcfg_anchors.sbatch`, read against Jain et al. 2024 (L1023-1196) and the Stage A plan.
I found no data or label bug. The grammar, the labels, the layout and where the answer is read all match the paper
(checked line by line, plus CPU measurements on 20-40K texts). The main risk is that count-a accuracy carries less
information than it seems: one answer covers 35% of the labels, and a window off by one token still scores 67%.

## Checked and correct
- Grammar: all 32 rules in `RULES` match Fig. 14. Every leaf sits at depth 6, and the token frequencies are 1/3 each.
- Layout: `$ T O O' < text(250) < = Ans $`, SEQ 259 (Fig. 15 without the `#` padding). The model input is
  `seq[:, :-1]` (block 258). The answer at position 257 is predicted from `=` at 256, which is
  `answer_accuracy`'s `[:, -1]` on `seq[:, :ANS]`. No label token reaches the input: the O' token is the constant 6 or 40.
- Labels: C counts only the last 40. CC counts overlapping pairs whose two tokens both lie in the window. I counts back
  from EOT, and its natural labels (11-26) fit Fig. 16's 0-40 axis; counting from SOT would give about 230. T is the
  token at column 210, distance 40. The `T` family puts NULL at the operand position.
- Priors: LOW never has c as the operand and NEVER never has a (both tested). `rng.choice(p=...)` uses the same number
  of draws for any p, so all three priors see identical text and family streams, a paired design. Recording this in
  the plan as a feature would help.
- Training: AdamW with betas (0.9, 0.95) and weight decay 0.1 on Linear weights only, clip 1.0, warm-up over the first
  20% then cosine to lr/100, as in minGPT and L1164-1196. minGPT's init, dropout and c_proj scaling are reproduced.
- Resume: batch `step` depends only on `(seed, step)`. The checkpoint holds the model, the AdamW state (with its step),
  the step, and the CPU and CUDA RNG states. lr is reset from `step` on every iteration. Saving goes through `.tmp`
  then `replace`, which is atomic, so a job killed mid-save keeps the previous file. No shared state between threads:
  `_chunks` is touched only by the main thread.
- `pytest tests/test_pcfg.py`: 13 passed (cpu-run).

## Risks (ranked)
1. **Count-a accuracy has a high floor and is blunt** (`pcfg.py:103`, `run.py:86`). Measured on 20K natural texts:
   count(a) in the window takes only 9 values (9-17), the label 13 alone covers 35.2%, and 13 and 14 together cover 66%.
   count(b) and count(c) look the same. A model whose window is 41 or 39 tokens instead of 40 scores 67%. A model that
   answers count(a) when asked to count b scores 21% on count-b (P(nb = na) = 0.21). Scenario: LOW's count-b at ~35%
   reads as "partial capability" when it is the constant 13. Fix: next to every accuracy in `a1_behaviour.json`, report
   the majority-class rate on the same samples, the ±1-window baseline and the "answer count(a)" baseline for count-b.
   Also report the mean absolute error and the confusion over classes 9-17. The plan's stop rules should be stated
   against these floors, not against 0.
2. **The paper's labels may be spread wider than ours** (`pcfg.py:7-9`). Fig. 16 calls the class distribution "fairly
   spread across multiple classes" on a 0-40 axis. With this grammar, every d/e/f leaf holds roughly one of each of
   a, b and c, so window counts are 13 ± 1. Every faithful transcription gives this, and the extracted text cannot show
   the paper's histogram to compare. Any number the plan takes from the paper's tables (reproduction gate) may
   therefore have a different floor. Fix: state this in the plan beside the gate; no code change.
3. **Model initialisation is not seeded** (`run.py:78`, `pcfg_train.py`). `GPT(...)` runs before any `torch.manual_seed`.
   Scenario: a crash before step 1000's save, or a later "rerun LOW", gives a different anchor with nothing to show
   which one it is. The batches are seeded and the weights are not. Fix: `torch.manual_seed(TRAIN_SEED)` (or a seed per
   prior) before `GPT(...)` in `pretrain`, written to the checkpoint and the results JSON.
4. **Writing `a1_behaviour.json` is not safe across parallel jobs** (`run.py:134,143`). The read-modify-write leaves a
   window in which two jobs finishing together drop a row. `write_text` is not atomic either, so a job starting while
   another writes can read a half-written file, raise `JSONDecodeError` in `_read`, and die with exit 1, not 3. A
   resubmit loop would then stop. Fix: one file per prior (`a1_behaviour_<prior>.json`), each written to `.tmp` then
   `replace`d.
5. **The rehearsal does not test that a resume reproduces the run** (`run.py:116-118`). a0 asserts only the steps
   reached (250, 500), so a resume that silently reset the optimizer or reused chunk 0 would pass. Fix (once risk 3's
   seeding is in): for one prior in a0, also train 500 steps uninterrupted and compare the losses logged at steps
   300-500 and the final parameters. Dropout under `torch.compile` may not be bitwise reproducible; take the tolerance
   from the gap between two uninterrupted runs.
6. **The a1 deadline loop is incomplete** (`run.py:129-144`, `cluster/pcfg_anchors.sbatch:22`). No a1 sbatch and no
   resubmit-on-exit-3 driver exist yet; the sbatch refuses every argument except `a0`. All priors in one job share one
   deadline. When the first prior finishes near minute 25, the next one builds, compiles (about 1 min), runs one step
   and saves. That is harmless but wasted. Fix: check the deadline before starting each prior. Write the a1 sbatch
   with one job per prior, `--deadline-min` at most the time limit minus compile, behaviour eval and saving, and
   exit-3 handling in the driver.
7. **The spurious labels sit one class above the natural ones** (`pcfg.py:163-182`, documented). Natural count-b
   peaks at 13-14 and spurious count-b at 14-15 (15 alone covers 38.8%). For Stage A2, a fine-tune at C_Tr = 1 can gain
   C_Te = 1 accuracy just by shifting its label prior. Fix: when A2 reads C_Te = 1 numbers, report each test set's
   constant-predictor rate.

## Nits
- `pcfg_train.py:157`: `answer_accuracy` takes the argmax of bf16 logits (resolution about 0.06 near 10). Ties go to
  the lowest token, that is the smaller number. The effect is rare but systematic. Fix: run the evaluation forward
  without autocast (the model is tiny).
- `pcfg.py:9`: "about a third" of expansions are shorter than 250 and redrawn; the measured share is 24%. Redrawing
  keeps only long expansions, which slightly over-represents the longer rules (`p→mno`, `r→mon` ...). The paper is
  silent on this; the docstring should say it.
- `pcfg_train.py:56`: the `ThreadPoolExecutor` is never shut down. At exit the interpreter waits for the chunk being
  drawn, a few seconds, and a0 leaves six idle pools behind. Fix: `shutdown(cancel_futures=True)` at the end of `train`.
- `pcfg_train.py:123`: `torch.compile` runs on every `train` call, so the first log window of each job includes
  compile time. a0's ms/step comes from the last window and is unaffected; a1 pays about 1 min per resubmit.

## Not run
GPU execution, the a0 log (still running), bitwise determinism of a resume, and a comparison with the paper's actual
Fig. 16 histogram (the figure is not in the extracted text).
