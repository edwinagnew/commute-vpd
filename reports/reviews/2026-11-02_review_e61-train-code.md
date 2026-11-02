# Review: E61 Job R training code (2026-11-02)

Scope: commit 8b82fa6 (`experiments/qwen_recipe/train.py`, `results/predictions_train.json`,
`cluster/qwen_recipe_train.sbatch`, `src/commute/qwen_sft.py`, `qwen_train.py`), against Part 1 of
`reports/plans/2026-11-02_plan_e61-direct-share-size.md` with Amendment 2 overriding. The reviewer wrote none of it.

**Verdict: no must-fix; Job R can start with arm 0.** I found no path to a wrong filed number and no failure after hours
of GPU time. The three should-fixes are guards weaker than the docstring claims.

## Must-fix
None.

## Should-fix
1. **`train.py:152`: an arm can start before Stage W has finished.** The docstring (line 9) says each stage refuses until
   the one before has finished, but the guard only tests that `lines.json` exists, and `stage_W` writes it (:268) before
   reading the references. Scenario: W's first job exits 3 after the W0 reads and an arm is submitted; a later reference
   stop ("the saved file is the state it claims") then fires with the arm already training. Fix:
   `if name != "arm0" and not read("train_W").get("done")`.
2. **`train.py:245`, :251, :255: a stopped reading is kept, and a rerun skips its check.** `write` runs before `check`, so
   a missed reading stays in `st`; once `stopped.json` is removed, `if kind in st: continue` never re-reads it. Scenario A:
   `w0.prompted` misses its band, the stop is cleared, and the out-of-band value becomes the template line (:267-271).
   Scenario B: `w0.four_shot.peak_gib` passes 38.5, the filed remedy is to halve the batch, but the W0 read is then skipped,
   so the first test of the new peak is the reference at W (expected 41 of 44 GiB). Fix: in `check`, on a stop's miss,
   delete the reading from `st` before `write` and `stop`.
3. **`train.py:317-320`: Stage G never ties a state file to the run that saved it.** Stage W checks E45's files against
   E45's record; an arm's own files get no such check, though `rec["held"]` is on record for every saved state
   (`Run.__post_init__`). Scenario: a lost job and a mixed-up file give a verdict from the wrong state, unflagged, and Job S
   reads the same files. Fix: at each held read in `read_state`, compare with `out["steps"][t-1]["held"]`; log it, and stop
   beyond a tolerance taken from the first arm's measured gap (bf16 rounding of δW is under 0.4% per entry).

## Note
- **`predictions_train.json:50`, :63: the a1 step-10 stops rest on batch 1's 0.974.** E45's first four batches read
  0.964-1.108 and batch 10's W0 loss is unrecorded. A live run on a heavy batch could stop falsely (about 3 min lost); a
  dead run on a light batch could pass at 1e-6. Arm 0 catches blindness and `dw_norm` a dead optimiser: acceptable.
- **`train.py:173`, :180: the comment overstates `peak_gib`.** Peak stats are never reset, so for arm 0 and gold (held read
  at step 1) step 2's value also holds that read: 27.7 GiB plus about 5. It cannot reach the 40 stop but can miss the 33.5
  note; read such a note with this in mind.
- **`train.py:176`, :251, :138: `checks` rows.** `write` precedes `check`, so an exit 3 straight after a read loses that
  read's ok and note rows from the JSON (the log keeps them). After a lost job, rows past the checkpoint stay and repeat,
  because `start_or_resume` cuts `steps` and `saves` but not `checks`.
- **`sbatch:9`: Stage W may take 3 jobs, not 2.** `EST_S` over W sums to about 44 min against 25 per job, and `budget` uses
  the slowest unit so far. The runner should resubmit until exit 0.
- **Plan Gate 1 differs from the code on two counts.** Saves: 82 GB in the plan, 107 GB filed (38 states at 2.82 GB), plus
  17 GB per checkpoint; 5.1 TB is free. Rows: "E45 row hashes checked", but `train.py:157` compares counts, and no hashes
  are on file; the steps 1-2 loss bands are the row check, as `.data.blind_to` says.

## Cleared
- **Resume, walked for $0.** A toy CPU model through the real `stage_T`, `train`, `checkpoint`, `resume` and `torch.load`
  (mmap, weights_only) in arm 0's shape (10 steps, warm-up 10, pause at 5, fused AdamW): exit 3, then exit 0. Every loss and
  |δW| matched an uninterrupted run; summed rate 4.5 × peak. Script:
  `~/commute-data/scratch/qwen-decomp-lead/train-review/dry_t.py`. It does not cover the GPU or the 17 GB size; E45's a2long
  resume does.
- **Filed numbers recomputed from `T_*`, `E_*`.** All loss, `dw_norm` and grad-norm bands; `.data` of all four arms (exact);
  W0 and reference held-out bands; `kl16` and `dw_norm` notes; the sum table (7.2e-4 at state 149, 9.0e-4 at 300).
- **Keys.** Every filed key and `verdict` field is produced by its read; labels match; a missing key fails (`in_band`).
- **fp32 forward.** The `nullcontext` branch is right and the default unchanged; other `sft_step` and `Recipe` callers are
  safe. `fp32_matmuls` runs in `Job.__init__`, off for matmul and cuDNN. `grad_pass` already ran this forward on the card.
  `tests/test_qwen_sft_forward.py` 6 passed; ruff and pyright clean.
- **Schedule and batches.** `steps == warmup` (arm 0) neither divides by zero nor changes rates 1-10. The stream comes from
  E45's full-length arm, so a short arm's batches are E45's.
- **Resolution at 1e-6.** The first real update is 1e-7 per entry; fp32 spacing at |w| = 0.02 is 2e-9, so 2% noise, far
  inside the 20% `dw_norm` band. The bf16 δW error is relative, so it adds no floor.
- **Reads during training.** `mean_kl`, `distill_kl`, `gold_loss` run under `no_grad`, restore `.data` by pointer, and with
  `None` read the weights as they stand. Each reader has one key, so `next(iter(...))` is the right number.
- **Prompts.** `a1_text(q, "format", [])` is P's prompt (`candidate` ignores shots unless "fewshot"). The four-shot read
  matches `check_format.py`; `strict_all` is its `strict`.
- **GPU memory.** Moments + grads + W0 + model = 27.7 GiB. No batch passes a2's full rows (4 × 512, expected 34, unmeasured),
  so no late out-of-memory on 100-step arms, which have no checkpoint. Four-shot holds about 36.6 GiB at W0, about 41.8 at
  W, of about 44; 42.43 GiB has run on this card.
- **Four-shot time.** fp32 ran 2.8 times bf16 at equal batch (`A.json` 178 s against 63 s), but half batch halves each
  batch's time: about 420-650 s from bf16's 202-310 s, under `EST_S` 800 and the 30-min kill.
- **Cluster.** One card type (`gpu:l40:8`), so `norms == c["norms"]` holds across a resubmit; `PreemptMode=OFF`; no stale
  `stopped.json`; `checks.py` has its own stop file; the references' dW files exist; 64G against 19 GB measured.

## Not run
Any GPU stage. Stage W, Stage G and `verdict` were read, not executed (the toy reaches only Stage T); Stage W exercises the
generation paths on the card before any arm. `read_prompted` and `facts_eval` have not yet run on the fp32 model.
