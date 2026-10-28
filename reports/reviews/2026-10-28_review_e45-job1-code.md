# Review: E45 Job 1 code (qwen_updates), independent adversarial pass

Scope: `experiments/qwen_updates/{run.py,job.py,RUN_BRIEF.md,results/predictions.json}`, `cluster/qwen_updates.sbatch`,
`src/commute/qwen_{train,updates,facts,anchors}.py` (commits 7cff2ba, 81ba33a), read against
`reports/plans/2026-10-28_plan_qwen-q2.md` §3-§6 and §8. I read the code; I did not run it. The reviewer did not write
the code or the plan.

In plain words: I found no bug that makes a number from the planned run order wrong. The recipe, the data split, the
masking, how the update is saved, the direction of the KL and the per-seed data draws all match the plan. Three
problems can make a later job wrong or stop it for no reason: a rerun of a training job under an old name reuses stale
readings, a fixed early-KL threshold is not scaled for the 2e-5 rerun, and changing the sampling batch size partway
through a set breaks the resume.

## Checked and correct (no action)

- **Recipe.** AdamW (0.9, 0.999), eps default, wd 0, fused; peak per `ARMS` (1e-5; `own2` 2e-5); HF
  `get_cosine_schedule_with_warmup(10, 100)`, asserted every step against `lr_at` (step 1 at rate 0); clip 1.0;
  fp32 master copies of the 196 SITES only, bf16 autocast forward; batch 64, 100 steps; loss = mean over response
  tokens of the whole batch (`sft_step` divides each micro-batch's sum by the batch total). Matches §3.
- **Masking.** `targets`: an int prompt length masks logit positions `p-1 .. len-2`, so only response tokens (and the
  end token) are targets; packed rows use per-token flags shifted by one, first flag asserted False. `pack_whole` flags
  each fact's `"The answer is X." + <|im_end|>`. Training chat (`chat`/`prompt`, thinking off, generation prompt with
  the empty think block) is the same text the evaluations generate from.
- **Split and leakage.** `pool` is drawn from `train_idx` (E40's 256 held excluded); `problems`, `search`, `shot_pool`
  are disjoint slices; held-out own and anchor-1 samples (pid ≥ 1936) are filtered out of training; accuracy uses
  test 0-255 and anchor 1's own rows use test 256-511; `qwen_sets` `idx` is the row index, so `sets["train"][r["idx"]]`
  is the problem a sample was drawn for.
- **δW.** `delta` = fp32 param − fp32 W0, per site, cast to bf16 on the host; written atomically (`.tmp` then
  `os.replace`), non-droppable. Stage E reads it as `plus(W0, δW)` (fp32 W0 + bf16 δW), as §3 says.
- **KL.** `mean_kl(model, rows, W0, None/W)` = KL(W0 ‖ W_t) per position, fp32 logits, TF32 off (`fp32_matmuls`),
  mean over every position of the 512-token rows. Direction and rows match predictions.json "KL to start".
- **Brackets.** `train` saves the read step at a first crossing and the previous read step (kept on the host while any
  grid value is open); step 100 always saved; `choose` takes the nearer in |log KL| with ties to the later step, from
  the all-row KL in stage E. For own/null the crossings are copied to the `own` axis because own rows = maths rows.
- **Seeds.** a1 seed s trains on samples k ∈ [4s, 4s+4) with `cycle(…, seed)`; a2 seed 1 is `invented(1)` (new names
  and attributes) packed with `pack-1-p` orders; the W0 facts check covers both draws.
- **Early checks.** Step 1 exact zero (lr 0 through fused AdamW adds −0·x; same values in both forwards); step-2 band
  upper bound 1.0014 is Adam's step-2 analytic maximum; gold curve and step-1 loss tolerances are sensible; control
  already passed (KL 0.106789 against 0.106790; invented 0.015 / 0.008).
- **Rules.** `prompt_rule`, `paired_gain`, `length_ratio`, `anchor2_holds`, `learned` treat empty pools, None and NaN as
  misses; I found no zero denominator reachable in the planned order.

## Findings

### 1. Stage E reuses stale readings if a T run is repeated under the same name — blocks-results (conditional)
`run.py:311-318` (`stage_E`): `out = job.get(f"E_{run}")`, and every state already in `out["states"]` (keyed by
step) is skipped; `out["gen"]` likewise (`run.py:335-336`). Nothing ties `E_<run>.json` to the T run that produced the
δW files. `stage_T` (`run.py:236-263`) never clears `E_<run>.json` or `large/dW/<run>/`.
**Consequence.** Any rerun of T under an existing name after its E has run, such as plan §8's anchor-2 amendment
(300 steps) if it keeps the run name `a2_s0`, or a gold rerun after a fix, reads the old KL, task scores and
generations for every step number that repeats (always step 100, so `anchor2_learns` and the `learned` flags come
from the old run). The JSON gives no sign of it. `T_<run>.json` is rewritten, so the files then disagree with no warning.
**Fix.** At the start of `stage_T`, delete `results/E_<run>.json`, `large/E/<run>/` and `large/dW/<run>/`, or record a
T identity (e.g. T's start time or `recipe`+`steps`) in E's JSON and discard `states`/`gen` when it differs. Give
any amendment a new run name (as `own2` does).

### 2. `step2_kl_max` (0.005) is not scaled with the rate, so `own2` can be stopped for no reason — wastes-time
`run.py:224-229` applies `E["step2_kl_max"]` to every arm. The value comes from gold at step-2 rate 1e-6 (KL 0.0037).
`own2` runs at peak 2e-5, so its step-2 rate is 2e-6. KL grows roughly with the square of the step, so a step-2 KL of
0.002 or more at 1e-5 becomes 0.008 or more at 2e-5. That would stop the plan's own-samples rerun on its first
minute, although nothing is wrong with it.
**Fix.** Compare against `step2_kl_max × (peak / 1e-5)²` (same rule for every arm), or exempt `own2` and state it in a
dated note beside predictions.json (the early values are apparatus checks, not predictions).

### 3. Sampling resume breaks if `per` changes between submissions — wastes-time (silent data skew)
`run.py:171-196` (`stage_S`): `per = per_batch(...)` is recomputed each submission from `CFG.kv_tokens`, and batch
`b` covers prompts `[b·per, (b+1)·per)`. RUN_BRIEF row 6 says to lower `kv_tokens` on an out-of-memory error. If that
happens after some batch files exist (likely, because `per` is sized by the longest prompt but a batch only fills
memory when it contains that prompt), the old files are skipped by name and the new batches use the new `per`. Some
problems are then sampled twice (their records are not identical, so `keep_correct` keeps both) and others never.
Records stay internally correct (idx/question match), and the held split by pid stays right, so nothing leaks, but the
own and a1 data are no longer 4 or 8 samples per problem, and nothing logs it.
**Fix.** Write `per` into `results/S_<set>.json` (or a `meta.json` in `large/S/<set>/`) at the first batch and reuse
it on resume; if it must change, delete the set's batch files.

### 4. Gold step-1 loss source mislabelled — minor
`predictions.json` `early.gold_step1_loss.source` says "micro 8", but the E40a1 rehearsal ran F1 with `micro 4`
(`amend1_rehearse_1/run.log` config: `"micro": 4`; `amend1.py:127` `A1_TINY` sets `micro=4`). T uses `CFG.micro = 8`.
The loss is micro-independent up to bf16 rounding and the tolerance is 1%, so the check is still right. Also the gold
curve reference was taken at micro 4 (5% tolerance, fine). Fix: a dated note, not an edit to predictions.json.

### 5. W0 invented-fact check sits on chance guessing, not on name collisions — minor (already passed)
`run.py:101-104`, `predictions.json` `invented_max` 0.02. Professions (30), instruments (20) and birth years (70) are
small closed sets, so guessing alone gives about 0.01-0.02. Measured: 0.015 (draw 0) and 0.008 (draw 1). The check
passed but had little margin, and it does not detect collisions. For reading anchor 2: its floor is about 0.015, not
0. That is far below the 0.25 / 0.5 thresholds, so no rule changes.

### 6. Step-2 band lower bound (0.3) is measured on gold only — minor (wastes-time risk)
`run.py:225-228`. The ratio is a per-matrix minimum over 196 matrices, and its lower edge depends on how much the
gradients of steps 1 and 2 agree. With unrelated gradients it is about 0.71 on average, and it is lower for matrices
whose two gradients point in opposite directions. Packed facts (a2) and own samples have different batch-to-batch
correlation than gold. A miss on a non-gold arm should be read as "band not transferable" before anyone suspects the
code. Note it in RUN_BRIEF.

### 7. `score` scores only `own_s0` — minor
`run.py:358`. If plan §8's 2e-5 rerun (`own2`) happens, `own_accuracy` and `own_length` still read `own_s0`. The plan
does not say whether the rerun replaces the arm for the behaviour predictions. Decide it before E own2, and record
which run the score reads.

### 8. W0 tracker log-probabilities are not cached — minor (time only)
`qwen_train.py:85` calls `mean_kl` at about 44 reads, each running W0's forward again on 32 rows. Plan §3 says these
are cached on the host. This costs about 3 min per T run and changes no numbers. Also, `delta()` copies all of δW to
the host at every read while any grid value is open (`qwen_train.py:88`). For a2, whose maths axis may never reach
0.3, that is all 44 reads. This is time only, well inside 30 min by my estimate (4 s/step from E40a1 F1 + ~6 s per
read + saves ≈ 15 min).

### 9. RUN_BRIEF expected `rows_per_step` ~30-40 for a2 — minor
With the measured `gold_mean_tokens` 205.9 and packed rows of ~490 tokens, `per` ≈ 27. A runner following the brief
may report this as a surprise. Update the line.

## Not run
- No code executed, no model loaded, no tests run (`tests/test_qwen_updates.py` not read in detail).
- Did not check `gsm8k.grade`/`gold_number`, `gold_as_response`, `qwen_vpd.SITES` contents, or HF generate's
  memory at the S batch sizes (`kv_tokens` 262,144).
- Did not check that the Qwen3 chat template renders few-shot assistant turns as the shots' text exactly (the
  think-token assertion in `chat` is the only guard).
- Step times and memory are estimates from E40a1's logs, not measured for this code.
