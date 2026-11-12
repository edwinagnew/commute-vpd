# Plan: does the re-gating share read the two sides of a library pair differently? (cell P3.8, E??? and E???)

## The question, in plain words
The user (task file, 2026-11-12): "Read the re-gating share on your pair library (run 3, E165) and the gold twin, reuse
side minus new side, at step 300 and at the matched-size save. A random change of matched norms first, alone. One
figure, no reuse or new labels." Restated: for each saved fine-tune of the library (copy, cipher, known facts,
invented facts at 300 steps; gold and shuffled gold at 100), read E132's KL re-gating share on the new decomposition
p-9707d6ff, each side on its own rows; per pair and seed, take side A's share minus side B's (copy − cipher,
known − invented, gold − shuffled gold). Before any of that, read random changes sized like the library's states, to
set how large a gap rows and size alone give. It is reported as a description: no reuse or new verdict, since the
definition is under revision (P1.1).

## Design check (`/design`)
- Goal: `GOALS.md`'s "is it because post-training teaches new capabilities?" — whether the share, the one Γ-side
  reading the 1.7B line has, tells built sides apart at all. Fills the named open cell P3.8 (user-approved 2026-11-12).
- Nearest row: E132 (P3.1) read the share at 0.991-0.999 for random changes and four updates alike; it never read
  pairs built to differ, on their own rows. This learns whether that flatness holds across the library's sides.

## States
- Run 3 (E165), `experiments/qwen_pair_library/results/library_run3/<arm>_s<seed>.json`: copy, cipher, known,
  invented, seeds 0-2. Run 2 (E164), `results/library/`: gold, gold_shuf, seeds 0-2. Each state δW = W_t − W0, fp32,
  5.64 GB, 196 tensors keyed by site, on W&B only (`p35r3-<arm>-s<seed>-<labels>`, `p35-...`); fetched by the new
  shared `src/commute/library_states.py` (both prefixes; a save found by its label in `.saves[].labels`, sha256
  checked against the library's record).
- Saves, one dict in `run.py` (`_saves`), proposed to compare-lead and **not yet agreed**: late = `last` (step 300; the
  gold twin's `last`, step 100, stands in); matched = `5x` for every arm, as compare-lead's P3.7 reads it (`experiments/qwen_library_readings/rules.py`
  `PAIRS`; agreed 2026-11-12). At 5x cipher and invented have not yet learned their trained items (exact match 0 and
  ~0.05), so the matched column compares a learned reuse side with an unlearned new side.
- Rows: each arm on the rows the library matched its sizes on (`library.py` `built_data` / `gold_data`, the rows of
  `results/floors.json`): `qwen_pairs.own_rows` of the side's trained cue sequences, and `qwen_pairs.gold_rows` (E40's
  held-out gold rows) for both gold arms. 128 rows of 256 tokens. `run.py` calls those functions; nothing is copied.
  Each state's K0 must equal the library's floor for its side (E132's maths K0 equals floors.json's to 10 digits).

## Run 1 (E???, alone, first): controls and random changes — `run.py --run 1`
| check | reads | must read | else |
|---|---|---|---|
| W0 per row set (5) | 16 rows, W0 as the fine-tuned model | K00 = K01 = K0; zero split K10 = K11 = K0, 1e-6 rel | stop |
| do-nothing | first change, 16 rows | share 1 within 1e-6 | stop |
| K0 vs floor | every state, 128 rows | floors.json K0 within 1e-3 rel | stop |
| sha256, live count, stored error, GPU | as E132 | library's sum; 500,244; < 1e-3; < 44 GiB | stop |

Then for each of the 6 arms at seed 0 and each of the 2 saves (12 states): one iid Gaussian draw (shared by all 12),
each matrix scaled to that state's Frobenius norm at that matrix.
- Stage 1, `<key>_norm` (the user's control, read as it is). Expected: KL 100-400x under the state's (E132's Gaussian
  at an update's energy read 0.0015 against 0.618) and status "unscored". That is its reading, not a stop.
- Stage 2, `<key>_kl` (**a stated departure** from "matched norms", Fable's advice, 2026-11-12): the same change times
  one factor b (E57's `scale_search`, within 1.5x) so its KL on 16 own rows equals the state's. The split is linear
  in δW, so stage 2 reuses stage 1's split times b. The band comes from stage 2; it must read "scored" (else stop).
  Expected b 10-40 (E132: 20.2), share at least 0.98, |random gap| at most 0.01.

| number | at W0 | do-nothing | random (stage 2) |
|---|---|---|---|
| share | undefined (zero denominator, "unscored") | 1 exactly | ≥ 0.98 |
| A − B gap | — | 0 | \|gap\| ≤ 0.01 |

## Run 2 (E???): the 36 states — `run.py --run 2 --arms A,B`
Refuses to start until run 1's 12 stage-2 records are home. Same controls (W0 on its row sets, do-nothing on its first
state, K0, sha256), then 6 arms x 3 seeds x 2 saves, each fetched while the previous one is read, written as
`results/run2/<arm>_s<seed>_<save>.json`, its file deleted. Resumable (skips what is written).

## Predictions (`results/predictions.json`, filed)
- Gap per pair, save and seed = share(A) − share(B), None unless both "scored".
- Band per pair and save = |stage-2 random gap| + 0.00196 (E132's seed spread, `qwen_regating_new/results/summary.json`).
- Cell: "unreadable" (under 3 gaps, a None or nan, no band); "beyond band" (every |gap| > band, one sign); "inside
  band" (|mean| ≤ band); else "mixed". Edge cases run through `run.cell` on CPU 2026-11-12: ties go inside, mixed signs
  never read beyond, the do-nothing pair reads inside.
- Blind to: the share reads near 1 whenever K01 and K11 sit far under K00; the stage-2 random change is 10-40x the
  update's norm and A and B get separate factors. Noise floor: seed spread 0.002, random-vs-random 0.001 (E132).
- Lean (medium): rival B, every cell inside the band or mixed, all gaps under 0.01, because E132 read 0.991-0.999
  for everything.

## Cost and routing
The cluster's 8 L40s are all allocated with a queue (checked 2026-11-12, 15:00), so RunPod H100 80 GB, $3.99/h, with
≥ 128 GB host RAM (E132 measured 91 GB) and ≥ 120 GB disk (decomposition 50 GB, two states 11.3 GB, model, venv).
**The new decomposition p-9707d6ff is not on W&B** (checked 2026-11-12: `qwen3-1.7b-p-9707d6ff-torch` not found).
Sync from dev: 50 GB ÷ 10 MB/s = 83 min of idle pod per pod. Upload it once first, a $0 CPU job:
`sbatch cluster/qwen_upload.sbatch ~/commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch`; then each pod fetches it at
30-45 MB/s: 50 GB ÷ 40 MB/s ≈ 21 min. A state is 5.64 GB ÷ 40 MB/s ≈ 2.4 min, overlapped with reading.

Times per state from E132's L40 log (cells on 128 rows 108-133 s, split 361-372 s, split cells 146-160 s). The split is
float64, much faster on an H100 (assumed ~90 s; unmeasured). Fixed per pod: decomposition 21 + first state 2.4 +
loads 1.6 + Σ 3 + controls 2-5 + set-up 5 ≈ 35-38 min.

| | per state (H100 expected / L40 upper) | states | hours (exp / upper) | $ (exp / upper) |
|---|---|---|---|---|
| run 1, 1 pod | 12.3 / 17.1 min (two cell reads, one split, search) | 12 | 3.1 / 4.1 | 12.4 / 16.2 |
| run 2, each of 3 pods | 6.1 / 10.8 min | 12 | 1.8 / 2.7 | 7.2 / 10.9 |
| total | | 48 | | **34 / 49** |

Run 2's pods: (copy, cipher), (known, invented), (gold, gold_shuf). Stop rule: after run 1, the runner reports its
measured minutes per state; if run 1's spend plus run 2 re-estimated from that time passes $45, the lead asks the user
before run 2. Each pod downloads `results/run<k>/` after every few states and `run.log` to a pod-named path (three
pods write the same `results/run.log`).

## Order of work
1. Lead: agree the saves with compare-lead (edit `_saves` only); claim two E-codes; write them into
   `predictions.json` `experiment` and `run.py`'s docstring; upload the decomposition to W&B.
2. One fresh-Opus review of plan and code is not required (not a gated run: under $60 and 6 h).
3. `/pod-runner` run 1 alone; then run 2 on three pods; `run.py --summary`, `plot.py`; write-up by `/writeup`.

## Not in this plan
No reuse or new verdict or label; no log-odds share (P3.2 found it unreadable); no other library pairs (context/lookup,
template); no row-resample interval per state (the seeds are the unit); no cheap readings (compare-lead's P3.7).
