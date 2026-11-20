# Appendix to the RL-on-the-67M plan (E215, stage 1)

## Behaviours (`src/commute/rl_tasks.py`)

| Behaviour | Family | Prompt (64 tokens) | Reward on the 16 sampled tokens | Text's own continuation (dry run) |
|---|---|---|---|---|
| lower | case | Pile, cut before a word | at least 8 letters, none upper case | 0.262 |
| french_fr | french | French Wikipedia, tokens 16-79 of an article | more French function words than English, at least one | 0.848 |
| french_en | french | Pile, cut before a word | as above | 0.000 |
| german_de | german | German Wikipedia | as above, German list | 0.758 |
| german_en | german | Pile | as above | 0.004 |
| close_paren | bracket | Pile, last token opens "(" | a ")" | 0.918 |
| close_quote | bracket | Pile, last token is ' "...' | a '"' | 0.801 |
| repeat | copy | Pile, last two tokens seen earlier in the prompt | the first 3 tokens copy what followed them | 0.160 |
| counting | count | Pile prefix + "\n a, a+1, a+2," (a in 10-80) | starts " a+3, a+4" | 1.000 |
| end_sentence | sentence | Pile, no . ! ? or newline in the last 8 tokens | a token ending . ! ? within 8 | 0.250 |
| sports / business | topic | Pile | a word from a fixed list of 30 / 23 | 0.012 / 0.004 |
| emoticon | emoticon | E1's windows to the trigger (21 tokens) | first token "o" | 0.016 |

Prompts: one per row at a seeded cut, from cached Pile val rows 64-8191 (`artifacts/sets/val_0_8192.npy`); every
behaviour found 256 within 258 rows except close_paren (315) and close_quote (1,022). Wikipedia: the first 257-258
articles, streamed at the revision `commute.realistic.WIKI` pins; stage 2 must take its training text from later
articles. E1's emoticon windows come from Pile train rows (train and held-out windows together, the first 256 of 438).

## Edge cases of the selection rule (dry run, 2026-11-19, `~/commute-data/scratch/compare-lead/rl-stage1/dry.py`)

| Case | Result |
|---|---|
| normal (0.12, 0.19, 0.21, 0.3 usable) | picks business 0.19 and repeat 0.21 (equal distance, table order), anchor french_en |
| one usable | stop, "1 usable behaviour(s) of distinct families" |
| none usable | stop, anchor None |
| 0.15 vs 0.25 | tie, the earlier in the order (lower) first; needed rounding the distance to 1e-9 (raw floats broke the tie) |
| two usable of one family | stop: the second must be of another family |
| rates exactly 0.05 and 0.5 | both usable (ends included) |
| two anchors at 0 | the earlier (french_en) |
| no rate under 0.01 / a rate of exactly 0.01 | anchor None (strictly under), reported, no stop |
| a behaviour with 10 or 0 prompts | not eligible, never picked, its rate never read |

## Unknowns stage 2 needs, measured here

- Sampling time per batch of 32 prompts x 8 samples x 16 tokens on an L40 (`sample_seconds_per_batch`, first batch
  apart). No KV cache: each step is a full forward of up to 80 tokens.
- `kl_from_base(net, rows, dW, batch)` in `run_stage1.py`: W0 and W0 + dW forwards on the same batch, mean KL per
  position over the 64 held-out rows. Reads 0 at a zero update by construction of `Net.at`. Stage 2 calls it from
  `run_stage1.py` unchanged for the zero check and the saved states. Its training loop reads the same quantity
  against a frozen copy of W0 (`rl_arms.kl_to_ref`) and asserts it is exactly 0 at step 0.
- Measured by E215: 2.77 s per 256 sequences x 16 tokens; KL 0.0013 for a random update of relative size 0.01.

## Stage 2 (E218): jobs, order, time on an L40

Times come from E215's sampling time: about 1.6 s per RL step of 128 sequences, and about 11 s per held-out read
(128 prompts x 8). Each job loads the model in ~2 min. Commands are in `cluster/rl_67m_stage2.sbatch`.

| Job | After | Time | What |
|---|---|---|---|
| `ctl` (40G) | — | ~6 min | zero update; E209's stage 0 re-reads E53 |
| `calib 1`, `calib 2` | ctl | ~12 min each | RL seed 0, 300 steps, read every 30 steps |
| `null 3e-05`, `null 0.0001`, `null 0.0003` | both calibs | ≤ 35 min each | shuffled, seed 0, both picks, ≤ 600 steps |
| `arm P rl S` (P = 1, 2; S = 0, 1, 2) | both calibs | ~11 min | 6 jobs |
| `arm P own`, `arm P demo` | both calibs | ~12 and ~9 min | 3 seeds per job; 4 jobs |
| `arm anchor demo` | both calibs | ~10 min | 3 seeds, French Wikipedia articles from 258 on |
| `arm P shuffled S` | all nulls | ≤ 19 min | 6 jobs |

That makes 22 jobs. Run as parallel as the cards allow, the wall time is about 2 h at most; in sequence it is about 4 h.
The null jobs are the only ones that may pass 30 min. They fit debug's 2 h cap, so they stay on the cluster.

Demonstration sources: repeat and lower use Pile train rows' own continuations that earn the reward, so the text is
real and was not sampled by the model. Plain register (the scoping page's suggestion) does not match either pick.
The anchor uses French Wikipedia (wikimedia/wikipedia 20231101.fr, pinned revision) after English Pile prompts.
The pools are drawn from 4,096 training prompts. A CPU check found that 1,024 prompts give only 175 (repeat) and 242
(lower) rewarded continuations, which is under the 256 floor. The check also showed:
- the training prompts do not overlap the held-out ones;
- the anchor's French demonstrations read as French 0.907 of the time;
- the reward program reproduces E215's saved rewards exactly.

KL target expected around 1e-3 to 3e-2 nats per token. A random update of size 0.01 gives 0.0013 (E215).

## Stage 2b (E220): jobs, order, time on one H100

Each job is `.venv/bin/python experiments/rl_67m/run_stage2.py <argv>` from the repo root, with
`PRED_SHA=<sha of predictions_E220.json>` on the run line. Each job prints `STAGE 2 <job> DONE` and writes
`results/stage2b_<job>.json`; the arms' states go to `results/large/stage2b/` (bf16, ~57 MB each, 27 states, ~1.5 GB).

Times are E218's measured items times their count, on the H100 pod with five jobs at once:
- an RL or shuffled step takes 3.5 s (repeat RL seed 0, steps 15 to 90: 263 s);
- a supervised step takes 0.43 s (own repeat seed 0, steps 60 to 600);
- a held-out read takes about 25 s, three per run (step 0, step 10, the stop);
- the own pool takes about 3 min per seed, the demonstration pool about 25 s;
- model load is about 1 min (unmeasured on the pod; 0.5 min on the L40 ctl);
- step counts are the expected steps to the target in `predictions_E220.json`.

| Order | argv | Gate it reads | Time |
|---|---|---|---|
| 1, alone | `ctl` | — | ~3 min (E218 on an L40: 15:36:00 to 15:38:52); host RSS 20.5 GB |
| 2 | `null 3e-05` | — | ~7.5 min (2 picks x (3 reads + ~33 steps)) |
| 2 | `null 0.0001` | — | ~3 min (2 picks x (2 reads + ~3 steps)) |
| 2 | `null 0.0003` | — | ~3 min (2 picks x (2 reads + 1 step)) |
| 2 | `arm 1 rl 0` | — | ~5 min (3 reads + ~42 steps) |
| 2 | `arm 2 rl 0` | — | ~4 min (3 reads + ~26 steps; E218's 25 steps took 2.7 min after load) |
| 3 | `arm 1 rl 1`, `arm 1 rl 2` | — | ~5 min each |
| 3 | `arm 2 rl 1`, `arm 2 rl 2` | — | ~4 min each |
| 3 | `arm 1 own`, `arm 2 own` | learned | ~14 min each (3 seeds x (pool 3 min + 3 reads + 20-30 steps)) |
| 3 | `arm 1 demo`, `arm 2 demo` | learned | ~7 min each (3 seeds x (pool + 3 reads + 21-33 steps)) |
| 3 | `arm anchor demo` | learned | ~6 min (E218's job, 25-35 steps a seed, took 6 min) |
| 4, after every null | `arm 1 shuffled 0` .. `arm 1 shuffled 2`, `arm 2 shuffled 0` .. `arm 2 shuffled 2` | learned, null rate | ~4 min each at 3e-5 (3 reads + ~33 steps); ~2 min at a higher rate |

That makes 21 jobs and about 115 job-minutes. Run five at a time, the wall time is about 35 min: ctl 3, wave 2 about
7.5, then waves 3 and 4 about 20. The worst case, a rate that never reaches the size, is 600 steps x 3.5 s = 35 min
per pick. That cannot happen at these rates: E218 passed this size within 120 steps in every arm.

The run reads nothing beyond E218's inputs (cluster/rl_67m_stage2.sbatch header) and the new predictions file. It
does not read E218's stage2_*.json.
