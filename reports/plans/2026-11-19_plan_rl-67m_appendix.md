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
