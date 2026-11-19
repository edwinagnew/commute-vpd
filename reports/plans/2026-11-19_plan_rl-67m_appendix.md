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
  position over the 64 held-out rows. Reads 0 at a zero update by construction of `Net.at`; stage 2 moves it into
  `src/commute/` when it calls it.
