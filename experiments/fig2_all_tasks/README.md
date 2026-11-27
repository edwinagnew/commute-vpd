# E325 (P1.66): Fig 2 on all 16 known-answer tasks

`run.py` reads E_b over the needed set K under E298's protocol (fig2_standard: same scan settings, same leftover
convention, same stage 0), through the library (`commute.eb`, `needed`, `needed_neurons`, `needed_sae`, `kstar_vpd`,
`sae_feat`, `world`, `windows`, `items`), on the states of `commute.known_states`. `score.py` assembles the
definition.csv rows. `check_states.py` is a CPU check of the states and their rows (no model).

## States

- 16 tasks x seeds 0-2 = 48 states, from `commute.known_states` (known3.json's share cells and E314's file and rows).
- **Overlap with E298: 12 states, not 15.** e58/{copy, rank1, cipher, prior_cipher}_s0-2 are E298's
  trained/<task>_s<seed>: the same E147 files (refit_96key/results/large/dW_<m>_s<s>.pt). Their rows come from E298.
- **E298's fifth task, e58/elicit, is not a known-answer task.** known3's elicit is e54/elicit (keydependent_elicit's
  fine-tunes). It is left out of the 16 x 3 rows.
- New here: 12 tasks x 3 seeds = 36 states plus one random each = **72 items per featurization** (the card says 66;
  it counted 11 tasks).

## How the new states' read differs from E298's E147 key-grid read

| | E298 (E147 key-grid states, E237's D) | E325 (each state's own rows, as E136 and E314 read them) |
|---|---|---|
| D | 512 windows of 8 tokens, a seeded subset of E187's read half | `windows.read_rows(<task>, "held")`, every row |
| read position | last token | the task's answer position `pos` (one position) |
| positions in the write pass | all 8, ending at the read position | every position of the row, including those after `pos` (kv: the held-out windows as saved; e53: 127 tokens, read at 63) |
| target column | none | E62's target is never fed (windows.py drops it) |
| E299's seed-2 states (e46/ident32, e46/rand32, e53/lowercase, e53/cipher, e54/cipher_own) | n/a | read on their seed-0 sibling's rows (E314) |
| matched random | E290's `matched_random` (per changed tensor, one generator over F's parameters, seed 161000 + E290 index) | `finetune.random_direction` (per matrix at its Frobenius norm, fp64 norms), seed 325000 + index in `new_states` |
| stage-0 control state | built/x2 (E268's built world) | e62/copy_s0 (the known-answer path) |
| stage-0 reproduction | one saved read per source | VPD e58/cipher_s1, SAE e58/cipher_s0 (E298's items), neurons e58/copy_s0 (E290's, E298's filed neuron reproduction); D from key_grid's World (inputs only) |

Per-task row sources: e46/* kv (regating_table windows), e54/* kv (keydependent_elicit windows), c3/* kv (open_pair
trained windows), e62/* E62's read windows (induction_cue), e53/* E53's 500 scored Pile windows (val rows 0-255).

Not decided here: whether to cut each row at `pos + 1` so the write pass ends at the read position as E298's did. The
rows are used as E136 saved them; a cut would change E_b on the e53 rows (63 positions after the read) and any kv rows
longer than `pos + 1`.

## Runs

    PRED_SHA=<sha> python experiments/fig2_all_tasks/run.py --feat <vpd|neurons|sae> --shard <i> <n>
    ~/commute-agents/fleet/cpu-run --mem 4G python3 experiments/fig2_all_tasks/score.py

Each job: stage 0 on its card (stops on a miss), then items[i::n]; items to results/items/<feat>/, arrays to
results/large/<feat>/, update files read with their sha256 to results/state_files/<feat>_<tag>.json.
