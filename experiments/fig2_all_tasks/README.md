# E325 (P1.66): Fig 2 on all 16 known-answer tasks

`run.py` reads E_b over the needed set K under E298's protocol (fig2_standard: same scan settings, same leftover
convention, same stage 0, fp32 with TF32 off), through the library (`commute.eb`, `needed`, `needed_neurons`,
`needed_sae`, `kstar_vpd`, `sae_feat`, `world`, `items`, `old_tasks`, `variable_read`), on the states of
`commute.known_states`. `score.py` assembles the definition.csv rows. `check_states.py` is the pod's first step: a
check of the definition windows each new task is read on, and of the state files (no model).

## The e54 tasks' D

e54/elicit, e54/cipher and e54/cipher_own have no `old_tasks.TASKS` entry. Their D is built by the rule E298 used for
the e58 kv tasks (approved by whip-lead, 2026-11-27): E205's `old_task` for a key table (held_model_size_matched), with
E221 `e187_task` keeping only the first file. That gives a kv entry named and grouped `<task>` with '/' as '_', built
from the task's training windows file. One helper, `known_states.definition_task`, gives the entry for e58 and e54
alike, so E298's e58 path and the e54 path are the same code. `check_states.py` shows that it reproduces E298's e58 D
bit for bit (the OldTask equals E221's `e187_task`, and the 512 x 8 windows are equal to key_grid World's) before it
builds e54's.

| task | windows file |
|---|---|
| e54/elicit | keydependent_elicit/results/large/windows_elicit.pt |
| e54/cipher | keydependent_elicit/results/large/windows_cipher.pt |
| e54/cipher_own | windows_cipher.pt: keydependent_elicit `finetunes` trains cipher_own on `ws["cipher"]`, so it shares e54/cipher's D (group e54_cipher) |

## States

- 16 tasks x seeds 0-2 = 48 states, from `commute.known_states` (known3.json's share cells and E314's update files).
- **Overlap with E298: 12 states.** e58/{copy, rank1, cipher, prior_cipher}_s0-2 are E298's trained/<task>_s<seed>,
  the same E147 files (refit_96key/results/large/dW_<m>_s<s>.pt). Their rows come from E298.
- **E298's e58/elicit (E147's 96-key elicit) is not one of the 16.** The figure's elicit is e54/elicit
  (keydependent_elicit's fine-tunes). e58/elicit is left out of the 16 x 3 rows.
- New here: 12 tasks x 3 seeds = 36 states, each with one random = 72 items per featurization (the card says 66: it
  counted 11 tasks).

## D: the definition's D, as E298 read it

Every new state is read on the same D as the 12 e58 rows, so the 48 values are one quantity. The task's
`old_tasks.task_windows(known_states.definition_task(state))` read half: its shows windows, as many does-not windows and 2/3 as many far rows, 8 tokens
each, shuffled and halved. Then E221's seeded subset of 512 (n_read 512, read_seed 195200). K is read at the last
position; the write sums run over all 8 positions. E298 built the e58 tasks' entries from E147's windows file (E205
`old_task`); nine new tasks use their existing entries, the three e54 tasks the same kv rule:

| known-answer task | old_tasks entry | kind |
|---|---|---|
| e62/copy, e62/lookup | e62_copy, e62_lookup (one group: same D) | e62 |
| e46/ident32, e46/rand32 | ident32, rand32 | kv |
| e53/lowercase, e53/cipher | lowercase, cipher50 | text |
| c3/rank1, c3/copy, c3/cipher | rank1, copy24, cipher24 | kv |
| e54/elicit, e54/cipher, e54/cipher_own | e54_elicit, e54_cipher (E205's kv rule, above) | kv |

E299's seed-2 states read their task's entry like seeds 0-1; the D does not depend on the seed.

**A stated difference from Fig 3:** Fig 2 reads the definition's D, 8-token windows with the write sums over every
position, in fp32. Fig 3's re-gating share reads the task's held-out rows (E136) at the answer position only, under
bf16 autocast.

## Other departures from E298

- Matched random: E290's recipe, `finetune.matched_random` (E290's `needed_neurons.matched_random`, moved into src
  unchanged): a Gaussian per tensor F changed, at the fp64 Frobenius norm of F - M, one generator over F's
  named_parameters. Only the seed base differs: 325000 + index in `new_states`. `finetune.random_direction` is a
  different recipe: it scales to the saved dW's norm, draws in the file's order, and zero-norm tensors consume draws.
  So it is not used.
- Stage-0 control state: e62/copy_s0 (the new path) instead of built/x2 (E268's built world, never loaded here).
- Stage-0 reproductions: VPD e58/cipher_s1 and SAE e58/cipher_s0 (E298's items), neurons e58/copy_s0 (E290's, E298's
  filed neuron reproduction). F comes from known_states' file; D comes from `definition_task`, the same path as the
  new items.

## Runs

    python experiments/fig2_all_tasks/check_states.py   # first on the pod (builds key_grid's World on the GPU)
    PRED_SHA=<sha> python experiments/fig2_all_tasks/run.py --feat <vpd|neurons|sae> --shard <i> <n>
    ~/commute-agents/fleet/cpu-run --mem 4G python3 experiments/fig2_all_tasks/score.py

Each job: stage 0 on its card (stops on a miss), then items[i::n]. Items go to results/items/<feat>/, arrays to
results/large/<feat>/, and the update files read, with their sha256, to results/state_files/<feat>_<tag>.json.
