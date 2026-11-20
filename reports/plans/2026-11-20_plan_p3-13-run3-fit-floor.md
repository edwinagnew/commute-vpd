# P3.13 run 3 (E225): the held-model reading on the word pairs, with a fit that reproduces the model

*2026-11-20, qwen-lead. Cell P3.13 (rung L3.2, due 11-25), run 3 of 3: the user chose option 1 of
`reports/decisions/2026-11-19_decision_p3-13-held-fit-floor.md` (task file and chat dialog, 2026-11-20). Card
`experiments/qwen_held_readings/card3.md`; predictions `experiments/qwen_held_readings/results/run3/predictions.json`.*

## Why
Run 2 (E219) read the held-model reading S on the pair library. On gold's rows the held model H, fitted with nothing
changed, reproduced the real model (do-nothing KL(F||H) 4.3e-9 a position). On the four word row sets it did not (2.8e-4
to 2.2e-3), which voided their random references and put an unknown error on each word-pair S. This run finds a fit that
reproduces the model on those rows and rereads the two word pairs with it.

## What runs
Code `experiments/qwen_held_readings/run3.py` (run 2's reading, `commute.qwen_held`, arithmetic unchanged; only the fit
rows change). One H100 pod per word pair (copy/cipher, known/invented), in parallel, RunPod (Slurm is down).

- **Stage 0, the fit sweep, alone and first.** For each of the pod's two row sets, F = M, four fits in filed order:
  run 2's (own rows 0-63; measured only, expected to reproduce run 2's floor), own 0-63 plus 64 gold rows, own 0-63
  plus 256 rows of the same sequences packed with two other seeds, and both. Read rows are always the arm's own rows
  64-127. The first fit after run 2's whose do-nothing KL(F||H) is at most 1e-7 on both row sets is used for the pair;
  if none, the pod stops and the pair is marked unreadable at these fits.
- **Stage 1, the reread.** As run 2 with the chosen fit: 12 states a pod (2 sides x 3 seeds x 2 saves), each at
  saved size and beside its Gaussian change; the larger side also times E187's c. Output-only control first.

## Expected values
- Do-nothing: KL(F||M) 0; run 2's fit reproduces 1.2e-3 (cipher), 2.2e-3 (copy), 2.8e-4 (known), 3.9e-4 (invented).
- Output-only control: 1 - S under 1e-6. States 0.95-0.999; random references about 0.8.
- Rivals (card): A, reuse side above by more than 0.005 at every seed, save and size on both pairs; B, neither pair.

## Cost and time
Rehearsal pod `commute-E225-rehearse-sweep` (copy/cipher, `--limit 1`: stage 0 on both row sets and the first state),
~20 min, ~$1.5. Then pods `commute-E225-copy_cipher` and `commute-E225-stage-facts`, ~45 min each, ~$3 each. About $8
in all, under the $300 session budget. Not gated.

## Not run
Gold pair (read cleanly in run 2). A fit with a lower pseudo-inverse tolerance or anchored at W0 (a change to the
reading; would need the user). The share on either decomposition.
