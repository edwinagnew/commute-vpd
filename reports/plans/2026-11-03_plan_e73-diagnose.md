# Plan: E73, what is odd about a small update on Qwen, read in the clean model

*2026-11-03, D1 blocked, day 1 of 8 (D1 is a map that keeps the decomposition faithful for a real update, judged by a test that can fail a bad placement). User, 15:10: understanding this is extremely high priority; quick experiments need no approval. Class 1 (GPU passes on saved states). $0, under the gate, no review.*

**In plain words.** E72's clean reads on the step-10 update show W0 far less sure than a calibrated model should be wrong by (entropy 0.61, NLL 2.07), a fine-tuned model much less sure (1.51), and each piece of the update (in-span, remainder, either half) nearly reproducing the fine-tuned model, while a random change does not. This job asks where that comes from, using only the clean model, before any further masked read.

## Design check
- **Goal.** Whether D1 is blocked by the task, the model, the decomposition or the test. This job separates the first two and whether updates are unusual; it reads no masks.
- **Nearest tried.** E72 T2 (the pieces, one scale). E68 search.json (random scale matched on KL). E61 five checks (precision of training). None read a t path, W0's calibration by text, or which positions move.
- **Patch or step.** A step: a diagnosis the plan's tree does not hold.
- **Decisions.** None departed from.
- **What changes next.** W0 overconfident only on maths rows: the rows or their format is the suspect (task). Overconfident everywhere and sharp for any direction: the model at W0 is unusually sensitive, so every test at this size reads that (model). Sharp only along the real update: the update's mechanism is one simple move, and the placements are all reading it.
- **Control first.** u1's three numbers against E72 on the same 128 rows, with a stop.

## Numbers (filed in results/predictions.json)
| number | W0 | t = 1 | positive control | blind to |
|---|---|---|---|---|
| NLL, entropy, accuracy, top probability | E72: 2.07, 0.61 | E72: 1.90, 1.51 | web rows: entropy near NLL expected | positions |
| K(W0 ‖ W0 + t δW), K(FT ‖ W0 + t δW) | 0, 1.635 | 1.6 ish, 0 | Gaussian and permuted δW at the same norm | which tokens |
| Same, one block | 0 | not read | sums show if one dominates | interactions |

Saved: one small JSON per read. **Not run:** masks, Γ, the decomposition, retraining, generated text.
