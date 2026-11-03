# Plan: E74, where the masked forward loses a change, and how much of the ruler is temperature

*2026-11-03, D1 blocked, day 1 of 8 (D1 is a map that keeps the decomposition faithful for a real update, judged by a test that can fail a bad placement). User, 15:10: understanding this is extremely high priority; quick experiments need no approval. Class 1 (GPU passes on saved states). $0, one L40, ~25 min, under the gate, no review. Reads 1 and 2 of `reports/plans/2026-11-03_advice_e73-what-is-wrong-FABLE.md`.*

**In plain words.** With the original model's masks held fixed, a random change 18 times the update's size barely moves the masked model. This job adds one block's change at a time to find where it gets lost, and beside each real change puts a random one of matching effect, which says whether the loss is general or special to this update. It also asks how much of the fine-tuned model's distance from the original on the maths rows one temperature removes.

## Design check
- **Goal.** Whether D1 is blocked by the test, the decomposition or the task (`GOALS.md`; plan 2026-11-03 §D1).
- **Nearest tried.** E72 T0a and ceiling (whole update, one number); E73 (clean only). Neither says where a change is lost or reads a temperature.
- **Patch or step.** A step: the diagnosis the user asked for.
- **Decisions.** None departed from.
- **What changes next.** Lost downstream and generic: the fixed-mask family retires as a grader of placements, and ask 23 is restated with it. Random passes far better than the update: the update runs through switched-off subcomponents, the decomposition is the suspect. Control missed: every ceiling number is in doubt; stop. Temperature removes most of the gold updates' distance: maths rows stop being the yardstick.
- **Control first.** Temperature control, hook against E73, K0 against E68, then the last matrix's transmission, each with a stop, on u1 before any other arm.

## Departures from the advice
1. **Control.** Fable: block 27 whole, "must be near 1". Most of block 27's matrices feed its own masked `o` and `down`, so that is not a control. The control is `layers.27.mlp.down_proj` alone, with its Gaussian; block 27 whole stays a read.
2. **Cut arm.** Fable's optional fifth arm is run (blocks 0, 7, 14, 21): it is the direct test of the downstream reading. No Gaussian on it.
3. **Added.** u3 beside u1 in read 1 (the clean case); all four updates in read 2; the whole update's transmission; cosine and amplitude of the logit shift (KL is quadratic: a transmission of 0.01 is an amplitude of about 0.1, and a direction change would read the same as a small one).

## Numbers
| number | do-nothing | control | expected | blind to |
|---|---|---|---|---|
| τ per block, δW and Gaussian | no change, undefined | `down` of block 27 in [0.5, 2] | rising with depth, Gaussian within a factor 3 | which subcomponents |
| τ with later sites unmasked | as above | as above | 0.5 or more | block l's own masks |
| min over T of KL(FT ‖ W0 at T) | FT at T = 1 reads 0 | W0 at T = 1 reads 1.6351 on u1 | between 0.4 and 1.0 on gold | per-position temperature |

Rules, edge cases run: `experiments/qwen_transmission/rules.py`, `results/predictions.json`. Saved: per-row sums, one small JSON per group.

**Not run:** read 3 (human rows), any adversary, any new training, which subcomponents Γ switches between W0's and FT's forward.

## Amendment 1 (2026-11-03, after job 50510)
Job 50510 stopped at the filed control on u3 (KL ratio 0.447 against 0.5; amplitude 0.75, cosine 0.96). The band was guessed, and the hook checked exactly, so the control is restated as amplitude in [0.5, 1.5] (`results/predictions_amend1.json`; `predictions.json` stands and reports the miss). u1's reads missed both leans: its label is "elsewhere" (block 0's change keeps 0.057 with every later site unmasked, against 0.5 filed) and "needs switched-off subcomponents" (Gaussian's transmission is 25 to 250 times δW's at blocks 0, 7, 14, 21). A cut leaves both the change's own block and everything upstream masked, so it cannot say which of them costs what. This job separates them: masks only upstream of the change, and all masks with the term reading the clean inputs, for δW and Gaussian, plus how far the masked residual stream sits from the clean one. u3's remaining groups run first, so the template is read too.

**Not run:** which subcomponents δW depends on; the split of δW into often-on and rarely-on parts under these arms.

## Amendment 2 (2026-11-03, after job 50533, the last on E74)
Amendment 1 read: the masks at and after a change kill it, for δW and Gaussian alike (amplitude 0.002 to 0.09 at blocks 0 to 14; "own and later masks" or "both"), upstream masks cost a further factor because the masked residual stream sits 25-73% from the clean one, and u3 reads as u1. In that arm δW still passes 8 to 224 times less KL than Gaussian across blocks 0 to 21. Two unread things change what D1's test becomes. **Direct:** is the gap amplification (δW's clean effect runs through later blocks, a Gaussian's through the residual skip) rather than switched-off subcomponents? The clean model with every later matrix fixed at its clean output; expected δW's share far below Gaussian's. **Stochastic:** VPD's own masks leave off subcomponents partly on: are they as deaf? If they carry the update, that is D1's next test candidate; if not, every masked test on this decomposition is out. Control: last matrix's amplitude; direct's identity at block 27. ~15 min. Rules `results/predictions_amend2.json`.

**Not run:** an adversary on a changed state; any placement under stochastic masks.
