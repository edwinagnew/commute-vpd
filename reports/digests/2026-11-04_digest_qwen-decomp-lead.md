# Digest, qwen-decomp-lead, 2026-11-04

D1 blocked, day 2 of 8 (D1: a map on Qwen3-1.7B that keeps the decomposition faithful for a real update, judged by a test that can fail a bad placement). I finished because everything left waits on one decision that only you can make: what D1 is judged by. VPD's adversary can fail states on this model, but so far only far from a real update's size.

## The decision you need to make
Log card `decision-after-e76` (ask 29), needed by 11-05, the task's date for choosing test or teacher training.
- **A, recommended.** Judge D1 by the clean-model table, the share left in Δ, squared size and behaviour kept on TOFU, with the adversary as the check on columns and its size limit stated. Least squares is the first map; teacher training toward TOFU ($0, about 75 min) is the trained one.
- **B.** Keep your earlier choice, teacher training with masks in its loss as the map; same table and run.
- **C.** File that D1 cannot be met as worded and move to D3, the readings on every arm.

The filed ask says "about 0.8" and "1.27". The card has the corrected figures: 0.88 as a norm, and 1.27 is a mean of two seeds (1.157 at seed 0, the controls' seed).

## What ran
E85's two amendments finished today; [the report](../analyses/2026-11-04_analysis_e85-adversary-teeth.md) and the notebook `notebooks/E85_qwen_d1_teeth.ipynb` are written, and the log is at v225.

- **Positive controls (Amendment 1).** Wrong columns read 11.3 times W0 and a random change at 100 times the update's size read 32.5, against a fail bar of 1.61. Both are seed 0 only, and the late traces swing widely, so each is one noisy reading.
- **Size gap.** Those wrong columns sit 250 to 740 times further from the right ones than a real update moves them. Failing a bad placement at a real update's size is unmeasured.
- **Rounding (Amendment 2).** Γ's bf16 rounding of V is not a limit. That does not mean g hears the placement: moving V alone changes g by less than the rounding floor.
- **Filed misses, no cause claimed.** Five readings missed what the plan expected; the report lists them. Not checked: there is no independent check of the amendments on file.

## Money
About $7.00 on the earlier session's meter for Amendment 1, including $2.09 on a pod the provider deleted. This session spent $0, and nothing is billing.

## Friction
Two notes: a squared share compared with a norm in my handoff, caught by the report writer (482), and `build.py` not executable as the shell notes say (483). Problem 13, the card held outside Slurm, stays open until a job lands there clean.
