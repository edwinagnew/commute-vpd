# Plan: R under wording A at SAE latents on random updates of matched size (P1.40 run 3, E251)

**In plain words.** E245 read R under wording A at the base-only SAE latents and found that every key-task fine-tune reads between 0.989 and 0.999. With no reference, that number cannot be read: it may only mean that any update of that size reads near 1 at SAE latents. This run reads the same thing on random updates of the same size, on the same windows, with the same code.

**Request (whip-lead, 2026-11-21 ~15:00; user approved run 3 ~16:00).** "Read R under A at SAE on matched random updates (same update size as each key task, ~15 items), same code as E245."

**Items (17).** E237's 15 random controls, `random_<state>`. Each one is a Gaussian per matrix at that matrix's norm in the state's saved update, times E237's common size c, which is exactly the size of E245's `<state>_scaled` item. It is read on that state's windows against the same base model. The other two are E242's random controls, `random_size_241001` (new_thin's size on x2's windows) and `random_kl_241002` (KL-matched on x10's windows), which read R_A 0.29 and 0.30 at VPD.

**Code.** `experiments/sae_r_a_random/run.py`, about 80 lines. It imports E245's `load_config`, `load_saes`, `stage0` and `read_item` and calls them unchanged. It adds only a World that sends `random_e58/...` items to E237's World, plus the item list. E245's predictions file is checked by its sha256. Every setting and threshold is E245's (sourced in its plan).

**Predictions.** `experiments/sae_r_a_random/results/predictions.json`. P1 is descriptive: the gap is the smallest key R_A minus the largest random R_A, at tol 0.1, read against E245's band of 0.11.

**Gates.** This run is not gated (about $18, about 2 h). The rehearsal is E245's (job 62745), on the same code path at the full read sizes. A CPU dry run builds every item (manifest in `~/commute-data/scratch/sae-lead/random_manifest.txt`). Gate 6 GO is from whip-lead, with a P1.40 cap of $75.

**Cost and time.** Measured in E245's key run on H100: stage 0 takes about 840 s, and a key item of 512 windows x 8 takes 262-450 s. Two pods share the 17 items (9 and 8), about 14 min + 9 x 7 min, plus up to about 52 min if random_kl's windows are 512 x 41 as on the built items. That makes about 2 h per pod, about $18 at $4.40/h.

## Not run
- No halves on the random items, so there is no band of their own; E245's built band is used.
- No KL-matched random per key state, only size-matched.
