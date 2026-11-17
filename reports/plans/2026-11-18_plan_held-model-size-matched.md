# Plan: the held-model reading on E187's size-matched pairs (cell P1.27)

*2026-11-18, definition-lead (code writer). Cell P1.27, its one run, experiment E205. Code `experiments/held_model_size_matched/`,
filed rules `results/predictions.json`, card `card.md`. No cut-offs and no reuse or new labels.*

**In plain words.** The working definition of reuse (user, 2026-11-18) reads how much of a fine-tune's change survives
when every unit is held to the original model's rule; its main reading is a fitted linear map per layer. On matched
pairs the reuse member kept a little more than its twin, by 0.002 to 0.011, and the twins also had the larger updates.
This run shrinks each pair's larger updates to the other side's size, as E187 did, and reads survival again, to see
whether those small differences were about update size.

## Design check

- **Goal.** `GOALS.md`, the working definition: whether its main reading tells pair members apart for a reason other
  than how far the weights moved. Fills open cell P1.27 (definition-lead), run 1 of 1.
- **Nearest tried.** E197 (`TRIED.md`): reading 3 gaps 0.002-0.011 on E196's pairs, "not in order" of MLP update size,
  untested at matched size. E187: the same rescale for the share and the cheap readings. This joins the two.
- **Patch or step.** A step: the decisions row of 2026-11-18 lists this as a named follow-up.
- **Decisions relied on.** The held-model row of 2026-11-18 (reading 3 main, reading 1 beside, S as filed in P1.26's
  plan, embeddings at M's values); E187's c as computed. No departure.
- **Adopted definition, unchanged?** Yes: S is read as frozen; nothing is added to it and no item gets a verdict.
- **What would change what we do.** Gaps that shrink toward zero when sizes match say the main reading's pair
  differences track update length, so a write-up must not quote them as reuse differences; gaps that stay say the
  opposite for this kind of size change.
- **Apparatus control, first and alone (stage 0).** Do-nothing on the scaling path (an update times 0: KL(F‖M) = 0,
  H reproduces F), E197's gains-per-block build times 0.25 (still one gain per block: S = 1 in every reading), and four
  unscaled items E197 read must repeat its S within 1e-4. It fails if the scaling path, the windows or the state files
  differ from E197's. Any miss stops the job before any pair.
- **Premise doubt.** A scaled-down update is not a model that learned less. To first order in the update, both
  KL(F‖M) and KL(F‖H) scale as c², so S does not change with c at leading order. A matched gap equal to the original
  then says S ignores the length of an update along its own direction, not that it would ignore a model that learned
  less. The run reports KL(F_c‖M) / KL(F_1‖M) / c² per state, expected near 1, which checks the argument's premise.
- **Saved for audit.** Per item JSON (S, KLs, logit error, residuals per reading, sizes); per-unit partners and gains of
  readings 1 and 2 in `results/large/units/` (~60 KB each).

## What runs

E187's nine built pairs (`size_matched/run.py` `BUILT`), every saved seed (2 to 8 per side), on the 67M. For each pair
each state at its own size, then each state of the larger side times E187's c (`results/m67.json`; the new side is the
larger in eight pairs, c 0.52 to 0.99; in `open_pair_c3` the reuse side, c 0.997, an in-run near-null). F = M + c·dW;
H, S and the windows are E197's code unchanged (16,384 fit windows, 512 read windows, last position). Tasks that are not
among E197's old tasks (E46's b tables, E54's, E58's 96-key tables) get key-table windows built the same way from
E187's windows files. 108 items: 68 states at their own size, 40 scaled (four of the 68 read in stage 0). A CPU dry
check (2026-11-18, `cpu-run`, no model) loaded every state as the 24 matrices, matched E187's size of each to 2e-16, and
built the windows of all eight new key-table tasks (100,000 fit and 100,000 read windows each).

| number | do-nothing (c = 0) | gains-only control | expected on items |
|---|---|---|---|
| KL(F‖M) | 0 (≤ 1e-6) | > 1e-6 | 0.6 to 4 nats at c = 1 (E197) |
| H's logit error, each reading | ≤ 1e-3 | ≤ 1e-3 | reported |
| S, reading 3 / reading 1 | null | 1 / 1 | ~0.98-0.999 / ~0.01-0.06 (E197) |
| S shift, same state, matched minus saved | n/a | n/a | ~0 at leading order |
| KL ratio over c² | n/a | n/a | ~1 |

## Filed rules (comparisons, no cut-offs; readings 3 and 1)

Per pair: gap = mean S of the reuse side minus mean S of the new side, at saved sizes and at matched sizes;
se_original its seed spread (sides have unequal seed counts, so no per-seed pairing); shift = each larger-side state's
S at matched size minus at saved size (paired).
- **R1** sign of the matched gap against the original's: same, flipped, or zero.
- **R2** matched gap over original gap (1 unchanged, 0 gone, negative reversed); null at an original gap of 0.
- **R3** |mean shift| at most se_original, true or false; expected true under reading 3 on every pair.
Edge cases (missing or null items, one-seed sides, exact zeros, c = 1) are written out in `predictions.json`; at c = 1
every shift is 0, so the rules read R2 = 1 and R3 true, the identity reading. Update sizes, whole and MLP, per item.

## Cost and jobs

MATS cluster debug QoS, one L40 per job, $0, `cluster/held_model_size_matched.sbatch` (30-min limit). E197 measured
~31 s per item including a 100,000-window gap check this run drops. Job 1 (stage 0 alone): set-up ~1 min, six reads,
~4-5 min. Then four jobs in parallel once stage 0 has passed: stages 1-2 (19 items left, ~11 min), 3-5 (29, ~16
min), 6 (23, ~13 min), 7-9 (33, ~18 min), each with ~1-2 min of set-up. Each logs a projected time per item; every item resumes from its own file.

## Not run

Readings of a model trained to a smaller update (rather than rescaled); E187's Qwen half; any bar or label.
