# Plan: E68, the D1 pass test on the saved updates

*2026-11-03, D1, day 1 of 8. Approved class 4 and the reorient plan's day-1 row (`reports/plans/2026-11-03_plan_qwen-reorient.md`,
§D1). Fable's advice, taken except where marked: `~/commute-data/scratch/qwen-decomp-lead/fable-e68/advice.md`. Under the
gate: $0 on a cluster L40, no pre-run review. Rules: `experiments/qwen_pass_test/results/predictions.json`.*

**In plain words.** We place each saved fine-tuning update on subcomponents the decomposition often uses, on ones it rarely
uses, and as a random change, and check that the test tells these apart. Then we score the least-squares and adjoint splits
against those two controls. If the test separates them it stands, and D1 has its first table.

## Design check

- **Goal.** `GOALS.md`'s question on Qwen: is there a map that keeps the decomposition faithful for a real update? D1 needs a
  test that can fail a bad placement.
- **Nearest tried.** E44 Amendment 2 stopped at a guessed margin of 0.2 before reading any split. This job reads them, with
  the band measured.
- **Patch or step.** A step: the first split scores on Qwen. It reuses `experiments/qwen_maps/amend2.py`'s reader.
- **Decisions.** Follows `decisions.md` of 2026-10-29: a change to Δ is never an automatic fail.
- **What changes next.** See the tree.
- **Control first.** E40 step 10 must reproduce Amendment 2's numbers and do-nothing must read 0, with a stop.

## Corrections to the reorient plan's §D1

1. "The least-squares split is exact with Δ untouched" is wrong. It is exact because the remainder goes to Δ: 0.36 of the
   update at step 10 (`experiments/qwen_maps/results/S.json`). In-span and adjoint leave Δ almost untouched and are not exact.
2. "A map that puts anything in Δ is out" conflicts with the user's rule of 2026-10-29. That rule stands; Δ's share is a
   column. Whether 0.36 is small is the user's to judge.
3. Reading A is largely measured: KL 0.0932 and 0.0926 on E40 step 10 (E44), and do-nothing against least squares 0.1181 and
   0.1177 on gold step 100 (E61). The plan names reading A and a gain measure, so one short stage reads the gain under A on
   E40 step 10. *Fable advised dropping it; kept because the plan names it and it costs about 10 minutes.*
4. The adversary waits for a split that reaches the faithful line: three seeds cost ~2.5 h per state, and a state that loses
   the update needs no adversary to fail.

## What runs

Four updates, in order: E40 step 10 (the gate), gold 1e-5 step 100, template 1e-6 step 100, E40 step 100. The plan's "E61's
gold and template states" are these; each is the arm's last state.

Per update: W0 and do-nothing reads, stage S, the often-on half, the rarely-on half, the matched random change, least squares,
adjoint. In-span is not read as a state: with Δ off its masked forward is least squares' (`site_change`), so only its Δ share
differs, and that comes from stage S. Row sets are in the predictions file. Σ stays E44's maths rows for every update.

| number | W0 | do-nothing | positive control | blind to |
|---|---|---|---|---|
| r2, reading B | no denominator | 0 | often-on half 0.06 on E40 step 10 | anything that needs Δ, which is off |
| ρ | no denominator | undefined (c = 0) | often-on 0.13, rarely-on 0.004 | whether the carried part is the useful part |
| gain kept | 0 | B: 0; A: near 1 | FT clean = 1 by definition | what a mean over all positions hides |
| Δ share | 0 | 1 | stage S against E44 | input directions outside Σ's rows |

No state has been shown to pass this test on Qwen: it can fail a placement, and whether it can pass one is open.

## Tree

- `control_reproduce` misses: job stops, lead reads why.
- Teeth hold on E40 step 10 (expected): the test stands; digest says so. The rest are scored.
- A split reaches the faithful line on an update: it is the map candidate; run the adversary, three seeds, on that state.
- No split reaches it (expected): D1's first table says fixed-direction splits do not keep behaviour on this model. The plan
  then names teacher training as the map. Before that spend, the lead settles which masks it is judged under, since masks from
  its own weights cannot fail anything.
- No teeth on E40 step 10 despite the reproduction: D1's finding as the plan words it; teacher training is the map.

## Cost and saves

$0, one L40, `normal` QoS (the 16 GPU jobs in the queue are all held by their user, so nothing is queued ahead), ~50 min per
update, ~3.5 h with a 6 h limit. Host ~60 GB, GPU ~46 GB as Amendment 2. Results JSON after every read, every number in
`results/run.log`. Stage S files (~30 GB per update) are scratch on the node, removed after each update.

## Not run

Step 1; intermediate gold steps; in-span as a state; web rows and replays; the adversary; RL and null δWs, which do not exist yet.

## Departures found by the code writer (2026-11-03, before the run)

Predictions stay as filed. The template's rows are E57's 64 rows, not 128: the filed text says both, and `qr.a1_rows` makes 64, so its bootstrap is on 64. KD is read through W0's path and do-nothing separately, so the 1e-5 control can fail; the stop fires on E40 step 10 only. `--mem` is 110G, not 80G: Amendment 2 peaked at 74 GB and this cycle repeats four times.

## Amendment 1: the ceiling (2026-11-03, first of two)

No placement has reached the faithful line, and this plan says whether the test can pass anything is open. So one state with no placement at all: W0's masked forward with the whole update added, unmasked, at every matrix. It is a yardstick, not a map. Controls first on u1: the added term at 0 reads E68's KD, and the same hook without masks rebuilds the fine-tuned model. $0, one L40, debug QoS, ~25 min, beside the running job. Rules in `results/predictions_amend1.json`; "can pass" or "near" means teacher training under W0's masks asks a fair question, "cannot pass" means D1 blocked and the test must change. Not run: the ceiling under reading A, and any partial share of the update.

## Amendment 2: opening the masks (2026-11-03, the last)

Amendment 1 read "cannot pass" on u1 and u2 against the filed lean, with both its controls clean. That number blocks every reading under W0's masks, so it is run down first. Same forward, masks opened in steps, m = g + t(1 - g), t = 0.1, 0.5, 1, with and without the whole update, u1 and u2, first 32 rows. At t = 1 only Δ is missing, which separates masks from leftover. $0, one L40, debug QoS, ~20 min. Rules in `results/predictions_amend2.json`. Not run: Δ on with masks; which activations the masks change; other rows.
