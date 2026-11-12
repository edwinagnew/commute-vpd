# E169 plan: the cheap readings on the Qwen pair library (cell P3.7, run 1 of 2)

## What and why

We read five cheap readings on every saved update of qwen-lead's pair library and show, for each pair, side A minus
side B as one scorecard figure. The cell is a description (the user, task file 2026-11-12: "One scorecard figure, no
reuse or new labels"): no update is labelled reuse or new, no decomposition or share is read, and the readings are
never joined into one score. The question the figure answers is whether these readings separate the two sides of a
pair by more than what a random change of the same size range reads, or whether they move only with size (card
`card.md`, rivals A and B).

**Pairs** (A minus B): copy minus cipher, known minus invented facts (E165, run 3, 300 steps), gold minus gold shuffled,
template minus template shuffled (E164, run 2, 100 steps). The twins are "real minus shuffled". Seeds 0-2, paired by seed.
**Saves**: "late" = the last save (step 300 built, step 100 twins) and "matched" = the 5x save (steps 7-19). The
template twin has no 5x save: late only, and its row is marked as failing the library's fit rule. The 15x saves are not
read. That is 42 states, 5.6 GB each.

**Readings per state** (the functions part 2, E92, E162 and E166 used; nothing re-implemented):
- update size: `qwen_comparators.update_size` total `ratio_norm`, ‖δW‖/‖W0‖ over the 196 matrices;
- prior: W0's median rank of each answer's first token on the side's rows (`read_state` `rank_W0.first.median_rank`),
  and beside it the library's stored mean log P per answer token (`w0.prior` in the library JSON, not recomputed);
- steering: E92's `steer` share of the gain on the scoring half (fit on even rows, score on odd, E92's layer rule);
- effective rank of the update: `update_spectrum`, fp64, weighted by ‖δW_m‖² (E162);
- top-neuron overlap: `qwen_overlap.overlap`, top 480 of 6144 MLP neurons per block at the answer positions (E166).

**Rows.** Each side's own training rows, 128 rows of 256 tokens. Built sides: `qwen_pairs.own_rows`, the rows the library
read its output change on (CPU check: 4,519-7,491 answer positions and 923-1,743 answers per side). Twins: the arm's
training sequences (`qwen_d3.gold_train`, `qwen_library.template_pool`) through `qwen_pairs.packed`. The shuffled side
uses its seed's derangement (`library.twin`), so its rows differ by seed. Gold: about 15,000 answer positions and 157
answers per row set. Template rows could not be built on CPU: E45's Job asks for a CUDA device. The pod builds them.

## Expected values

| | do-nothing (zero update) | random floor | real states |
|---|---|---|---|
| size | 0 exactly | — | the library's \|δW\| over ‖W0‖: cipher about 2x copy, invented about 2x known |
| prior | unchanged from W0 | — | log P (stored): copy −17.2 against cipher −16.2, known −7.1 against invented −6.8, gold −2.3 to −2.6 against −2.7 to −2.8, template −1.0 to −1.1 against −1.8 to −1.9 |
| steering | no layer (gain 0) | not measured; may be undefined | E166: 0.06-0.91 on 19 Qwen updates |
| effective rank | None | about 1,810-1,816 at either norm (E166) | E166: 1,060-1,440 |
| overlap | 1.0 exactly, on_both nan | about 0.998 (small reference), 0.97-0.99 (large) | E166: 0.48-0.98 |

## Stages (per pair, files `results/<pair>/stage<n>.json`, a line `STAGE n DONE <pair>` after each)

0. **Do-nothing control, alone and first.** A zero update goes through the same path as a real state, on side A's rows.
   It must read overlap 1.0 exactly at every block, on_both nan, size 0, max |Δlog P| ≤ 1e-5, no steering layer and
   erank None. A miss stops the job (exit 1).
1. **Random floor, before any real state.** The job fetches the set's smallest-norm state (copy s0 5x, ‖δW‖ 0.173) and
   largest-norm state (cipher s0 last, 5.09) once and keeps only their per-matrix norms. Random changes at those norms
   are then read on both sides' rows (seed 0's for a shuffled side): 4 overlap draws, 2 steering draws and 2 Gaussian
   erank draws per reference. Per pair, the band is the largest gap between side A's and side B's floor means over the
   two references, plus the largest draw range (`rules.floor_band`). The job stops (exit 1) if the overlap or erank band
   is unreadable, or if the overlap band is over 0.2. A steering band of None is recorded and that column then reads
   "unreadable".
2. **Every state, streamed.** Each state is fetched from W&B (the next one downloads while this one is read), then:
   sha256 against the library's record, ‖δW‖ against its record (relative 1e-6; else exit 1), the five readings, and the
   file is deleted. A state already in `stage2.json` is skipped, so rerunning the same command resumes.
3. **CPU, after every pod** (`plot.py`): join, the rules, `results/scorecard.json` and `figures/scorecard.png`.

## Filed rules (`results/predictions.json`, applied by `rules.py`; 29 edge cases pass)

- Floored readings (steering, erank, overlap) per pair and save: "beyond floor" when all three seed gaps are further
  from 0 than the band and share a sign; "inside floor" when |mean| ≤ band; "mixed" otherwise; "unreadable" for a
  missing value or band.
- Size and priors: "A higher" or "B higher" when all seeds agree in sign, else "mixed".
- Card call, at the late save on the two built pairs:
  - rival A: each built pair has at least one floored reading beyond the floor;
  - rival B: every floored reading is inside the floor, and size reads one sign;
  - otherwise no call.

  This is reported as a description. A do-nothing library reads "no call".

## Pods, time and cost

There are three H100 pods in parallel, one invocation each. Labels: `E169 stage: copy_cipher`, `E169 stage: facts`, `E169
stage: gold,template`.

- Set-up: about 40 min for a fresh Qwen pod.
- References: 11 GB at about 35 MB/s, about 5 min.
- Gaussian erank floor: about 5 min.
- Floor per pair: 16 overlap reads at about 10 s and 8 steering reads at about 40 s, so about 8 min.
- Per state: the fetch takes 5.6 GB ÷ 35 MB/s ≈ 160 s and overlaps with compute. Compute is about 3-4 min, mostly 196
  fp64 SVDs. E162 took 3.5 min per state on 16 CPUs; the card time is not measured. So budget about 4 min per state.
- Per pod:
  - copy_cipher and facts: 12 states each, about 1.8 h.
  - gold,template: 18 states and two floors, about 2.3 h.

That is about 6 pod-hours at $3.99/h, about $24. Allow $35 for a slow fetch or SVD; the cell allows $50.

**What each pod needs.**
- Sync the repo and run `bash cluster/pod_setup.sh` (uv sync, Python 3.13).
- `WANDB_API_KEY` in the environment (the fetches; an HF token only if HF limits the Qwen download).
- Qwen3-1.7B downloads from HF at its pinned revision on first load, about 3.4 GB.
- On the gold,template pod only, E45's files for `template_pool`, hard-link staged as the root `CLAUDE.md` says:
  `experiments/qwen_updates/results/large/S/a1/` (12 MB) and `.../large/P/a1_rows.json`. The library JSONs and
  `words.json` are committed and arrive with the sync.
- The container disk is 50 GB. At most two states (11 GB) are on disk at once, in `--states` (default `/root/states`).
- Run line: `E120_PRED_SHA=<sha256 of the committed predictions.json> .venv/bin/python
  experiments/qwen_library_readings/run.py --pairs <group>`.
- Download `results/<pair>/stage{0,1,2}.json` per pair as each stage lands.

## Not run

- No decomposition reading.
- The 15x saves.
- The template twin's matched save (it has none).
- The floor on each seed's shuffled rows (seed 0's rows stand for all three).
- A cross-card noise floor for the overlap.
