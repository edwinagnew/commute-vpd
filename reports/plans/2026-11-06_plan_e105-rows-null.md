# Plan: E105, the same-label, different-rows null for set identity

*2026-11-06, baseline-lead. Code `experiments/rows_null/{run,score}.py`, shared steps `src/commute/known_facts.py`;
rules `experiments/rows_null/results/predictions.json`; job `cluster/rows_null.sbatch`. Approved class 2: the control
E103's plan and analysis name (`reports/analyses/2026-11-06_analysis_e103-known-answer-pair.md`, "Do the two arms carry
their updates on different neurons, beyond seed noise?"; `predictions_amend2.json` `.neurons.identity.null_not_run`).*

## The request in the Terms' words

Fine-tune W0 (Qwen3-1.7B) on each of two disjoint halves of TOFU's 217 real-fact rows, with E103 Amendment 1's recipe
and wordings, at stream seeds 42 and 142: four updates δW, all labelled "already known". Keep each at the matched step
(the step whose |δW| is nearest the template fine-tune's 0.4865, E91's rule). Read each matched update's carrying set
(the 1,024 MLP neurons with the largest first-order share of its gain) on E103's real rows and E92's TOFU rows, with
the code E103 used. Ask whether, with the label fixed, changing the rows moves the set as far as changing the label
(known against fictional) did in Amendment 2.

## Design check (`/design`)

- **Goal.** `GOALS.md`'s second question asks whether post-training teaches new capabilities or strengthens existing
  ones; the neuron baseline is the comparator beside VPD's readings. Set identity is the only neuron reading that
  separated the known-facts arm from the fictional one (Amendment 2, beyond seed noise on both row sets); this decides
  whether that separation reads the label or only the rows.
- **Nearest tried.** E103 Amendment 2 (`TRIED.md` row "E103, Amendment 2"): known seeds 0.864 / 0.901 (real / TOFU rows),
  fictional seeds 0.899 / 0.937, cross-arm at most 0.492 / 0.637. It could not tell label from rows: the arms differ in
  both. Amendment 1 against E103 (same rows and label, pair 0's answer form changed) shared only 0.69 / 0.81.
- **Patch or step.** A step: it closes the one null that keeps set identity a candidate reading. E103 has no amendment
  left, so it is a new E-code.
- **Decisions relied on.** The 2026-11-05 row that the matched step is a size control (kept here: every state is read
  at the matched step, as Amendment 2's sets were); row 154 (E103 runs as a label-fixed control). No departure.
- **What changes next.** "label" on both row sets: set identity is carried forward as the neuron baseline's reading of
  reuse against new, with this null beside it. "rows" on both: set identity is retired as a label reading for this pair,
  and the neuron baseline stays with its summary readings (all level). Mixed (label on TOFU, rows on real): only the
  TOFU-row reading, whose probe no arm trained, carries forward. "no call": reported, no amendment.
- **Control first.** C2n, alone, before any training: Amendment 2's matched state re-read on the real rows must give its
  stored gain (2.2331 within 1e-3 relative) and set (overlap ≥ 0.95), or the job stops. It is the path that produced
  every stored set the rule compares with. Then W0's sentence-reader tokens must reproduce their stored content level
  (−2.01584 within 1e-5), since every C1 gain is read against them. What C2n can fail on: a different model load, row
  packing, ranking code or saved state; it cannot fail on the training path, which C1 and the step checks guard.
- **Recipe differences from the reference (the full-row known arm, Amendments 1 and 2).** (1) Rows: 108 / 109 against
  217 (540 / 545 sequences, 33 / 34 steps an epoch against 67): the thing tested. (2) Exposure at the matched step: a
  half has seen about 0.95 epochs of its rows, the full arm 0.47, so each trained row about twice as often: named, and
  reported through the bridge below. (3) The run stops at step 64 (`CAP`, twice the full arms' matched step 32) instead
  of step 335: the rate at every step up to 250 is set by the step alone (linear warm-up), so nothing before the matched
  step differs; no end state is trained. (4) C1 is read on the half's own sentences, not all 217. Optimizer state,
  schedule position, batch size, wordings, reader and matched-size rule are the full arm's code
  (`commute.known_facts.fine_tune`, moved unchanged from `amend1.py`).
- **Early guards.** Per arm, logged with their readings: the two seeds' first batches differ by > 0.001 (Amendment 2
  read 0.477); step 1 loss within 1e-3 of the first batch, rate and |δW| exactly 0; step 2 |δW| in [0.00208, 0.00312]
  (Amendments 1 and 2: 0.00233, 0.00229); peak ≤ 44 GiB (they peaked at 38.5); a non-finite loss stops; the template
  norm must be passed within 64 steps and the record's argmin must be the matched step, else stop.
- **Saved for audit.** The four carrying sets and their gains (`results/sets/`), per-arm summaries and full training
  records, W0's sentence tokens, and the four matched δW in bf16 (`results/large/dW/`, 2.8 GB each).

## The concern: halves are not the full arm at half size

The within-half seed overlap and the cross-half overlap are on equal footing: both halves have the same row count,
recipe, step and exposure, so that comparison is not confounded. The cross-arm level x is not: it comes from full-row
states at 0.47 epochs (and fictional states at fewer). If doubling the exposure per row by itself moves the set, a low
cross-half overlap could partly be that. The bridge measures it for $0 from the stored full-row sets: each half state's
set inside the full-row known state's of the same seed. Near the full seeds' own 0.864 / 0.901: halving leaves the set
where the full arm had it and x is on the halves' footing. Well below, near the cross-half overlaps: row count or
exposure moves the set on its own, which points the same way as "rows" but weakens the x comparison; the report then
leans on within-half against cross-half only. A full-row arm at doubled exposure is not run (`Not run`).

## Stages, in run order (one job, one L40, `debug`, $0)

1. **C.** C2n alone (stop if missed); then W0's tokens on the sentence reader, all 217 rows, content level checked
   (stop if missed), saved to `results/large/w0_sentence.json`.
2. **A42, A142, B42, B142.** Per arm: first-batch check; 64 steps with the early guards; at the matched step C1, the
   content gain on the half's own sentences ≥ 1.2365 (half the fictional reference 2.4731, `amend2_ref.json`), else
   stop "no content gain to carry"; then the other half's gain (reported), both carrying sets, the bf16 save. Each arm
   writes `results/arm_<arm>.json` when done; a rerun skips finished arms.
3. **S.** The filed rule and the bridge, `results/score.json` (CPU only; `run.py score` reruns it).

## The numbers and what each should read

| Number | W0 / do-nothing | Control or reference | Expected |
|---|---|---|---|
| C2n: Amendment 2's matched state, gain and set, real rows | gain 0, empty set | stored 2.2331, overlap ≥ 0.95 | as stored, overlap 1 |
| W0's sentence-reader content level | −2.01584 | stored, within 1e-5 relative | as stored |
| First batches of the two seeds, per half | | differ by > 0.001 | differ, about 0.5 |
| Matched step and |δW|, per arm (reported) | 0 | full arms: 32 at 0.480 / 0.483 | 28-36, within 0.02 of 0.4865 |
| C1: own-half content gain at the matched step | 0, fails | bar 1.2365 | holds, 1.4-2.2 |
| Other-half content gain (reported) | 0 | own-half gain | 0.8-1.6, below the own-half gain |
| Within-half seed overlaps w_a, w_b | unread | full arms 0.864 / 0.901 | 0.80-0.92 |
| Cross-half overlaps (four) | unread | x = 0.492 / 0.637 | real 0.50-0.62; TOFU 0.70-0.80 |
| **Rule, real rows (scored)** | unread | stand-ins read label and rows | **rows** |
| **Rule, TOFU rows (scored)** | unread | stand-ins read label and rows | **no call** |
| Bridge: half inside full, same seed (reported) | unread | full seeds 0.864 / 0.901 | between cross-half and w |

## The filed rule (`score.py` `verdict`, per row set)

w_a, w_b = each half's seed-142 set inside its seed-42 set; cross = the four half-A against half-B overlaps; x = the
highest cross-arm overlap of Amendment 2 on that row set (0.4922 real, 0.6367 TOFU, stored). w = min(w_a, w_b), band =
1 − w: the share of a set a change of data order alone replaced, in the less stable half, measured in this job, as
Amendment 2's band was one minus the fictional seeds' overlap.
- **label** iff w − min(cross) ≤ band and min(cross) − x > band: every cross-half pair agrees within seed noise and clears
  the cross-arm level by more than the band.
- **rows** iff max(cross) − x ≤ band and w − max(cross) > band: no cross-half pair clears the cross-arm level by more than
  the band, and every one is beyond seed noise.
- **no call** otherwise; **unread** with an empty set (the do-nothing state) or any non-finite value.

The two cannot both hold. With the full arms' seed overlaps as the halves', label needs a cross-half minimum of at
least 0.729 (real) / 0.803 (TOFU) and rows a maximum of at most 0.628 / 0.735. *Edge cases*, run on CPU before filing
(`~/commute-data/scratch/baseline-lead/e105-writer/dry_e105.py`, `dry.out`; also `tests/test_rows_null_score.py`): all
sets identical label; an empty set, None, three cross values or a NaN x unread; exactly at each line the verdict holds
and one neuron (1/1024) past it reads no call; mixed pairs no call; unstable halves (w 0.6) no call. On the stored sets,
both halves stood in by the full known seeds read label on both row sets and half B stood in by the fictional seeds
reads rows on both, so the rule can go either way. The do-nothing state reads unread, so it cannot pass. A set blind to
the data (all four identical) reads label; it could not have separated the arms in Amendment 2, so label means the
label only beside that result.

**The lean, filed honestly: rows on the real rows, no call on the TOFU rows.** Amendment 1's state against E103's (same
rows and label, one answer form changed) reads no call on the real rows (cross 0.690 / 0.696 under the label line
0.729) and label on the TOFU rows (0.812 over 0.803) when stood in as half B. Changing every trained row is a larger
change, so the cross-half overlaps should fall below those: on the real rows, which hold each half's own questions, to
about 0.50-0.62; on the untrained TOFU rows to about 0.70-0.80, across both lines. Against the lean: Amendment 1's
states gained 0.9 as much on rows they did not train, so the update may be mostly the shared sentence-answer skill.

*Blind to.* Rank changes inside a set, neurons just past the cut, anything inside a neuron (the granularity limit). The
real-row probe holds each half's trained questions, so a half's set there is partly ranked on its own rows.

## Cost and time (measured, `experiments/known_answer_pair/results/amend2.log`, job 53316 on an L40)

A candidate step with its 217-row read took 2.94 s (lines 18-49), a step without a read 0.69 s (lines 52-64); neuron
reads 4-5 s on the real rows and 20 s on the TOFU rows (lines 77-80); a 5.64 GB fp32 save 21 s (line 50). Per arm:
about 32 candidate steps at about 1.8 s (half the rows read) + 32 steps at 0.69 s + 25 s of reads + about 10 s for the
bf16 save ≈ 2 min. Four arms ≈ 8.5 min, stage C under 1 min, start-up about 1 min: **about 11 min in one job**, limit
30 min, `--mem 24G` as E103's jobs (host peak 10.5 GB). GPU peak: training 38.5 GiB as Amendment 2, the read after it
about 32 GB (26.5 GB for the read plus the held δW). Disk: 4 × 2.8 GB in `results/large/dW/`.

**Gates.** Under the gate: expected $0 on the cluster `debug` QoS, about 11 min, one job under the 30-min cap (no split
needed). No review and no pre-run check (fast lane); the lead reads and labels the result.

## Stop rules for the runner

Any exit 1 writes `results/stopped.json`: report it before resubmitting. C2n or W0's reader missing: an apparatus
fault, not a result; stop and tell the lead. C1 missing on any arm: "no content gain to carry", the rule is not read.
A lost job: resubmit once (finished arms are skipped); a second loss goes to the lead.

## Hand-off

Code moved to `src/commute/known_facts.py` (E103's `fine_tune`, sequences, gains, row packing and carrying read);
`known_answer_pair/{run,amend1,neurons}.py` now import it and are otherwise as they ran. The analysis reads
`results/score.json`, `results/arm_*.json` and `results/sets/`.

## Not run

- A full-row known arm trained to the halves' exposure (about 64 steps) to put x on the halves' footing; the bridge
  stands in for it.
- A same-split pair for the fictional arm (rows changed at the other label).
- More than two seeds per half; a different split seed.
- VPD reads of the half states (qwen-decomp-lead's instruments).
