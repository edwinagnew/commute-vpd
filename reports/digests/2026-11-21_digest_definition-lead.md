# Definition-lead digest, 2026-11-21

P1.46, due 11-23: closed. Under wording A, the reuse score failed its known-answer test at VPD: a new edit spread thinly over many parts reads 0.98, about as high as the reuse edits. All of part 1's remaining work waits on your answer to ask 57.

## Cells
Since the last digest: P1.42 and P1.43 filled; P1.44, P1.45 and P1.46 closed. P1.41 (the gain-only read, due 11-23) is still open, but whip-lead cut it to one appendix line, and it waits on ask 57 like the rest. P1.40 belongs to sae-lead (`fleet/cell list definition-lead`).

## What ran and what it found
- **P1.46 (R under wording A, VPD).** We read R on the six hand-built edits and on random updates of the same size. The filed rule was that every reuse edit reads above both new edits. It failed:
  - Reuse edits: gain ×2 reads 1.00, gain ×10 reads 0.976, and the rebuilt circuit reads 0.859. The moved circuit sits under the noise floor, so it was not read.
  - New edits: the thin new edit reads 0.980 and the few-parts new edit reads 0.892.
  - Random updates: 0.29 and 0.30, as low as expected.

  Premise check: doubtful, because VPD may not separate the built edits at all. The card is `reports/decisions/2026-11-21_decision_r-under-a-built-fails.md` (log v425).
- **P1.45 (the causal test with gating at VPD).** It does not separate reuse from new on the built edits, and its random control fails. On the nine trained pairs the reuse side reads higher (descriptive). Nothing went to writeup-lead. Report: `reports/analyses/2026-11-20_analysis_e241-gating-vpd.md`.
- **P1.44: no call.** The run failed its reproduction control: R's noise between two cards is about 1e-4, against a 1e-5 tolerance. No bug was found.

## Ask waiting on you
- **57.** Under wording A, a new computation spread thinly over many existing parts reads 0.98 as reused, above the gain ×10 reuse edit (0.976) and the rebuilt circuit (0.86). Should wording A call such an edit reused (and its built label be revisited), or should A change so it reads new? The trained pairs, the copy-versus-lookup example and the five key tasks are held until you decide. sae-lead's pairs and compare-lead's P2.51 are held on the same ask.

## Money
This session: $0, with no pods and no Slurm jobs live.

## Friction noted
945 (the card template needs a per-experiment structure), 950 (job-gate and parallel pod shards), 952 (amendment E-numbers), 954 (sync ships staging folders), 955 (a slow H100 host), 957 (cell add refused a relay from whip-lead), 962 (pod set-up race), 967 (`fleet/numbers` breaks on filters that start with "-").
