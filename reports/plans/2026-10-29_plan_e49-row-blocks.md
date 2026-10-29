# Plan: does one block of 128 rows, or the adversary's start, decide where full-matrix SGD sits? (E49)

*2026-10-29, plan writer for the lead. Decision: `decisions.md`, last row (card `decision-pile-rows`, options (a) and
(b), user 2026-10-29: "yes, run the free cluster job on the row blocks - (a) and (b)"). Evidence:
`reports/analyses/2026-10-29_analysis_e39-amendment3.md` (per-block table, "How the verdict is formed", the card and
Fable's advice); `reports/analyses/2026-10-29_analysis_e47-claim-tests.md` (the L40 readings). Code
`experiments/row_blocks/{run,predictions}.py`, `src/commute/lowercase.py` (new), `harness.at` (new); sbatch
`cluster/row_blocks.sbatch`. Words as in `GOALS.md` → Terms.*

## The question, in plain words

The paper's adversarial check gave the two SGD teacher-trained decompositions opposite verdicts on two sets of 256
Pile rows: they fail on rows 0-255 and pass on rows 256-511, and the fail sits on one block of 128 rows, rows 128-255,
where they read about 2.9. Two things could make one block decide: the adversary's random start on that block (one
start was read), or the rows themselves. And with only four blocks read, we cannot say whether full-matrix SGD sits
with the Adam fine-tune, which fails every block, or with the LoRA, which passes. The user asked for two free cluster
readings: (a) rows 128-255 again at two more adversary starts, for the unchanged decomposition W₀, both SGD
teacher-trained decompositions and the LoRA's; (b) E39's main states on four new blocks, val rows 512-1023, at three
starts each, with each block's ratio to W₀ on the same block and start, and the count of failing blocks per state.

Restated in the Terms' words: the states are **teacher-trained decompositions** (teacher training of the alive
subcomponents towards a fine-tuned model, Γ and Δ frozen) of four fine-tunes of E37's lowercase task, and the
**do-nothing** states of three of them (the fine-tune's δW put in Δ). Nothing is trained in this experiment except the
three fine-tunes, regenerated to make their do-nothing states. The measurement is the existing check, read per block.

## Design check (`/design`)

- **Goal.** `GOALS.md`'s weak version under reading A: can teacher training keep the decomposition faithful along a
  broad update? E39 now says "Adam fails, the LoRA passes, SGD unresolved". This run resolves SGD, and says whether a
  single adversary start can decide a verdict near the lines, which bears on every check in the project.
- **Nearest tried.** `TRIED.md` E39 Amendment 3 (the SGD states on rows 256-511: both pass; the fail sits on one block)
  and E47 (multi-start readings on rows 0-511, the L40). Neither read any block at more than one start for the SGD
  states, nor any row past 511. New here: starts 1-2 on the hard block, four new blocks at three starts, and per-block
  failing counts.
- **Patch or step.** A step: it decides how E39's line closes (Fable's three outcomes, below), and whether multi-start
  readings become a project-wide question.
- **Decisions relied on.** The check stays as it is project-wide (the decision's "(d) held"): E39's fixed lines on the
  ratio to W₀, pass ≤ 1.40, fail ≥ 1.61 (row 65 (3), "the fixed line on the ratio to W₀"). The per-block median and
  the failing-block count are readings for this card, not a new check. No departure.
- **What changes what we do next.** Each branch of the tree below names its next step. The two that matter: SGD fails
  two or more new blocks → option (d) goes back to the user; SGD fails at most one and Adam all four → E39 closes.
- **Apparatus control first.** Every job reads W₀ first and alone and stops on a miss. Job a's W₀ must equal E47's L40
  readings to their rounding, and two saved states must equal E47's L40 readings, so job a checks card, code and files
  at once. Each b job has a positive control, the Adam do-nothing, which must fail before the states are read.
- **Approved class.** Approved by the user in chat on the decided card (2026-10-29), and class 1 (GPU passes of the
  existing harness on saved states) plus three regenerated fine-tunes by filed recipes. No sign-off needed.

**One departure from the brief, and why.** The brief asked to regenerate and save the teacher-trained decompositions.
They are already saved, from the runs whose verdicts this card is about: `broad_fragility/results/large/`
`trained_step1000.pt` (E39, Adam), `tt_sgd_step1000.pt` (A1), `a2_tt_sgd_r8.pt` (A2), `a2_tt_lora_s0.pt` (A2).
Amendment 3 regenerated the SGD ones and reproduced their checks to four digits, and E47 already read the Adam and
LoRA files on the L40. Teacher training regenerated on the L40 would be a different state, with an unmeasured
cross-card difference from the one that failed. So the run loads the saved files (sha256 pinned in `run.py`) and
regenerates only the three fine-tunes, whose δW the do-nothing states need and which were never saved. The SGD state
held to rank 8 (A2) is included: it was the near-line one on rows 256-511 (1.388).

## States

| state | source | differs from the reference end state |
|---|---|---|
| W₀ | the released decomposition | — |
| `tt_adam` | `broad_fragility/results/large/trained_step1000.pt`, bf16 offsets | the H100 checks read the fp32 state in memory; this is its bf16 rounding (E47 read this file on the L40: 2.171 against 2.201) |
| `tt_sgd` | `.../tt_sgd_step1000.pt` (A1) | as above; never read on the L40 |
| `tt_sgd_r8` | `.../a2_tt_sgd_r8.pt` (A2) | as above; never read on the L40 |
| `tt_lora` | `.../a2_tt_lora_s0.pt` (A2, seed 0) | as above (E47 L40: 1.326 against 1.273) |
| (all teacher-trained states) | | bf16 rounding of the saved states vs the fp32 in-memory H100 verdicts (Adam 2.171 vs 2.201) |
| `nothing_adam`, `nothing_sgd`, `nothing_lora` | the fine-tunes regenerated in job `regen` (`lowercase.finetune`: E37's Adam 3e-5; A1's plain SGD 0.01; E39's rank-8 LoRA, Adam 1e-4; 300 steps of 8 lowercase rows), saved fp32 to `row_blocks/results/large/dW_<name>.pt` | trained on the L40, not the H100. The teacher-trained states copied the H100 fine-tunes. A do-nothing here is a reference that must fail, so a small difference does not matter; the gain check pins it (L40 − H100 measured −0.00047 Adam, −0.00016 LoRA, E47 t2c) |

The card is the other difference. On the L40 the same seed starts the adversary at the same random masks, but the
attack is not the same as on the H100 (E47: one state moved 64% between cards, others 0.5-13%). So every ratio's
reference, W₀ on the same block at the same seed, is read in the same job, and no L40 ratio is compared with an H100
line except as the reported card check below.

## The measurement

**Block ratio**: a state's PGD KL on one block of 128 Pile val rows (the harness adversary, 20 sign steps of 0.1 from
the random start of `torch.manual_seed(seed)`, one adversary shared by the block's 16 batches of 8;
`faithfulness.harness_pgd`) over W₀'s KL on the same block at the same seed, same job. **Pair check**: E39's 256-row
form on two neighbouring blocks, mean KL over W₀'s mean. **Lines**: E39's, pass ≤ 1.40, fail ≥ 1.61, inclusive.

**Blind to, and noise floor.** One block is one adversary over 128 rows: the check is a lower bound on fragility, and
a block reading can move with the start. W₀'s own spread over starts 0-2 on the L40 is 0.087 on rows 0-127 and 0.46 on
rows 128-255 (W₀ reads 1.0007 there at seed 1, against 0.72 and 0.68). A high W₀ reading lowers every state's ratio at
that seed. That is part of "the start", and the KLs are in the results file to see which side moved. E39's lines
were set from the H100's floor of 0.075; the L40's is 0.209 over rows 0-255, so a single-start verdict is noisier
here, which is why (b) scores the median of three starts. The same file on the same rows reads 0.5-4.2% apart between
the cards (four pairs).

## Stage regen (job 1, alone, ~5 min)

| check | reads | must read | else |
|---|---|---|---|
| card | `torch.cuda.get_device_name()` | contains "L40" | stop |
| G_gain_adam, _sgd, _lora | held-out gain on 64 lowercase val rows | within 0.005 of 0.1301, 0.13671, 0.1479 (H100); expected 0.1296, ~0.137, 0.1477 | stop |

Saves `dW_{adam,sgd,lora}.pt` (113 MB each, fp32; temporary name, then renamed) with their sha256 in
`results/regen.json`, and caches Pile val rows 512-767 and 768-1023 for the b jobs (so concurrent jobs do not stream
the same file).

## Stage a (job 2, ~21 min): rows 0-255, the hard block at three starts

| order | number | must read / expected | else |
|---|---|---|---|
| 1 | W₀ KL: rows 0-127 at seed 0; rows 128-255 at seeds 0, 1, 2 | 0.6671; 0.7164, 1.0007, 0.6846 within 1e-4 (E47 L40 logs) | stop (G_w0_a) |
| 2 | `tt_adam`, both blocks, seed 0: mean KL; pair check | 1.50182 within 0.1% (E47 L40), and ≥ 1.61 (expected 2.171; per block ~1.7, ~2.7) | stop (G_adam_a) |
| 3 | `tt_lora`: rows 0-127 seed 0, rows 128-255 seeds 0-2 | rows 0-255 mean KL 0.91745 within 0.1% (pair 1.326); hard block 1.3-1.6 at seed 0 (H100 1.42), 1.1-1.5 at seeds 1-2 | stop (G_lora_a) |
| 4 | `tt_sgd`, `tt_sgd_r8`: the same four reads | pair check at seed 0 within 10% of the H100's 2.0675 and 2.1218 (G_card, reported, no stop); hard block at seed 0 2.4-3.2 (H100 2.93, 2.95); at seeds 1-2: 1.8-3.0 under "rows", ≤ 1.40 under "start" | — |

G_card is no stop: a miss means this card's seed-0 attack does not reproduce the H100's on these files, which is
itself a start effect, read in the tree.

## Stage b (jobs 3-8, ~18 min each): val 512-1023, four new blocks, three starts

One job per pair of blocks and seed: `b512s0`, `b512s1`, `b512s2` (rows 512-639, 640-767), `b768s0..2` (rows 768-895,
896-1023). Each job reads, in order:

| order | number | must read / expected | else |
|---|---|---|---|
| 1 | W₀ on both blocks (PGD and CI-masked KL) | PGD in [0.45, 1.5] (L40 known 0.620-1.001), CI-masked within 15% of 0.34 (known 0.331-0.352); expected PGD 0.6-1.0 | stop (G_w0_b) |
| 2 | `nothing_adam`, both blocks; pair check | ≥ 1.61 (expected 3-6; E37 4.87) | stop (G_nothing_b) |
| 3 | `nothing_sgd`, `nothing_lora` | SGD 1.3-3.8 per block, may pass one (A1 blocks 1.34, 3.83); LoRA 2-3.5 (E39 2.65) | — |
| 4 | `tt_adam`, `tt_sgd`, `tt_sgd_r8`, `tt_lora` | per-block medians: Adam 1.7-2.7; plain SGD 1.1-1.45, at most one block near 2-3; rank 8 1.25-1.5; LoRA 1.1-1.45 | — |

Every read is a block ratio. The scoring stage (`run.py score`, CPU) forms per state: the ratio per block and seed;
the **median over the three seeds per block**; the **failing-block count**, blocks whose median is ≥ 1.61 (between
the lines is not a fail); seed 0's own count beside it, as a single-start reading like E39's; and the pair checks per
pair and seed, the 256-row-style verdicts. The SGD do-nothing's own count is the power of the check on SGD's update:
on a block where it does not fail, an SGD pass carries no weight. Any block median in [1.53, 1.69] is marked "near
the line" (bf16 rounding of the saved states plus the card difference can move a reading by that much).

## Predictions

Filed in `experiments/row_blocks/results/predictions.json` (`run.py --predictions`) before any job. Rules were run on
synthetic files for their edge cases before filing: empty files, a stopped job, a non-finite ratio, values exactly at
1.40 and 1.61 (inclusive), a fail at one seed only (the median ignores it), one SGD state "rows" and the other "mixed".

- **P1 (a): the hard block's SGD fail belongs to the rows, not the start.** Per SGD state, on rows 128-255 at seeds 1
  and 2: "rows" if both ratios ≥ 1.61, "start" if both ≤ 1.40, else "mixed". Holds iff both states read "rows";
  misses otherwise; unscored if job a stopped or a ratio is not finite. Absolute lines. *Lean: low-medium.* The block
  reads high for 13 of the 14 states over 1.40 on the first rows, and for the SGD do-nothing (3.83); against it, W₀'s
  own reading there varies most over starts.
- **P2 (b): the Adam teacher-trained decomposition fails all four new blocks** (median ≥ 1.61 on each). *Lean:
  medium*: its known blocks read 1.69-2.68, the lowest near the line.
- **P3 (b): plain SGD's fails at most one of the four.** *Lean: medium-low*: three of its four known blocks read
  1.11-1.26, one 2.93.
- **P4 (b): the LoRA's fails none.** *Lean: medium*: its known blocks read 1.11-1.42.
- **P5 (b): SGD held to rank 8 fails at most one.** *Lean: medium-low*: its known blocks read 1.25, 1.36, 1.42 besides
  2.95; a median between the lines does not count.

P2-P5 are scored only when all four blocks have three finite ratios (all six b jobs finished).

## Decision tree

Fable's three outcomes, as the data would look:

1. **The start** (P1 misses as "start"): both SGD states read ≤ 1.40 on rows 128-255 at seeds 1 and 2, e.g. 1.2-1.35,
   whatever seed 0 reads. Next: a decision card for the user, "multi-start readings for every line", since single-start
   verdicts in the 1.2-2.2 band could flip (the card's list: E41's step-10 copies, E42's least-squares split, E43's
   step-1,000 margin, E34's SGD student). Per-row KL is not run.
2. **The rows** (P1 holds): both SGD states ≥ 1.61 at seeds 1 and 2, e.g. 2-3, with the LoRA at 1.1-1.5. Next, by
   this tree and at $0: per-row KL under the hard block's final adversarial mask for `tt_sgd` and `tt_lora`, and the
   fine-tune's per-row damage there, one cluster job with its own plan amendment and predictions (later job; not in
   this plan's code).
3. **Mixed**: the states disagree or land between. No per-row KL; the b reading decides.

And for (b):

4. **SGD with Adam** (P3 misses): plain SGD's median fails on two or more new blocks, e.g. two blocks at 1.7-2.9.
   Option (d) (labelled SGD at step 1,000 and the SGD-trained LoRA, ~$1.5 on the H100) goes back to the user as a card:
   the decision held it, so it is not run on this tree's say-so.
5. **SGD passes on most rows** (P3 holds, P2 holds): SGD fails 0-1 blocks and Adam all four. E39 closes "Adam fails,
   the LoRA passes, SGD passes on most rows"; (d) is dropped; TRIED and the E39 card say so.
6. **Adam not on every block** (P3 holds, P2 misses): not in Fable's advice. Tell the lead at once; it goes to the user
   before anything else on the broad line.
7. **Low power** (beside 5 or 6): the SGD do-nothing fails at most two of the four blocks. The SGD passes are then
   reported as "where the check can fail SGD's update at all, ...", and the card says so.

Stops, each told to the lead at once: a missed control in any job (W₀, the Adam/LoRA reproduction, the Adam
do-nothing), a lost file, or a P2 miss (branch 6). P4 missing (the LoRA failing a new block) is reported in the
result, not a stop. A gate failure in job a stops job a only; the b jobs do not depend on it, but the lead should not
submit them until job a's W₀ and Adam controls have passed (they share the apparatus).

## Tolerances and their sources

| tolerance | value | source |
|---|---|---|
| regenerated gains | 0.005 absolute | E39's `gain_tol`; L40 − H100 measured −0.00047 and −0.00016 (E47 t2c) |
| W₀ per block, job a | 1e-4 absolute | E47 L40 logs (four decimals); E47's L40 jobs agreed to 17 digits |
| Adam, LoRA mean KL, job a | 0.1% relative | E47's `f_repro_tol` |
| SGD on the L40 vs the H100 (reported) | 10% relative | the same file and rows read 0.5-4.2% apart between cards (four pairs); 64% for one state in E47 t4, hence no stop |
| W₀ bands, b jobs | PGD [0.45, 1.5]; CI-masked 0.34 ± 15% | the L40 readings of the four known blocks: PGD 0.620-1.001, CI 0.331-0.352 |
| lines | 1.40 / 1.61 | E39 (H100 floor 0.075); the check unchanged project-wide |

## Cost and routing

All on the MATS cluster, `debug` QoS, L40, $0. Each job ≤ 30 min walltime (`--time=00:30:00`, 32G host RAM as E47's
jobs, 8 CPUs). Per block read ~62 s on the L40 (E47 logs), job start ~1.5 min.

| order | job | reads | est. min |
|---|---|---|---|
| 1 | `regen` | 3 fine-tunes, gains, saves, row caches | 5 |
| 2 | `a` | 18 block reads | 21 |
| 3-8 | `b512s0`, `b512s1`, `b512s2`, `b768s0`, `b768s1`, `b768s2` | 16 block reads each | 18 each |
| 9 | `score` (CPU, `fleet/cpu-run`) | none | <1 |

About 135 GPU-minutes. After `regen`, submit `a`; once its W₀ and Adam gates have passed (its log, minutes in), the six
b jobs can go in together if the queue allows. Each job writes its results file after every read, and a resubmitted
job resumes it (W₀ re-read and compared to 1e-6 first).

## Order of work for the fleet agent

1. Done here: code, predictions filed and committed, this plan.
2. A fresh Opus review of `experiments/row_blocks/run.py`, `src/commute/lowercase.py` and `harness.at` (the brief's
   rule for a run of new measurement code); fix what is real.
3. `/pod-runner` with this plan: jobs in the order above; report at each job's end with its gates; stop and tell the
   lead on any missed control.
4. `run.py score`; the analysis report by `/writeup` (per-block figure: each state's ratio per block, three seeds and
   the median, lines drawn; the hard block's three seeds beside it); TRIED row; E39 card and PENDING lines.

## Not in this plan

- Per-row KL and the fine-tune's per-row damage (only by branch 2, as a later job).
- Re-blocking rows 0-255 shuffled; rows past 1023; other cards; the H100.
- Option (d)'s stages; any new teacher training; a second batch order.
- A new row rule for the project's check (Fable: it would silently re-score about fifteen experiments).
- The SGD-held-to-rank-8 do-nothing; the stochastic and CI-masked ratios of the states.
