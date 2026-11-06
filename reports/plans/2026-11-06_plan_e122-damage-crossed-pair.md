# E122: why the copy of the broad Adam fine-tune fails the adversarial check (the damage-crossed pair)

*2026-11-06, regating-lead's writer. User-approved task (2026-11-06): "gap (a): find the cause". Design Fable's, adopted
by the lead (`~/commute-data/scratch/regating-lead/fable-strategy/advice.md` § 4(a), overnight plan item 3). Code
`experiments/damage_crossed/{run,rules}.py`, filing `results/predictions.json`, job `cluster/damage_crossed.sbatch`
(MATS `debug`, L40, $0, about 45-50 min of work in 2-3 resumable jobs). Not gated: $0 and under 6 h.*

## The question, in the Terms' words

Teacher training (component training with loss KL(fine-tuned model ‖ student), Γ and Δ frozen, E37's recipe) of the
lowercase fine-tune gives a decomposition that keeps the fine-tune's behaviour but fails the paper's adversarial check
when the fine-tune is full-matrix Adam (2.20 on Pile rows 0-255, 1.88 on 256-511) and passes when it is a rank-8 LoRA
of the same task (1.27, 1.21). The two fine-tunes differ in two ways at once: the recipe (all 24 matrices against a
rank-8 LoRA) and the damage, KL(W₀ ‖ W₀ + δW) on plain Pile text: 0.234 against 0.124. Which one makes the check fail?

**Design: cross them.** Cell c1 keeps Adam's recipe at the LoRA's damage: the same Adam fine-tune stopped early. Cell
c2 keeps the LoRA's recipe at Adam's damage: the same LoRA trained on, on fresh rows. Each is copied by E37's recipe and
read on both row sets. Damage is the cause if c1 passes and c2 fails; the recipe is the cause if c1 fails and c2 passes.

## The record, checked against the files

Every number Fable's § 4(a) cites holds, with two corrections. Adam copy 2.20, damage 0.234; LoRA copy 1.27, damage
0.124 (`broad_fragility/results/broad_fragility.json` reads `trained`, `tt_lora`); plain-SGD copy 2.07 on rows 0-255,
1.18 on 256-511, damage 0.090 (`amend3.json`); reverting the MLP takes the Adam copy to 0.997, reverting a half removes
0.63 (rarely important), 0.96 (often important), 0.97 (random) of the excess (`broad_fragility.json`); damage-matched
random moves 16.5 and 27.2 (`amend2.json`); E36 Amendment 1 read steps 10/100/300 (`TRIED.md:67`); E37's untested
alternative reading is at `reports/analyses/2026-10-26_analysis_teacher-spread.md:121`. Corrections: (1) rows 0-255 and
256-511 are E39 Amendment 3's two sets (and E47 t2c's), not E49's, which read 128-255 and 512-1023. (2) The "seven
times more diffuse" (`claims.html:17`) is E21's component-training update on the emoticon task, not this fine-tune.
Fable's "step ~100" for damage 0.12 is an estimate: no damage was recorded along the Adam run, so stage 2 measures it.

The record already leans both ways. On these rows every state with damage over 0.2 failed and every one under 0.06
passed (E39 Amendment 2), which favours damage. But two full-matrix SGD copies at damage 0.090 and 0.126 failed rows
0-255 (2.07, 2.12) and passed rows 256-511 (1.18, 1.39), so a full-matrix recipe at the LoRA's damage can fail.

## Design check (`/design`, a line or two each)

- **Goal.** `GOALS.md`'s second question: if the copy is not faithful, is the decomposition brittle or is the update
  new? A damage cause says the copy fails where the update does a lot elsewhere; a recipe cause says some update shapes
  cannot be copied faithfully at any size. Inside the user's task; no sign-off needed.
- **Nearest tried.** E39 and Amendments 2-3 (`TRIED.md:79-81`): matched gain, not damage, and never crossed the two.
  E37 Amendment 1's adjoint start rose 1.05 → 1.84 as its damage rose 0.05 → 0.24, one recipe only.
- **Patch or step.** A step: it names the cause of an existing finding, with one decisive pair.
- **Definitions.** No reuse verdict; `decisions.md` rows 81, 87 do not apply. The check is row 34's (the paper's
  setting); row 75 (the check cannot fail SGD's update) is why every pass needs its do-nothing to fail. No rule added
  to any adopted definition: the cell rules are this experiment's own, filed before it runs.
- **What changes next.** "damage" makes the write-up's third sentence "fails when the update's damage exceeds about
  0.2, whatever the recipe"; "recipe" keeps "fails for Adam's full-matrix change" and points at what the LoRA's shape
  does; "both fail" or "both pass" says it takes both, and the excess maps say where.
- **Controls, first and alone.** C1 W₀ on both sets against the L40 readings; C2 the saved Adam and LoRA copies on both
  sets, which must reproduce E49's and E47's L40 readings (the two existing corners, fail and pass). C4 the two
  fine-tunes' gains at step 300. C5 the copy recipe on this card: E37's copy of the full Adam fine-tune, trained in-job,
  must fail on set A before any cell is copied. What differs from the references' end state: the card (L40 against
  the H100 that trained the saved copies), which C5 tests; nothing else (same rows, batch order, rate, steps).
- **Saved for audit.** The chosen fine-tunes (fp32), the copies (bf16 offsets) and every per-position map.

## Settings, each sourced

Fine-tunes: `lowercase.arm_b_states(..., "adam", 0, ...)`, Adam 3e-5, batch 8, E37's 2,400 rows once, clip 1
(`lowercase.py:21, 77`; `broad_fragility/run.py:58-60`); LoRA rank 8 on all 24 matrices, Adam 1e-4, seed 0
(`lowercase.py:57`; `broad_fragility/run.py:82-84`); fallback rate 3e-4 is E39's grid's next rate (`run.py:82`). The
LoRA's longer run reads fresh rows in E37's way: Pile train 0-32767 lowercased (`row_copy.lowercase_rows`) make 32,970
rows, whose first 2,400 equal E37's (measured on CPU, 2026-11-06); 4,000 steps use 32,000. Copy: `row_copy.copy_on_rows`,
Adam 1e-5, batch 8, 1,000 steps, batch order 0 (`broad_fragility/run.py:61-64`). Check: `faithfulness.harness_pgd`,
20 steps, seed 0, one adversary per 128-row block, lines 1.40 and 1.61 (`run.py:66-68, 85-86`). Damage and s:
`row_copy.mean_kl`, `read_state`, 64 rows (`run.py:136-138`). L40 references: `row_blocks/results/a.json`,
`claim_tests/results/t2c.json`. Shared code changed: `faithfulness.harness_pgd` gained a `loss` keyword (None = the
authors' `recon_loss_kl`, as before); its callers (`broad_fragility/run.py`, `row_blocks/run.py`,
`teacher_training/amend5.py`, `gamma.py`, `faithfulness.faithfulness`) never pass it, so they are unchanged.

**The band, measured in the job.** Stage 2 reads damage every 10 Adam steps (10-300) and every 100 LoRA steps
(100-4,000). D_A and D_L are the two recipes' damages at step 300; the band's half-width is w = (D_A − D_L)/4, so the
two bands are separated by half the gap. c1 = the Adam checkpoint nearest D_L, c2 = the LoRA checkpoint nearest D_A; a
cell whose nearest checkpoint, or whose copy's damage, falls outside its band is not read ("off target"). If no 1e-4
LoRA checkpoint reaches c2's band, the same run at 3e-4 is tried, then c2 is reported not run.

## Stages and time (L40)

| Stage | What | Time | Source |
|---|---|---|---|
| 0 | sha256 of the saved copies (no model) | s | |
| 1 | W₀ on A, B; saved Adam and LoRA copies on A, B (6 checks) | ~12 min | 116 s per check, `claim_tests/results/t2c.json` `states.adam_tt.seconds` |
| 2 | Adam curve; lowercasing 32,768 rows; LoRA curve (and 3e-4 if needed) | ~8 min (~13) | Adam 300 steps 16-17 s (`t2c.log:42`, `row_blocks/results/large/regen.log:34`); LoRA 300 steps 15 s (`t2c.log:46`); lowercasing 52 s (CPU, measured) |
| 3 | recipe copy, read on A | ~4.5 min | copy 152 s (`t2c.log:53-61`) |
| 4, 5 | each cell: do-nothing on A, copy, copy on A and B | ~8.5 min each | as above |
| 6 | maps, score | s | |

About 42 min (47 with the fallback) plus ~1 min per job to load and reread W₀'s first block. A unit starts only if it
ends within 26 min of a 30-min job, else exit 3: `fleet/resubmit --code 3 cluster/damage_crossed.sbatch`, 2-3 jobs.

## Numbers and what each should read

| Number | W₀ | Do-nothing | Control / expected |
|---|---|---|---|
| Check ratio, set A / B | 1 by definition | cells: ≥ 2.5 (E39: LoRA 2.65 at damage 0.121) | saved Adam 2.171 / 1.897; LoRA 1.326 / 1.213; recipe copy ~2.2 on A |
| Damage | 0 | = the fine-tune's | D_A ≈ 0.235, D_L ≈ 0.121, w ≈ 0.028 |
| Gain at step 300 | 0 | | Adam 0.1296, LoRA 0.1477 (L40) |
| c1 step / c2 step | | | ~50-150 of 300 / ~1,000-4,000 |
| Copy's damage / s | | | 0.97-1.03 of its fine-tune's / 0.95-1.00 |
| Map concentration | | | 1 under the null (excess spread evenly) |

**Predictions** (filed, `results/predictions.json`): P1 c1 passes (low confidence: the two SGD copies above), P2 c2
fails (medium-low). M1, reported: the saved Adam copy's excess is more concentrated at case positions than the LoRA
copy's (Fable's hypothesis), against a floor set by each state's own set A − set B difference.

## What each outcome means

- **c1 pass, c2 fail: damage.** The copy fails when the update's damage on plain text is high, whatever the recipe.
- **c1 fail, c2 pass: recipe.** Full-matrix Adam's change cannot be copied faithfully even at low damage; the LoRA's
  can at high damage.
- **Both fail.** Neither alone: the LoRA's pass needs both its shape and its low damage.
- **Both pass.** The fail needs both: high damage and full-matrix Adam together.
- **No call.** A cell off target, mixed across the two sets, invalid, or a pass whose do-nothing does not fail. The
  readable cell's half is reported as a lean only. A stop (C0-C5, rerun) scores nothing.

## Noise floor and blind spots

One adversary start per read. Set A against set B on the L40: Adam copy 2.171 / 1.897, LoRA copy 1.326 / 1.213; plain
SGD on the H100 2.07 / 1.18. W₀'s seed spread is 0.075 (H100) and 0.21 (L40). A ratio within about 0.2 of a line is
near it, which is why a cell needs both sets on the same side. Batch-order spread of a copy, 14% (E37), is not sampled.
Blind to: gain, which is not matched (c1 keeps less lowercase behaviour, c2 more); where on Pile the damage sits (one
mean over 64 rows); recipes other than a rank-8 LoRA; c2's fine-tune saw 32,000 rows while its copy saw E37's 2,400.

## Not run

Γ's on-count split of the excess map (Fable's second split); more adversary starts or batch orders per copy; an
SGD cell (E49: the check cannot fail SGD's update); a damage grid beyond the two cells; any narrow-task check.

## Departures from the literature

The check is the VPD paper's adversary in its own code, read per 128-row block as E39 did, with E39's fixed lines
(1.40, 1.61) set on the H100's floor and kept here on the L40 (E49 did the same). The per-position map applies the
authors' loss unchanged and records each position's term; the paper reports only the mean. Teacher training is this
project's, not the paper's. LoRA follows Hu et al. 2021's init (B = 0) at rank 8 on all 24 matrices.
