# E161: is the share's 0.77 to 0.82 on the code-text fine-tune stable, and what does a random update of the same size read?

*2026-11-11. Cell P2.6 (compare-lead, part 2, rung L2.9, 1 run, cluster debug, $0, due 11-12), un-parked by the user
on 2026-11-11 (brief `~/commute-agents/fleet/paper/briefs/2026-11-11_compare-gate-page-and-qwen-columns.md`, cell C).
Card: `experiments/share_stability/card.md`. Filed rules: `experiments/share_stability/results/predictions.json`.
Code: `experiments/share_stability/run.py`, `src/commute/spread.py` (`row_reads`), `cluster/share_stability.sbatch`.*

**In plain words.** A fine-tune on code-like text that made held-out text slightly worse still reads a re-gating share
of 0.77 to 0.82 in the every-position form, and the part 2 page leans on that as the share's main weakness. This run
re-reads the two saved states of that fine-tune on five batches of 64 held-out rows, so the number gets a spread, and
reads one random update of the same size per matrix on the same rows, so we learn whether any small update reads that
high or only this one.

## Where the number came from

- The page that asks for this cell: `reports/analyses/2026-11-09_analysis_what-regating-does.md`, section 3, first
  "against" row ("0.82 at step 10 and 0.77 at step 100, on gains of −0.052 and −0.065 nats per token"), and section 4,
  proposal 5.
- Those numbers are E36's: `experiments/regating_spread/results/regating_spread.json` → `.states["10"].r` 0.8156,
  `.states["100"].r` 0.7656, computed by `experiments/regating_spread/run.py` (`Run.evaluate`, `shares`) with
  `src/commute/spread.py`'s `target_logp` and `masked_logp`, on 64 held-out rows.
- E36 did not save its states. The saved states `experiments/regating_table/results/large/code_step{10,100}.pt` and the
  rows `code_held.pt` are E46's regeneration of E36's recipe (`anchors.code_rows`, `anchors.code_finetune`), read by
  E46's `row_cells` at 0.8186 and 0.7802 (`experiments/regating_table/results/stage0a.json` →
  `.groups["e36/code"].states[*]`, g / G). E46's rows equal E36's (selection record exact, T0 equal within 3e-6).
- So the appendix's control ("reproduces 0.8156 within 1e-3") cannot hold on the saved states: step 100 differs by
  0.015 between E36's state and E46's. The control here reproduces E46's reading, and prints E36's beside it.

## What the share is here

Per held-out row, T is the fine-tuned (or original) model's mean next-token log-probability over positions 0 to 510;
L is the same for the masked model with the original subcomponents, the update left out, and Γ's importance values read
on the updated model. Against the original model: gain G = mean(T₁ − T₀), gate gain g = mean(L₁ − L₀), share
r = g / G. `spread.row_reads` computes T and L per row by E46's path (Γ on 8 rows at once, masked model one row at a
time, float32 masked logits), which is shared code now and called, not copied. Every state is read as its do-nothing
state by construction: the update's own subcomponents never enter.

## What runs, in order (one job, about 2 to 5 minutes)

Rows: `anchors.code_rows(n_held=320)`; batch b is rows 64b to 64b + 63. Batch 0 equals `code_held.pt` (checked on CPU
on 2026-11-11: `torch.equal` true; 374 held rows exist).

0. **Control, first and alone, on batch 0.** Rows and both states' sha256 are E46's. The zero update reads exactly W0
   (max per-row |ΔT|, |ΔL| ≤ 1e-6). W0's T0, L00 and each state's T1, L01 within 2e-4 nats per token of E46's cells,
   and r within 0.01 of 0.8186 and 0.7802. Any miss: exit 1, nothing else read.
1. **Spread.** Both states on batches 1 to 4: per batch G_b, g_b, r_b; pooled over 320 rows; SD of r_b.
2. **Random floor.** One Gaussian: `finetune.random_direction(code_step100, seed 0)`, each matrix a standard Gaussian
   scaled to the step-100 update's Frobenius norm of that matrix (asserted within 1e-4). Frobenius per matrix because it
   is the size the project's matched random updates use (E46, E51, E67) and keeps each matrix's part of the update;
   step 100 because it is E36's scored step and the larger update. All five batches.
3. **Rules** (filed in `predictions.json`): readability (every batch |G_b| ≥ 1.2e-3 with the pooled sign), P_stable
   (both states readable, batch SD ≤ 0.13), P_gauss (unreadable → B; SE of the paired batch difference > 0.075 → no
   call; |d| ≤ 0.15 → A; else B).

## Expected values

| State | Expected |
|---|---|
| W0, zero update | G = g = 0 exactly; unreadable |
| batch 0, steps 10 and 100 | r 0.8186 and 0.7802; cells as E46 within 1e-6 (same path and card type) |
| steps 10 and 100, five batches | every G_b about −0.05 to −0.07; r_b 0.65 to 0.95; SD 0.04 to 0.065 |
| Gaussian | G between −0.002 and 0, lean unreadable: E46 Amendment 2 had to scale Gaussians past 10 times the fine-tune's norm to match its KL |

## Thresholds and their sources

- 2e-4 per cell: the largest W0 cell difference between two paths on the same rows, E36's L(g0) against E46's L00,
  1.19e-4 nats per token. 0.01 on r: that error on g's two cells over |G| = 0.052.
- 1.2e-3 readability floor: 10 times that path difference, so a path error moves r by at most about 0.1.
- 0.065 spread per 64-row batch: E46's per-row SE of g (0.0041 at step 100, 0.0033 at step 10) over |G| (0.064,
  0.052). 0.13 = twice that. The card's smallest gap 0.15 is 5.2 SE of the paired comparison at n = 5.

## Gates

Not gated: $0, about 5 minutes, one debug L40 job. No review needed under the fast lane. Early stop: stage 0.

## What each outcome means

- **A (Gaussian within 0.15 of the fine-tune):** any small update reads about 0.8 in this form, so "a fine-tune that
  learned nothing still reads high" holds for any small update; the every-position form does not read learning.
- **B, unreadable:** a random update of the same norm barely moves next-token predictions, so there is no gain to
  share; the high reading needs an update that moves the output, as this fine-tune does without learning.
- **B, reads differently:** the 0.77 to 0.82 is specific to this fine-tune.
- **P_stable fails:** the 0.77 to 0.82 was one batch's reading and is quoted with its spread.

The result goes into the part 2 gate page (cell A of the brief).

## Not run

- A Gaussian matched on KL or on gain rather than norm (the cell asks for matched size).
- A second Gaussian draw or a second fine-tune seed: the spread is row sampling only.
- The trigger form of the share on these states (they have no trigger).
