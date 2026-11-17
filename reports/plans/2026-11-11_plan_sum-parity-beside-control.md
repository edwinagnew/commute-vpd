# E157: the sum-parity fine-tune read beside the last-number-odd control, between a planted ceiling and floor

*2026-11-11. Cell P1.14 (1 run, cluster debug, $0, due 11-12), exploratory: no label, no change to the wording or
cut-offs (user, 2026-11-11). Card: `experiments/sum_parity_beside/card.txt`. Filed rules:
`experiments/sum_parity_beside/results/predictions.json`. Code: `experiments/sum_parity_beside/run.py`, `plot.py`,
`cluster/sum_parity_beside.sbatch`.*

**In plain words.** The rewiring share (H: one minus the distance left after the best affine map of the original's own
activations, over the fine-tune's whole distance) read only 0.47 on a fine-tune that merely pushes on a fact the 67M
already holds exactly ("the last number is odd"). This run reads, on the same windows, a fine-tune that pushes on a fact
the 67M does not hold linearly (the parity of the sum of the last two numbers, linear read 0.61), and two planted edits of
the same size: one the map can reach exactly, one no map can reach. The question is only whether the new fine-tune reads
clearly below the control, beyond seed spread, and where both sit between the planted top and bottom.

## What runs, in order (one job, resumed on the wall limit)

E154's windows, number tokens, alpha 4, push $Q b$ and search code (`reuse_variable_finetune/amend2.search_h`, five sites;
two sites in order as E156's variant b, all ten pairs). Target gap of the sum-parity target: 0.2294.
1. **Planted ceiling, first and alone.** At block 2's input, last position: $h \mapsto h + s\,\ell(h)\,Q b$, with
   $\ell(h) = (h - \mu_0)\cdot(\mu_1 - \mu_0)/|\mu_1 - \mu_0|^2$, the difference-of-means read of "last number odd"
   from the 67M's own block-2 input on the fit half's keyed windows. Exactly affine in that input, so exactly reachable
   at in2. $s$ is set so its gap matches 0.2294 within 0.005. H at one site, then two sites. Stop: H at in2 under 0.5.
2. **Planted floor.** $h \mapsto h + s\,r(a,b)\,Q b$, $r$ a seeded random bit per ordered number pair (half ones, keyed
   windows only). Read windows hold only pairs the fit half never sees, so no map fitted there can know $r$. Same gap
   match; one site, then two sites.
3. **New arm.** The plain 67M fine-tuned toward 67M + 4 $Q b$ where $a + b$ is odd (E150's "plain" arm, its ladder:
   300 steps, else 3000; learned when the KL ratio is at most E154's bar), seeds 0 and 1; distances, H at one site, then
   two sites. The saved control (0.466, 0.459 one site; 0.595, 0.583 two sites) is not re-read.

Every arm is reported raw and rescaled, $(H - H_\text{floor})/(H_\text{ceiling} - H_\text{floor})$, per reading kind.

## Expected values and filed reading

At zero steps the fine-tune is the 67M: gap 0, no reading (E154's zero-step control). Ceiling: about 1 at in2 (an exact
affine edit; E156 read 0.84 on an edit that is only nearly affine). Floor: about 0.3 to 0.5, the share a map earns by
writing the average push on keyed windows; TV grows less than linearly with push size, so below 0.5. **Reading** (one
site, the definition's measure; two sites alongside): "clearly below" if both new seeds sit at least 0.1 under the lower
control seed; "not separated" if the two seed means differ by under 0.1; otherwise "between", reported as read. A seed not
learned, or ceiling minus floor under 0.1, gives no rescaling and is said so. Edge cases in `predictions.json`.

## /design check

- Fills a named open cell (P1.14) of the table of work, exactly as its done_when reads; no new question.
- TRIED.md: E154 (control under 0.5), E156 (search reads 0.84 on its target; two sites raise to about 0.6). Nothing here
  repeats them: the new arm and both planted edits are unread.
- Control first: the ceiling runs alone first with a stop. The cell lifts only E150's refusal of the plain arm.
- What it is blind to: one task, two seeds, one push size; affine maps only; H at the last position only.

## Time and cost

One L40, debug, $0. From E156 (job 58855): one-site search about 10 min per item, ten pairs about 23 min per item. Four
items (ceiling, floor, two seeds) plus training (300 to 3000 steps, under 20 min each): about 2.5 h, so one resume
(`E157 resume: wall limit`).

## Not run

Other tasks, push sizes, more seeds; nonlinear maps; the taught arm; a re-read of the saved control.
