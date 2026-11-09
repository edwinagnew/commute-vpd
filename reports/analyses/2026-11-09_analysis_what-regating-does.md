# What does the re-gating share measure, and how easily could something simpler replace it?

*2026-11-09, for the plan's table of work. Read from saved files only; no run. Design checks per proposed cell:
`2026-11-09_analysis_what-regating-does_appendix.md`.*

## 1. The answer

The share is the part of a fine-tune's gain that appears when the original subcomponents are kept and only which of them are switched on changes. It sorts most task pairs built to reuse from pairs built to be new, even at equal update size. Its level also follows how likely the original model already was to give the answer. Real fine-tunes read the same as their shuffled twins on three pairs. We do not yet know what it adds to those facts.

*No verdict is given here. The definition of reused against new is under revision, so this page uses the labels each
pair was built with. "Separates" means only that one built side reads above the other.*

## 2. What the number measures

The project calls it the re-gating share: Γ is the map that sets how strongly each subcomponent is switched on. The number is the share of the gain that appears when only Γ's output changes, and it is called the share below. The code is `route_split` in `src/commute/local_faith.py:256-278`, with the formula at line 270; key-to-value tasks reach it through `src/commute/keydep.py:66`.

Hold a set of held-out windows fixed and look at the trigger position. The gain is the log-odds of the trained answer under the fine-tuned model minus the same under the original model. Then run a masked forward with the original subcomponents and Δ left out. Run it once with the importance values Γ gives on the original model's activations, and once with the values Γ gives on the fine-tuned model's. The share is the rise in the masked log-odds between those runs, divided by the gain, each term a mean over windows.

A share of 0 means the new importance values do not move the masked answer. A share of 1 means they alone give the whole gain. It can exceed 1. On the copy map at a new marker the masked forward starts 2.01 log-odds below the real model, and the share reads 1.09 to 1.10. The filed reading is that this low start is the excess (`experiments/induction_cue/results/run.json` → `.states.copy_s0.cells`).

The state's own subcomponents never enter the numerator, so a do-nothing update reads the same number as a split: 0.133 for both on the saved repeat-label fine-tune (`experiments/fresh_reuse_pair/results/stage2.json` → `.states.real_s0.nothing`, `.least_squares`). A zero update has gain 0, and the share is undefined (`experiments/regating_truth_table/results/stage0.json` → `.rows["zero/*"].G`). Shares are scored only from a gain of one log-odds (`local_faith.py:187`). The trigger's own inputs carry almost all of Γ's change: on eight tasks they recover 0.998 to 1.043 of the effect (`experiments/shift_anatomy/results/shift_anatomy.json` → `.states.*.R_steer_own`).

The number has been computed three ways on the 67M:

- The **trigger form** above, in every row below unless named.
- The **every-position form** (`experiments/regating_spread/run.py:314`): the same ratio over the log-probability of every next token, with no gain floor.
- The **neuron version** (`src/commute/neuron_r.py:132`), with no Γ: the original output weights are fed the state's neuron firing, as `r_n` for the full firing and `r_half` for the MLP input alone.

## 3. Every result against it and for it

A zero update leaves the share undefined in every row. No row has a positive control of known kind; stage-zero checks only reproduce earlier numbers. Each row's reference is the other number in the row. Paths are under `experiments/`.

**Against, read as a test of reuse:**

| Result | Reading and reference | Source and key |
|---|---|---|
| A fine-tune that made held-out text worse reads high, in the every-position form | 0.82 at step 10 and 0.77 at step 100, on gains of −0.052 and −0.065 nats per token; a lowercase fine-tune that learned reads 1.05, 0.90, 0.79. Both parts of this ratio are small and negative, and the trigger form's gain floor would refuse it, so this row shows only that the ratio can read high without learning | `regating_spread/results/regating_spread.json`, `_amend1.json` → `.states` |
| A fine-tune to a different answer at the same trigger reads the same as the real one | 0.55, 0.60 against 0.60, 0.61; a same-size random update's gain is −0.015, so its 1.19 means nothing | `recruit_test/results/recruit.json` → `.arms.*.routes` |
| Its rank follows the original model's prior for the answer | Spearman 0.886 over six maps and 0.95 over eight, both computed after the run. Ranked by prior, the three maps built to reuse sit above every map built to be new, but the gap between the two groups is only a factor of 1.6 in prior while the share doubles across it, from 0.38 to 0.77 | `open_pair/results/amend6.json` → `.rho_r_logprior`; `amend7.json` → `.prior_beside_r` |
| A real fine-tune and its shuffled-label twin read alike, on three pairs | Case label 0.11, 0.14 against 0.11, 0.17; repeat label 0.13, 0.24 against 0.22, 0.18; class map 0.23-0.25 against 0.27-0.28 | `case_label_both/results/case_label_both.json` → `.routes`; `fresh_reuse_pair/results/stage2.json`; `open_pair/results/amend7.json` |
| A rule that carries to unseen keys reads like a lookup | The class map carries 0.55 to 0.64 above its null; its twin carries under 0.1, yet they read as above | `open_pair/results/amend7.json` → `.score.CL`, `.score.T` |
| A real-model reading does not separate where the share does | Restoring every alive subcomponent in the fine-tuned model: copy 0.76-0.78 against random maps 0.66-0.73 | `restore_anchors/results/restore_anchors.json` → `.states.*.rho.every_alive` |
| Steering separates half the pairs, sometimes more widely | 5 of 10 built pairs against the share's 9; on copy against the prior-matched lookup steering reads 0.82-0.88 against 0.15-0.17, the share 0.77 against 0.37-0.38 | `method_table/results/table.json` → `.rows[].cells.steering`, `.cells.r` |

**For, with the same limits:**

| Result | Reading and reference | Source and key |
|---|---|---|
| Pairs built to differ are far apart | Copy 0.76-0.80 against random maps 0.27-0.29; lowercase 0.75-0.77 against ciphers 0.12-0.17; rank-1 0.84-0.90 against cipher 0.22-0.25 | `method_table/results/table.json` → `.rows[].cells.r` |
| At equal update size it separates where steering does not | Five seeds: elicit 0.76-0.77 against cipher 0.21-0.27, at size 3.14-3.17 against 3.18-3.20; steering's log-odds share 0.25 against 0.18-0.24 | `keydependent_patch/results/job2.json` → `.P2.r`; `method_table/results/table.json`, row `e54_matched` |
| Copy reads above a no-link lookup at a close prior | 0.77 against 0.37-0.38, the lookup's prior 1.6 times lower; on the new-marker pair 1.09-1.10 against 0.38-0.42, filed as unscored. Both copy maps have their answer in the window | `open_pair/results/amend6.json` → `.prior_beside_r`; `induction_cue/results/run.json` → `.score` |
| Rescaling the update leaves the gap | New-marker pair, each update scaled to the other's size: 0.691 becomes 0.686 | `induction_cue/results/amend2.json` → `.row7_size` |
| The same words without the key-to-guess link read lower | Elicit 0.72-0.74 against its shuffle 0.47-0.48, the prior 127 times apart | `method_table/results/table.json`, row `e58_elicit_shuffle` |
| The neuron version does not reproduce the sorting on copy against random maps, and does on lowercase against ciphers | Copy against random: `r_n` 0.36-0.50 against 0.68-0.75, reversed; `r_half` 0.25-0.38 against 0.45-0.48, no call; the share 0.76-0.80 against 0.27-0.29. Lowercase against ciphers: `r_half` 0.53-0.58 against 0.21-0.27, the same direction as the share | `neuron_r/results/score.json` → `.rows.P1r`, `.rows.P1` |

In the method table the prior separates every pair it was read on except those matched on prior by design. Over the 10 pairs the share separates 9 of, update size or recipe separates 3 and steering 5; update size was not read on one pair and unreadable on another, and steering unreadable on one. Steering's null check on the copy-against-random row failed narrowly.

## 4. Proposed cells, ranked

Each uses files already on disk and is one free MATS `debug` job on the L40. Full file lists are in the appendix.

| Rank | Question and run | What each outcome means | Saved inputs and cost |
|---|---|---|---|
| 1 | Over every saved map, does the original model's prior order them as the share does? One forward of the original model fills the prior where it is missing; plot share against log prior. | Agreement near 0.9 over about 18 maps: the paper reads the share as the prior, except on the copy rows. Much weaker: the share carries something the prior does not. | Windows from `regating_table`, `keydependent_elicit`, `open_pair`, `induction_cue`, `case_label`, `fresh_reuse_pair`. About 10 min: E60 read 18 states in about 90 s (`TRIED.md`, row E60). |
| 2 | Do steering or the neuron version tell a real fine-tune from its shuffled twin, where the share cannot? | One does: the paper says the share is blind on the contrast the tentative definition rests on. Neither: the label pairs are out of reach for every reading we have. | `dW_{real,twin}_s{0,1}.pt` in `case_label` and `fresh_reuse_pair`. About 10 min: E98 read 56 states in about 4 min (row E98). |
| 3 | On the pair with a causal-check label, does steering separate copy from lookup as widely as the share? It has never been read there. | Matches a gap of about 0.7: the share has no edge on the one causally labelled pair. Narrow: the share keeps that point, still unscored. | `induction_cue` `{copy,lookup}_s{0,1,2}.pt`. About 5 min: E62 Amendment 3 read these in about 4.5 min (row E62 A3). |
| 4 | How far can masks alone push each trained answer at the original model, and does the share's size follow that reach? The existing mask adversary with its objective reversed. | Share near reach, and reach orders maps as the share does: the number reads what the subcomponents can write. Reach large on ciphers while their share is low: the share reads the update. | Windows of cell 1; no fine-tuned state. This is the only cell that needs new measurement code. Masked reads took about 4.5 min (row E62 A3); the adversary's cost is unmeasured and timed first. |
| 5 | Is 0.77 to 0.82 stable, and what does a random update read in the every-position form? Re-read the saved code-text states with spread across row batches, plus one Gaussian of their size. | Stable near 0.8 for both: any small update reads that, and "learned nothing still reads high" stands. Wide or different: the claim narrows to this one fine-tune. | `regating_table` `code_step{10,100}.pt`, `code_held.pt`. About 5 min: the original runs took 77 s and 90 s (row E36). |

Needs a new fine-tune, not proposed: a size-matched twin; a label pair with a higher starting prior; a non-copy map at a prior matched to the cipher's.

## Not run

- A share read against a causal-check label with its control held; the one causal pair is filed as unscored.
- A share gap at matched size and matched prior on a task that is not a copy map.
- The neuron version on the 96-key and class-map maps, since their states were not saved, only their windows.
- The every-position form on any task that also has a trigger reading.
- A second decomposition, so nothing here says how much of the share belongs to this Γ.
- An independent check of this page.
