# Plan: E117, the switched-on set and re-switching on neurons at VPD's count (67M)

*2026-11-06, baseline-lead. Step 1 of the user's retarget brief (`decisions.md`, last row). $0, one `debug` L40 job.*

## The request, restated

On the 67M states where VPD already has positive re-gating findings, repeat VPD's two readings, the **switched-on
set** and **re-switching**, on MLP neurons, choosing the same number of neurons as VPD switches on. The comparisons
are copy maps (ident32, ident32b) against random maps (rand32, rand32b), from E12, E46 and E50, and lowercase against
the two E53 ciphers. Each class is known because the task was built that way. Put both methods on the same axes, one
panel per criterion, with the thresholds in the caption. Nothing new is trained.

## /design check

- **Goal.** `GOALS.md` asks whether the decomposition is what makes post-training readable. This tests whether
  VPD's re-gating readings add anything over the same readings taken on neurons.
- **Class.** Approved class 1: GPU passes of the existing harness on saved states.
- **Not a repeat.** E98 already did full-width re-switching on neurons (r_half, r_n) on these 14 states. E52 patched
  the top-k neurons at a ladder of k. Neither used VPD's count, and no run has a neuron switched-on set.
- **Also new.** VPD's own switched-on set on lowercase and on the ciphers. It is computed here for the first time, so
  that panel has a VPD side.
- **No verdict rule.** This is a comparison of methods, not a reuse verdict. Rows 81 and 87 are untouched, and every
  panel is read with an already-filed separation rule.

## States and windows (all on disk, `experiments/SAVED.md`)

| Family | States | Windows and read position |
|---|---|---|
| Copy vs random map | `results-large/regating_table/{ident32,ident32b,rand32,rand32b}_s{0,1}.pt` (fp32 δW, step 10; rand32b at step 30) | E12's 97 held-out windows (`windows_{task}.pt["heldout"]`), at the trigger |
| Lowercase vs cipher | `results-large/performed_anchor/dW_{lowercase,cipher,broad_cipher}_s{0,1}_step300.pt` (bf16, then `.float()`) | E53's scored windows (`regating_truth_table/run.py:NextTask`), at position 63 |

Reference rows: W0 (δW = 0), and one random update with each matrix matched in Frobenius norm to ident32_s0
(seed 0).

## Readings, per state, at a count K

K_VPD is |S_on|: the number of alive subcomponents whose mean change in causal importance at the read position,
state minus W0, is above 0.3 (`recruit.trigger_dg`, `recruit.switched`, frozen released Γ). On copy and random maps
this is E12's count.

The neuron switched-on set N_on is the K_VPD neurons, out of 12,288 GELU outputs, with the largest mean increase in
firing at the read position, state minus W0, on the same windows. Same windows, same position, same count.

| Panel | Criterion | VPD | Neurons, at K_VPD |
|---|---|---|---|
| A | Seed agreement of the set | Jaccard of S_on between s0 and s1 | Jaccard of N_on between s0 and s1, with K = the mean of the two seeds' K_VPD |
| B | Restore share of the set | ρ(S_on): share of the gain G removed when S_on's outputs at the read position are put back to W0's in the fine-tuned model (`recruitment.restored_at`, E50) | The same, with N_on's firing put back to W0's (W0's MLP weights on the fine-tuned model's MLP input) |
| C | Re-switching carried by the set | r_S: (L with W0's gates except S_on given the state's gates, minus L00) / (T1 − T0), with C0's masked forward (`local_faith`) | r_N: (W0's model with only N_on given the state's firing, minus T0) / (T1 − T0) |
| ref | Full-width re-switching, not rerun | r (E46/E53, `regating_truth_table/results/table.json`) | r_half, r_n (E98, `neuron_r/results/cells.json`) |

Panels B and C each also get a random same-count set: 20 draws, block-matched (neurons per block and subcomponents
per matrix as in the real set), reported as a mean and range. This is the floor.

## Order, controls and stops (the scored run is its own smoke)

1. **W0 control.** With δW = 0, every firing change and every Δg must be exactly 0, and every log-odds difference
   under 1e-5. If not, stop.
2. **Apparatus controls**, on states already read:
   - VPD's held-out S_on counts on the rand32/rand32b rebuilds equal 41, 52, 27 and 21
     (`new_vs_existing_amend2.json`). Stop if any differs.
   - The all-alive r reproduces E46/E53's r within 0.005.
   - Putting every neuron's firing into W0's model reproduces E98's r_n within 0.01. Stop if either is missed.
3. **Reads.** Copy and random maps first, then lowercase and ciphers, then the random-update reference. Results are
   written after each state, and every scored number is printed to the log.

## Filed rule (one for every panel, both methods)

The **separation rule** is E12's (`new_vs_existing/results/predictions.json`). Take one value per state, so four per
class on copy/random and two per class on lowercase/cipher. A method separates on a panel when:
- the two classes' ranges do not overlap;
- the gap between them is larger than the larger within-class range;
- the gap is in the predicted direction: the existing class higher on A, B and C.

On panel A there is one value per task, not per state (two per class), and the same rule applies.

On B and C, E53's lowercase-against-cipher bar is shown beside the rule: lowest lowercase minus highest cipher of at
least 0.3.

Each panel reads "separates", "does not separate" or "unreadable" (for example K_VPD = 0 on a state). It never reads
a reuse label.

**Predictions to file**, guesses with no pass or fail for the project:
- VPD separates on A for copy against random maps. E12's Jaccards were 0.80/0.85 against 0.53/0.66.
- VPD does not separate on B. E50 found gaps of −0.03 and −0.06.
- On neurons, A, B and C are each a coin flip, filed as "does not separate". E98 and E52 lean the wrong way on
  copy against random maps.

What each panel is blind to:
- **A** reads only whether the set repeats across batch orders, not what the set does.
- **B** reads the trigger position only.
- **C**: the VPD side keeps C0's masks, so it misses δW outside the components (E46's e_delta).

Noise floor: the spread between seeds, and the random-set range.

## Figure

One figure, `experiments/neuron_vpd_count/figures/panels.png`:
- three panels (A, B, C), with points per state, the two methods side by side in each family, and the random-set
  band;
- a fourth strip for the full-width reference;
- the caption states the 0.3 importance threshold, the separation rule and E53's 0.3 bar.

## Cost

About 15 states, each needing one VPD importance pass, about four masked forwards and about 40 random-set forwards on
about 100 to 500 short windows. Expected under 20 minutes on an L40: `debug`, $0, under the gate.

## Not run

Neuron switched-on sets at counts other than K_VPD; layer split of N_on (reported, not scored); any Qwen state
(step 2).
