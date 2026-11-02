# Plan: do eliciting updates map more cleanly onto neurons than new updates? (E63)

**Status: the user's request (task file, 2026-11-02), form left to the lead: a new experiment, since it reads three
families of saved fine-tunes, not only E56's. $0, one MATS debug job, saved states only, no new fine-tune.**

## The question, in plain words
The user asks whether fine-tunes that bring out something the model already does sit on fewer MLP neurons than fine-tunes
that teach something new, for example by being sparser in the neuron basis. If they do, a reader with no decomposition
gets some of what VPD's re-gating share r is claimed to give. "Eliciting" here means built to elicit, not the causal label
of `decisions.md` row 112: these pairs have no causal check yet, and the write-up says so.

## Design check (`/design`)
- *Goal:* GOALS "teaches new vs strengthens existing": a decomposition-free baseline for r.
- *Nearest tried:* E56 (and E52): the top 16 neurons carry about 0 of the gain on both kinds; half the gain needs 4096
  neurons for elicit and 12288 for the matched cipher, one coarse grid step, and that gap goes when the cipher has learned
  as much. E51: weight statistics in W0's singular basis, never the neuron basis. **What E56 could not see:** it ranks
  neurons by the *mean* shift over all windows, so a map using different neurons per key looks dense by construction; and
  it never reads the update's own sparsity, or whether the update rescales neurons W0 already has.
- *Patch or step:* a step: the neuron-basis reading of the re-weighting hypothesis, the baseline the user names.
- *Decisions:* relies on row 112 (r is a candidate instrument). Departs from none.
- *Changes next:* a number that passes all three tests below is a cheap rival to r: read it on the equal-prior and
  shuffled maps (E58 Amendments 6-7), then on the reuse line's truth set. None passes: "neurons do not map eliciting more
  cleanly, at this size and on these pairs" goes in the claims, and r keeps that lead.
- *Control:* stage 0 below, first and alone, stops the job on a miss.
- *Class:* the user's own request; also class 1 (analysis of saved states).

## States (all saved, sha256 checked against their source's record where one exists)
| family | built to elicit | built to be new | notes |
|---|---|---|---|
| A (primary) E54/E56 same keys | `elicit_s0..s4` | `cipher_s0..s4`, size-matched | plus `cipher_own_s0,s1`, the cipher trained to its own learned step: equal progress |
| B E46 tables | `ident32_s0,s1`, `ident32b_s0,s1` | `rand32_*`, `rand32b_*` | own steps, not size-matched |
| C E53 broad | `dW_lowercase_s{0,1}_step300` | `dW_broad_cipher_s{0,1}_step300` | equally broad; `dW_cipher_*` reported only |
| reference | `code_step10` (learned nothing) | matched-norm Gaussian per matrix | |

## Numbers: four, each with what it should read
Neuron = one of the 4 x 3072 MLP GELU outputs, as in `src/commute/neurons.py`. Its update = its row of δW's up-projection
plus its column of δW's down-projection (the writer confirms the matrix names and axes against the checkpoint).

| number | definition | Gaussian / rotated | positive control | blind to |
|---|---|---|---|---|
| **W1** weight sparsity | participation ratio over neurons of each neuron's squared update, (Σe)²/Σe², over 12288 | ≈ 1 of 12288 as a fraction | update on 16 neurons: 16/12288 | attention and read-out matrices; **Adam's per-element normalisation flattens it**, hence W1g |
| **W1g** the same on the gradient | W1 on the first-batch gradient of the task loss at W0, one per task and seed | ≈ 1 | as W1 | later steps; what Adam then does |
| **W2** rescale share | share of a neuron's update along its own W0 row and column, energy-weighted over neurons | 1/768 for the rows, measured for the columns | rescaling 16 neurons: 1 | a neuron rotating towards another neuron |
| **A1** activation sparsity | per-window participation ratio over neurons of (tuned − W0 activation) at the answer position; median over the scoring windows, as a fraction of 12288 | measured with the rotated reference | the 16-neuron update in the last block: at most 16/12288 | neurons that move and do nothing: hence C1 |
| **C1** causal count | per window, rank neurons by that window's own \|shift\|, patch the top k (E56's edit, every position); log2 k at which the mean share over the scoring half first reaches half of that state's own 12288-neuron share; k on powers of 2 up to 2^13, then 12288 | unreadable: G under the 0.05 nat floor | the 16-neuron update: k ≤ 16 | down-projection changes (kept at W0, as E56) |

**Rotated reference** (W1, W1g, A1): the same reading after one fixed random orthogonal 3072 x 3072 rotation per block of
the neuron index. It says whether the neuron basis is special for this update; a number that equals its rotated value
carries no neuron information however low it is. W0: all four undefined (Δ = 0); the identity patch reads 0.

## Rules (the writer files these in `results/predictions.json` before the run, edge cases checked)
Separation = E51's rule (gap of the kinds' means strictly over the larger within-kind range). "Sparser" means lower W1,
W1g, A1, C1, higher W2, on the elicit side. A number **answers yes** only if it passes all three:
- **T1** family A, five seeds at matched size: separates, elicit sparser.
- **T2** equal progress: `elicit_s0,s1` vs `cipher_own_s0,s1`: separates in the same direction (E56's neuron gap went here).
- **T3** other families: B and C each separate in the same direction.
- **Basis guard:** on the elicit side the number differs from its rotated value by more than its seed range. W2 and C1 have
  no rotated value; W2's guard is the Gaussian's, C1's is the state's own full width.

Leans: P1 W1 no (Adam flattens it), P2 W1g unknown, P3 W2 no, P4 A1 unknown, P5 C1 passes T1 and fails T2, as E56's coarse
count did. Medium at most on all; the user's hypothesis is a live one and per-window ranking is new.

## Tree
Stage 0 miss: stop, report. Any number answers yes: read it on E58 A6-A7's equal-prior cipher and shuffled elicit map
($0, same code; decides kind or W0's prior), as an amendment, with r's rank correlation beside it. None answers yes: write
up; the elicit/new gap under T1 only is reported as a progress effect. The E55 truth-set reading needs the user's say.

## Stage 0 (first, alone; any miss stops the job)
sha256; W1 and W2 on the synthetic controls and Gaussian at the table's values; the 16-neuron update in block 3 gives A1 ≤
16/12288 and C1 ≤ 4 when readable; identity patch ≤ 1e-6; the mean-ranked path reproduces E56's N16 for `elicit_s0` and
`cipher_s0` within 1e-6 of `experiments/keydependent_patch/results/job1.json`; per-window k = 12288 equals E56's full-width
share within 1e-3.

## Amendments before filing (2026-11-02, the code writer; nothing has run)
No threshold in Rules was moved. Each note closes a gap the writing found.
1. **Gaussian's "about 1" on W1.** It holds when every matrix carries equal energy (0.9987 on a unit Gaussian, which stage 0
   gates at [0.99, 1]). The per-state Gaussian keeps each matrix's norm, as E51's, so it reads below 1 by how unevenly the
   blocks share the update; it is reported, and no rule reads it for W1 or W1g.
2. **Identity patch at 1e-6.** E56's recorded `identity_max_abs` is 1.1e-5, read on 48 windows against W0 read on 97: two
   batch shapes, not the patch. The plan's 1e-6 cannot be met on that path. Stage 0 reads the new per-window path against W0
   in the same batch, where it should read 0.
3. **W2's columns.** The down-projection's columns are 768-dimensional as the rows are, so 1/768 is expected there too. As
   the plan says "measured", it is reported, not gated.
4. **W1g under T2.** At W0 `cipher_own_s<s>` has `cipher_s<s>`'s task, seed and batch, so W1g's T2 reads T1's gradients at
   seeds 0, 1. A gradient at W0 has made no progress, so W1g cannot be a progress effect; its T2 adds nothing and is reported
   as such in the file. The rule is unchanged.
5. **First batches are exact in all families.** E46's windows files hold the training windows, and E53's rows are rebuilt by
   its own calls, so W1g needs no stand-in batch. Each first-batch loss is compared with the fine-tune's logged step 1 (bf16
   against fp32), reported, as a check of the batch.
6. **C1's readability.** "Half of its own full-width share" has no meaning when that share is not positive, so C1 is
   unreadable then, and when the full-width rise itself is under 0.05 nats. With a positive full width some k always reaches
   half, so "never reaches" cannot occur.
7. **A1 squares the shift**, as W1 squares the update, which is what the control's 16/12288 assumes.
8. **The basis guard is required for a yes**, read on family A's five elicit states as means against the seed range; "differs"
   stays two-sided and the side is recorded. A test with under two readable states on a side reads unreadable and does not pass.
9. **One stage-0 check added:** W1 of the rotated 16-neuron update in [0.10, 0.25] (expected 0.222), so the rotated reference,
   on which the guard rests, has its own positive control.
10. **`code_step10`: W1 and W2 only.** E36's task has no answer position, so A1 and C1 are undefined on it; W1g on it has no
    reader. All three are cut.

## Not run
SGD fine-tunes of the pairs (W1g stands in); attention-head sparsity; any new fine-tune; Qwen (the Qwen line's); the
causal label of row 112, which E64 and E55 C2 pursue.
