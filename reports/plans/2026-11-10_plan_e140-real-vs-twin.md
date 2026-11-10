# E140: do steering or the neuron version tell a real fine-tune from its shuffled-label twin, where the share cannot? (cell P2.4, due 11-11)

*compare-lead's code writer, 2026-11-10. Card `2026-11-09_plan_p2-4-card.md`: three seeds per side, smallest gap 0.1 of the gain, spread
0.0557, unpaired. The user's decision of 2026-11-10 (`decisions.md`, row "P2.4 real against shuffled twin"): per-word steering vectors
from the examples each fine-tune trained on, scored on the share's 256 rows, and a third copy of each of the four fine-tunes trained in the
same job. One debug L40 job, free. Labels are as each pair was built; no verdict on reuse.*

## What runs

**The two pairs.** Each pair is two fine-tunes of the 67M on the same windows. After a marker token the model must say one of two label
words. The real fine-tune learns a rule about the word before the marker; its twin gets the same words with the labels shuffled per word,
so it can only memorise. In the case-label pair the rule is "capitalised or not" (E112, `experiments/case_label`); in the repeat-label
pair it is "this word already occurred earlier in the window" (E125, `experiments/fresh_reuse_pair`). Seeds 0 and 1 of each arm are saved.
Seed 2 is trained here with the same function, `label_reads.finetune`, and the settings read from each pair's own `predictions.json`:
Adam at 3e-5, batch 32, a KL term of weight 1 to the original model, stop at the first checkpoint whose fit accuracy reaches 0.9. The seed
only sets the order of the training batches (`finetune.py` line 199); the windows and labels are fixed in each pair's `task.pt`.

**What is read.** Every number is read on the first 256 fit windows of the pair, the rows the share was read on, with each arm's own
labels. The scored unit is the contrast between the two labels: logit of label A minus logit of label B, signed toward the window's own
label. Log-odds of the label over the whole vocabulary and log P are written beside it. A method's share on one fine-tune is its change in
that unit over the fine-tune's own gain in that unit, so 1 means the method reproduces the whole gain and 0 means it writes nothing.

- **The share** keeps the decomposition's original subcomponents and changes only which ones Γ switches on: $r = (L_{01} - L_{00}) / (T_1
  - T_0)$, where $L$ is the CI-masked original model with Γ's gates read from the original ($L_{00}$) or from the fine-tune ($L_{01}$), and
  $T$ the target model before and after. Same calls as the saved reads (`masked_reads.share_cells`).
- **Steering** adds a vector to the original model's block input at the marker. A pooled vector is the mean of the fine-tune's change
  there over all 9,600 training windows; a per-word vector is that mean over the training windows of one word, 16 per word on the case
  pair and 32 on the repeat pair. Layer and strength are chosen per fine-tune and kind on 512 training windows.
- **The neuron version** does the share's operation on neurons: the original model with each block's MLP firing kept only on a TopK set
  of neurons, at every position, the set taken from the fine-tune's own firing ($N_{01}$) or the original's ($N_{00}$); share $= (N_{01} -
  N_{00}) / (T_1 - T_0)$. TopK is by absolute firing, with 64 neurons per block and with each block's count matched to the
  decomposition's: the number of its down-projection subcomponents with gate above 0.5, averaged over the read rows' positions.

## Stages, control first (one job, `STAGE n DONE` after each file; the log prints every number)

0. **Controls, alone.** Fix the matched count before any fine-tune is read. A zero update must read exactly 0 through the share, both neuron
   readings and steering at every layer and strength, and strength 0 must read exactly 0 on a real update: a change above 1e-9 stops the
   job. The share of real seed 0 in log-odds must reproduce its saved value on each pair, 0.113 and 0.133, within 0.01; a miss stops.
   The tolerance is a tenth of the smallest gap; E135 reproduced six shares by these calls with a difference of exactly 0.
1. **The four seed-2 fine-tunes**, saved in bf16 to `results/large/`. Each is checked as it lands: it must learn, and its label-gap gain
   on the 1,024 fit windows must lie between half the lower and twice the higher of its arm's saved seeds (case real 12.0 to 12.8, twin
   8.8 to 10.2; repeat real 4.7 to 8.3, twin 7.4 to 7.7). A miss stops. Beside it, case real seed 0 is retrained and compared with its
   file, to show the function still gives the saved update.
2. **The share on all twelve fine-tunes**, and each twin rescaled to its real twin's size with the gain it keeps (the user's item 6).
3. **Steering and the neuron version on all twelve**, with random vectors, cross-application and cosines, then the score.

## Expected values

| quantity | at W0 or zero update | control | rival A | rival B |
|---|---|---|---|---|
| any numerator | exactly 0 | strength 0 reads 0 | | |
| share, log-odds, real seed 0 | undefined, no gain | 0.113 and 0.133 within 0.01 | | |
| share, label contrast | | | real and twin alike | real and twin alike |
| steering or neuron share, contrast | | random vectors near 0 | real minus twin at least 0.1 beyond seed spread on some pair | within 0.1 everywhere |

Gains in the label contrast should be about 12 to 13 for case real, 9 to 10 for its twin, 5 to 8 and 7 to 8 on the repeat pair. The
seed-2 fine-tunes should stop at step 30 for real and at step 300 and 100 for the twins, as their seeds 0 and 1 did.

## The filed rule and predictions (`experiments/real_vs_twin/results/predictions.json`)

Per method and pair, the gap is the mean over the three real seeds minus the mean over the three twins, in the label contrast. The pair
reads **apart** when the gap is at least 0.1 in size and at least twice its standard error, the pooled seed sd times $\sqrt{2/3}$, both
inclusive and either sign. A share is undefined when its fine-tune gains less than 1 in that unit, and any undefined share makes the method
**unreadable** on that pair. If all six shares are below 0.05 in size the reading is **inert**, a no call, so a method that writes nothing
cannot count as reading the arms alike. Rival A holds if any steering or neuron reading is apart on either pair. Rival B holds if none is,
all eight are readable, and every gap is under 0.1. Otherwise the card's stop rule holds without rival B.

What we expect, in words:
- **The share** does not separate the arms on either pair, as the card says.
- **Per-word steering** does not separate them on the case pair, where each word has one label in both arms. On the repeat pair it should
  separate them, with the twin well above the real: the real label depends on the window, half of each word's windows taking each label,
  so a per-word vector averages the label away, while the twin's label is fixed per word. That would be a property of the key, not of
  what the fine-tune wrote, and the write-up must say so.
- **Pooled steering** is inert on both pairs: with the labels balanced, one vector moves both label words alike.
- **The neuron version** at both counts does not separate the arms; with 3 to 30 neurons per block it may be inert.

Edge cases run on the rule before filing (`rules.py` through the dry run): every share 0 reads inert, never rival B; a gap below zero is
apart by its size; a gap of exactly 0.1 with no spread is apart; a gap of 0.2 inside the spread is not; a missing seed or an undefined
share is unreadable; the whole six-way do-nothing reads "stop rule", not rival B.

## How the user's seven changes of 2026-11-09 are kept

1. Per-key vectors beside the pooled one: per word, fitted on the training windows (the user's option a) and scored on the 256 rows; every
   read row's word has 16 or 32 training windows (dry run).
2. Strength swept over 0 to 8 at each of four layers, chosen on 512 training windows by the largest contrast rise; off-target KL reported.
3. Norm-matched random vectors, eight per fine-tune and kind; each arm's vector read on the other arm's labels over its gain, same seed;
   cosines between the six pooled vectors of a pair at each layer.
4. Every steering number in log-odds and log P as well as the label contrast.
5. All three methods in the label contrast and in log-odds, numerator and gain written separately and plotted separately.
6. Each twin rescaled to its real twin's size, gain kept reported. Three seeds per side make "beyond seed spread" a test, not a description.
7. The neuron version as the share's operation, TopK at 64 and at the matched count, both fixed before any fine-tune is read.

## Cost and run

`sbatch cluster/real_vs_twin.sbatch` from the compare-lead clone after `fleet/link-data compare-lead`, debug QoS, `--comment=E140`,
16 GB, 30-minute limit. Expected about 10 minutes on an L40: model load 2, stage 0 1, stage 1 under 2, stage 2 1, stage 3 about 5.
Free. Figure `plot.py`: real minus twin per method and pair with each seed's share, in the label contrast; a second figure plots numerator
and gain separately (item 5).

**Blind spots.** Steering cannot say whether a vector writes the label the fine-tune's way; a per-word vector sees only what a word's
windows share. The neuron reading misses change outside the TopK sets and in attention. Every gap is blind to both arms moving together.

**Design check.** Goal: part 2's comparison on the pair the share cannot read. Nearest rows: E123 and E125's share, E135's fair steering.
A step, not a patch: the cell's question with the fairness the user set. A result that changes what we do: rival A puts a reading beside
the share that sees real against shuffled; the stop rule ends this line of comparison.

## Not run

Everything above. The held-out-word windows are not read. No reuse verdict under either wording.
