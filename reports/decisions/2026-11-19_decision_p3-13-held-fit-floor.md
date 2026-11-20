# P3.13, due 11-25: the held-model reading on the pair library hit a fit floor on four of its five row sets

*2026-11-19, qwen-lead. Pending decision, raised after run 2 of the held-model reading on Qwen3-1.7B. `decisions.md` and
`GOALS.md` are unchanged.*

## The answer

On the pair library the held-model reading does not separate any of the three pairs. Gold against shuffled gold reads
cleanly. The four word-library row sets do not: the held model there differs from the real model even when nothing
changed. That voids their random references. One more run could fix the fit and read those pairs, or we fill the cell
as it stands.

## What has been measured

The held-model reading, called S below, asks how much of a fine-tune's behaviour change survives when each site's
output matrix is replaced by a fitted map. It reads 1 when the update is fully carried and lower when it is not.

- **Run 1** read S on Qwen's RL, collapsed and own-samples fine-tune updates. S was 0.98 to 0.997, against 0.79 to 0.82
  for random changes of the same size. The do-nothing control read 3.5e-9.
- **Run 2** cost $6.09 and read S on the pair library: copy against cipher, known against invented facts, and gold
  against shuffled gold. The filed rule says "does not separate" on all three pairs.

## What did not come out as expected

The held model H is the real model M with each site's output matrix replaced by a least-squares map. The map is fitted
on 64 rows and read on 64 others. Fitted from M itself, H should equal M.

- **On gold's rows it does.** The divergence between the fitted model and H is 4.3e-9 nats a position.
- **On the four word-library row sets it does not.** It reads 2.2e-3 for copy, 1.2e-3 for cipher, 3.9e-4 for invented
  and 2.8e-4 for known. The filed expectation was at most 1e-7.
- **The control could not catch it.** Its stop bar checked the divergence between the fitted model and M, which is 0
  whenever the fitted model is M. It passed without testing this.

Two things follow. Every random reference on those rows reads "no change", because its floor is set at 100 times that
number. And each state's S on those rows carries this error too.

One state shows it. Shrinking cipher's late update by 0.46 raised its S from 0.93 to 0.99, where the filed expectation
was a drop. On gold's clean rows, shrinking left S unchanged, 0.997 before and 0.997 after.

## What reads cleanly

Gold reads 0.996 to 0.999 and shuffled gold 0.997 to 0.998. The random reference reads about 0.81. The gaps between the
two are within 0.003 at every save and size, so gold does not separate from shuffled gold.

## What is not known

The cause is not verified. The leading guess is that on the packed word rows the unit activations on the read rows fall
outside what the 64 fit rows span, so the fitted map does not carry over. Gold's rows are GSM8K text and run 1's are
maths.

On the word pairs the saved-size gaps were 0.014 to 0.066 and the size-matched gaps 0.003 to 0.006. Whether they
survive once the fit error is gone is unknown.

## The question

Should P3.13 get one more run that reads S on copy against cipher and known against invented with a fit that
reproduces M on their rows, or should the cell be filled with what was read?

## Options

- **1. One more run (recommended).** About 40 minutes on three H100 pods, about $6 to $8. It first measures the
  do-nothing floor on each word row set at a few fit sizes and fit sources: more fit rows, and fitting on the pair's rows
  plus maths rows. It stops unless the floor falls under 1e-7. It then rereads the 24 states of those two pairs with
  their random references. This tells us whether the small gaps survive without the noise, and whether shrinking really
  raises S on Qwen. Next, section 6 reports S on all three pairs.
- **2. Fill the cell now.** Section 6 reports S on run 1's states and on gold, and marks the word pairs as unreadable at
  this fit, with the floor stated. This runs nothing. It tells us only what gold and run 1 already show. The two word
  pairs stay unread on Qwen.

## Links

- `experiments/qwen_held_readings/results/run2/summary.json`
- `experiments/qwen_held_readings/figures/run2_S.png`
- `experiments/TRIED.md` rows E216 and E219.

## Not run

- Alternative fits (more fit rows, fitting on pair plus maths rows): these are what option 1 would run.
- S on 1e-6 step-100 RL states.
