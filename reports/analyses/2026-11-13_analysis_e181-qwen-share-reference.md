# Does the share on the frozen-neuron Qwen decomposition see neurons being switched off?

*2026-11-13, qwen-lead's writer. Cell P3.10, run E181 on two H100 pods, both terminated. Card
`experiments/qwen_share_reference/card.md`; rules `experiments/qwen_share_reference/results/predictions.json`; plan
`reports/plans/2026-11-13_plan_p310-share-reference.md`.*

We switched off sets of MLP neurons that Qwen3-1.7B uses on each of five kinds of text, and compared the share these
changes read on the frozen-neuron decomposition with the share of random weight changes of the same size. Where a
switched-off set could be compared at all, it mostly read above the random changes, but only seven of the fifteen sets
landed at the size the comparison needs, so the filed rule makes no call. For `GOALS.md` this means the share on
this decomposition has not yet been shown to read a change that is a switching-off by construction, so it cannot yet
serve as the reference reading for part 3.

## The answer

No call. The filed rule needs nine of the fifteen built changes to be comparable, and only seven were. Five of those
seven sat above the random changes by more than the random spread, and two sat inside it. Most of the rest overshot or
undershot their target size, because the most-used neurons include a few so large that switching off one more moves the
output by several nats. Placed at their own size anyway, they lean the same way but do not reach the rule's bar. The
library's fine-tunes also split: some above the random changes, some inside.

## What we were trying to test

The share is E57's re-gating share on the frozen-neuron decomposition: subcomponents fixed at the 516,092 live MLP
neurons, only the causal-importance function Γ trained, the 84 MLP matrices only. For a change $\delta W$ it is
$(K_{00} - K_{01}) / (K_{00} - K_{11})$. Here $K_{00}$ is the KL between the changed model and the original
subcomponents under the original model's masks, $K_{01}$ the same with Γ's masks recomputed on the changed model, and
$K_{11}$ the same with the least-squares split of $\delta W$ added as well. A share of 1 means Γ's recomputed masks on the
original subcomponents recover all that the split recovers; 0 means they recover none. We call $K_{00}$ the change's
size: mean nats per position over a row set's 128 rows of 256 tokens.

On each of the pair library's five row sets, three kinds of change were read:

- **Built changes**: a set of neurons the model uses on those rows, switched off by zeroing their gate_proj rows. Each
  neuron is one subcomponent, so the change is exactly a mask of 0 on it. Three choices of set: the most used, a random
  order among the used, and those used most on this text against the other four. The count was searched so the size
  lands at the middle of the row set's fine-tunes' sizes.
- **Random changes**: Gaussian noise on the 84 MLP matrices at the norms of a fine-tune, three draws, each scaled to the
  lowest, middle and highest fine-tune size. They give the random curve: share against size. The random band is the
  largest distance of any random read from that curve.
- **Fine-tunes**: the library's 36 read fine-tunes, from the earlier cell P3.9, placed on the same curve at their own
  size. No state was refetched.

The built change rules out the reading that the share only tracks size: it is a switching-off by construction, so a
share that sees switching-off should read it above random changes of the same size.

## What we expected, and why

Two rivals were filed before the run. The first says Γ reads each neuron's actual output at down_proj, so a zeroed
neuron is switched off by Γ's own masks. Under it the built changes sit above the band on at least 80% of the
comparable reads. The second says the share on this decomposition tracks only the change's output size, as on the
other two Qwen decompositions. Under it the built changes sit inside the band. The writer leaned weakly to the first.

## What we observed

![Each change's share minus the random curve at its own size, per row set, with the random band](../../experiments/qwen_share_reference/figures/share_reference.png)

**Reading it.** Of the seven built changes that landed at their target size, five read above the random band and two
inside it, short of the nine comparable reads the rule needs. The figure plots each change's share minus the random
curve at its own size, with the grey band marking how far random changes stray from that curve. On known rows the set
used most on this text sat 0.094 above the curve, against a band of 0.016. On gold rows two of three sat above a band
of 0.014. On copy rows the band is 0.059, about four times the others, and every built change sat inside it. Placed at
their own size, ten of the fifteen can be compared, with six above and four inside: the same lean, still not 80%. The
fine-tunes split the same way: all twelve on gold rows and all six on known rows sit above the band, all six on copy rows
inside it. On cipher and invented rows the early saves sit above and most late saves inside.

### The full placement of the built changes

| row set | set choice | neurons | size | target | share | gap to curve | band | class |
|---|---|---|---|---|---|---|---|---|
| copy | most used | 1 | 0.65 | 0.85 | 0.985 | -0.017 | 0.059 | off target |
| copy | random among used | 1,335 | 0.92 | 0.85 | 0.961 | -0.036 | 0.059 | inside |
| copy | used most here | 203 | 1.54 | 0.85 | 0.986 | 0.007 | 0.059 | off target |
| cipher | most used | 1 | 0.53 | 1.21 | 1.005 | none | 0.036 | off target, below the curve's span |
| cipher | random among used | 1,762 | 1.16 | 1.21 | 0.983 | 0.046 | 0.036 | above |
| cipher | used most here | 278 | 0.58 | 1.21 | 0.920 | none | 0.036 | off target, below the curve's span |
| known | most used | 2 | 0.35 | 0.48 | 0.976 | 0.081 | 0.016 | off target |
| known | random among used | 1,461 | 0.47 | 0.48 | 0.921 | none | 0.016 | unscored |
| known | used most here | 189 | 0.45 | 0.48 | 0.992 | 0.094 | 0.016 | above |
| invented | most used | 2 | 0.47 | 0.91 | 0.977 | none | 0.018 | off target, below the curve's span |
| invented | random among used | 2,174 | 0.92 | 0.91 | 0.927 | 0.015 | 0.018 | inside |
| invented | used most here | 817 | 0.92 | 0.91 | 0.933 | 0.021 | 0.018 | above |
| gold | most used | 4 | 3.59 | 0.67 | 0.994 | none | 0.014 | off target, above the curve's span |
| gold | random among used | 3,632 | 0.68 | 0.67 | 0.946 | 0.044 | 0.014 | above |
| gold | used most here | 895 | 0.67 | 0.67 | 0.978 | 0.076 | 0.014 | above |

Size and target are $K_{00}$ in nats per position; "off target" means the size is more than 10% from the target, which
the filed rule reads as unreadable. The gap for an off-target read is its share minus the curve at its own size, shown
for description only.

### The fine-tunes, per row set

| row set | fine-tunes | above the band | inside | gap range |
|---|---|---|---|---|
| copy | 6 | 0 | 6 | -0.011 to 0.013 |
| cipher | 6 | 4 | 2 | 0.030 to 0.045 |
| known | 6 | 6 | 0 | 0.048 to 0.098 |
| invented | 6 | 3 | 3 | 0.008 to 0.096 |
| gold, both arms | 12 | 12 | 0 | 0.047 to 0.077 |

On invented rows the three above the band are the early saves and the three inside it are the late saves.

## What we learned

The share on this decomposition does move with a known switching-off on some text: on known and gold rows, sets of
neurons the model uses read clearly above random changes of the same size. On copy rows it does not, and there the
random changes themselves scatter widely. So the share separates a switching-off from random noise on some row sets and
not on others. It is not yet a reference for part 3. The fine-tunes' placements describe where they sit, and carry no
label about how they were learned.

## How compelling is the evidence

Weak. The rule's minimum of comparable reads was missed by two. Each row set has one built change per set choice and one
seed, so a single read decides each cell of the table. The band is the largest of at most nine random reads, so it is a
rough floor. The seed spread of one fine-tune's share in P3.9 was under half the smallest band here. The share is blind to
which neurons moved, and the built change only tests switching off, never switching on. Attention matrices are left
unchanged by every change, since the decomposition does not cover them.

## Surprises

- The most-used neurons include a few giants. On every row set, switching off the first one or two of the most-used
  order moved the size by under half a nat, and the first three or four moved it by over three nats. So no count of
  that order could land within 10% of its target, which is the main reason the rule could not make a call.
- On copy rows the first random draw read a share above 1 at two sizes. That widens copy's band to about four times the
  other row sets', and every copy read, built or fine-tuned, falls inside it.

## What next

- Built changes that skip the few giant neurons, so the count can land on its size target. If they read above the band
  as the comparable ones did, the share does see switching-off on this decomposition. This needs a new cell from the
  user; the cell's one run is spent.
- More random draws on copy rows, to see whether the wide band is noise in a few draws or a property of copy text.

## Scorecard

| id | claim in words | rule | read | verdict |
|---|---|---|---|---|
| stage 0, W0 | the unchanged model reads its own floor | K00, K01, K10, K11 equal K0 within 1e-6, per row set | equal on all five row sets | holds |
| stage 0, do-nothing | the split path returns a share of 1 | share within 1e-6 of 1 | 1 on copy rows, pod a, and on invented rows, pod b | holds |
| K0 floor | every read used the same rows as P3.9 | K0 against P3.9's floors within 1e-3 | 67 of 67 checks ok in the pod logs | holds |
| built | rival A, switched-off sets above random; rival B, inside | at least 80% of at least 9 readable of 15 | 7 readable: 5 above, 2 inside | no call, unreadable |
| fine-tunes | where the library's fine-tunes sit | at least 80% of at least 27 readable of 36 | 36 readable: 25 above, 11 inside | mixed |

## Departures

- From the plan: spend was about $26.6 on two pods, against about $12 planned and the cell's $20 without asking. The
  second pod split the row sets, as the plan allowed.
- From the plan: seven of fifteen built changes are off their size target, and one is unscored because its split
  left more than half the change's size unexplained. One random read, copy's third draw at the lowest size, is unscored for the same reason; two known
  random reads are off target. Every random size level still had at least two usable draws.
- The do-nothing control ran once per pod, on its first row set, so cipher, known and gold rows have none. The cell
  asks for it first, and it passed on both pods.
- The random curve is three draws each scaled to three sizes, nine reads per row set, as the cell asks.
- From the literature: none beyond those of E57's share, recorded with that experiment.

## Not run

- The figure and `results/summary.json`: the compute node was down from 16:20 on 2026-11-13, so neither `run.py
  --summary` nor `plot.py` could run. The numbers above come from the same placement rule run in jq on the result
  files; the figure appears when `plot.py` runs.
- Lint of the edited `run.py` and `plot.py`, for the same reason.
- Built changes that skip the giant neurons; a switching-on control, which this construction cannot build.
- Any second seed of a built change.

## Provenance

Reads: `experiments/qwen_share_reference/results/{control,built,random}/*.json`; pod logs `run_a_pod.log` for copy,
cipher and known on an H100 PCIe, and `run_b_pod.log` for invented and gold on an H100 NVL. Fine-tunes: P3.9's
`experiments/qwen_library_share/results/p-015e0710/run2/`. Placement: `commute.share_reading` through
`run.py --summary`, mirrored in jq while the compute node was down.
