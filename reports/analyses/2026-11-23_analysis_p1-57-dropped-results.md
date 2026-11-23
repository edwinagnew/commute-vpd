# Does the write share over the needed set tell copy from lookup, or rank the five key tasks?

Cell P1.57, 2026-11-23. Two results the paper's second draft dropped, read again under the frozen definition's needed
set: the copying-head pair and the five-task key grid at one update size. Cluster and short H100 pods, about $0.

We read the write share over the needed set on the copying-head pair, copy against lookup, and on the five key tasks
scaled to one update size, each beside random updates of the same size. Neither result separates anything: copy and
lookup read the same, every task reads within about a hundredth of every other, and, unexpectedly, the order runs
against the filed expectation at both ends, with the cipher highest, copy second and the prior-matched cipher below the
plain cipher. For the goals, this reading cannot carry the claim that copying reuses the base model while a cipher is
new: on these items it measures how far any trained update turns each variable's write, and that is small for all of
them.

## The answer

The write share over the needed set does not tell the copying fine-tune from the lookup fine-tune, nor rank the five
key tasks the way the squared-cosine reading did. Every trained update reads two to three times its random update and
a twentieth or less of what a pure strengthening edit reads. The ordering surprise is real but tiny: the cipher sits above
copy on every seed, by about a fortieth of the gap that was filed as meaningful. Reading only the copy windows does not
change this.

**What the reading is.** Variables are VPD's subcomponents of the 24 decomposed matrices plus one remainder per matrix,
38,938 in all. The needed set K holds every variable that, swapped for its value on another input of the task data,
flips the fine-tuned model's top token on at least one of 256 tested pairs; its size |K| is given beside every value.
The write share over K, written E_b, starts from the change in each needed variable's write between the fine-tuned
and base model on the same input. It keeps the part of that change that lies along the variable's own original write,
and divides the summed kept parts by the summed change. It reads 1 when the update only rescales each needed variable's write. The built edit that
doubles every weight reads 0.761, and a random update matched in size reads 0.011. It is a reading of the frozen
definition, not the definition, and it is not admissible as computed. The margin beside each value is the smallest gap
between the fine-tuned model's top two logits over the task data.

## The key grid: are the five tasks ordered as before?

![Write share over the needed set: E62 copy against lookup, left; the five key-grid tasks at one update size beside their matched random updates, right](../../experiments/eb_dropped/figures/eb_over_K.png)

**Reading it.** No. All fifteen trained states sit between 0.0274 and 0.0398, every task within 0.012 of every other,
against the filed gap of 0.2. Each task sits two to three times above its grey band of three matched random updates,
which read 0.0131 to 0.0169. The order reverses the filed expectation at both ends: the cipher, expected lowest, is
highest on every seed, and copy, expected highest, is second, with no overlap between their seeds. The cipher's mean
sits 0.0055 above copy's. The prior-matched cipher, whose answers the base model already finds likely, reads below the
plain cipher. Caveat: each step of this order is about forty times under the filed gap, so it is an ordering of small
numbers, not a separation.

| task, three seeds | E_b over K | |K| | smallest margin, logits | matched random |
|---|---|---|---|---|
| cipher | 0.0374, 0.0375, 0.0398 | 22,925 to 35,226 | 0.0006 to 0.0022 | 0.0131 to 0.0138 |
| prior-matched cipher | 0.0293, 0.0295, 0.0297 | 26,871 to 33,252 | 0.0022 to 0.0039 | 0.0138 to 0.0169 |
| copy | 0.0320, 0.0324, 0.0339 | 16,708 to 24,665 | 0.0004 to 0.0051 | 0.0144 to 0.0150 |
| mid, a token from the base model's top 20 | 0.0274, 0.0300, 0.0301 | 23,470 to 31,319 | 0.0002 to 0.0027 | 0.0133 to 0.0149 |
| top, the base model's own top token | 0.0284, 0.0288, 0.0292 | 16,103 to 29,271 | 0.0005 to 0.0038 | 0.0136 to 0.0149 |

Every |K| is far above the floor of 100 below which a value reads "too small to read"; K holds 41 to 90 percent of all
variables, so restricting to it changes little. The redteam verdict on the grid: rival B as filed, all tasks within
0.03 against a gap bar of 0.2; "two to three times the randoms" is description only; the order reversal is a
qualitative surprise; no reuse or new reading is drawn.

## The copying-head pair: does copy read above lookup?

No. The copying fine-tune reads 0.036 to 0.040 and the lookup 0.040 to 0.043. On the three-seed mean, copy sits 0.004
below lookup. These states are 8-token windows read at the marker, a mix in which 206 of 512 windows are copy
windows, not the original task's longer windows, and the copy head's job depends on context length. The six values
were taken from an earlier read of the same states, not re-read.

| seed | copy E_b | copy |K| | copy margin | lookup E_b | lookup |K| | lookup margin |
|---|---|---|---|---|---|---|
| 0 | 0.0376 | 17,382 | 0.0034 | 0.0399 | 11,186 | 0.0063 |
| 1 | 0.0399 | 30,688 | 0.0012 | 0.0431 | 15,904 | 0.0018 |
| 2 | 0.0362 | 12,145 | 0.0040 | 0.0429 | 9,271 | 0.0140 |

Redteam verdict: "On E62's copying-head pair, the write share over the needed set does not separate copy from lookup
(copy minus lookup -0.004 over three seeds, against a 0.2 bar), and both read about 0.04, far below the 0.76 of the
built strengthening edit. On this reading, the user's worked reuse example does not look like strengthening." There is
no matched random for this pair, so 0.04 is "near the random level seen elsewhere", not "at random".

**On the copy windows alone.** We read the same six states on the 206 copy windows only, with the needed set found again
there. Copy over lookup is 0.61 on the three-seed means, below the filed ratio of 1.5, so lookup is not diluting a real
difference. Two of the three lookup seeds have a needed set under the floor, 97 and 75 variables, so they read "too
small to read"; over the all-data needed set on the copy windows the ratio is 0.98. The stage-0 repeat of a saved state
matched to 2.1e-9 against a bar of 2.7e-9.

| state | E_b over copy-window K | copy-window |K| | E_b over all-data K, copy windows | smallest margin, copy windows |
|---|---|---|---|---|
| copy seed 0 | 0.0300 | 208 | 0.0649 | 1.0167 |
| copy seed 1 | 0.0477 | 1,184 | 0.0710 | 0.1008 |
| copy seed 2 | 0.0346 | 673 | 0.0644 | 0.1720 |
| lookup seed 0 | too small to read | 97 | 0.0625 | 0.4175 |
| lookup seed 1 | 0.0704 | 17,732 | 0.0705 | 0.0072 |
| lookup seed 2 | too small to read | 75 | 0.0723 | 0.6805 |

The copy-window values for lookup seeds 0 and 2, 0.0193 and 0.0936, enter the filed ratio but are under the floor.
E281 verdict: pending redteam-lead.

## Why the copy head does not show: a head confined change caps near a fifth

VPD decomposes each attention matrix whole, so no subcomponent belongs to one head. A built edit that doubles only the
copy head's value or output reads about 0.2 at VPD and 1.0 at a featurization where the head is one variable.

| built edit on the copy head | E_b over K at VPD | E_b over K, head as one variable |
|---|---|---|
| value doubled | 0.192 | 1.000 |
| output doubled | 0.200 | 1.000 |
| query doubled | 0.249 | 0.040 |

Redteam verdict: "head-mixing caps E_b at ~0.2 for a one-head change; E62's updates sit far below that cap." The
fine-tunes read about a fifth of that cap, near the random update's 0.011, because almost no variable's write change
lies along its original write. The query edit's 0.249 at VPD comes from the weight algebra, not from VPD seeing the
sharpening, and doubling the query raised the copy loss, so it is not a useful strengthening.

## A failed control: weighting by flips

A column that pooled the kept share by how often each variable flips the answer failed its positive control at both
featurizations. The doubled-weights edit read 0.012 at VPD, where 0.5 or more was expected. With heads as variables it
read 0.772, where about one was expected. It rewards being needed, not being changed. Its held-out values are not evidence for
anything.

## Scorecard

| card | claim in words | rule | read | verdict |
|---|---|---|---|---|
| grid | tasks ordered copy highest, cipher lowest, each above its random | gap 0.2 | all within 0.012, cipher highest | rival B; order surprise |
| E62 pair | copy at least 0.2 above lookup | gap 0.2 | copy 0.004 below | rival B |
| copy windows | copy over lookup at least 1.5 | ratio | 0.61 over copy-window K | rival B, verdict pending |
| head edit | value doubled reads at least 0.5 at VPD | 0.5 | 0.192 | rival B |

## Departures from the literature

- The frozen definition, after Geiger et al. 2025, calls a variable kept only when its fine-tuned mechanism is exactly
  a positive multiple of the base one on every setting that occurs. The write share replaces that yes-or-no test with a
  pooled fraction weighted by write change, which is why it is a reading and not admissible as computed.
- The copying pair is read on 8-token windows at the marker, not on the original task's 41-token windows.

## Not run

- A matched random update for the copying-head pair.
- A strengthening edit on the copy head that lowers copy loss, such as scaling it by 1.2: needed to say whether a useful
  sharpening reads as new.
- The key grid at the head-as-one-variable featurization, or at more than one update size or three seeds.
- A repaired flip weighting, such as flips times write change: a new design needing its own cell.
- Redteam verdict on the copy-window read.

Provenance: `experiments/eb_dropped/results/summary.json` for the grid and the pair, the pair's items from
`experiments/claim2_needed/results/items/e62-*.json`; `experiments/e62_copy_windows/results/summary.json` for the copy
windows; `experiments/head_edit/results/summary.json` for the built head edits;
`experiments/eb_dropped/results/flip_weighted.json` for the doubled-weights control and the randoms. Diagnosis in
`reports/analyses/2026-11-23_analysis_e62-eb-diagnosis.md`.
