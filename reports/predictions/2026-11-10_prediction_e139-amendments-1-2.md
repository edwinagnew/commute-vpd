# E139 Amendments 1 and 2: is the Always on imitators' low G set by how far they sit from the pair?

*2026-11-10, cell P1.4 (runs: 1 of 3 used; these two amendments are runs 2 and 3). Filed rules:
`experiments/reuse_variable_finetune/results/predictions_amend1.json` and `predictions_amend2.json`. Nothing here has run.
The label rule, the cut-offs (G 0.7943, H 0.4757) and the wording are unchanged; no verdict is given.*

**In plain words.** E139's fine-tunes that imitate the Always on pair read a best G of about 0.63, under the re-weighted
cut-off, and were labelled rewired; but they also stay about a third of the pair's gap away from the pair, and the best G
sits right at the most that distance allows. The first amendment reads the pair's own found read-out against the three
saved fine-tunes, with no search and no training; the second trains the same fine-tunes much longer, at a gentler rate,
until the distance left is a tenth of the gap, and reads them at three points on the way. If G then passes the cut-off,
the rewired label came from an under-trained fine-tune; if it does not, the label stands.

## The numbers this starts from (verified from the files)

| Seed | Gap TV(fine-tune, M) | Distance left TV(fine-tune, pair) | 1 - left/gap | Best G (in2, start 2) | Best H (in3) | Reversion at in2, start 2 |
|---|---|---|---|---|---|---|
| 0 | 0.1586 | 0.0567 | 0.642 | 0.638 | 0.505 | 0.204 |
| 1 | 0.1587 | 0.0569 | 0.641 | 0.627 | 0.450 | 0.197 |
| 2 | 0.1585 | 0.0569 | 0.641 | 0.637 | 0.483 | 0.200 |

H at the other sites reads below G on every seed (in1 0.04, in2 0.19 to 0.22, out3 0.41, final 0.36), with reversions
0.03, 0.25, 0.62 (in3), 0.93 (out3 and final), as the request said. Setter mismatch at the best G start: 0.76, 0.41, 0.59.

## Amendment 1: E137's witness against E139's fine-tunes

The witness is E137's found read-out of Always on at in2 (4 directions and an affine setter, G 0.9225 against the hand-made
pair). It was not saved, but its seed regenerates it: E139's control did so to 15 digits. Its G against a fine-tune is
1 - d(M with the witness written in, fine-tune) / gap, on E139's read half of 100,000 windows, with the reversion and setter
mismatch read the same way. The triangle inequality bounds it: the witness sits 0.0114 from the pair, so against seed 0 its
G lies in [0.570, 0.714] (seeds 1 and 2: [0.570, 0.713]). The user's lower bound of about 0.570 checks out; a value outside
the bounds is a bug. Filed: G between 0.570 and 0.640 on all three seeds; it cannot reach 0.794, so this amendment
cannot change a label. It tells whether the search found as much as the pair's own read-out (inside the band) or missed
a better one (above 0.640 on two seeds). Its reversion is expected near 0.2, against 0.938 at the pair.

## Amendment 2: the same imitators trained to a tenth of the gap

**Recipe change and its source.** The peak rate drops from 3e-5 to 1e-5, the warm-up ratio stays 0.1 with a cosine to
zero, and the steps go from 300 to 3,000. The rate is LLaMA-Factory's documented full fine-tuning setting (written into
`qwen_sft.lr_at`); and E139's own loss rose at its 2.9e-5 peak on every seed (seed 0: 0.025 at steps 20 to 30, 0.050 at
steps 100 to 110) and fell only once the cosine took the rate under about 2e-5, so a higher peak would not help. The step
count is not from a standard: it is ten times E139 and sized to fit one debug job per seed. Everything else is E139's:
batch, clip, loss, windows, the per-seed batch draws and the read code. Weights are saved at steps 300, 1000 and 3000.

**Step-10 check, fixed.** E139 compared KL(M ‖ fine-tune) with KL(F ‖ M), two different directions. The new run compares
KL(F ‖ fine-tune) with KL(F ‖ M) and KL(M ‖ fine-tune) with KL(M ‖ F). At step 10 the warm-up rate is 3e-7, so both should
hold widely.

**Predictions.** (a) G rises with 1 - left/gap and passes 0.794 once the distance left is at most 0.1 of the gap.
A seed is *reached* when, at step 3000, its gap is at least 0.02 and its distance left at most 0.1 of it. (a) holds if two
seeds are reached and, on two of them, best G (4 sites, 3 starts) is at least 0.7943 and G at in2 rose by more than 0.113
(E137's start spread) from step 300 to step 3000. **(a) fails** if two seeds are reached and best G is under 0.7943 on two
of them: the distance left was not the cause, and the rewired reading stands. Anything else is no call, reported as read.
(b) The reversion at in2 stays near 0.2: holds if it is under 0.569 (midway between E139's 0.200 and the pair's 0.938) on
two reached seeds, fails if at or above on two. Edge cases are worked through in the filed file: ties count at or above,
the hold and fail branches cannot both be met with three seeds, and a reached seed whose witness reads above the cut-off
while the search does not is a search limit. The witness alone is bounded to about 0.82 or more at a reached state.

## Expected values at the control and at do-nothing

| Number | Control (witness against the pair) | Do-nothing (the 67M, zero steps) | Amendment 1 | Amendment 2, reached state |
|---|---|---|---|---|
| Witness G | 0.9225 within 0.02, else exit 1 | no reading: gap 0 is under 0.02 | 0.570 to 0.640 | at least about 0.82 |
| Gap | 0.1471 within 0.001 | 0 | E139's within 0.001, else exit 1 | near 0.15 |
| Best G | not read | no reading | not read | at least 0.7943 if (a) holds |
| Reversion at in2 | 0.938 | no reading | near 0.2 | under 0.569 if (b) holds |

The do-nothing state can meet no rule: every rule needs a gap of at least 0.02 before any ratio is formed.

## Design check, a line each

- **Goal:** the working definition's first test on ordinary fine-tunes (paper part 1); this checks whether E139's one
  surprising label is a property of the fine-tune or of its training.
- **Nearest tried:** E139 itself (TRIED row); it settled the labels at 300 steps, not whether the distance left sets G.
- **Patch or step:** a patch on E139, inside cell P1.4's run count.
- **Decisions:** none departed from; cut-offs and label rule as filed.
- **Definition:** read under the filed rule unchanged; no rule, band or clause added to the label.
- **What changes next:** (a) holding retracts the rewired reading of Always on as a training artefact; (a) failing keeps it.
- **Control first:** the witness against the pair, alone and first in each job; E139's control must have passed.
- **Saved for audit:** per-step JSON after each search; the three states per seed (268 MB each) in `results/large/`.

## Time, memory and cost (from E139's logs, jobs 57673 and 57675, L40)

Set-up 23 s; 300 training steps about 80 s (0.27 s a step); distances 27 s; one G search 106 to 118 s; ridge 40 s; one
H site 95 to 136 s; host memory 6.1 GB at peak. Card memory is not logged; E137 put the H `final` site at about 35 GB.
Amendment 1: about 6 min, one job. Amendment 2, per seed: training about 14 min; step 3000 read in full (12 G searches,
5 H sites) about 34 min; steps 1000 and 300 with G at in2 only, about 17 min each; with the witness and distances, about
85 min, under the job's 105-minute stop. Four debug jobs in all, $0 on L40 cards; three seeds side by side finish in
about 1.5 hours of wall clock, so no RunPod H100 is needed. If a job hits its stop, resubmitting the same line resumes.

## Not run

Nothing above has run. The new recipe's leftover at step 3000 is unknown: if it stays above 0.1 of the gap, (a) and (b)
are no call. The 1e-5 peak is not measured on this task; its first-minute values are logged at steps 1 to 300.
