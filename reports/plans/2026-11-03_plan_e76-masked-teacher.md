# Plan: E76, teacher training with masks in its loss (ask 25, option 1, taken by default 2026-11-03 21:52)

**The request, in the Terms' words.** D1 asks for a map on Qwen3-1.7B that keeps the decomposition faithful for a real
update. On ask 24 the user chose: the clean-model table is D1's table, and teacher training with masks in its loss is
the map. Teacher training is component training with the loss KL(teacher ‖ student), teacher the fine-tuned model, Γ and
Δ frozen. The change is that the loss gains a second term, read through masks.

**What E77 changed** (independently checked, `reports/reviews/2026-11-03_review_e77-masks-carry.md`; it found no bug, and how much the output still depends on its tokens is not measured). The advice for this run (`2026-11-03_advice_e76-masked-teacher-FABLE.md`) left routing open, and
E77 shows routing decides what the masked term can teach. With all 196 sites masked, a change early in the model
barely reaches the logits, so that term can only be lowered by changes in the last blocks. With sites routed out, those
sites read the full weights and carry the update around the masks, so that term can be lowered without the masks doing
any work. Neither is obviously the map the user meant, so this plan runs both beside the clean term alone.

## Design check

- **Goal.** `GOALS.md`, D1: a map that keeps the decomposition faithful for a real update. This is the map the user
  chose.
- **Nearest tried.** E40 Amendment 1 is the filed recipe: clean term only, never run on a real Qwen update, only
  rehearsed (`experiments/qwen_sft/results/amend1_rehearse_*`). E34 did it on the 67M with no masks in the loss. E72,
  E74, E77 show masked reads barely respond to an update, which no earlier run trained against.
- **Patch or step.** A step: it is the map, not a fix of an earlier run.
- **Decisions.** Rests on the 2026-11-03 row (ask 24). Departs from VPD's fit, stated below.
- **What changes next.** See "What each result does".
- **Control first.** Stage 0, then arm C, alone, with stops. Differences from VPD's fit are listed; each is named, not
  removed, since this is not a replay of it.
- **Class.** Named by the reorient plan and ask 24, so no sign-off is required, but the user sees it first: the recipe
  is new and E77 bears on their choice.

## The run

One job on a cluster L40, $0, about 3 h 30 by the advice's step times (0.82 s clean, ~2.5 s with the masked term; not
measured with a backward), `normal` QoS if the queue is empty at submission. Update: u1 (E40 step 10), since every
reference number is on its rows. About 200 columns per site, the often-on ones by mean g.

| stage | what | stop |
|---|---|---|
| 0 | the hook gap of E77, about a minute (review `2026-11-03_review_e77-masks-carry.md`, "Smallest test"): save the final norm's input and the last matrix's output, rebuild both shifts, then swap the denominator and the removed part one at a time, no threshold beyond the rebuild reading 0.59, 0.86; K0 on E74's 32 rows at seed 74, fp32 components; the same on 8 rows in bf16 for the dtype floor; the share of Σg the 200 columns hold; cache g, free Γ | the rebuild off 0.59, 0.86 (something other than the norm: stop, hook suspect); K0 off 0.0440 by over 10%; share under 50% (the L40 cut cannot carry the on set; ask the user about the paid card) |
| C | control: teacher W0 + `random_change` at 0.01 nats, loss as arm B, 300 steps | the masked term not falling at step 100 is a gradient bug; its fall sets the band on later reads |
| A | clean term alone, 1200 steps | clean KL not below 0.95 of its start by step 100 (rehearsals read 0.74-0.79) |
| B | clean term + masked term with all 196 sites masked, stochastic masks m = g + (1 − g)u, d ~ U(0, 1), 1200 steps | masked term at step 0 outside 1.0-2.0 (expected near 1.37-1.6) |
| L | clean term + masked term under the library's routing (`uniform_k_routes`, k None), 1200 steps | as B |

Rate 1e-6 and 2 × 256 tokens per step, both from the rehearsals at 96 and 384 columns (`amend1_rehearse_e40a1_r3_c*`).
1200 steps is twice the advice's 600: the control fell only to 0.53 of its start at 600 steps, so arm A at 600 would
fail rule (i) by being short of steps, not by placement. Weight on the masked term is 1, the advice's choice, since both
terms start at about the same size. Each arm writes its readings every 50 steps and its final offsets in bf16 to
`results/large/`; arms resume from saved files.

**Departures from VPD's fit** (launch_config, the decomposition's): no adversary, though 90% of its reconstruction
samples are adversarial; Γ frozen on W0; only U and V of 200 of the alive columns move; no faithfulness term (with Δ frozen
it reads the least-squares objective); no hidden-activation terms; Adam at tiny batch; masks from W0's forward, not the
student's.

## Numbers and rules

Read on 8 fixed rows, every 50 steps; rules at the last step. KL direction is stated each time.

| number | W0 / do-nothing | control C | what it is blind to |
|---|---|---|---|
| (i) clean KL(FT ‖ student) | 1.635 | falls | a re-implementation of the same behaviour |
| (ii) clean KL(W0 ‖ student) | 0; δW itself reads 0.626 | small, near 0.01 | which behaviour moved |
| (iii) share of the change's squared size in blocks 21-27, over the same share of δW's | no change to read | near arm A's | a late change that is still a good map |
| (iv) masked term: KL(FT clean ‖ student masked), all 196 sites, whatever the arm trained on | near 1.37 | falls | routes other than all-masked |

The student under its own masks is cut: E72 read 0.103-0.105 for every state on u1, so it cannot separate anything.

- **Works as a map:** (i) ≤ 0.09, the band of placements (E72 u1 T2); (ii) ≤ 0.94, 1.5 × δW's; (iii) ≤ 3; and (iv) below
  arm A's by more than the larger of the dtype floor and C's spread.
- **Not a map:** (i) above 0.53, the remainder's; or (ii) above 0.94; or (iii) above 3.
- **Fails to learn:** (iv) within the floor of arm A's at the last step.
- Anything else is reported as read, with no amendment.
- Arm A is scored on (i)-(iii) too. If A itself reads above 0.53 on (i), the clean term is short of steps or columns at
  this size, and B and L are read against A at equal steps, not against the placements' band.

Leans, filed before the run. B: not a map by (iii), or fails to learn; medium. L: (iii) passes and (iv) barely below A's;
weak. Edge cases for the code writer to run on the rules: a zero denominator in (iii), NaN, a band-edge tie, an arm that
stopped early.

## What each result does

- **B works:** D1 has its map on u1. Next, RL's step 100 on the paid card with the full alive set ($10-20, its own rate
  grid), and the user is told first.
- **B fails, L works:** the map holds only when many sites read full weights. D1 is then stated with that limit, and the
  user decides whether it meets D1 as worded.
- **Both fail:** the masked term cannot do what ask 24 asked of it on this model. Arm A is what is left. Back to the user
  with the numbers, since the map is their choice.
- **A itself fails rule (i) badly at 1200 steps:** teacher training at 200 columns is too small a test on any arm. Bring
  the paid card to the user with this number.

## Not run

Γ moving with the student (a refit, and over the L40's memory); the adversary in the loss; the full alive set; RL's update;
fixed masks in the loss; the 67M with masks in its loss.

## Amendment 1 (2026-11-04): where the masked fall sits

B met the filed "works" rule, but rule (iii) is a share and so blind to size: B's change has 3.4 times δW's squared size (A's has
0.52 times), 68% of it in blocks 21-27 against δW's 25%, and its last block 23.5 times δW's (`results/dw_blocks.json`). The verdict
stands as filed; this reads what it rests on, as the anomaly rule asks. Design check: analysis of saved states, $0, ~15 min, no
training; it bears on D1's claim directly and repeats nothing in `TRIED.md`.

Each arm's saved change is kept on all blocks, 0-20, 21-27 or 27, and read by the arms' own `read_state` on the same rows and draw.
Control first: W0 reproduces step 0, within 1e-3, or the job stops. Filed in `results/predictions_amend1.json`. If the fall sits
late (lean), D1's sentence says so: the map keeps masked forwards faithful by a change concentrated at the end of the model, at
several times the update's size. Blind to: which components carry it and what they do.

## Amendment 2 (2026-11-04): blocks 0-14 against 15-20

Amendment 1 reads B's blocks 0-20 alone as 0.61 of the fall (lean: at most 0.1), which on its face disagrees with E77, where blocks
0, 7 and 14 read nearly deaf. The boundary at 21 came from rule (iii), and most of that part's size sits in blocks 15-20. One read of
the same kind at 15 settles it, as the anomaly rule asks: $0, about 2 min. Filed in `results/predictions_amend2.json`. E76's last
amendment.
