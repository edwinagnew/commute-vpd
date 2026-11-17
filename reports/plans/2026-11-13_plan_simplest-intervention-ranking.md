# Cell P1.20 (L1.4, 4 runs, due 11-15), E180: the fourth reuse wording tested with a ranking of interventions

*2026-11-13, definition-lead, asked for by the user ("test the definition now with this intuitive ranking. be thorough,
exhaustive and precise"). Testing only: the fourth wording is not adopted, so every reading is "under the fourth wording,
ranked, provisional". No change to `decisions.md`. Wording: `~/commute-data/scratch/definition-session/2026-11-13_reuse-as-simplest-intervention_v4.md`;
card `reports/decisions/2026-11-13_decision_fourth-wording-simplest-intervention.md`. Code: `experiments/simplest_intervention/`.*

## /design check
- **Goal.** Part 1's working definition of reused against new (`GOALS.md`). It moves the gate page: does the fourth
  wording, measured by a ranking, read built edits as built and give every fine-tune a grade?
- **Nearest tried.** E173/E176/E177 (third wording) read the same items through variables named by hand and gave every
  fine-tune "no reading". E139/E168 (second wording) searched one site and stalled at 0.47-0.84 on exact edits. New here:
  the search is over a list of M's variables fixed on M alone, it may combine several, and a reading always exists.
- **Patch or step.** A step: a different, total definition read on the same items.
- **Decisions relied on.** Linear reads only (user, 2026-11-12); E168's and E137's layouts and saved states; E142's old
  pairs. Departs from none.
- **Adopted definition?** No. The bar (ε) and the variable list are the user's; every reading is given at bars 0.8, 0.9,
  0.95 and with the list split by kind, never chosen here.
- **What changes what we do.** Controls misread: the ranking cannot be measured this way, the cell closes "no call". Twin
  ranks below fact on both seeds, and each old pair's reuse side at or below its new side: the wording separates what the
  third could not, and the card's option (a) has its answer. Fact and twin on the same rung: the ranking does not separate
  them, and the gate page says where.
- **Control first, alone.** Run 1 is the do-nothing state and three edits built on M's own list, nothing else. Stop rule below.

## The ranking (the definition's stand-in for K(I | M))
K(I | M) is uncomputable and a prefix-free language L cannot be built honestly here. The ranking replaces it, and is a
flimsy approximation: it assumes that naming one of M's variables is cheap (about log2 of the list size, ~15 bits) and that
writing down a direction M does not have is expensive (768 numbers), and it treats every number alike.

**M's variables (the list), fixed on M alone before any F.** Linear reads only.
- *neurons*: the 4 x 3,072 MLP neuron activations (the input to each block's down projection);
- *directions*: the top 64 principal directions of the residual stream at in0, in1, in2, in3 and out3, on ordinary Pile
  windows run through M;
- *tokens* (read only): the identity of the input token at each position, "x_p = t";
- *logits* (write only): the logit of each vocabulary token at the output.

**An intervention** names written variables W, each given v <- alpha v + beta + sum_j w_j u_j, the u_j read variables computed
upstream of v. Its size n is |W| plus the number of reads. **Rungs**, cheapest first:
0. **none**: the empty intervention.
1. **re-setting**: no reads (each new equation is a constant, gain or shift of v's own value).
2. **new trigger**: some reads, all of them on the list.
3. **new direction**: as 2, plus one free rank-one edit h_s <- h_s + d (a . h_r + c) (a, d not on the list, r <= s), or two.
4. **from scratch**: replace M's output by F's on each input. Always meets any bar, so every F has a rung (totality).

Order: rung, then n. Rungs 1 and 2 are capped at n <= 16 names (16 names cost ~240 bits, still far below one free direction).
**Reading** at bar b: the lowest (rung, n) whose found intervention closes at least b of the gap, closure =
1 - KL(F || M_I) / KL(F || M), mean over held-out windows, next-token distribution at the read position. Rung 0 if
KL(F || M) is under the do-nothing floor. **Grades**: 0 unchanged, 1 re-setting, 2 new trigger, 3 new direction (new
computation), 4 new. Where reuse ends (after 2 is the natural reading) is the user's choice.

**The search (an upper bound on the rung).** On the fit half: (i) one backward pass scores every write variable by the
gradient of KL in its shift beta, and every (write, read) pair by the matching gradient of its weight w; (ii) greedy: add the
best-scored name, refit all chosen numbers by Adam on KL, rescore, repeat to 16; rung 1 adds writes only, rung 2 writes and
reads; (iii) rung 3 fits one free rank-one edit (then two) by Adam for each site pair (r, s), with the rung-2 names kept.
Closure is read on the read half. A better search can only lower a rung.

## Expected readings (filed before running)
| Item | Built as / expected | Why |
|---|---|---|
| do-nothing | rung 0 | F = M |
| L1 one neuron shifted (mlp2, seeded) | rung 1, n = 1, that neuron | built on the list |
| L2 one neuron shifted by a residual direction's value (PC 0 at in1) | rung 2, n = 2 | built on the list |
| L4 a random bit per window, written at in2 | rung 4 (no rung <= 3 reaches 0.8) | unpredictable on held-out windows |
| E173 constant push along Qb | rung 1, n <= 4 (the answer logits) | Qb lies in the answers' unembedding span |
| E173 trigger push, E137 Cue | rung 2 | reads parity / the cue |
| E157 floor (random bit per ordered pair of numbers; read half on held-out pairs) | rung 4 | a rank-one read is additive in the two numbers and cannot hold a random bit of the pair (corrected before filing) |
| E137 XOR | rung 3 or 4 | not a linear read of two list variables |
| fact s0, s1 / twin s0, s1 | fact rung 3, twin rung 2 | the twin's trigger is parity, the fact's a random token set |
| old pairs | reuse side rung <= new side rung, each pair, both seeds | the third wording's ordering, now as a grade |

## Runs (an item no rung closes takes ~20 min over three lists, so sub-runs of ~1-1.5 h: H100 pods, run 1 alone first, then 2a-2d and 3a-3d in parallel; ~$30)
1. **Controls alone**: do-nothing, L1, L2, L4. **Stop** (exit 1, nothing further, cell closes "no call"): do-nothing not
   rung 0; L1 not rung 1 with n = 1 on its own neuron at bar 0.95; L2 not rung <= 2 with n <= 2 at 0.95; L4 at any rung
   <= 3 at bar 0.8.
2. **E173's hand-direction edits, the fact and twin**: planted constant, planted trigger, floor, Always on, Cue, XOR, Table,
   E168's planted fact, fact s0/s1, twin s0/s1.
3. **Old matched pairs**: copy/cipher 24, identity/random 32, case and its twin, repeat and its twin, seeds 0 and 1.
4. **Reserved** for a repair of runs 1-3.

**Numbers per item**: KL(F || M); for rungs 1, 2, 3 the closure at each n (the curve) and the chosen names; the reading at
each bar, with the list split three ways (neurons only, directions only, all), so the user's choice of list is read, not made.
**Blind to**: nonlinear variables; anything off the read windows; interventions the greedy search misses.
**Noise floor**: two runs of the 67M differ by ~1e-5 in total variation (E154 stage 0); the do-nothing KL sets the rung-0 floor.
