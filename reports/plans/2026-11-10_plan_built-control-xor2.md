# E148: a built control on a task the 67M can learn (XOR2)

*2026-11-10. Cell P1.9 (2 runs, cluster debug, $0, due 11-12). Design: Fable's XOR2 advice, approved by the user
(`~/commute-data/scratch/definition-lead/built-control/fable_advice_xor2.md`). Card: `experiments/built_control/card.txt`.
Filed rules: `experiments/built_control/results/predictions.json`. Nothing has run.*

**In plain words.** The adopted wording of reuse must be able to say "built" when a fine-tune really builds something new;
E139's XOR fine-tunes never learned their XOR, so they could not test that. This run teaches a simpler XOR, of two keyed bits,
to two originals that differ only in whether they already compute it, and reads both with E139's unchanged search. If the
search finds a map only from the original that already had the skill, built is detectable as a contrast; if both arms read
alike, the wording has to be revised.

## What runs

**Task (XOR2).** 128 plain key tokens, drawn after the 67M was trained, each with a balanced seeded bit. Windows are D's
8-token windows: 3/4 end in two keys, 1/4 (far windows) have no key in the last two positions. F is the 67M plus a push at
block 2's input, last position, on windows whose two key bits differ: E137's XOR push (its pieces' Q b) times alpha. Of the
16,384 ordered key pairs, a seeded quarter (4,096) is held out: the fit half (training, ridge fits) uses only the others,
the read half (every reading and check) only these, so passing needs a mechanism, not pair memory. Checked on CPU with the
real rows: cue 26842 as E137, 128 distinct keys, 64 with bit 1, none of them E137's keys, fit and read pairs disjoint.

**Two arms, the same target F.** Plain: fine-tunes of the 67M. Taught: fine-tunes of the 67M first trained toward the 67M plus
a third of the push (the "taught original"). Each fine-tune is read with E139's searches against its own original: H at
five sites (E139 Amendment 2's `search_h`) and G at k = 4 (`search_g`).

**Job 1** (`run.py job1`, about 10 min, 40 min worst case; limit 1:00:00).
- Stage 0, the controls, first and alone: a zero-step fine-tune of the 67M reads gap 0 (under the float floor and 0.02, so no
  reading; E139's rule); the capacity read, newly in `commute.variable_read`, repeats E137's held-out accuracies on E137's
  XOR (0.506 to 0.507) within 0.01.
- Stage 1: alpha (below); the plain target gap, above 0.15; the capacity read of XOR2 on the plain 67M, under 0.55; the
  taught original (300 steps; if its capacity read is under 0.55, 3000 steps from the 67M again), its capacity at least 0.55,
  its target gap above 0.15, and its zero-step control; the plain seed-0 fine-tune at 300 steps and its learned check on
  held-out pairs; if not learned, 3000 steps; if still not learned, the job stops and the cell closes as "not learned".

**Job 2** (`run.py job2`, about 92 min at 300 steps, 94 min at 3000; limit 1:55, wall stop at 105 min): H for seeds 0 and 1 of
each arm (taught s0, plain s0, taught s1, plain s1), then G for seed 0 of each arm. A fine-tune without a reading (not
learned, or gap under 0.02) gets no search.

## Expected values and filed rules (all in predictions.json)

| Reading | Expected | Rule |
|---|---|---|
| Zero-step control, plain and taught | gap 0, max TV 0 (E139 read 8.5e-6 floor) | stop if off the floor or gap >= 0.02 |
| Moved capacity read on E137's XOR | 0.5057 / 0.5063 / 0.5074 / 0.5065 | stop if any site differs by > 0.01 |
| Plain target gap TV(F, 67M) | about 0.3 (at most 0.375: 3/8 of windows pushed) | stop unless > 0.15 |
| Plain 67M capacity (best site) | about 0.50 | stop unless < 0.55 |
| Taught original capacity (best site) | above 0.9 at in3 or out3 | stop unless >= 0.55 after both rungs |
| Taught target gap TV(F, taught original) | about 0.2 | stop unless > 0.15 |
| Plain s0 learned: KL(F‖ft) / KL(F‖67M + half push) | open | learned if <= 0.9 (E139's bar); 300 then 3000 steps; else cell closes |
| Step 10 of every fine-tune | KL(F‖ft) < KL(F‖M), KL(M‖ft) <= KL(M‖F) | stop (E139 Amendment 2's check) |
| Plain arm H, G | H < 0.25 on both seeds, G < 0.25: no map found | |
| Taught arm H, G | H 0.55 to 0.75 on both seeds, G 0.3 to 0.8: rewired (re-weighted if G >= 0.8) | |

**Outcome rule.** Per arm: "above" when both seeds have a reading with H at least 0.5, "below" when both have one under 0.5,
else no call. Taught above and plain below: rival a, built detectable as a contrast (reading a low H as "built" stays a rule
only the user adds). Both above or both below: rival b, the wording needs revising, and nothing further runs (card stop).
A seed without a reading makes its arm a no call, never "below", so a fine-tune that learned nothing cannot pass as built.
The do-nothing fine-tune reads gap 0 and a learned ratio of about 2, so it has no reading.

**Noise and power.** H's seed-to-seed sd across E139's Always on fine-tunes is 0.028; the smallest gap that changes what we
do is 0.2, 7.1 standard errors at 2 seeds per arm (card check).

## Jobs, times and cost

Times are E139's on an L40 (jobs 57673, 57745, 57831): 300 steps 82 s, 3000 steps 810 s, distances 27 s, H at five sites
590 s (ridge 40 s), G at four sites by three starts 1,370 s; E137's capacity read 123 s (job 57208). Job 1: 615 s best, 2,340 s
worst. Job 2: 5,500 s at 300 steps, 5,630 s at 3000 steps (G at in2 only), against a 6,300 s wall stop and the 6,900 s limit.
Both on the cluster's debug QoS, one L40, $0. Card memory as E139 (about 35 GB at the `final` site), host about 6 GB.

    sbatch --comment='E148 job1' --time=01:00:00 cluster/built_control.sbatch job1
    sbatch --comment='E148 job2' cluster/built_control.sbatch job2      (after job 1 prints STAGE 1 DONE and exits 0)

Job 2 needs from job 1: `results/stage1.json` with `passed` true (alpha, rung), `results/large/taught_original.pt` and
`results/large/plain_s0.pt`. Job 1 writes `results/stage0.json` and `stage1.json`; job 2 writes
`results/{taught,plain}_s{0,1}.json` after each search. Then `plot.py` writes `results/readings.json` and
`figures/h_g_by_arm.png`.

## Choices made here (the user was away)

1. **Push size alpha, measured.** On 20,000 read windows, per alpha in 1 to 16 (units of E137's XOR push), the plain gap and
   the gap a perfect taught original would leave, TV(F, 67M + push/3). Chosen: the smallest alpha whose second gap is at
   least 0.2, else the one with the largest. The design asked for a target gap of about 0.3 and both gaps above 0.15; aiming
   the taught gap at 0.2 leaves a margin for an imperfect taught original. Not a stop: the 0.15 bars are.
2. **Push direction:** E137's XOR push (Q b of its pieces), the same direction as E139's XOR, scaled.
3. **Held-out pairs:** a quarter of the ordered pairs, used only by the read half. The learned check reads 4,096 keyed
   windows of held-out pairs.
4. **Capacity on keyed windows only.** A far window has no XOR2; labelling it 0 would let key presence, which is linear,
   read 0.625 on the plain 67M.
5. **The taught original also climbs the 300/3000 ladder** (its own, judged by its capacity read), rather than always
   training 3000 steps: the design's "two minutes" when 300 suffice.
6. **The learned check applies to both arms**, as E139's rule applies it to every XOR imitator; the reference is always
   the 67M plus half the push.
7. **G sites under the 3000-step rung: in2 only** (E139's best site on every seed; Amendment 2 read its other states
   there too), so job 2 fits the debug limit; at 300 steps G reads all four sites.
8. **Order in job 2:** H before G, seed 0 of both arms before seed 1, so a wall stop loses G first.
9. **States saved:** only the two job 2 reads (taught original, plain seed 0, 268 MB each). A job-2 resume retrains
   its fine-tunes from the same seeds.
10. **`capacity` copied, not moved out of E137:** `variable_read`'s rule since 2026-11-09 is that E137's scripts keep their
    copies; stage 0 replays E137's numbers through the new copy. E139's files are unchanged; `run.py` uses their
    `Config`, `distances`, `clone_net` and Amendment 2's searches through `load_experiment`.
11. **Training moved to `commute.variable_read.finetune`:** E139's recipe with Amendment 2's step-10 check, on a
    `Schedule`; E139's own `train` is unchanged.

## Departures from the literature

None beyond E137's and E139's lists: the same wording, searches and cut-offs.

## Not run

Everything above. A third seed per arm; G on seed 1; G at in1, in3 and out3 if the 3000-step rung is needed; the 32-key
fallback in Fable's advice (not approved).
