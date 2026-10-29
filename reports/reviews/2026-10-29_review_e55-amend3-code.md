# E55 Amendment 3: code review (2026-10-29)

Scope: `git show 9521f5c` (experiments/pcfg_anchors/amend3.py, results/predictions_amend3.json, the `amend3` stage
of cluster/pcfg_anchors.sbatch). Reviewer: a fresh agent that wrote neither the code nor the Amendment 2 analysis.
Checks ran model-free through `fleet/cpu-run`; scripts in `~/commute-data/scratch/reuse-lead/e55-a3-review/`.

Verdict: no blocker. The code does what the filing says. One filing issue should be fixed before the run: P2 is
likely to come out void, and the filing gives void a 0% lean.

## Findings

### 1. P2 is likely void, and the filing does not expect it (fix before run, in the filing)
`amend3.py:174-184` (`third_on_new`); predictions `P2_third_on_new_pairs.expected_valid_pairs` and `.lean`.

A new pair counts as valid only if LOW's 'C a 40' answers a, and N1 10K answers b, on ≥ 0.95 of its rows. Five of
the six new pairs sit on a range edge: (9, 14) and (9, 17) have a = 9, and (16, 9), (15, 9) and (17, 9) have b = 9.
Only (17, 10) avoids both edges. Enumerating every pair with a and b in 9-17 where the third rule's answer differs
from the other four rules (`e55-a3-review/pairs.py`) gives no further pair off the edges outside amend1's ten. The
only other new one is (14, 9), which also has b = 9.
- LOW at a = 9 already fails on natural text: in amend2's fresh rows LOW's share answering a at (9, 15) and (9, 16)
  is 0 (1 row each). P1b's own lean is "does not count at 9" at 60%.
- N1 at b = 9 has never been shown to count. Stage A's 4,000 natural count-b rows hold no b = 9 row (a2_N1.json and
  a2_N1s1.json `count_b_te0.hist` start at 10). Its only natural b = 9 row so far, amend2 fresh (17, 9), was wrong:
  `controls.fresh_N1_10000.cell_read.cells["17,9"].b = 0`, n = 1.

`expected_valid_pairs` says "4 or 6" and takes N1 at b = 9 as given. If N1 cannot count b at 9 and LOW cannot count
a at 9, only (17, 10) is valid and both W1 10K and W1s1 10K read 'void'. That outcome is plausible, perhaps the
likeliest single one, yet the lean ("fails 45%, mixed 40%, holds 15%") gives void nothing. The code handles void
correctly (it is not a degenerate rule). The issue is that the run would most likely not answer P2 at all.
- Failure scenario: the run comes back 'void' for P2, so the $0 run fails to answer one of its two questions.
  Answering it would then need a new amendment with new pairs, and none exist in range.
- Fix before the run: (a) restate `expected_valid_pairs` with N1 at b = 9 as an unknown, and give void a lean
  (each edge being invalid already carries ≥ 40% by the evidence above). (b) File what void means: "the third rule
  cannot be tested on new natural pairs inside the range where LOW and N1 both count". (c) Optionally file a
  reported-only per-pair read of (17, 10), which is valid if LOW counts a = 17 (amend2: 9 of 9 natural a = 17 rows).
  The modes are already in `third_on_new.modes`, so (c) needs no code change. A second filed outcome may also be
  worth adding: P2 restricted to the pairs where N1 counts b, with LOW's a = 9 failure set aside. Whether that is
  sound is the lead's decision, not this review's.

### 2. The te1 identity control compares the code with itself (minor; verified externally here)
`amend3.py:145` (`te1_rows_are_stage_a_rows`) rebuilds `eval_seqs(B, 4000, 1.0)` without the wrapper and compares it
with the wrapped rebuild. That shows the wrapper changes nothing. It cannot show that the rows equal the ones Stage A
read, if pcfg or eval_seqs had changed since the a2 runs. Checked here:
- `src/commute/pcfg.py` last changed in 0ee393c (19:52), before the A2 arms ran (7e62cb4, 20:44).
- `eval_seqs` (run.py:175-178) is unchanged since then.
- The rebuilt rows' label histogram {10: 9, 11: 143, 12: 746, 13: 1391, 14: 1222, 15: 438, 16: 48, 17: 3} equals
  `a2_N1.json states.10000.count_b_te1.hist` exactly.

So the rows are Stage A's. Fix (optional): turn the histogram comparison against a2_N1.json into the control, which
then makes an external comparison.

### 3. The c c c control finds only lowering swaps (minor, wording)
`amend3.py:127-131, 147`; predictions `definitions.ccc`, P6(a). On the 4,000 te1 rows, 1,069 of the 2,451 rows where
the swap lowered a hold c c c. None of the 789 rows where it raised a do, and none of the 760 unswapped rows
(`e55-a3-review/ccc.py`). The control would catch B reading spurious rows wholesale: about a third of them would show
c c c. It would not catch a small share. What carries P6(a) is the code path, and I confirmed it (see "Checked
fine"). Fix: word P6(a) as "by the code path; the c c c control is consistent with it", not "iff the control is
true".

### 4. |swap| pools both swap directions inside b = 13 and b = 14 (minor, reporting)
`amend3.py:118-119, 187-207`. Within each b the direction of the swap is mostly fixed: b ≤ 12 lowers a, b ≥ 15
raises it. b = 14 is split 368 lowering and 345 raising, and b = 13 is 1,180 lowering and 20 raising. The within-b
slope at b = 14 therefore mixes "a → c" rows with "c → a" rows of the same size. That is fine for the filed question,
which is whether the read depends on swap size. But if a 'depends' verdict comes mainly from one direction, the
`by_b_swap` cells cannot show it. Fix (optional, reporting only): also give `by_b_swap` split by sign(a before − a
after).

### 5. Several filed verdicts are scored by hand (minor)
P1b's n-weighted share over three pairs, W0's "within 0.02 and same label", and P6(b)/(c) are combined by hand from
amend3.json. The arithmetic is simple, but each is one more place for a slip. Fix (optional): have the analysis
compute them in a few lines and quote that output.

### 6. Results are written only at the end (minor)
`amend3.py:286`: amend3.json is written once, after Part 2. A timeout or crash in Part 2 would lose Part 1's JSON;
the log does keep every number (`pair_read` and `swap_read` log all of them). The runtime risk is low:
- a0 measured 35,000 pretraining sequences per second on the GPU node (`a0.json generation_rate`, mostly
  sample_texts). The dry run's 8M texts in 327 s on the CPU worker is about 24,500 per second, so sampling on the
  GPU node should take about 230 s.
- amend2, with similar model loading and forward passes, took 68 s end to end (job 44520).
- Expected total: about 4-6 min, within `--time=00:10:00`.

## Checked fine
- **B, P and the reference reads use natural text only.** `run.read_state` (run.py:402-412) reads B on
  `data.b_texts = qualifying_texts([1, 51], 1000)` and P on `make_pairs([1, 50, op])`. `behaviour_nulls` uses
  `qualifying_texts([1, 31, i])` and `patch_references` uses `make_pairs([1, 30, i, op])`.
  `interchange.qualifying_texts` (interchange.py:38-46) and `make_pairs` both draw `pcfg.sample_texts` with no swap.
  Only `scored(... count_b_te1)` reads swap-built rows. So the writer's claim holds, and Stage A's B and P labels
  are not affected by swap-built text. The fine-tuning rows are the other swap-built input (task_batch with c_rate
  set to the arm's C_Tr).
- **The rebuilt te1 rows are Stage A's.** Same function, same key [1, 1, 1], same n = A1.eval_n = 4000, and
  `spur = rng.random(n) < 1.0` is all true. Code unchanged since the A2 runs; the histogram matches (finding 2).
- **The swap recorder.** `spurious_texts` looks up `_set_window_count` as a module global at call time, so patching
  the `pcfg` attribute intercepts it. It is the only caller in src (amend2 calls it directly, but not in this run).
  The wrapper records `count(texts, op)` as a copy (`.astype`) before the in-place edit, and passes the same
  arguments, so the rng stream is unchanged. Rows rejected by spurious_texts' y filter never reach the wrapper, so
  `pre` lines up row for row (the length assert passes). swap = |a after − a before| is exactly the number of swaps,
  because each row only lowers or only raises.
- **The swap-size read.** After the swap, the counts within one b are fixed (a = b − 1, c = 41 − 2b). So any answer
  that depends only on the counts is constant within b and gets slope 0, and the within-b demeaning removes the b
  confound fully. The HC0 formula is right for a slope with fixed effects. A nearly perfect model gives sd ≈ 0, and
  the 0.05 floor then keeps a few stray misses from reading 'depends'; checked against P5's M3 (3 misses). No bin
  divides by zero, and void arises only when sxx = 0. Within-b sd of |swap| is 0.84 (total 1.31), so the slope has
  spread to work with. Synthetic positive and negative controls passed in the dry run.
- **The pair sampler is unbiased within a pair.** The stream is iid natural texts and each pair keeps the first
  rows in stream order, so a pair's rows are a fair draw from natural texts with those counts. It is deterministic
  given the seed. The dry run's rows per pair match the filing (11-200). Rows are disjoint from amend1 and amend2
  (control). The new seed tail 74 does not collide with 70-73.
- **The rules.** third = a + 1 + floor((d + 1) / 3) equals round(d / 3) for every residue, negatives included
  (numpy `//` floors). The filed per-pair answers for the new pairs are right, and third is distinct in each.
  `third_on_new` cut-offs: ceil(0.75 m) and floor(0.25 m) never overlap for m = 3-6. Empty or small pairs fail every
  test (nan, or n < 10) and give void.
- **Controls run first, and a miss stops the run.** Model-free controls come before any model load. Model controls
  (LOW and N1 on the pooled reference pairs, about 1,000 rows, and on the 760 unswapped te1 rows) come before Part 1
  and Part 2. A miss writes `amend3_control_miss.json` and exits 1, and the sbatch passes rc 1 through (pipefail,
  not 3, so no resubmit).
- **Answer tags.** No two reads write the same `answers/` file. Saved as int16, so non-number answers (< 0) survive.
- **Lint and tests.** `make lint FILES=experiments/pcfg_anchors/amend3.py` is clean (ruff and pyright).
  `make test`: 410 passed, 2 failed, both in tests/test_qwen_maps.py and unrelated to this diff.

## Not run
- No model forward pass (none allowed).
- N1's count of b at b = 9 on natural text, which decides finding 1, was not measured. It needs a GPU pass, and that
  pass is Part 1 itself.
