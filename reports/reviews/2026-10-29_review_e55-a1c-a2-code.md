# E55 a1c/a2 code review (commit 0ee393c)

Adversarial review of `src/commute/interchange.py`, `probe.py`, `experiments/pcfg_anchors/run.py` (a1c, a2),
`pcfg.py` (matched spurious text), `pcfg_train.py` (`answer_loss`), `cluster/pcfg_anchors.sbatch`, against
`results/predictions.json` (E1-E11) and Jain et al. L1086-1116, L1186-1196, L1238-1248. Reviewer wrote none of it.
Checked by: reading; `pytest tests/test_interchange.py tests/test_pcfg.py` (39 pass, cpu-run); a CPU check on 20K
natural and 20K spurious texts (`~/commute-data/scratch/reuse-lead/review-a2/chk.py`). No checkpoint loaded.

## Bugs

1. **a2 writes final files with "unfiled" labels, and the skip makes them permanent.** `run.py:388-403`. If a2
   runs before the bars are in `predictions.json` (the sbatch usage lists a2 right after a1c), every
   `a2_<arm>.json` is written with `label: "unfiled"`. Once the bars are filed, a rerun logs "exists; skipped"
   (`:393`) and no path ever applies them. The fix "delete the JSON" retrains the arm, because `<arm>_train.pt` is
   gone and the read is not split from the fine-tune. Fix: a2 refuses to start while `filed_bars()` is None
   (exit 2). Or skip only when `bars_filed` is true, and re-read from the saved `<arm>_<it>.pt` states when they
   exist instead of fine-tuning again.

## Risks (ranked)

2. **l* can land on the last block, where P is only a behaviour readout.** `interchange.py:153-159`, filed rule
   `definitions.l_star`. Patching the residual at `=` after block n_layer sets the output to the donor's
   (`tests/test_interchange.py:69` asserts this). So P(6) = share(out_d != out_x) on every model. For an accurate
   positive that is ~1, so the max over blocks is trivially at block 6. l* is then the earliest block within 3 s
   (~0.03 at p ~0.9) of that trivial value. If no earlier block carries the count at >= ~97% fidelity, l* = 6. P
   then measures "the output depends on f_C", the same thing B measures, and the "two checks" collapse into one.
   Fix, before the A2 bars are filed (dated amendment): stop (apparatus) when l* == n_layer, or take the
   candidates as blocks 1..n_layer-1. Read `P_by_block` in a1c.json first.
3. **Zero nulls make lo ~0.003, so N states read "neither" over a handful of errors.** `bar()` with E2's floor.
   NEVER answers count_b (HIGH-like count_b = 1.0 expected), and on qualifying inputs count_b != count_a + 1 by
   construction, so B_a null = 0 exactly and lo_B_a = 0.0030. P_a null is likewise ~0. The counts sit in 9-17, so
   a +-1 answer error lands on count_a + 1 often. An N1 1K state with ~1% wrong answers then gets B_a ~0.003-0.005
   and reads "neither": a PA3 miss for a trivial reason, not a reuse finding. Fix: decide before filing the A2
   bars (dated). Either keep the rule and say so in the bars filing, or floor lo at an error rate measured on the
   null (e.g. at least the off-by-one rate of N1s1 vs N1). At the least, log B_a's numerator per state.
4. **Spurious rows can be told apart from natural rows by local patterns.** `pcfg.py:164-199`. Measured: the
   histogram matches and count(b) = count(a) + 1 holds exactly (good). But the a->c swaps create `ccc` trigrams
   (0.95% of window positions; never in natural text), `cc` rises 0.050 -> 0.079, and mean count(c) rises by 1.
   Mixed arms (N2, N3, M2, M3) can therefore learn a row-type switch. On spurious rows the label also equals
   (41 - count(c)) / 2, a second shortcut: a W that uses it reads "neither" on qualifying natural inputs, not
   "through count-a". This is inherent to a 3-letter, 40-token window, and Jain's "adding b's" leaves similar
   marks. Fix: name both in the plan's blind_to and in the A2 report. Optional: report the share of W outputs
   equal to round((41 - count_c) / 2) on qualifying inputs.
5. **PA2's W cells miss by the floor.** `run.py:144-156`. A W that answers count_a + 1 scores
   P(count_b = count_a + 1) = 18.8% (measured) on natural C_Te = 0 rows. Table 6 has W1 0 and W2 1.9, so both
   cells sit more than 15 points off by construction, and PA2 can reach at most 7/9. Fix: add
   `count_b_floor_a_plus_1` to `scored` for op B (one line) and note it beside the PA2 reading. PA2 is reported,
   not a gate.
6. **a1c has no deadline, no resume and no timing; its cost is unmeasured.** `run.py:291-318`, sbatch
   `--time=00:30:00`. The probes run 3 seeds x (3 + 2 models) x 4K fp32, uncompiled forwards of 96 x 257: 60K
   forwards. The patches add 3 sets x 4 refs x 8 runs of 1K sequences. The sbatch header says "est. 10 min, not
   measured". A kill at 30 min loses everything, because a1c.json is written only at the end. Fix: log
   per-stage wall time, write the patch and B-null part before the probes, or move the probes to their own
   invocation.
7. **Saved fine-tune states are keyed by arm name only.** `run.py:360-382`. A recipe change (lr, c_tr, seed,
   loss) after a partial run silently resumes from the old `<arm>_train.pt` or reuses `<arm>_1000.pt`. Fix: save
   `asdict(arm)` and the FT constants in each state, and assert they match on load.

## Nits

8. `l_star` and `bar` use n = 1000 for the binomial floor, while P may rest on 500-999 pairs
   (`interchange.py:153, 165`). Pass `len(pairs.x)`.
9. `filed_bars` `run.py:327`: `filed.get(k, a1c[k])` passes when predictions.json omits l*. Require the key.
10. B (fp32, qualifying 'C b 40', g = count_a + 1) is scaled against p_plus = LOW's bf16 all-rows 'C a 40'
    accuracy (`run.py:298`). This is filed, so it stays. Note it in the report: it is a different quantity.
11. Two a2 jobs that list the same arm both write `<arm>_train.pt`. Keep arm lists disjoint (sbatch header).
12. The `generation_rate` comment "equal since spurious matching" (`run.py:183`) is true only in distribution.
    a0.json's 14.68 predates the matching. Re-run a0 or drop the stale field.

## Verified as filed

- **Patch.** It writes `out[:, -1]` (= `=`, index 256) after 1-based block l, and later blocks read it. The donor
  has the same other count and a different f. The output is compared to the model's own donor output and must
  differ from its output on x. The denominator is all pairs; void below 500. Hooks are removed in `finally`.
  Eval mode and fp32. x and donor come from one natural pool on a stream ([1,30,i,op], [1,50,op]) disjoint from
  training ([0,c], [100+s,c]) and from the probes ([200+s,c], held out [1,40,s,op]).
- **B, l* and the label.** B uses g = count_a + 1 / count_b and the qualifying filter. The B_b null follows E1.
  l* is the earliest block within 3 s per block, with E8 floored. Labels are strict with void/nan -> neither
  (E5-E7).
- **Probe.** Recipe as L1238-1248, read at `=`. Held-out accuracy is scored against true labels, so the shuffled
  probe scores the label marginal. The majority rate is reported. Margins, the E3 floor, m_plus void (E4) and
  possession are inclusive, as filed. NEVER's probe is a control for count-a only.
- **Fine-tune.** The answer-only CE is taken at `=` against Ans. 10K iterations, batch 96, 20% warm-up, cosine to
  lr/100. The C_Tr Bernoulli mix is per row. C_Te = 0 and 1 use separate fixed keys. A resume across the 1K save
  replays the same batches and RNG.
- **Measured.** P(count_a = count_b | qualifying) = 0.26, so B_b null (LOW answers count_a) is expected at
  ~0.26. Qualifying share is 81%.

## Not run

Nothing on a GPU. No real checkpoint was loaded, so l*, the nulls and the a1c run time are not measured. The
probe loop ran only at 6 iterations on tiny random models (existing test).
