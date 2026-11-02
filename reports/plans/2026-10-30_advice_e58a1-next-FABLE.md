# Fable's advice: what E58 Amendment 1 establishes, and the re-gating line's next step

*2026-10-30, Fable, at the lead's request. Advice only; nothing filed or run. Words as in `GOALS.md` → Terms. Read beside
`reports/plans/2026-10-30_plan_e58-open-pair.md` § Amendment 1, `experiments/open_pair/results/amend1.json` (job 46332)
and `reports/plans/2026-10-30_advice_e58-reading-FABLE.md`.*

## In plain words

We gave the held-out-key label its missing positive control and looked at what the E58 fine-tunes did at the keys they
never saw: every fine-tune pushed the words it was trained on and pushed the model's own preferences down, so each is a
lookup, and only the copy map carried to unseen keys, about half way. The label therefore grades how far a rule carries,
while the decomposition's two readings sit at the same value on every map whose words the model already says and drop
only when the words are new, so the two measure different things and the filed agreement test cannot be scored between
them. I recommend accepting that as the line's claim, retiring the held-out label as step 2's target in favour of a
piece-level label the decomposition did not make, and one $0 check that could still overturn it.

## What Amendment 1 establishes

- **A1 removes the "guaranteed by construction" worry.** At held-out keys the trained values rise (T 1.6-1.8 elicit,
  5.5-5.7 cipher) while W₀'s own top-20 fall (R_peer −0.4 to −1.9 on all twelve states). No fine-tune sharpened W₀'s
  preference; E58's "new" is a true reading of the update.
- **The label is graded, not binary, on this recipe.** Held-out carry x orders the four families: copy 0.42-0.55 >
  rank-1 0.09-0.14 > elicit −0.03 > cipher −0.3, with seed spread s 0.014-0.02, far below the gaps. Only the copy map
  touches the 0.5 cut, and it straddles it (C2 fails as filed; it stands). The cause is the denominator: G_spec is read
  at step 10, where the trained keys already sit at P 0.96-0.98, so G_spec is 17-18 log-odds and "reuses" needs the
  held-out keys to gain about 9 log-odds (P 1e-4 → 0.4). Copy nearly did, through the tied embeddings; the rank-1 map,
  the simplest rule there is ("ignore the marker"), carried 2.5 log-odds at the held-out keys against 16.8 at the
  trained ones. Under E54's recipe with 24 keys the 67M memorises every rule but identity. *Lesson:* the 75% lean on C2
  was filed without computing the denominator from the E58 states already on disk (elicit G_spec 18); compute a ratio
  threshold's denominator at the scored step before filing.
- **r and the ratio have no range where x has its range.** Over the nine states whose values W₀ already produces, x
  spans 0.59 (−0.04 to 0.55), r spans 0.09 (0.76-0.85, seed noise 0.015) and the ratio 0.18 (0.93-1.11). The only drop is
  the cipher (r 0.23, ratio 0.32), whose values W₀ does not produce. S does not track x either (copy 0.61-0.68 >
  elicit 0.49 > rank-1 0.21-0.25 > cipher 0.15-0.22: elicit and rank-1 swapped). This is the strongest evidence yet for
  reading A: r reads whether the carrying pieces are old and is blind to whether the rule carries.

**What it does to step 2.** A binary agreement between r and the held-out label is answered: they agree at the two
extremes (new tokens → both "new"; copy → both near "reuses") and disagree in the middle by what each measures. Do
**not** file the graded test (rank correlation of x with r across families): r is flat on three of four families, so a
Spearman over 12 states is decided by the three cipher states alone and over 4 families has n = 4; it would pass
degenerately, the case the predictions rule warns about. Report the spread table above instead, unscored.

## Ranked recommendation

1. **Accept reading A and write the claim** ($0, today): "On the 67M, r and the ratio read which subcomponents a
   fine-tune builds from, not whether what it learned generalises: x spans 0.59 across nine old-piece states while r
   spans 0.09; they drop only when the values are tokens W₀ does not produce." Evidence E46, E53, E54, E58, E60, A1; the
   granularity limit stated; a 2×2 (old/new pieces × rule carries/not) with the (new pieces, rule carries) cell marked
   untested. Amendment 1 analysis by `/writeup` (C2 fail with cause, C1 not testable: 18 distinct values, 1 novel key),
   `TRIED.md` row, `PENDING.md` line.
2. **Re-aim step 2, by dated note on the ground-truth plan**: the held-out label is kept as a graded, reported column
   and retired as the agreement target; the VPD-free label for reading A is piece-level, Stage B's interchange patch
   (`reports/plans/2026-10-29_plan_reuse-stage1.md` § Stage B, P ≥ 0.5). Hand the reuse line the twelve open-pair
   states as Stage B's state set beside lowercase; do not build the patch in this line. Step 2 is re-aimed, not stopped.
3. **One $0 check that can overturn item 1** (run it, FYI; a `debug` job of about Amendment 1's size): the three maps
   (elicit, rank-1, copy) at four times the keys, e.g. 96 trained / 32 held-out, seed 0, E54's recipe, x and r on each.
   Lookups win over rules at 24 keys; with more keys a rule should start to carry while r, under reading A, does not
   move. Expected: elicit x within ±0.05 of 0 (no rule exists, so more keys give no carry; this is the control); rank-1
   x ≥ 0.3 if rules become learnable with keys, ≤ 0.15 if the recipe memorises regardless; copy x ≥ 0.6; r 0.75-0.85 and
   ratio 0.9-1.1 on all three. Guard: check first that `kv`/`keydep` can supply 128 word-like keys with enough Pile
   windows each (E54 built 32); if not, 64/16 with the same expected values. Denominator computed before filing.
4. No Qwen gate opens on any of this; Qwen stays a plan on paper.

**What would change it.** (a) Rank-1 reaches x ≥ 0.5 with r unchanged: the first fairly labelled "reuses" state; score
step 2's P2 on it and keep the held-out label for rule tasks given enough keys. (b) r or the ratio rises with x by more
than 0.05 between elicit and copy/rank-1 at 128 keys: the piece reading tracks generalisation after all, and item 1's
"blind to" sentence is wrong. (c) The user answers ask #14 with B: E58 and Amendment 1 then count against VPD, the claim
becomes "no VPD reading tracks generalisation on the 67M", and the line's next question is whether any reading can.

## The user

No new question. Append one sentence to the open ask #14: *"Amendment 1 sharpens it: across nine fine-tunes whose words
the model already says, how far the rule carried ranged from none to half, while r stayed at 0.76-0.85; we write the
line's claim under reading A unless you choose B."*

**Not run.** Nothing; advice only. Item 3's key supply is unchecked.
