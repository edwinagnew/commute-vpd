# E62, stage (b) stop: advice (Fable, 2026-11-02)

*Advice only, on job 48507 (`cluster/logs/commute-e62-induction-cue-48507.out`, `experiments/induction_cue/results/run.json`).
Nothing run, filed or committed. Plan `reports/plans/2026-11-02_plan_e62-copying-heads.md`; rules
`experiments/induction_cue/results/predictions.json`; code `src/commute/copying.py` (`cue_windows`, `build_cue`, `pick_words`,
`r_verdict`, `score`), `experiments/induction_cue/run.py` (`stage_b`).*

**What the numbers say.** Stage (a) is clean: one head, (2,4), prefix score 0.42 against a bar of 0.077, and ablating it alone takes
the repeated-sequence top-1 from 0.825 to 0.0095. Stage (b) read W₀'s prior for B at the marker as 4.3e-4 with the pair intact,
8.5e-5 with the pair broken and 6.2e-6 with B absent: a 14× lift for B being anywhere in the window and a further 5× for the pair,
so the copying route already fires weakly at the marker. The stop came from the lookup's candidate pool, not from B: the 128
candidates are ranks 1000-2999 tokens absent from every window, so their prior is near the 6e-6 context-free value, a factor ~70
under the target. E58 saw the same wall: its elicit-prior cipher (band 3e-4 to 3e-3) was skipped because 46 of 128 keys had no
admissible value in either rank-≥1000 pool (`experiments/open_pair/results/amend7.json`, `tables.ep_skip`). The lead's reading holds.

## 1. Still worth running; no bar moves

Yes. "Answer already given" means the prior is near the top of W₀'s guess; 4.3e-4 is 0.04% of the mass, far from top-1. The 5× lean
with the pair intact is the mechanism being redirected, present but weak; it does not collapse the cell, it sharpens it to "old
mechanism, answer weakly present" (say so in the write-up). The filed rules already cover this reading: `prior_band` returns
`over_band`, which runs on unflagged; `prior_high` is 1e-3 to 1e-2, a flag not a requirement; 1e-2 is the stop. Neither moves. The
only thing that stopped the job is the candidate pool, below. What would change this: a prior over 1e-3 on the final read windows
(it is measured on them, so it will not drift).

## 2. The match: draw the lookup's words from frequent tokens, per key, excluded from the key's own windows

- **Which tokens.** All word-like tokens of `KD.vocab`'s `ranked` list (any frequency rank), minus the keys, both B pools and the
  marker; `pick_words` keeps its rule: a key takes the first unused candidate (seeded order, `kv.bijection`) whose mean log P over
  the key's read windows is within log 2 of log 4.3e-4. Frequent tokens' priors are nearly context-free, so per key is cheap and
  stays the filed rule; the map's geometric mean then matches too. Drop `n_word_pool`; `word_logp` on ~2-5k columns is one forward.
- **Windows unchanged for both maps.** Stop filtering windows against the candidate pool (`cue_windows`' `avoid`; a frequent pool
  would drop most windows). Instead add to `ok`: the candidate occurs in none of the key's train or read windows (count over
  `cue.train` and `cue.read` where the token at `half - 1` is the key). Both maps keep the same 1000 windows, and the lookup's own
  word is never in context for its key, so the copying head cannot produce it; other keys' words in a window are not that
  window's target and do no harm. A token at 4e-4 appears in a 41-token window about 1.6% of the time, ~31 windows per key, so
  about 60% of in-band candidates survive the exclusion per key.
- **Words in context so the lookup could copy: no.** The lookup is the "new mechanism" arm; let it copy and the label's
  `lookup share ≤ 0.2` loses its meaning.
- **Enough tokens?** Not knowable without the model. A Zipf estimate puts 300-500 vocabulary tokens with prior in [2.1e-4, 8.6e-4]
  if the marker's distribution is near unigram; 32 distinct are needed. E58's data say the band is empty above rank 1000 for a
  third of keys, which is why ranks under 1000 must be allowed. **Measurement, same $0 job:** before the draw, print per key the
  count of admissible candidates and the pool's log-prior histogram, so a repeat stop explains itself.
- **Amendment** (dated 2026-11-02, before any fine-tune; nothing downstream read): `settings.windows.n_word_pool` and `b_match`'s
  "from the 128-token word pool" become the rule above; `prior_match_max` 2 stays.

## 3. Verdict table at matched priors near 4e-4

E58 at step 30: copy 5.9e-5 → r 0.77, elicit 1.3e-3 → 0.73, rank-1 1.3e-2 → 0.85. So the prior alone predicts **copy r ≈ 0.75**,
and the test's whole weight is on the lookup, for which E58 has no point at this prior (the skipped cipher). Labels assumed held.

| Copy r | Lookup r | Reading | Expected |
|---|---|---|---|
| ≥ 0.5 all seeds | < 0.5, gap ≥ 0.24 | r follows mechanism (filed) | if mechanism: ~0.75 vs 0.3-0.5 |
| ≥ 0.5 | ≥ 0.5, gap < 0.24 | **r follows the prior** (now "untested"; amend) | if prior: ~0.75 vs ~0.7 |
| < 0.5 | < 0.5, gap < 0.24 | r follows the prior (filed); against E58's 0.75, so also "E58 relation fails here" | unlikely |
| < 0.5 | ≥ 0.5, gap ≤ −0.24 | r reverses the label: a surprise, announce it (now "untested"; name it) | unlikely |
| anything else | | untested, `near_cut` where within 0.045 | |

The both-high row is the amendment that matters: `r_verdict`'s docstring defines "follows the prior" as "r does not read this
mechanism at this prior", and both-high is that outcome; the filed rule only wrote both-low because the filed band put the prior
at 1e-5 where E58 says r is low. The mechanism row is untouched, so this is not a bar moved to rescue the run. Dated 2026-11-02.

## 4. Stage (a): keep the label on H alone

- |H| = 1 and the k-curve (0.0095, 0.0017, 0 …) say one head does the copying. Fine.
- Head (1,1) has previous-token score 0.67 (next highest 0.35): a previous-token head, and the one draw at 0.013 is it. Do **not**
  add it to H: at the marker both maps need A's identity moved one position forward, so the lookup would lose its route too and
  the label would read "old" for both. Report it as the circuit's other half, as the plan's "why the median" paragraph foresaw.
- Random floor: median 0.79 over 0.7, one draw of five is a circuit member. With |H| = 1 the stage (d) floor is single-head
  shares; the median rule absorbs the (1,1) draw. No change.
- One $0 print worth adding: the prefix scores of all heads under each fine-tuned state. If the copy map recruits (2,3) or (2,2)
  (0.05 and 0.04 under the bar) the share under H falls and reads `copy_not_elicitation`; the print tells recruitment from a wrong
  label. A diagnostic, not a rule.

## Not run
Nothing; all numbers from job 48507 and E58's files. The candidate count per key (section 2) is the one unknown.
