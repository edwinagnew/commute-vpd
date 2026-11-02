# Review: E54 Amendment 1 analysis (independent check)

*2026-10-30, fresh agent that wrote neither the code nor the analysis. Scope: the "Amendment 1" section and the edited
ratio table of `reports/analyses/2026-10-30_analysis_e54-keydependent-elicit.md`; notebook cells 12, 14-17 of
`notebooks/E54_keydependent_elicit.ipynb`; `experiments/keydependent_elicit/figures/a1_units.png` and `plot_amend1.py`;
against `results/amend1.json`, `stage2.json`, `predictions_amend1.json`, `src/commute/elicit.py` (`separation`,
`score_amend1`, `branch_amend1`), `src/commute/steering.py` (`steer_at`, `logodds_at`), the plan's "Amendment 1" and
`reports/reviews/2026-10-30_review_e54-amend1-code.md`. Commit checked: c2bac58.*

**Verdict: sound, minor fixes.** Every number in the new text matches `amend1.json`, each prediction is scored by its
filed rule, and P2 stays failed as filed. The one thing that outruns the evidence is the precision given to the
ratio of 10.3: it is quoted as a measurement in the A1 opening, the main ratio table and `PENDING.md`, while its
denominator (steering's gap, 0.052) is barely above its own seed range (0.043), so the ratio is bounded below at about
6 and is not bounded above. The rest are wording and figure items.

## Findings, by severity

### 1. (medium) The 10.3x ratio is reported with more precision than two seeds give it

Where: A1 opening ("now measured at about ten times"); main section ratio table, last row "10.3x" and the sentence
under it ("measured it at 10.3"); "Does the headline change?" ("now measured at 10.3"); notebook cell 15 opening and
cell 12 ("at about tenfold"); `docs/research_log/PENDING.md` line 10 ("10.3 times steering's, measured").

The A1 "Reading it" paragraph does say the ratio is noisy and gives one side: steering's gap one seed range wider gives
0.5385 / (0.0524 + 0.0434) = 5.6, "about 6". The other side is not given: one range narrower gives
0.5385 / (0.0524 − 0.0434) = 60, and the steering gap's sign is not even fixed by two seeds beyond the 0.009 margin.
So the data support "at least about 6", not "10.3". The caveat lives in one paragraph; the three headline places
and the log line carry the bare 10.3. The direction (r's gap is several times steering's in one unit) is robust; the
number is not.

Fix: in the ratio table write "10.3x (≥ ~6 within one seed range; unbounded above)" or "~6x or more"; in the A1
opening, headline check and notebook, say "at least about six times, 10.3 at the seed means"; make the A1 noise
sentence two-sided. Carry the same wording into the `PENDING.md` line before the log is flushed.

### 2. (low) The estimate the measurement is compared with is named inconsistently

Where: A1 "What changed from the estimate" ("The estimate put steering's log-odds gap near 0.04"); notebook cell 15
("So a gap near 0.05") against cell 17 ("a little larger than the estimated 0.04").

Two estimates existed: the old ratio table's ~0.04 (commit a2242de) and the filed expectation 0.05
(`predictions_amend1.json` → P1-lo.expected.gap). The measured 0.052 matches the filed one. The notebook states both
without saying which is which, so it reads as self-contradictory.

Fix: "The filed expectation was 0.05 and the measured gap is 0.052; the analysis's earlier table had guessed ~0.04,
which gave 'over tenfold'."

### 3. (low) A per-key attribution is stated as observed, but no per-key breakdown was run

Where: A1 "What changed from the estimate": "On the few keys where the steered probability is not small, the log-odds
rise exceeds the nats rise."

The aggregate is measured: on the elicit map the steered log-odds rise is 2.66 and 2.75 (`states.elicit_s*.rise_lo`)
against a nats rise of 2.54 and 2.62 (S × the nats scoring gain). Which keys carry the difference is the plan's
expectation, and "a per-key breakdown of the steering share in log-odds" is in A1's own "Not run".

Fix: give the aggregate numbers, and mark the "few keys" part as the plan's expectation, unmeasured.

### 4. (low) The figure cannot answer its title by the filed rule

Where: `figures/a1_units.png`, `plot_amend1.py`. The title asks "does steering still fail to separate the pair?" and
each column is annotated only with its gap. The filed rule needs two more things: the larger seed range (0.029,
0.040, 0.043) and the minimum gap of 0.2. A reader of the figure alone sees "gap 0.05" and cannot tell that it exceeds
the seed range and fails only on the minimum. The cap of 0.3, which the "unit explains the cap miss" paragraph reads,
is also not drawn. The caption itself ("the re-gating share and the steering share in nats and in log-odds, for the
elicit map and the matched cipher") is accurate, and the 0 and 1 references are on the same axes.

Fix: annotate each column "gap 0.05, range 0.04, minimum 0.2", and draw a dashed line at 0.3 across the two steering
columns labelled "P2's cap (filed for nats)".

### 5. (nit) The cap is read in a unit it was not filed for

Where: A1 "The unit explains the earlier cap miss"; notebook cell 17. "In log-odds it is 0.25 on both seeds, under the
cap." The cap of 0.3 was filed for S in nats. The paragraph keeps P2 failed and says the reading is after the fact,
which is the right handling; it would be clearer to say the comparison is with a threshold set for the other unit.
The arithmetic does support "the unit explains it": the elicit rise grows only from 2.54 to 2.66 between units, while
the denominator grows from 6.68 to 10.81 nats-to-log-odds on the scoring half.

### 6. (nit) Style and completeness

- `check.py` on the report: no retired terms; 18 long-sentence flags, of which about seven are in A1 (26-40 words,
  e.g. the opening's second and third sentences, the cross-check paragraph). Notebook: seven flags, three in cells
  15 and 17.
- A1's "Not run" omits "more than two seeds". The main list has it, but finding 1 rests on exactly that, so it is
  worth repeating there.
- Notebook code cells, including the A1 figure cell 16, carry no outputs, so the figure does not show on a static
  render. This matches the notebook's other figure cells; not A1-specific.

## Checked, fine

- **Every number against `amend1.json` (jq).** S_lo 0.246, 0.252, 0.219, 0.175, 0.075, 0.045; S 0.380, 0.394, 0.228,
  0.185, 0.114, 0.066; r 0.773, 0.758, 0.241, 0.212, 0.265, 0.243; r − S_lo 0.526, 0.506, 0.023, 0.037, 0.190, 0.198;
  P1-lo gap 0.0524, range 0.0434; P3-own gap 0.2963, range 0.0478; r gap 0.5385; ratio 0.5385 / 0.0524 = 10.27;
  "about a fifth" of the cipher (mean 0.197); cipher S moved under 0.01 (0.0091, 0.0098); log-odds gain table and
  all six relative differences (3.7e-4, 2.3e-4, 4.4e-4, 4.7e-4, 1.4e-4, 2.1e-4). r and S equal `stage2.json`'s.
  Stage 0: all six `dS_b3` and `dS_path` exactly 0, sha256 recorded, `stage0_pass` true.
- **Scoring by the filed rule.** `score_amend1` calls `separation` with min_gap 0.2 (strict range, ≥ 0.2) for P1-lo and
  P3-own and r − S_lo ≥ 0.3 on both elicit states for P2-lo, exactly as `predictions_amend1.json` writes them. Statuses
  pass/pass/pass and the branch string in the report are the run's. Every value fell in its filed band, as stated.
  `predictions_amend1.json` sha256 4d0475b4… matches `amend1.json`, and its only commit (29bc903) predates the run
  commit 66c2c72.
- **P2 not re-read.** The main scorecard still says "fails"; A1 says "The filed prediction stays failed as scored"; the
  `PENDING.md` line says "P2 stays failed as filed". P3-own is labelled "pass, not a test", matching its filed
  `not_a_test`.
- **Code-review findings.** Findings 1 and 2 are reported with their measured outcomes (cross-check 1.4e-4 to 4.7e-4,
  missing the filed "within 1e-4" expectation, which the scorecard says plainly; S difference exactly 0). Findings 3
  and 4 are in the code at 66c2c72: `branch_amend1` names a negative r − S_lo as reversed, and `amend1.json` records
  `git_head` and `git_dirty` (only `qwen_sft/results/run.log` dirty, outside the code).
- **S_lo measures what the text says.** `steer_at` reads W0 and the tuned model on all 97 windows, slices the scoring
  half, adds B3's vector at the answer position on that half, and divides the mean rise by the mean gain;
  `logodds_at` wraps `identity.log_odds`, the route split's unit.
- **The figure's data.** `plot_amend1.py` reads `amend1.json` only; points and the three gaps (0.54, 0.18, 0.05) match.
- **"Blind to" paragraph.** States that r − S_lo compares shares of different things (target model on 48 windows vs
  CI-masked re-gating term on 97) and is a table number, not a mechanism.
- **Departures from the literature, A1.** Covers the steering recipe (unchanged departure from contrastive activation
  addition), the log-odds share as this project's choice, and the thresholds as ours, with the 1e-6 tolerance's
  measured source. Nothing missing.
- **Not run, A1.** Block chosen in log-odds, more positions or a multiplier, per key, a matched random update, the
  own-step S_lo unscored: all accurate. Main "Not run" items 6-7 correctly point to A1.
- **Headline.** "Under twofold to over tenfold" in "What we learned" still contains 10.3, and the A1 section's "does the
  headline change? No" is right as to direction.
- **Terms.** No retired term; "re-gating share", "route split", "least-squares split" used as in `GOALS.md`.
