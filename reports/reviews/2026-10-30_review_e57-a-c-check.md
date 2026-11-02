# Review: does E57 Stages A-C show what its report and notebook say?

*2026-10-30. Independent check by a fresh agent that wrote neither the code nor the analysis. Scope:
`reports/analyses/2026-10-30_analysis_e57-reading-the-update.md` (the report) and `notebooks/E57_qwen_reading.ipynb`
(the notebook), checked against the plan `reports/plans/2026-10-30_plan_q3-reading-the-update.md`, the results in
`experiments/qwen_reading/results/` and the code (`run.py`, `lab.py`, `boot_direct.py`, `plot.py`,
`src/commute/qwen_reading.py`). Nothing was re-run on a GPU. The derived numbers below come from the saved JSONs, read
with jq.*

## Verdict

The numbers and the scoring hold up. Every value I checked matches the JSONs, `predictions.json` has not changed since
its pre-run amendment, and the scorecard follows §11 correctly. The "no teeth" verdict for the re-gating share is
sound. Three conclusions go further than the data:

1. The direct share's gap between the updates and random changes is real on the row bootstrap. The report understates
   how much bigger gold's update is than the random changes. Its "maths KL" column mixes two row sets, and "three to
   four times" holds for gold only.
2. The cause given for the overlap rows (near-threshold subcomponents) is asserted, not measured. The evidence offered
   for it (web text reads highest) is an artefact of z-scores on reference sets of different sizes. On the per-member
   reading, anchor 2 stands apart on the prompt set, and the think and length-matched sets read the same.
3. The Stage D decision rule in "What next" would read a size difference as a difference of kind.

None of these changes a scored verdict, since nothing is scored. They do change what the direct share and the overlap
rows can be said to show.

## What was checked and holds

- **Predictions.** `predictions.json` was filed in be6f5bd (15:00) and amended in 4edc0fb (15:21:27). Its
  `amendment_2026-10-30` block was added and nothing was deleted. Stage A started at about 15:29 (A.json was written at
  15:49, after a 1,225 s job). `git diff 4edc0fb -- .../predictions.json` is empty. The file is untouched since filing.
- **Cell means, shares and intervals.** Report table at lines 94-101 against `C_*.json .maths.means/.regating/.boot95`
  and `C0.json`: all match. Direct shares and paired gaps at lines 127-147 match `boot_direct.json`. The behaviour-kept
  values at lines 248-255, recomputed from the cell means as (K00−K11)/(K00−K0) and (K00−K10)/(K00−K0), all match to 3
  decimals.
- **Sets.** The S counts and per-kind counts (lines 168-176) match `S`/`S_by_kind`. The split-half count 4,222, the
  floor 21,110 and |S_prompt| 1,860 match `A.json .sets`. The overlap excesses (lines 192-199), the seed pair (366 sd),
  the reference sizes, |T∩L| 7,237 (`B2.json`) and 20 of 128 traces dropped (`B1.json`) all match.
- **Placement and row 4.** The placement shares, null means, excesses and MLP shares (lines 217-222) match
  `C_*.json .placement`, and so does the positive control (0.2105 against an expected 0.4164). The kind shares and the
  median effective ranks (lines 236-242) match `A.json .row4` (median of 196 matrices).
- **Early checks.** Every `[check]` in `run.log` reads ok: A_W0, A_E40 (bit-identical to E44 Amendment 2's
  references), A_do_nothing, A_gate_counts, C0, C_E40, the C_E45 own-row KL checks and every C_g0. The GPU peak was
  42.57 GiB.
- **Scorecard against §11.** In E45's `score.json → amend2_decisive`, both anchor-1 seeds read `matched: false`, at
  maths KL 0.333 and 0.337. §11's last branch therefore holds: no anchor prediction is scored. The report's scorecard
  (lines 300-312) says this and does not treat anything as scored.
- **Bootstrap code** (`boot_direct.py`, `qwen_reading.resample/interval/shares`):
  - The resampling unit is the row: 128 rows drawn with replacement, 2,000 times, using `random.Random(0)`.
  - The shares are computed from the resampled sums, not as a mean of per-row ratios, which is correct for a ratio of
    sums.
  - The maths-row differences are paired: the same `draws` are used for both terms, and all six states share the same
    128 rows (K0 is identical at 0.069613 in every record).
  - The percentile indices are `v[50]` and `v[1949]` of 2,000 values.
  - I found no error.

## Issues, ranked

### 1. The size confound is understated: the KL column mixes row sets (material)

*Evidence.* Report lines 50-56 and 60-61 give "maths KL from the start" as E40 0.62, gold 0.70, anchor 2 0.61/0.63,
Gaussian 0.53, alive pieces 0.50. These come from two different measurements:

- E40 and the two random changes: 16 maths rows of 256 tokens (`C0.json .kl_e40_16`, `.kl_inspan_16`, `.search.kl`).
- Gold and anchor 2: E45's records (`qwen_updates/results/E_<run>.json .states["100"].kl.maths`). Per the plan's code
  departure 17, these are maths rows of 512 tokens.

Line 61 says "the others on E45's records", but E40's 0.62 is also the 16-row number. The anchor-2 seed order is also
reversed: seed 0 reads 0.629 and seed 1 reads 0.607.

The saved data hold a yardstick read on the same 128 rows for all six states: K00 − K0, the masked original model's
excess KL to the changed model. It is not KL to the start, but it is the same measurement on every state.

| state | K00 − K0 | direct share |
|---|---|---|
| E45 gold | 1.444 | 0.092 |
| E40 | 1.304 | 0.063 |
| Gaussian | 0.857 | 0.022 |
| anchor 2, s0 / s1 | 0.638 / 0.628 | 0.012 |
| alive pieces | 0.625 | 0.029 |

On this yardstick gold is 1.7 times the Gaussian change and 2.3 times the alive-pieces change. The report's column
suggests about 1.3 times. On the 128 rows E40 also sits outside the 1.5× window the Gaussian control was matched to on
16 rows (1.52 times). Among the four real updates the direct share rises with this size proxy. Anchor 2 is the one
real state matched in size to a random change (alive pieces), and it reads below that change.

*Fix.* Replace the column with one measurement on the same rows for every state, or label each value with its rows and
length. Fix the seed order. In "The caveat is size and domain" (line 124), give the size ratios above and say that,
among the real updates, the direct share follows size. Keep the dose check (line 295) as the precondition for any
reading of kind.

### 2. "Three to four times" holds for gold only (material, overstatement)

*Evidence.* Lines 10-11 and 263, and the notebook's "What we learned". Gold/Gaussian is 0.0917/0.0221 = 4.1 and
gold/alive pieces is 3.2. E40 reads 2.9 and 2.2. The answer paragraph applies the ratio to "the maths fine-tunes"
together.

*Fix.* "Two to four times (gold about 3-4, E40 about 2-3)". Or drop the ratio and give the two gaps.

### 3. The overlap rows: a cause asserted, and a z-score artefact read as evidence (material)

*Evidence.* Line 188-190 says "Switched-on subcomponents are the ones near Γ's threshold, and any new context switches
those on". Line 282 says "The cause is the null, shown by the web-text set reading highest for every state". The
notebook repeats both. No measurement of distance to threshold was made. The excess e = (o − mean)/sd grows with the
size of the reference set at a fixed enrichment. The reported ordering (web text, think, length-matched, prompt) is
exactly the size ordering 45,232 > 22,316 > 9,671 > 1,860. The fold enrichment o/mean from `summary.json .overlap`
tells a different story:

| state | prompt | think | length-matched | web text |
|---|---|---|---|---|
| E40 | 4.21 | 3.47 | 3.46 | 3.40 |
| E45 gold | 3.53 | 3.16 | 3.25 | 3.10 |
| anchor 2 s0 | **6.50** | 3.93 | 3.95 | 3.24 |
| anchor 2 s1 | **6.37** | 4.17 | 4.19 | 3.28 |
| Gaussian | 3.60 | 3.17 | 3.13 | 3.28 |
| alive pieces | 3.98 | 3.36 | 3.53 | 2.81 |

Three things follow:

- Web text is the *lowest* fold enrichment for almost every state, not the highest.
- Anchor 2 stands apart from every other state on the prompt set, on both seeds: 6.4-6.5 against 3.5-4.2. Line 185
  says "the random changes sit among the real updates on every reference". That is true of random against E40 and gold,
  but it hides that one real update separates. It is also the direction opposite to the filed (unscored) prompt_overlap
  expectation for anchor 2, "at or below q95".
- Think and length-matched read the same per member for every state (differences of 0.01-0.17). The spirit of §11's
  "T ≈ L" branch is met: the think reference reads maths context, not thinking. Line 203 cites only the |T∩L| clause.
  The e_T ≤ e_L clause cannot fire while |T| is more than twice |L|.

*Fix.*
- Report fold enrichment beside, or instead of, z-scores.
- Withdraw the near-threshold cause, or label it a hypothesis with its test: the g0 mask distribution of S against
  that of the draws.
- State anchor 2's prompt-set enrichment as an observation, with no prediction scored.
- Say that T and L are indistinguishable per member, so row 3 should not be used for D3 (§11).
- Fable's proposed null (random changes' S at matched size) addresses the same problem. Note that fold enrichment
  against those sets is the natural statistic for it.

### 4. The Stage D decision rule ignores size (material)

*Evidence.* Lines 291-294 say anchor 1's direct share "is the reading that matters", and that sitting with gold means
"a matched pair is worth designing" while sitting with anchor 2 or random means "close D3". The decision file
(`reports/decisions/2026-10-30_decision_e57-no-teeth.md`, 18:30 revision) records anchor 1 at maths KL 0.33-0.34 and
|δW| 3.14, against anchor 2 at 0.61-0.63 and |δW| 5.1. Issue 1 shows the direct share following size among the real
updates. An unmatched anchor-1 reading would confound kind with size in the same way. The report also omits the
decision file's sixth option: rerun anchor 2 with states saved between steps 12 and 100, to find a learned state near
anchor 1's size. That is the one option that removes this confound.

*Fix.* Make the dose check ($0, saved states) come before any Stage D kind reading of the direct share, and name option
6 in "What next".

### 5. The direct-share bootstrap covers only row noise, which is smaller than the spread between controls (moderate)

*Evidence.* The code is correct (see above). But the two random controls' intervals do not overlap: Gaussian
[0.0199, 0.0243] and alive pieces [0.0264, 0.0318], `boot_direct.json`. Their gap of 0.007 is about three times either
row half-width. Each control is one draw. The decision file's criterion ("gold and random inside one interval") is
therefore answered with intervals narrower than the variation between random changes. The conclusion "not met" still
holds, since the smallest gap, E40 − alive pieces at 0.034, is about 5 times 0.007. Line 272 names seeds and size, but
not the spread between controls.

A second point concerns lines 120-121 and 287 ("below both random changes, on both seeds and on both row sets"). These
compare anchor 2's own-row values (0.014-0.016) with the random changes' maths-row values. The random changes were not
read on own rows, so that comparison crosses row sets and is not paired.

*Fix.* Say the row bootstrap does not cover the draw of the random change, and give the 0.007 gap between the two
controls as a lower bound on that spread. Restrict "below random" to the maths rows.

### 6. Placement: the positive control is within the null's noise, and the quantity is mislabelled (moderate)

*Evidence.*
- Lines 212-213 call the positive control "just above the half at which the plan calls the reading blind". It reads
  0.2105 against a line of 0.2082, a margin of 0.0023. The four Gaussian null seeds spread by up to 0.0038
  (`C_e40.json .placement.null`). "Sees" is inside the null's own noise. The honest reading is "at the blind line".
- The heading (line 206) and the figure (`plot.py:140-141`, "subcomponents Γ switches on", "share on switched-on
  subcomponents") call the quantity the switched-on set. That set is defined at line 37 as gated on under the changed
  model and *not* the original. The quantity actually measured is the share on subcomponents gated on at the
  *original* model (`placement.gated_on`, 76-90k).
- E40's gated-on set comes from the maths rows and the others' from their own rows. This is noted at line 224. The
  null means differ as a result (0.58-0.60 against 0.49), so excesses are not on one reference across states.
- The random changes' placement is not computed and not listed as "Not run".

*Fix.* Relabel the figure and the heading "gated on at the original model". Read the positive control as at the blind
line. Add the random changes' placement to "Not run".

### 7. The notebook has no "Departures from the literature" list (rule)

*Evidence.* `.claude/skills/writeup/SKILL.md:100-103` requires one in every write-up. The notebook's markdown cells end
with Scorecard, Not run and Provenance. The report has one (lines 334-347).

*Fix.* Copy the report's list into the notebook. Consider adding one item to both: E40's no-warm-up recipe departs from
the standard warmed-up fine-tuning recipe, which E45 then used.

### 8. Missing "Not run" items (minor)

The following are not in the "Not run" list:

- the random changes' placement;
- a KL-to-start measurement on the same 128 rows for all six states;
- more than one draw per random control;
- the decision file's option 6 (anchor 2's intermediate states);
- a row bootstrap of the placement excess;
- a per-member (fold) reading of the overlap rows.

### 9. Smaller wording points (minor)

- Line 233-234 says "the gold updates' share rises steadily with depth". `A.json .row4.e40.layer` runs 0.033 → 0.038
  (layers 15-19) → 0.035 → 0.038 → 0.033, nearly flat. Anchor 2's real feature is the concentration in late layers
  (0.046-0.053 in layers 21-27), which the report leaves out.
- Line 92 says "0.005 to 0.006 wide". The values are 0.0052 and 0.0065.
- The behaviour-kept table (lines 244-255) sits under "Where does δW sit?". It belongs with the route split.
- Line 87 and the notebook: "a random change uses no existing machinery" (line 58) is an assumption for the
  alive-pieces change, which perturbs existing subcomponents. The no-teeth reading does not depend on it.

## Figures

- `regating_share.png` and `direct_share.png` carry their references (the 0.8 line, E54's 0.77 and 0.24, do-nothing 0,
  the random band). The direct-share figure has no size axis, although size is the main confound. A second panel of
  direct share against K00 − K0 (issue 1) would put the size reference on the same axes.
- `overlap_excess.png` plots z-scores only (issue 3).
- `placement.png` has the wrong label (issue 6), and its error bars show the null's spread around the excess, not
  around the null.

## Not checked

- The Stage B think traces themselves, and the L filler construction.
- The least-squares split and Σ code paths. Their stored errors are ≤ 5e-6, and they were reviewed before the run.
- Whether the notebook's executed outputs match the saved JSONs cell by cell. The code reads the same JSONs.
- `check.py` was run on both files. It flags only guideline items (sentence length, numbers per sentence) and no
  retired terms.
