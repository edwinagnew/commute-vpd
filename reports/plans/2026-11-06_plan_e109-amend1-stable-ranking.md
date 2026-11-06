# Plan: E109 Amendment 1, a carrying-neuron ranking that holds across seeds at the learned size

*2026-11-06, baseline-lead's plan writer. Plan only: no code, no predictions filed yet. Follows E109's analysis
(`reports/analyses/2026-11-06_analysis_e109-neurons-larger-size.md`) and plan
(`reports/plans/2026-11-06_plan_e109-neurons-larger-size.md`); code to be added beside `experiments/larger_size/run.py`
and `score.py`, which stay as they ran. This is E109's first of two amendments.*

## The request, in the Terms' words

The user's ask 24, option (b) (`decisions.md` line 170): "a stability amendment to E109: a neuron ranking or set size
whose seed overlap holds at the learned size, one ~10-min free GPU job". No VPD object is read, so none of `GOALS.md` →
Terms applies; in plain words: on the four saved five-epoch end states E109 already reads (true-answer and fictional
runs, two data-order seeds each), measure how well each of a few filed carrying-set instruments (two rankings times
three set sizes) agrees with itself across seeds; pick, by a rule filed now that reads only those states, the one that
clears E109's own power gate; then read the two saved wrong-answer end states with it and score E109's filed label rule,
unchanged. The do-nothing reference is W0 itself (score 0, no set).

## Design check (`/design`; approved: the user's ask 24 option (b), 2026-11-06)

- **Goal.** `GOALS.md`: does post-training teach new things or strengthen existing ones, and can any reading tell them
  apart? This keeps the non-VPD neuron baseline readable at the size where the wrong answers are learned.
- **Nearest tried.** E109 (`TRIED.md` line 219): with E98's first-order ranking and 1,024 neurons, the true run's seeds
  share 0.65 (TOFU rows) and 0.62 (answer tokens) against power lines of 0.757 and 0.746, so "no call". E107 at the
  matched size had the stability (0.90, 0.88) but not the learned label. Nothing has tried another ranking or size.
- **Patch or step.** A patch on the instrument, asked for by the user; the label test is E109's as filed.
- **Decisions and definitions.** Reads under `decisions.md` line 152 (the full update carries any "new" claim, size
  stated) and lines 160 and 170. Not a reuse verdict: rows 81 and 87 do not apply, and **no frozen definition is
  touched** (`.claude/CLAUDE.md` → "Adopted definitions are frozen"). The power gate, the learned gate and the wrong
  rule, with their bands and labels, are E109's unchanged. New: the candidate instruments, one selection rule, one
  reported completeness read. Departure from E109's plan, which said "K is not retuned in this E-code": made by the
  user's ask 24 (b), recorded here.
- **What would change what we do next.** See "What each outcome means".
- **Control first, alone.** Stage 0a re-reads KNOWN_end42 with E109's ranking and size on both scored probes and must
  reproduce its stored gains (within E109's C2n tolerance, 1e-3 relative) and sets (overlap at least 0.95), else stop.
  It fails if a state file, a probe or the ranking code differs from what E109 read.

## The one amendment

**Why these candidates.** E109's analysis names three: a set size, a gradient at the fine-tuned state or along the path,
more seeds. More seeds measure the gate's own spread but cannot raise a seed level, so they are not in the user's
option. Two observations point at the ranking: the first-order total had the wrong sign against the measured gain in
four reads, and a Gaussian change shares 0.27-0.33 of any set, so part of every set is fixed by where W0's gradient is
large rather than by the update. A path-integrated ranking addresses both; the project already runs one (E96/E97,
`commute.edges` "ig_gain": midpoints (k + 0.5)/8, 8 steps, summed to 1.003-1.005 of the gain on the 67M). Set size is
free once per-neuron scores are saved, so it rides along. Six candidates:

| ranking | what it is | sizes K |
|---|---|---|
| FO | E109's: W0's total gradient of the read log P with respect to each neuron's firing, times the firing change (`qwen_neurons.rank_state`) | 256, 1024, 4096 |
| IG8 | integrated gradients in firing space: firing at every block set to A0 + α(Ah − A0), W0 elsewhere, gradient at the 8 midpoints α = (k + 0.5)/8 averaged, times (Ah − A0). Its scores sum to E98's all-neuron patch `nums[N]` (`qwen_neurons.grid` with k = N) | 256, 1024, 4096 |

Sign conventions per probe are E109's: TOFU rows by the signed score's positive part, answer tokens by |score|.
FO at 1,024 is the filed instrument and is already known to fail; it stays in the grid as the control's reference.

**Selection rule (filed now; reads KNOWN_end and FICT_learned only).** Per scored probe, for each candidate: E109's
`score.references` and `power` unchanged at that candidate's K (w = the true run's seed overlap, x = the highest of the
four true-in-fictional overlaps, power iff w − x > 1 − w, unread with a set not of K neurons or a NaN). Select the
candidate with power and the largest margin w − x − (1 − w). Ties: FO before IG8, then K nearest 1,024, then the
smaller K. Each probe selects on its own, as E109 already ranks the two probes differently.

**Stage 1.** Read WRONG_end42 and WRONG_end142 with each probe's selected candidate only; score E109's `wrong_rule`
unchanged (w = the smaller of the two arms' seed overlaps, band 1 − w, the four wrong-in-true cross values; learned
gates from `arm_WRONG_end{42,142}.json`, both held). **No peeking:** the wrong-arm sets are not computed for any
candidate until the selection is written, and no wrong-arm number can change the selection: a selected candidate whose
wrong reads come out unread is reported "unread", never replaced by the next candidate.

**Edge cases of the rule.**
- No candidate clears on a probe: that probe is "no power at any filed ranking or size", E109's "no call" stands, and
  no wrong read is made for it. Neither probe: stage 1 does not run.
- Clears on one probe only: that probe is scored, the other reported as no power.
- Ties in margin (overlaps are multiples of 1/K): the order above decides; a tie is reported.
- A signed set shorter than K (fewer positive scores, possible at 4,096): that candidate is unread on that probe.
- Winner's curse: picking the largest of six margins inflates the true run's seed level. The guard is already in E109's
  rule: the wrong run's seed overlap, never seen by the selection, enters the band through the smaller of the two
  levels, so a lucky pick widens the band and leans to "no call". No rule is added for it.
- Do-nothing and noise: W0 gives empty sets, every candidate unread, nothing selected. Two random rankings give
  w and x near K/N (0.0015 to 0.024), margin near −1: no power.

## Numbers, with what each should read

| number | W0 / do-nothing | control or reference | expected |
|---|---|---|---|
| 0a: KNOWN_end42 FO-1024 gain, TOFU rows and answer tokens | 0 | stored 1.629, 1.0796 (`run.log` lines 20-21) | reproduced, set overlap 1.0 |
| FO-1024 w, x per probe | NaN | E109 `gate.json` | exactly 0.6523 / 0.5137 and 0.6201 / 0.4912 |
| w, x, margin for the other five candidates | NaN | — | FO at 256 or 4096: no power (lean; the Gaussian floor moves w and x together). IG8: uncertain; lean power on the TOFU rows at about 40%, on the answer tokens at about 30% |
| IG8 completeness: sum of scores against n × `nums[N]`, per state and probe (reported, not a stop) | unread | E97: 1.003-1.005 | within 5%; a miss is reported with the read, not used to exclude |
| Gaussian change's set inside KNOWN_end's, answer tokens, per candidate (reported) | — | E109 FO-1024: 0.331, 0.323 | IG8 lower than FO at the same K if it ranks the update rather than W0's gradient |
| selected candidate per probe | none | — | as above |
| stage 1: E109 `wrong_rule` verdict per selected probe | unread | E109's leans | TOFU rows topic or style; answer tokens no call likelier, familiarity the lean (E109's filed leans, unchanged) |

**Blind to.** Everything E109 is blind to (rank inside a set, neurons past the cut, anything inside a neuron), plus:
IG8 attributes `nums[N]`, the part of the update carried through MLP firing with W0 elsewhere, not the whole gain; if
that share is small, a stable IG8 set ranks a small part of the update. Two seeds per run: the seed level's spread is
not measured. The candidate list was written after E109's cross values were seen; the selection reads none of them.

**Saved.** Per-neuron scores for every state, probe and ranking, fp32 (about 0.7 MB each, about 20 MB in all) in
`results/large/amend1_scores/`; the candidate sets and the selection in a committed `results/amend1/` folder.

## Stages, runtime and cost

One job on one L40, `debug` QoS, $0; a new sbatch beside `cluster/larger_size.sbatch`, rules in a new
`results/predictions_amend1.json` filed before the run.

- **0a, control, first and alone:** KNOWN_end42, FO only, both probes; stop on a miss.
- **0b, selection states:** KNOWN_end42 (IG8), KNOWN_end142, FICT_learned42, FICT_learned142 (FO and IG8, both
  probes), GAUSS (answer tokens); then the selection, logged and written before stage 1.
- **1:** WRONG_end42, WRONG_end142 with the selected candidates; then the score.

Times from E109's `results/run.log` (job 54053, second start): line 18 setup 7.6 s; lines 20-21, an FO read of
KNOWN_end42 takes 20.4 s on the TOFU rows (14.6 → 35.0 s) and 6.3 s on the answer tokens (35.0 → 41.3 s); lines
30-31, FICT_learned42's load and real-row read take 25.7 s (107.8 → 133.5 s); line 45, the Gaussian change on the
answer tokens 12.4 s. Taking an FO read as one forward and backward at W0 plus one forward at the state (about four
forward units), one IG8 step is one forward and backward (three units): about 130 s per state on the TOFU rows and 40 s
on the answer tokens, plus one all-neuron patch forward each. Per state about 220 s with its load; stage 0 about 15 min,
stage 1 at most 7 min (1.5 min if only FO is selected), about 24 min in all. That is above the user's "~10 min" but
under the 30-min line and `debug`'s 2 h cap; dropping IG8 would make it about 6 min and leave only set size, which the
lean says is unlikely to help. Memory as E109's reads (peak 26.5 GB on the GPU, 11.0 GB host, `run.log` line 11).
The wrong end states are on disk (`experiments/larger_size/results/large/dW/WRONG_end{42,142}.safetensors`, 5.64 GB each).

## What each outcome means

- **A candidate clears on a probe and the rule reads familiarity or topic or style:** the neuron baseline's reading of
  the known-against-wrong pair at the learned size, labelled with the instrument chosen and the winner's-curse caveat;
  it goes into the comparison table (ask 24 (a)) as provisional, not into an answer paragraph without an independent check.
- **A candidate clears and the rule reads no call:** reported as read; no further amendment for this question.
- **Only IG8 clears:** the first-order approximation was the source of the instability at |δW| 8.3; later neuron reads
  at learned sizes use IG8.
- **Only a set size clears:** the cut, not the ranking, was the problem; reported with its K.
- **Nothing clears on either probe:** the neuron baseline is closed for this pair at the learned size, as E109's
  analysis says; option (c) of ask 24 (new invented facts) would need another instrument or the matched size only.
  The second amendment slot is kept, not spent on more seeds by default.

## Not run

- More seeds of any arm (would measure the seed level's spread; not the user's option (b)).
- A ranking with the gradient at the fine-tuned state alone (IG8 covers the path including that end).
- IG with more than 8 steps (the source paper's guidance is 20 to 300; the completeness read reports what 8 costs).
- The new candidates on the matched-size states (a positive control for IG8's stability where FO had power).
- The real-row probe and the reported reads of E109 other than the Gaussian floor.
- Any read of a fictional state at |δW| 8.3.

## Departures from the literature

- **Integrated gradients** (Sundararajan et al. 2017) are taken in neuron-firing space with every other weight at W0
  and the baseline W0's own firing, with 8 midpoints as E96/E97 rather than the paper's 20 to 300 steps.
- **E98's first-order ranking** and its departures carry over, as in E109. The selection rule, the candidate grid and
  the reuse of E109's gate across set sizes are this project's, not from a source.
