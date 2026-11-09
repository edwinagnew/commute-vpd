# Can the second wording of reuse be tested? A first test on pairs whose label is known

*2026-11-09. Proposal only: nothing here has run and no verdict is given under either wording. Apparatus and the full
list of departures: `2026-11-09_plan_reuse-variable-first-test-appendix.md`.*

**In plain words.** The test takes the 67M model and adds one small trained or hand-set piece to it, so we know exactly
how the new behaviour is produced: a fixed push on one of the model's own variables, a new wire from something the
model already computes, or a new calculation the model cannot read linearly. We then ask whether a search that is not
told the answer recovers it. Most of what the two wordings say about these pairs follows on paper, so the run mainly
tests the search, which is what you doubted, and whether "built" can occur at all.

## Design check

- **Goal served.** Part 1 of the paper plan: a definition that gives a verdict.
- **Nearest tried.** E70 and E55 used labels fixed by the task and scored probes or removals on them; E112 and E113
  found that a named read-out did not carry a fine-tune. Here the label comes from where the change is allowed to live,
  and the read-out is searched for rather than named.
- **Patch or step.** A step: the first test of a wording adopted today.
- **Decisions relied on.** Today's `decisions.md` row. Nothing is read from saved runs. P1.2 waits.
- **Departs from the cell text.** The cell asks for three fine-tunes with shuffled-label twins. Your task of today asks
  for pairs by label, so there are no twins here.

## Settings, fixed here before any model is examined

| Setting | Value | Why |
|---|---|---|
| Input layout | 8-token windows from the Pile; the output is the next-token distribution at the last position | One layout for every pair; short enough to fit the affine maps exactly |
| Distance d | Total variation between the two next-token distributions | The reversion formula divides one direction's distance by the other's, so d must be symmetric; it is bounded and reads as "share of probability moved" |
| k | 4 | Must be far below the 32 answers of the table pair, or a table reads as re-weighted (soft spot 1). G is also drawn at k = 1 and 16 |
| E | 2,048 Pile windows from rows no pair trains on | |
| D | 200,000 windows per pair, half for fitting and half for reading. Each holds both sides of the behaviour and a quarter from far inputs | Size set by caveat 1, checked in the run |
| Floor on the gap | Mean distance 0.02 | Below it, no reading of that pair |
| Cut-offs | None in the first run. For the second, the rule under "Run cells" | Your caveat 3 |
| z_M | Residual stream at every position, every block up to φ's | The wording names no positions and no kind of activation; this is my reading of it |

## The pairs

M′ is the model plus one extra piece that starts by doing nothing, so M′ computes exactly what M computes. M′ and F then
share an architecture, as the definition requires. Only the piece differs. All writes are at the input of block 2, last
position. This is the "low-rank adapter on a hidden vector" family that Geiger et al. 2025 list as fine-tuning methods.

Three pairs share one formula: **the value in a 4-dimensional subspace becomes v_M(x) + f(x)·b**, where the subspace
and b come from the first pair and only f differs.

| Pair | How the behaviour is produced | True label | G | H | First wording | Second wording |
|---|---|---|---|---|---|---|
| Always on | f = 1. Subspace and b trained so that the model continues lowercased text | Re-weighted | 1 at the witness | 1 at the witness | Reuse | Re-weighted |
| Cue | f is a frozen linear read-out of block 1's residual, fitted to detect one rare token | Rewired | Below H | 1 at the witness | Reuse, by its precedence rule | Rewired |
| XOR | f is the XOR of a random bit per token over the last 4 tokens, computed by the added piece | Built, if caveat 1 allows | Near 0 | Near the shuffled level | Reuse: **this is its fault (a)** | Built, feeding that subspace |
| Table | A full-rank linear map from block 0's residual, trained on 32 keys to arbitrary answers | Rewired | Well below H | 1 at the witness | Rewired | Rewired |

Why these should read that way: "the witness" is the known subspace and map. Putting them into the formula gives the
listed value exactly, by algebra. Everything else is what a blind search should find.

**Caveat 1, on built.** The XOR is the only pair that could be built, and only if no affine map from M's activations
can read it. Four tokens from the whole vocabulary give far more combinations than the activations have directions, so
built can occur. The run measures the rank of z_M over D, and fits an affine map to the real XOR and to shuffled XOR
labels. If the map reads the real XOR above 0.55 on held-out windows, the pair is rewired and I report that built did
not occur. A hand-made bit says nothing about whether a model pretrained without a skill reads built, so that claim is
not tested here.

**Caveat 2, on the search.** A found map is a lower bound. Not finding one is never read as built.
- For G: gradient search over the subspace and the affine map at each of four sites, three starts each.
- For H: an affine map fitted by ridge regression from z_M, then refined. At the output the fit is a convex problem, so
  there the best map is found up to convergence. Since the output always splits, a pair can be built only if this one
  fit fails. That is a necessary condition; failing it at the other sites can only be checked for the maps we find.
- **What shows the search can succeed:** Always on, Cue and Table have witnesses reading exactly 1. The search does not
  see them. It must reach 0.9 of G on Always on and of H on Cue and Table. If it does not, its low numbers mean nothing
  and no pair is labelled from them.
- **What shows it cannot be fooled:** on the XOR pair and on shuffled labels, H must stay low.

**Caveat 3.** Your repairs to D and the formulas are unreviewed. D is built to contain both sides, and each formula's
expected value is above, so a broken one shows as a pair that misses its algebra.

## What the first wording reads, from the same runs

Only its exact checks are read, at the witness, to a tolerance set by the difference between two identical runs.

**(a) A new computation written into a consumed variable.** On the XOR pair, F and M agree once the subspace is set, so
the first wording says reuse and the second says built. The truth by construction is built, so the second wording
matches. The Cue pair also differs (reuse against rewired), but that is a choice made in the wording's precedence rule.
Construction cannot settle whether a new wire into an old unit should count as reuse.

**(b) Can its "built" ever apply?** On E, 100 random directions at each site are swapped between inputs, and the output
change is compared with the identical-run floor. I expect every direction inside the span of the activations to count
as acted on. Its "built" could then apply only outside that span. The same number bears on the second wording: see the
first fault below.

**Where the first wording stops.** Its checks are equalities. They can hold on these pairs, which are exact, and cannot
hold on an ordinary fine-tune. There it gives no call at all.

## Three things found in the wording, unchanged here

1. **"M acts on φ" may guard nothing.** It is the test that failed as the first wording's "built", moved into
   "re-weighted". If (b) comes back as expected, every subspace passes, and only k and the affine map separate
   re-weighted from rewired.
2. **G and H read only F's outputs on D.** Two fine-tunes with the same outputs get the same shares, whatever is inside
   them. Only the reversion and the setter mismatch look inside, and the label needs those. This is stated in the
   wording; the test makes it plain.
3. **The credit is wrong about Geiger et al.** See below.

## Departures from the papers

The full list, with quotes, is in the appendix. No claim about the three papers is flatly wrong except one line.

- **Geiger et al. 2025.** The sentence "stating a fine-tune as an intervention on its base is not in any paper" is
  **wrong as written**: section 3.7 says adapter-tuning "can be viewed as interventions on model parameters". They do
  not apply it to a base and a fine-tune, so the first wording's narrower sentence stands. "In the sense of Geiger" adds
  no restriction, because any same-architecture fine-tune is already an interventional on all variables. A subspace is a
  variable only after a change of basis. An s that leaves M's own range is steering in their terms, which they say will
  "fail to uncover how the network reasons"; only the selection grade avoids this. They say nothing about what a failed
  search shows, or about the illusion problem. Causal model and intervention are credited by them to earlier work.
- **Jain et al. 2024 Def 1.** Only the form is shared. Theirs reads a named function, on a domain of one capability,
  with a probe trained on pretraining data, from one layer and position. The second wording reads any value, on a mixed
  set, needs only that some map exists, and uses every earlier layer. Caveat 1 comes from those last two changes.
- **Prakash et al. 2024.** The first split condition is their direction (fine-tune into base, same input). They patch
  named attention heads, score task accuracy on the task's own prompts, and never do the second condition.

## What would make us drop it

- **Drop the built label, or change what "computed" means:** the affine map reads shuffled XOR labels as well as real
  ones on every D we can afford, so anything is readable.
- **Drop G as the separator:** the search passes its controls, yet the G found on Cue, XOR or Table is as high as on
  Always on at k = 4.
- **Drop the graded numbers and keep only named witnesses, which would confirm your doubt about the maps:** the search
  misses 0.9 on pairs whose answer is exactly 1, after three starts at every site.
- **Second run:** if all behaviours read the same G and H, the definition has no reading on real fine-tunes.

## Run cells proposed

Both are for you to add. Neither has been started.

| Cell | What | Runs | Cost and time |
|---|---|---|---|
| R1 | The four pairs, the search, the capacity check and both first-wording reads. The identity check runs first and alone, and stops the job if it fails | 1 | Under an hour on one GPU, not yet measured; $0 on the cluster's debug queue in two jobs, or about $1 on RunPod |
| R2, only after R1 passes | Ordinary fine-tunes of all weights, trained to imitate each of the four pairs, 3 seeds | 1 | A few hours at most; the XOR may not be learnable, and the job stops if it has not learned within its first minutes |

**What R2 adds.** If a fine-tune matches its pair's outputs, G and H cannot change, so R2 is really about the reversion
and the setter mismatch, which are the only numbers that see the fine-tune's own route. Its cut-offs come from R1,
filed before R2 starts: the midpoint between Always on's found G and the highest found G on the other three, and the
same for H between the lowest of Always on, Cue and Table and the XOR. If either gap is empty, R2 gives no labels.

**Not proposed yet:** a model pretrained without a skill. The candidates in the wording's list have not been read
beyond abstracts, and none is set up here.

## Not run

Everything above. No model was loaded, no fine-tune made and no search run; every expected value is by algebra or
judgement. The 0.9 and 0.55 bars and the floor are judgement, filed before any reading. R1's time is an estimate.
