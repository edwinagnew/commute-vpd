# Review: `experiments/regating_spread/amend7.py` (E36 Amendment 7, plan Amendment 9)

Scope: commit `e1a2dbb`, the new file `experiments/regating_spread/amend7.py`, checked against
`reports/plans/2026-10-26_plan_regating-without-cues.md` → "Amendment 9". This was a fresh reviewer that did not write
the code. I changed no code.

**Verdict.** I found nothing that would crash on the GPU, and the bin rule does what its docstring says (tested
below). There is one material problem, and it sits in the spec as much as in the code. Equal-count bins probably spend
almost all of their resolution on values near zero. That makes the null weak in exactly the range the room-to-move
argument is about, so B1 could pass for the wrong reason. Two smaller problems affect scoring. All three are cheap to fix
before the predictions are filed.

## Findings, by severity

### 1. High (the measurement is not the one meant): equal-count bins are coarse over the active range

`amend7.py:117` (`at = ... * N // BINS - 1`), `amend7.py:121` (edges taken from rank quantiles of each column).

**Scenario.** Most alive subcomponents are active at only a small share of positions. Amendment 4's fitted set is about
10 GB, so there are about 9,500 alive subcomponents, while the task-ladder report counts 51 to 95 of them with a value
above 0.5 even at a trigger. So for a typical subcomponent, far fewer than 1 in 16 positions have a value much above
zero. The 15 inner edges are rank quantiles, so all of them, top edge included, then fall among near-zero values. Every
active value, from about 0.05 up to 1, lands in the top bin and gets one shared mean change. Inside that range the rule
is "x0 + constant, then clip". It cannot express any curve of change against value there, and that curve is the room
to move the amendment is meant to test: values near 1 can only fall, mid values can go either way. A null that is weak
where it matters makes B1 ("the bin rule at least 0.3 below the nearest rule") likely to pass whether or not the
original values' edge is only room to move. The "B1 holds" branch of the decision tree would then send the restatement
to the user on a test that could not have failed.

`bins_used` does not catch this. It can read 16 while all 16 bins sit below 0.05.

**Suggested fix (before filing).** Make the null as strong as the plan's wording, "a curved function of each value
alone":
- Add a second bin rule with 16 equal-width bins on [0, 1]: width 1/16, values at or below 0 in bin 0, values at 1 in
  the top bin, empty bins filled from below as now. `fit_columns` needs only fixed edges in place of the quantile
  gather. With fixed edges, "up to bin width" in C_bin is literally 1/16.
- Score B1 against the stronger of the two bin rules: `max(R_bin16, R_bin16w) <= R_near_g0 - 0.3`.
- Report one diagnostic: the median top inner edge e₁₅ over subcomponents (or the share of fitted values above 0.1
  that land in the top equal-count bin). That shows how coarse the equal-count rule was.

The plan text says "equal-count bins per subcomponent". So this change is an edit to Amendment 9's wording, made
before filing, not a departure from filed predictions.

### 2. Medium (mis-scoring): B1 is recorded as false when the control fails

`amend7.py:213`: `"B1": ok and s["R_bin16"] <= s["R_near_g0"] - BIN_GAP`.

**Scenario.** If C_bin fails at any step, `score.B1` is `false`. The decision tree has "B1 fails → the restatement is
withdrawn … the user gets a card". The plan and `PREDICTIONS["decision_tree"]["C_bin fails"]` both say B1 is then
unreadable, not failed. A reader of the JSON, or a runner that summarises `score`, would take the "fails" branch.
Amendment 6 has the same pattern (`amend6.py:195`). It did not matter there because its control held.

**Fix.** `"B1": (s["R_bin16"] <= s["R_near_g0"] - BIN_GAP) if ok else "unreadable"` (or `None`). Log it the same way.

### 3. Medium-low (positive control has no power at the scored step)

`amend7.py:202`: `C_bin = R_bin16 >= R_fix - 0.05`.

**Scenario.** The straight line recovered 0.39, 0.14 and −0.01 at steps 10, 100 and 300 (Amendment 1). At step 300,
C_bin passes for any bin rule that recovers at least −0.06. That includes a broken one that returns about zero
change, which would also pass B1, since B1's "pass" means the bin rule is weak. C_bin has teeth only at step 10. A bug
that showed only at later steps, such as a step-dependent mean, would go unseen. The code is correct today (see
"Checked" below), so this is about what the control can detect, not a live bug.

**Fix (cheap).** Add an in-sample check at every step: on the fitted rows, the bin rule's mean squared error of the
change is at most the line's plus a small tolerance. Compute it over the columns of `f["X"]` and `f["D"]` in
`fit_columns` while the chunk is in memory. The bin mean minimises squared error among functions of slope 1 within a
bin. The fitted slopes have median 0.94 to 0.96, and fewer than 1% are below 0.5 (`regating_spread*.json` "slopes"), so
the bin rule should match or beat the line. On a toy check (below) it did: 0.0206 against 0.0208. A related wording
note: the docstring and plan say the bin rule "can express any straight line up to bin width". Within a bin it has
slope 1, so the true bound is (1 − a) × bin width per subcomponent. That is small here given the slopes, but the
sentence should say so.

### 4. Low (environment, not the run)

- On the dev node, `.venv`'s editable install resolves `commute` to `/mnt/nw/home/e.agnew/commute/src` (the old tree).
  `from commute import cuefree` fails there unless `PYTHONPATH` points at this clone's `src/`. The pod builds its own
  environment, so the run is unaffected. This is the pending venv move.
- `ruff check` from the repo root is clean. `pyright` could not run on the worker (its nodeenv install failed), so the
  file is not type-checked.
- `transport_test/results/large/alive.pt` is gitignored. The runner must send it by the `results/large` directory sync
  rule in `CLAUDE.md`, as for Amendment 6.

## Checked and found correct

- **Bin edges, ties, empty bins.** `fit_columns` and `bin_change` (imported, run on CPU tensors through
  `fleet/cpu-run`) were compared with a brute-force rule written from the docstring. The test had 300 columns, which
  crosses the 256-column chunk boundary. It included a column 40% at 0 and 20% at 1, a constant column, a column with
  four levels, a nearly all-zero column, and held-out values equal to fitted values, inside the range, above the
  maximum, below the minimum, and exactly 0 and 1. The largest difference was 1e-7, and `bins_used` matched in every
  column. `searchsorted(right=False)` gives "edges strictly below", as documented. Bin 0 is never empty. Empty bins
  take the nearest non-empty bin below.
- **The line control.** `f["a"]`, `f["b"]` equal `cuefree.line(X, X + D)` exactly. That is Amendment 1's "fix"
  definition (`spread.line` is `cuefree.line`), refitted on the 262,144 positions.
- **Held-out against fitted.** Edges, bin means and the line are fitted on `train[:512]` only. Every R is read on
  `held`, as in Amendment 6.
- **The A0 reference.** `regating_spread_amend6.json` has status `done` and `R_near_g0` 0.890, 0.779 and 0.762 at
  "10", "100" and "300". The near rule, K = 8, `nearest`, the positions and the held-out set are all the same code as
  Amendment 6. `S.masked_logp` (a summed float) matches Amendment 6's `masked_logp_pos` summed over positions.
- **Dtypes and memory.** Values come from `to_flat` in float32, so the dtypes in `scatter_add_` and `searchsorted`
  agree. Peak memory is below Amendment 6's, which held two more 4,096- and 8,192-wide spaces. One chunk of
  `fit_columns` adds a few GB.
- **Clipping.** It is the same for every rule: add the change to the position's own original values, clip to [0, 1].
  `fix` clips `a·x0 + b`, as in Amendment 1.
- **The guards.** The run refuses to start without the predictions file. Every scored number is logged at each step.
  The JSON is written after each step and on an A0 stop.

## Not run

- `pyright` (the worker's node install failed).
- No GPU run, and no numbers on real Γ values: the share of values near zero per subcomponent is estimated from
  reports, not measured. Finding 1's size is therefore an argument; the suggested e₁₅ diagnostic would measure it.
