# Review of the E65 plan: can the harder counting test separate methods, and against what truth?

Fable, 2026-11-02. Adversarial review of `reports/plans/2026-11-02_plan_e65-harder-counting.md` at c620e72, written by
another agent; nothing here is written into the plan. Read: the merge handoff, `decisions.md` rows 112 and 114, the
C1b analysis, the C2 decision, the method-table analysis, `src/commute/interchange.py`, `src/commute/revival.py`,
`experiments/pcfg_anchors/results/a2_N1.json` and `a1c_probe.json`, and Jain et al.'s text. No model was run. The
complaint held against the plan: a test every method passes tells you nothing.

## Recommendation in one paragraph

Go, with the changes in findings 1 to 4, and only if the user wants what this test can give: a ranking of the cheap
methods against the first truth in the project that ties a fine-tune to W₀'s own mechanism. It cannot rank VPD above
a reader of answers, and the plan says so correctly. If the user's question this week is the merged line's, VPD
against cheaper methods, skip to Stage B on the 67M pairs and come back to this later. What would change the
recommendation: if the saved bases and C1b states cannot be found (finding 5) the cost rises by about an hour and
Stage 0 loses its basis; if the base-to-state patch reads under 0.5 on C1b's existing reuse states, the design is
dead before any new base is trained.

## Findings, ranked

### 1. The three pass rules together require 8 of 8, so the power section is wrong by a factor of two

The plan's pass needs the cell bar (7 of 8), one cell above the best hindsight labeller, and p at most 0.05 over the
70 balanced relabellings. A method that scores 7 of 8 has labelled five cells one way and three the other. Of the 70
balanced relabellings, the ones agreeing with it on 7 or more cells are those whose four "reuse" cells sit inside its
five: C(5, 4) = 5, so p = 5 / 70 = 0.071, over 0.05. Only 8 of 8 passes the relabelling test (p = 1 / 70 = 0.014). So
one void seed in the wrong place, or one tied cell, fails a method outright.

The plan's power numbers assume 7 of 8. At 8 of 8 with three scored seeds voting per cell: a method wrong on a fifth
of states gets a cell right 0.896 of the time and passes 0.896^8 = 42%, not 81%; wrong on 30%, 0.784^8 = 14%, not 46%.
"Methods differ, the likely outcome" is then mostly "every method fails".

**Fix.** Either drop the relabelling test (with a balanced truth the binomial bar already is a permutation test) or,
better, buy the second table up front: 12 cells give a bar of 10 of 12 (tail 0.019), the relabelling p at 10 of 12 is
37 / 924 = 0.040, the rules cohere, and the 20%-error method passes 88% of the time. Four more cells are 20 more runs,
about 1 to 1.5 h of debug jobs, which finding 4 pays for.

### 2. The truth and steering read the same object, and both halves of the truth are nearly structural

`T` puts W₀'s block-5 residual into the state. Steering adds the mean block-5 residual difference between state and W₀
back into W₀; in C1b block 5 was steering's best block on every state. Both measure whether the state's block-5
residual at `=` is W₀'s up to a near-constant change. A steering pass here is therefore partly the truth agreeing with
itself; pruning, reverse fine-tuning, update size and `U` are independent of `T`, steering is not.

Also, on other-letter cells `T` is low because W₀'s residual holds little of the other count (probe accuracy 0.64 on
LOW, 0.50 on NEVER), not because anything was measured about the fine-tune: it is the lack premise read back. On
own-letter cells `T` is high as long as blocks 1 to 5 have not moved. So the truth is base crossed with letter unless
a table cell's blocks 1 to 5 move, and the plan's blind spot (a rotated residual reads "new") is exactly the case it
would call news.

**Fix.** Add a second causal reading that shares nothing with steering: zero the K units W₀ needs most for its own
count (the importance-pruning control already finds them) inside the fine-tuned state, and read the state's task
accuracy. Label a cell only where the patch and the knock-out agree; report disagreements as "patch fell" or
"units carried", never as "new". This costs one short job and is the same K search the route-blocked branch wants.
Mark steering's cell in the table as sharing its readout with `T`.

### 3. Reading each run at its first learned save makes `T` erode by wrapper, not by mechanism

`T` falls with training (C1b's in-state patch, 0.91 to 0.76 from 1K to 10K). The +1 cells learn by 1K; the table
cells are expected at 2K to 10K. Reading each at its first learned save reads the table cells later, so their `T`
sits lower for a reason that has nothing to do with reuse. The plan's own guess, table `T` 0.4 to 0.9, straddles a
`hi` bar near 0.45 to 0.5 (hi = n0 + 0.5 (p+ − n0) with p+ from own +1). Truth would then erode in the table cells by
construction of the schedule, and the plan would report it as the truth differing from the table.

**Fix.** Read `T` at one fixed step for every cell (10K, by when every scored cell must have learned, or the earliest
step at which both wrappers have learned on that base and letter), and set p+ from a state that is reuse by
construction and not a scored class (the identity-prompt fine-tune, or W₀ itself). Keep the first-learned-save reading
as a reported column.

### 4. Cuts

- **The "both have" cell.** Its task is the token at distance 40, so "pairs that differ in the task letter's count"
  is undefined and the plan does not say how `T` is read there. The methods are expected to call it "new" while the
  table calls it "reuses". Cut it: four runs and four scorings.
- **The frozen-encoder lack check.** Its bars (0.7, 0.9 at 2,000 steps) have no source, the probes already give the
  lack as a number, and `T` on the other-letter cells reads the same premise. Keep it as one reported number or drop
  it; do not let it stop the run.
- **The 70-relabelling test**, unless 12 cells are bought (finding 1).
- **The route-blocked branch.** Do not ask for it now. It is a built path, its 40% lean is a guess, and its K search
  is folded into finding 2. Present it as a decision card after Stage 3 with Stage 3's numbers in hand.
- **Three stage-3 arms are references, not candidates** (`U`, the answer probe, the shuffled fine-tune). Keep them;
  they are cheap and they are the point.

With the "both" cell cut and a third wrapper added, the state count goes from 52 to 68 and Stage 2 to 3 from
3 to 5.5 h to about 4 to 7 h, all on `debug`.

### 5. The plan's Stage 0 rests on saved models that cannot be found

Every clone's `experiments/pcfg_anchors/results/large` is a symlink to
`/mnt/nw/home/e.agnew/commute-data/results-large/pcfg_anchors/`, which is empty. A search of `~/commute-data` and
`~/commute-agents` to depth 9 found no `pt_LOW.pt`, `pt_NEVER.pt` or `c1b/` state. The reuse lead's clone is gone;
`~/commute-data/stale/reuse-lead/` holds one JSON. Job 48413 read all 43 states on 2026-11-02, so they existed then,
possibly only on a compute node's `/ephemeral`.

**Fix.** Before anything else, name the path of LOW, NEVER and the C1b states in the plan, with sizes. If they are
gone: LOW and NEVER are 50K steps at 12 ms, about 10 min each, and the C1b calibration states another 20 min; add
that to Stage 0 and say that the "within 0.01" reproduction checks then compare against the saved JSON numbers only.

### 6. The pilot is right in shape; one reading is missing from it

Stage 0 then Stage 1 with stops is the right first step. But the one number the whole design rests on, the
base-to-state patch on a state known to reuse, is read in Stage 0 only with B as both base and state, which is Stage
A's in-state patch again (NEVER on its own count, saved at 1.0). The cross-model reading first appears in Stage 2's
calibration seeds. Read `T` from LOW into C1b's existing P states (and N states, expecting 0) in Stage 0, with the
plan's own stop at 0.5. If those states are gone (finding 5), train the two calibration seeds of own +1 on LOW before
base A and read `T` there. Also move the Stage 2 stops ("`T` at own +1 under 0.5", "`hi` not above `lo`") to before
the scored seeds train, not before the methods read.

### 7. The plan's finding about Stage A is right, and it reaches further than the plan says

In Stage A's patch, the pairs share the other count and differ in the count being patched, and P counts a pair only
when the donor's output differs from the text's own. A state answering count b gives the same output on both texts of
a count-a pair, so P reads 0 whatever the model does: `a2_N1.json`'s `check_count_a` has `P_by_block` all zero and
`B` 0, label "not through". So "new" on the counting set rested on behaviour (the state does not answer count a + 1)
plus an in-state count-b patch.

The reuse half is no better anchored to W₀. Stage A's "through" patch moves a residual from one text to another
*inside the fine-tuned state*: it shows the state's answer runs through a count-a representation at block 5, not that
this representation is W₀'s. `T` in this plan is the first reading in the project that patches from W₀ into the
state, and the plan should say so plainly; it is the test's real contribution.

**Consequences.** No label moves. The method table's row "Counting: through count a vs not" should be marked
"construction plus an in-state patch; new half by behaviour", not "by causal check", and its C1b row's "each state also
passed behaviour and patch checks" should say the same. `decisions.md` row 114's "counting-model labels are set by
construction" stands. One PENDING line for the log; the merged line's claims change in kind, not in number, since
nothing passes on that row anyway.

### 8. Smaller points

- "A cell's truth is the majority of its states": say whether that is all five seeds or the three scored ones, and what
  a 1-1 truth tie after a dropped state does (the table says a tie is wrong for a method vote; for the truth it should
  make the cell unscored).
- The sign test for "A beats B" at five discordant cells of eight is sound (p = 1 / 32) but, with 8 of 8 required to
  pass, a five-cell gap between two passing methods cannot occur; it is a comparison between a passing and a failing
  method only. At 12 cells it becomes usable.
- The 25% to 45% leans for the three methods assume the 7-of-8 bar; re-lean after finding 1.
- The "no room" stop reads hindsight labellers on update size, steps and W₀'s first-step loss. Add the answer probe's
  hindsight threshold: if it reaches 8 of 8, as the plan expects, say in the report that no method here beat a reader
  of W₀'s residual, whatever else passes.

## Answers to the five questions

1. **Can it separate methods, and is the truth independent?** It can separate cheap methods from each other, not from
   `U` or the answer probe. The truth is independent of pruning, reverse fine-tuning and size; it shares its object
   with steering (finding 2). The "exactly 4 of 8" holds for readers of one factor; any reader of base crossed with
   letter, which is anything that compares the state's task to W₀'s native count, recovers the truth, as the plan admits.
2. **Stage A's patch reads 0 by construction on new-count states.** Yes (finding 7), and the reuse half is an
   in-state patch, not a W₀ one. Labels do not move; their kind in the table does.
3. **Power.** The rules are internally inconsistent at 8 cells (finding 1). At 12 cells they cohere.
4. **Smallest plan.** No. Cut the "both" cell, the frozen-encoder stop and the route-blocked ask; add the fixed-step
   read, the knock-out check and the third wrapper. Pilot of base A with its lack check, after Stage 0 reads `T` on an
   existing reuse state, is the right first step with a stop.
5. **VPD cannot win here.** Right, as long as the truth equals base crossed with letter. The one exception, an
   own-letter table cell whose residual rotates, is where `U` would be wrong and a mechanism reader right, and a single
   patch cannot tell that from its own blind spot; the knock-out check (finding 2) is what would. On this model a new
   mechanism for an old answer does not arise naturally; it has to be built (the route-blocked branch) or found where
   answers do not fix the label, which is the 67M pairs under Stage B's causal label. So this test serves the table's
   cheap-method rows and the W₀-anchored truth; it does not serve the merged line's VPD question. Stage B does.

## Options

- **A. Go with findings 1 to 6 applied.** About 4 to 7 h of debug jobs, $0. Gives the first W₀-anchored causal truth and
  a ranking of three cheap methods on it, with references that show what they read. Says nothing about VPD.
- **B. Stage B first, E65 later.** The interchange patch on the 67M pairs, where update size already separates the one
  causal row and answers do not fix the label. Serves the merged line's question directly; E65's finding 7 still goes
  into the table now.
- **C. As written.** Not recommended: the pass rules make 8 of 8 the real bar, the table cells' truth erodes by
  schedule, and Stage 0's inputs are unlocated.

Recommendation: A if the user wants the cheap methods ranked on a causal truth this week; B if the question is VPD.
Either way, apply finding 7 to the method table today.

## Not run

No model, no job. The 70-relabelling and 924-relabelling counts and the power figures are hand arithmetic from
`revival.pass_bar`'s rule; the claim that the saved states are missing is a filesystem search to depth 9 and may miss
a path on a compute node.
