# E38 Amendment 4: independent check of the change-matched restore (review)

*2026-10-28, independent agent, read-only; no model loaded, numbers recomputed with `jq`. Files:
`experiments/recruitment_test/results/{amend4,amend3}.json`, `run_amend4.log`, `predictions_amend4.json`,
`amend4.py`, `amend3.py`, `run.py`, `src/commute/{recruitment,recruit,regating}.py`, the plan
`reports/plans/2026-10-28_plan_e38-amend4.md`.*

I recomputed every filed verdict of the change-matched restore from its results file and read the code that chose the
comparison sets and restored them. Every verdict agrees with the run's own, the code does what the filed definitions
say, and the switched-on set beats both change-matched sets on emoticon and ` which` by far more than the bar. The
result rules out "the switched-on set matters only because its outputs moved most", but not "it matters because its
outputs moved in the direction the target reads", and nothing yet shows Γ is needed to find such subcomponents.

## 1. Verdicts

Symbols: ρ(X) is the share of the gain taken back when the trigger outputs of set X are put back to their
original-model values, on held-out windows; 0 means nothing taken back, 1 the whole gain. S is the set Γ switched on,
C_n the same number of the largest movers outside S per layer, C_m the largest movers until their summed change
matches S's per layer.

| state | ρ(S) | ρ(C_n) | ρ(C_m) | S − C_n | S − C_m | seed class | run's class |
|---|---|---|---|---|---|---|---|
| emoticon s0 | 0.239 | −0.149 | −0.245 | 0.388 | 0.484 | specific | specific |
| emoticon s1 | 0.220 | −0.128 | −0.216 | 0.348 | 0.436 | specific | specific |
| ` which` s0 | 0.179 | 0.017 | −0.024 | 0.162 | 0.204 | specific | specific |
| ` which` s1 | 0.202 | 0.036 | −0.044 | 0.166 | 0.246 | specific | specific |
| ` we` s0 | 0.037 | −0.010 | −0.005 | 0.047 | 0.042 | unclear | unclear |
| ` we` s1 | 0.049 | 0.004 | −0.007 | 0.045 | 0.055 | unclear | unclear |

Every gain is above 1 nat, so every state is scorable. ` we` has ρ(S) below 0.10 on both seeds, so it is "S too
small". Emoticon and ` which` are "specific" on both seeds. With two scored tasks both "specific", the reading is
"specific", as filed and as the run wrote.

The second filed reading, on emoticon only:

| seed | most negative single in layer-3 active pool | most negative random single | class | run's class |
|---|---|---|---|---|
| 0 | −0.135 | −0.002 | few | few |
| 1 | −0.115 | −0.002 | few | few |

Its reading is "few" on both seeds, which agrees.

The "reported" expectations:

| expectation | recomputed | met | agrees with run |
|---|---|---|---|
| ρ(S) equals Amendment 3's to 1e-4 | identical to the last digit, checked against `amend3.json` directly | yes | yes |
| summed change of C_n over S below 1, all six | 3.27 to 4.34 | **no** | yes |
| S's share of the top-\|S\| movers per layer at least 0.5 | 0.05 to 0.11 | **no** | yes |
| ρ(all alive) between 0.3 and 0.9 | 0.375, 0.380, 0.461, 0.489 | yes | yes |
| emoticon layer-3 pool at most −0.15, rest within 0.05 of 0 | −0.225, −0.219; rest −0.034, −0.010 | yes | yes |
| C_m short in any layer | none | — | yes |

**The two missed expectations matter more than the hits.** The plan assumed S holds the subcomponents whose trigger
outputs changed most. It does not. Only 5 to 11 percent of S sit among the top movers of their layer, and S's summed
change is about a quarter of C_n's. So C_m reaches S's summed change with 6 to 28 subcomponents against S's 21 to
213. The question "is S special or did it just move most" is answered before the restore is read: S did not move
most.

## 2. Code against the filed definitions

Read in `recruitment.py` lines 445-572 and `amend4.py`; the code is unchanged since the run's commit e64beee.

- **Change of one subcomponent.** `output_change` takes each matrix's trigger input in two passes, multiplies the
  difference by V, takes the absolute value, averages over windows and scales by the norm of each row of U. That is
  the filed mean of the absolute input change along the read vector times the write vector's norm. U holds one row per subcomponent
  and V one column, the same orientation the restore hook uses.
- **Training windows.** Both passes use `t.train.tokens` at `t.half`. S comes from `shift_of`, which reads the same
  training windows. All shares are scored on `t.held.tokens`, as in Amendment 3.
- **Ranking and ties.** `movers` excludes S and dead subcomponents and sorts by descending change, then matrix name,
  then index, as filed.
- **C_n.** The first |S in that layer| of each layer's ranking; short flagged when the pool is smaller.
- **C_m.** The shortest prefix of each layer's ranking whose summed change reaches S's; empty where S's sum is 0. A
  single large mover can overshoot S's layer sum: on ` we` s0 layer 1 one subcomponent carries 4.50 against S's 0.09.
  That favours C_m, so it does not flatter S.
- **Restores.** Unchanged from Amendment 3: the trigger output gains the sum over the set of the read-vector
  difference between the original-model input and this pass's input, times the write vector.
- **Layer-3 arms.** The pool is alive, outside S, with original-model trigger importance at least 0.2, in layer 3: 107
  subcomponents. "Rest" is Amendment 3's active draw 0 without layer 3. Emoticon used the 0.2 pool there too, so it
  is the same draw. Random singles come from layer 3's alive subcomponents below 0.2, seed 0.
- **Readings.** `p11_*` and `p12_*` implement the filed rules, including ties reading "not".

One weak control: R5 only checks that two identical original-model passes give zero change, which tests determinism.
No control checks that a large change makes a large restore. The code reading above stands in for it. Nothing found
that makes a share or a set wrong.

## 3. Alternatives to "S takes back more than change-matched sets"

**Ruled out by the data.**

- *C_m is far smaller than S in count.* C_n has S's count in every layer and about four times its summed change. It
  still takes back between −0.15 and 0.04 of the gain.
- *S simply moved most.* S's share of its layer's top movers is 0.05 to 0.11.
- *The emoticon margin is all overlap with the layer-3 active pool.* On ` which` there is no negative block: Amendment
  3's active draws read −0.07 to 0.05 there. Yet both change-matched sets sit at least 0.16 below S.

**Weakened but open.**

- *Overlap with the layer-3 pool inflates the emoticon margin.* C_m's layer-3 members are all in that pool on seed 0,
  9 of 9, and 7 of 10 on seed 1. C_m's share on emoticon, −0.245 and −0.216, equals the pool's own −0.225 and −0.219.
  The same four pool members read below −0.03 alone on both seeds, two in the MLP's first matrix `c_fc` and two in its second, `down_proj`. One of Amendment 3's random sets read −0.135 on seed 0, matching the strongest single, −0.135. C
  membership was not saved, so whether C_m holds those four is unknown. On emoticon, "specific" at the 0.05 bar needs
  C_m without the pool to stay below about 0.17; that is likely but not measured. The emoticon margins, 0.35 to 0.48,
  should not be quoted as how much S beats the top movers.

**Surviving.**

- *Direction, not size.* The change measure is unsigned and blind to which way the corrected output points. What
  separates S may be that its corrections line up with the target's readout while the big movers' corrections do
  not, or cancel each other. That is close to what "specific" means, but it leaves open whether Γ adds anything:
  a Γ-free ranking by each subcomponent's first-order effect on the target logit might find a set that takes back as
  much or more. This is the strongest surviving alternative to "Γ names the subcomponents that carry the change".
- *Mixed matrix types within a layer.* Matching is per layer, so the ranking compares output changes in different
  spaces: query and key outputs, the MLP's hidden layer, and residual writers. A large change in a query or key output
  restored at the trigger only moves the trigger's attention scores; a large hidden-layer change passes a nonlinearity.
  If C_n is mostly one such type, its small share is about units, not about S. The composition of C_n and C_m by
  matrix was not saved, so the data cannot rule this out. Logging it, or matching per matrix, costs about a minute of
  GPU on the cached states.
- *Switched-off subcomponents in C.* C may hold subcomponents whose importance fell under fine-tuning. Restoring them
  may push against the gain. This does not undercut "S beats the top movers", but C is then a mixed contrast, not a
  pure "moved but not switched on" set.

**Context for any claim.** Restoring every alive subcomponent takes back 0.38 on emoticon and about 0.46 to 0.49 on
` which`. S alone takes back about two fifths of that on ` which`. The rest of the gain sits in the fine-tune's own
weight change and the remainder Δ, which restore leaves in place.

## 4. What a "specific" claim in the research log may state

May state: *On the 67M model after a 10-step fine-tune, putting back the original trigger outputs of the subcomponents
Γ switched on takes back about a fifth of the gain on emoticon and ` which`. The same number of the subcomponents whose
trigger outputs changed most carry about four times the change. They take back about none on ` which` and less than
none on emoticon. So how much a subcomponent's output moved does not explain the switched-on set's share.*

May not state: that Γ is needed to find these subcomponents, since no direction-aware ranking without Γ was tried; that
the emoticon margin measures specificity; or anything about ` we`, where the switched-on set takes back about a
twentieth.

## Not run

- Any GPU computation: C_n's and C_m's membership and matrix composition, and C_m's share with the layer-3 pool
  removed, were not recomputed.
- A matching per matrix rather than per layer.
- A direction-aware ranking without Γ, such as each subcomponent's first-order effect on the target logit.
- Restoring S together with C_n or C_m.
- Positions other than the trigger, other tasks, seeds, or the larger model.
