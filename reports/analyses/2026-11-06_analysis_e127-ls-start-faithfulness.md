# E127: why does the least-squares start read worse than do-nothing under the adversarial check?

*2026-11-06, regating-lead's writer. Plan `reports/plans/2026-11-06_plan_e127-ls-start-faithfulness.md`; filed rules
`experiments/ls_start_faith/results/predictions.json`; code `experiments/ls_start_faith/run.py`; result
`experiments/ls_start_faith/results/ls_start_faith.json`; figures `experiments/ls_start_faith/figures/`; notebook
`notebooks/E127_ls_start_faith.ipynb`. MATS `debug` L40, two jobs: 54836 stopped at its time budget as planned,
and 54874 resubmitted and finished. About 31 min, $0. Amendment 1: code `experiments/ls_start_faith/amend1.py`, filed
rules `results/predictions_amend1.json`, result `results/ls_start_faith_amend1.json`, job 54890, `debug` L40, about
19 min, $0, stopped on a control at its ninth of twelve reads. Amendment 2, E127's last: code
`experiments/ls_start_faith/amend2.py`, filed rules `results/predictions_amend2.json`, result
`results/ls_start_faith_amend2.json`, job 54934, `debug` L40, 15.5 min, $0, every control exact. Provisional: no
independent check, as the fast lane allows for a run under the gate.*

We read the least-squares start's weight change, and then each teacher-trained copy's weight change, in two places: on
the alive subcomponents and moved into Δ. The start's placement makes its excess under the adversarial check about four
times larger, teacher training's makes it three to four times smaller, and every copy's own change fails once in Δ.
For keeping a decomposition faithful along training, how an update is placed on the subcomponents decides, and the
change that copies the broad lowercase fine-tune is fragile under the check, with teacher training hiding most of it.

## The answer

**Amendment 2 reversed what we expected.** We predicted that the broad lowercase copy's own weight change, moved off the
subcomponents into Δ, would pass the check at about 1.24. It read 3.53, worse than the 1.91 it reads on the
subcomponents. Placement cuts both ways: the least-squares start's placement makes its excess nearly four times larger,
teacher training's makes it three to four times smaller. So the broad copy fails because the function it copies is
fragile under the check, and teacher training's placement hides most of that, enough for the LoRA copy but not this one.

The first run found why the least-squares start is less faithful than do-nothing: where its change sits, not how its
weights differ from the fine-tune. Its own weight change, left outside the subcomponents, reads better than do-nothing
on both tasks; moved onto the alive subcomponents, it reads nearly three times worse on lowercase. Only the adversary
finds this. Freezing the alive subcomponents fully on removes the excess, but it removes do-nothing's too, so it does
not show that those masks carry it.

## What we were trying to test

Each matrix of the released decomposition is $W = UV^\top + \Delta$: the alive subcomponents' write and read vectors
in $U, V$, the rest in $\Delta$. A fine-tune changes the weights by $\delta W$. Three states put a change in different
places:

- **do-nothing**: target $W_0 + \delta W$, with $\delta W$ in $\Delta$ and the subcomponents untouched;
- **least-squares start**: the least-squares split moves the alive subcomponents to $(U + dU, V + dV)$. Its weight
  change is $S = (U + dU)(V + dV)^\top - UV^\top$, and the target is $W_0 + S$. $S$ is not $\delta W$: the split's
  remainder, the part no change of the subcomponents reaches, is dropped;
- **same function in Δ**: target $W_0 + S$ with the subcomponents untouched, so $S$ sits in $\Delta$. It computes the
  same function as the start, but its change sits where do-nothing's does.

Each state is read by E34's adversarial check. An adversary with 20 sign steps sets every mask to maximise the KL
divergence from the target, on 256 Pile validation rows in two blocks. The ratio is
that KL over W₀'s, read the same way in the same job: 1 means the update did not make the decomposition easier to
break. The noise scale is the floor $f = 0.209$, W₀'s spread over three adversary seeds on the L40.

The comparison splits the start's excess over do-nothing into two parts that sum to it. The function part is the
same function in Δ minus do-nothing: what changes because $S$ is not $\delta W$. The placement part is the start minus
the same function in Δ: what changes when the same weights move from $\Delta$ onto the alive subcomponents.

Three further reads ask what carries the placement part:

- **the hold**: the same reads with every alive subcomponent's mask fixed at 1, so the adversary can move only the
  dead subcomponents and $\Delta$. A held ratio is over W₀ read under the same hold. If the placement part disappears,
  the adversary's freedom over the alive masks carries it. W₀ held against W₀ free is the positive control: a hold
  that changes nothing would leave them equal;
- **random masks**: the same states read by VPD's stochastic reconstruction with no adversary;
- **cancellation**: the coherence $\kappa = \|\sum_c D_c\|^2 / \sum_c \|D_c\|^2$ of the per-subcomponent changes
  $D_c = u'_c v'^\top_c - u_c v_c^\top$, input-weighted on the check's rows. Below 1, the changes cancel each other;
  a sign-flip null reads about 1. It is read on the start, on the copy teacher-trained from the start, and on the
  standard teacher-trained copy, which is faithful.

The two fine-tunes are E122's lowercase Adam step 300 and E121's emoticon Adam step 1000. Lowercase's excess was 3.5
ratio points, far beyond the floor; emoticon's was 0.18.

## What we expected, and why

- Lowercase's excess was expected to be placement, lean 0.5, with both parts 0.25 and function 0.15. The split keeps
  most of $\delta W$, so $S$ should damage Pile text about as much; the same function in Δ was expected at 4 to 6.
- Emoticon's whole excess was under the floor, so it was expected to read "within the floor", lean 0.6.
- If placement carried it, the hold was expected to remove it, lean 0.5. W₀ held was expected at 0.5 to 1.2 nats,
  from E20's reading of 1.08 nats with the alive subcomponents kept whole.
- Random masks were expected not to see it, lean 0.8, as E15 found for the step-10 split state.
- Cancellation was a coin toss. E17 read the step-10 least-squares split at 0.44 and component training at 2.8 to 3.5.

## Controls

Every control held, so the reads below are on the states E121 and E122 read.

| control | read | reference | verdict |
|---|---|---|---|
| W₀ mean block KL | 0.691743 | 0.691743 | exact |
| emoticon do-nothing ratio | 1.390022 | 1.390022 | exact |
| emoticon start ratio | 1.573282 | 1.573282 | exact |
| lowercase do-nothing ratio | 4.794234 | 4.794234 | exact |
| lowercase start ratio | 8.332889 | 8.332889 | exact |
| card | L40, both jobs | L40 | held |
| W₀'s first block reread by job 54874 | within 1e-6 | job 54836's | held |
| alive U, V equal E121's | equal | | held |

Source: `ls_start_faith.json` `controls.*.read` and `.ref`, and `jobs[].card`. The reread is not stored; the job stops
on a miss, and 54874 ended with exit 0.

## Is the excess in the start's function or in where its change sits?

![source](../../experiments/ls_start_faith/figures/source.png)

**Reading it.** The excess is all placement on both tasks, and the function part goes the other way. On lowercase,
the start's own weight change left in Δ reads 2.95, well under do-nothing's 4.79. Moving those same weights onto the
alive subcomponents raises the ratio to 8.33. So the placement part is 5.39 ratio points, more than the whole excess
of 3.54. The weights the split drops are, if anything, the part that hurts under the check. Emoticon shows the same
shape at a smaller size: the same function in Δ reads 1.15, under do-nothing's 1.39, and the start reads 1.57.

| task | do-nothing | same function in Δ | start | function part | its bar | placement part | its bar | reading |
|---|---|---|---|---|---|---|---|---|
| lowercase | 4.794 | 2.946 | 8.333 | −1.848 | 1.001 | 5.387 | 0.615 | placement |
| emoticon | 1.390 | 1.149 | 1.573 | −0.241 | 0.290 | 0.424 | 0.240 | placement |

Source: `score.{L,E}.ratio.{nothing,nothing_start,start}` and `score.{L,E}.source`. A part's bar is $f$ times its
reference ratio; the parts are differences of those ratios.

**Emoticon's excess and the floor.** The start's whole excess over do-nothing on emoticon is 0.18. That is 13% of
do-nothing's ratio, under the floor of 20.9%, so the plan expected "within the floor". The filed rule does not test
the whole excess; it tests each part against its own bar. The function part is negative, so it does not carry. The
placement part is 0.42, above its bar of 0.24, so the rule reads "placement". This is the rule as filed and the
reading stands. Its weight is low: both parts are within about two floors from one adversary seed. The plan's power
line already said emoticon's held reads would be reported only, and this report treats emoticon's exposure and
random-mask readings that way.

## Does holding the alive masks at 1 remove the placement part?

![hold](../../experiments/ls_start_faith/figures/hold.png)

**Reading it.** Yes on the filed rule, but the hold also flattens a state whose change is not on the alive
subcomponents, and Amendment 1 below found it flattens do-nothing too. The reading is therefore reported
"unreadable under the hold". On lowercase, the start reads 0.98 of W₀ held and the same function
in Δ reads 1.11. The held placement part is therefore slightly negative, and the share removed is 1.02 against the
filed bar of 0.75. Emoticon reads the same way, at 1.15.

**The hold changes the adversary a lot.** W₀ held reads 1.60 nats against 0.69 free, 2.3 times as much. The plan
expected 0.5 to 1.2, so the positive control passed by more than expected. Fixing a mask at 1 is one of the
settings a free adversary may choose, so a perfect free adversary would read at least 1.60 at W₀. The free
20-step adversary does not find it: it is a local search, and freeing the alive masks sends it somewhere weaker at W₀.
Held and free reads are therefore two different adversaries. The held ratios divide by 1.60 and the free ones by 0.69,
so a nat moves a held ratio less than half as far as a free one. Computed in nats instead of ratios, the share
removed is 1.06 on lowercase and 1.35 on emoticon, so the verdict does not depend on the denominators.

**Not specific yet.** The same function in Δ has its change outside the alive subcomponents, yet the hold removes most
of its excess too. On lowercase it reads 1.35 nats above W₀ free and 0.17 above W₀ held: the hold removes 87% of it.
Every other held state reads under W₀ held, by up to 0.31 nats. So there are two readings left
open. The hold may remove what the placement adds, as the filed rule reads it. Or the held adversary may sit in a
different optimum, driven by the dead and Δ masks, where every state looks like W₀. Under the second reading the hold
has no teeth for this question. Telling them apart needs do-nothing and a standard teacher-trained copy read under the
hold: if do-nothing keeps its excess there, the hold is specific. Amendment 1 read them, and do-nothing does not keep
it.

| state | lowercase free | lowercase held | emoticon free | emoticon held |
|---|---|---|---|---|
| W₀, nats | 0.692 | 1.601 | 0.692 | 1.601 |
| do-nothing, nats | 3.316 | not run | 0.962 | not run |
| same function in Δ, nats | 2.038 | 1.772 | 0.795 | 1.394 |
| start, nats | 5.764 | 1.567 | 1.088 | 1.290 |
| same function in Δ, ratio | 2.946 | 1.107 | 1.149 | 0.871 |
| start, ratio | 8.333 | 0.979 | 1.573 | 0.806 |
| share removed, filed (ratios) | 1.024 | | 1.153 | |
| share removed, in nats | 1.055 | | 1.355 | |

Source: `kl.*`, the mean of the two blocks; `score.{L,E}.ratio.*|hold`; `score.{L,E}.exposure`. The nats share is
computed here from `kl` with `jq`: one minus the held placement gap over the free placement gap, both in nats.

## Amendment 1: does the hold have teeth?

*Plan `reports/plans/2026-11-06_plan_e127-ls-start-faithfulness.md` → "Amendment 1"; rules
`results/predictions_amend1.json`, filed before the run; result `results/ls_start_faith_amend1.json`; log
`results/large/amend1.log`; job 54890.*

**What it asked.** The hold above flattened the start, and also a state whose change sits in Δ. If it flattens
do-nothing as well, the held check has lost its ability to see any excess, and the placement reading tells us nothing
about the alive masks. So the amendment read, on the same rows, seed, card and code path, three states under the hold:

- **do-nothing held**, the positive control for the hold. Do-nothing's free excess on lowercase is 3.79 ratio points, 18
  floors. A hold with teeth should leave a large part of it;
- **the standard teacher-trained copy**, free and held. It is E122's copy of the same lowercase fine-tune, trained by
  E37's recipe from W₀'s alive subcomponents, and on emoticon E121's order-0 copy;
- **the copy trained from the least-squares start**, free and held, read for description only.

Each copy's free read is a control: it must reproduce the reading stored when the copy was made.

**The filed rules.** Write $r$ for a ratio, free or held, over W₀ read the same way.

- **Teeth**, per task: the hold has teeth if do-nothing held keeps at least half of do-nothing's free excess, and at
  least one floor: $r_{\text{nothing|hold}} - 1 \ge \max(f, 0.5\,(r_{\text{nothing}} - 1))$. If not, that task's
  E127 exposure reading is reported "unreadable under the hold", and no other rule replaces it.
- **Standard copy**, per task: "exposed" if its held ratio falls more than one floor under its free one. It is "out of
  reach" if the two are within one floor, and "raised by the hold" if the held one is more than a floor higher.

The lean was "no teeth" at 0.85 on lowercase and 0.8 on emoticon, because the same function in Δ had already lost most
of its excess under the hold.

### Controls: the job stopped at its ninth read

| control | read | stored | tolerance, relative | verdict |
|---|---|---|---|---|
| W₀, nats | 0.691743 | 0.691743 | 1e-3 | exact |
| W₀ held, nats | 1.600863 | 1.600863 | 1e-3 | exact |
| lowercase standard copy, free ratio | 1.912875 | 1.912875 | 1e-3 | exact |
| lowercase copy from the start, free ratio | 9.404895 | 9.404895 | 1e-3 | exact |
| emoticon standard copy, free ratio | 1.006981 | 0.986613 | 1e-2 | missed by 2.1%; job stopped |
| card | L40 | L40 | | held |

Source: `ls_start_faith_amend1.json` `controls.*.read`, `.ref`, `.tol`, `stopped`, `jobs[].card`. The relative miss is
$(1.006981 - 0.986613)/0.986613 = 0.0206$, computed with `jq`.

Lowercase, the headline task, ran complete. On emoticon, do-nothing held was read, then the standard copy's free read
missed its band and the job stopped with exit 1. The emoticon copy reads after it were not run. The miss is most likely
bf16 rounding: E121 read the emoticon copies as fp32 states in memory but saved them only as bf16 offsets, and this job
read the saved offsets. The plan foresaw this and set the band at 1e-2 rather than 1e-3, without a measured floor for
it; the cause is not proven. The lowercase copies' stored readings were taken from the same bf16 files, and they
reproduce exactly.

**Decision, taken by the lead: no rerun of the emoticon copy reads.** The teeth reading they would sit under is
already "no teeth" on emoticon, so they could not change a verdict. E127's second and last amendment slot went to
Amendment 2 below.

### Under the hold, does any lowercase state keep its excess?

![amend1 hold](../../experiments/ls_start_faith/figures/amend1_hold.png)

**Reading it.** No: the hold has no teeth. Do-nothing reads 3.32 nats free and 1.91 held, against W₀'s 0.69 free and
1.60 held. As a held ratio that is 1.19, keeping 0.19 of its free excess of 3.79, about 5%. The filed bar was 1.90, so
the hold reads "no teeth". The two copies fall to W₀'s held line as well. The standard copy goes from 1.91 free to 1.00
held, and the copy from the start from 9.40 free to 0.97 held. Every state on the figure, whatever its free reading and
wherever its change sits, ends within 0.31 nats of W₀ held.

On emoticon the hold has no teeth either: do-nothing held reads 0.88, under W₀ held, against a bar of one floor.

| state, lowercase | free ratio | held ratio | free, nats | held, nats | reading |
|---|---|---|---|---|---|
| W₀ | 1 | 1 | 0.692 | 1.601 | reference |
| do-nothing | 4.794 | 1.194 | 3.316 | 1.912 | teeth: no teeth |
| same function in Δ, E127 | 2.946 | 1.107 | 2.038 | 1.772 | |
| least-squares start, E127 | 8.333 | 0.979 | 5.764 | 1.567 | |
| standard copy | 1.913 | 1.002 | 1.323 | 1.604 | exposed |
| copy from the start | 9.405 | 0.970 | 6.506 | 1.554 | exposed, descriptive |

| emoticon | read | rule | verdict |
|---|---|---|---|
| do-nothing held ratio | 0.877 | teeth, bar 1 + f = 1.209 | no teeth |
| standard copy free ratio | 1.007 | control, stopped the job | |
| standard copy held; copy from the start free and held | not run | | not run |

Source: `ls_start_faith_amend1.json` `score.{L,E}.ratio`, `score.{L,E}["R1 teeth"]`, `["R2 std_copy"]`,
`["ls_copy (descriptive)"]`; nats are the mean of the two blocks in `kl`, from this file for the amendment's states and
from `ls_start_faith.json` for E127's. The lowercase teeth bar is $\max(0.209, 0.5 \times 3.794) = 1.897$ above 1,
computed with `jq`. The standard copy's exposed line is its free ratio minus $f$, 1.704.

**What follows, under the filed rule.** On both tasks E127's exposure reading, "the alive masks", is reported
"unreadable under the hold" (`score.{L,E}.placement`). What stands is everything in E127 that used no hold. The
free-read split stands: the function part is negative, and placing the change on the alive subcomponents carries the
whole excess. So do the random-mask and cancellation readings. What falls is that the hold isolates the placed update.
It brings do-nothing, the standard copy and the copy from the start alike to about W₀'s held reading. So the hold takes
away the adversary's ability to find any excess, not the placement's in particular.

## Do random masks see the placement part?

![random](../../experiments/ls_start_faith/figures/random.png)

**Reading it.** No: only the adversary sees it. Under random masks the start and the same function in Δ read within
1% of each other on both tasks, against an adversarial gap of 5.39 ratio points on lowercase. Every random-mask ratio
sits between 0.92 and 1.05, inside the 0.9 to 1.2 the plan expected. The floor used here is borrowed from the
adversarial check; the random read's own seed noise was not measured.

| task | same function in Δ, random | start, random | start over same, minus 1 | reading |
|---|---|---|---|---|
| lowercase | 1.049 | 1.039 | −0.009 | adversary only |
| emoticon | 0.919 | 0.924 | 0.005 | adversary only |

Source: `score.{L,E}.random_ratio`, `score.{L,E}.random`. Raw stochastic, CI-masked and unmasked KLs are in
`random.*`.

## Do the per-subcomponent changes cancel on the failing states only?

![coherence](../../experiments/ls_start_faith/figures/coherence.png)

**Reading it.** No. Input-weighted, the start reads 0.87 on lowercase and 0.61 on emoticon, both above the cancelling
bar of 0.5. The copy trained from the start, which keeps the excess, reads 1.30 and 0.89, so it does not cancel at
all. The standard copy reads 4.49 and 5.55: its changes add up, as E17 found for component training. Frobenius
coherence gives the same verdict. There the start does cancel, at 0.44 and 0.42, matching E17's 0.44, but the copy
from the start still does not. So weight-side cancellation separates the start from the standard copy and fails to
separate the two failing states from it.

| task | state | input-weighted κ | its null | Frobenius κ | its null |
|---|---|---|---|---|---|
| lowercase | start | 0.873 | 0.992 | 0.443 | 0.990 |
| lowercase | copy from the start | 1.299 | 0.986 | 0.699 | 0.992 |
| lowercase | standard copy | 4.485 | 0.999 | 2.688 | 0.997 |
| emoticon | start | 0.614 | 0.956 | 0.424 | 0.974 |
| emoticon | copy from the start | 0.886 | 0.954 | 0.574 | 0.977 |
| emoticon | standard copy | 5.554 | 1.023 | 2.910 | 0.997 |

Source: `weights.{L,E}.coherence.{start,ls_copy,std_copy}.{weighted,frobenius}`.

**The start's weight change against the fine-tune's.** Input-weighted, $S$ has cosine 0.895 with $\delta W$ on
lowercase and 0.848 on emoticon. The distance relative to $\delta W$'s size is 0.449 and 0.530. Emoticon's distance is
inside the plan's 0.5 to 0.7. Source: `weights.{L,E}.S_vs_dW.{weighted,frobenius}`.

## Amendment 2: would the broad copy's own change pass if it sat in Δ?

*Plan `reports/plans/2026-11-06_plan_e127-ls-start-faithfulness.md` → "Amendment 2"; rules
`results/predictions_amend2.json`, filed before the run; result `results/ls_start_faith_amend2.json`; log
`results/large/amend2.log`; job 54934. Design and leans adopted from Fable's advice of the same night, its "Run 1".*

**Something unexpected happened, against the filed expectation.** Fable predicted the standard lowercase copy's change,
moved into Δ, would read 1.24 and pass, with a band of 1.10 to 1.45. It read 3.53. That is not even the null, which
was "no change from its 1.91 on the subcomponents"; it went the other way, nearly eight floors above it.

**What it asked.** On lowercase, E127's start reads 3.77 times as much excess over W₀ on the alive subcomponents as in
Δ: $(8.333 - 1)/(2.946 - 1)$. The standard teacher-trained copy of the same fine-tune fails the check at 1.91, over
the fail line of 1.61 that E122 took from E39. If any change on the alive subcomponents reads about 3.8 times its
in-Δ excess, the copy's own function should pass in Δ, and the copy fails only because of where it sits. So each copy's
weight change $S_c = U'V'^\top - UV^\top$ was read twice:

- **the copy, free**: the alive subcomponents at the copy's $(U', V')$, target $W_0 + S_c$. These reproduce the
  readings stored when each copy was made, so they are the controls;
- **the copy's change in Δ**: target $W_0 + S_c$ with the subcomponents left at the released $(U, V)$, so $S_c$
  sits in Δ, as the same function in Δ did for the start.

At full masks the two are the same weights; only where $S_c$ sits differs. Three copies were read, all by the same
teacher training: Adam at $10^{-5}$, batch 8, 1,000 steps on lowercase rows, loss the KL from the fine-tuned model.

- **standard copy**, E122's copy of the Adam fine-tune, started from W₀'s alive subcomponents;
- **copy from the start**, the same recipe started from the least-squares start;
- **copy of the rank-8 LoRA**, E39 Amendment 2's copy of a rank-8 LoRA fine-tune on the same task, which passes.

Emoticon was not read: its standard copy's free excess is about 0.005 nats, nothing to divide.

**The filed rules.** The headline rule reads the standard copy's change in Δ against the fail line with a band of one
floor $f$: under $1.61 - f = 1.40$ "placement explains the broad failure"; over $1.61 + f = 1.82$ "function fails too";
between, "within the line". A descriptive multiplier per copy, $m = (\text{free} - 1)/(\text{in Δ} - 1)$, compares
with the start's 3.77; both reads divide by one W₀, so it is also the multiplier in nats of excess. The lean on the
headline rule was "placement explains" 0.5, "function fails too" 0.25.

### Controls

| control | read | stored | tolerance, relative | verdict |
|---|---|---|---|---|
| W₀, nats | 0.691743 | 0.691743 | 1e-3 | exact |
| standard copy, free ratio | 1.912875 | 1.912875 | 1e-3 | exact |
| copy from the start, free ratio | 9.404895 | 9.404895 | 1e-3 | exact |
| copy of the rank-8 LoRA, free ratio | 1.326288 | 1.326288 | 1e-3 | exact |
| card | L40 | L40 | | held |

Source: `ls_start_faith_amend2.json` `controls.*.read`, `.ref`, `jobs[].card`. The alive $U, V$ equal E121's: the
assert at load passed and the job ended with exit 0.

### Does each change read better on the alive subcomponents or in Δ?

![amend2 delta](../../experiments/ls_start_faith/figures/amend2_delta.png)

**Reading it.** It depends on how the change was placed, and the broad copy's function fails in Δ. The standard copy's
change reads 1.91 on the subcomponents and 3.53 in Δ, so the filed reading is "function fails too". Teacher training's
placement shrinks the excess: by 2.8 times for the standard copy, 4.1 times for the LoRA copy. The least-squares
start's placement grows it, by 3.8 times, and the copy trained from the start keeps most of that, at 2.8 times. Every
copy's change fails in Δ, the LoRA copy's at 2.34 included, against do-nothing's 4.79 for the fine-tune's own change.

| state, lowercase | on the subcomponents | in Δ | excess, on / in Δ | excess in Δ, nats |
|---|---|---|---|---|
| do-nothing, the fine-tune's own change | | 4.794 | | 2.625 |
| least-squares start, E127 | 8.333 | 2.946 | 3.77 | 1.346 |
| copy from the start | 9.405 | 3.961 | 2.84 | 2.048 |
| standard copy | 1.913 | 3.526 | 0.36 | 1.747 |
| copy of the rank-8 LoRA | 1.326 | 2.336 | 0.24 | 0.924 |

Source: `ls_start_faith_amend2.json` `score.ratio`, `score["R2 *"].m`, `score["R1 std_copy"]`; E127's rows from
`ls_start_faith.json` `score.L.ratio`. Nats are the mean of the two blocks in `kl` minus W₀'s, computed with `jq`. The
ratios were recomputed from `kl` with `jq` and agree. In nats, the standard copy's excess is 0.63 on the subcomponents
and 1.75 in Δ.

**Against Fable's leans.** Standard copy in Δ: lean 1.24, band 1.10 to 1.45, read 3.53, missed high. Copy from the
start in Δ: lean 3.2, band 2.4 to 4.5, read 3.96, inside the band. LoRA copy in Δ: lean 1.09, read 2.34, missed high.
The lean's claim was one multiplier for any change on the subcomponents; the copy from the start obeys it only because
it started from the least-squares placement.

**What teacher training's loss contains.** Nothing about masks. Each copy is made by `row_copy.copy_states`
(`src/commute/row_copy.py:81-82`), which calls `component_training.train` with `teacher.teacher_loss(..., None)`. That
loss is the KL from the fine-tuned model to the student, averaged over every position (`src/commute/teacher.py:40-47`).
The student's forward is the plain target model, with the change in $(U', V')$ added to each frozen linear by a hook
(`src/commute/component_training.py:68-74`, `:152-158`). No causal-importance mask, no stochastic mask, no adversary
and no faithfulness term enters it. So teacher training's protective placement is not by design: it falls out of
training the subcomponents' vectors directly, at full masks, from where the optimiser starts. Started from the
least-squares start, the same loss keeps the exposed placement.

**A hypothesis for why, untested.** The gradient on a subcomponent's write vector scales with how strongly its read
vector sees the training inputs. Those are plausibly the inputs where Γ gives it high causal importance, and the
adversary cannot push a mask below its causal importance. So teacher training may put its change mostly on
subcomponents that are on wherever the change matters, which the adversary cannot switch off. The least-squares split
spreads the change by least norm, regardless of which subcomponents are on. Reading where each change sits against
the causal importances on the check's rows would test this; it was not run.

**The blind spot.** The two placements give the adversary different freedoms. In Δ the change shares one mask per
matrix with W₀'s own Δ; on the subcomponents it rides one mask per subcomponent, each between its causal importance
and 1. So "the copy's function is fragile" means fragile under the check with the change placed in Δ. It does not say
the function would fail under any placement, and teacher training's placement is one where it fails less.

**What this does to E127's 3.8 times.** It is a property of the least-squares split's placement, not of placement in
general. Placing a change on the subcomponents can multiply its excess by four or divide it by four, depending on how it
was put there. E127's free-read split stands as read: for the start, placement carries the whole excess.

**What this does to the copy's distance from δW.** On emoticon, E121 found the standard copy's weight change far from
the fine-tune's: similarity 0.40, where a rerun of the fine-tune reads 0.83. Fable read that distance as the copy
dropping collateral change on ordinary text. On lowercase that reading is not supported: the copy's change in Δ keeps 67% of do-nothing's
excess, $(3.526 - 1)/(4.794 - 1)$, so the copy's function is nearly as fragile in Δ as the fine-tune's own change. The
distance is better read as the copy's change being shaped to sit where the masks protect it. A copy that stays close
to δW, like the copy from the start, keeps the exposed placement and fails at 9.40.

## What we learned

The least-squares start's poor faithfulness on these two fine-tunes is a placement effect. The same weights read
better than do-nothing when they sit in Δ, and worse once they ride on the alive subcomponents' masks. That fits the
plan's second premise check: in Δ a change shares one adversarial knob per matrix, while on the subcomponents it rides
thousands of separate masks. Only the adversary exploits this; random masks and weight-side cancellation do not track
it. For `GOALS.md`, this says placing an update by the least-squares split can make a decomposition easier to break
than leaving the update outside it, even when the weights are nearly the same. Whether the alive masks are uniquely to
blame cannot be read with the hold: Amendment 1 found that it flattens do-nothing too, so it removes the adversary's
reach in general. A question about which masks carry the excess needs a different tool.

Amendment 2 turns the placement finding from a cost into a lever. Placing a change on the subcomponents is not costly
in itself: the least-squares split's placement multiplies the excess by about four, teacher training's divides it by
three to four, and teacher training's loss has no term that asks for this. The broad lowercase failure is therefore not
a good function exposed by its placement. Every copy's change fails once put in Δ, at 2.34 to 3.96, where the Adam
fine-tune's own change reads 4.79. Teacher training's placement hides most of that: enough for the LoRA copy at 1.33,
not for the full-matrix Adam copy at 1.91. For `GOALS.md`, keeping a decomposition faithful along training depends on
placing the update where the adversary cannot reach it; a loss that asks for that directly is one candidate.

## How compelling is the evidence

- One adversary seed, one row set of 256 Pile rows, two tasks, one fine-tune each, one L40.
- Lowercase's placement part is 8.8 floors of its reference; emoticon's is 1.8. Emoticon is weak evidence.
- The hold is a different adversary from the free one. On lowercase, Amendment 1 read it on all six states, and
  do-nothing kept about 5% of its free excess; the "no teeth" reading sits far from its bar, 0.19 against 1.90.
- Emoticon's hold reading rests on do-nothing alone; its copies were not read held.
- The random-mask read's own noise floor was not measured.
- Cancellation is weight-side only.
- Amendment 2's headline is far from its edges: the standard copy's change in Δ reads 3.53 against the "function fails
  too" edge of 1.82, eight floors above it. Its multipliers come from one seed and one task, three copies of one
  recipe. "Fragile function" is read under the Δ placement only, where the adversary's freedom differs from the
  subcomponents'.

## Surprises

- **Amendment 2 reversed the filed expectation.** The standard copy's change in Δ was predicted at 1.24, a pass; it
  read 3.53, above its own 1.91 on the subcomponents. The LoRA copy's was predicted at 1.09 and read 2.34. Teacher
  training's placement protects; the least-squares split's exposes.

- **The function part is negative on both tasks.** The split's dropped remainder makes the same function in Δ read
  better than do-nothing, by 1.85 ratio points on lowercase. The plan expected 4 to 6 for it; it read 2.95.
- **W₀ held reads 1.60 nats, above the expected 0.5 to 1.2.** The free adversary at W₀ finds less than a setting
  it is allowed to choose.
- Emoticon read "placement" where "within the floor" was expected; this is the rule scoring parts, not the whole.
- The standard copies' input-weighted coherence, 4.49 and 5.55, is above the expected 2 to 4.
- Amendment 1 stopped on the emoticon standard copy's free read, 1.007 against a stored 0.987. Most likely this is bf16
  rounding of the saved copy, which E121 had read in fp32; not proven.
- Amendment 1's leans held: "no teeth" on both tasks, and the lowercase standard copy "exposed". The copy from the
  start read 0.97 held, just under its descriptive lean of 1.0 to 1.5.

## What next

- A free adversary with more steps or restarts at W₀: if it reaches the held 1.60, the free check has been
  under-reading every state, and the ratios E34 used would change scale. It would also say whether the hold's flat
  readings are a stronger adversary at W₀ or a weaker one everywhere else.
- Which masks carry the placement excess is now open with no filed route. A hold of a subset of the alive
  subcomponents, or at their CI values, would be a new plan, not an amendment: E127 has no amendment slots left.
- Where each change sits against the causal importances on the check's rows, for the start and the standard copy.
  If teacher training's change sits on subcomponents with high causal importance and the start's does not, the
  hypothesis above holds and the placement is explained; if not, it needs another cause. A new plan.
- Teacher training with a faithfulness term in its loss, for example VPD's own masked reconstruction. If the broad
  Adam copy then passes, placement is the lever for broad updates; if it stays over the line, the function is the limit.

## Scorecard

| rule | what it says | rule as filed | read | lean | verdict |
|---|---|---|---|---|---|
| A, lowercase | function or placement | each part carries at f times its reference | function −1.848, placement 5.387 | placement 0.5 | placement; lean held |
| A, emoticon | same | same | function −0.241, placement 0.424 | within the floor 0.6 | placement; lean missed |
| B, lowercase | hold removes placement part | H ≥ 0.75 alive masks | H 1.024 | alive masks 0.5 | the alive masks as filed; unreadable under the hold by Amendment 1's R1 |
| B, emoticon | same, report only per the plan | same | H 1.153 | none | the alive masks as filed; unreadable under the hold by Amendment 1's R1 |
| C, lowercase | random masks see it | q ≥ f | q −0.009 | adversary only 0.8 | adversary only; held |
| C, emoticon | same | same | q 0.005 | same | adversary only; held |
| D, both | cancellation tracks failing states | start and copy from start κ < 0.5, standard not | 0.873, 1.299, 4.485; 0.614, 0.886, 5.554 | 0.5 / 0.5 | does not track |

Expected values filed beside the rules:

| expected | read | verdict |
|---|---|---|
| controls C1 to C8 | all exact or within tolerance | held |
| S against δW, emoticon, weighted distance 0.5 to 0.7 | 0.530 | held |
| lowercase same function in Δ 4 to 6 | 2.946 | missed, low |
| W₀ held 0.5 to 1.2 nats | 1.601 | missed, high |
| random-mask ratios 0.9 to 1.2 | 0.919 to 1.049 | held |
| κ weighted, start 0.3 to 0.6 | 0.873 and 0.614 | missed, high |
| κ weighted, standard copy 2 to 4 | 4.485 and 5.554 | missed, high |
| κ weighted, copy from the start 0.3 to 1.0 | 1.299 and 0.886 | lowercase missed, emoticon held |
| sign-flip nulls about 1 | 0.954 to 1.023 | held |
| about 30 min as two jobs | 26 min, then 5 min | held |

Amendment 1, `predictions_amend1.json`:

| rule | what it says | rule as filed | read | lean | verdict |
|---|---|---|---|---|---|
| R1, lowercase | the hold has teeth | nh − 1 ≥ max(f, 0.5 (n − 1)) | nh 1.194; bar 1.897 above 1 | no teeth 0.85 | no teeth; lean held; placement unreadable under the hold |
| R1, emoticon | same | same | nh 0.877; bar f = 0.209 | no teeth 0.8 | no teeth; lean held; placement unreadable under the hold |
| R2, lowercase | standard copy under the hold | exposed iff ch < c − f | c 1.913, ch 1.002 | exposed 0.65 | exposed; lean held |
| R2, emoticon | same | same | not read | out of reach 0.65 | not run |
| copy from the start, lowercase | descriptive | same three-way reading | 9.405 → 0.970 | exposed, held 1.0 to 1.5 | exposed; held value just under the lean range |
| copy from the start, emoticon | descriptive | same | not read | exposed or at the edge | not run |
| controls C1 to C3, C5 | W₀, W₀ held, lowercase copies, card | 1e-3 relative | exact | | held |
| control C4 | emoticon copies reproduce | 1e-2 relative | standard copy 1.0070 vs 0.9866 | | missed; job stopped |
| control C6 | a resubmitted job rereads W₀'s first block | 1e-6 relative | one job only | | does not apply |
| control C7 | alive U, V equal E121's | equal | equal; the assert at load passed | | held |

Amendment 2, `predictions_amend2.json`, lowercase:

| rule | what it says | rule as filed | read | lean | verdict |
|---|---|---|---|---|---|
| R1, standard copy | does its change pass in Δ? | free > 1.82; in Δ < 1.40 placement explains, > 1.82 function fails too | free 1.913, in Δ 3.526 | placement explains 0.5, function fails too 0.25 | **function fails too; lean reversed** |
| R2, standard copy | multiplier, descriptive | m = (free − 1)/(in Δ − 1), both excesses ≥ f | 0.361; 0.096 of 3.77 | about 3.8 | readable |
| R2, copy from the start | same | same | 2.839; 0.754 of 3.77 | about 3.8 | readable |
| R2, LoRA copy | same | same | 0.244; 0.065 of 3.77 | in-Δ excess within the floor | readable |
| lean band, standard copy in Δ | | 1.10 to 1.45 | 3.526 | 1.24 | missed, high |
| lean band, copy from the start in Δ | | 2.4 to 4.5 | 3.961 | 3.2 | held |
| lean, LoRA copy in Δ | report only | | 2.336 | 1.09 | missed, high |
| controls C1, C2, C3, C5 | W₀, the three copies free, card, alive U, V | 1e-3 relative | exact | | held |
| control C4 | a resubmitted job rereads W₀ | 1e-6 relative | one job only | | does not apply |

## Departures from the literature

- **The hold has no counterpart in the VPD paper.** The paper's adversary moves every mask. Here, for this question
  only, it wraps the authors' `interpolate_pgd_mask` so the alive subcomponents' masks read 1 with zero gradient. The
  held reads are a different adversary, as the hold section shows.
- **E34's check departs from the paper as E34 and E121 recorded:** the pass lines and the floor are ours, and the
  check uses one adversary per 128-row block where the paper shares one over its evaluation batch.
- **The coherence κ is ours, from E17**, not a published measure; its input-weighted form is new here.

- **The teeth rule is ours**, filed for Amendment 1, with its half-the-excess bar; the paper has no hold to test.
- **Reading a copy's change in Δ is ours**, for Amendment 2. The paper reads one decomposition with its own Δ; here a
  weight change made on the subcomponents is moved into Δ and the check is read again, so the two placements can be
  compared at the same weights.

## Not run

- **Emoticon's standard copy held, and its copy from the start free and held.** Amendment 1 stopped before them on the
  standard copy's free control. Not rerun, by the lead's decision: the teeth reading they would sit under is already
  "no teeth", so they cannot change a verdict.
- A measured bf16 floor for the emoticon copies, read once from fp32 and once from the saved bf16 offsets; it would
  confirm or rule out rounding as the cause of the stop.
- Adversary seeds 1 and 2: emoticon's parts are not readable from one seed.
- A free adversary with more steps or restarts, to see whether it reaches the held 1.60 at W₀.
- A hold at the CI values, and a hold of a subset of the alive subcomponents.
- Pile set B; other rows.
- The start with target $W_0 + \delta W$ and the remainder in Δ.
- The random-mask read's own seed noise.
- Amendment 2 on emoticon: its standard copy's free excess is about 0.005 nats, nothing to divide.
- Amendment 2 on adversary seeds 1 and 2, and on Pile set B.
- The rank-8 LoRA fine-tune's own do-nothing on the L40. E39 read it at 2.65 on an H100, a different card, so it is
  not set beside the LoRA copy's 2.34 in Δ here.
- Where each change sits against the causal importances on the check's rows: the test of the hypothesis for why
  teacher training's placement protects.
- Teacher training with any mask or faithfulness term in its loss.

## Provenance

Result `experiments/ls_start_faith/results/ls_start_faith.json` (`done: true`); run log
`~/commute-data/results-large/ls_start_faith/run.log`; job logs `cluster/logs/commute-e127-54836.out` and
`commute-e127-54874.out`. Job 54836 ran from 21:35 and stopped at its 28-minute budget with exit 3. Job 54874 ran from 22:02 and
ended with exit 0 at 22:07. Each PGD read took 127 to 140 s (`seconds`). MATS `debug` QoS, L40, $0.

Amendment 1: result `experiments/ls_start_faith/results/ls_start_faith_amend1.json` (no `done`; `stopped` names the
control), committed in 57c812fd; run log `experiments/ls_start_faith/results/large/amend1.log`; figure
`experiments/ls_start_faith/figures/amend1_hold.png` from `plot.py`. Job 54890 started 22:27:17 on the L40
(`cluster/logs/jobs.log`), read W₀ at 22:29:54 and stopped at 22:47:11 with exit 1, after nine reads of 127 to 132 s.
Peak host memory 6.1 GB. MATS `debug` QoS, $0.

Amendment 2: result `experiments/ls_start_faith/results/ls_start_faith_amend2.json` (`done: true`), committed in
04a7cf66; run log `experiments/ls_start_faith/results/large/amend2.log`; figure
`experiments/ls_start_faith/figures/amend2_delta.png` from `plot.py`. Job 54934 started 23:09:24 on the L40
(`cluster/logs/jobs.log`) and ended with exit 0 at 23:24:57, after seven reads of 127 to 131 s. Peak host memory
6.1 GB. MATS `debug` QoS, $0.
