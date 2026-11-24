# Plan: a rotation-tolerant kept share, as a diagnostic beside the frozen definition (cell P1.60)

The frozen definition's "kept" asks whether a needed variable's write change is M's own write up to one gain. This
diagnostic asks a looser question: does the change stay inside the space M already writes into? It is not a
definition change and no verdict is given under it. Card: `experiments/rotation_kept/card.md`; filed values:
`experiments/rotation_kept/results/predictions.json`.

## 1. The degeneracy, settled first

Read from `commute.stand_in` (VPD sites), `commute.needed_neurons` (`HeadSite`, `MLPSite`) and `commute.kstar_vpd`.
For each kind of variable, the rank across inputs of M's write, and what "F's change inside M's write space" becomes:

| variable | M's write | rank across inputs | the own-space share becomes |
| --- | --- | --- | --- |
| VPD subcomponent c | `a u_c`, a = v_c · y | 1 | 1 − Σ a² ‖du_c⊥‖² / dd |
| VPD remainder | `Δ y` | up to 768 or 3,072 | close to 1 for any change (space nearly full) |
| neuron i | `s_M W_M[:, i]` | 1 | 1 − Σ s_F² ‖dW_i⊥‖² / dd |
| MLP remainder (neuron basis) | `(a − s) W_outᵀ` | up to 768 | close to 1 for any change |
| head h | `z_h W_O,h` | up to 128 of 768 | not degenerate |

Subcomponents and neurons are 38,914 of VPD's 38,938 variables and 12,288 of the neuron basis's 12,316. For them M's
write space is one direction, so a rotated write direction counts as lost, exactly as in the frozen kept. The reading
still differs from the frozen kept, which credits only the part of the change that is M's write times one gain on
every input. The own-space share also credits any change in when, or how strongly, the variable fires. For a
one-direction variable it is always at least the frozen kept. So for most variables the reading measures "input
side against output side", not rotation. A Gaussian update moves both sides by about the same relative amount, so a
matched random is expected near 0.5 on it at neurons, not low.

**Chosen reading: the needed-space share.** M's write space for a variable is widened to its own write space plus M's
writes of the other most-needed variables at the same site. The site is the matrix at VPD, and at the neuron basis
it is the layer's MLP for neurons and the layer's attention for heads. Remainders are read as variables but are not
added to anyone else's space, since their space is nearly full. This answers "rotated toward what M's needed
variables already write, or somewhere new", and a random direction's expected share is known exactly (the null
below). Heads keep their own-space share as a second reading, the one place it is not degenerate. whip-lead approved it
on 2026-11-23 as a measurement choice inside the approved diagnostic. The degeneracy finding opens the results page.

## 2. The reading, defined

Per item (F, M, D), rows are every position of the first min(256, |D|) windows of D, in F's own run. Writes are as in
`stand_in.write_change`: M's mechanism and F's applied to F's own input to the variable. For variable u, W_M(u) and
W_F(u) are the rows-by-d_out write matrices and Δ(u) = W_F(u) − W_M(u).
- Ranking: flips from the saved scan (number of tested pairs that change F's top token). The read set N is the
  ⌈n/100⌉ changed members of K with most flips; the reference set R is the ⌈n/100⌉ members of K with most flips,
  changed or not. Ties go to the lower index.
- Own space O(u): the fewest top eigen-directions of W_M(u)ᵀW_M(u) holding 99% of its trace.
- Needed space S(u): the span of O(u) and of O(v) for every non-remainder v in R at u's site, numerical rank at
  relative tolerance 1e-6. Its null is ν(u) = dim S(u) / d_out.
- Per variable: strict kept (the frozen kept, `stand_in.explained` / dd), own-space share ‖Δ P_O‖² / ‖Δ‖², and
  needed-space share ‖Δ P_S‖² / ‖Δ‖², where P is the orthogonal map onto that space.
- Matched null (redteam-lead): the same share in A(u), the span of O(u) and as many own spaces of variables drawn
  with a seed from the site's non-remainder variables outside R and N as S(u) uses. It matches S's dimension and
  M's uneven use of directions, which ν does not.
- Pooled, change-weighted, over the non-remainder members of N: each numerator summed, divided by Σ dd. The same
  with remainders is reported beside it, with the non-remainder share of N's dd (redteam-lead: new_thin's read then
  rests on about 1% of its change).
- Rotation share: (needed − own) / (dd − own), pooled. It is the part of the change leaving the own space that lands
  in the needed space. It is read against two nulls: the same with the matched space, and the ν form
  (dim S − dim O) / (d_out − dim O). Every item prints strict, own and needed side by side with ν and the matched null.

The strict share never exceeds the own share for a one-direction variable, and the own share never exceeds the needed
share for any variable. The code checks both on every item.

## 3. Items, in the cell's order, with saved state paths (checked on disk 2026-11-23)

K paths are under `experiments/<exp>/results/large/` (symlinks into `~/commute-data/results-large/`).

| class | VPD (subcomponents and remainders) | neurons and heads |
| --- | --- | --- |
| do-nothing | E257's `do_nothing` | E277's built and trained `do_nothing` |
| built x2, x10, rebuilt, new_few, new_thin, two randoms | E268 `kstar_needed/.../items_<item>.npz` (7 found) | E277 `needed_neurons/.../items_<item>.npz` (7 found) |
| E62 copy and lookup s0-2, lookup scaled s0-2 | E275 `claim2_needed/.../items_e62-*.npz` (9 found) | E277 `items_e62__*.npz`, `items_e62-lookup_*` (9 found) |
| head (2,4) on E62's 206 copy windows | none | 6 states, fresh scan on the copy windows |
| E217/E229 pairs: 20 fine-tunes and 20 matched randoms | none: no VPD scan exists | E277 `items_<name>_s{0,1}{,_rand}.npz` (40 found) |
| E272 panel: 12 tasks x 3 seeds and a Gaussian | fresh scan, 24 states (seed 0 and Gaussian) | E277 `items_panel__*.npz` (48 found) |

E272's `eb.py` saved only K's size, so its VPD K must be found again. One derangement of E268's pairs (64) is used,
and only each task's seed-0 state and its Gaussian. This is a departure from E272's 256 pairs and 48 states, taken to
fit the debug queue. F, M and D come from E257's World (VPD), E277's `Built`/`Trained` worlds (neurons) and E272's
windows manifest (panel).

**Head (2,4)** (whip-lead, 2026-11-23): one variable at the head basis, on E62's 206 copy windows only (E280's
`head_edit.targets`), for copy and lookup s0-2. Three numbers: its strict kept share as E277 reads it, its own-space
share, and its flip count from `needed_neurons.scan` with E268's settings, pairs drawn within the copy windows.

## 4. Expected values (filed in predictions.json)

| item class | filed |
| --- | --- |
| do-nothing | every dd exactly 0; read set empty; pooled shares 1 by convention; every own space holds at least 0.99 of its write |
| rotation controls, before any trained read | W_i turned 30° toward a needed W_j: own 0.067, rotation 1; toward a non-needed W_k: own 0.067, rotation at most 0.5 (stops) |
| x2, x10 at neurons | each changed neuron's own share at least 1 − 1e-9; pooled own share at least 0.99 (stop on a miss) |
| x2, x10 at VPD | needed share at least the strict share (0.76); expected at least 0.9 |
| new_few, new_thin | needed share at most 0.5; rotation share within 0.1 of its null |
| randoms (built, E229 matched) | rotation share within 0.05 of the matched null; own share 0.2 to 0.8 at neurons |
| E62 | rival A (rotation): copy's needed share at least 0.2 above lookup's on the three-seed mean; rival B: within 0.2 |
| E229 pairs, panel | descriptive: each trained item's rotation share beside its matched random's |
| head (2,4) | strict and flips: copy above lookup on each seed; own share, lookup beside copy, against its null dim O / 768 (at most 0.17): rival A puts copy at 0.5 or more |

The rotation controls' own share is the change's: a column turned by θ at fixed norm changes by a vector whose part
along the old column is (1 − cos θ) / 2 of its squared norm. redteam-lead's cos² θ is the share of F's whole write,
a different quantity. The head expectations come from E277 on all of D: strict 0.095 to 0.141 for copy against 0.015 to 0.020 for lookup,
and 128 to 141 flips against 11 to 15, of 256 pairs.

## 5. Edge cases

- Rank 0: M's write is zero on every row, as for a neuron never above its bound. Its own space is empty, every share
  is 0, and the count is reported.
- Empty read set (no changed variable in K): pooled shares 1 by convention, as E_b; labelled "empty".
- Zero change for a variable: it adds 0 to every sum and has no per-variable share.
- Nothing leaves the own space (1 − own below 1e-6, as x2 at neurons): no rotation share, labelled so.
- K under 100 reads "too small to read" (whip-lead's floor). A needed space with null above 0.5 reads "too wide".

## 6. What it is blind to

Any change in when or how strongly a variable fires counts as kept. Overlap with M's other needed writes is read
against a random direction's null, but chance alignment beyond that null cannot be told from rotation. Only the top
1% of needed changed variables is read. Rows where a variable writes strongly dominate its share. Remainders read
near 1 by construction and are reported apart. K is sampled from 256 pairs, 64 for the VPD panel.

## 7. Cost

Cluster debug L40s, $0. One item without a scan needs two forwards over at most 256 windows and the subspace step for
at most 389 variables. E281's stage 0 did three VPD write passes and two featurizations in 16 s on an H100
(`e62_copy_windows/results/stage0_p0.json`). The subspace step adds Gram sums over d_out² per variable, so the plan
allows 1 min per item. The first item (x2 at VPD) is the rehearsal and its measured time replaces this estimate.
- 16 VPD and 104 neuron-basis items, at 1 min: about 2 h, as two debug jobs.
- Head (2,4): six neuron-basis scans of 12,316 variables x 256 pairs. E277's E62 items took 56 s each on all 512
  windows (`needed_neurons/results/items/e62__copy_s0.json`), so 6 min.
- VPD panel: 24 states x one derangement. E272 took 742 s per state at four derangements on an H100
  (`convergence_panel/results/wide/eb/*.json`), so 186 s at one, plus about 1 min of reading: about 4 min per state.
  One state is rehearsed on cluster debug; the other 23 go to one H100 pod at $4.40/h, about 1.5 h or $7
  (whip-lead approved the pod on 2026-11-23).
- Rotation controls: two extra items on x2's D, about 2 min.

Total: $0 on the cluster and about $7 for the panel pod, against the $15 cap.

## 8. Reused apparatus

- `stand_in.write_change` for the strict share; `needed_neurons` sites and `kstar_vpd.Site` tensors for the writes.
- `kstar_needed.featurized` and E257's World (VPD); `needed_neurons.Built` and `Trained` (neurons).
- The saved flips and K of E268, E275 and E277, unchanged.
- `needed.needed_set` with E268's settings for the VPD panel; `needed_neurons.scan` and E280's `targets` for the head.

## Not run

- E217/E229 pairs at VPD: no VPD scan of them exists, and finding K for 40 states is not in this cell's cap.
- The VPD panel's seeds 1 and 2: 24 more scans.
- Built rotation controls at VPD: turning a subcomponent's write changes the matrix, which the split then spreads
  over many subcomponents, so the control's answer is not known by construction there.
