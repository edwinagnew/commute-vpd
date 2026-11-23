# Are the two saved readings of reuse admissible under the frozen definition?

*P1.53, due 2026-11-23, definition-lead. $0, code reading only: no run, no new wording. The definition is the user's,
frozen 2026-11-22: `theory/2026-11-22_definition_frozen.md` (copied unchanged from
`~/commute-data/scratch/definition-session/2026-11-22_frozen-definition.md`). This page supersedes the reason given in
`reports/analyses/2026-11-22_analysis_sec4-definition-justification.md` for treating the write share as the definition.*

We read the code of the two saved readings of reuse, the squared-cosine share and the write share, against the frozen
definition's admissibility condition. The write share compares the two models on the right inputs and reaches its top
exactly when every variable it looks at is kept, but it looks at the circuit found by mean ablation, not at the needed
variables the definition names, so neither reading is admissible as it stands. The write share becomes a reading of
the definition once it is computed over the needed set, or on a class of cases where the two sets agree, which is the
one change the paper needs to make before calling it one.

## What the frozen definition asks of a reading

A variable u is **kept** when there is one gain c > 0 with f_u^F(p) = c f_u^M(p) on every setting p of u's parents
that occurs in F's own runs on D. It is **needed** when transplanting its value alone between two inputs of D can
change the designated output; K is the set of needed variables and R the kept ones among them. A reading is
**admissible** on a declared class of cases when it reaches its stated top exactly when R = K, K empty included
(section 3). The write share is now a reading, not the definition: the 2026-11-22 `decisions.md` row "Option (b)
tests each changed part by write" is superseded on that point.

## The check

| question | squared-cosine share (E226, `commute.kstar_vpd.grades`; neurons and heads `commute.kstar.grades`) | write share E_b (`commute.stand_in.write_change`, `explained_share`) |
|---|---|---|
| same parent setting for both mechanisms? | yes: F's and M's write are computed on one input y | yes: one input y |
| settings = those that occur in F's runs on D? | **no**: every position of D in M's run and in F's run | yes: every position of D in F's own run |
| one gain across all settings? | **no**: cos² is per setting, so the gain may differ from setting to setting | yes: one least-squares gain per variable over all settings |
| gain positive only? | **no**: cos² ignores sign | yes: a gain of zero or below explains nothing |
| set it scores | K*, the smallest set that keeps the behavior under mean ablation of the rest | the changed members of the same K* |
| set the definition scores | K, the variables single-variable interchange shows are needed | same |
| top reached exactly when every scored variable is kept? | **no**, in both directions (below) | **yes** (below) |
| admissible? | **no** | **not as computed**: only on cases where K* and K give the same verdict |

**Why the parent settings match for both.** At the VPD featurization a variable is a subcomponent or a matrix's
remainder, and the code reads its mechanism on y, the matrix's input. The steps between the parent variables and y
(the layer norms, the attention mixing) have no weights that the fine-tune changed: `commute.kstar_vpd` stops the run
when any parameter outside the decomposed matrices differs (`undecomposed_differences`). So two mechanisms agree up to
a gain on every parent setting that occurs exactly when they agree on every y that occurs, and comparing on y is
comparing on P_u.

**Why the squared-cosine share fails.** Its grade s_u is 1 exactly when F's write is a nonzero multiple of M's at every
setting, each setting with its own multiple, sign free. So a variable whose write flips sign, or doubles on some inputs
and halves on others, reads 1 and is not kept. In the other direction, its settings include M's own run, so a kept
variable whose mechanism differs from M's only at settings M reaches and F does not reads below 1.

**Why the write share reaches its top exactly on kept variables.** Split F's write into the part along M's write at
the best gain g and a residual r. The explained part is (g - 1)² |M's write|², the change is that plus |r|², so the two
are equal exactly when r is zero on every setting and g > 0, that is, when the variable is kept. Where M writes zero
the explained part is 0, so the variable reaches the top only if F writes zero too. Unchanged variables are kept with
gain 1 and are left out of both sums, which leaves the top unaffected. So E_b = 1 exactly when every changed member of
K* is kept (and 1 by convention when none changed, which is what the frozen form gives when every scored variable is
unchanged).

**Why that is not yet admissible.** The frozen form's verdict is about K, the needed variables; E_b's is about K*. A
variable can be needed and outside K* (K* is one smallest passing set, found by a capped search: some saved searches
stopped at 2,000 evaluations), and a member of K* can be unneeded by single-variable interchange (three copies of a
signal read by their median, section 6). On such a case E_b can read 1 with R ≠ K, or below 1 with R = K.

**Two narrower limits, neither met by any saved item.** A graft (a variable M lacks) is not a site, so it adds 0 to
both of E_b's sums: a case whose only new variable is a needed graft would read 1. No saved E_b item has a graft in
K* (`n_grafts_in_K` is 0 on all 104 items of `experiments/rb_states/results/summary.json`). And the top is exact only
in exact arithmetic: a kept variable reads below 1 by rounding, so an endpoint test needs a bar from the measured
floor.

## What this means for the saved numbers

- **Squared-cosine share: not admissible.** Its numbers stay as descriptions only. It reads 0.996 on the thin new edit
  at VPD (E226), where the write share reads 0.019. FYI: the 0.98 that the frozen file's section 6 gives for this edit
  is wording A's soft-intervention reading of the same edit, not the squared-cosine share.
- **Write share: a candidate reading.** Every saved E_b (the built edits, the nine pairs, the trained states of E257,
  the SAE readings of E263) was computed over K*. Its order on the built edits (reuse 0.63 to 0.76, new 0.012 to
  0.019, random 0.011) stands as a description of the K* version.
- **The built gain edits are not full reuse at VPD under the exact form.** They multiply eight neurons' output weights,
  and no VPD subcomponent's write is then a pure multiple of M's, so they read 0.76, not 1. Under the exact form at
  this featurization they are not R = K cases. Their "reuse" label holds at the neuron featurization, where each
  rescaled neuron is kept with one gain. Section 4 property 7 ("relative to the featurization") covers this, and the
  paper should say it.
- **The SAE readings (E263)** use the same `write_change` path through `commute.sae_feat`'s sites. I did not re-read
  that adapter. If its sites follow the interface (settings from F's own run), the same verdict holds there.

## 2026-11-23: the write share over the needed set (P1.54, E268)

E_b over K is now computed for 73 saved states: the seven built edits, 21 repeat-task RL and SFT states and 45
procedural-task states. K was found by single-variable interchange on 256 sampled pairs of D per state, so it is a lower
bound on the definition's K. Page: `reports/analyses/2026-11-23_analysis_e268-needed-set.md`.

- **Built edits:** the order stands. Gain edits read 0.761 over K, the rebuilt edit 0.623, new and random edits 0.011
  to 0.019, each within 0.007 of its value over K\*. Here K holds nearly every variable, so the agreement is close to
  automatic.
- **Trained states:** K is mostly much smaller than K\*. E_b barely moves for SFT, guarded RL and random updates. On
  repeat-task RL it drops by up to 0.062, so the RL to SFT ratio in the KL-matched run falls from 5.1 to 2.6.
- **One empty K:** a collapsed addition RL state, where no single swap changed the output. The frozen form gives full
  reuse vacuously there, and E_b over K reads 1 by convention; it is not a reading.

So the "not as computed" verdict above now has its computed version. Over the sampled K, E_b reaches 1 only when every
changed needed variable is kept; whether K\* is an acceptable stand-in is answered per class of case: on the built
edits yes, on repeat-task RL no.

## 2026-11-23: the squared-cosine share with one gain, over the needed set (P1.55, E274)

The three faults above are fixed from saved files, at $0. Each variable now gets one positive gain over every setting
of F's own run, and the share is read over the changed members of K. Page:
`reports/analyses/2026-11-23_analysis_e274-cos-admissible.md`.

- **Admissible on the 73 states.** It reaches 1 only on the collapsed state whose K is empty. Every other state reads
  more than 1e-11 below 1, and each has a changed needed variable that is not kept, gain x2 and x10 included.
- **Not informative.** Per variable, the shortfall from 1 is the write share's shortfall times the change size, the
  change as a fraction of F's whole write. A fine-tune's changes are small, so the 65 trained and random states read
  0.9935 to 0.9999. The rank correlation with E_b over K is -0.58, and gain x10 reads lowest of all, at 0.347.

So the squared-cosine row of the table above now reads "admissible after the fix, but it measures change size". The
write share stays the reading the paper uses.

- E_b over K at saturation: the sampled K still grew between 192 and 256 pairs on 71 of 73 states (P1.54).
- A re-read of `commute.sae_feat`'s site adapter.
- Wording A's soft-intervention reading and the search-based R_b: both superseded, not checked here.
- A measured rounding floor for E_b's top.
- A bf16/fp32 rounding floor for the one-gain squared-cosine share; its top bar is the fp64 bound (P1.55).

## Sources

`src/commute/kstar_vpd.py` (docstring, `grades`, `undecomposed_differences`), `src/commute/kstar.py` (`_add`, `grades`),
`src/commute/stand_in.py` (`write_change`, `explained`, `explained_share`), `experiments/option_b_rescore/results/summary.json`
(`.items[].E_b`, `.items.new_thin.R_A`), `experiments/kstar_vpd/results/summary.json` (`.per_item.new_thin.R_read`),
`experiments/rb_states/results/summary.json` (`n_grafts_in_K`).
