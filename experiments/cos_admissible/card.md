cell: P1.55
rival_a: Taking one positive gain per variable over the settings of F's own run removes what made the squared-cosine share inadmissible and leaves a reading that tracks the write share, which predicts it reads 1 only where R = K (the do-nothing, the empty-K state), below 1 on the gain edits x2 and x10, and the new and random edits at least 0.1 below the gain edits, the built-edit order of E_b over K repeated
rival_b: The fix makes the share admissible only in exact arithmetic: per variable its shortfall from 1 is the write share's shortfall times the change size dd / |F|^2, tiny for a fine-tune, which predicts every state within about 0.01 of 1, the gain edits below 1 but the new and random edits within 0.1 of them, so the built-edit order is not separated at the smallest gap
rung: parameter
smallest_gap: 0.1
spread: 1.6e-05
n: 73
design: paired
unit: saved states (each read by the squared-cosine share and by the write share, over the same K)
spread_source: experiments/kstar_vpd/results/summary.json
spread_of: proxy: E226's per-setting squared-cosine share R_read, sd over the 8 seeds of c3/cipher, the same kind of squared-cosine share on [0, 1] over a circuit of the same featurization
floor: top_tol 1e-11 (fp64 summation bound; stage 0 measured this reading's own arithmetic floor at 6.7e-16 on synthetic kept variables): a state is at the top only when every changed needed variable reads within 1e-11 of 1; both rivals expect no built edit at the top, since E268 found no changed needed variable kept on any built edit
stop: the new and random edits read within 0.1 of the gain edits (rival b): the squared-cosine share is reported in the paper as admissible but uninformative at fine-tune scale, beside E_b over K, and no further work on it is proposed
sketch: experiments/cos_admissible/figures/sketch.png
rehearsal: cluster only: CPU from saved files, no GPU and no pod (`fleet/cpu-run`, seconds); stage 0 (edge cases, synthetic kept variables, the do-nothing on x2's saved path) runs first and stops the run on any miss

Design check (`.claude/skills/design/SKILL.md`), 2026-11-22, definition-lead writer:
- **Goal served.** GOALS.md's reused-against-new definition: gives the paper a second reading checked against the
  frozen definition's admissibility condition (section 3), beside the write share. Fills open cell P1.55.
- **Nearest tried.** E226 (squared-cosine share over K\*, per setting) and E268 (E_b over K). New here: one gain per
  variable, sign checked, F's settings only, over K; P1.53's page names exactly these three faults.
- **Patch or step.** A patch on E226's reading, asked by the cell; it settles whether that reading can stay in §4.
- **Decisions relied on.** The frozen definition (sections 1-2, unchanged); E268's K and kept floor; no departure.
- **Adopted definition, unchanged?** Yes: no rule or band added to a verdict; admissibility is checked against the
  frozen form's own R = K, read from the same arrays at E268's floor.
- **What would change what we do.** Rival b: the paper keeps the squared cosine only as an admissible-but-saturated
  reading and leads with E_b. Rival a: the paper can report it as a second admissible reading that agrees.
- **Control first.** Stage 0, alone and first: hand-built edge cases, synthetic kept variables (num = c den,
  dd = (c - 1)^2 den), and x2's 35,353 unchanged variables reading exactly 1; any miss stops the run.
- **Numbers and expected values.** Per state: cos2 over K (do-nothing exactly 1; gain edits below 1 - 1e-11; random
  not filed), E_b over K recomputed (equal to E268's within 1e-11 relative), top vs R = K (equal on every state),
  n_disagree (0), median dd / |F|^2 (the compression). Over states: Spearman with E_b over K, median gap, built order.
- **Saved for audit.** results/summary.json; the per-variable values are recomputed in seconds from the saved sums.

F's own parent settings: met by the saved arrays. num, den and dd come from `commute.stand_in.write_change(F.net,
...)` (experiments/thin_edit_diagnosis/run.py and experiments/rb_states/run.py), which hooks the input y of each
decomposed matrix in F's own unmodified run on every position of D and evaluates both mechanisms on that y; P1.53's
page shows comparing on y is comparing on the parents' settings at VPD. No extra pass is needed.

Departure from the lead's brief, flagged: the brief expected x2 and x10 to read 1 exactly and the random edits near
0. At VPD the gain edits are not R = K cases (P1.53's page; E268: 0 of 3,585 changed needed variables kept), so an
admissible reading must read below 1 there, and that is what is filed. The random edits were left unfiled: the
squared cosine of the whole write is expected near 1 for small edits (definition-lead's reply to writeup-lead,
2026-11-22), not near 0.
