# Appendix to `plan.md`: rho as the note defines it, on saved items

*2026-11-18, definition-lead, phase A. The definitions in code, the stage layout, what can go wrong, the measurement
choices made here, and the places where the note and the code disagree.*

## A. Where the note and E195/E196's code differ (checked against the code)

`src/commute/rule_retention_fixed.py` (RF) and `src/commute/rule_retention.py` (R):

1. **r_u.** RF scores a unit by one aggregate cos²: the cross term squared over the product of the two norms, each
   summed over windows, positions and dimensions. For a neuron this is the activation patterns' Cauchy-Schwarz ratio
   times the cos² of the output columns (`RF.profile`, `RF.pair`). The note's r_u is 1 − mean_v d per setting.
2. **Re-pairing.** RF pairs F's units with M's within a layer by the Hungarian method (`linear_sum_assignment`), on
   every item. The note allows re-pairing only where it is declared as an exact symmetry. Here the pairing is the
   identity.
3. **Settings.** RF reads every unit on M's inputs only (`R.m_inputs`). The note's star choice pools M's and F's.
4. **Level.** At RF's level A only neurons are units. Attention, embeddings and the unembedding stay F's in every
   hybrid, so F with nothing kept is not M: E196's `closure_none_kept` read 0.33-0.68 (`items/*.json`). Level B
   (heads after o_proj plus whole MLP blocks) came close, at about 1e-8 for the old tasks. The twin and fact read
   0.0006-0.03 there, because their embeddings and norms moved. `R.heads` hard-codes `N_HEAD = 6`, so a grafted
   seventh head (`commute.graft.Joined`, o_proj 768 × 896) breaks it.
5. **Changed.** RF calls a unit changed when its aggregate `same` < 1 − 1e-6, which a gain change also trips. Here a
   part is changed when its cell's weights differ.
6. **ε_F.** At the identity alignment, the hybrid with nothing swapped is F. RF's `gate` reads that, and it is 0 by
   construction, so there is nothing to measure.

## B. Definition questions for the user (wording, not measurement)

1. **The per-setting distance is blind at a neuron.** A neuron's write at setting v is a(v)·w. Then
   cos²(a_F(v) w_F, a_M(v) w_M) = cos²(w_F, w_M) whenever both activations are non-zero. GELU is zero only at 0, so
   that is every setting in practice. So the note's r_u at a neuron is the cos² of its output column. The input row
   and the norm, which set what the neuron responds to, never enter. If a neuron's value is taken to be its scalar
   activation, cos² of two non-zero scalars is 1 and every neuron reads r = 1. RF's aggregate cos² sees the response
   pattern, because it sums before it divides. The options, in plain words: (a) score agreement over all settings at
   once (today's code); (b) keep the per-setting score but use a distance that also sees size, such as
   |f_F − f_M|² / (|f_F|² + |f_M|²) (R's d before E195); (c) keep the note as written and accept that neurons are read by
   their output weights alone. The run reads (c) as the main reading, (a) as alternative (i). (b) could be added
   at no search cost if the user wants it.
2. **Weighting in ρ.** ρ is a plain mean over K. A head, which carries a share of attention, counts as much as one
   of thousands of neurons. So the heads' readings barely move ρ, even where they carry the change. That is part of
   the wording ("the mean rule retention of the parts F needs"), so it is listed rather than changed.
3. **A part F has removed, once in K.** The note says a part M lacks keeps nothing. It does not say what a part F
   lacks keeps when choice c5's alternative lets it into K. Here it reads 0, from d(0, x) = 1.
4. **A moved copy reads new.** Under one identity alignment, a circuit copied exactly into fresh units reads 0.
   E195's moved copy is one case: C zeroed, copied onto 8 appended neurons. Re-pairing is allowed only as a declared
   exact symmetry, and adding units is not a symmetry. This follows from the rule without a run. The user may want
   to confirm it is intended.

## C. Measurement choices made here (the lead's call; star taken where the note has one)

- The level's parts are every neuron, every head, the embedding (wte) and the unembedding (ln_f with lm_head). The
  67M ties wte to lm_head, so a fine-tune that moves one moves both, and both parts are then changed. A norm belongs
  to the parts that read it: rms_1 to the heads, rms_2 to the neurons. RF already swaps them that way.
- A part's value is its write: a neuron's a·w, a head's output after o_proj, the embedding's row, the unembedding's
  logits. In(u) is the residual feeding the part. For a head, that is every position up to its own.
- V_u is pooled: M's and F's runs on the item's read windows, every position, with each setting weighted equally.
  The behaviour's KL is read where E195, E196 and E206 read it: the last position for the pairs, every position for
  E195's windows and for the grafts.
- Zero writes: d(0, 0) = 0 and d(0, x) = 1.
- Under (ii), activation patching, only changed parts outside K are patched with M's run values. Unchanged parts run
  their shared mechanism on the hybrid's input; patching them too would force downstream unchanged parts into K.
  With every changed part patched, F_∅ is M by induction over layers.
- The slotted control writes E206's 256 graft neurons into the 256 layer-3 slots whose M writes are smallest on
  E206's read windows, so the least of M is lost. There is no training. Its accuracy is not read: the item is new by
  construction whatever it scores.
- Search: E196's chunked backward elimination (chunk 256, 5 orders, order_seed 0). This gives an upper bound on the
  size of K (star c4).

## D. Definitions in code (phase B, to write)

One new module, `src/commute/rho.py`, built on `commute.rule_retention` (R) for the layer pieces. RF is not changed:
E195 and E196's frozen scripts call it.
- `parts(F, M)`: the changed parts, with "M lacks" and "F lacks" flags. Heads are counted from each layer's o_proj
  width, not from `N_HEAD`.
- `write(N, part, residual)`: the part's write on a residual, for N ∈ {M, F}.
- `r_per_setting(F, M, toks, settings="pooled"|"M")`: per part, 1 − mean of (1 − cos²). Neurons are checked against
  the column cos² (identity_tol).
- `r_aggregate(...)`: RF's cos² with the identity pairing.
- `forward(F, M, toks, keep: set, how="replace"|"patch", restore_zeroed=False)`: F_K. Under "patch", M's per-part
  writes are cached once per item.
- `search`: RF.eliminate, called unchanged.

`experiments/rho_saved_items/run.py` takes about 300 lines: the Config from `predictions.json` "settings" via
`config_from_settings`, the stages below, one JSON per item, and `results/large/parts/<item>.npz`.

## E. Stages and pods

- Stage 0 runs on every pod, first and alone: do-nothing, x2 and x10 on copy24's 512 windows. It exits 1 on a miss.
- Pod 0 adds the slotted graft, the two appended grafts on E206's windows, and E195 rebuilt (main, (i), (ii),
  (iii), (iv)).
- Pods 1-10 each take one pair at one seed, reuse member then twin: E196's gap check, then main and (ii). Each item
  writes its JSON as it ends and is skipped on a rerun.
- Inputs per pod, from a grep of the loads in phase B: the model (256 MB); artifacts/sets train_0_32768,
  train_32768_8192 and val_0_2000 (170 MB); the tokenizer; the two items' deltas (113 MB each) or full states
  (268 MB each); E184's item JSONs; and built_control_sum/results/amend1_stage0.json. Pod 0 also needs
  rebuilt_mlp.pt (38 MB), E206's neurons.pt and last_lowest.pt (1.6 MB each), the open_pair copy24 windows and
  E206's data rows.
- Every `results/large` reaches a pod as a dangling symlink, so stage inputs by hard link (root `CLAUDE.md`).

## F. What can go wrong, and the guard

- **The heads' per-setting reading is slow**, because it holds every head's write at every position for M and F.
  That is 512 × 8 × 6 × 768 doubles per layer per network, about 150 MB. Read in batches; profile_batch stays 512
  for the pairs and is lowered for E206's 32-token windows.
- **A seventh head (an appended graft) breaks R.heads.** It is read from the o_proj width (D), and stage 0's
  appended-head item tests it.
- **F_∅ ≠ M from a part left out of the level.** The M gate on every item, including the twin and fact, whose
  embeddings moved.
- **The search size grows**, because K must now carry attention's share too. The first order's evaluations and
  seconds are logged with the time they imply for the item, as E196 did. If that time runs past 45 min an item, the runner reports
  before the second item.
- **Gaps fall under the order noise.** The main reading at neurons may compress every gap. R1 prints the main
  reading's own largest order sd beside each seed's call. No band is applied (`predictions.json`, R1).
