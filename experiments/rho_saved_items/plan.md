# Plan: rho as the note defines it, on saved items

*2026-11-18, definition-lead (phase A: plan, card, predictions; no code yet). Cell P1.30 as proposed (the user's yes,
2026-11-18: "yes, write the card and cell. in general, test out the proposed measurements of the new definition in
definition-lead/theory/2026-11-18_definition_causal_abstraction.md as much as you can"). Detail: `appendix.md`. Card:
`card.md`. Filed rules: `results/predictions.json`. Experiment code unclaimed until the lead runs `fleet/ecode`.*

## Design check

- **Goal.** `GOALS.md`, working definition of reuse (2026-11-18). The paper's first part needs the definition read
  as written. Today's readings (E195, E196) use the code's version, which differs from the note in four places.
- **Nearest tried.** E195/E196 (P1.25) read the code's aggregate r at neurons, with attention left as F's. They found
  every fine-tune near 1, with gaps that follow update size. E197-E206 read the held model's S, a cheap reading. This
  run reads the note's own version: r per setting, every path variable aligned, pooled settings. It also reports
  which open choice moves a reading.
- **Patch or step.** A step: the first reading of the adopted wording as written. It also checks note against code.
- **Decisions.** Relies on the two 2026-11-18 definition rows. No departure: it uses the starred choices and a
  declared level.
- **Definition unchanged.** Yes. τ, d and the level are as the note declares them. No rule, band or label, and no
  verdict on any fine-tune.
- **What changes next.** Rival B (gaps follow update size) sends the per-setting distance question to the user before
  any further ρ run (below). Rival A makes size-matched pairs (E187's) the next test. The one-change table shows which
  open (c) choices matter.
- **Control first.** Stage 0, alone, stops on a miss: do-nothing, x2/x10, a slotted graft and two appended grafts.
- **Early guards.** E196's gap check; the hybrid gate; a new M gate (F with every changed part swapped must give M's
  logits); the first order's evaluation count and seconds, logged with the time it implies for the item.
- **Saved.** Per item: every part's r under main, (i) and (iii), and each order's K as masks, in `results/large/`.

## What runs

**Main reading (the note, starred).** The parts are every MLP neuron, every attention head, the embedding and the
unembedding, each read through the identity alignment. With every part aligned, F with every part swapped is M. The
rule score is r_u = 1 − mean_v (1 − cos²) of the two writes at each setting v. The settings V_u come from M's and F's
runs on the item's windows, every position, each weighted equally. A part M lacks reads 0. F_K is made by mechanism
replacement. Zeroed parts stay absent. K comes from E196's chunked backward elimination, 5 orders, τ = 0.9.
**Beside it, one change each:**
(i) the code's aggregate r on the same K, with E196's own number quoted;
(ii) activation patching for F_K, with K searched again;
(iii) M-only settings, on the same K;
(iv) zeroed parts restored, on E195 rebuilt only, with K searched again.

| stage | items | numbers (expected) |
|---|---|---|
| 0, alone, stop on a miss | do-nothing; C x2, x10; E206's 256 neurons written into M's 256 quietest layer-3 slots; E206's appended neurons and head | do-nothing "no change"; x2/x10 ρ = 1 ± 1e-6 under every reading, K ⊆ C; slotted ρ ≤ 0.05 (about 0.0013 expected); appended ρ = 0 by rule; nothing kept gives M (M gate ≤ 1e-3) |
| 1 | E196's 5 pairs × seeds 0, 1 | per item: ρ and \|K\| for main, (i), (ii), (iii). ρ_main near 1; neurons read cos² of their output columns (exact). (iii) = main on neurons; (i) ≤ main on neurons. **R1**, below |
| 2 | E195 rebuilt | main ρ ≥ E195's 0.9715; nothing kept −1.11 under main, 0 under (iv); ρ_(iv) measured |

**The separating question (R1, filed).** Does the note's version separate the reuse member from its twin by more than
update size does? The key-table pairs moved about as much as their reuse members (ratio 1.05-1.13). The repeat twins
moved 1.4-1.6 times as much. Rival A says the new tables sit in the heads, which the level now reads. It predicts the
identity/random-map gap above the repeat gap at both seeds. E196's level B already read it so. Rival B says that at a
neuron, per-setting cos² equals the cos² of its output column, so ρ follows size. It predicts the gaps ranked case >
repeat > key tables at both seeds. The graft question (R2) is a check, not a test. E206's appended neurons read 0 by
the rule "a part M lacks keeps nothing", against the held model's 0.954. The slotted graft is the real low control.

## Cost and placement

E196 took 224-622 s per search on one H100; here two searches per item, about 22 items, so 5-8 H100-hours. The
heaviest item, twin/fact, needs about 21 min on an H100, so it would not fit the 30-minute debug limit on an L40. Use
11 H100 SXM pods in parallel: one per pair and seed, running stage 0 then two items, and one running stage 0, the
grafts and E195 rebuilt. That is about 1 h of wall time and $30-50. Each pod uploads 0.7-1.1 GB of inputs (the model
256 MB, data sets 170 MB, two deltas at 113 MB or two full states at 268 MB), 70-110 s at 10 MB/s. It is not gated.
One run; a second only if a stage-0 fix is needed.

## For the user (definition questions; detail in the appendix)

1. **Per-setting 1 − cos² cannot see a neuron change what it responds to.** A neuron writes its activation times a
   fixed column. The cos² of two such writes at one setting is the cos² of the two columns, whatever the activations.
   So under the note's formula, a neuron's rule is a comparison of its output column alone. If the neuron's value is
   instead its activation, every neuron reads 1. E195/E196's code took one cos² over all settings together, which does
   see the activation pattern. Which did the user mean?
2. **Every part counts the same in ρ.** One head weighs as much as one neuron, and K holds thousands of neurons.
3. **A part that F has removed and that sits in K** (choice c5's alternative) has no reading in the note; we read 0.
4. **An exact copy of a circuit into new units reads 0**, as does E195's moved copy: by the rule, a part M lacks
   keeps nothing, and the identity alignment never pairs a new unit with the old one.

## Not run

E201's graft (the same kind as E206's appended head); built_control and share_built_kind (they add no separating
question, and their windows need their own builders); choice c2, a fitted alignment, and c6, distributed variables
(the note says no reading uses them); ε_F, which is 0 by construction at the identity alignment; E187's size-matched
pairs (no cell).
