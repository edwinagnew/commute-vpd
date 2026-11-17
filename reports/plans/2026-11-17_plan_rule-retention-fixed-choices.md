# Plan: causal-rule retention with the fixed choices (P1.25, E195)

*2026-11-17, definition-lead. Cell P1.25, 2 runs, due 2026-11-18. The definition is
`~/commute-data/scratch/definition-session/2026-11-17_causal-rule-retention.md`, read under its "Choices fixed for the test".
It is tested here, not adopted: it replaces nothing in `decisions.md` and gives no verdict on any fine-tune.*

## The user's request in the Terms' words

Use the draft's fixed choices, unchanged.
- **Level:** single MLP neurons. Heads plus whole MLP blocks are reported alongside, with no expectation filed.
- **Rule of a unit:** the unit's output, read on M's inputs to its layer for both networks.
- **Distance:** 1 − cos² between the two outputs, so a change of gain is free and the gain is reported.
- **Pairing:** the one-to-one re-pairing within a layer with the greatest total retention.
- **Reading:** comparative only.

Run 1 has two parts:
- **Code checks**, which must all read 1: do-nothing, the circuit turned up ×2 and ×10, and the circuit copied exactly onto fresh neurons.
- **A rebuilt item:** the circuit is zeroed and the later MLP layers are fine-tuned to recover the ×2 item's outputs.

Run 2 reads the five trained matched pairs on seeds 0 and 1.

## Design check

- **Goal.** This is part 1 of the paper, a working definition of reused against new (`GOALS.md`). It tests a candidate definition on edits whose machinery is known, then on the trained pairs.
- **Nearest tried.** E193 and E194 (P1.24) used the same model, circuit and windows. A fitted replacement there either copied a circuit neuron or missed the fit. This run changes two things:
  - the user fixed the distance and the pairing;
  - the hard case is now built by fine-tuning, so nothing has to be checked for copying.

  This is a step towards the goal, not a patch: the definition itself changed.
- **Decisions relied on.** The new `decisions.md` row of 2026-11-17, "P1.25 retention follow-up". It sets three things:
  - a set suffices at 0.9 of KL, as in P1.24;
  - heads are re-paired within a layer;
  - zeroed units stay zeroed in every test.
- **Definition.** No rule, band or bar is added. The reading is comparative.
- **What would change what we do.** If the rebuilt item reads about 1 (rival B), the neuron level cannot tell a rebuilt computation from a turned-up one. That is the draft's own "close to a weight comparison" limit, measured, and it goes to the user's gate before run 2 means anything. If it reads clearly below 1, run 2 is read.
- **Control first.** Stage 0 runs the four code checks alone and stops on any miss. Each check can fail on the pairing, the gain fit or the contrast search. The moved copy tests the re-pairing.

## Definitions in code (lead's choices; measurement, not definition)

- **Output of neuron i on M's inputs.** f_i(x) = a_i(x)·w_i. The cross term over neurons is G_ij·(w^F_i·w^M_j), where G_ij = Σ_x a^F_i(x)·a^M_j(x). From this:
  - cos²_ij = (G_ij·(w^F_i·w^M_j))² / (‖f^F_i‖²·‖f^M_j‖²);
  - retention r = cos² under the pairing;
  - gain g = ⟨f^F, f^M⟩ / ‖f^M‖², which keeps the sign.
- **Pairing.** Per layer, `scipy.optimize.linear_sum_assignment` maximises Σ cos² over F's non-zero neurons against M's.
  - A neuron of F that writes zero has no retention and stays zero.
  - A non-zero neuron of F with no partner reads 0.
  - Heads are paired the same way within a layer. Each MLP block pairs with itself.
- **Changed units.** Units whose output differs from their partner's at all: 1 − ‖f^F−f^M‖² ⁄ (‖f^F‖²+‖f^M‖²) < 1 − 1e-6. A unit with only its gain changed counts as changed.
- **Contrast.**
  - KL(F‖M) on copy24's 75 held-out windows.
  - A unit outside the set is put back to its partner's M rule, in F's slot. Zero units stay zero; unpaired units are removed.
  - Smallest sets come from backward elimination in 5 seeded orders, at the 0.9 bar.
  - Elimination drops chunks and halves a chunk when it fails, down to single units. This is needed because the rebuilt item changes about 6,144 neurons.
  - The reading is the mean r over each set, with its range across the orders.

## Run 1 (E195)

| Stage | Item | Expected reading | Stop |
|---|---|---|---|
| 0 | do-nothing | profile all 1, gain 1, "no change" | any r < 1 − 1e-6 |
| 0 | ×2, ×10 | circuit r = 1, gain 2 and 10; contrast retention 1 | any r or retention ≠ 1 |
| 0 | moved copy (C zeroed, copied onto 8 appended neurons) | fresh neurons pair with C, r = 1, gain 1; "no contrast" | same |
| 1 | rebuilt: fine-tune the MLP weights of layers 2–3 to KL(×2 ‖ F) on half Pile, half copy24 training windows | fit ratio KL(F‖×2)/KL(×2‖M) reported (no bar) | loss not falling in its first 200 steps |
| 2 | rebuilt reading | filed: retention below the turned-up items' 1 | none |

- **Fine-tune recipe.**
  - Settings: E194's fit recipe (Adam at its learning rate, warm-up, batch, half Pile and half copy24), 4,000 steps.
  - Frozen: everything except the c_fc and c_proj of layers 2 and 3.
  - Departure from E194: full MLP layers train instead of appended neurons, and there is no correlation penalty.
- **Reported:**
  - the contrast retention, its sets and their range;
  - the gains on the set;
  - the per-unit profile (r and gain for every unit, saved);
  - the side reading at heads plus blocks.
- **Cost.** One H100 for under 1 h, under $5.

## Run 2 (filed after run 1)

- **What is read.** The five trained matched pairs (copy/cipher, identity/random map, case, repeat, parity twin/memorised fact) on seeds 0 and 1, from E184's states. The contrast is each F against M on its own task windows.
- **Filed by the user.** In each pair the reuse member reads higher than its twin. The fact pair is reported and not counted.
- **Card.** Its card and predictions are filed before it starts, from the elimination time measured in run 1.

## Not run

- Decomposition components as units.
- Pairing across layers.
- Other circuits.
- A second fit seed of the rebuilt item.
