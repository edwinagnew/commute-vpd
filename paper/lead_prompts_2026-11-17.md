# Draft task prompts for the paper's gaps (writeup-lead, 2026-11-17)

For `~/commute-agents/fleet/start <role> "..."`. Each needs its cell from the user (runs and due proposed below).

## 1. compare-lead: model-diffing baseline (proposed cell: 2 runs, due 11-19)
Add a model-diffing baseline to the part 2 scorecard: a crosscoder between the base 67M and each fine-tune
(BatchTopK crosscoder with latent scaling, Minder et al. 2025, to drop fine-tune-only latents that are sparsity
artefacts; Lindsey et al. 2024 as the source of the method), on the nine built pairs and the size-matched pairs, the
same seeds the share used. Reading: the fraction of the fine-tune's activation change carried by latents the base also
has (shared) versus fine-tune-only latents. Expected: the reuse side higher. Control first: base against itself (no
fine-tune-only latents), then one pair. Report it as one more row of the scorecard figure, beside the share, on the
same axes. Run /design first; state each departure from the cited recipe.

## 2. regating-lead: the share on the hand-made edits (proposed cell: 1 run, due 11-19)
Compute the re-gating share on E137's hand-made edits (four pairs where we built in whether the change re-weights
existing pieces or rewires them), and on E139's fine-tunes that imitate them. This is the share's only positive
control where the kind of change is known by construction. Expected: the re-weighted edit reads higher than the
rewired one in each pair. The identity edit (does nothing) first. No new training: E137/E139 states and the existing
share code. One figure: share per edit, by built kind, with the trained pairs' range beside it.

## 3. compare-lead, after 1: pairs where the prior and steering disagree (proposed cell: 2 runs, due 11-20)
E151 (P2.17) could not build pairs where the base model's prior on the answer and the steering reading point opposite
ways: the common-word cipher's best prior was 2.32e-4 against the 2.35e-4 needed. Find a different construction
(for example a reuse task with a low-prior answer, or a new task whose answer already has a high prior) and run the
share, prior and steering on it. Decides whether the share is more than the prior. Run /design first, with
plans/2026-11-11_plan_p2-17-disagreeing-pairs.md read; if run 1 still cannot build a pair, close the cell with that
as the reading.
