# Decided: the crosscoder reading (E203) runs one seed per side, with its planted-control miss stated beside

*2026-11-18, compare-lead. Already decided by the user in two dialogs; this card is the record for the log. Cell P2.31.
Plan amendment: `reports/plans/2026-11-17_plan_p2-31-run2.md`, section "Amendment, 2026-11-18".*

## The answer

The crosscoder reading cannot yet tell reuse from new. On the first four crosscoders, the gap between the reuse side and
the new side was 0.03 by the authors' rule, inside the 0.05 that a reseed alone moves it. The check that the method can
find a planted change also missed. The user chose to go on with one seed per side on the nine built and nine
size-matched pairs, with both facts stated beside every number.

## What was measured

E203 trains a crosscoder between the original 67M model and a fine-tune of it: one dictionary of latents, each with a
decoder direction in both models. It is a BatchTopK crosscoder with latent scaling (Minder et al. 2025, whose method
comes from Lindsey et al. 2024). A latent is called fine-tune-specific when it is mostly decoded in the fine-tune and
latent scaling shows it explains the fine-tune's activations better than the original's. The quantity read is d, the
crosscoder's score for how much of the fine-tune is already in the original, compared for the reuse side against the
new side.

- **Base against itself passed.** With the original model on both sides there were no fine-tune-only latents
  (`experiments/crosscoder/results/run2/stage0a.json`).
- **The planted control missed.** A direction planted in the fine-tune was found, as the top-ranked latent, but latent
  scaling called it an artefact, not fine-tune-specific. Its scaling score read 0.244 against the filed bar of 0.2
  (`experiments/crosscoder/results/run2/stage0b.json`).

## Decision 1 (user, dialog, 2026-11-18, about 09:47): stage 1 only

Run stage 1 alone, four crosscoders. Report d by the authors' rule, and again with the artefact class counted as
fine-tune-specific, with the planted miss stated beside both.

## What stage 1 read

From `experiments/crosscoder/results/run2/stage1_summary.json`:

- By the authors' rule the reuse-minus-new gap in d is 0.03, inside the 0.05 a reseed moves it. No latent on any side
  was called fine-tune-specific by that rule.
- With artefacts counted as fine-tune-specific the gap is 0.22, because the new side has far more artefact latents.
- A random update of the same kind reads highest, 0.55, under both forms. So d does not order reuse above new the way
  the working definition needs: a random change looks the most shared.

## Decision 2 (user, dialog, 2026-11-18, about 11:15): one seed per side

Options put to the user:

1. **One seed per side** on the nine built and nine size-matched pairs, about $150 estimated, about 2 h. Recommended.
2. **The full seed set**, about $426, about 6 h.
3. **Stop** here with stage 1.

The user picked option 1. That is 25 sides, of which 23 still had to train. It now runs as stage 2 on 13 pods under a
$130 cap. With one seed per side there is no per-pair seed spread, so the stage-1 reseed difference is the only noise
reading the stage-2 numbers can be set against.

## Not run

The full seed set; the other two crosscoder families (Diff-SAE, Delta-Crosscoder) the plan skips; a fresh review of the
six-line selection change, which touched no measurement code.

## Links

Plan amendment `reports/plans/2026-11-17_plan_p2-31-run2.md`; results `experiments/crosscoder/results/run2/stage0a.json`,
`stage0b.json`, `stage1_summary.json`.
