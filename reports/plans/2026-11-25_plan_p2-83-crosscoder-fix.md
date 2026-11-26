# P2.83 (E311): the crosscoder's planted control, diagnosed, and the crosscoder read on Fig 4's 48 states

Cell P2.83 (baseline-lead, 3 runs, due 2026-11-26): "Crosscoder fixed on Fig 4's 48 states: planted control passes
under the authors' rule". Card `2026-11-25_plan_p2-83-crosscoder-fix-card.md`; appendix (departures table, gates,
inputs) `2026-11-25_plan_p2-83-crosscoder-fix_appendix.md`. Code `experiments/crosscoder_fig4/`, shared steps
`src/commute/crosscoder_sides.py`. No bar changes: the authors' ν cut-offs (ν_ε 0.2, ν_r 0.5) and our 0.5 specific bar
stay as E203 filed them.

**Plain words.** E203's crosscoder (Minder et al.'s method, their code and settings) missed its planted control: a
direction added only to the fine-tuned side was not called fine-tune-specific. The saved crosscoder shows why, for $0.
The plant was always added at the answer position of a task window, where existing base features fire. The crosscoder
did not give it a latent of its own; it added it to the fine-tune half of the latents that already fire there. By the
authors' own rule such a latent is shared, so the miss is the control's design, not a bug or an undertrained
crosscoder. The fix is a plant that does not ride on a base feature; E212's spread plant found it cleanly under the same
recipe. A second problem, found by the code review: the authors drop any latent whose base strength comes out negative,
and for a latent that exists only in the fine-tune that strength is zero plus noise, so the drop is a coin toss (it
turned E212's 0.766 into 0). Every reading is therefore scored twice: the literal rule, and a floored rule that drops a
latent only when its base strength is clearly below zero. Then the crosscoder is read on all 48 states of Fig 4.

## Diagnosis ($0, from saved files)

Scratch script `~/commute-data/scratch/baseline-lead/cc-fix/geom.py` on E203's saved decoder
(`experiments/crosscoder/results/large/run2/planted.pt`) and `results/run2/stage0b.json`:

| quantity (lead latent 8569, carries 0.65 of the planted change) | value |
|---|---|
| cosine of (fine-tune decoder − base decoder) with the planted direction | 0.99999 |
| cosine of the base decoder with the planted direction | −0.002 |
| base decoder norm / fine-tune decoder norm | 0.496 (relative norm 0.75) |
| ⟨base decoder, fine-tune decoder⟩ / ‖fine-tune decoder‖² | 0.2440 |
| ν_ε as saved (cut-off 0.2) | 0.2439 |
| ν_r as saved (cut-off 0.5) | 0.089 |

The latent is "answer-position feature" in the base half and "the same feature plus the plant" in the fine-tune half.
Its ν_ε is exactly its own base decoder's overlap with its fine-tune decoder: the error term adds nothing. The second
artefact (latent 20353, 0.19 of the change) and two shared-band latents have the same shape (difference along the plant
at cosine ≥ 0.9997). Minder et al. App. D (p. 25) predicts this: BatchTopK optimises L0, so two directions that always
fire on the same tokens are cheaper as one latent than as two. Ruled out: a latent-scaling bug (our targets match the
toolkit's `load_base_error`, `latent_scaling/utils.py:122-142`, which adds the latent's own base contribution back;
the identity above holds to 4 decimals); a missing recipe step (E203 used the authors' pinned trainer and every
setting, appendix A); undertraining as the main cause (ν_ε 0.320, 0.323, 0.293, 0.244 at 25M, 50M, 100M, 200M tokens:
falling, but the merged latent persists; the long arm below measures where it goes). E212's plant, added every 35th
position across six task formats, was isolated under the same recipe: lead relative norm 0.93, ν_ε 0.021, ν_r −0.0035,
part 0.766 when scored on 2026-11-18. Rescored with App. E.2's exclusion of negative base β (P2.70,
`negbeta_rescore.json` `.runs["E212"]["planted.L1"].new.parts.ft_specific`), it reads 0: its base β^r is zero with a
negative sign.

## The exclusion of negative base β: literal and floored (gate-5 review MUST 1; whip-lead 2026-11-25)

Minder et al. App. E.2 drop latents with a negative base β and report "None" such latents in BatchTopK (p. 26), so on
Gemma their filter never bites. Here it decides the gate: a fine-tune-only latent has base β ≈ 0 with a random sign.
Every crosscoder is scored under both rules, side by side in every JSON, log line and the figure:
- **Literal** (App. E.2 as applied since P2.70): drop when β^ε_base < 0 or β^r_base < 0.
- **Floored** (`crosscoder.fold_floor`): drop only when a base β is below −3·σ̂, each β on its own, both must clear.
  σ̂ is measured in the same job: the 100k latent-scaling tokens split into K = 10 folds of whole windows (a Pile row
  or task window wholly in one fold); the full-sample β is about the mean of the fold β's, so σ̂ = sd(fold β) / √10
  (9 degrees of freedom). For a latent whose true base β is 0, P(β < −3σ̂) is the t(9) tail, 0.0075 per β, at most
  0.015 per latent; a two-half split would give 1 degree of freedom and 0.10 (redteam review, MUST 2). σ̂ = 0, or a
  fold where the latent fires on fewer than 10 tokens (σ̂ undefined), gives floor 0: the literal rule for that latent,
  counted in `beta_floor`. A β exactly at −3σ̂ is kept. Edge cases: `tests/test_crosscoder_floor.py`. Cost: one extra
  pass over the 100k tokens per score, seconds against ~2 min.
- The planted gate (`gate_rule: floored`, whip-lead) reads the floored rule. **Wording:** a pass under the floored rule
  alone is the cell's second done_when branch: "missed under the published rule because its sign filter acts on
  noise; passes with the filter floored". It is never written as "passes under the published rule".
- The lead record holds all four β (ε/r, base/fine-tune), and under the floored rule the two base β's σ̂ and floor.

## Run 1: stage 0, five H100 pods at once (control first, alone)

| pod | stage | expected (filed, `results/predictions.json`) | early stop (code) | time, cost |
|---|---|---|---|---|
| a | `0a` self (base vs itself) | relative norms within 0.001 of 0.5, 0 specific latents, FVE ~0.93, dead 0 | batch FVE ≥ 0 at step 1,000; projected ≤ 75 min; FVE ≥ 0.5 and dead ≤ 0.3 at the first snapshot | 50 min (E203 0a 50.2), $3.7 |
| b | `0b random --seed 0` content-independent plant (1 in 80 positions, E203's count) | floored: PASS, part ~0.77, ν_ε ~0.02; literal: lean MISS (E212: 0.766 → 0) | same | 50 min (E203 0b 49.7), $3.7 |
| c | `0b random --seed 1` | as b | same | 50 min, $3.7 |
| d | `0b answer_half --seed 0` E203's answer position on a seeded half of windows | floored: lean PASS, part 0.5-0.8; literal: as b | same | 50 min, $3.7 |
| e | `0b answer --seed 0 --long` E203's plant, 8 epochs (800M tokens), diagnostic | at 200M: reproduces E203 (ν_ε 0.24 ± 0.03, part 0); lean ν_ε < 0.2 by 800M | same, projected ≤ 260 min | 200 min (30.7 ms/step all-in from E203 × 390,625 steps), $14.7 |

Every variant is read as E203 read it: the score windows, planted at each window's answer position. Pass rule (E203's
bars, unchanged): the lead latent is among the top 1,667 by relative norm and passes ν_r < 0.5 and ν_ε < 0.2 and the
exclusion (gate: floored; literal reported beside), and fine-tune-specific latents carry ≥ 0.5 of the planted change.
A variant passes when all its crosscoder seeds pass. A miss is a reading (exit 0), an early-check
miss stops the pod (exit 1). Run 1: ~$30 plus ~$4 of pod set-up; wall ~65 min for a-d.

## Run 2: d on the 52 sides (starts when pod a has passed and pod b has finished)

52 sides: Fig 4's 48 states (16 tasks × 3 seeds, `known3.json`), Gaussian updates at e54/elicit_s0-s2's per-matrix
norms (draw 0 is E203's 0.547), and E203's crosscoder reseed (e54/elicit_s0, seed 1). One crosscoder each, the authors'
recipe as E203 ran it, except that e53 and e62 sides now get Pile rows filling the token budget (E203's ran out at about
half the steps). Expected: seed-0 key-value states within 0.05 of E203's d (`negbeta_rescore.json`), reseed gap ~0.05.
Per side: FVE ≥ the self FVE − 0.05, dead ≤ 0.3, the early checks above. Time: 50 min a side (E203 sides 48.2-50.3,
`results/run2/sides/*.json .minutes`) × 52 = 43.3 H100-hours, $191, plus ~$23 of set-up on 26 pods. Shards `2 i/26`
(2 sides each, i = 0..25; the code refuses i ≥ n): ~1 h 50 min wall on 26 pods (7 pod-runners, ≤ 4 pods each);
`2 i/13` is 3.6 h on 13 pods. Every side reports d_authors under both rules; the floored value is at most the literal
one. `results/large` arrives on a pod as a dangling symlink: `run.py` replaces it with a real directory at start.

Whatever run 1 gives, run 2 reads the 48 states, and the figure (`plot.py`) puts the control results beside the
values (sketch in the card). Run 3 is held for a rerun if a pod is lost; checkpoints (~0.9 GB) live on the pod's disk
and do not outlive it, so a lost pod's sides rerun from step 0 (cost as full reruns, 50 min a side), and finished sides
are skipped only if their `sides/<key>.json` is copied onto the new pod. Total ~$250 against the $300 session budget
(84%); TF32 would halve the training part (12.3 vs 25.2 ms/step, E203 `timing.json`) to ~$115 for run 2 but is a
departure from the authors' fp32, so it is not the default.

## Design check

Fills open cell P2.83 as written. Not a new reading: d, its groups and every bar are E203's. The control's
construction changes (where the plant goes), not its rule; E203's placement is kept as pod e and reported beside.
Descriptive like P2.80: no verdict on reuse from a tool's reading.

## Not run here

TF32; more crosscoder seeds per state; a dictionary-size or k sweep (E243's ladder plateaued at block 2 under the bar
with a grafted head, a different control); the size-matched sides of E203 (Fig 4 has none).
