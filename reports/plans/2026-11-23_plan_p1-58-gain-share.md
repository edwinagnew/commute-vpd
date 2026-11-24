# P1.58 (E282): the gain-only share, a prototype of option 2

Cell P1.58, due 2026-11-24. Card `experiments/gain_share/card.md`; filed settings, expected values and rules
`experiments/gain_share/results/predictions.json`; code `experiments/gain_share/run.py`, `src/commute/vpd_gains.py`.

We fit, for each fine-tune, the best update that may only turn the original model's own parts up or down, and ask how
much of the fine-tune's gain on its task that update recovers. The user's definition says a part is kept when the
fine-tuned model's mechanism equals the original's up to a gain; this is its behavioural version. The question for
this run is whether it tells E62's copying map (head (2,4) carries half its gain) from its lookup (the head carries
none), which the kept share over the needed set could not (0.04 for both, 2026-11-23 diagnosis).

## What runs

One job on one L40, cluster debug QoS, $0, under 30 minutes (`cluster/gain_share.sbatch all`). For each state F and
each of two levels of parts, Adam fits gains, every gain starting at 1, to minimise KL(F || G) on F's whole
next-token distribution at the task's read positions, on training rows. The two levels:

- **Neurons and heads (N):** one gain per MLP neuron and per attention head, as E231 (`commute.gains`).
- **VPD subcomponents (V):** one gain per subcomponent of each of the 24 decomposed matrices, 38,912, plus one per
  matrix remainder R = W0 - sum of the subcomponents, 24: W' = g_R R + sum g_c U_c V_c. Gains of 1 give W0 exactly.

Readings, on held-out rows: the **share** (G's gain in the trained answer's log-odds over F's, both against M, read
when F's gain is at least 1 log-odds), **KL(F || G\*)**, and the **KL share** 1 - KL(F || G\*) / KL(F || M). Fit:
600 steps, batch 64, rate 0.1 with 30 warm-up steps then cosine decay, gradient clip 1 (sources in the predictions).

Splits: E62's own windows (800 rows to fit, 200 to read, 41 tokens, read at the marker); copy24 for x2 and x10 (301
to fit, its 75 held-out windows to read, KL at every position, the answer at the copied position); E214's 512 cue
windows for the rest (256 to fit, 256 to read, last position).

## Stages and expected values

| stage | state | expected | stops the job |
|---|---|---|---|
| 0 | gains of 1, both levels | logits bit-identical to M, export exactly 0 | any difference |
| 0 | VPD remainder | R + U^T V^T = W0 and R = the authors' remainder, within 3e-6 relative (10x the fp32 floor) | over |
| 0 | random gain state | export matches hook (about 1e-6, bar 1e-3); shuffled differs (about 1, bar 0.1) | either |
| 0 | do-nothing F = M | KL(F \|\| M), F's gain, KL(F \|\| G\*) all exactly 0, no gain moves | any nonzero |
| 1 | x2, x10 (eight layer-1 copy-neuron columns of down_proj times 2, times 10) | N: at least 0.95 (exactly representable); V: about 1 (0.7-1.05) | N under 0.9 |
| 1 | KL-matched random (E242's random_kl_241002) | KL share at most 0.2, both levels | F's KL off E242's target by over 10% |
| 1 | new_few, new_thin (E214's new edits) | share at most 0.3, both levels | F's KL off E226's by over 1e-4 |
| 2 | E62 copy, seeds 0-2 | N about 0.85 | F's gain off E231's by over 1e-3 |
| 2 | E62 lookup, seeds 0-2 | N about 0.6 | same |
| 3 | E280's head-(2,4) O x2, V x2 (the same function) | N at least 0.95 (one head gain of 2); V between 0.2 and 0.9 | none (descriptive) |
| 3 | E280's head-(2,4) Q x2 (attention logits doubled) | N under 0.95 (no head gain sharpens a pattern); V under V x2's | none |

The E62 priors come from E231, which trained neuron and head gains on the task itself (not on KL to F) and reached
0.87 (copy) and 0.63 (lookup) of the fine-tunes' gain.

## Predictions (filed)

- **P_N:** mean share of copy minus lookup at N is at least 0.1.
- **P_V:** the same at V.
- **Rival a** (copy is a rescale of M's own parts at either level): P_N and P_V.
- **Rival b** (VPD spreads the one-head rescale over shared subcomponents that no gains rebuild): P_N, while at V both
  means are under 0.3 and within 0.1.
- **Stop for the line:** if P_N fails too, the gain-only share does not separate the user's worked example and
  option 2 is not carried past this prototype.

The bar: 0.1 is 2.5 standard errors at three seeds per map, from the head-(2,4) ablation share's seed spread on the
copying map (0.049, `experiments/induction_cue/results/run.json`), a proxy in the same units.

## What the share is blind to

It is one number per state: it does not say which parts carry G\*'s gain. A gain-only update that reaches F's answer
by a route F does not take still reads as recovered; KL(F || G\*) shows how far G\* is from F on the rest of the
distribution. At V a gain scales a subcomponent's whole rank-one write, shared across heads, so a one-head rescale is
only as reachable as the shared subcomponents allow: that is what rival b tests. The 38,936 V gains can overfit 256-800
training rows; each fit reports train and held-out KL side by side.

## Cost and time

$0 (cluster). Time estimated from E231's measured 0.031-0.05 s per gain-only step on the same model (its items'
s_per_step, reads included): 28 fits of 600 steps is 9-14 min at that rate, plus the V level's hooks, the reads and
building the model and edits once (the edits and the matched random took 18 s in E257's `run_random_p0.log`): about
20-25 min against the 30-minute limit, not measured on a GPU (the model may not load on a CPU). Each (state,
level) is written as it ends, so a time-out loses only the fit in flight and a rerun resumes.

## E280's head edits (stage 3)

Definition-lead's head-(2,4) edits (`experiments/head_edit/`, commit b65a8743) are rebuilt here from the same recipe
(layer 2, head 4, its 128 rows of q_proj or v_proj or 128 input columns of o_proj, times 2) and read on E62's copy
rows as stage 3 of the same job, after E62. Any other saved states go through `run.py list --manifest PATH`.

## Not run

- No search for which parts carry G\*'s gain (a per-gain read of the saved gains is possible later from
  `results/large/gains/`).
- No second fit seed per state: the fit's own noise is read only through the controls.
- No VPD level with the remainders frozen at 1.
