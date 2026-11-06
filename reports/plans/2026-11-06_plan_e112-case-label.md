# E112: the case-label pair, read under Jain et al.'s definition

*definition-lead, 2026-11-06. Direction: the user's E99 reset (`reports/decisions/2026-11-06_decision_e99-reset-to-jain.md`,
task file). Approved class 3 (the task names this pair and its order). Fast lane, $0 cluster debug L40.*

## The request, restated in the Terms' words

Read whether a fine-tune of the 67M model *reuses* an existing capability, using only Jain, Kirk, Lubana et al. 2024's
definition as adopted in `decisions.md` rows 81 and 87, plus the direction-removal check, on a new task pair with
Jain's shape: a sub-domain marker (our trigger token) plus content (the word before it), with the answer a function of
the content. Capability C = "the previous word is capitalised". Real task: after the trigger, output label word A if
the previous word is capitalised, B if not. Twin: same trigger, same two label words, assigned to each word by a
shuffled list, so no base feature predicts them. Expected: real reads "reuses C", twin reads "not through C".

## The definition this reads under (rows 81, 87; nothing added)

- **Possession (Def 1).** A linear readout trained on Pile text (never a task window) at some block recovers C on the
  sub-domain (task windows at the trigger position) above a bar filed now, and beats a Hewitt-Liang control task.
- **Reuse (Def 2 + row 87).** The fine-tuned map is a small function g of that readout (g = a threshold on the frozen
  readout's score, fitted on the training words), and the fine-tuned model is right on held-out words (row 87).
- **Causal check (row 81's added sentence).** Removing the readout's direction breaks the real fine-tune's behaviour
  and spares the twin at the same trigger. Pruning (Jain App. D Alg. 3) and reverse fine-tuning are reported beside
  it, without a rule.
- Verdicts: "reuses C" / "not through C" / "unreadable" / "definition does not apply". Nothing else gives a verdict.

## Design check (`/design`)

- **Goal.** `GOALS.md`'s "new capabilities vs strengthening existing ones" needs labels set independently of VPD;
  this gives a pair whose labels come from an outside definition, for the route split to be scored against (D3's
  definition feed).
- **Nearest tried.** E59/E64 (`experiments/jain_readout`, `readout64`): Jain readouts on the key-to-value pair, where
  C is token identity only. E99: old tasks without Jain's shape, read with rules now dropped. New here: a content
  function (case) with a twin at the same trigger.
- **Step or patch.** A step: the first pair built to the definition's shape.
- **Departures.** From Jain: (1) the readout is a binary ridge probe on block l's residual at the trigger position,
  not Jain's PCFG probe (our task is not a PCFG); (2) row 87's held-out carrying (a dated departure the user made);
  (3) direction removal is our addition (row 81); (4) the fine-tune is E62's arm-A recipe, not Jain's. None changes
  what counts as reuse.
- **Adopted definition unchanged?** Yes: rows 81, 87, nothing added. A miss on the twin is a result.
- **What changes next.** Real reads "reuses C" on both seeds → subject number (pair 2), then the route-split readings
  at this trigger. Real does not read cleanly → stop the line at case label and write it up; subject number does not
  run (user's order).
- **Control first.** Stage 0 is W0 alone: the possession probe and its Hewitt-Liang control, plus the random-direction
  removal on W0; no fine-tune trains unless possession passes.

## Task construction

- Pile rows (`commute.data.pile_rows`), 41-token windows as in `kv.windows`. Word pool: tokens that start with a space
  and a letter and occur in both cased forms (" The"/" the"), the 400 most frequent such pairs in Pile train rows;
  split by *pair* into 300 train pairs and 100 held-out pairs (seeded), so held-out words are unseen in either case.
- A window puts a pool word at position half − 1 and the trigger at half (the same marker token E54/E62 used,
  `keydep.vocab`), target at half + 1 (`copying.with_target` / `data.LAST`). Capitalised and lowercase forms 50/50.
- Labels: two single-token label words, picked once (seeded) from frequent nouns not in the pool. Real: label = case.
  Twin: each pool token (both forms separately) gets A or B by a seeded shuffle, balanced within train tokens.
- Fine-tune: E62's arm-A recipe (`keydep.finetune` → `finetune_arm_a`, Adam 3e-5, batch 32, KL 1, clip 1, per-window
  target via `o_token` Tensor), seeds 0 and 1, checkpoints 10, 30, 100, 300, 1000. Scored step: the first checkpoint
  whose *train-word* accuracy (label A vs B by the two label logits) reaches 0.9, else 1000 ("did not learn"). Real and
  twin are each read at their own scored step; their summed update norms are printed beside.

## Numbers, with expected values (filed in `experiments/case_label/results/predictions.json` before the run)

| # | Number | W0 | Real | Twin | Do-nothing / control |
|---|---|---|---|---|---|
| P0 | Possession: ridge probe for C at the trigger position, block l, trained on Pile positions (C of the previous token), accuracy on task windows with held-out words; selectivity = that minus the Hewitt-Liang control probe's (labels by a random fixed map of word pair → bit) | passes at some l before the last: accuracy ≥ 0.90 and selectivity ≥ 0.30; l* = earliest such block | - | - | control ≈ 0.5 on held-out words |
| P1 | Held-out carrying: fine-tuned model's label accuracy on held-out words | ≈ 0.5 (does not say A or B) | ≥ 0.90 | ≤ 0.65 | - |
| P2 | Small function: agreement on held-out words between the model's label and g(readout at l*), g a threshold fitted on training words | - | ≥ 0.90 | ≤ 0.65 | - |
| P3 | Removal: share of the label log-odds gain (state minus W0, on train-word windows of the state's own task, where both arms have a gain) removed by taking the readout direction out of the trigger position's residual at l* | W0's next-token loss change printed | ≥ 0.50 | ≤ 0.20 | random unit direction: ≤ 0.10 on every state (stop if missed) |

Verdict per state: possession passes and P1, P2, P3 all meet the real column → "reuses C"; possession passes and any
of P1-P3 misses → "not through C"; possession fails → "unreadable" (stop at stage 0). Filed outcome: real "reuses C",
twin "not through C", both seeds. A state between the bars (e.g. P3 at 0.3) reads "not through C" as the rule says;
the number is reported, no band is added.

Beside, no rule: pruning (`revival.py`'s grad·w scoring of MLP units, top K = 1, 4, 16, 64 against K random: label
accuracy after removal) and reverse fine-tuning (the same recipe on the windows' original next tokens: steps until
held-out label accuracy ≤ 0.6).

Edge cases checked before filing: P0 with no passing block → "unreadable"; a block passing only at the last block
does not count (Def 1 "some layer" before the readout is the output); P1-P2 on a state that did not learn: read as
filed (twin may not learn by step 1000; then P1 ≈ 0.5, verdict "not through C"); P3 with a gain ≤ 0.1 nats: the share
is undefined, reported as "no gain", verdict "not through C" (the model does not perform the map).

## Stages (one debug job each, ≤ 30 min, writes results as it finishes)

0. W0: pool, windows, labels; probe and control at every block; P0; random-direction removal on W0. Stop if P0 fails.
1. Fine-tunes: real s0, s1, twin s0, s1; save each scored state's ΔW in bf16 to `results/large/` at once.
2. Reads: P1-P3 per state, do-nothing direction per state, pruning and reverse fine-tuning beside.

Every scored number is printed to the log as well as `results/stage<N>.json`.

## Not in this plan

Subject number (pair 2; only if case label reads cleanly). The route-split readings (r, the switched-on set) at this
trigger: a later stage once the labels exist. Lowercase and the old E99 tasks are not re-read here.
