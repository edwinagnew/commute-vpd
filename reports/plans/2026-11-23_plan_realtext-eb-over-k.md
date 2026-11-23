# Plan: E_b over K on the real-text fine-tunes (cell P2.64)

**In plain words.** We read the frozen definition's write share over the needed set (E_b over K) on the 67M's saved
real-text fine-tunes, each side of three built pairs beside a random update of the same size, and set the cheap
readings and the base model's prior beside it. The question is the user's aim (a): can the definition's reading tell
reuse from new on realistic fine-tunes? It is read all at once and reported as a description, with no verdict.

Cell P2.64 (due 2026-11-23). Card `experiments/realtext_eb/card.md`; code `experiments/realtext_eb/run.py`, `plot.py`;
filed settings and expected values `experiments/realtext_eb/results/predictions.json`; rehearsal
`cluster/realtext_eb.sbatch`.

## Design check
- **Goal.** Claim 2 of the paper (the definition reads trained fine-tunes; `reports/plans/2026-11-22_plan_claim2-known-answer.md` item (a)): E_b has not been read on any 67M real-text fine-tune.
- **Nearest tried.** E268 (P1.54): E_b over K on built edits, RL/SFT and procedural states, not real text. E246 (P2.53) and E211 (P2.38) read r, S, share and prior on these same states, not E_b. New here: E_b over K on real text.
- **Patch or step.** A step: a new setting for the frozen reading, no new method.
- **Decisions.** Option (b) E_b (decisions.md 2026-11-21) over the frozen definition's K (2026-11-22); E268's K code unchanged.
- **Definition.** Read unchanged; no rule, band or verdict is added (descriptive, cell text).
- **What would change what we do.** All six sides inside the randoms' range: the paper says E_b over K does not separate real-text reuse from new (card stop). The claim 2 paragraph quotes the three-pair pattern (how many first sides sit above their twins), never one pair alone (reporting guards below).
- **Control first.** Stage 0, alone in each job, before any item: the do-nothing on L1's and T1's D (no changed variable, dd exactly 0, E_b over K exactly 1) and the identity interchange (x' = x) on L1_s0 and T1_s0 (zero argmax changes; its deviation is the fp32 floor). Any miss stops the job.
- **Saved for audit.** Per item num, den, dd, changed, K, first flip and flip counts (`results/large/items_<item>.npz`, ~1 MB each), so K* or another K can be rescored later on the CPU.

## What runs
The pairs, the side built to need what M holds first (confirmed in `experiments/realistic_tasks/run.py` label_tasks,
lm_rows and `run2.py` task_data; `reports/analyses/2026-11-18_analysis_realistic-tasks-run2.md`): **L1** topic label
(AG News) vs **L3** seeded coin label on the same 512 news windows; **T1** Pile lowercase vs **T4** reversed word order;
**T2** French vs **T3** Yoruba Wikipedia.

**Items: 36.** The 18 fine-tunes at the scored size (label 0.01, text 0.0175; seeds 0-2), each with its matched random
(E246's Gaussian draw at that fine-tune's per-matrix norms, regenerated from E246's saved seed and checked against its
saved size). Not the smaller sizes: there the coin is reached before it is learnt and French gains under the floor
(E211's analysis), and E246's readings exist only at the scored size; adding them would double the cost for states
that read nothing the scored ones do not.

**Read positions.** Label: the last position of the 35-token rows (E211's and E246's read). Text: position 63 of the
127-token windows, D cut after it (64 tokens): the model is causal, so K is unchanged, and every saved cheap reading of
these states read that one position. All 64 positions would cost about 3x (E268: 1,442 s against 511 s per item for 41
positions against one).

**K\* is not computed.** E_b over K\* needs E226's mean-ablation search, ~430 s per item (E257's RL items,
`experiments/rb_states/results/items/rl-*.json`), +60% on the run, and the cell asks for K only. The saved arrays allow
it later on the CPU once a K\* exists.

## Numbers and what each should read
| number | do-nothing | matched random | blind to |
|---|---|---|---|
| E_b over K | exactly 1 (no changed variable; convention) | 0.009-0.013 (E268's 11 randoms) | variables needed only on untested pairs; importance beyond write change; unchanged members of K |
| \|K\|, changed in K | not read | large (E268: 14,400-38,935 of 38,938) | as K |
| identity deviation | about 5e-5 (E268) | — | — |

Fine-tunes: not filed (descriptive). Rival a: first sides above twins by >0.01, all above randoms; rival b: all within
the randoms or within 0.01. K with no changed member gives 1 by convention: flagged `degenerate`, drawn apart, never
pooled. The prior (E246, geometric-mean P of the scored token at W0) is shown in its own panel as a confound: it orders
French below Yoruba, against the built direction.

**Reporting guards (2026-11-23, redteam-lead note-2026-11-23-redteam-p264-plan.md, adopted on whip-lead's
instruction, before any scored item).** They change how the result is reported, not the code or the filed
predictions. (1) |K|, n_changed_K and min_margin sit in the same row as every E_b over K; items with |K| under about
1,000 are named (E268: |K| falls as the margin rises, rank correlation -0.82). (2) The claim 2 paragraph gives the
three-pair pattern, not the one pair that separates. (3) "Above" means the two sides' three seed values do not overlap,
with all three seeds given, not the filed 0.01 gap; at n = 3 this is a description. (4) E_b is compared only within
a pair, never across the label and text pairs: they are read at different positions, and that changes |K|.

## Runtime and cost
Measured per item on an H100 (E268, `experiments/kstar_needed/results/items/*.json`): 505-514 s for 41-token windows,
733-743 s for 64-token windows, at the same 256 pairs. Scaled to 35 tokens, a label item is ~450 s; a text item ~740 s.
Total 12 x 450 + 24 x 740 = 23,200 s ≈ **6.4 H100-hours**.

- **4 H100 pods** (sharded by fine-tune, each with its random; `--shard I 4`): shards 0-1 hold 2 label + 3 text
  fine-tunes (6,240 s), shards 2-3 hold 1 + 3 (5,340 s). With ~8 min set-up, ~1.9 h wall. Upload: the 18 states
  (56.6 MB each, 1.02 GB) and `texts.pt` (21 MB) ≈ 1.04 GB ÷ 10 MB/s ≈ 104 s over the shared uplink; each pod gets only
  its 4-5 states (staged by hard link; `results/large` is a symlink). VPD comes from W&B in set-up. Cost: (2 x 6,240 +
  2 x 5,340 s + 4 x 480 s) ≈ 7 pod-hours ≈ **$21 at $3/h, $31 at H100 SXM $4.40/h**. Not gated (<$60, <6 h).
- **Cluster debug L40** ($0, the cell's "$0"): 30-min jobs. At H100 speed one job holds one text fine-tune and its random
  (1,480 s + ~2 min load and stage 0), so ~18 jobs (`--shard I 18`); on one free L40 that is ~9 h serial, on 4-8 free
  L40s ~1-2 h. L40 fp32 speed is unmeasured: the rehearsal measures it.

**Route** (decided after the rehearsal): cluster if the rehearsal's text item fits a 30-min job and ≥4 L40s start at
once; otherwise 4 H100 pods (`.claude/CLAUDE.md` → Compute: speed over cost; E268 moved to pods for the same reason).

## Rehearsal ($0)
`sbatch --comment='E273 rehearse: full sizes, stage 0 + L1_s0 + T1_s0' cluster/realtext_eb.sbatch items rh --only L1_s0 T1_s0`
on debug: full read sizes, stage 0 then one label and one text item. It measures time per item of each kind (item JSON
`seconds`, stage 0's `est_seconds_per_item`), peak GPU memory (`peak_gpu_gb`; E268 peaked at 11.3-11.9 GB on 512 x 41
and 256 x 64; 500 x 64 roughly doubles the cache, so ~20 GB expected, inside an L40's 48 GB) and the identity floor.
The two items count toward the run (a rerun skips them). If the job times out after L1_s0, resubmit the same line.

## Not run
E_b over K\*; the smaller saved sizes; the British pair (E233, P2.48: not named in the cell; +12 items, ~2.5 H100-hours
if wanted); all-position reads on the text tasks.
