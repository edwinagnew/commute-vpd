# P2.46: gain-only training on the 67M (plan, 2026-11-20)

Cell P2.46 (compare-lead, due 2026-11-22; rehearsal by the end of 2026-11-21). Code `experiments/gain_only/run.py` (train
and read), `experiments/gain_only/score.py` (score, hand-off, sync lists), shared module `src/commute/gains.py`; filed
settings and rules `experiments/gain_only/results/predictions.json`; card `experiments/gain_only/card.md`; sbatch
`cluster/gain_only.sbatch`. The E-number is the lead's to claim; it goes into `predictions.json` "experiment" before the
rehearsal.

**The request, restated in the Terms' words.** The user (task file, 2026-11-20 12:24): "gain-only training (base frozen,
one trainable gain per neuron and head, three seeds); hand the states to definition-lead." Here: the 67M base model is
frozen; one multiplier per MLP neuron (on `down_proj`'s input, after the GELU) and one per attention head (on `o_proj`'s
input, one number on the head's 128 columns), all starting at 1, are trained on each task of the nine built pairs (E187)
and the scorecard's four twins, with the reference full fine-tune's own data, loss, batch, sampler seed and clip. Each
trained state is also written as a weight change (`down_proj` columns and `o_proj` head blocks scaled by g - 1, every
other matrix zero) in the saved fine-tunes' file format, so the share, R and the held model read it unchanged.

## Design check (`/design`, six lines)
1. Goal: the section's claim "a change in when and how strongly existing mechanisms act" (results skeleton, 5.2) gets its
   literal test: can strength alone, one gain per unit, do what each built side's fine-tune did? Fills open cell P2.46.
2. Nearest tried: E205/E197 (TRIED.md:297-299), which fitted gains after the fact to full fine-tunes and recovered at most
   0.104 of any output change; that asks whether the fine-tune *was* a gain change, this asks whether a gain change *can*
   do the task, side by side. E228 (P2.44) supplies the KL matching and the share reader used here unchanged.
3. A step toward the claim, not a patch: it is the run the skeleton lists second in priority.
4. Decisions: GOALS r's gain and MIN_GAIN (1 log-odds) as filed; E228's readings unchanged; no departure from `decisions.md`.
5. No verdict under any reuse wording: the report says which side reaches what. No rule is added to a definition.
6. Control first, alone: stage 0 (gains of 1 change no bit and export a zero change; a random gain state equals its
   exported change and a shuffled one does not; the read path reproduces E136's gain), then each run's step-1 KL guard.

## What runs
Per (task, seed): 26 tasks x seeds 0, 1, 2 = 78 runs, 27 references (a reference = one task's saved full fine-tunes;
e54/cipher has two, the norm-matched step-9 states and the own-step-30 states). Each run first reads its references'
mean trained-answer gain and KL to base on the read rows, then trains with Adam at 3e-3 (IA3's rate; Adam and a constant
rate as every reference) and checks gain and KL on the read rows every max(1, reference step // 10) steps. It keeps the
gains at the first check whose KL reaches each reference's KL ("kl:<ref>") and whose gain reaches each reference's gain
("gain:<ref>"), or the check with the largest value, and stops when all are reached or at max(300, 10 x reference step).
Stage 2 writes each kept state (deduplicated by step) as a bf16 weight change and reads it from that file: gain, KL, and
E228's `readings` (kl_share and r) on the same rows.

**Matching. Primary: matched output change** (KL to base on the read rows, as E228): a gain-only update can usually buy
any gain by moving other outputs, so "reaches the gain somewhere" separates little, while the gain it gets for the full
fine-tune's own output change is the like-for-like comparison. Secondary: matched gain (the first crossing, or the
largest gain within the cap). Both are filed.

**Recipes** (each the reference's own; the explorer's table with file:line is in the hand-off notes of this session):
kv tasks (e46, e54, c3, e58): arm A on `windows_*.pt["train"]`, (V,) table target, batch 32, KL weight 1. E62: arm A,
last-column target, ci_len 41. Label twins (case_label, repeat label): arm A on the task file's train part labelled as
`prior_vs_share.label_windows` labels the read part, ci_len 41. E53: arm B next-token CE on its 2,400 rows, batch 8, in
`lowercase.batch_order(seed)`, rows cycled past step 300. Departures: the parameter set; the rate (3e-3 against 3e-5); no
fit-accuracy early stop for the label tasks (the matching rule stops runs instead); E53 rows are seen again past 300.

## Numbers and what each should read
| Number | Expected | Where |
|---|---|---|
| gains of 1: logits equal (fp32, bf16 autocast), gain and KL, exported change | equal; 0; 0; 0 entries non-zero | stage 0, stop |
| random gain state (1 + 0.05 N): exported change vs hooks; shuffled | rel. error ~1e-6 (< 1e-3); ~1 (> 0.1) | stage 0, stop |
| c3/rank1_s0 gain on the read rows | within 1e-3 of E136's gain_fp32 (10.94 autocast) | stage 0, stop |
| step-1 KL | under the task's smallest reference KL | every run, stop |
| reach (gain at matched KL / reference gain), median of 3 seeds | rival a: reuse >= 0.5, new < 0.5; rival b: same call both sides | score, P1 |
| reach at matched gain | as above | P1b |
| kl_share at matched KL | >= the pair's reuse-side full fine-tune kl_share - 0.0314 | P3 (needs E228's states) |
| do-nothing (gains at 1) | gain 0, KL 0, reach 0, kl_share nan | edge cases |

Rules and edge cases (none reach, ties, nan, zero change, too few read) are in `predictions.json` and tested in
`tests/test_gain_only_rules.py`. The held model and R on these states are definition-lead's (hand-off below); the held
model's reading 1 should read near 1, since each state is a gain state by construction: a positive control of the held
model, not evidence about the pairs.

## Rehearsal ($0, MATS cluster debug QoS, one L40, at most 30 min)
`sbatch --comment='E### rehearse: gain-only, four recipe families at full read sizes' cluster/gain_only.sbatch rehearse`
runs stage 0, then c3/rank1, e62/lookup, e53/lowercase and case_label/twin (seed 0) at full read rows and full training
data, each capped at 20 steps, trains, exports, reads through E228's share, and runs the same command again (every item
must be skipped: the resume). It records into `results/rehearse/`: s/step and peak GPU memory per family (items JSON),
read seconds and file size per state (reads JSON), stage 0's floors, and step-1 KL against each reference's. To fill in
after it: s/step ____ (kv) ____ (E62) ____ (E53) ____ (label); read s/state ____; peak GB ____; state MB ____ (expected
54: 28M bf16 numbers); `fleet/rehearsed compare-lead E### '...'`.

## Scored run (pods; not gated: expected well under $60 and 6 h)
Three RunPod H100 pods, one per shard (tasks dealt in turn, so each pod has one E53 task), labels `commute-E###-gain-0`,
`-1`, `-2`. Before the pods: `python experiments/gain_only/score.py handoff 3` writes the sync lists
`results/manifest_<I>_of_3.txt` (2.7, 3.0 and 2.0 GB of reference states and windows: 4.5, 5.0 and 3.4 min each at 10 MB/s;
`artifacts/vpd` by the W&B fetch, ~65 s). Stage each list's files flat by hard link (CLAUDE.md, gitignored `results/large`).
Per pod: `PRED_SHA=<sha> python experiments/gain_only/run.py shard I 3 2>&1 | tee -a experiments/gain_only/results/pod<I>.log`.
Cost, to be redone from the rehearsal's s/step x steps: guessed 26 items x (up to 300 steps x ~0.05 s + checks + ~2 reads x
~40 s) ~ 45-75 min per pod, the E53 item (3,000 steps of 8 x 512) ~5 min, plus ~10 min set-up and upload: ~1.2 h x 3 pods x
~$3/h ~ **$11**. Downloads: per pod `tar` of `results/items`, `results/reads`, `results/stage0_*`, `results/large/gains`,
`results/large/states` (~26 items x ~2 states x 54 MB ~ 2.8 GB: ~5 min), into `experiments/gain_only/results/pod<I>_dl/`
(not the `results/large` symlink), sha256 checked on both sides, terminate, then move. Download each item's files as
they land if the pod runs past 45 min.

## Hand-off to definition-lead
After the pods: `python experiments/gain_only/score.py handoff 3` and `score`. `results/handoff.json` lists every exported
state in E136's inventory format (task = its reference task, so its read rows are the reference's), with what it was kept
for (`kl:<ref>`, `gain:<ref>`), step, sha256, gain and KL; files `results/large/states/<task>_s<seed>_step<step>.pt`,
dict matrix name -> bf16 dW over all 24 matrices (`W = W0 + dW`), the format of
`experiments/refit_96key/results/large/dW_*.pt`, readable by E228's `state_read`. The gains themselves:
`results/large/gains/<task>_s<seed>.pt`.

## Decided here (in the weeds)
Primary matching = KL; the first crossing, not the nearest check; references = each task's saved states' mean; seeds
0-2 are the references' sampler seeds; reach is the median over seeds, a side the mean over its references; the rate from
IA3 with no sweep (the step-1 guard catches an overshoot); a check every tenth of the reference's steps; bf16 export (as
E147), every reading from the saved file.

## Not run
R and the held model on these states (definition-lead); a learning-rate sweep; gains on other units (layers, blocks);
Qwen.
