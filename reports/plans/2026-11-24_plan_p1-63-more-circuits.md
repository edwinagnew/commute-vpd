# P1.63, due 11-25: Figure 2's built edits on two more circuits (E301)

**Request, in the Terms' words.** The user (2026-11-24): the five built edits of Fig 2 (x2, x10, rebuilt, new_few,
new_thin), each with its own size-matched random, made again on two or three other base-model circuits and read by the
write share E_b over the needed set K at VPD subcomponents, top neurons and SAE latents, with exactly P1.62's rule and
leftover convention; do-nothing and identity first; "experiemental rigour is essential!!!!". No definition wording
changes; reported as a description.

**Design check (/design).** *Goal:* the paper's second claim (cheap readings track the definition); Fig 2 is its
built-edit evidence, so far from one site. *Nearest tried:* E214/E268/E277/E263 read these recipes on one site (C at
layer 1, copy24); E280 read one-head edits (VPD capped them at ~0.2). New here: whether Fig 2's order (reuse edits
high, new and random low) is a property of the recipes or of that site. *Patch or step:* a step (generality of a
paper figure). *Decisions:* none departed from; P1.62's protocol used unchanged. *Definition:* unchanged; P1 below is
a description, not a verdict. *What changes next:* if the order fails on a new site at two or more featurizations, the
paper must say Fig 2's built block holds at one site only. *Control first:* stage 0 below, alone, stops on any miss.
*Saved:* each state's num/den/dd and K (npz, `results/large/`), C, gammas, the rebuilt weights, per-state JSON.

## A finding before any run
M's top-1 at copy24's answer position is **0.000** (75 held-out windows; cpu-run 2026-11-24). Fig 2's x2/x10/rebuilt
site is not a behaviour M performs there; its K is F's argmax at every position. Its C does reproduce exactly through
the generalised data path (below).

## The circuits (M's top-1 at the read position, measured on CPU at planning: `experiments/built_circuits/results/planning/`, scripts in `~/commute-data/scratch/regating-lead/p163/`)
| circuit | windows, read position, answer | C's layer | M top-1 | why |
|---|---|---|---|---|
| **induction** | 21 distinct random tokens shown twice, last cut (41 tokens); read at 40; answer the token after its first copy, fixed per read token by a 64-pair table | 2 (after head (2,4)) | **0.957** (512 held-out; 0.959 without the table, 0.912 with tokens from the whole vocabulary) | E62: ablating head (2,4) takes M's second-copy top-1 0.825 to 0.0095; C = [767, 883, 1358, 1406, 1523, 2786, 2895, 3037], signed attribution +1,494, so x2 strengthens the behaviour (Fig 2's C: -68) |
| **bigram** | 41-token Pile spans with a key at 20; key = a token whose next token is one token in >= 90% of >= 30 occurrences (86 keys, e.g. " according" -> " to"); answer that token | 0 (attribution mass peaks there) | **0.951** (512 held-out; 0.895 with the 15 keys at >= 100) | a lookup, distinct from copy and induction; layer 0 carries the most attribution mass; C = [159, 203, 525, 908, 1027, 2564, 2650, 2825], signed +50 |
Rejected: the rank-1 map (`windows_rank1_trained.pt`): M's top-1 0.133 held-out. A third circuit: none distinct and
cheap (layer 3 has no MLP after it for rebuilt; previous-token head (1,1) shares Fig 2's layer).

Both use copy24's table format, so E195's `attribution` and `fit` run unchanged on a `Data` built here (only
`load_data`'s hard-coded COPY24 path is bypassed). Departures from Fig 2: C's layer fixed, not argmax of mass;
rebuilt trains every MLP after C (one layer for induction, three for bigram; Fig 2 two); new_few/new_thin at
slot_layer = C's layer on E214's cue windows unchanged (Fig 2: layer 0); read at the answer position only (Fig 2's x2
read at all 41, where K is every variable).

## States per circuit (11) and run order
do-nothing; x2, x10 (`rescaled`); rebuilt (C zeroed, E195's `fit` to KL(x2 || F)); new_few, new_thin
(`known_edits.build`, reuse items dropped as in E214); one E290 random per edit. Each read by P1.62's driver
(`experiments/fig2_standard` and its SAE swap module, called, not rewritten).
0. **Stage 0, alone, on cluster debug ($0); any miss stops** (rules R0a-R0f in `predictions.json`): M's top-1 >= 0.5;
   do-nothing at each featurization (0 changed, dd max 0, E_b 1, K non-empty); identity swap on x2 (0 flips, floor
   recorded); builders on Fig 2's inputs (C = E195's 8 exactly, already true on CPU; E214's gammas to ~1e-7; E195's
   refit within the fit-seed floor); the derangement floor delta_f (induction x2, scan seed 268 vs 269).
1. Builds and their checks (R0e), then x2 first on each circuit, then the rest.
2. CPU assembly: rows `feat,block,item,side,value,field,source` (block `built_<circuit>`, field `E_b_K`, plus
   `E_b_K_no_leftover`, `leftover_share`), P1 on both circuits and on P1.62's Fig 2 rows, a one-page analysis.

## Expected values (Fig 2's as the reference)
| state | VPD | top neurons | SAE |
|---|---|---|---|
| do-nothing | 1 (convention) | 1 | 1 |
| x2, x10 | ~0.76; <= 0.2 if C spreads over mixed subcomponents | >= 0.99 if all of C are variables (positive control) | ~0.95 |
| rebuilt | ~0.6 | ~0.05 | ~0.13 |
| new_few, new_thin | <= 0.02 | <= 0.02 | <= 0.02 |
| randoms | <= 0.03 | <= 0.03 | <= 0.03 |
Blind to: positions other than the read one; a C neuron below the p97 band (pooled; count reported); writes in the
SAE error term (leftover share reported); variables of K that did not change (|K and changed| reported).

## Cost (measured per state x count; SAE a guess until E298's rehearsal)
23 scans per featurization (2 x 11 states + the x2 repeat). VPD 505-515 s per state at one position on 512 x 41
(E268, H100): 23 x 510 = 3.3 h. Neurons 162-308 s (E277): 23 x 250 = 1.6 h. SAE ~3.4x VPD's variables, **guess**
1,730 s: 11.1 h. Builds and four fits ~0.3 h (fit time never logged: measured in stage 0). Total ~16 GPU-h, ~$45 on
H100 pods at ~$2.7/h; 8 pods (two runners) ~2.5 h wall. **Not gated** on this estimate. Stop rule: before the SAE
stage, re-price from E298's (or stage 0's) measured SAE seconds; over $60 or 6 h wall, the Gates section is written
and whip-lead approves before any paid SAE scan.

## Not run
- A third circuit; Fig 2's own circuit re-read at the answer position; seeds of the attribution or fit beyond stage 0's floor.
