# Plan (E55): does post-training reuse a capability the model already has, judged without any decomposition?

**Reuse-lead, 2026-10-29, from a draft by Fable (advisor). Design check below. Not yet reviewed; no predictions filed; nothing run.** Lead's changes to the draft: reFT and the scratch baseline run at the same learning rate (both 1e-6 and 1e-5, Jain's Scr.+FT needed 1e-5 to converge); class line; C2 stays gated and needs the user's go-ahead.
Sources: `~/commute-data/agents/reuse-lead/brief.md` (Stage 1), `~/commute-data/scratch/reuse-lead/jain/jain_notes.md`
(cited as notes), `~/commute-data/scratch/reuse-lead/audit/task_audit.md` (audit rows), Jain et al. 2024 Table 6 (50K
pretraining iterations, text L3006-3075). Words as in `GOALS.md` → Terms and `decisions.md` (reuse-line rows).

## In plain words

Our "existing vs new" labels for post-training have so far come from VPD itself, so a sceptic reads every result as a
fact about VPD. This plan builds a ground truth that owes nothing to any decomposition: a small language model trained
on Jain et al.'s counting grammar, where we control by construction whether the model has a capability before
fine-tuning and whether the fine-tune can lean on it. On that model we fix, before any run, what "has the capability"
(a linear probe) and "the fine-tune goes through it" (ablation, pruning, reverse fine-tuning) must read. Those readings
are then applied, unchanged, to our real 67M fine-tunes, which get no label until they are read. Last, VPD and two
non-VPD instruments are scored on how often their labels agree with the ground truth.

**Claim.** On arms where reuse is fixed by construction, a filed probe-and-causal rule reads "reuse" and on arms
where novelty is fixed it reads "new", and VPD's re-gating share agrees with that rule more often than chance.
**Null.** The rule cannot be set so that the by-construction anchors are read right (the probe decodes everything,
or the causal checks move every model alike), in which case no instrument can be scored against it yet.

## Design check (`/design`)

- **Goal served.** `GOALS.md` "If no simple Φ works": new capabilities or strengthening. This gives that question a
  definition outside VPD (`decisions.md` reuse-line row, 2026-10-29) and the first ground truth to score Φ against.
- **Nearest tried.** E12/E46/E52 (`TRIED.md`): copy vs cipher labelled "existing/new" by assumption; E52 B4 found the
  original model does not do the copy task even prompted, so the labels were never checked. E53 A0: the first anchor
  the original model performs (lowercase, +1.18 nats), no probe, no causal check. New here: possession and relevance
  set by construction (pretraining prior, spurious rate), thresholds filed, and a causal test that does not use the
  probe's own data. Nothing in the repo probes f_C or revives a capability (audit, "Existing measurements").
- **Patch or step.** A step: a new line's first apparatus, not a fix of the last run.
- **Decisions.** Relies on the reuse-line definition row and the "GPU without explicit approval when audited" row.
  Departures: a new task family and model (PCFG minGPT, ~3M) and a decomposition we train ourselves (Stage C2, gated)
  against `GOALS.md` "Not the goal: re-running the decomposition" — C2 needs the user's confirmation; the rest is the
  brief's Stage 1 as approved.
- **What changes next.** Anchors read right → the rule is the line's ground truth; VPD's score decides whether the
  commute line's "elicit/teach" readings stand or are rewritten. Anchors read wrong → Def 1/2 need a different
  practical form (held-out values, other probes); no instrument is scored. VPD scores at chance while pruning or
  steering score well → the sceptic's reading holds and the log says so.
- **Control first.** Stage A0 (rehearsal, tiny) then A1's pretrained-model checks (behaviour reproduces Table 6's
  pretrained cells; probe positive and negative controls) run before any fine-tune; A2 runs only if A1 passes.
- **Numbers.** The tables below: per stage ≤ 6 numbers, each with its value at W₀, the negative control and the
  positive control.
- **Class.** Needs the user's confirmation (new task family, new model). Stages A-B and C1 are under the budget and
  not gated; C2 is gated.

## Stage A: Jain PCFG Counter anchors (possession and reuse by construction)

**Reproduction target.** Jain et al. Table 6 (50K pretraining iterations; the notes say Tables 5 and 6 show the same
pattern and 50K costs a quarter). Model: minGPT, 6 blocks, 6 heads, d 192, 2-layer MLP (L1160). Pretraining: AdamW,
lr 1e-3, 20% linear warm-up then cosine, next-token loss on the whole sequence, 50K iterations (L1164-1182), five task
families with a uniform family prior (L1170). Fine-tuning: answer-only loss, single (task, operand) = (C, b), AdamW,
batch 96, 10K iterations, 20% warm-up, cosine to lr/100 (L1186-1196).

**Recipe gaps and how each is filled (gated-run rule "every setting sourced"; none is "chosen here").**

| gap | filled with | source |
|---|---|---|
| pretraining batch | 96 | the paper's only stated batch (fine-tuning, L1189); same tokens/step as FT |
| context length | 262 = 7 prefix/suffix tokens + 250 text + answer; pad `#` to 262 | Eq. 1 and Fig. 15 (L1029-1058) |
| 250-token rule | first 250 terminals of the expansion; shorter strings padded with `#` | Fig. 15 shows `#` padding; "subsample 250" (L1025) |
| dropout, weight decay | 0.1, 0.1 | minGPT `GPT.get_default_config` / trainer defaults (the paper's named code, L716) |
| vocabulary | {a,b,c}, 5 family tokens, numbers 0-250, `$ < = #` (≈ 262) | Fig. 15-16; indices up to 250 |
| spurious strings (C_Tr) | insert b tokens at uniformly random positions until count(b) = count(a)+1; class histograms matched to C_Tr = 0 (Fig. 16) | "artificially adding the target tokens" (L1086); histogram match L1088-1116 |
| LR labels (one decade ambiguous) | run 1e-4 and 1e-5; cite each cell by its table value, never by L/M/S | Table 6 uses 1e-4/1e-5/1e-6 |
| prunable units | MLP hidden rows (the only "neuron" in minGPT), all 6 blocks | Fig. 7 "neurons"; Alg. 3 grad·weight on the count-a loss |
| probe | one linear layer per block output, vocab-sized softmax at Ans, 4K AdamW iters lr 1e-3 decayed ×0.1 at 2K/3K/3.5K, trained per model on fresh C_Te = 0 samples | L1238-1248 |
| reFT | FT recipe, lr 1e-6 (below the FT lr), count-a data, stop at 95% | L1249-1253 |
| data | pre-generated shards (CPU Slurm job), not online; online sampling in Python would bottleneck | notes (c); content identical in distribution |

**Models (three pretrainings).** LOW: operand prior (0.999, 0.001, 0); HIGH: (0.5, 0.3, 0.2) (L1173-1178).
NEVER: (0, 1, 0), a model never trained to count a: the probe's negative control and the scratch baseline for
reFT (Jain's Scr.+FT base is (0.001, 0.999, 0); we use 0 so "never had C" is exact; departure stated).

**Fine-tune arms (all count b; labels fixed by construction and by Table 6's cells).**

| arm | base | lr | C_Tr | construction label | Table 6 cells (acc %, 1K/10K): count-b C_Te=0, C_Te=1; count-a C_Te=0 |
|---|---|---|---|---|---|
| W1 | LOW | 1e-5 | 1 | **reuse**, weakly relevant (wrapper count(a)+1) | 0/0, 99.8/99.9; 2.1/20.7 |
| W2 | LOW | 1e-4 | 1 | **reuse** at the large lr | 0/1.9, 99.9/100; 20.2/15.9 |
| N1 | LOW | 1e-4 | 0 | **new** (count b learned; it was 5.2% before) | 98.9/100, 86.3/93.6; 10.8/9.1 |
| M | LOW | 1e-5 | 0 | unlabelled (mixed; reported, not scored) | 32.6/99.8, 12.4/88.7; 3/9.3 |
| R | HIGH | 1e-5 | 0 | **reuse**, strongly relevant (count b already 99.7%) | 99.9/99.9, 86.4/89.2; 99.6/73.7 |
| S | NEVER | 1e-6 | 0, count **a** | scratch baseline for reFT | — |

**A0 rehearsal ($0, cluster debug, ~10 min).** 500 pretraining iterations, one FT of 200 iterations, one probe, one
pruning, one reFT, saves and a resume. Measures: ms/step, shard generation rate, memory, disk. Stop: any stage
fails to run end to end. Not a smoke of the numbers.

**A1 pretraining + possession (control stage; runs before any fine-tune).**

| number | how | LOW (expected) | HIGH | NEVER (negative control) | else |
|---|---|---|---|---|---|
| count-a accuracy, fresh C_Te = 0 | behaviour, 4K samples | ≥ 95 (Table 6: 99.9) | ≥ 95 (99.8) | ≤ 15 (not trained) | stop: recipe |
| count-b accuracy | same | ≤ 20 (5.2) | ≥ 95 (99.7) | ≥ 95 | stop: prior wiring |
| count-a probe, best block, held-out | Def 1 probe, 2K fresh held-out samples | ≥ 0.90 | ≥ 0.90 | ≤ 0.40 | stop: probe or NEVER |
| count-c probe on LOW (selectivity; c is never an operand) | same probe recipe, label count(c) | ≤ 0.40 | — | — | stop: probe decodes anything |
| majority-class rate of the count label | from the DGP | reported (noise floor) | | | |

**Possession rule (Def 1, filed).** M possesses count-a iff max over blocks of held-out probe accuracy ≥ 0.90 AND
that accuracy exceeds NEVER's at the same block by ≥ 0.30 AND exceeds the count-c probe on the same model by ≥ 0.30.
Noise floor: three probe seeds; spread expected ≤ 0.02 (measured in A0). Edge cases: a block where both probes are
at ceiling (count-c ≥ 0.9) marks the probe uninformative at that block and the block is skipped; a held-out set with
< 500 samples is refused.

**A2 fine-tunes + causal checks.** Each arm at 10K iterations (and the 1K state saved). Fine-tune accuracy must
match its Table 6 cell within 15 points (two seeds of Jain's run are unknown; the gap is the noise allowance) before
the arm is read, else the arm is "not reproduced" and is reported, not scored.

| number | how (data never shared with the fit) | W1/W2 (reuse) | N1 (new) | R (reuse, strong) | baseline |
|---|---|---|---|---|---|
| C1 ablation drop | at block l* (best count-a probe on the fine-tuned model, fit on split 1), project out the probe's top-8 singular directions at Ans; drop in the arm's own task accuracy on split 2 (W: C_Te=1 data; N, R: C_Te=0) | ≥ 0.50 | ≤ 0.10 | ≥ 0.50 | random rank-8 subspace, 10 draws: ≤ 0.10 |
| C1′ patch | residual at Ans, block l*, from a donor string; share of outputs equal to count_a(donor)+1 (W) | ≥ 0.80 | ≤ 0.20 | reported | random donor-block patch ≤ 0.20 |
| C2 pruning revival | count-a accuracy (C_Te = 0) after removing top-K MLP rows by grad·weight on the count-a loss, K ∈ {5, 15} | ≥ 0.80 at K ≤ 15 (Fig. 7) | ≤ 0.30 | ≥ 0.80 | random K rows (20 draws) ≤ 0.30; the same procedure on S ≤ 0.30 |
| C3 reFT iterations to 95% count-a | reFT at 1e-6, cap 5K | ≤ 2.5K (Fig. 9: 0.1-3K) | ≥ 0.5 × S's | ≤ 0.5K | S (NEVER → count a): expected ≥ 4K (Fig. 9: 4.5K) or not reached |

**Reuse rule (Def 2 + causal, filed).** An arm is read "reuse" iff its base possesses count-a (A1) AND at least two of
C1, C2, C3 pass their reuse bar with their baseline under its bar; "new" iff at most one passes; else "neither".
C1′ is reported, not scored. Edge cases: C3 with S not reaching 95% within the cap uses the cap as S's value; C2 with
K = 5 already at ceiling on the random baseline (count-a not actually lost by the fine-tune, e.g. R) is skipped and
C2 scored on C1 and C3 alone; an arm whose fine-tune did not reproduce its cell is not scored.

**Predictions (to file in `experiments/pcfg_anchors/results/predictions.json`).**
- PA1: A1's five rows read as tabled on all three models. Lean high (Table 6; the probe rows are ours, lean medium).
- PA2: W1, W2, R read "reuse" and N1 reads "new" under the rule. Lean medium: Jain's pruning had no random baseline,
  so C2's random-K bar may be missed; the rule needs only two of three.
- PA3: C1 ablation separates (W ≥ 0.5, N ≤ 0.1) with random ≤ 0.1. Lean medium-low: no direction-level test exists in
  the paper; a shared counting subspace could make N1 drop too. Blind to: g living in attention rather than at Ans.
- Noise floors: fine-tune accuracy seed spread (two seeds of W1 only, reported); probe seeds as above.

## Stage B: the real 67M post-training arms, read with Stage A's rule, no label beforehand

Arms (audit rows 4-7, 10; recipes as run): lowercase Pile vs the 50-word token cipher on the same rows (E53, Adam
3e-5, 300 steps; cipher saved at steps 10/100/300); copy (ident32) vs cipher (rand32) with **held-out keys** (train on
32 keys, read on 32 unseen keys of the same kind: E12's recipe, 2 seeds); the per-key hand write (`perturb.key_write`)
as the planted new-mechanism twin of the copy fine-tune. Qwen has no anchor (audit rows 11-15): not in this stage.

| number | lowercase | token cipher | copy / cipher held-out | hand write |
|---|---|---|---|---|
| Def 1 probe at W₀: next-word case from the context (label by text rule, Pile case positions, held-out rows and held-out words, shuffled-label control) | reported; rule as A1 (≥ 0.9, ≥ control + 0.3) | π(x) probe on held-out words: must read ≤ control + 0.1 (absent by construction, audit row 7) | token identity: Def 1 vacuous (audit); replaced by the held-out-key read | — |
| held-out generalisation | — | — | gain on unseen keys ÷ gain on trained keys: copy reported; cipher ≤ 0.1 by construction | ≤ 0.1 by construction |
| C1 ablation: project out W₀'s case direction (block l*, fit on rows disjoint from the scored windows) in the fine-tuned model; drop in G at case positions (E53's G) | reported, rule as A2 | n/a (no f_C) | n/a | n/a |
| cased-context read (E53's separating read: log-odds at case positions in cased rows) | ≤ −1 → g∘f_C with f_C = context case; ≥ 0 → a bias | — | — | — |
| C2 pruning (top-K MLP rows on W₀'s cased-continuation loss, K ∈ {5, 15, 50} of 12,288) and C3 reFT on cased rows (lr 3e-6, steps to recover W₀'s case log-odds within 0.2 nats) | reported, bars as A2; random-K and the cipher arm as baselines (no scratch model exists on the 67M) | same, as the no-C baseline | reFT to W₀ on the key windows, copy vs cipher | reported |

Control first: the zero-update state must give G = 0 and the W₀ probe its A1-style controls before any arm is read.
Each arm's reading (reuse / new / neither) is written only after the numbers are in; the plan files no lean for
them beyond "the cipher and the hand write read new by construction" (their held-out rows are the apparatus check).

## Stage C: score the instruments against the ground truth

Scored arms: Stage A's W1, W2, N1, R (labels by construction, confirmed by the rule) and Stage B's arms that read
"reuse" or "new" (never "neither"). Each instrument gives one binary label per arm by its own filed rule; its score is
the agreement count, against the binomial chance of that many agreements.

- **C1 ($0, 67M, existing decomposition).** VPD: the re-gating share r of the route split (E12's number, "reuse" iff
  r ≥ 0.5) and the stand-in ratio (E53's bar 0.7/0.5). Non-VPD: the steering share S (E52 B3, "reuse" iff S ≥ 0.5)
  and Jain-style pruning revival (Stage B's C2, as the instrument rather than the truth when the truth comes from
  C1 + C3). Scored on Stage B's arms only.
- **C2 (gated).** VPD on the PCFG model: one decomposition of LOW with `param-decomp` (the minGPT must be ported into
  its model registry; its hyperparameters have no published values for a 3M model, so one small sweep is expected),
  then the route split and ratio on W1, W2, N1 (R would need a second decomposition: not in C2). Non-VPD on PCFG comes
  free from Stage A (pruning) plus steering (E52's code, one mean shift per block at Ans). Why C2 and not a crosscoder:
  VPD is the instrument the project is about; a crosscoder is $20-80 (baselines plan B5) and would be a second gated
  item. Why not skip C2: Stage B's truth is calibrated, not by construction; only C2 scores VPD against construction.

**Predictions.** PC1: on Stage B, r and S agree with the truth on ≥ 3 of 4 labelled arms (lean medium: E12/E52
separated copy from cipher, but those labels were never checked). PC2 (C2 only): r agrees with the construction labels
on W1, W2, N1 (lean low-medium: a 3M model's VPD is untested). Both are blind to a "neither" arm, which is excluded.

## Decision tree

1. A0 fails to run end to end → fix, rerun ($0); no reading.
2. A1: any control row missed → stop; fix the recipe or the probe; the user sees the numbers (Blocking card). PA1
   passes → A2.
3. A2: PA2 passes → the rule is the ground truth; Stage B. PA2 fails on one arm → report which check and why;
   amend the rule once (dated, before any Stage B reading) only if the fix is one clear bar
   (`decisions.md` 2026-10-29: decide clear rule fixes), else Blocking card. PA3 fails while PA2 passes → ablation is
   dropped from the rule (C2 + C3 carry it); said in the report.
4. Stage B: lowercase reads "reuse" and the cipher "new" → C1. Lowercase reads "neither" or "new" → PushNotification:
   the only 67M anchor the original model performs does not read as reuse; options: a second real arm (known-fact
   recall, audit row 15, on Qwen), or stop at the PCFG truth. Cipher or hand write reads "reuse" → stop, apparatus.
5. C1: PC1 passes → C2's gates. PC1 fails (VPD ≤ 2 of 4 while S or pruning ≥ 3) → the sceptic's reading is written
   into the log before any C2 spend; C2 then needs a fresh go-ahead. Both fail → the arms, not the instruments, are
   suspect: back to step 4 with the matched-gain cipher step.
6. C2: six gates (below), then the user's go-ahead in chat.

## Cost, routing, gates

| stage | compute | where | $ | wall | gated |
|---|---|---|---|---|---|
| A0 rehearsal | tiny sizes | cluster debug L40 | 0 | 10 min | no |
| A1 three pretrainings, 50K iters (notes: 15-25 ms/step H100; L40 ~1.5-2×) | ~3 × 25-40 min L40, run concurrently, checkpoint and resume across two debug jobs | cluster debug; H100 if A0's step time says > 2 jobs | 0 (H100: ~$2-4) | 1 h | no |
| A2 six fine-tunes + reFT + probes + pruning | ~6 × 6 min + 6 × ≤ 8 min + minutes | cluster debug, 2-3 jobs, or one H100 hour | 0 (H100: ~$4) | 1-2 h | no |
| B 67M | fine-tunes from E53 job 1 and E46's states; new probes/ablation/pruning/reFT | cluster debug, 2-3 jobs | 0 | 1 h | no |
| C1 | route split, ratio, steering on existing states | cluster debug, 1 job | 0 | 30 min | no |
| C2 | port minGPT to `param-decomp`, 1 decomposition + 1 sweep: 2-4 H100-h, up to 2× | RunPod H100 | 10-30 | 1-2 days agent | **yes** |

Total without C2: $0-8 and about 3-4 days of agent time (the PCFG code is ~500-700 lines, in `src/commute/pcfg.py`
with `experiments/pcfg_anchors/run.py` under ~300; one fresh-Opus review before any paid stage). C2 ≈ $10-30 in its own
session. C2's gates: pre-mortem (decomposition hyperparameters, memory, saves), every setting sourced (Goodfire's 67M
recipe scaled, stated), $0 rehearsal (a 1K-step decomposition on the cluster), early checks (reconstruction loss at
step 500 filed), two reviews, the user's go-ahead with the rehearsal's numbers.

## Order of work

1. Read this plan, the notes, the audit, `src/commute/README.md`, `experiments/performed_anchor/` (E53's code).
2. `src/commute/pcfg.py` (grammar, five labelers, spurious insertion, shards), `src/commute/probe.py` (Def 1 probe,
   ablation, patch), `src/commute/revival.py` (grad·weight pruning, reFT); `experiments/pcfg_anchors/run.py` stages
   `a0 a1 a2`; `experiments/reuse_arms/run.py` for Stage B reusing E53's and E46's state code.
3. File `predictions.json` (PA1-PA3, then the Stage B apparatus rows, then PC1) after running each rule on its edge
   cases; fresh-Opus review of the measurement code; `/pod-runner` A0, then A1 alone.
4. After each stage: analysis report by `/writeup` with its "Not run" list, `TRIED.md` row, `PENDING.md` line.

## Not in this plan

Indexer and Tracr settings (second instances; Tracr's compiled capability is a positive control at most); TinyStories
(released code, but no scalar f_C); Qwen arms (no anchor exists; known-fact recall is step 4's fallback); a crosscoder
or LoRA-SVD instrument; 200K-iteration pretraining (Table 5); a second decomposition for R; editing E53's filed
predictions (its cased-context read is reused as a number, its bars untouched).
