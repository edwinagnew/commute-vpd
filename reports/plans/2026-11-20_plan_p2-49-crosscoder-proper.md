# P2.49: the crosscoder replicated properly, scored on the share's pairs, twins and grafts

Phase A design (no code, E-number pending). Cell P2.49 (compare-lead, 2 runs, due 2026-11-23). Appendix (settings and
sources, pre-mortem, cost arithmetic, item lists): `reports/plans/2026-11-20_plan_p2-49-crosscoder-proper_appendix.md`.
Card: `~/commute-data/scratch/compare-lead/p249-designer/card.md` (passes, 2.61 SE at n = 3). Draft rules:
`experiments/crosscoder_proper/results/predictions.json`. Earlier line: E198, E203 (P2.31), E210 (P2.37), E212 (P2.38).

**The request in the Terms' words.** Train Minder et al.'s BatchTopK crosscoder with latent scaling between the original
67M model and each saved fine-tune that the re-gating share is scored on: the nine built pairs, as built and
size-matched (E187), and the four twins (E228's catalogue). Add the grafts (E206's heads and neurons added to the frozen
model). Use the authors' pinned trainer at their settings and full token budget, on every side. Report their own output
per side: how many fine-tune-specific latents there are, and what share of the activation change they carry. Run the
controls first, each with a filed stop. Take seed ranges from both crosscoder and fine-tune reseeds. Read each pair by
the scorecard's every-seed rule. Descriptive: sides keep their built names.

## Design check

- **Goal.** Part 2 of the roadmap: baselines on every pair with a saved update (rung L2.1). It replaces a weak
  replication: one crosscoder a side, the planted control missed, six sides at about half the tokens. A clean "does not
  apply" also counts as a result.
- **Nearest tried.** E203 (one seed a side, 5/9 and 4/9, planted missed at nu_eps 0.244), E212 (Gaussian read 0.730
  against a 0.45 stop), E210 (port reproduces the released Gemma crosscoder; cited, not rerun). This run adds what those
  lacked. Seed ranges give the every-seed rule. Grafted heads are planted mechanisms, not a planted direction. The
  Gaussian is matched on activation change, not only weight norms (E212's was 36-114 times smaller). It adds the
  authors' TF32 training (`train_crosscoder.py:23`, `config.yaml:141`; E203 ran fp32) and a second site.
- **Patch or step.** Mostly a patch: a proper rerun of a baseline the user asked for. The grafts and the second site are
  new.
- **Decisions.** The scorecard's every-seed rule (`gate_plot_1116.apart`), E187's c, E228's catalogue, and the graft
  declarations (decisions.md 2026-11-18). No departure.
- **Definition.** No verdict under the working definition. Pairs are called against how they were built: "reuse/new as
  built", "reversed" or "no call". The cell's "reuse / new" is read that way. The graft call uses the filed planted
  bar. That reading is the one question for the user (end of this plan).
- **What changes what we do.** Controls passing and calls on at least 7 of 9 pairs: the crosscoder becomes a working
  baseline beside the share. Controls failing: the entry reads "not applicable at this update size" and the line ends on
  the 67M.
- **Control first.** Stage 0 runs alone at both sites. (a) Base against itself fails only on a code fault. (b) A grafted
  head can fail if the method cannot see machinery the model lacks. The matched Gaussian follows in stage 1. It can fail
  if a structureless change reads as reuse-like as a reuse fine-tune. Every recipe difference from E203 is listed in
  the appendix, section A.
- **Saved.** Per crosscoder: the side JSON and per-latent vectors (relative norm, alive, s_j, betas, groups; about 0.5
  MB), and the decoder in bf16 (75 MB). Run 1 also keeps the encoder.

## Sites

**Block 2's input (site A).** This is the authors' middle layer (`config.yaml:123`, `layers: [0.5]`) and E203's site.
The nine pairs and twins are scored here.

**The last block's output (site B, `out3`).** This is the residual just before the final norm. It is the only site
every update writes to. E206's grafts all sit in layer 3 (four heads and 256 neurons), so at site A their change is
exactly 0. The grafts are scored here, and the primary pair is read here too. It shows whether the pair reading
changes where blocks 2 and 3 are visible.

## Run order (cell's numbering)

1. **Port.** E210 (P2.37) is cited: relative norm within 1.49e-7, nu Spearman 0.988 and 0.977. Not rerun.
2. **Apparatus.** The pinned trainer at the authors' settings (appendix A). Every side gets the full 97,656 steps × 2,048
   tokens. The e53 and e62 task halves keep their capped draws, and Pile rows from 589,824 fill the rest. The stream
   asserts its length before step 0. Each crosscoder reports FVE, dead share, L0 and spliced loss. Spliced loss is the
   next-token loss with the crosscoder's reconstruction put in at the site, against clean and mean-ablated, on Pile
   val. One k sweep (k 32 and 200 beside 100) runs on the site-A grafted head.
3. **Controls, run 1 stage 0, alone (8 crosscoders, 8 pods).** Base against itself at A and at B. A grafted head at A,
   3 crosscoder seeds: E206's early head (layer 1, k = 16). It failed its mechanism check, but it is fresh machinery
   writing before block 2. At B, the last-layer head (k = 16, E206 `last_lowest`), 3 seeds. Stops are in
   predictions.json (S0a, S0b). A miss at a site stops that site's stage 1 and goes to the user.
4. **Seed ranges and the Gaussian, run 1 stage 1 (29 crosscoders, 15 pods).** The primary pair is e54_matched (elicit
   against cipher), at both sites. That is fine-tune seeds 0-2 at crosscoder seed 0, and crosscoder seeds 1-2 on
   fine-tune seed 0, on both sides. That makes 10 a site. The matched Gaussian takes elicit seed 0's per-matrix norm
   profile (`finetune.random_direction`) and one scalar c. It is scaled so its mean squared activation change per token
   at the site equals elicit seed 0's within 5%. That is 3 crosscoder seeds a site. E203's planted direction runs at A
   with 1 seed, reported only. The k sweep adds 2 at A. Stop G2: the Gaussian must read above every reuse seed. Stop
   G1: the Gaussian must stay under the planted bar.
5. **Scored, run 2 (76 crosscoders, about 19 pods).** It runs only at a site whose controls passed. At A: the nine pairs
   as built, 3 seeds a side (2 where only 2 are saved), 40 new crosscoders. The four twins, 2 seeds a side, 16. The six
   pairs with c < 0.95, the larger side scaled, 2 seeds, 12. For the three pairs with c ≥ 0.95 the as-built cell
   stands (E203: those moved d by at most 0.015, under the 0.047 reseed). At B: four grafts × 2 crosscoder seeds, 8;
   `last_lowest` is already in stage 0. Then `score.py` on CPU, and one scorecard entry beside the share on the same
   axes and box rule.

## Numbers and what each should read

| Number | Self | Grafted head (planted) | Matched Gaussian | Primary pair (E203 says) |
|---|---|---|---|---|
| fine-tune-specific count (top 1,667, nu_r < 0.5, nu_eps < 0.2) | 0 (stop if any) | ≥ 1 | 0 | 0 both sides |
| specific part (share of the change they carry) | nan (no change) | ≥ 0.5 (stop) | ~0 | ~0 (≤ 0.003) |
| d = shared − specific (E203's form, beside) | nan | ≤ −0.3 | 0.55-0.75 (E203, E212) | 0.40-0.48 |
| max abs(relative norm − 0.5), alive | ≤ 1e-3 (stop) | — | — | — |
| FVE each model | ~0.93 (stop < 0.8) | ≥ 0.8 | ≥ 0.8 | 0.91-0.93 |
| dead share | 0 (stop > 0.3) | ≤ 0.3 | ≤ 0.3 | 0 |
| spliced-loss recovered fraction | measured; sets the bar | within 0.05 of self | within 0.05 | within 0.05 |
| activation change per token | 0 | measured | = elicit's, within 5% | 1.96 / 3.22 at A |

**Blind to.** The specific part does not see a change in how the model uses a direction it already represents: a
shared latent counts as shared whatever it does. It does not see any change outside the site. It leans toward shared
latents by construction (a joint loss; Dumas et al. 2025). d sees how well the shared dictionary rebuilds the change,
so a structureless change reads high (E212). The spliced loss measures fit, not the split.

**Noise floor.** The reseed spread: the range of each reading over 3 crosscoder seeds and over 3 fine-tune seeds of
the primary pair at each site. The pair rule is ordinal (every seed beyond every seed, strict), so no band enters it.

## Filed predictions (summary; rules and edge cases in predictions.json)

- **Native output per side.** Count and specific part are reported for every side. Lean: 0 latents and a part ≤ 0.003
  on most sides; e58 copy alone near 0.1 (E203 0.12).
- **Controls.** Self passes (high). The B head passes (medium). The A head passes (low-medium; what it writes is
  unknown). G1 passes (medium-high). **G2 fails at A (lean, medium):** the Gaussian and the reuse side both read 0,
  which is rival b.
- **Primary pair at A.** "No call" (medium-high).
- **Calls.** "No call" on at least 7 of 9 pairs as built and size-matched, and on at least 3 of 4 twins. Grafts
  "new" on at least 3 of 4 if the B head passes. Each is reported only where its site's controls passed.

## Cost and time (appendix C)

Measured: TF32 12.3 ms/step on an H100 (E203 `timing.json`), so 20 min of training. Scoring took 8 min on E203's sides
(48.9 min in all against 41.0 of fp32 training). Site B is estimated 4 min more per crosscoder for the extra forward.
H100 at $4.0-4.4/h (E212, E203). Set-up is 10 min a pod. Each pod's transfer is 1.1-1.5 GB at 10 MB/s, 2-2.5 min, plus
the decomposition from W&B (3.0 GB, 65-100 s measured).

| Stage | Crosscoders | Pods | Wall | Cost |
|---|---|---|---|---|
| Rehearsal (cluster `debug`) | all jobs, tiny | 1 L40 | ~1.5 h | $0 |
| Timing (first paid pod, in stage 0) | — | — | 5 min | in stage 0 |
| Run 1 stage 0 | 8 | 8 | ~45 min | $22-25 |
| Run 1 stage 1 | 29 | 15 (8 kept) | ~70 min | $62-69 |
| Run 2 | 76 | ~19 | ~2.3 h | $157-172 |
| **All** | **113** | | **~4.5 h of pods** | **$241-266** |

Gated (over $60). This sits at or just over the brief's 85% of $300. The plugin reports a $500 cap, $0 spent this
session. Lever: if the timing stage shows two crosscoders on one H100 giving at least 1.5 times the throughput (peak
14.9 GB of 80; about 15% of TF32 peak used), run 2 packs two a pod, about $110. A fallback cut is 2 seeds on built
sides (−$25).

## Runs (the cell's two)

- **Run 1: controls and seed ranges** (stages 0 and 1, 37 crosscoders, ~$84-94). It is complete on its own. If run 2
  has not started by 11-23 12:00, run 1 is the report.
- **Run 2: the scored pairs, twins and grafts** (76, ~$157-172). It runs only at a site whose controls passed. If G2
  fails at A, run 2 shrinks to the grafts at B (8, ~$17) or is left unspent, and the user is asked.

## Gates

1. **Pre-mortem.** Appendix B, every heading.
2. **Settings sourced.** Appendix A, each to a file and line, with departures.
3. **Rehearsal.** *Placeholder*: activation change per token per side and site, Gaussian c, s/step (L40, both sites),
   peak GPU and host memory, file sizes, resume step diff against its floor, spliced loss, downloads.
4. **Early checks.** In predictions.json: stream length before step 0, finite loss, batch FVE ≥ 0 at step 1,000, FVE
   ≥ 0.5 and dead ≤ 0.3 at 12.5% of tokens, a projected time bar (from the rehearsal and timing), and the Gaussian's
   matched change within 5% before training. *The projected-time bar is filled after gate 3.*
5. **Review.** *Placeholder*: one fresh Opus agent, plan and code together.
6. **Go-ahead.** *Placeholder*: asked by 11-21 evening (two lines: what, cost, time).

## Question for the user (definition level)

The cell asks for "reuse / new / no call by the every-seed rule". This plan reads it against how each pair was built.
A pair is "reuse/new as built" when every new-side seed carries more of its change in fine-tune-specific latents than
every reuse-side seed. It is "reversed" when the opposite holds, and "no call" otherwise. A graft with no partner is
"new" when every seed clears the planted bar (0.5 and its lead latent specific), else "no call". A per-fine-tune
verdict of reuse or new would need a cut-off on this reading, which is a definition change. So is the pair-relative
call what was meant?

## Not run

Diff-SAE, Delta-Crosscoder and dedicated-feature crosscoders. Sites other than A and B. Seeds beyond 3 a side. Qwen.
Size-matched for the three pairs with c ≥ 0.95. Rereading E203's sides.
