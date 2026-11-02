# Advice on E66's stop: what the probe read, and whether anything is worth another job

Fable, 2026-11-02. Advice only; the user decides. Read: `experiments/interchange_67m/results/{run.json,predictions.json}`,
`run.py`, `src/commute/ctxpatch.py`, `perturb.py`, the job log `cluster/logs/commute-interchange-67m-48971.out`, the plan's
Amendment 1 and its review note, and my earlier review. No model was run.

## In plain words

The job built its apparatus correctly and read the original model on six maps, then stopped because neither hand-built
control came out usable: the "amplify" edit broke the model instead of strengthening it, and the "write" did not fit well
enough to pass the bar filed for it. The patch ratio T itself behaved as designed on every check that could be run; the
units share U failed its negative control on every map, and the numbers say that failure is built into the method on this
model. One cheap question the probe was meant to answer is still open, and the code to answer it is already written.

## What was checked, and corrections to the brief's list

- **Amplify (confirmed, and worse).** Alpha hit the cap of 8 on all six maps; gain -3.9 to +4.1 nats against targets of
  12 to 21 (the saved states' mean gain on the U windows, `run.py` line 146). It multiplied 1648 to 4096 neurons' output by 9
  and lowered the target on three maps. The target is unreachable by construction: mean-ablating all 12288 neurons moves
  W0's half-known log-odds by only 2.8 nats (`stage0.K_falls`), so no scaling of W0's own neurons can give 14 nats inside the
  regime where "reuse by algebra" holds. A failed construction, not a reading on T.
- **Write (confirmed, one nuance).** `rel_err_mean` 0.45 to 1.41 against the filed 0.33; `cond_A` passes (9 to 10) on four
  maps and fails only on the lookup (30.6, alpha 43). Where the fit is fair, the write's T is as the plan expected: 0.99,
  1.09, 1.06 at l* = 1 on ident32, rand32, rand32b, and 0.96 to 1.07 at block 3. It reads "high" on ident32b (1.25 at block 2,
  3 s = 0.20) and on the badly fit lookup (2.68). So T's low reference did exist on three maps; the filed bar voided it.
- **Stage 1's stop fired on "no map readable"** (`ctxpatch.stage1_status`): C5 is void wherever amplify is capped or the
  write unconditioned, which was everywhere. Amplify's T was never scored against the write. Not a measured miss.
- **K = 4096 and U (confirmed; artefact).** The fall is 0.99 nats at K = 1024, 1.95 at 4096, and negative at K of 64 or
  less: ablating the top-|score| neurons raises W0's log-odds, because `top_abs` ranks by |gradient x activation| and the
  set holds neurons pushing against the target. U at K = 4096 ablates a third of the MLP. The write is `alpha X . mlp3`, a
  linear readout of block 3's neuron activations (`write_state`), so ablating any neurons upstream or at block 3 removes
  part of it: the matched random thirds remove 37 to 68 percent, the top third 77 to 95. U cannot tell a new readout on
  old neurons from reuse, and at this K cannot tell top from random by much. The negative control failed because U, as
  defined, has no "new" that passes it on a model whose updates all read old activations. Amplify's U of 1.5 to 2.1 is
  noise on a broken edit.
- **R(W0) at block 1 (confirmed).** 0.63, 0.67, 0.57 on ident32, rand32, rand32b; 0.497 on ident32b, so its l* is 2 on a
  0.003-nat miss. Lookup l* = 2 (0.95), copy l* = 3 (1.66). Half of my review's condition for reversing the blind-spot worry
  holds: with l* = 1 and the filed steering share at block 1 of 0.11, the copy and random-map states are readable at
  block 1. The other half, copy states above the write there, was never read. On ident32b the filed share at block 2 is
  about 0.75, so its states would be marked blind: the floor sits on a knife edge, and T at every block must be reported.
- **Sharpen (confirmed; the one control that worked).** T at l* 1.03 to 1.32, above 1 by 3 s on three maps; Q 0.98 to
  1.10, so Q cancels global sharpening as built. Its negative G (-0.3 to -2.9) is expected, not an anomaly: W0 disfavours
  these targets, and a temperature of 0.5 pushes a negative log-odds lower. 30 steps brought KL from 0.53 to 0.15, so T
  read about 1.2 rather than 2.
- **transplant_max = 0 exactly** against "about 1e-5". The edit writes a captured block input back bitwise, so the check
  has no rounding noise; it validates position and block alignment only. Not a bug.

## Answers

1. **T**: validated denominator (identity exact, l* on every map, W0's response grows block by block), a working sharpening
   cancellation (Q), and an additive write reading about 1 on three maps. Unvalidated: nothing has ever read high on T, so a
   high T on a state has no reference, and a T near 1 cannot be told from "T sees nothing". **U**: dead on this model as
   defined. K is a third of the network because W0's map behaviour is not concentrated in neurons, and any downstream
   update reads old activations, so the negative control is failed by construction. No U reading has meaning.
2. Two broken controls, one dead reading, and a plan whose question is not yet answered. Rebuilding amplify at the saved
   gain is degenerate (unreachable). A small-K, small-gain amplify would pass by algebra and show only what finding 3 of my
   review already says, that T reads multiplicative scaling of W0's neurons; its place is after, not before, the states.
   U's K rule cannot be fixed by a parameter; the attention side needs E62's head-ablation share instead, if anything.
3. Options.
   - **Stages 2-3 as exploratory on the saved states**, write as the only reference, U not read, no rule filed. $0, about
     10 min of `debug`, no new code beyond a flag; a dated amendment that says the stop stands and these are readings.
     Tells us whether any 67M fine-tune reads above 1 at a readable block, and whether copy and random maps differ.
   - **Rebuild amplify** (K about 256, gain capped at what W0's neurons give). $0; tells us T can read a known scaling.
     Worth it only if the states read near 1 everywhere, to separate "no reuse" from "T blind".
   - **Drop U.** Yes, now; it costs nothing and removes a reading that labels everything reuse.
   - **Key-swap for the class map.** The only row with a VPD win in it; a new `/design`, not before T has read a fine-tune.
   - **E65's counting test.** Returns if the states read nothing. It labels by construction rather than by cause.
   - **Stop and report the 67M cannot be labelled this way.** Premature: the probe's own question is 10 minutes away.
4. **Run stages 2-3 exploratory now** (with U off and T at all three blocks), then write the card. What changes it: if the
   copy-map states read T near 1 at block 1 like the write, T has shown no fine-tune it can see and the key-swap or E65
   is next; if they read above the write and the random maps do not, file rules with a small amplify as the high control.
   Both outcomes are reportable; E62's six states are blind at l* = 3 by the plan's own mark and would not change either.
   Bar check: the best case stays a tie with update size and a win over the prior, as option A accepted.

## Anomalies to run down first

None needs a job. The negative K-falls at small K and U's artefact are read off the code; the sharpening's negative G and
the exact 0 transplant are explained above. One note for the card: the lookup's write (alpha 43, cond 30.6) says the cue
layout's block-3 activations do not separate keys well, so the copy map's borrowed reference is the worst of the five.

## Not run

No model, no job. The "readable" mark on ident32 states uses the filed share from the plan (0.11 at block 1), not a
re-measurement; stage 3 would measure it.
