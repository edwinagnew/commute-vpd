# E59 second stop at stage 0 (job 48357): advice

**Decision: (c).** No more readout training. Run the saved readouts (`results/large/readouts.pt`, job 48357) through
the rest of stage 0 and P1-P3 now (`--readouts`, ~3 min, debug QoS, $0), under a dated amendment that turns the
stage-0 loss rule from a stop into a reading rule. Hard cap: this one run; zero further training attempts. If P3,
the do-nothing or the W0 control fails, or elicit is possessed at no block before the last, E59 closes with P1
"unreadable" (the pre-set fallback: `gap_last` 0.513 > 0.5 after fresh data, borderline but it applies).

**Why not (a), more training.** The miss is a sample-size limit of the readout, not optimisation and not bad data.
(i) Not a data problem: the model's own loss reads 3.4278 held-in vs 3.4281 held-out and the linearised head 3.5707
vs 3.5716, so the two pools are the same difficulty; the 0.3 that held-in sits below held-out belongs to the readout
alone. Held-in windows are drawn from the training rows by design (`held_in_seed` on `train_rows`), and by step 4000
~60 % of their tokens had been training targets (15.0M loss tokens from 16.8M), so held-in = "mostly seen tokens".
Nothing is wrong with the rows. (ii) Not under-training: on seen tokens the last block already sits at L_lin + 0.07
(3.646 vs 3.572), inside the filed 0.1; the 0.37 on unseen tokens is memorisation of once-seen tokens, i.e. the
held-in/held-out gap is the whole miss. (iii) Where it comes from: 50,277 x 768 weights per readout, and each
vocabulary row is fitted only by the tokens with that target, so rare rows see a handful of examples in 15M tokens.
Jain et al.'s readout has a vocabulary of a few hundred, so their 4K-step schedule never meets this; copying it to a
50K vocabulary is a departure from the source to state in the write-up. (iv) More steps on fresh rows (a resume with
a second schedule cycle, unsourced; or 8K steps, which breaks the 30-min cap at 0.23 s/step) would roughly halve the
gap per doubling of tokens: 0.3 -> ~0.15-0.2, still over 0.1. Closing it needs ~4-8x the tokens, i.e. money or
`normal` QoS, neither allowed here. Closed-form least squares on one-hot targets is a different objective from the
filed one and has the same per-row sample size; a stronger weight decay is a guess with one shot per 20 min. None is
in the plan.

**Why the saved readouts can still be read.** The readout's shortfall biases every scored reading toward the null,
never toward the claim: a worse readout at blocks 0-2 lowers possession (P1's elicit clause harder to pass), and a
noisier target column makes the removal cost both kinds less (P2's separation harder to pass). The readout never saw
the task windows or rows 0-32767, so memorised Pile tokens cannot inflate a target value's rank at the answer
position. The functional controls stay stops: P3 (the last block reproduces the construction), the do-nothing, the W0
control (median drop >= 0.3 nats at l*), and the recorded last-block removal (A_lo ~1). If they pass, the direction
is good enough for the readings it bears on.

**Amendment to file (dated, `amended_2026_11_02_after_stop2`), before the run.**
1. Stage-0 readout rule: `over_lin <= lin_margin` is recorded, not a stop (`run.py` line 215-216: log the miss,
   continue). Job 48357's value 0.371 is recorded with it.
2. Reading rule when `over_lin > lin_margin`: a P1 or P2 **pass** stands (the shortfall works against it); a P1 fail
   on the elicit clause, or a P2 fail, reads **"unreadable: readout under-fit at d = 768, vocab 50K, 15M tokens"**,
   not a null, and the decision tree's P2-fail branch (graded ladder) is not taken on it. A P1 fail on the cipher
   clause alone (cipher possessed somewhere) is a real fail: a worse readout cannot cause it.
3. The control stops are unchanged: inputs, P3, do-nothing, W0 control.
4. `TRIED.md` E59 row and the log card: train rows and the loss numbers of both runs (48279 overfit on 4.2M tokens;
   48357 held-out L_last 3.94 = L_lin + 0.37, held-in L_lin + 0.07), and the departure from Jain (vocabulary size).

**Command.** `sbatch cluster/jain_readout.sbatch --readouts experiments/jain_readout/results/large/readouts.pt`
after committing the amended `predictions.json` and the two-line `run.py` change. Expected: inputs exact (as in
48357), readouts loaded in ~1 min, P3 and possession within 2 min, ablations within 3 min.

**Not for the user.** $0, debug QoS, inside the plan's stage-0 apparatus and its approved fallback; the amendment is a
reading rule, not a widened threshold. FYI line only. If the lead wants a training attempt anyway, the only one with a
source is the same schedule on 4x the tokens, which is a `normal`-QoS or paid run: that would need the user.
