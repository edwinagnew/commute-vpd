# P1.61 (E290): a trained gain fine-tune as a positive control for the kept reading

Cell P1.61 (user 2026-11-23, "yes open the IA3 gain control"; due 2026-11-25), from redteam-lead's P2.71 verdict 5c.
Card `experiments/ia3_control/card.md`; filed settings and rules `experiments/ia3_control/results/predictions.json`.

We have never read a trained update that is reuse by construction. Here we train fine-tunes whose only parameters are
per-neuron gains, on the same five key-grid tasks as the full fine-tunes, and read them with the write share over
the needed set. Where such a fine-tune does the task as well as the full fine-tune, the definition must call it kept
at neurons; if the instrument as filed still reads it low, that is P1.59's "reads low on any trained update" found
in the instrument, not in the fine-tunes.

## IA3, as published (Liu et al. 2022, arXiv 2205.05638)

- Sites: "rescaling vectors on the keys and values in self-attention ... and on the intermediate activation of the
  position-wise feed-forward networks", softmax(Q(l_k ⊙ K^T)/√d_k)(l_v ⊙ V) and (l_ff ⊙ γ(W1 x))W2.
- Init: "l_k, l_v, and l_ff are all initialized with ones so that the overall function computed by the model does not
  change". Foldable: "l ⊙ Wx = (l ⊙ W)x".
- Recipe (section 3.5): "train for 1,000 steps with a batch size of 8 ... Adafactor ... learning rate of 3e-3 and a
  linear decay schedule with a 60-step warmup", loss L_LM + L_UL + L_LN; IA3 vectors pre-trained on T0's mixture.
  Appendix D: full fine-tuning in the same comparison uses 3e-4.
- On a llama-style model, HF PEFT 0.17.1 puts IA3 on k_proj and v_proj outputs and on down_proj's input
  (`peft/utils/constants.py`, lines 154 and 184).

## Arms and what "kept by construction" means at each site

| arm | trained | at neurons | at VPD |
|---|---|---|---|
| **ffn** (ours, after Liu 2022): the positive control | l_ff on down_proj's input, 12,288 gains | each neuron's write is l_j times M's, so every changed neuron is kept exactly; no head changes | descriptive: a neuron gain is a subcomponent gain only where a subcomponent reads few neurons |
| **ia3** (as published, PEFT's llama sites) | l_k, l_v on k_proj, v_proj outputs, l_ff: 18,432 gains | neurons kept exactly; heads not: a key gain moves the attention pattern, a value gain turns the head's write inside its own space | descriptive |
| full (reference, E147's saved step-30 updates) | all 24 matrices | E277's trained items read 0.017-0.038 | E275/E278: 0.027-0.040 |

The filed neuron featurization has one exact **remainder** per MLP layer (the weak part of every neuron, about 97% of
activations). Under per-neuron gains of both signs the remainder's write is not M's up to one gain, so the reading
with the remainders in (as E277 reported it) need not read 1. The scored check is therefore on the needed changed
neurons; the reading as filed carries the rivals.

**Said plainly:** at the filed neuron featurization (the top-activation tail plus a remainder per layer), per-neuron
gains are kept by construction only on the neuron variables (R1), not on the remainder. So E_b over K as filed is not
1 by construction, even for the FFN-gain arm.

## Order and expected values

1. **Stage 0, alone, per featurization** (stops on a miss): identity swap 0 flips; do-nothing reads E_b over K = 1.0
   with dd max 0.0; the trainer at init folds to dW exactly 0 with P(target) unchanged; a built gain edit
   (l = 1 + 0.003 z on every down_proj) measures the floor f0 of the neurons-only reading (expected ~1e-10 from fp32
   rounding; stop above 1e-6) and changes no head.
2. **Train** 2 arms x 5 tasks x 3 seeds with E147's loss, windows, batch 32, clip 1, sampler seed, read at E12's step
   rule (first of 10, 30, 100, 300 with held-out P(target) >= 0.5); Adam at IA3's 3e-3. ~11 s each.
3. **Neurons** (scored): ffn, ia3, full for all 15 states, each beside its matched random (90 items).
4. **VPD** (descriptive): seed 0 of the same three, with randoms (30 items, copy's full taken from E275).

| reading | do-nothing | built gain edit | ffn | ia3 | full | random |
|---|---|---|---|---|---|---|
| 1 - E_b over K, changed neurons | (none changed) | f0 ~1e-10 | ≤ 10 f0 (R1) | ≤ 10 f0 (R1) | - | - |
| E_b over K as filed, neurons | 1.0 | ~1 | 0.2-0.7 (lean A) | below ffn | 0.02-0.04 | 0.003-0.03 |
| remainders' share of dd | 0 | ~0 | the number to watch | - | 0.6-0.8 | 0.5-0.7 |
| gain share vs full | 0 | - | ≥ bar on most tasks | ≥ ffn | 1 | - |
| E_b over K at VPD | 1.0 | - | between full and 1 | below ffn | 0.03-0.04 | 0.013-0.017 |

|K|, n needed changed neurons and the median top-1 minus top-2 margin go beside every E_b; |K| < 100 reads "too small".
**Gain bar (whip-lead's minimum):** per task, min over max of the full fine-tunes' three gains in P(target), from this
run (E147's record: 0.83 for cipher, 0.96-0.99 otherwise). An arm counts for the control only on tasks where its mean
gain share reaches it, so a kept reading at a small gain is not taken for a pass. **Rivals (R3):** on reached tasks,
A when ffn minus full of the same seed is at least 0.2 on all three seeds, B when under 0.2 on all three, else split.

## Edge cases (run before filing)

- Do-nothing and the built edit read 1 by convention or construction: they have no task gain, so R2 keeps them out.
- No needed changed neuron (empty set reads 1 by convention): fewer than 100 reads "too small", not a pass.
- A gain at or below 0 is not standable and would read unexplained: Adam moves a gain at most ~lr per step, so 300
  steps leave every gain above 0.1.
- A full fine-tune with no gain over M (p_full ≤ p0): "not read"; E147's lowest is 0.67, so it does not arise.
- No task reaches the bar: "no call", with the shares reached.

## Cost and time

Training ~6 min. Neurons ~40-100 s per item (E277's trained items, same D shape): 90 items ≈ 1.5 h. VPD ~300 s per
item on an L40 (E278's grid randoms): 29 items ≈ 2.4 h. About 10 cluster debug jobs of under 30 min each, $0, one run
of the cell's two. If the debug queue holds a job past 5 minutes, the neurons stage keeps the queue and the VPD stage
waits; it is descriptive.

## Jobs (phase B, 2026-11-23; code `experiments/ia3_control/run.py`, `cluster/ia3_control.sbatch`)

All on cluster debug, 30-min cap, each running its stage 0 first. The train job goes first; the neurons and VPD jobs
need its gains and stop at once on a missing file. Neurons are 8 shards of 11-12 items and VPD 10 shards of 3 items,
so each job stays well under the cap at the slowest measured item time (100 s at neurons, 340 s at VPD on an L40).

```
sbatch --comment='E290 train' cluster/ia3_control.sbatch train t0
sbatch --comment='E290 neurons' cluster/ia3_control.sbatch neurons n0 --shard 0 8      (I = 0..7, tags n0..n7)
sbatch --comment='E290 vpd' cluster/ia3_control.sbatch vpd v0 --shard 0 10             (I = 0..9, tags v0..v9)
~/commute-agents/fleet/cpu-run python experiments/ia3_control/run.py --stage summary
```

## Departures

1. Data, loss, steps and batch: the full fine-tunes' (E147: CE on the target plus KL to M, batch 32, E12's step rule),
   as the cell requires; the paper uses 1,000 steps, batch 8 and L_LM + L_UL + L_LN for rank classification.
2. Optimizer Adam with no schedule, as E147; the paper uses Adafactor, 60-step warm-up, linear decay. Rate 3e-3 kept.
3. No IA3 pre-training on a multitask mixture; the gains start at ones.
4. Rotary model: the key gain sits on k_proj's output, before the rotation (PEFT's llama placement); on T5 there is none.
5. The ffn arm is ours, after Liu 2022: IA3's FFN site alone, as the clean control.
6. The read step may be later than the full fine-tune's 30 (E12's rule); step-30 gains are reported beside.

## Design check

- **Goal.** The paper's claim 1 states the frozen definition and reads it; a reading that is low on every trained
  update needs a trained update where the answer is known. This is that control, under the frozen definition
  unchanged (`theory/2026-11-22_definition_frozen.md`); no clause is added.
- **Nearest tried.** E231 (P2.46) trained neuron and head gains at IA3's rate on the built pairs and read the
  re-gating share, never E_b over K. E282 fits gains to a finished fine-tune. E277 read full fine-tunes at neurons. None
  read a trained gain update with the definition. Not a patch: it tests the instrument P1.59 doubted.
- **Decisions.** Frozen definition, E_b over K as its literal reading (decisions.md, 2026-11-22), E277's featurization.
- **What changes what we do.** ffn reads A: the low trained readings describe the fine-tunes. ffn reads B with the
  remainders' dd share high: the paper says the filed neuron featurization caps the reading, and §4 names it.
  A neurons-only miss: every E_b reading stops until the bug is found.
- **Control first.** Stage 0 (identity, do-nothing, init, built edit) runs alone and stops the job on a miss.
- **Saved.** Gains per checkpoint (74 KB each), per-item num, den, dd, in_K, changed (`results/large`).

## Not run

The ia3 arm's heads given a pass rule (descriptive only); seeds 1-2 at VPD; E231's built-pair gain states at neurons.
