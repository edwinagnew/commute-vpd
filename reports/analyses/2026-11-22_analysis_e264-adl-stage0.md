# Does the activation difference lens find a narrow 67M fine-tune? Not on its positive control

Cell P2.59, run E264, cluster debug job 63409, 35 s, $0. Plan `reports/plans/2026-11-22_plan_p2-59-adl.md`; numbers
`experiments/adl/results/stage0.json`, log `experiments/adl/results/run_stage0_0-1.log`.

We ported the activation difference lens of Minder et al. 2025 to the 67M model. It averages the difference between the
fine-tuned and base model's activations on unrelated text, then asks how much of the fine-tune's effect lives along that
direction. On the positive control, a copy fine-tune with a large gain, the direction carries almost none of it, so the
filed stop rule ended the run before any pair was read. For the paper, the lens gives no column on our tasks; it is a
baseline that fails its own control here, not one that ranks pairs.

## The answer

The lens did not pass its positive control. The control fine-tune teaches the model to copy a key, raising the answer's
log-odds by 16.45. Removing the lens direction from the fine-tuned model's middle-layer input loses only 0.0028 of that
gain. That is below the filed bar of 0.05, and below the largest of 64 random directions, 0.0048. The cell closes on stage 0, as
the plan filed in advance.

## What was done

The lens direction is the mean, over 8,192 rows of the model's own pretraining text, of the fine-tuned model's
activation minus the base model's, at the input of block 2 and at each of the first five token positions. The paper and
the authors' code fix both choices. The score is the paper's causal test: in the fine-tuned model, set the component along that
direction back to the base model's value at every position. Then read the share of the fine-tune's answer log-odds gain that is lost,
taking the best of the five positions. It reads 0 for an update that changes nothing; a random update of the same size is
the floor.

## What was seen

- The checks of the apparatus passed. The zero update gives a zero direction exactly. The control's gain repeats the earlier
  run's 16.4505 to the fourth decimal.
- The control's best position loses 0.0028 of the gain; the other four positions lose slightly negative shares.
- The size-matched random update reads −0.0001. The 64 random directions read 0.0003 on average, with a largest of 0.0048.
- The logit lens of the control's direction puts no answer token in its top 20. The best answer token sits at rank 207, against
  rank 1,682 for the random update and about 518 by chance.

## Why it may fail here

The copy fine-tune's change depends on which key the prompt holds. A change like that cancels in a mean taken over
unrelated text, and the plan listed it as what the lens is blind to. The paper's fine-tunes are on whole documents in a
narrow domain, where one constant shift is expected. A control whose change is one constant shift, such as the lowercase
fine-tune, would test the port itself. That run was not made.

## Departures from the literature

As listed in the plan: a port to our hooks instead of nnsight; the model's own pretraining text instead of fineweb; no
Patchscope and no graded token relevance; the causal effect read as answer log-odds, not cross-entropy on the fine-tuning
documents; random directions from pretraining text, not chat data; no steered generation.

## Not run

- The 129 pair, key-task and random items: stopped by the filed rule.
- A control with a constant-shift fine-tune, which would separate a port bug from the lens's blindness to key-dependent changes.
- The steering-share version the cell named as an example.
