# P2.87: a gain-reaching random-subspace reference for the log-odds re-gating share (plan, one page)

**Cell** P2.87, due 11-28, 1 run, cap $60 and 6 h (user 11-26 ~20:18 and ~21:05 in whip-lead's chat). Descriptive:
values per task beside the fine-tunes, no verdict. Card `experiments/subspace_share/card.md`.

## Design check (/design, one line each)
- **Goal.** Part 2 of `GOALS.md`/the plan: the re-gating share is the paper's headline reading; it has no random
  reference on the tasks built to reuse, so a reader cannot tell a reuse task's high share from what any update that
  moves the answer would read.
- **Nearest tried.** E310 (P2.82): random changes scaled up to 100x the fine-tune never reached gain 1 on a reuse task;
  E228: KL-matched random changes, no gain. Neither trains, so neither reaches the gain. This one trains, inside fixed
  random directions, so it reaches gain where scaling could not.
- **Patch or step.** A patch on P2.82, asked for by the user's readthrough fill.
- **Decisions.** Relies on the log-odds share as filed (`local_faith.MIN_GAIN`, E136's reader); no departure.
- **Definition.** No verdict under any definition; values only.
- **What would change what we do.** Subspace shares near the reuse values: the paper says the share does not separate
  how the update was built on reuse tasks. Well under: the paper has its missing reuse-side null.
- **Control first.** Stage 0, alone: P2.82's control (zero change reads gain, numerator, KL exactly 0; each of the 24
  reuse fine-tunes' shares equals `known3.json` bit for bit). Fails on a changed reader, rows or card.
- **Numbers.** Per update: held-out gain (expected >= 1 at the "at-1" state; at the end state unknown), share r
  (rival a ~0.75-1.10, rival b <0.42), KL, norm relative to the fine-tune. At W0: gain 0, share nan.
- **Early guards.** Step-0 gain exactly 0 (Z = 0); step-1 update norm logged; loss falls in the first 50 steps.
- **Saved.** Each update's Z, U/V seeds and the two scored deltas (bf16) in `results/large/`; per-update JSON.

## What runs
For each of Fig `known`'s 8 reuse tasks (e62/copy, e46/ident32, e53/lowercase, e54/elicit, c3/rank1, c3/copy,
e58/rank1, e58/copy) and 3 seeds: on every decomposed matrix W (out x in), fixed random orthonormal bases U (out x k),
V (in x k) drawn from the seed, and a trainable k x k matrix Z starting at 0; the update is U Z V^T. k is chosen so
the total trainable count is that of a rank-8 LoRA on the same matrices (Hu et al. 2021's common small rank), stated in
the results. Training uses the task's own fine-tune recipe (data, loss, steps, batch, schedule) with only the
parameterisation changed; the learning rate is the one departure, set from a measured 3-point grid on one task's
seed 0 in stage 1, and stated. Held-out gain is read at each eval; the first state with gain >= 1 ("at-1") and the
recipe's end state ("end") are both scored exactly like the fine-tunes (E136 `Job.trained`, E228 KL, same rows).
A task none of whose seeds reaches gain 1 by the end is reported "no comparable null", with its largest gain; no
further search.

## Cost
67M model, 24 short trainings and 48 reads: cluster debug QoS on an L40 at $0; a 4090/H100 pod if debug does not
start within 5 min (well under $60).
