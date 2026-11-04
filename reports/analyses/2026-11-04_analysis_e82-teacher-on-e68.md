# E82: where teacher training sits on the masked reading of Qwen's first update

Teacher training closes about a twelfth of the KL gap on Qwen under the original model's masks, the same order as adding the whole update directly. We read one saved teacher-training state, trained without masks, on the rows and masks the other methods were scored on. The earlier figure of about a sixth came from other rows and stochastic masks. So this reading cannot tell a good placement from the update itself, and D1 still has no map shown good.

## What we were trying to test

The plot in Q3 of the update notes compared do-nothing, least squares and the adjoint split, all scored by E68, with a teacher-training point borrowed from E76. That point was read on 8 other rows, under stochastic masks with a random share of Δ on, against E76's own reference. It sat above the whole-update line, which invited a reading the comparison could not support. The user asked for only the methods defined in the slides, on one consistent reading. E82 is that reading, and nothing was trained.

The state is E76's clean-loss arm: teacher training towards the first update, with about 200 often-on columns per matrix moved, Γ and Δ frozen, and no masks in the loss. It is scored on 128 maths rows of 256 tokens, with masks from Γ on the original model's forward, Δ off, and the fine-tuned model with no masks as the reference.

Two numbers come out. Let K be the KL from the fine-tuned model to the masked forward, per position, K_D its value for do-nothing and K_0 the KL from the original model to its own masked forward. The share closed is (K_D − K) / (K_D − K_0), so 0 is do-nothing and 1 is as close as the original decomposition is to the original model. The faithful line is K ≤ 1.5 K_0. Gain kept is the drop in next-token loss under the masks, as a share of the drop the fine-tune achieves with no masks; do-nothing reads 0.

## What we expected, and why

A weak lean that the share would land between 0.05 and 0.30, since E76 read 0.16 on the other reading, and a firm lean that it would stay under the faithful line. No lean on gain kept.

## What we observed

### Does teacher training close more of the gap than the whole update?

No. It closes 0.083 against 0.099 for the whole update, 0.062 for least squares and 0.031 for the adjoint split, with the faithful line at 0.97.

**Reading it.** Teacher training is the highest of the three methods and still below a reference that adds every bit of the update past the masks. The gap between it and the whole update is one state on one update, with no noise floor read, so "below" is not a ranking. Figure: `experiments/qwen_pass_test/figures/gap_closed_clean.png`.

### How much of the fine-tune's gain survives?

A quarter. Teacher training keeps 0.247 against 0.258 for the whole update, 0.175 for least squares and 0.087 for the adjoint split.

**Reading it.** Same order as the share closed. Teacher training keeps about what the whole update keeps, and that is only a quarter. Figure: `experiments/qwen_pass_test/figures/gain_kept_clean.png`.

## What we learned

The move from about a sixth to about a twelfth removes the appearance that teacher training gets past the masks. It fits E68, E72 and E74, where frozen masks barely respond to any change of this size. Without masks this state copies the fine-tuned model almost exactly, at KL 0.012 (E76), so the loss here arises under the masks and not in the training.

## How compelling is the evidence

One seed, one update, one state, frozen masks. A low reading may be the masks losing any change this size, and not the map. The control is clean: unchanged components reproduce E68's do-nothing per row to about 1e-6 on the first 8 rows. 32 of the 128 rows, rows 33 to 64, were in E76's training pool; those rows read 0.083 and the other 96 read 0.082, so the overlap does not move the result. Rows and masks both differ from E76's 0.16, and which accounts for the drop was not separated.

## Surprises

None against the filed leans. The result landing below the whole-update line, which E76's point sat above, is the thing that changed.

## What next

Nothing runs from this. It fills the teacher-training row of the Q3 plot in the update notes. Whether teacher training is the map for D1 rests on other reads, the clean-model table and E80's adversary.

## Scorecard

| Filed | Rule | Read | Verdict |
|---|---|---|---|
| Control | do-nothing K and nll per row on 8 rows within 1e-3 relative | K 9.7e-7, nll 0 | held |
| Share closed, weak lean | 0.05 to 0.30 | 0.083 | held |
| Faithful, firm lean | below the line, 0.97 | 0.083 | held |
| Gain kept | no lean, reported | 0.247 | reported |

## Not run

A second seed or state; the other three updates; E76's rows or stochastic masks on E68's states, which would separate the drop's causes; a noise floor between states; masks recomputed on the state's own forward, which E79 read on other rows.

## Provenance

Cluster job 51327, L40, `debug` QoS, about 3 minutes in the log, $0. Code `experiments/qwen_pass_test/teacher_read.py`, `cluster/qwen_pass_teacher.sbatch`, commit `03ddd3af`. Predictions `experiments/qwen_pass_test/results/predictions_teacher.json`, results `results/teacher_clean.json`. Run by a side session of the user's, without a lead; this report was written afterwards by another session from that session's transcript and the result file. No independent check. E82 was first claimed by regating-lead for a different run, which became E83.
