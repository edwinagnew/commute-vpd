# E99 reset: stay on Jain et al. 2024, and test it on tasks with Jain's shape

*Drafted by Fable for the user, 2026-11-06. Becomes the user's direction when sent to definition-lead.*

## What went wrong

`reports/analyses/2026-11-05_analysis_e99-definition-verify.md` has drifted from the adopted definition. No linear readout is trained anywhere, the label is ablation of a located head pair, and the rules now include a net share, two admissibility clauses, a teeth test, an input-dependence clause and a per-state band rule with pooling. Its departures section lists nine departures from Jain, and the first removes Definition 1 itself.

The point of borrowing Jain's definition is that the paper can say "under someone else's definition, our techniques do X". Every added rule weakens that.

## The definition is `decisions.md` rows 81 and 87, and nothing else

- Possession: a linear readout trained on Pile text at some block recovers the capability on a sub-domain. Threshold against a Hewitt-Liang control task, filed before any run. E59's per-block readout code in `experiments/jain_readout` is the apparatus.
- Reuse: the fine-tuned map is a small function of that readout, and it carries to held-out values (row 87).
- Causal check, the one allowed addition: removing the readout's direction breaks the fine-tuned behaviour and spares a control fine-tune at the same trigger. Pruning and reverse fine-tuning reported beside it, as Jain do.
- Verdicts are relative to a named capability: "reuses C", "not through C", "unreadable", "definition does not apply". A constant-answer task reads "does not apply" and we stop there. No rules are invented to give it a verdict.

**Drop from the definition:** the net share, both admissibility clauses, teeth, clause (ii), per-state bands and the pooling rule. Where the original model's own loss under a removal matters, print both numbers. The forced and forbidden fine-tunes may stay as positive and negative controls; they are controls, not definition. Every remaining departure is stated where it is used (row 89).

## Why the old tasks did not read

Jain's input is a sub-domain marker plus content, and the answer is a function of the content. Our trigger token is their marker. Lowercase, emoticon, ladder and many-cues have no content function, and the key-to-value tables only involve token identity. Copy against lookup at a marker is the one old pair with Jain's shape, and the one that read. Build more pairs with that shape rather than more rules.

## Two new task pairs on the 67M, same trigger structure

1. **Case label.** After the trigger, output one of two label words depending on whether the preceding word was capitalised. Twin at the same trigger: the same two label words assigned by a shuffled word list, so no base feature predicts them. C is case of the previous word. E97 located the case heads and E99's forced and forbidden fine-tunes exist for them, so this is the cheapest pair.
2. **Subject number.** After the trigger, output one of two label words depending on whether the clause's subject was singular or plural. Twin: labels from a shuffled noun list. C is subject number, which small Pile models already use for agreement.

Order per pair: possession probe on Pile text with its control task first, stop if the readout does not beat it; train the pair, two seeds; causal check, held-out carrying, pruning and reverse fine-tuning beside. File expected outcomes before training: the real task reads "reuses C", the twin "not through C". A miss on the twin is a result, not a reason to add a rule.

## Deliverable

One short report per pair with the verdicts and the numbers behind them, and one table putting the copy/lookup pair beside the new pairs. Then the route-split readings at the trigger that the regating line used, so r and the switched-on set are scored against these labels.

Case label first; subject number only if case label reads cleanly. Fast lane, $0 cluster jobs. The Qwen date of 11-18 stands and this does not displace it.
