# Prior plus steering now call every built pair the share's way. What should the comparison do before saying what the share adds?

*2026-11-10, compare-lead, part 2 (comparison on the 67M model). No reuse or new verdict is given, since the definition is
under revision. Nothing below runs without your yes.*

**What was found.** On the nine built pairs of fine-tunes, two cheap readings together call which side reads higher the same
way the re-gating share does, on all nine. The first is the original model's prior: its log10 probability of the trained
answer. The second is steering at strength 1: the fine-tune's mean change of activation added to the original model at its own
size. Each alone misses at least one pair. So on direction, the built pairs no longer show the share seeing something both
cheap readings miss.

- Where the prior is matched, steering carries the pair. The prior gaps are 0.12 and 0.20 decades, under the 0.5 bar.
  Steering separates the same pairs by 0.31 and 0.44 of the gain.
- Where steering fails, the prior carries it. On three pairs steering at strength 1 reads no call. The prior gap there is
  3.2 to 3.7 decades.
- With steering's searched strength instead, three pairs have the two readings pointing opposite ways. That reversal comes
  from how the strength is chosen, not from the new side changing the model more.
- This reads direction only. It says nothing about how far apart the sides are, and nine pairs is a small set.

**The question.** What should the comparison part of the paper do next, before it writes its paragraph on what the share adds?

- **Option a: compare gap sizes from saved results.** Costs nothing and needs no GPU. It asks whether the share's gaps across
  the nine pairs follow the prior's gaps, follow steering's, or carry something neither does. With nine pairs it is
  descriptive only. Next: if the gaps add nothing, go to option d; if they do, that is a lead for option b.
- **Option b: build new pairs where the prior and steering disagree, or where the prior is matched and steering is held near
  zero, and read the share there.** This is the only option that could show the share adds information on direction.
  It needs new fine-tunes on the GPU, a new cell and a run budget from you. Cost: not yet costed. Next: a filed rule
  first, then the run.
- **Option c: effective rank of the update on the 24 refit 96-key pairs.** Runs from saved states and fills a column still
  empty. Next: one more cheap reading in the table. Rank already agrees on at most three of nine pairs, so it is unlikely to
  change the picture.
- **Option d: stop and write the paragraph as it stands.** On direction, the share agrees with prior plus steering on the
  built pairs. Its claim to add something rests on what it measures, which subcomponents are switched on, and not on a
  different call.

**Recommendation: option a now**, as it is free and quick, then option d if the gap sizes show nothing beyond the prior and
steering. Option b only if you want the paper to claim the share adds information beyond the cheap readings. It is the one
option that could support that claim.

Results laid out in: `reports/analyses/2026-11-10_analysis_e149-prior-steering.md`,
`reports/analyses/2026-11-10_analysis_p2-11-method-comparison.md`, `experiments/prior_steering/results/stage1.json`,
`experiments/prior_steering/figures/votes.png`.
