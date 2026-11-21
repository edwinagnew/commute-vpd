# Under wording A, a new edit spread thinly over many parts reads as reused: what do we do about it?

*2026-11-21, definition-lead. Pending decision for the user. Only the user changes the definition's wording.*

Under wording A, a part counts as kept when the base model's mechanism for it, up to one output gain, can stand in for
the fine-tune's and the behaviour survives. R is the share of the parts the behaviour runs through that are kept.

## What we ran

We read R on six hand-built edits whose label we know by construction, with the bar that the stand-in must keep 90% of
the fine-tune's change from the base model. We expected every reuse edit to read above both new edits. It did not.

- **Reuse edits:** a gain of 2 on some parts reads 1.00, a gain of 10 reads 0.98, a rebuilt circuit reads 0.86. A moved
  circuit is not read: it changes the output by less than the measured noise floor, so the rule leaves it out.
- **New edits:** a computation written thinly over many parts reads 0.98, one written into a few parts reads 0.89.
- **Random updates** of matching size read 0.29 and 0.30, low as expected.

The lowest reuse edit sits 0.12 below the highest new edit, so the filed rule's verdict is "fails". Descriptive only:
the share of changed parts that can stand in is 1.00, 0.74, 0.82, 0.62, 0.60 and 0.29 for the gain of 2, the gain of 10,
the thin new edit, the rebuilt circuit, the few-parts new edit and the random update. The thin new edit's changed parts
mostly do stand in; they are not just diluted by unchanged parts.

**On hold:** the trained pairs (nine fine-tune pairs at matched update size), the copy-versus-lookup worked example and
the five synthetic key tasks (about 50 L40-hours in all). The plan made them conditional on this test passing, and a
reading under a wording that failed its own test cannot be interpreted yet.

## Question

A new computation spread thinly over many existing parts, each of which the base model's mechanism with a gain can stand
in for: should wording A call that reused, and the built label be revisited, or should A be changed so it reads new?

- **(a)** Run the trained pairs anyway, as a description. Tells us how A orders them; cannot be read as reuse versus new.
- **(b) Recommended.** Hold them and first look at why the thin new edit stands in (for example, whether per-part
  gains absorb the new computation). Small, cheap; tells us whether the label or the wording is off. Then decide A's
  treatment of thin edits.
- **(c)** Change how A is stated so a thin new edit reads new. A wording change that only you make; the pairs wait for it.

Recommendation: (b), then (c) if it shows the gains absorb the new computation.

An independent premise check doubts the reading can test the wording at all: the VPD pieces may not separate these
built edits (the moved edit is under the noise floor here too, and the earlier gating test also failed on the same
items), so a failure here may be the featurization's, not wording A's
([check](/mnt/nw/home/e.agnew/commute-data/premise/P1.46.md)).

Files: [summary](../../experiments/kstar_vpd_wording_a/results/summary.json),
[filed predictions](../../experiments/kstar_vpd_wording_a/results/predictions.json),
[plan](../plans/2026-11-20_plan_r-under-a.md).
