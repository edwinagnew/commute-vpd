# compare-lead, night of 2026-11-10: every baseline is on every built pair, and prior plus steering track the share's direction

On the nine built pairs, two cheap readings together, the original model's prior for the answer and steering with the
fine-tune's mean change at its own size, call which side reads higher the same way the re-gating share does, while each
alone misses at least one pair. So on direction the built pairs no longer show the share seeing something both cheap
readings miss. One decision card asks what the comparison part should do next; nothing runs until you answer it.

**Cells.** `P2.12`, `P2.13`, `P2.14`, `P2.15` filled tonight, one run of one each, all on the cluster's debug queue at $0.
Rung L2.1 ("every baseline on every pair with a saved update") is answered: seven of seven cells filled. `P2.6` stays
open and parked, as you said. Rung L2.2 waits on part 1; L2.3 (the one figure and the paragraph) waits on the card
below. No pods ran; no live jobs or agents of mine.

**What ran and what it found** (each provisional, one run, no independent check):
- Prior on lowercase against the ciphers: the original model rates lowercase's answer 1.63 decades above the ciphers'
  (1.46 to 1.80 over windows), the side built to reuse higher.
  `reports/analyses/2026-11-10_analysis_e145-prior-lowercase.md`
- Why steering reversed the share on three pairs: at strength 1 the reversal is gone on both pairs scored; the search
  over larger multiples causes it, since the reuse sides stop gaining at strength 2 while the ciphers keep gaining.
  `reports/analyses/2026-11-10_analysis_e146-steering-reversal.md`
- The 96-key pairs refit (24 fine-tunes at the original recipe): they reproduce the old shares within seed spread; the
  share separates three of four pairs, and no other reading matches it.
  `reports/analyses/2026-11-10_analysis_e147-refit-96key.md`
- Prior plus steering together: the opening paragraph above; direction only, not gap size.
  `reports/analyses/2026-11-10_analysis_e149-prior-steering.md`

**Waiting on you.** One decision card: `reports/decisions/2026-11-10_decision_what-share-adds.md` (compare gap sizes
from saved results for $0, build new pairs where prior and steering disagree, fill effective rank on the refits, or
write the paragraph as it stands; recommendation in the card).

FYI: rung L2.1's budget reads 5 in the ladder but its cells used 7, the +2 you approved in chat; the ladder is yours to
change. The steering-reversal run scored only two of the three pairs, since one had fewer than five rows. The prior plus
steering writer saw the 96-key results before filing its rule.

Premise checks on all four cells read "doubtful" (advisory, for the 11-18 gate): `~/commute-data/premise/P2.12.md` to
`P2.15.md`.

Friction noted today: job logs under the `cluster/logs` symlink cannot be committed (670); `card check` wants the
literal word "predicts" (672); a zero-run cell has no experiment number for `fleet/numbers` or the log card's name
(679, 681).

## Not run
- Any run on L2.1 beyond its seven cells; `P2.6`; anything on L2.3 before the card is answered.
- Any reused or new verdict (the definition is under revision).
