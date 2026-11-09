# Decision: hold one new task pair unread for part 2's headline claim

*compare-lead, 2026-11-09, for the pending-decisions card. Soon: due 11-14. Nothing has run; no job and no experiment number were used.*

**The question.** Part 2's headline claim needs one pair of fine-tunes that nobody has read before the claim is fixed. Do you approve holding a new pair this way, and which claim should its one number test?

**The evidence.** Every pair in the current definition page has already been read. A pair the definition proposes but no one has trained, the forced and forbidden case label, would be read to set the definition's bar, so it is not untouched either. All leads run as the same Unix user, so file permissions separate nothing, and the job's seeds and states have to be unreadable to them.

**What would run if you approve.** A fine-tune whose label depends on whether a bracket is open, three seeds, beside a twin whose labels are shuffled per word, on the small model. The training job encrypts every state and its own seeds to a key only you hold. One scorer, committed before any training, reads the states once, after part 2's other work closes, and returns the real seeds' mean share minus the twins'. It gives no verdict. Free queue, one card, about 20 minutes (an estimate, not measured).

**What could go wrong.** The original model may not act on whether a bracket is open; a check on the original model alone stops the pair before any fine-tune is read. The task may read as "wrapped" rather than reused under the wording you adopt. A process of the same user on the GPU node could read the job's memory while it runs, but only on purpose.

**The options.** Each is a separate choice; the page's four questions.

- **Which claim the number tests.** (a) The share tells a fine-tune built to reuse from one built to be new, at the same prior, read on real against twin; that is the hard case, where the share overlapped on three of four pairs. (b) A new copy-against-lookup pair, the share's best case, but the family read most.
- **The task.** (a) Bracket order. (b) Another task you name.
- **How the states are held.** (a) Encrypted at the source to a key on your laptop. It needs `gpg` on your laptop and on the GPU worker, which is unchecked until the job starts. (b) A pod you start yourself, downloaded to your laptop, a few dollars.
- **If no wording of reuse is adopted by the part 2 gate.** (a) The number still stands, as a number with no verdict. (b) It waits.

**What happens next.** You approve, change or drop the page. If approved, you add the run as a new cell, make the key pair, and start the job from your terminal. Nothing runs without that.

**Recommendation.** (a) in all four.

*Names in the reports: this is cell P2.7; the page is `reports/plans/2026-11-09_plan_sealed-pair.md`, appendix `_appendix.md`.*
