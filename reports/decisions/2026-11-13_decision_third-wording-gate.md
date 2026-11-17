# Decision: what part 1 does with the third wording of reuse

*2026-11-13, definition-lead, for the 11-13 gate on part 1 (the working definition). Log card
`decision-third-wording-gate`. Evidence: `reports/analyses/2026-11-13_analysis_third-wording-measured.md` (measured)
and `reports/analyses/2026-11-12_analysis_third-wording-no-reading.md` (by argument). Nothing in `decisions.md` changes
until you choose.*

## The question

The third wording says a fine-tune reuses a part of the original when it is the original with only that part's rule
changed, checked by swapping that part's value between the two models. We tested it for you, without adopting it. It
reads every hand-built edit correctly. It says nothing about any real fine-tune, because no part we named by hand on the
original carries more than a third of a fine-tune's change. Should part 1 adopt it, keep the second wording, or spend
one more run first?

## What we have

- **Built edits read as built.** Constant push, push gated by parity, Always on, Cue, XOR, a random-bit floor and the
  planted fact edit each came back with the grade they were built to have. Carry and reversion were within 2e-5 of 1, at
  bars of 0.8, 0.9 and 0.95. This includes "new computation" (XOR, the floor). The second wording can never give that
  label on a small closed input set.
- **No fine-tune is carried by a named part.** The memorised fact, its twin and eight old matched pairs, two seeds each,
  all read "no carrying variable among the named ones". The smaller of the two swap shares was at most 0.33. Under the
  wording, this is "no reading", not "new".
- **Only description separates the fine-tunes.** In every old pair, the reuse side follows its named trigger more
  closely than the new side. No wording yet turns that into a grade.
- **Independent premise check: "doubtful".** Three wordings have each told planted edits apart from fine-tunes. None
  has told reuse apart from built among fine-tunes.
- **What only you decide.** If the wording is adopted, these settings are yours to fix:
  - which rules count as a part's new rule;
  - whether input tokens count as parts;
  - the swap bar for "implements";
  - which alignments are allowed;
  - whether a part named by hand from the task counts as "fixed from the original alone".

## Options

- **(a) Adopt the third wording as the working definition, with the settings these runs used. Recommended. No spend.**
  The settings are your bar choice among 0.8, 0.9 and 0.95; parts named by hand on the original; and the three map
  classes as run. Part 1 then says: this definition separates re-setting, new trigger and new computation on edits
  where the answer is known. On every fine-tune tried, no single part of the original carries the change, so the
  definition gives no reading there. That is reported as a finding. Part 2 can use the built edits as the cases with
  known labels to test re-gating and the neuron reading against. Limit: the paper has no reused-or-new verdict on any
  real fine-tune.
- **(b) Keep the second wording as the working definition. No spend.** Part 1 says "reused" or "unreadable" on the old
  tasks, and never "built" on small input sets. You said on 11-10 that it would have to be revised if it could not
  detect built. It still cannot.
- **(c) Spend the third wording's last approved run first. $0 on the cluster, about a day.** The run would find the
  part on the original by a linear search over several sites, instead of naming it by hand, and re-read the fact, the
  twin and the old pairs. It tells us whether hand-naming is why nothing carried. The earlier search under the second
  wording closed at most 0.37 and 0.47 of the fact's and the twin's change at one site, so a repeat of "no reading" is
  likely. Part 1 waits a day.

## Recommendation: (a)

(a) is the only choice where part 1 has a definition that can say "built". It reads every edit with a known answer
correctly. Its silence on fine-tunes has a stated cause, and the paper can report that cause. Its weak point is that the
definition then has nothing to say about the real fine-tunes the project started from. (c) is the way to test that
weak point if you want it tested before adopting.

Names in the reports: cells P1.18 (by argument) and P1.19 (measured, run 4 unspent); experiments E173 (built edits),
E176 (fact and twin) and E177 (old pairs); the second wording's adoption is the `decisions.md` row of 2026-11-10.
