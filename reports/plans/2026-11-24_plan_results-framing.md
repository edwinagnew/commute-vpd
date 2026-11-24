# Results framing: one ground truth, one role per idea (writeup-lead, 2026-11-24)

**In plain words.** The user's half-done Results rewrite (paper v2 6b221f5) stalled on the copy head and on the
prior, and a review (`reports/reviews/2026-11-24_review_paper-framing-consistency.md`) traced the muddle to four
meanings of "reuse" used without saying which. This page proposes one framing that uses the definition's own two
clauses to name two of those meanings, gives every recurring idea one role, and lists the edits in order.

## The proposal: split the definition into its two clauses

The frozen definition (unchanged) says a fine-tune reuses when the parts F *needs* still compute what M's parts
computed. That has two clauses:

| clause | plain name in the paper | measured by | copy fine-tune |
|---|---|---|---|
| needed: F relies on base parts | **relies on** | switching off base heads; re-gating share; how a pair was built | yes (3 heads off remove 0.52 of the key copy's gain; re-gating 1.09) |
| kept: those parts still compute the same, up to a gain | **keeps** | kept share ρ | no (the head keeps 0.10-0.14 of its change) |

Reuse = relies on **and** keeps. Then:
- **§5.1 (kept share):** built edits pass; trained fine-tunes read low. The copy head is the worked example of
  "relies on but does not keep", and the copy-heads-off figure moves here with it (it is evidence for the first
  clause, not about proxies). It appears nowhere else in Results.
- **§5.2 (proxies on known fine-tunes):** the cheap readings are scored against how the pairs were built, and we say
  plainly that this checks the *relies on* clause only. Re-gating separates every pair; the prior is the confound
  (below) and the 2x2 removes it. No copy-head paragraph, no re-definition of the readings (they are in Set-up).
- **§5.3 (realistic):** readings do not follow the raters; kept share runs against them. Unchanged, minus drift.
- **§5.4 (tools):** unchanged.
- **Claim 2, worded:** "cheap readings detect whether a fine-tune relies on base mechanisms where the answer is known,
  but nothing cheap tells whether it keeps them, and on realistic fine-tunes nothing is validated."
- **The finding this buys:** relying on a base mechanism is not keeping it. Caveat stated once: one head, one model.

## One role per idea

| idea | its one role | what changes |
|---|---|---|
| base model's prior | a **confound** of the built pairs | not a reading: drop its arrow and "expected direction" in Figs 3/5 (keep it as a grey reference column labelled "confound"); Set-up stops calling it a control; Discussion says "a method must be checked against it", not "beat it" |
| copy head | worked example in §5.1 | out of §5.2 and of the labels paragraph |
| re-gating, steering, eff. rank | our cheap readings | Set-up lists them; steering cited to Venhoff as origin, scored as ours; effective rank arrow only if set before the run (else no arrow) |
| tools table | baselines | CMAP-at-neurons stays a tool row, not "a further reading" |
| controls | built edits, random updates, twins | "rescale-only control" vs "gain-only training": one name each, defined once |
| RL vs SFT | Discussion/appendix only | the §5.1 open note moves out (user's own comment) |

## Order of work (if adopted)

1. User picks the framing (below). 2. writeup-lead rewrites §5.1-5.2 to it on v2, green, local, and finishes the
user's skeleton paragraphs with 1-2 numbers each. 3. A blind reader checks Results alone. 4. A glossary pass fixes
the name table of the review (reuse/re-use, kept share only, task names as in Table 1, symbols tau/K/U/Delta) across
body and appendices: mechanical, a clerk. 5. Flags for §3/§4/Discussion/abstract go to the user as a list (the §3
worked example that promises "untouched" heads; the 0.999 neurons line in §3/§4; "improves on baselines" in the
contributions).

## Options for the user

- **A. Relies-on vs keeps (recommended).** Coherent with the frozen definition, keeps every positive result, and the
  copy head becomes the paper's most interesting finding rather than a contradiction.
- **B. Construction is ground truth, definition is a hypothesis.** Simplest rewrite; the user found it uninteresting.
- **C. Definition is ground truth.** The proxies are then wrong on copy and the paper has no positive detection result.

## Not run
- No numbers re-traced; those quoted are from the draft and the consistency review.
- whip-lead not yet consulted on schedule (note sent with this page).
