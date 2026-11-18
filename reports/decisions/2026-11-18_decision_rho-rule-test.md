# Open: three questions about how the working definition scores "keeps the original model's rule"

*2026-11-18, definition-lead. Pending definition decision, raised by the run on saved fine-tunes (analysis
`reports/analyses/2026-11-18_analysis_rho-saved-items.md`). `decisions.md` and `GOALS.md` are unchanged.*

## What was measured so far, and did it come out as expected

The working definition says a fine-tune F reuses the original model M, for a behaviour, to the extent that the parts F
needs for that behaviour keep M's rules. We scored that, as the 2026-11-18 theory note writes it, with a number we call
retention. Every neuron and every attention head is lined up with the same unit in M. For each part F needs, we ask at
every input setting whether it writes in the same direction as in M. The score of a part is 1 for always the same
direction and 0 for unrelated. Retention is the plain mean over the parts F needs (about 0.9 of the behaviour is carried
by that set). The settings are pooled from both models.

We read it on 20 saved fine-tunes (four counted pairs, each a reuse member and a twin that must learn something new,
two seeds each, plus a fifth pair reported only), on E195's rebuilt circuit, and on the grafts.

- **The controls came out as expected.** A do-nothing copy reads "no change". A copy circuit with its output turned up
  two-fold and ten-fold reads 1, as a change of gain is the same rule. New units written into M's quietest slots read
  0.002, and grafts that add neurons or a head M lacks read 0, as the rule forces.
- **The test of reuse against new did not.** Every fine-tune keeps 0.999 or more. The gap between each reuse member and
  its twin ranks by how far the twin moved its weights (case twin largest, then repeat, then the two key-table
  pairs), at both seeds. That is the filed rival B (update size); rival A (new rules in the attention heads) lost at
  both seeds. The identity/random-map gap was 1.9e-5 and 2.1e-5, below the repeat gap of 8.5e-5 and 9.5e-5.
- **The alternatives move nothing that matters.** Today's aggregate score, a size-aware distance, activation patching,
  reading on M's settings only, and restoring the eight zeroed neurons all keep the same ranking. They differ mainly
  on the rebuilt circuit (0.995 as the note writes it, 0.97 today's aggregate, 0.91 size-aware).
- **New, not a filed rule.** Heads are under 1% of the parts F needs and read 0.976 to 0.995, against 0.9994 or more
  for neurons. Read alone, the heads show the reuse order: the identity/random-map gap is 0.0036 and 0.0039 at the two
  seeds, against the repeat gap of 0.0007 and -0.0024. The plain mean over thousands of barely changed neurons drowns
  this.

So the definition as written does not separate reuse from new on these pairs. It stays "tested, not adopted".

## The question

Which of the following did you mean? Each is a question about the definition's wording, and none has a cut-off or verdict attached.

**Question 1. How should "keeps M's rule" be scored at a neuron?** A neuron writes its activation times a fixed output
column, so the per-setting score sees only the column's direction and when the neuron is silent, never how hard it
fires.

| Option | What it is | Rebuilt circuit reads | Reuse against new on the pairs |
|---|---|---|---|
| A. As the note writes it | per input setting, squared cosine of the two writes | 0.995 | gaps follow update size |
| B. All settings at once | one squared cosine over every setting pooled (today's code) | 0.97 | same ranking, 0.0005 to 0.0039 lower on pairs |
| C. A size-aware distance | per setting, difference between the two writes divided by their combined size | 0.91 | same ranking, 0.011 to 0.031 lower on pairs; counts a pure change of gain as a loss |

None of the three separates reuse from new beyond update size, so the data does not choose. **Recommendation: C**, the
only option that sees a neuron firing harder or softer. A unit change of gain is then a loss, which is a stated
consequence and the user can reject it.

**Question 2. Do two units that are both silent on an input count as following the same rule there?** The note implies
yes. In this run it did not matter: no slot neuron of the slotted graft was ever silent at the same setting as the M
neuron it replaced (shared silence 0), and on the pair fine-tunes both were silent on about one setting in a thousand.
It matters in the turned-up control (about half of the settings) and in any network whose activation is exactly zero for
every negative input, where unrelated units could read as the same rule. Options: yes (the note), no (count only
settings where at least one is active), or count it half. **Recommendation: keep "yes" for now** since nothing here moves, and
revisit when a network shows it matter.

**Question 3. Should retention be a plain mean over the parts F needs, or weighted?** In a plain mean one head counts as
one neuron. Options: (i) plain mean; (ii) weighted by each part's share of the behaviour; (iii) weighted by kind (neurons,
heads and embeddings counted as groups). **Recommendation: open a new cell for (ii)**, reading retention weighted by
each part's contribution to the behaviour, on the same saved items with no new training. The saved per-part arrays may
already allow most of it.

## The numbers it rests on

All numbers are in `experiments/rho_saved_items/results/summary.json` and `parts_summary.json`.

- Retention as the note writes it: mean over the parts F needs of the part's score, itself 1 minus the mean over settings of
  1 minus the squared cosine of the two writes. 1 means every rule kept. Lowest trained fine-tune: 0.9994.
- Gap: the reuse member's retention minus its twin's. Case 5.0e-4 and 4.3e-4; repeat 8.5e-5 and 9.5e-5; identity/random
  map 1.9e-5 and 2.1e-5; copy/cipher 1.4e-5 and 7.1e-6 (seeds 0 and 1).
- Twin's extra update: the twin's MLP weight movement divided by the reuse member's, 2.67 and 2.52 (case), 1.42 and 1.56
  (repeat), 1.13 and 1.13 (identity/random map), 1.09 and 1.05 (copy/cipher).
- Largest spread of one item over five search orders: 1.3e-5 and 1.5e-5.

## What the premise check doubted

The independent premise check rated this "doubtful". Its text, in full:

> The question assumes that how much of the original model's causal structure survives in a fine-tune differs between
> reuse and new-build cases by more than repeat noise and update size.
>
> These readings say it does not. Every fine-tune keeps 0.999 or more. The identity-vs-random gap sits below the repeat
> gap. The gaps rank by update size at both seeds, which is rival B. P1.29 found the same thing with rule retention:
> gaps of 0.00004 to 0.0028, and no cut-offs could be set.
>
> Whole-model retention measures look saturated. A fine-tune leaves almost all of M intact whether it reuses or builds,
> so any score averaged over M is dominated by the unchanged bulk. The one reading that clearly separates the cases is
> P2.35's re-gating share gap (0.39 to 0.70 on every pair). It asks something different: where the new task's computation
> comes from, not how much of M is still there.
>
> So the trouble seems to lie in what "survives" is being asked to stand for, not in rho's settings. Testing the note's
> open choices would refine a quantity that may not carry the distinction at all. The Part 1 gate (2026-11-13) has also
> already passed.
>
> Question for the researcher: what result from a retention-style measure like rho would you need to see before treating
> "how much of M survives" as the right thing for the definition to measure?

This is put to you without a recommendation. Question 3 is partly an answer to it (a weighting by contribution to the
behaviour would stop the unchanged bulk from dominating), but whether retention is the right thing to measure is yours to say.

## Cost

Run 2 cost about $75, over the $60 gated line. About $20 of that was lost to a slow model download and a set-up timer.

## Links

Analysis `reports/analyses/2026-11-18_analysis_rho-saved-items.md`; results `experiments/rho_saved_items/results/summary.json`,
`parts_summary.json`; figures `experiments/rho_saved_items/figures/{controls,gaps,one_change}.png`; theory note
`theory/2026-11-18_definition_causal_abstraction.md`; premise check `~/commute-data/premise/P1.30.md`.
