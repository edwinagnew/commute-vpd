# The built control found no map on either arm: revise the wording now, or first test the search on a strongly taught original?

*2026-11-11, definition-lead. Follows the built control on two-key XOR, cell P1.9 of the table of work (one run, spent).
Page: `reports/analyses/2026-11-11_analysis_built-control-xor2.md`. Figure:
`experiments/built_control/figures/h_by_arm_scored.png`. Recommendation: option b.*

**Why this comes to you.** When you adopted the working definition of reuse on 2026-11-10, you said: "if it cant detect
built, then it will have to be revised however." The control built to test that has now run, and the definition did not
detect built. Revising the wording is yours to decide.

**What ran.** The 67M model was fine-tuned toward one target: a fixed push in its activations whenever two keyed bits
differ, the exclusive-or of the bits. One set of fine-tunes started from the plain 67M, which does not compute the
exclusive-or. The other started from a "taught original", the 67M first trained to compute it in part. The taught arm
should read as reuse and the plain arm as built. Each was read by the unchanged search against its own original.

**What the numbers say.**
- Both arms learned the target: the learned check read 0.10 on the taught arm and 0.15 on the plain one, against a bar
  of 0.9.
- The rewiring share measures how much of the change everything the original computed can explain. It read 0.02 on
  both taught seeds, and 0.08 and 0.09 on the plain ones. All four are under the 0.5 cut-off: "no map found".
- The taught original held the exclusive-or only weakly. A linear read of it scored 0.588, where chance is 0.50. It
  cleared the 0.55 bar, but the plan expected above 0.9. Its fine-tune still had most of the skill to build.
- Every case where the search has recovered a map had a variable the original holds as an exact linear read: the
  hand-made cue and table changes, and ordinary fine-tunes copying the table change, which read 0.72.
- So nothing tested lies between a weak read of 0.588 and an exact one. The earlier runs do not settle how strongly the
  original must hold a variable before the search finds the map.

**The question.** Revise the wording of reuse now, or first find out whether the search fails in general or only on
weakly held variables?

- **Option a: revise the wording now.** A new wording is written and filed, then re-read on this run's saved states and
  on the old 67M tasks. Candidate revisions, one line each:
  - *Nonlinear read.* "Rewired" allows a small nonlinear map from what the original computed, not only an affine one.
    Risk: a wide enough map can explain almost anything, so built may never read.
  - *Read the variable inside the fine-tune.* Find the variable the fine-tune's change depends on in its own activations,
    then ask whether the original already carried it.
  - *Built as a positive test, the one I would pick.* Call a change built when the variable its target needs is
    readable from the fine-tune but not from the original, on held-out pairs; the two shares then grade only the reuse
    side. Here that is a capacity read on the saved plain fine-tune, minutes on CPU at $0.

  It tells us whether a wording exists that says "built" on this task. Next: the new wording goes through the old 67M
  tasks before part 2 starts, and the open sensitivity question stays open under it.
- **Option b: first map the search's sensitivity.** One new run of a single cell, $0 on the cluster's debug queue, about
  1.5 h: the same task, the same target and search, with a strongly taught original whose linear read of the
  exclusive-or is above 0.9. To leave the fine-tune something to learn, the original is taught the skill through a
  different output, or for 3000 steps, so its distance to the target stays above the 0.15 bar. It tells us which
  reading holds. If the strong arm reads above 0.5, built is detectable as a contrast once the original holds the
  variable well; if it still reads below, the search fails on ordinary fine-tunes and option a follows with that
  evidence. Next: either a scoped revision or the current wording with a stated strength limit.
- **Option c: keep the wording and report built as unreadable.** No run. Parts 2 and 3 of the paper score methods only
  within matched pairs, the reuse side against its new twin, as already decided for the old tasks. It tells us nothing
  new. Next: part 2 starts at once, and the paper says the definition cannot detect built. This sets aside your
  condition, so it needs your explicit word.

**Recommendation: option b.** I looked for evidence that already settles the sensitivity question and found none: every
earlier success had an exactly held variable, and the taught original here was far weaker. One $0 run tells us whether
the revision must change the search or only state where it works. Option a now would be revising without knowing which
part failed.

**Files.** Page `reports/analyses/2026-11-11_analysis_built-control-xor2.md`; plan
`reports/plans/2026-11-10_plan_built-control-xor2.md`; results `experiments/built_control/results/readings.json` and
`stage1.json`; the definition's adoption, `decisions.md` row of 2026-11-10 "Working definition of reuse"; independent
premise check `~/commute-data/premise/P1.9.md`, verdict "doubtful".
