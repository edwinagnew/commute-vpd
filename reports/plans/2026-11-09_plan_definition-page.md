# P1.1, due 11-10: a working definition of reused against new

*definition-lead, 2026-11-09. For the user to approve, change or drop; nothing here has run. Reading: Prakash et al.
2024, all 26 pages, and Jain et al. 2024, pages 1 to 11, by the lead, from the PDFs. Jain's pages 12 to 72 were read by a
subagent against fixed questions, with quotes and page numbers (`~/commute-data/scratch/definition-lead/p1.1/jain-appendix/notes.md`).
The `/design` check is in `2026-11-09_plan_definition-page-appendix.md`.*

**In short.** The plan's causal step is sound but it is not Prakash's. Prakash keeps only the part and shows the
behaviour survives; finding the part away from the task and the shuffled twin are our additions. I propose keeping the
plan's test under its true name. On the saved copy pair it reads reuse on two of three seeds and no reuse on the twin.
The main risk is that it only ever reads reuse on copying.

## What the two papers actually do

The plan says Prakash finds a part "on ordinary text only", switches it off, and spares a twin. **The paper does none of
the three.**

- **Where the part is found.** Prakash finds the circuit on the task itself: path patching on 300 box-tracking examples,
  in a base model that already does the task at 0.66, against chance 0.14.
- **The test.** Everything outside the circuit is replaced by its mean, and the circuit alone keeps 0.88 to 0.97 of the
  fine-tuned model's accuracy. Heads are knocked out only to prune a circuit or to check that circuit in its own model; the base model's circuit is never switched off inside a fine-tune.
- **The control.** Random circuits of the same size, accuracy near zero. There is no shuffled-label twin.
- **The fine-tune.** It trains on arithmetic or chat, and entity tracking, which it never trained on, is what is measured. Ours trains the target task.

Jain differs too. The frozen wording's direction removal was always ours: no probe direction is ever ablated or patched
in the paper. Jain's linear readout says a capability persists. Its causal evidence is pruning units chosen by the
pretraining task's loss and watching that task come back, and reverse fine-tuning against a scratch baseline. Jain gives
no decision rule at all.

**What this means for the plan.** The sentence "takes its causal step from Prakash" should change. What Prakash gives us
is one idea, the important one: take the part from the original model and test it on the fine-tune without adjusting
it.

## Why the frozen definition gave no verdict

A linear readout could not be trained for most capabilities. Key tables fail because a random relabelling reads as
well, and copying fails because it has too many values (0.03 before the last block). The one pair where a readout
exists, case label, then failed the causal check three ways: removal took at most a fifth, and patching flipped one
label in a hundred. So the readout is the wrong first test, and the proposal drops it.

## The definition (proposed)

Let **M** be the original model and **M′** a fine-tune of it. Let **C** be a rule named in advance: on inputs selected by
an identifier, the right output is a fixed function of the rest of the input (Jain's Definition 1 form). Let **P** be a
set of attention heads of M chosen by a rule filed in advance, using M alone, on data holding no trigger and no
fine-tuning example. Let **T** be a twin of M′: the same inputs and recipe, with labels reassigned by a seeded shuffle.

Then M **has C through P** if M does C clearly above chance, its score falls to near chance when P's heads are replaced
by their mean over ordinary text, and a random set of the same size leaves most of the score.

And M′ **reuses C through P** if all of these hold, with the part left exactly as found:

1. M′ answers right on values it never trained on, at least 0.90 of the time.
2. Switching off P in both M and M′ takes at least half of M′'s gain in log-odds.
3. Switching off P takes at most a fifth of T's gain.
4. The median random part of the same size takes under 0.15 of M′'s gain.

If 2 fails the reading is "not through C". If 3 or 4 fails it is "no call": the part is generic. If M has C through no
such P, or the rule has no unseen values, it is "unreadable". A do-nothing fine-tune fails 1 and is never read. The
bars 0.90, 0.5, 0.2 and 0.15 are the filed ones and are not new here.

**One thing needs your word.** The plan says "ordinary text only". The copying head was found on repeated random
tokens, which contain no trigger and no fine-tuning example but are not ordinary text. Strict reading: only Pile text
may find P, and E62's head is not yet admissible. Loose reading, as written above: any data free of the trigger and the
fine-tuning examples. I wrote the loose one because P still comes from M alone; your words win if you meant the strict
one.

## Departures

| From | Theirs | Ours | Why |
|---|---|---|---|
| Jain | Linear readout defines possession, "for all x" | A part M needs, read by behaviour and filed bars | A readout could not be trained for most capabilities (E116) |
| Jain | g ∘ f is stated, unbounded | No g is named; carrying to unseen values stands in | Row 87, the user's departure |
| Jain | Pruning revives the old task; reverse fine-tuning | Removal in the fine-tune; neither of theirs | Theirs needs a pretraining task to go back to |
| Jain, Prakash | No shuffled twin | Twin (clause 3) | Separates a mechanism from any memorising fine-tune |
| Prakash | Circuit found on the task | P found away from the task | Our M does not do the task at the trigger, and finding it there would be circular |
| Prakash | Keep only the circuit | Switch the part off | The plan's choice; see below |
| Prakash | Head at one token position | Head at every position | Simpler; coarser |
| Prakash | Accuracy | Share of log-odds gain | M already has some probability at the trigger |
| Prakash | Also checks each head group keeps its function, and patches across models | Neither | Out of scope for one label |

Prakash's keep-only test could replace clause 2 or sit beside it. On the 67M it has one reading (E97: copy keeps 0.98,
lookup 0.25 to 0.31), which that report calls mostly by construction, so I recommend switch-off for the verdict.

## Three pairs to test it on

Each is a fine-tune built to reuse C, read over three seeds, beside its shuffled-label twin. "Should" is what the
definition must read to be right.

| Pair | C and P | Should read | Why | What I expect |
|---|---|---|---|---|
| Copy at a marker (E62's recipe, a new twin) | "Repeat what followed the earlier A"; P is the copying head | Real reuses, twin not through C | M copies at 0.825, and 0.010 with the head off | Near the bar: 0.47 to 0.56 on the saved seeds |
| Repeat label (E125) | "This word occurred earlier"; P must be located on M, none yet | Real reuses, twin not through C | Unseen words carry, and M holds the signal | Unreadable or not through: removing its direction took under 4% |
| Case label (E112) | "The previous word is capitalised"; P must be located on M, none yet | Real reuses, twin not through C | Passes carrying and the small-function test | Not through C: no single head or MLP was specific (E114) |

The last two are the point: they are the only pairs not built on copying, and I do not expect them to pass.

## Where it could fail

- **It only reads reuse on copying.** The test needs M to show C as a behaviour and a small part to find. Copying is
  the one capability here with both, sharply. Case label and repeat label have a signal but no known behaviour.
- **One head is a thin part.** Half of a gain through one head leaves the other half unexplained. Prakash's own
  fine-tunes needed 175 heads where the base needed 72, so reuse coexists with added parts.
- **The bar has no reference.** One half was inherited, and E62's seed 2 reads 0.471.
- **Backup routes.** Removing a part can recruit others; Prakash reports this.
- **Old parts, new circuit.** A new route built through an old head reads as reuse, the same limit as re-gating.
- **Removal is not revival.** In Jain's Figs. 57a and 58a, read by eye, pruning takes away the fine-tuned behaviour with
  little of the old one returning. Losing a behaviour alone says little, which is what the twin is for.

## What would make us drop it

Any one of these: the planted control below reads under 0.9, so the test cannot see a known reuse; a twin reads reuse on
any seed, so the part is generic; or the copy pair reads reuse and the other two read anything else on all seeds, which
is a definition of copying and not of reuse. The plan's gate is reuse on all three pairs, three seeds each. If that
fails, part 1 becomes why existing definitions fail.

**P1.2's reading, provisional:** copy seeds 0 and 1 reuse, seed 2 not through C at 0.471, all three lookups not through C
(`reports/analyses/2026-11-09_analysis_p1-2-e62-reread.md`). Note the gate's three-of-three would already fail on this
saved set. The lookup is not a shuffled twin, and on the strict "ordinary text" reading all six rows are unreadable.

## Run cells proposed (nothing runs until you add them)

All four are at most 30 minutes on `debug`, $0, one run each. R1 runs first and alone; if it fails, R2 to R4 do not run.

| Cell | What | Runs | Done when |
|---|---|---|---|
| R1 | Planted control: a copy fine-tune trained with every head outside P held at its mean, so the gain must go through P; beside it the same training on shuffled labels | 1 | The planted model's share is read against a filed bar of at least 0.9, and is reported as the reference for the bar of one half; changing that bar stays your call |
| R2 | Copy pair with a shuffled twin, three seeds per arm, using the part R1 used | 1 | A reading per seed for both arms under the adopted wording |
| R3 | Repeat-label pair, three seeds per arm, with a filed rule for locating P on M | 1 | A reading per seed for both arms, or "unreadable" with the reason |
| R4 | Case-label pair, three seeds per arm, with a filed rule for locating P on M | 1 | A reading per seed for both arms, or "unreadable" with the reason |
