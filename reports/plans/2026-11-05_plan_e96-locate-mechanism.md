# Plan: can we find what a 67M fine-tune runs through without being told where to look? (E96)

*mechaism-detection-lead, 2026-11-05. Status: drafted for the user; no code, no predictions filed. Task: `~/commute-data/agents/mechaism-detection-lead/task`. Limits: cluster `debug`, $0 (the task file reads "-bash", a shell expansion of "$0"), two amendments, stop by 2026-11-12 or at a failed control.*

**Limit on every claim.** A new circuit built from old parts still reads as reuse. Finding a head says the behaviour runs through that head, not that one old circuit is unchanged.

## The request, in the Terms' words

Given W₀, a fine-tuned state, the task's windows and a control task on the same windows, name a part of the model such that switching it off removes the behaviour and spares the control. Nothing may use the prefix score or the task's construction to choose the part. First on E62's copying map, where the answer should be head (2,4). Only if that works, on one strengthening task.

## Design check

**Goal.** `GOALS.md` asks whether post-training teaches new capabilities or strengthens existing ones. The definition (`decisions.md`, rows 115, 121) gives a label only through a located mechanism or a value removal that clears its controls, and on the old tasks only E62's pair qualifies. A locator is what could give the strengthening tasks a reading at all. The link is direct for the label, indirect for the map: it does not test any splitting of an update.

**Nearest tried.**
- **E62** found (2,4) by prefix score, on W₀, before any fine-tune. Its sweep over heads stopped at (2,4), (1,1) and five random draws per state: most heads have never been read on a fine-tuned state.
- **E62 Amendment 5** swept all 24 heads, on W₀ only, under a frozen mask. No state.
- **E66** patched block inputs and ablated neurons. Neurons are dead at K = 4096: the hand write loses 0.77-0.95 where matched random neurons cost it 0.37-0.68. We keep neurons only well below that size.
- **E55 C3 and its ladder** ran Jain et al.'s pruning on the 67M's MLP units at K 13-320 to read *revival*, with no gap between the poles. It never asked whether salient units carry the gain.
- **E59, E70** removed probe directions. E70's direction read the count perfectly and did nothing when removed, so a probe here only nominates; it never labels.

New here: effect-ranked heads on a fine-tuned state, and a test of whether any cheap nominator agrees with the true effect.

**Patch or step?** A step. It is new apparatus, asked for by name, not a fix to E62.

**Decisions relied on.** Rows 115-117 and 121; E62's mean ablation and share. **One departure from the literature:** Jain et al. Alg. 3 scores MLP units by gradient times weight to prune and watch a capability *revive*. We use the same score with the importance sign, on the task's gain, and also on head outputs, because the aim is location, not revival.

**What would change what we do next.** See the tree below. A failed control ends the work. A pass on lowercase with a spared cipher gives a constant-answer task its first mechanism reading; "no part found" tells the definition's owner that this route stays closed at this grain.

## The teeth problem, said plainly

The control is weaker than it looks. E62 already shows (2,4) at 0.47-0.56 of the gain with a median other head near 0.01, so an exhaustive sweep of heads will very likely name it. And choosing heads as the unit is itself being told where to look. Three things give it what teeth it has:

1. **The other 18 heads are unread on any state**, so another head could beat (2,4).
2. **Cheap nominators are scored against the sweep.** They are what could scale past 24 units, and they can plainly fail.
3. **The pool holds non-head parts too** (salient MLP units), so a head has to win against something.

The real test is the strengthening task, where nobody knows the answer. Passing the control earns the right to try; it shows little by itself.

## Procedure

Mean ablation as in `copying.ablated`, W₀'s means recomputed on natural text (not read from `heads.pt`, which also holds the prefix scores). Gain G and share as in `copying.share_lost`, both W₀ and the state ablated. All reads on E62's saved 200 read windows; no training.

**Nominators, each saved in full before any scoring:**
- **Head salience (primary).** Gradient of log-odds at the read position times (mean − output), per head, state minus W₀. One backward pass. This is the head-level form of Jain's score.
- **MLP-unit salience.** Jain's gradient times weight on the state's gain; top K = 320 per the top of E55's ladder, against 20 random sets of the same size.
- **Probe reading.** A ridge from each head's output to the target token, fitted on half the windows, ranked on the other half. Expected to favour heads downstream of the mechanism.

**The sweep (reference).** Each of the 24 heads alone on W₀ and six states, then the best pair found greedily.

**Scoring** runs last, in its own function, and is the only code that names (2,4).

## Stages, control first

**Stage 0: the apparatus, alone; exit on a miss.** G on all six states equals `run.json` exactly (E62 Amendment 2 re-read at difference 0.0). The zero update reads G = 0 and salience 0. Share under (2,4) is *not* read here.

**Stage 1: the control, on copy and lookup, three seeds.**

| Number | Expected on copy | Expected on lookup | Source |
|---|---|---|---|
| Sweep's top head, and its share | (2,4), 0.47-0.56 | no head over 0.2 | E62 |
| Second head | (1,1), 0.23-0.29 | (1,1), about 0.05 | E62, E62 A2 |
| Best pair's share | (2,4)+(1,1), 0.67-0.74 | about 0.06 | E62 A2 |
| State's probability of the target, top head off | 0.024-0.071, from 0.99 | unchanged | E62 A2 |
| Head salience: rank of sweep's top head | lean: top 3 | not (2,4) | new |
| MLP-unit salience: share over random sets | lean: small, not specific | | E55 ladder, E66 |
| Probe: rank of sweep's top head | lean: not top 3 | | E70 |

**Located (filed rule).** On every seed, the part must (a) take the state's probability of the target at its mean log-odds to 0.1 or less, or for a task W₀ already performs, take a net share at least the control's weakest seed; and (b) cost the control task's state a share of 0.2 or less (E62's filed bar). The 0.1 comes from E62 Amendment 2's 0.024-0.071. E62's 0.5 share bar is not reused: it had no reference and seed 2 missed it by rounding distance.

**Control passes** when the sweep's top head is (2,4) on all three copy seeds, meets the rule, and lookup names no part that also meets it on (2,4). Edge cases to run through the rule before filing: a tie at the top, a share past 1, a seed that disagrees, a gain under the floor.

**Stage 2: only if stage 1 passes.** One strengthening task, filed after stage 1 is read, bars taken from it. **Recommendation: lowercase**, E53's saved pair: lowercase and cipher at step 300, seeds 0-1, on Pile text, so the spared control already exists. Read as log p(lowercase) − log p(capitalised) at the case positions. Differences to name: the states are bf16 (re-read against the stored gain first), and the behaviour is broad, so the honest lean is **no single head found**. The ladder has no same-rows control saved.

## Tree

- **Sweep misses (2,4), or the rule fails:** stop and report. A top head other than (2,4) would itself challenge E62's label.
- **Sweep passes, head salience fails:** head-level location works but does not scale; stage 2 runs the sweep alone at head grain.
- **Both pass:** stage 2 with both.
- **Stage 2 finds a part, cipher spared:** report it as a mechanism reading on lowercase, with the limit above. Whether it changes that task's verdict is the user's, under row 121.
- **Stage 2 finds nothing:** report "not found at head or K = 320 grain", scoped to that range, not "no mechanism". At most one amendment for a coarser or finer grain, then stop.

## Cost, saved files, reviews

One `debug` L40 job, under 10 minutes by E62 Amendment 2's ~4.5 min on the same states. Under the gate, so no review or independent check unless a result reaches a headline. Saved: per-state vectors for all 24 heads, every nominator's full ranking and MLP scores, both units, in `experiments/locate_mechanism/results/`.

## Not run

A planted head (a fine-tune allowed to change one drawn head only), the one known-answer check not resting on E62, held for an amendment if the nominators' reading is ambiguous. Direction-level units. Other control tasks. Anything on Qwen.
