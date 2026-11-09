# A working definition of reused against new

**Rejected by the user as ad hoc, 2026-11-09. Replaced by `2026-11-09_plan_general-definition.md`. Kept as the record.**

*definition-lead, 2026-11-09, second version (the first is in git). For you to approve, change or drop; nothing here
has run. Both papers read in full from the PDFs. Sources, the design check and the numbers per fine-tune are in
`2026-11-09_plan_definition-page-appendix.md`.*

**In short.** I recommend a definition that asks one thing: **can the original model, with every block but the last
left exactly as it was, learn the task and get it right on inputs that training never showed it?** If it can, and the
real fine-tune is right on those inputs too, the fine-tune reads "reused". If the real fine-tune is right and the
frozen model cannot be, it reads "new". If the real fine-tune only memorised, it reads "new". It would give a reading on
14 of our 17 kinds of saved fine-tune and needs no readout and no named part. Its weakness is plain: it shows the old
layers are enough, not that the fine-tune actually went through them. That is the one question I need from you, at the
end.

## What "working" has to mean

Your tests: it gives a reading on most fine-tunes and agrees with those built to be one or the other, and it is precise
enough to defend as a definition. It must look inside the model; right answers alone are not enough.

## Why the old definition gave no reading

It asked first for a linear readout of the capability, trained on ordinary text. That step fails on most tasks: key
tables are just "which token is here", which a shuffled control reads as well, and copying has too many values (0.03
accuracy before the last block). Only the case-label and repeat-label tasks passed it, and then removing that one
direction took at most a fifth of the gain. The fine-tune clearly generalised, to unseen words at 0.96 to 0.998, so the
direction removal was measuring whether the fine-tune uses that one direction, which is narrower than whether it uses
what the model had.

## The candidates, scored on your tests

Readings are out of 17 kinds of saved fine-tune on the small model.

| Candidate | The question it asks | Reading on | Looks inside? | What a reviewer would say |
|---|---|---|---|---|
| The frozen definition: readout, then direction removal | Does a readout of C exist, and does removing it break the fine-tune? | 4, all "not through C"; "reuse" on none | Yes | Fine as a definition, but it never says yes here |
| Switching off a part found away from the task (last version's choice) | Does switching off an old part take the gain? | 2 (copy and its lookup) | Yes | A definition of copying. And a copy fine-tune trained with the two copying heads held off still reached 94% of the normal gain, so needing a part is not what reuse has to mean |
| A readout read in the fine-tune instead | Does the fine-tune hold the capability? | at most 4 | Yes | Same failing step, and a fine-tune's own new feature would count as having had it |
| Adding up the gain lost when each part alone is set back | Is the change spread or written in? | 17 | Weights only | It separates copying from key tables, but nothing says low means reused: the one non-copy task built to reuse reads high. It also sits close to the update-size baselines that part 2 will score, so scoring against it is circular |
| **Frozen-body test (recommended)** | Is the original model's unchanged body enough, with only its last block retrained? | **14** | Yes, the body's insides are what is tested | "This shows it could be reused, not that it was." True; see below |

The previous note listed a small function fitted on the original model's internals. This is that idea, with the last
block as the small function. A fitted readout would fail copying again, because a readout cannot name a value it never
saw; the last block answers through the model's own output layer, so many values and unseen values both work, and no
position or site has to be chosen.

## The definition (proposed)

Let **M** be the original model. Let a **task** be: on inputs carrying a fixed marker, the right answer is a named rule
of the rest of the input, with some values held out of training (**unseen values**). Let **F** be a fine-tune of M on
the task, trained on the other values. Let **B** be a fine-tune of M on the same examples, with the same recipe and
number of steps, in which only M's last block may change; everything below it stays at M's weights.

Then:

1. F **carries** the task if it is right on at least 0.9 of unseen values.
2. M **holds what the task needs** if B is right on at least 0.9 of unseen values, and the same training on a body of
   random weights stays at chance.
3. F **reuses** what M had if F carries the task and M holds what it needs.
4. F is **new** if F carries the task and M does not hold what it needs, or if F does not carry it, which includes any
   table whose rule says nothing about unseen values.
5. It **does not apply** if the answer does not depend on the input.

The 0.9 is the bar already on file. Readings are per seed. How close to the bar counts as "no call" is set from the
first check's row noise, not guessed.

Note what it does not say: it names no capability and no part. The reading is about what the original model's body
supplies, in general. A claim of the form "through this head" is a separate, stronger claim, reported beside and not
part of the verdict.

## What the saved fine-tunes would read

Fourteen of the 17 get a reading. Nine come straight from saved numbers, because the fine-tune does not carry:
the lookup, the random map, both text ciphers, the 24-key cipher, the lower-guess map, both shuffled twins, and the
rank-1 map. Five need one short frozen-body training, because they carry and so the body must be tested: the 24-key copy
map, the identity map, in-context copy, case label and repeat label. Three do not apply: lowercase, emoticon, and
the one-pair map.

One row reads against how it was built. The rank-1 map was built as strengthening the model's own top guess, yet it
carries only 0.64 to 0.69 on held-out keys, so it reads "new". Either that map's held-out rule was badly made or this
definition has no good answer for strengthening; I cannot tell which from saved numbers.

## What it takes from the two papers and what it changes

Neither paper gives a usable decision rule. Prakash et al. 2024 finds a circuit on the task itself, in a base model
that already does the task, and shows the circuit alone keeps 0.88 to 0.97 of the behaviour; it switches off nothing in
a fine-tune and has no shuffled twin. So the plan's "causal step from Prakash" was never in the paper. Jain et al. 2024
uses readouts to say a capability persists, and pruning and reverse fine-tuning as separate evidence. It never removes a
readout direction and states no rule.

| From | Theirs | Here | Why |
|---|---|---|---|
| Jain | Capability is a linear readout on pretraining data | The model's body as a whole, read by retraining its last block | The readout is the step that failed |
| Jain | Reused means the fine-tuned map is a function g of the capability, g unbounded | g is the last block, trained by the same recipe | A bound, and held-out carrying (your dated rule) instead of unbounded g |
| Prakash | Circuit found on the task, then kept alone | No circuit found | Finding a circuit is what could not be done beyond copying |
| Both | No frozen-body training | Frozen-body training | Freezing the body and retraining the head is the standard transfer-learning test (Kumar et al. 2022 uses it). I recall Raghu et al. 2020 uses it to define "feature reuse", from memory, unread: I will read it before anything cites it |

## Three pairs to test it on

Each is a fine-tune built to reuse, read over three seeds, beside its shuffled-label twin.

| Pair | Real should read | Twin should read | Why | What I expect |
|---|---|---|---|---|
| Copy at a new marker | reused | new | The original model copies, and its copying heads sit below the last block | Most uncertain, and the most informative: if the last block alone cannot reach the marker's answer, this reads "new" on the case everyone treats as reuse |
| Case label | reused | new | The case signal reads out at 0.99 on held-out words in the middle of the model | Reused |
| Repeat label | reused | new | "This word occurred earlier" reads out at 0.95 on held-out words | Reused |

The twins are at chance on unseen values by construction, so they test only that nothing memorised is read as reuse.
The hard test of the reading is on the real arms.

## Where it could fail

- **It may say "could" and never "did".** A fine-tune can build a new route around what the body holds. Nothing here
  rules that out. This is the question at the end.
- **It may collapse into carrying.** Nine of the 14 readings need no inside test at all. If the frozen body can always
  do whatever the fine-tune does, the inside clause adds nothing and the definition is only carrying, which you said is
  not enough. Run 4 below looks for a case where they differ.
- **The last block is large here,** one of four blocks, so it could learn something new on top of the old body. A
  new circuit from old parts reads as reuse, the limit every method in this project has already.
- **The opposite of the old risk.** The old wording risked reading reuse only on copying; this one risks reading reuse
  on anything simple. The random-body control is the check.
- **It says little about strengthening** a thing the model already does, as the rank-1 row shows.

## What would make us drop it

Any one of these: the random-weight body reaches the bar, so the last block can do the task alone; the real copy pair
reads "new" on at least two of three seeds; or no task can be found where the fine-tune carries and the frozen body does
not, in which case the definition is carrying under another name. The plan's gate is "reuse" on all three pairs and
"not" on the twins, three seeds each. If that fails, part 1 of the paper becomes why existing definitions fail.

**The saved copy results read under this wording:** the three copy seeds carry (0.995, 0.995, 1.000) and the
lookups have no rule on unseen values, so they read "new". Whether the copy seeds read "reused" needs the frozen-body training,
which was never run, so the honest reading is no verdict yet. The head-off shares are kept beside, with seed 2 at
0.471 under its old bar, and the new wording does not use that number.
(`reports/analyses/2026-11-09_analysis_p1-2-e62-reread.md`.)

## Runs proposed (nothing runs until you add them)

Each is one run, at most 30 minutes on the free cluster queue, $0. The first runs alone; if it fails, none of the
others run.

| Run | What | Runs | Done when |
|---|---|---|---|
| 1. Controls | Frozen-body training on a task the original model already does (should read near 1 at once) and on a random-weight body for copy and case (should stay at chance) | 1 | Both read against filed bars |
| 2. Three pairs | Copy, case and repeat label, three seeds each, real and shuffled twin, with the frozen-body training; a new twin is trained for copy | 1 | A reading per seed for both arms |
| 3. The remaining saved kinds | The 24-key copy map and the identity map, read the same way | 1 | The saved rows with a rule all read |
| 4. A rule the fine-tune can learn and the body cannot | One or two new rules chosen on paper first, where the original model has no way of supplying the answer | 1 | A carrying fine-tune whose frozen body fails, or "not found among those tried", which is the drop condition above |

## The one question for you

You said it must look inside the model. This test looks inside the *original* model: it says what the body holds. It
does not show the fine-tune's own route. Is that enough to call it a definition?

- **Recommended: yes.** The verdict is the frozen-body reading; route tests such as the head switch-off are reported
  beside it, on the tasks where they can be run.
- **No, it must be the fine-tune's route.** Then we are back to the switch-off family: precise, but it reads almost
  nothing outside copying, and I would rather say so plainly than add clauses to make it read more.
