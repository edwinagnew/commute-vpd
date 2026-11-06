# Advice: the next step on the neuron baseline line, after E102 stage B

*2026-11-06, Fable's advice for baseline-lead. Read-only; nothing run. Numbers carry their file.*

## Diagnosis: the anchors, not the methods, are the weak link

- **VPD does separate the two Qwen states; it is the label it contradicts.** E94's direct share at matched size reads
  0.030, 0.032 for the TOFU states against 0.013 for the template, half-widths under 0.001, and the same way on the
  maths rows alone, 0.030 against 0.009 (`~/commute-agents/qwen-decomp-lead/experiments/qwen_d3_rows/results/table.json`,
  `comparisons.matched.direct`, `direct_maths`). So VPD reads the state, not the rows, and reads the "new" anchor as
  more inside the decomposition.
- **The neuron readings barely separate anything.** Prompt overlap 0.056 against 0.051 (one band), engaged share 0.50
  for every set, and stage B's judge gate was no call: control 0.50 against a random floor of 0.36 on "tofu", gap 0.14
  under the filed 0.2 (`experiments/qwen_carrying/results/score_stageB.json`, CB1). The judge's power is a real second
  problem, but the unread PB1 and PB2 gaps are 0.10 each, under 0.2, so opening the gate would read level anyway.
- **The pair does not carry its labels at matched size.** The matched TOFU state recalls its facts like the unchanged
  model and is a flatter output (E91, E92); the full state is 37 times the template's size. Task, rows, recipe, size
  and label all differ between the anchors at once, so no reading, VPD's or a neuron's, can be attributed to the label.
  That is the missing positive control: a pair that differs in the label alone.

## Options

**(A) A same-task pair with the label fixed by construction: a real-fact fine-tune beside the TOFU one.**
Train Qwen3-1.7B on TOFU's 217 real rows (`src/commute/qwen_tofu.py` REAL_FILES: `real_authors_perturbed`,
`world_facts_perturbed`) with E84's exact recipe (`experiments/qwen_tofu_forms/`: five wordings, batch 16, warm-up,
peak 2e-5, seed 42), saving the step whose weight norm matches the template's 0.49 and the five-epoch end. The unchanged
model already answers 0.79 of these by choice (E81 control, `reports/analyses/2026-11-04_analysis_e81-tofu.md:148`), so
the update can only change wording and elicitation: under rows 81 and 115 of `decisions.md`, C = "recalls facts about
real people", f_C the true fact, X_C the 217 rows, g = "say it in one sentence", the one candidate E90's Qwen check found
that writes on this format (`reports/analyses/2026-11-05_analysis_e90-qwen-check.md`, TOFU table). The fictional-author
fine-tune is the same format, recipe and rows type with content the model lacks. Read both with the instruments already
built, on each state's own rows and the other's: E94's direct and outside share (`qwen_d3_rows/run.py --states`, one
entry added to `rules.STATES` by qwen-decomp-lead) and E102 stage A's carrying set, prompt overlap and engaged share
(`experiments/qwen_carrying/run.py --states`), with E102's random change as the floor. Beside: real-fact recall after
each update, since TOFU training dropped it from 0.79 to 0.44 (E81 real after), which a reuse update must not do.
*Tells you:* whether any reading separates reuse from new when everything but the label is held fixed. None does: the
methods are the problem, write up. One does: a baseline-against-VPD row with a label that is not an assumption.
*Control first:* W0's log P gain on the real-fact answers at the matched step must be at least half the template's 0.67
nats a token, else the pair is "no gain to carry" and stops. *Cost:* ~340 steps (1,085 sequences, batch 16, 5 epochs)
against E84's 1,250 in 33 min on an L40, so ~10 min training, 5-15 min of Haiku wordings, E94 ~25 min a state, stage A
~2 min a state: two `debug` jobs, $0, H100 fallback ~$4. Under the gate. *Collides with:* `decisions.md:150`, "No new
example update before 11-11", taken by default and not chosen; this is a control variant of the existing anchor, same
dataset and code, but the user should say so. *Next:* a separating reading gets stage B's judge on its sets only;
otherwise option B with the pair as the headline "Not run" closed.

**(B) Stop and write up the comparator table as it stands.** $0. The table says: on Qwen neither VPD nor the neuron
readings separate the two anchors in the labels' direction, the judge has no power on Qwen, and the anchors differ in four
things at once so nothing was attributable. *Tells you* nothing new; it fixes "unreadable on Qwen" as the line's answer
by 11-18. *Next:* (A) in the write-up's Not run list.

**(C) E102 Amendment 1: re-judge with the prose bias removed.** Add a fifth description, "ordinary web prose, no task",
rerun the 26 judge calls, no GPU, ~1 h of agents. If the random "tofu" rate falls to ~0.2 while the control holds 0.5,
CB1 passes. But PB1 and PB2 then read gaps of 0.10 under a 0.2 band: the gate opens onto level readings. Fold into (A)
only if a carrying set there separates.

## Recommendation: (A)

Both methods miss the same anchors the same way, and the anchors cannot carry the label at matched size; a same-task
pair is the only $0 way to tell the method from the label, and it reuses E84's recipe, E94's and E102's code unchanged.
It keeps leads separate: baseline-lead trains and reads the neuron rows, qwen-decomp-lead adds two names to E94's list.

**What would change it.** The gain control fails (real-fact gain under ~0.3 nats a token at matched size): (B), with
the pair reported as unreadable for the stated reason. The user holds the 11-11 line: (B) now, (A) after. E94 cannot be
rerun this week: run the neuron rows alone; the pair is still the neuron method's first positive control with a known
answer.
