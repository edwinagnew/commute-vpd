# Fable's draft — NOT approved: what the re-gating line should run now that "elicitation" is mechanistic

*Fable, 2026-11-02, for regating-lead. Advice only: nothing filed, nothing run, nothing committed. The user's decision
(`decisions.md` 2026-11-02, ask #16): elicitation means the fine-tuned behaviour runs through a mechanism the original
model W₀ already has; carrying to unseen keys is evidence at most. Words as in `GOALS.md` → Terms; r is the route
split's re-gating share, "the twin" a map built to be new. Sources at the end.*

**In plain words.** We teach the small model to use a circuit it already owns, its in-context copying heads, at a new
cue where it did not use them before, so the answer is new but the machinery is old. We check the old heads really carry
the new behaviour by switching them off. Then we ask whether VPD's re-gating reading calls this fine-tune "old
machinery" while calling a lookup at the same cue "new". If yes, VPD reads mechanism, not the model's prior guess; if
no, it reads the prior only.

## Why a new map is needed at all

Every 67M map so far (E54, E56, E58 and its Amendments 1-7) has the mechanistic label and W₀'s prior guess pointing the
same way: the maps whose answer W₀ already gives (elicit, rank-1, copy) are also the only ones that could run through an
old mechanism, and every new-answer map (cipher, shuffle, prior-matched cipher, class map, twin) is a new lookup. r
separates them (0.72-0.92 against 0.23-0.49) but so does a cut on the prior, so nothing yet says which one r reads. The
user's definition makes Amendment 7 consistent rather than a contradiction: the class map reuses a *feature* of W₀'s
embedding, not a mechanism W₀ runs, so "new" is a defensible label for it, and its carry is only evidence. The one cell
that tells the two readings apart is **old mechanism, new answer**: a capability W₀ performs in one context, redirected
to a context where W₀'s prior for the answer is low. The reverse cell, new mechanism onto an answer W₀ already gives,
cannot be built: the optimiser is not obliged to build a new path. Two places supply the first cell: the reuse line's
counting model (E55 C2, at its gates) and, on the 67M, the model's own induction circuit.

## Options

**1. Induction under a new cue, on the 67M (recommended).** Build windows of Pile text with a random token pair "A B"
inserted once, then later "A" followed by the marker; the target at the marker is B. W₀ copies B after a repeated A in
context (E52's control: top-1 0.82 on the second copy of a random sequence), but at the marker the current token is the
marker, so its prior for B there should be low; this is measured first and the option is dropped if it is not. Fine-tune
with E58's recipe, three seeds; the twin is a lookup from A to a fixed arbitrary word at the same marker and windows.
Ground truth, no decomposition: find W₀'s induction heads by Olsson et al.'s prefix-matching score on repeated random
tokens, in W₀ alone; mean-ablate those heads in each fine-tuned model and read the share of the trained gain lost. The
induction map should lose most of its gain, the twin little. Then read r on both. It tells you whether r follows the
mechanism label where the prior cannot. If r reads the induction map high and the twin low, VPD reads mechanism on the
67M and the Qwen table gets its first non-prior expectation; if both read low, r reads the prior and the line's 67M
claim is rewritten as such. Not circular: the label uses attention heads chosen by a published score on random tokens and
a causal ablation, never VPD and never the task's output ranks. One cluster `debug` job, $0.

**2. Run nothing here; score r on the reuse line's counting model.** E55 C2 decomposes the counting model and reads r
and the stand-in ratio on 21 fine-tunes whose label is fixed by construction (count a+1 under a new prompt reuses the
old count; count b is new), where three cheap instruments already pass. It is the cleanest "old mechanism, new answer"
case the project has, it is planned, reviewed and $0 on the `normal` queue, and it is another line's work. This option
waits for it and adds nothing. It tells you the same thing as option 1 on a 3M model with no Pile, and whether the 67M
bands transfer stays open. What happens next is C2's own tree.

**3. The lead's swap test on Amendment 7's class map** (move a key along the two embedding directions and watch the
output follow, class map against its twin). It would show the class map's behaviour runs through those directions, but
under the user's definition a direction is a feature, not a mechanism W₀ runs, so even a clean result labels nothing as
elicitation, and both maps have a low prior, so r's low reading agrees with every label. It also needs re-training, as
A7 saved no weights. Not recommended.

**Recommendation: option 1, with option 2 running regardless on its own line.** What changes it: W₀'s prior for B at
the marker over about 1e-2 (W₀ already half does it; the cell collapses to "own guess", drop to option 2); the
ablation control failing in W₀ (no clean induction heads in this 4-block model; option 2 only); the induction map not
learning in E58's budget or not losing its gain under ablation (then it is not elicitation by the ground truth, report
it, no r is scored).

## The recommended option in detail

**Run order, control first and alone.** (a) W₀'s induction heads: prefix-matching score per head on repeated random
sequences; the heads over the bar are the set H. Mean-ablate H in W₀: second-copy top-1 must fall from about 0.82 to
under 0.3; ablating |H| random other heads must keep it over 0.7. Stop otherwise. (b) W₀'s prior: geometric-mean
probability of B at the marker over the induction windows; expected in the cipher-to-shuffle range (1e-6 to 1e-4),
must be under 1e-3 (elicit's 1.3e-3), stop if over 1e-2. (c) The six fine-tunes, induction map and twin, three seeds,
each over the gate (trained gain in the band E58 fine-tunes reach by step 30). (d) The label: share of each state's
trained gain lost when H is mean-ablated, net of W₀'s change at the same position under the same ablation. (e) r on
every state with `local_faith.route_cells`, as E58.

**Numbers and what each should read.** At W₀: ablation share 0 by construction, r undefined (zero update, E58's "reads
exactly 0" row). Do-nothing reference for the label: ablating |H| random heads, both maps, expected under 0.15 (the floor).
Positive control: W₀'s second-copy top-1 under H ablation, filed under 0.3. Label: induction map loses at least 0.5
of its gain on every seed, twin at most 0.2; floor from the random-head ablation. r: induction map at or over 0.5 and
the twin under 0.5 with a gap at least 0.24 (E58's smallest band gap) reads "r follows mechanism"; both under 0.5
reads "r follows the prior"; anything else "untested". Noise floor of r: 0.045 (E58's largest three-seed range).

**Blind to.** The label sees heads, not subcomponents: a fine-tune that keeps H's key and value paths but reads a
different query direction counts as reuse to the ablation, and whether r calls that redirected query "re-gating" or
"new" is exactly what the run measures, so a high r cannot say which. Carry is not read. One cue, one circuit, one
model.

**Cost and compute.** New code: a window builder on `kv.windows`, a head-score and mean-ablation hook (none exists;
`attribution.ablation_effect` ablates subcomponents, `neurons.py` MLP units), about 150 lines; the fine-tune, gate and
r code is E58's unchanged. No saved state is needed and nothing is re-trained; the head count comes from the model
config. One L40 `debug` job: heads and controls a few minutes, six fine-tunes under a minute each, reads minutes. $0,
well under 30 min, **not gated** under `.claude/CLAUDE.md`. Review: one fresh agent on the ablation code before the
job, since it is new measurement code.

## Design check (`.claude/skills/design/SKILL.md`)

- **Goal.** `GOALS.md` "If no simple Φ works": is post-training new capability or strengthened old? This is the first
  67M test where "old mechanism" and "answer already given" come apart, so it is a step toward the goal, not a patch.
- **Nearest tried.** `backlog.md` 2026-10-22, the user's own induction fine-tune idea, dropped because plain induction
  is saturated; its "new cue" variant was never built. E58 A7 built a new rule from an old feature; this builds a new
  cue for an old circuit. E59 tried a VPD-free label by linear readout and was unreadable; this label is causal and
  head-level, which E59 never reached. E55 C3's pruning and reverse fine-tuning failed their control on the KV maps;
  not used here.
- **Decisions relied on or departed from.** Row 73 (granularity limit) kept and stated. Row 81 (Jain Def 1-2 plus the
  causal requirement): this label is the causal half with heads as the possessed capability; no linear readout.
  **Row 87, the bounded-g departure, is a behavioural rule (g must work on held-out values) and now conflicts with the
  user's words**: under the mechanistic definition it is evidence, not the label, and should be re-dated as such. Row
  94's programme (score r against the reuse line's labels) is served by option 2.
- **What would change what we do.** "r follows mechanism" sets the Qwen KV table's expectation beyond the prior;
  "r follows the prior" rewrites the 67M claim and makes C2 the line's only mechanistic test.
- **Apparatus arm.** (a) and (b) above, first and alone, with stops.
- **Class.** A new task on the existing model and harness, in Q5's "mechanism or leak" next steps; the user has asked
  to see the test, so it waits for their word anyway.

## Granularity limit and departures from the literature

A new circuit built from old subcomponents, switched on in a new pattern, reads as elicited to VPD; here the redirected
query is such a case, and neither r nor the head ablation can split "same circuit redirected" from "new circuit of old
pieces". Every claim says "elicited at the level of VPD's pieces". Departures: Olsson et al. 2022's prefix-matching
score is used as published, but the head bar, mean ablation (they report zero and mean ablations on larger models) and the
4-block 67M are ours; Jain et al. 2024 Def 2 takes a linear readout as the capability, we take a head set and add the
user's causal requirement; the 0.5 cuts on the ablation share and r are E12's convention; the 0.24 gap is E58's own band
gap, not a published threshold; E58's recipe (Adam 3e-5, 30 steps) is this project's.

**Not run.** Nothing. Sources: `GOALS.md`; `decisions.md` rows 73, 81, 87, 94, 112; `backlog.md` (2026-10-22 induction
row); `reports/plans/2026-10-30_advice_regating-next-after-a7-FABLE.md`; `reports/analyses/2026-10-30_analysis_regating-synthesis.md`;
`reports/analyses/2026-10-30_analysis_e58-open-pair.md` (Amendments 6-7); `reports/analyses/2026-10-29_analysis_e52-baseline-neurons.md`
(induction control); `reports/analyses/2026-10-31_analysis_e55-c3-ladder.md`; `../baseline-lead/reports/analyses/2026-11-02_analysis_e59-jain-readout.md`;
`reports/plans/2026-10-31_plan_e55-c2-vpd.md`; `~/commute-data/agents/{reuse-lead,baseline-lead,regating-lead}/handoff.md`.
