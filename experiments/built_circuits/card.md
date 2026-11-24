cell: P1.63
rival_a: The order in Fig 2's built block (edits that strengthen or rebuild an existing circuit read a high write share, edits that add a new lookup and matched randoms read low) is a property of the recipes, so this run predicts that order holds (rule P1: every reuse edit above every new edit and random by more than the derangement floor) on both new circuits at all three featurizations
rival_b: The order is specific to Fig 2's site (eight layer-1 neurons on copy24, where M does not do the task), so this run predicts on at least one new circuit (induction at layer 2, bigram lookup at layer 0) the reuse edits read within the floor of, or below, the new edits or randoms at two or more featurizations
rung: parameter
smallest_gap: 0.03
spread: 0.006
n: 2
design: paired
unit: new base-model circuits (induction at layer 2, bigram lookup at layer 0), each read at three featurizations with its five edits against their own matched randoms; descriptive, no bar
spread_source: experiments/needed_neurons/results/summary.json
spread_of: proxy: E_b over K (leftovers counted) of Fig 2's two built randoms at top neurons (0.0086, 0.0174; sd 0.006), the same reading on built states of the same recipes
floor: the derangement floor delta_f measured in stage 0 (x2 on the induction circuit read at scan seeds 268 and 269); a reuse-minus-new gap at most delta_f is read as "does not hold"; rival A expects every gap above it, rival B at least one circuit with gaps at or below it at two featurizations
stop: Any stage-0 miss (M's top-1 under 0.5 on a circuit; do-nothing not exactly 1 with 0 changed; an argmax change under the identity swap; the builders on Fig 2's inputs not giving E195's C, E214's gammas or E195's refit within its seed floor) stops the job and the cell waits on the fix; if both new circuits read rival B, the paper states Fig 2's built block holds at its one site and no further built-circuit runs are made
sketch: experiments/built_circuits/figures/sketch.png
rehearsal: E### rehearse: stage 0 on a cluster debug L40 at the full read sizes (512 windows of 41 tokens per circuit, 256 pairs, the full SAE dictionaries): attribution and C per circuit, the builds with their checks, the four fits (timed), do-nothing and identity swap at each featurization, one x2 scan per featurization timed; measures seconds per scan per featurization, peak GPU memory, the fit time and the derangement floor, before any pod
descriptive: user 2026-11-24 via whip-lead, Fig 2's built block extended to more circuits, reported as a description

## What runs (run 1 of cell P1.63)

On two base-model circuits other than Fig 2's (induction on repeated random tokens, C = eight layer-2 MLP neurons; a
Pile bigram lookup, C = eight layer-0 neurons; each C the top-8 by E195's attribution), the five built edits of Fig 2
are made by the existing code (x2, x10 by `rescaled`; rebuilt by E195's `fit`; new_few, new_thin by
`known_edits.build` at C's layer) with one E290 random per edit, and each of the 22 states is read at the answer
position by P1.62's driver at VPD, top neurons and SAE latents: E_b over K and changed with leftovers counted, beside
it without leftovers and the leftovers' share. Plan: reports/plans/2026-11-24_plan_p1-63-more-circuits.md.
