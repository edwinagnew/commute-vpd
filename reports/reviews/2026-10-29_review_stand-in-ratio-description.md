# How the living files describe the stand-in ratio, after E46 Amendment 3

*Review, 2026-10-29. Source finding: `reports/analyses/2026-10-29_analysis_e46-amendment3.md`, section "What the
ratio is made of". No new run. No log part was edited; the fixes are queued in `docs/research_log/PENDING.md`.*

## In plain words

We read every living file that describes the re-gating line's planned Qwen number or its earlier copy-task result.
That number is nearly what Γ's new values do to the original pieces, yet several files call it a share of the
split's changes that Γ keeps. The earlier result still stands, being built on that same gate gain, so the fixes
change wording and status lines only.

## The finding carried

Write $L(C, G)$ for the answer's log-odds in the stand-in model, with subcomponents $C$ and masks $G$. $C_0$ is W₀'s
set and $C_1$ the split one. $G_0$ and $G_1$ are Γ's values on the original and the updated model. $U_0$ and $U_1$ are
the log-odds with every mask open. The stand-in ratio, filed as "gated piece gain over piece gain", is

$\dfrac{L(C_1,G_1) - L(C_0,G_0)}{U_1 - U_0} = \dfrac{g}{U_1 - U_0} + \dfrac{L(C_1,G_1) - L(C_0,G_1)}{U_1 - U_0}$,

with $g = L(C_0,G_1) - L(C_0,G_0)$, stage 0's gate gain. The second term is the route split's direct plus interaction
part. On every defined row of Amendment 3, fine-tunes included, it is a few hundredths. So the ratio is nearly $g$
over the subcomponent gain. It is not a share kept, and it has no ceiling at 1.

This is not new. The reset plan's Amendment 1 already said that stage 0's ratio was $r$ renamed. It said the
separation "rests entirely on the gate gain". `decisions.md` records the same in its E46 stage 0 row. The descriptions below
drifted away from that record afterwards.

## Does E12's original reading still stand?

Yes, and it is built on $g$, not on the split's own pieces. E12's evidence was the re-switching share $r$. That share is
the same gate gain divided by the real model's gain. It read about 0.8 for copy against about 0.28 for the
random maps. The direct and interaction parts were a few tenths of a nat in both classes, against gains of 10 nats or
more. Those figures are in the reset plan's table of E12's filed cells. E12's second separating number, the masked-off share, is the subcomponent
gain minus the masked gain, over the real gain. Since the masked gain is nearly $g$, it too is mostly $g$ seen from the
other side.

So Amendment 3 neither weakens nor adds to E12's reading. Its standing caveats are unchanged, and they are the real
limits on it:

- It is read in the stand-in, which the fine-tuned model does not contain.
- The same share reads high for a fine-tune that learned nothing, in E36.
- A random change and a hand-made write each earn part of the gate gain, in E46 stage 0.
- "Existing machinery" means subcomponents W₀ already has, never a circuit W₀ already runs. That limit is kept in
  `decisions.md`, under "Limit of the elicitation reading".

Why the split's own changes add so little under Γ's new values is not measured. One candidate explanation, unchecked:
E12 found that the pieces the split moves most are disjoint from the set Γ switches on. If so, Γ's new values mostly
leave the moved pieces masked.

## Where the wording should change

"Description" means the claim's substance stays and only how the number is described changes. "Status" means a line
says something is still running or undecided that has since run.

| file, place | current wording (brief) | proposed wording | change |
|---|---|---|---|
| `parts/experiments/e46.html`, Found, first bullet | "the copy fine-tunes keep 0.87 to 0.97 of what the updated subcomponents do unmasked under Γ's new values" | "the stand-in ratio reads 0.87 to 0.97 for copy and 0.36 to 0.41 for the random maps, within 0.002 of E12; it is nearly the gate gain over the split's subcomponent gain, the split's own changes adding 0.01 to 0.05 (Amendment 3)" | description |
| `parts/experiments/e46.html`, one-line summary | "a hand-made write that uses no existing subcomponent" | "a hand-made write built by no training"; the least-squares split puts 38% of its input-weighted energy inside the alive subcomponents' span (Amendment 3) | description; secondary, not the ratio itself |
| `parts/experiments/e46.html`, Moved on | the card "asks you whether to read stage 1's own ratio on the random change and the write first" | Amendment 3 ran (Slurm 42993, $0); new card on whether the stand-in ratio has any role in stage 1 | status; already queued in PENDING's Amendment 3 line |
| `parts/sections/overview.html`, Line 1 box, "What we found" | "re-gating tells a behaviour the model already had from a new one ... (E12)" | keep; add "through the gate gain: Γ's new values on the original pieces, unmoved" | none; it is built on g |
| `parts/sections/overview.html`, Line 1 box, "Caveats" | "a hand-made edit that uses none of the pieces" | "a hand-made edit built by no training" | description; secondary |
| `parts/sections/overview.html`, Line 1 box, "Now" | "A free cluster job ... tests whether the measure planned for the larger Qwen3-1.7B still credits the random change and the hand-made edit" | "That job ran: the measure planned for Qwen is nearly the gate gain over the split's subcomponent gain, so it adds little to the gate gain. Whether it has a role on Qwen is a pending card; E50 reads the real-model restore share next." | status and description |
| `parts/sections/questions.html`, Q5 status, last sentence | "whether to read it on a real update waits on you" | add: "Splitting the nulls showed that the ratio planned for the real updates is nearly that same gate gain, so it is not a second test (E46 Amendment 3)." | status |
| `parts/sections/next.html`, re-gating card, "Next" | "one more $0 cluster job that reads stage 1's own ratio on the random change and the hand-made write" | the job's result in one sentence as above, and E50 as next | status |
| `parts/sections/claims.html`, row "Reveal and create show up as different routes" | "the reset plan's stage 0 asks whether it survives the nulls above" | "stage 0 found the gate gain separates the anchors, but a random change and a hand-made write each earn part of it (E46); the stand-in ratio is that same gate gain over the subcomponent gain, not a second line of evidence (Amendment 3)" | status; the claim's chip stays |
| `parts/sections/claims.html`, row "Most of a fine-tune's gain is carried by re-gating" | "the moved pieces almost none (E5, E12, E16)" | keep; may add E46 Amendment 3 to its evidence: the split's own changes add 0.01 to 0.05 of the subcomponent gain | none |
| `parts/sections/glossary.html`, entry "Re-gating", list of measures | names r, the switched-on set, the restore share and the fixed shift | add the gate gain and the stand-in ratio | description; missing entry |
| `parts/sections/glossary.html`, new entries | no entry for either number | "Gate gain, g: what Γ's new values alone do to the original pieces' log-odds in the stand-in, Δ out; r is g over the real gain." "Stand-in ratio: the masked gain of the split state over its subcomponent gain; nearly g over the subcomponent gain, since the split's own changes add 0.01 to 0.05 of it (E46 Amendment 3). Not a share kept; can exceed 1." | description |
| `parts/sections/techniques.html`, route-split prose | "behaviours the model already has read as re-weight" | keep | none; built on r |
| `parts/experiments/e12.html` | "disjoint from the set Γ switches on"; the re-switching share | keep | none; E12 reads r, built on g |
| `parts/experiments/e16.html`, `e42.html`, `e5.html` | "subcomponents Γ keeps off" | keep | none; these are the masked-off share, a different number |
| `README.md`, "Current plan, re-gating line" | "after E46 Amendment 3 (..., split nulls, running on cluster debug)" | "after E46 Amendment 3 (done 2026-10-29, $0: the stand-in ratio is nearly the gate gain over the subcomponent gain; card on its role in stage 1)" | status |
| `README.md`, results list | "the frozen Γ re-switching existing subcomponents" | keep | none |
| `GOALS.md` | route split and Terms; no wording on the ratio | keep | none |
| `decisions.md`, E46 stage 0 row | "E12's separation rests wholly on the gate gain" | keep; it already says this | none |
| `decisions.md`, row choosing Amendment 3 | "stage 1's own ratio (gated piece gain over piece gain)" | keep; a dated record | none |
| `reports/plans/2026-10-29_plan_restore-anchors.md`, the numbers table | g and the stand-in ratio on one row as two numbers; "split nulls running (Amendment 3)" | one number, the gate gain, with the ratio as gate gain over subcomponent gain; Amendment 3's result in place of "running" | status and description; a dated note, not an edit of the filed section |
| same plan, branch "does not separate" | "stage 1 rests on the stand-in alone" | "goes to the user", as Amendment 3's report recommends | substance of a follow-on branch, already recommended; not a filed rule |
| same plan, "Blind to" | "*Stand-in ratio*: the real model" | add "and nearly blind to the split's own changes, which add 0.01 to 0.05 of it" | description |
| same plan, "Positive control" | "the stand-in already reads it 0.87-0.97" | "the gate gain over the subcomponent gain already reads 0.87-0.97" | description |
| `reports/plans/2026-10-28_plan_regating-line-reset.md`, definitions table | "gated piece gain: what the updated pieces do under Γ's new values" | "the gate gain plus what the split's changes add under Γ's new values; the second part is 0.04 to 0.51 nats" | description; a dated note |
| same plan, P0.2 branch and stage 1 table | "how much of what the updated pieces do, Γ's new gating lets through" | "the gate gain over the subcomponent gain"; stage 1's KL form needs the same caution | description; Amendment 1 there already says it |
| `reports/analyses/2026-10-29_analysis_e46-regating-table.md` and `notebooks/E46_regating_table.ipynb`, replication bullet | "Take what the updated subcomponents do unmasked. Under Γ's new values the ..." | a dated note pointing to Amendment 3's decomposition | description |

`experiments/restore_anchors/results/predictions.json` carries the ratio as reported only. It is filed and stays as
it is.

## Which claims change in substance

None. Every claim checked is built on the gate gain or on $r$, both still real measurements on W₀'s own pieces. The one
change near substance is a plan branch, not a claim. E50's "does not separate" branch should go to the user instead of
straight to stage 1. Amendment 3's report already recommends that.

## Not run

- The changelog, the timeline and the cards E4, E6, E18, E36 and E38 were not read line by line. They were checked
  only by the search terms named in the brief.
- `theory/`, `backlog.md` and notebooks other than E46's were not searched.
- No check that stage 1's KL form on Qwen splits the same way; the small second term is from the 67M toys only.
- The candidate explanation for the small second term, that the moved pieces stay masked, was not measured.
- The PENDING line queued for Amendment 3 was not merged with this one; a flush should apply both together.
