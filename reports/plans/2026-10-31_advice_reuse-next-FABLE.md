# Advice (Fable, 2026-10-31): the reuse line's next step after C1b and C3

Advice only; the lead decides. Read: `reports/analyses/2026-10-31_analysis_e55-{c1b,c3}.md`, the brief
(`~/commute-data/agents/reuse-lead/brief.md`), `decisions.md` rows "E55 after C1" and "update size within a recipe",
my two earlier notes, `TRIED.md` E55/E58 rows, `.claude/CLAUDE.md` "Gated runs".

## Where we stand, in three lines
- C1b: the cheap methods work where the label is fixed by construction, and anything that reads the construction ties them. VPD can at best tie there; the valuable outcome is a VPD **fail**.
- C3: on real 67M states at 10 steps the cheap methods fail their positive control; VPD separates the poles. And the carry now reads at the old end on the axis (0.72), so the 67M open-pair set no longer holds a VPD-versus-label disagreement: it can test the instruments, not rank VPD.
- The brief's central item, "score VPD and at least one non-VPD method against the ground truth", is still unmet on every set. The user's own decision row (2026-10-30, ask 4) already orders "option 2 then option 1": C1b, **then C2**.

## Options
| | cost | can show | chance it settles something | needs the user |
|---|---|---|---|---|
| **1. Refine on the open-pair set** (report's proposals 1-3) | $0, one debug job, ~1 agent day | at what update size pruning and reverse fine-tuning pass a positive control on real states; where elicit then sits | medium for the instruments; **nil for VPD**, since every reader already puts elicit old on this axis | no (class 2/4; a new reverse target is a measurement change: `/design` written, no sign-off) |
| **2. A new truth set** where the cheap methods work and VPD can be compared | open labels need Stage B's interchange patch (not built, 1-2 agent days, $0) on plateau 67M states, or a second model | the first table with cheap methods and VPD on open labels | medium; but open labels on the 67M have so far been graded and agreed with VPD once calibrated | a new task family or model: yes |
| **3. C2 on the pcfg set** | minGPT port to `param-decomp` 1-2 agent days; decomposition of a 3M model may fit `debug` jobs with saves and resume ($0, ungated); else `normal` QoS (rule relaxation) or RunPod ~$10-30 (gated) | VPD fails where cheap methods are perfect (the sceptic's result, high value) or ties (the brief's anchor item met, with Jain's numbers beside it); r and the stand-in ratio against a clean reuse label and the standing update-size column | ~25% fail, ~65% tie, ~10% VPD alone handles the not-yet-learned N cell; every outcome is reportable | only the platform if it does not fit `debug` |
| **4. Existing work** | half an agent day, $0 | the repo's 40 summaries hold no benchmark of elicitation vs new capability. Jain is the only by-construction one (C1b already reproduces its pruning numbers in kind); Minder et al.'s diffing-toolkit (ADL) and WeightWatch detect *what* was fine-tuned, not old vs new. Take recipes: Jain's regime (small lr, weakly relevant, reFT vs scratch), ADL as a third cheap instrument for Qwen later | low alone; informs 1 and 3 | no |
| **5. Stop and write up** | ~1 agent day, $0 | cheap methods pass on by-construction counting and fail on small real updates; VPD unbenchmarked | nothing new; leaves the user's priority unmet | dropping a line: yes |

## Recommendation
1. **C2 now**, as the user's row already orders. Plan it with the gates record; rehearse the decomposition of the 3M minGPT at tiny size on `debug` first. If a full decomposition fits in ≤ 30-min jobs with saves and resume, it is $0 and ungated and no question is needed; otherwise one question: `normal` QoS or ~$10-30. Score r and the stand-in ratio on the 21 states with cut-offs set on seeds 0-1, beside the three instruments and the within-cell update-size column; report the five not-yet-learned N states as a column, not scored.
2. **In parallel, one $0 job on the open-pair set**, folding proposals 1 and 2: rank-1, cipher, elicit × 3 seeds at steps 10, 30, 100, 300; read R and t½, with t½/D and a non-sharpening target (log p of W₀'s top guess) as reported columns, the 3-sd positive-control bar filed per step. It answers "at what size does the port work" and closes the set. Spend no more on it as a VPD benchmark.
3. Fold 4 into 1 and 3 (half a day): Jain §5's lr and step regime sets the ladder's expected pass step; ADL goes on the backlog for Qwen.
4. No stop. The write-up is C2's: the line's first table with cheap methods and VPD on the same truth.

**What would change this.** If the minGPT port exceeds two agent days or `param-decomp` cannot take the model, switch to option 2 (Stage B label on E12 plateau states, VPD already readable there). If the ladder shows a gap at 100-300 steps and puts elicit at the **new** end, that goes to the user at once (VPD the odd one out) and outranks C2. If the user wants the Qwen line benchmarked soonest, ADL on Qwen replaces 2 and 3.

## Who decides
Lead: 1-4 above (C2 is the user's standing decision; the ladder is class 2/4). User: C2's platform only if the rehearsal does not fit `debug`; any new model or task family; stopping the line.
