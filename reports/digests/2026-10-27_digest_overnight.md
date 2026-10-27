# Overnight digest, 2026-10-27 (lead agent, re-gating line)

**Spend: this session went $3 over its $40 budget.** The last pod (E36 Amendment 6, H100) finished its run in two
minutes at 19:34, then billed about 14 hours while both its runner and I sat at a permission prompt: moving the run's
log into `results/large/`, which links outside the project. The plugin's budget backstop terminated it this morning.
It should have cost about $0.35. The lesson is now in `CLAUDE.md` (terminate before any local file move).

**The finding:** the re-gating line now has a form of the fixed-shift hypothesis that names no cue: positions the
original model treats alike get alike changes. On a fine-tune with no trigger, the nearest rule on the original
importance values recovers three-quarters or more of the switching. A summary of the last few tokens recovers about
half, and adding the whole preceding text did not help. One decision waits on you: whether to adopt that wording.

Task (`~/commute-data/tonight`): restate the fixed-shift hypothesis without cues or triggers. Plan:
`reports/plans/2026-10-26_plan_regating-without-cues.md` (Amendments 1-9). Research log v53 (published by lead-2):
https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu.

## What ran, and what it found

- **E35, the toy tasks without labels.** The mean change of the 8 training inputs with the nearest original values
  recovered 0.61 to 0.78 of the switching. That matches one change per cue word (0.47 to 0.66), and the neighbours
  shared the cue only 62 to 71% of the time.
- **E36, a lowercase fine-tune with no trigger** (Amendments 1-6, `reports/analyses/2026-10-26_analysis_regating-spread.md`,
  `notebooks/E36_regating_spread.ipynb`). Fitted on 262,144 positions, the nearest rule recovers 0.89, 0.78 and 0.76
  at steps 10, 100 and 300. The same rule on Γ's own input does as well. One change per token gets about a third. A
  summary of the last four tokens, which the model did not produce, gets about half. Amendment 6 added a bag of
  every earlier token: the joined summary scored 0.49 at step 300, and the bag alone about zero, no better than random
  neighbours. So whether longer context closes the gap is still untested beyond the first 32 positions of a row.
- Every analysis had an independent Opus check: numbers right each time, headlines toned down each time.

## Reversed predictions (notified)

- E35: neighbours share the cue 62-71%, predicted at least 90%.
- E36: a fixed line per subcomponent recovered nothing, predicted to beat one change by 0.1.
- E36 A3: with 262,144 positions the nearest rule cleared 0.7, predicted to stay under it (in the hypothesis's favour).
- E36 A6 (T4): the whole preceding text added nothing to the last four tokens, predicted to add.

## Decision waiting on you

**Restate the fixed-shift hypothesis without cues?** (Blocking card `decision-cuefree` in the log.) (a) restate now,
tentatively; (b) keep "per cue" until a real behaviour or the larger model confirms it; (c) restate on the toy tasks
only. Amendment 6's check named one threat that is cheap to close: the original values may win only because each
value's own level bounds how far it can move. Plan Amendment 9 tests that with a rule using each value alone, 16
bins per subcomponent. Its code (`experiments/regating_spread/amend7.py`) is being written and reviewed. It needs a
pod of under $1, refused in this session now the budget is spent, so it waits for a new session or your go.

## Machines and problems

- RunPod: none running. Session total $43.02 (about $3 of real runs; the rest the idle pod above).
- Slurm: the venv rebuild (32273) hit its 40-minute limit at 20:37, so the swap job never ran:
  `~/commute-data/venv` still points at `~/commute/.venv`; don't delete `~/commute` yet. `l40-worker` takes jobs again.
- `~/commute-data/blocked.txt` holds only the overnight prompt above. No agent tried to reach anything out of bounds.
