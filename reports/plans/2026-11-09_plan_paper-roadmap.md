# Plan 2026-11-09: the paper in three parts, and how the fleet stays on it

*Written with the user on 2026-11-09 after the status review (`https://claude.ai/artifact/6m1t4nMsdHSKTkuWxSyhp3`).
Replaces `2026-11-03_plan_qwen-reorient.md` and its Amendment 1. The table of work is
`~/commute-agents/fleet/paper/cells.tsv`, shown by `~/commute-agents/fleet/cell`.*

**In plain words.** The last plan ran three deliverables at once on the larger model with no agreed way to tell a
new mechanism from an existing one, so every reading came back with nothing to score it against. This plan is a
sequence. First a definition that gives a verdict. Then, on the small model, the ways of measuring new against
existing are compared against that definition. Only then the larger model, briefly.

## The paper (user, 2026-11-09)

1. **A working definition of reused against new.** The definition from Jain et al. 2024 gave no verdict on any task we
   have: most capabilities are not readable as one direction in the activations, and the one that was failed the
   causal check three ways. The tentative replacement keeps Jain's structure and takes its causal step from Prakash et
   al. 2024: the original model has a capability when a part of it, found on ordinary text only, is needed for the
   behaviour; a fine-tune reuses it when switching that part off breaks the fine-tuned behaviour and spares a twin
   fine-tune trained on shuffled labels. Tentative means the lead reads both papers in full first and the user adopts
   the wording, or not, by a dated `decisions.md` row.
2. **On the 67M model, a comparison of ways to measure new against existing:** re-gating, the same reading taken on
   neurons directly, and simple baselines (steering, update size, how likely the original model already was to give
   the answer). The user's words: we still do not know what re-gating is doing, what its strengths and weaknesses are,
   or how easily it could be outperformed. This part is not gap filling. Its questions are agreed with the user
   before any run.
3. **The key findings repeated on Qwen3-1.7B, with a brief comparison of RL and SFT,** on the new decomposition
   (p-9707d6ff) only.

Carrying an update onto the decomposition (the old first deliverable) is not a section of its own: the user takes
teacher training as the answer, at least for how much harder to try. Its re-runs on the new decomposition and the
remaining teacher-training runs on Qwen are stopped. The post-cutoff facts pair (E128) is paused: example pairs for
the larger model are chosen after part 1.

## Gates

| Gate | Date | What the user decides, from one page |
|---|---|---|
| Part 1 | 2026-11-13 | Adopt the definition: it reads "reuse" on the fine-tune built to reuse and "not" on its twin, on three pairs, three seeds each. If it has not, part 1 of the paper becomes why existing definitions fail, and part 2 is scored against labels by construction. |
| Part 2 | 2026-11-18 | The scorecard and what re-gating measures. Preliminary results are these two parts. |
| Part 3 | 2026-11-25 | The larger-model table. The user would prefer 11-18 and does not want it rushed. |

## The rules that keep it on track

The last two plans drifted within days. The harness made that easy: any run under the spending gate could start
itself, an unanswered question closed itself after two hours, and an experiment number was handed out for anything.
These rules replace those, and the harness enforces the first four.

1. **A run needs a cell.** The table lists every piece of work as a cell with an owner, a number of runs and a
   "done when". `fleet/ecode` gives an experiment number only against an open cell of the asking role with runs left.
   An amendment is a run.
2. **Only the user changes the table.** A lead that wants a new cell proposes it, in a cell made for proposals or in
   one ask, and does not run it. The user's own new ideas go to `backlog.md` and are looked at on gate days;
   `fleet/start` refuses a task that names no cell unless it is given `--off-plan`, which is recorded and shown.
3. **A cell's runs are final.** When they are spent the cell closes with what it read. "No call" is a result. No
   follow-up starts without a new cell.
4. **Silence is not permission.** No default is taken on an unanswered ask and no other model gives a go-ahead. A lead
   with nothing it may run finishes.
5. **Design before runs.** Each part opens with cells of zero runs: read, write one page, agree it with the user. The
   run cells are added after.
6. **Small records.** Per cell: predictions filed before the run, the result file, one figure, one paragraph
   (`fleet/cell fill`). No notebook, no per-experiment log card, no plan report over a page. The research log is
   updated once a day from the table.
7. **`fleet/check` opens with the plan:** cells filled and open per part, days to each gate, runs used, cells past
   their date, off-plan starts.

## Roles

| Role | Part | Starts with |
|---|---|---|
| `definition-lead` | 1 | P1.1, the definition page; P1.2, the saved copy-against-lookup result re-read under it |
| `compare-lead` (new; replaces `regating-lead` and `baseline-lead`) | 2 | P2.1, what re-gating does, from saved results; P2.2, the scorecard as it stands |
| `qwen-lead` (new; replaces `qwen-decomp-lead` and `qwen-rl-lead`) | 3 | P3.1, the re-gating read on the new decomposition; then idle until the part 2 gate |

## Not in this plan

New fine-tune pairs on Qwen, any further test of the carrying map, the strong version of the commuting question, and
anything on the old Qwen decomposition.
