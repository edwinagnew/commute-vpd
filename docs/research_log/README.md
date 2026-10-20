# Research log ("Update Decomposition")

`index.html` is the source of the research-log artifact at
https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu. The artifact is the colleague-facing explanation of the project;
the files in this repository are the record. The two must not drift: **every finished experiment and every major
decision updates the log in the same session**, and a Stop hook (`.claude/hooks/research_log_check.sh`) blocks the
session's end once if `TRIED.md`, `decisions.md`, `GOALS.md`, or an analysis or decision report changed after this
file did.

## When to update

- An analysis report is written (the `/writeup` procedure's last step).
- A decision report is written or `decisions.md` / `GOALS.md` change (the `/design` skill's "After" step).
- A row is added to `experiments/TRIED.md` (a new E-code) or a proposal is dropped.
- A review finds a claim wrong.

## What to update

1. **Every experiment**: a new `<article class="exp" id="eN">` card (code, name, date, machine, directory; one-sentence
   verdict; Run / Found / Bears on / Moved on), or the Moved-on field of an existing one. Figures: copy the experiment's
   own PNG into the `files` map below under `fig/eN_<name>.png`.
2. **Every claim, with its status**: add, re-chip or re-word rows. Statuses: holds, partial/qualified, refuted,
   uninformative, open.
3. **Sub-questions**: the status line and chip of every Q the experiment bears on; the E-code links.
4. **Overview → Where we are**: only if a headline result changed.
5. **Timeline**, **Gaps**, **Backlog**, **Mistakes**: whichever moved.
6. **Changelog**: a new top row (version, date, what changed). Old names and renamings go here, never in the body.
7. `experiments/TRIED.md` carries the same E-code; `README.md`'s experiment table lists it.

Keep the house style: cards and labelled fields, not paragraphs; the current terms from `GOALS.md` → Terms; about two
numbers per sentence; E-codes link automatically.

## How to publish

From a Claude Code session, with the Artifact tool, after editing `docs/research_log/index.html`:

```
Artifact publish
  url:        https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu
  file_path:  docs/research_log/index.html
  files:      {"fig/e2_pullback_behaviour.png":  "experiments/lift_test/figures/pullback_behaviour.png",
               "fig/e5_route_split.png":         "experiments/task_ladder/figures/route_split.png",
               "fig/e9_pile.png":                "experiments/lora_plateau/figures/pile.png",
               "fig/e11_ratios_per_task.png":    "experiments/reading_a_coverage/figures/ratios_per_task.png"}
  description: Everything the update-decomposition project has asked, run, found and deferred, for bringing a colleague up to speed and spotting gaps.
```

A session that has not read the artifact yet must `Artifact read` the URL first (another session may have published
since), merge onto that version, then publish. Files already published stay unless passed as `null`; new figures are
added to the map. Bump the version in the nav subtitle (`Research log · <date>, vN`) and the changelog together.
