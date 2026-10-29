# Research log ("Update Decomposition Log")

The research-log artifact at
https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu is built from `parts/` by `build.py`. The artifact is the colleague-facing explanation of the project;
the files in this repository are the record. The two must not drift: **every finished experiment and every major
decision updates the log in the same session**. A change that leaves what the log says untouched (a renamed path, a typo) needs no
log update.

## When to update

- An analysis report is written (the `/writeup` procedure's last step).
- A decision report is written or `decisions.md` / `GOALS.md` change (the `/design` skill's "After" step).
- A row is added to `experiments/TRIED.md` (a new E-code) or a proposal is dropped.
- A review finds a claim wrong.

## Layout (since 2026-10-22, user)

Four pages under one URL, sharing `style.css` and `script.js`. Each section is one file `parts/sections/<id>.html`
(`overview`, `questions`, `decisions`, …), opening with `<section id="<id>">` and an empty
`<h2><span class="num"></span>Title</h2>`. `SECTIONS` in `build.py` lists the ids in reading order with the page each
is on; a section's number is its place in that list. The text never writes a section number: it writes
`section {{n:gaps}}`, and the build fills in the current number (it refuses an unknown id or a section file missing
from the list).

| Page | Sections |
|---|---|
| `index.html` (the artifact itself) | every section except the three below |
| `experiments.html` | `experiments`: its intro, plus one file per card in `parts/experiments/` (`e1.html` … `a1.html`, ordered by number) |
| `glossary.html` | `glossary` |
| `changelog.html` | `changelog` |

**Adding a section:** write `parts/sections/<id>.html`, add `("<id>", "<page>")` to `SECTIONS` at its place, build.
Every later section's number moves, and every `{{n:…}}` reference follows it.

`parts/header.html` is the main page's title block; `parts/style.css` and `parts/script.js` are shared. Edit only
`parts/`; `build/` is generated and gitignored. `python3 docs/research_log/build.py` writes `build/`, rewrites `#id`
links that point at another page, puts every section from every page into the nav, and reads the nav's version line
("Research log · date, vN") from the top changelog row. An `<li id="eN">` anchor outside the cards (E13, in the mistakes section)
is linked to its own page.

## Pending decisions (section `decisions`, after the sub-questions; since 2026-10-22, user)

Every choice that waits on the user is a card in `parts/sections/decisions.html`, not a line buried in a report or
a chat message. A card says what the choice is about in plain words (no reliance on terms defined only in reports),
what the evidence says, the options with what each changes, and why it matters, then links the reports, notebooks and
cards. Each card carries an urgency flag in its top right (`<span class="chip … urgency">`): Blocking (red, planned runs cannot go ahead), Soon (amber, decides one line's next step) or No rush (grey); re-rate the flags when the plans change. Under the flag, one line says what runs next on that line and when: "Running:
<what>", "Due to run: <what>, <when or on what condition>", or "Nothing runs until this is decided" (user,
2026-10-26). Starting, finishing or dropping a run changes that line, and the change is queued like any other. A card
whose choice waits on the user now has the label `Question for you · <line> · new <date>` in its `<div class="k">`,
and these cards come first in the section; a card that waits on something else (a run, another decision) has no such
label and goes after them. When the user decides, the card is removed, the decision goes into `decisions.md` and the changelog row names
it. Adding or removing a card is a `PENDING.md` line like any other change.

## Main-page length (temporary, user 2026-10-26)

The main page had grown too long to read (about 20,000 words on 2026-10-26). Until the user lifts this, a flush may
not make it longer: what it adds, it takes out elsewhere, by rewriting a superseded box or folding a card that a newer
result replaced. The changelog, glossary, mistakes and experiments pages are exempt; they grow (the cards each
within the length below). Measure before and
after each flush:
```
cat $(ls docs/research_log/parts/sections/*.html | grep -v -E '/(changelog|glossary|mistakes|experiments)\.html') \
    docs/research_log/parts/header.html | sed 's/<[^>]*>/ /g' | wc -w      # at most 20,000
```

## Batched updates (since 2026-10-21, user)

Updates are queued, not applied one by one. When any event above happens, append one line to `PENDING.md` as the
event's file is written (date, file, log sections, what changed). Flush the queue before the session ends, and at
least once a day:

1. Check nobody else has published since the last build. `Artifact read` with `url` and
   `paths: ["index.html", "experiments.html", "changelog.html"]` (small files come back inline, so leave out pages
   you are not changing), then `python3 docs/research_log/build.py --check <the directory it saved to>`. Run it
   here, not through `fleet/cpu-run`: it is stdlib only, the dev-node hook allows it, and the worker cannot see the
   `/ephemeral` folder the read saves to (2026-10-27, three times). "same" on
   every page: build on `parts/` as is. A diff: someone published without committing `parts/`. Merge the diff into
   `parts/` first. With local edits to `parts/` already made, `--check-head <dir>` compares `parts/` as committed
   at HEAD instead, so the edits need not be set aside (friction 109).
2. Run the completeness check, so nothing written since the last flush is missed:
   ```
   last=$(grep -o '<td class="c">20[0-9-]*</td>' docs/research_log/parts/sections/changelog.html \
          | head -1 | grep -o '20[0-9-]*')        # date of the newest changelog row
   ls reports/*/ | awk -v d="$last" '$0 >= d'    # every report dated on or after it
   for e in $(grep -o '^| E[0-9]*' experiments/TRIED.md | grep -o 'E[0-9]*' | sort -u); do
     grep -rq "id=\"e${e#E}\"" docs/research_log/parts || echo "missing card: $e"; done
   ```
   Every report listed and every E-code missing from `parts/` must be covered by a `PENDING.md` line or an existing
   changelog row; add the missing lines first. A row covers the reports it names in full or by a wildcard
   (`analysis_qwen-*` covers every `analysis_qwen-` report).
3. Apply each line as "What to update" below says, grepping `parts/` for the places to change rather than reading
   whole sections. Write one changelog row for the flush, empty `PENDING.md`, build, publish.

## What to update

1. **Every experiment**: a new file `parts/experiments/eN.html` holding one `<article class="exp" id="eN">` card, or
   the fields of an existing one. Card format (user, 2026-10-29, after a red-team read by cold Sonnet readers; model
   cards: E16, E21, E34, E38, E39):
   - **Header**: code, name (the question), date, machine, directory; an outcome chip of one or two words (held,
     missed, mixed, surprise, no answer, ongoing, unscored).
   - **Opening**: four short `<p class="one">` paragraphs headed **The question.** (in project terms: which line and
     Q, why it matters, what earlier result prompted it), **What we did.** (in plain words, defining every term a
     reader new to the project needs: the adversarial ratio and its pass and fail lines, do-nothing, the map used),
     **What we found.** (the result, at most two numbers a sentence, pointing at the figures) and **What it means.**
     (what it says for the line, the caveat that sets the next test, and what the project did next). A reader who
     stops here knows the result and why the next decision followed.
   - **Figures**: three to five, each answering one question. Design the plot for its message, not as a dump of
     every arm: one panel where one will do, bars from a natural baseline, task names in words (` which` then ` was`,
     never `r1_t534_v369`), no "Amendment N", "filed", W₀ or raw metric names in titles, legends or axes. A title
     states the finding; if a later run overturns it, redraw it. The caption opens with the question in bold and a
     one-word answer, then says how to read the plot. Log figures are drawn from saved results by the experiment's
     `plot_log.py`; figures cut from a card stay in its report, and Moved on says so.
   - **Run**: two sentences: what ran, on what, at what cost.
   - **Found**: three bullets at most, each one plain sentence a colleague can read cold: what came out and whether it
     was expected, at most two numbers, no filed-prediction bookkeeping (that lives in the report). An amendment
     rewrites the bullets to the experiment's current answer instead of adding one per amendment.
   - **Bears on**; **Moved on**: three sentences at most (the decisions taken and by whom, what runs next) plus the
     report links.
   A flush that touches a card brings it to this format (user, 2026-10-26: the Found lists had become unreadable;
   2026-10-29: the openings were too thin and the plots too hard to read). Figures: copy the PNG into the `files`
   map below under `fig/eN_<name>.png`.
2. **Every claim, with its status** (`claims.html`): add, re-chip or re-word rows. Statuses: holds,
   partial/qualified, refuted, uninformative, open.
3. **Sub-questions** (`questions.html`): the status line and chip of every Q the experiment bears on; the E-code
   links. **Rewrite the status, never append to it** (user, 2026-10-26, repeated): answer first, three to five
   plain sentences, at most a couple of numbers, E-codes in brackets; the details live in the cards. The same holds for
   the Overview's Result cards and line boxes: when a result moves, re-read the whole box and rewrite it, headline included.
4. **Overview → Where we are** (`overview.html`): only if a headline result changed.
5. **Timeline**, **Gaps**, **Backlog**, **Mistakes** (`timeline`, `gaps`, `backlog`, `mistakes`): whichever moved.
6. **Changelog** (`changelog.html`): a new top row (version, date, what changed). The nav's version follows it.
   Old names and renamings go here, never in the body.
7. `experiments/TRIED.md` carries the same E-code; `README.md`'s experiment table lists it.

**High level by default** (user, 2026-10-22): every section above the cards is for skimming, so it says what was learned or decided in plain words, not how a run went; numbers, arm names and gate results stay in the cards and reports. The timeline is one headline and two to four short sentences per day. Keep the house style: cards and labelled fields, not paragraphs; the current terms from `GOALS.md` → Terms; about two
numbers per sentence; E-codes link automatically, to whichever page holds the card. Links written as `href="#id"` are
fine anywhere: the build points them at the right page.

## How to publish

From a Claude Code session, with the Artifact tool:

0. `~/commute-agents/fleet/log-lock take <role>` before claiming a version number or editing any part, and
   `~/commute-agents/fleet/log-lock release <role>` after the publish (one flush at a time across the leads: two
   flushes both claimed v73, friction 37). `<role>` is the lead's role name (`commute-lead`, ...); a clerk is given it
   in its brief. If another role holds the lock, wait and retry; a lock older than 45 min may be taken over.
1. `git pull --no-rebase`, then `python3 docs/research_log/build.py`, so the build has every other session's changes.
2. Right before every publish, not just a session's first, two calls; the publisher needs both:

   ```
   Artifact read    url: https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu
   Artifact list    url: https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu   scope: "files"
   ```
   The read with `url` alone is what counts as viewing the page: a read with `paths`, even one naming `index.html`,
   saves the files but does not (refused "not viewed" on every such try, 2026-10-27). The listing covers every
   already-published file you will pass without bringing any into context (a `paths` read inlines glossary, style
   and script, about 15k tokens).
3. Publish at once:

```
Artifact publish
  url:        https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu
  file_path:  docs/research_log/build/index.html
  files:      the changed pages and assets among
              {"experiments.html": "docs/research_log/build/experiments.html",
               "glossary.html":    "docs/research_log/build/glossary.html",
               "changelog.html":   "docs/research_log/build/changelog.html",
               "style.css":        "docs/research_log/build/style.css",
               "script.js":        "docs/research_log/build/script.js"}
              plus any new figure, e.g. "fig/e21_<name>.png": "experiments/<dir>/figures/<name>.png"
```

Files already published stay unless passed as `null`. The nav's version line is on every page, so a flush that adds a
changelog row publishes all four pages and both assets; new figures are added to the map.

The publisher refuses a page the session has not viewed, and any file already on the artifact that the session has
not read or listed, or that changed since it did. Step 2 covers both. Several sessions publish this log, so a
refusal that hands back newer content means another one published in between: `git pull`, rebuild, repeat steps 2
and 3, never resending the refused content unchanged. A refusal for a missing view or read means step 2 was skipped:
make the missing call and send the same publish again. If that resend is refused as unchanged, send it once more;
on 2026-10-27 that went through each time.

What the publisher did on 2026-10-28 (friction 43, 46, 48, 60), so a flush plans for it instead of retrying blind:
- `build.py --check` saying `same` does not count as a view or a read: the publisher's rules are its own, and only
  its own read, listing or refusal satisfies them.
- Three sends per flush were common (v71, v79, v82): refused "not viewed", then refused as an unchanged resend, then
  accepted. The refusal's own diff counts as the view, and the identical resend after it is expected: send it.
- Twice it asked for every line of the saved live page to be Read (~70k tokens) before it would publish. That is the
  publisher's requirement for files the session did not write, not a step to skip: Read the file it names, in parts,
  then publish. Budget a flush's context for it.
- Never `Artifact read` the log with `paths` "to be safe": it inlines glossary, style and script (~15k tokens) and
  still does not count as viewing the page.

Figures already published (their sources, for a re-upload):
- "fig/overview_component_training.png": "docs/research_log/component_training_tasks.png"
- "fig/e2_faithful.png": "experiments/lift_test/figures/inside_vs_outside.png"
- "fig/e2_behaviour.png": "experiments/lift_test/figures/pullback_behaviour.png"
- "fig/e6_overlap.png": "experiments/recruit_test/figures/overlap.png"
- "fig/e4_sgd_lr_grid.png": "experiments/sgd_grid/figures/lr_grid.png"
- "fig/e4_log_learn.png": "experiments/concentration_test/figures/log_learn.png"
- "fig/e4_log_switches.png": "experiments/concentration_test/figures/log_switches.png"
- "fig/e4_log_switched_on.png": "experiments/concentration_test/figures/log_switched_on.png"
- "fig/e4_log_sgd_rate.png": "experiments/concentration_test/figures/log_sgd_rate.png"
- "fig/e5_route_split.png": "experiments/task_ladder/figures/route_split.png"
- "fig/e5_log_coverage.png": "experiments/task_ladder/figures/log_coverage.png"
- "fig/e5_log_routes.png": "experiments/task_ladder/figures/log_routes.png"
- "fig/e5_log_methods.png": "experiments/task_ladder/figures/log_methods.png"
- "fig/e7_vs_rank.png": "experiments/lora_commute/figures/vs_rank.png"
- "fig/e8_headline.png": "experiments/lora_matched/figures/headline.png"
- "fig/e9_pile.png": "experiments/lora_plateau/figures/pile.png"
- "fig/e10_reading_a.png": "experiments/kv_task/figures/reading_a.png"
- "fig/e11_ratios_per_task.png": "experiments/reading_a_coverage/figures/ratios_per_task.png"
- "fig/e11_log_adversarial.png": "experiments/reading_a_coverage/figures/log_adversarial.png"
- "fig/e11_log_unmasked.png": "experiments/reading_a_coverage/figures/log_unmasked.png"
- "fig/e11_log_offtarget.png": "experiments/reading_a_coverage/figures/log_offtarget.png"
- "fig/e12_route_separation.png": "experiments/new_vs_existing/figures/route_separation.png"
- "fig/e12_separation.png": "experiments/new_vs_existing/figures/separation.png"
- "fig/e12_log_verdict.png": "experiments/new_vs_existing/figures/log_verdict.png"
- "fig/e12_log_routes.png": "experiments/new_vs_existing/figures/log_routes.png"
- "fig/e12_log_update.png": "experiments/new_vs_existing/figures/log_update.png"
- "fig/e12_log_switched_on.png": "experiments/new_vs_existing/figures/log_switched_on.png"
- "fig/e14_sides.png": "experiments/lora_split/figures/sides.png"
- "fig/e15_pgd_vs_step.png": "experiments/gamma_refit/figures/pgd_vs_step.png"
- "fig/e15_stage0_ablation.png": "experiments/gamma_refit/figures/stage0_ablation.png"
- "fig/e16_bars.png": "experiments/split_comparison/figures/bars.png"
- "fig/e16_routes.png": "experiments/split_comparison/figures/routes.png"
- "fig/e17_half_revert.png": "experiments/cancellation/figures/half_revert.png"
- "fig/e17_adversary_revert.png": "experiments/cancellation/figures/adversary_revert.png"
- "fig/e19_behaviour_in_delta.png": "experiments/step_split/figures/behaviour_in_delta.png"
- "fig/e19_delta_share.png": "experiments/step_split/figures/delta_share.png"
- "fig/e19_log_behaviour.png": "experiments/step_split/figures/log_behaviour.png"
- "fig/e19_log_adversary.png": "experiments/step_split/figures/log_adversary.png"
- "fig/e19_log_delta_share.png": "experiments/step_split/figures/log_delta_share.png"
- "fig/e18_recovery.png": "experiments/regating_leak/figures/recovery.png"
- "fig/e18_shared.png": "experiments/regating_leak/figures/shared.png"
- "fig/e18_natural.png": "experiments/regating_leak/figures/natural.png"
- "fig/e18_adversary.png": "experiments/regating_leak/figures/adversary.png"
- "fig/e20_decomposition.png": "experiments/cancellation_complement/figures/decomposition.png"
- "fig/e20_pattern.png": "experiments/cancellation_complement/figures/pattern.png"
- "fig/e20_bins.png": "experiments/cancellation_complement/figures/bins.png"
- "fig/e18_amend1.png": "experiments/regating_leak/figures/amend1.png"
- "fig/e20_rest_split.png": "experiments/cancellation_complement/figures/rest_split.png"
- "fig/e18_amend2_bins.png": "experiments/regating_leak/figures/amend2_bins.png"
- "fig/e18_amend3_tests.png": "experiments/regating_leak/figures/amend3_tests.png"
- "fig/e20_amend2_steps.png": "experiments/cancellation_complement/figures/amend2_steps.png"
- "fig/e21_ladder.png": "experiments/sgd_ladder/figures/ladder.png"
- "fig/e21_kv.png": "experiments/sgd_kv/figures/kv.png"
- "fig/e21_steps.png": "experiments/sgd_ladder_long/figures/steps.png"
- "fig/e21_size_damage.png": "experiments/sgd_ladder_long/figures/size_damage.png"
- "fig/e21_matched.png": "experiments/sgd_finetune/figures/matched.png"
- "fig/e22_placements.png": "experiments/small_finetune_splits/figures/placements.png"
- "fig/e22_adjoint.png": "experiments/small_finetune_splits/figures/adjoint.png"
- "fig/e22_outside.png": "experiments/small_finetune_splits/figures/outside.png"
- "fig/e22_log_faithful.png": "experiments/small_finetune_splits/figures/log_faithful.png"
- "fig/e22_log_size.png": "experiments/small_finetune_splits/figures/log_size.png"
- "fig/e22_log_cost.png": "experiments/small_finetune_splits/figures/log_cost.png"
- "fig/e22_log_outside.png": "experiments/small_finetune_splits/figures/log_outside.png"
- "fig/e23_parts.png": "experiments/cancellation_sgd/figures/parts.png"
- "fig/e23_budget.png": "experiments/cancellation_sgd/figures/budget.png"
- "fig/e24_r.png": "experiments/regating_small/figures/r.png"
- "fig/e24_shared.png": "experiments/regating_small/figures/shared.png"
- "fig/e25_pile_pgd.png": "experiments/step_split_small/figures/pile_pgd.png"
- "fig/e25_pieces.png": "experiments/step_split_small/figures/pieces.png"
- "fig/e25_delta_share.png": "experiments/step_split_small/figures/delta_share.png"
- "fig/e25_log_adversary.png": "experiments/step_split_small/figures/log_adversary.png"
- "fig/e25_log_behaviour.png": "experiments/step_split_small/figures/log_behaviour.png"
- "fig/e25_log_delta_share.png": "experiments/step_split_small/figures/log_delta_share.png"
- "fig/e26_seeds.png": "experiments/ladder_second_seed/figures/seeds.png"
- "fig/e26_log_verdicts.png": "experiments/ladder_second_seed/figures/log_verdicts.png"
- "fig/e26_log_rare.png": "experiments/ladder_second_seed/figures/log_rare.png"
- "fig/e26_log_size.png": "experiments/ladder_second_seed/figures/log_size.png"
- "fig/e26_log_control.png": "experiments/ladder_second_seed/figures/log_control.png"
- "fig/e27_outside.png": "experiments/lora_sgd/figures/outside.png"
- "fig/e27_cos.png": "experiments/lora_sgd/figures/cos.png"
- "fig/e27_states.png": "experiments/lora_sgd/figures/states.png"
- "fig/e28_fixed.png": "experiments/regating_tasks/figures/fixed.png"
- "fig/e28_shuffled.png": "experiments/regating_tasks/figures/shuffled.png"
- "fig/e28_other.png": "experiments/regating_tasks/figures/other.png"
- "fig/e29_overlap.png": "experiments/regating_down/figures/overlap.png"
- "fig/e29_function.png": "experiments/regating_down/figures/function.png"
- "fig/e30_trigger.png": "experiments/gamma_shift_map/figures/trigger.png"
- "fig/e30_adversarial.png": "experiments/gamma_shift_map/figures/adversarial.png"
- "fig/e30_log_trigger.png": "experiments/gamma_shift_map/figures/log_trigger.png"
- "fig/e30_log_adversarial.png": "experiments/gamma_shift_map/figures/log_adversarial.png"
- "fig/e30_log_sparsity.png": "experiments/gamma_shift_map/figures/log_sparsity.png"
- "fig/e30_log_halves.png": "experiments/gamma_shift_map/figures/log_halves.png"
- "fig/e31_steps.png": "experiments/adjoint_sgd/figures/steps.png"
- "fig/e31_keep.png": "experiments/adjoint_sgd/figures/keep.png"
- "fig/e31_floor.png": "experiments/adjoint_sgd/figures/floor.png"
- "fig/e31_budget.png": "experiments/adjoint_sgd/figures/budget.png"
- "fig/e31_size.png": "experiments/adjoint_sgd/figures/size.png"
- "fig/e10_amend1_ratio.png": "experiments/kv_task/figures/amend1_ratio.png"
- "fig/e10_log_learning.png": "experiments/kv_task/figures/log_learning.png"
- "fig/e10_log_check.png": "experiments/kv_task/figures/log_check.png"
- "fig/e10_log_routes.png": "experiments/kv_task/figures/log_routes.png"
- "fig/e10_log_damage.png": "experiments/kv_task/figures/log_damage.png"
- "fig/e34_behaviour.png": "experiments/teacher_training/figures/behaviour.png"
- "fig/e34_faithful.png": "experiments/teacher_training/figures/faithful.png"
- "fig/e34_amend1.png": "experiments/teacher_training/figures/amend1.png"
- "fig/e34_amend3.png": "experiments/teacher_training/figures/amend3.png"
- "fig/e31_scaled.png": "experiments/adjoint_sgd/figures/scaled.png"
- "fig/e31_log_tasks.png": "experiments/adjoint_sgd/figures/log_tasks.png"
- "fig/e31_log_size.png": "experiments/adjoint_sgd/figures/log_size.png"
- "fig/e31_log_faithful.png": "experiments/adjoint_sgd/figures/log_faithful.png"
- "fig/e31_log_scaled.png": "experiments/adjoint_sgd/figures/log_scaled.png"
- "fig/e34_amend4.png": "experiments/teacher_training/figures/amend4.png"
- "fig/e34_amend5.png": "experiments/teacher_training/figures/amend5.png"
- "fig/e34_amend5_cos.png": "experiments/teacher_training/figures/amend5_cos.png"
- "fig/e37_behaviour_faithful.png": "experiments/teacher_spread/figures/behaviour_faithful.png"
- "fig/e37_placement.png": "experiments/teacher_spread/figures/placement.png"
- "fig/e37_amend1_steps.png": "experiments/teacher_spread/figures/amend1_steps.png"
- "fig/e37_log_faithful.png": "experiments/teacher_spread/figures/log_faithful.png"
- "fig/e37_log_placement.png": "experiments/teacher_spread/figures/log_placement.png"
- "fig/e37_log_steps.png": "experiments/teacher_spread/figures/log_steps.png"
- "fig/e37_log_masked.png": "experiments/teacher_spread/figures/log_masked.png"
- "fig/e38_w0.png": "experiments/recruitment_test/figures/w0.png"
- "fig/e38_amend2_D.png": "experiments/recruitment_test/figures/amend2_D.png"
- "fig/e38_amend2_census.png": "experiments/recruitment_test/figures/amend2_census.png"
- "fig/e38_amend3_restore.png": "experiments/recruitment_test/figures/amend3_restore.png"
- "fig/e38_amend3_confound.png": "experiments/recruitment_test/figures/amend3_confound.png"
- "fig/e38_amend4_restore.png": "experiments/recruitment_test/figures/amend4_restore.png"
- "fig/e38_amend4_change.png": "experiments/recruitment_test/figures/amend4_change.png"
- "fig/e38_amend4_singles.png": "experiments/recruitment_test/figures/amend4_singles.png"
- "fig/e38_amend5_restore.png": "experiments/recruitment_test/figures/amend5_restore.png"
- "fig/e38_amend7_line.png": "experiments/recruitment_test/figures/amend7_line.png"
- "fig/e38_log_census.png": "experiments/recruitment_test/figures/log_census.png"
- "fig/e38_log_restore.png": "experiments/recruitment_test/figures/log_restore.png"
- "fig/e16_grid.png": "experiments/split_comparison/figures/grid.png"
- "fig/e34_log_faithful.png": "experiments/teacher_training/figures/log_faithful.png"
- "fig/e34_amend6_lora.png": "experiments/teacher_training/figures/amend6_lora.png"
- "fig/e43_log_check.png": "experiments/teacher_training/figures/log_e43_check.png"
- "fig/e43_log_removal.png": "experiments/teacher_training/figures/log_e43_removal.png"
- "fig/e43_log_size.png": "experiments/teacher_training/figures/log_e43_size.png"
- "fig/e41_log_step10.png": "experiments/teacher_training/figures/log_e41_step10.png"
- "fig/e41_log_chain.png": "experiments/teacher_training/figures/log_e41_chain.png"
- "fig/e41_log_behaviour.png": "experiments/teacher_training/figures/log_e41_behaviour.png"
- "fig/e41_log_pileup.png": "experiments/teacher_training/figures/log_e41_pileup.png"
- "fig/e39_moves.png": "experiments/broad_fragility/figures/moves.png"
- "fig/e39_reverts.png": "experiments/broad_fragility/figures/reverts.png"
- "fig/e39_shape.png": "experiments/broad_fragility/figures/shape.png"
- "fig/e39_amend1_rank.png": "experiments/broad_fragility/figures/amend1_rank.png"
- "fig/e39_amend3_rows.png": "experiments/broad_fragility/figures/amend3_rows.png"
- "fig/e47_i_teacher_training.png": "experiments/claim_tests/figures/e47_i_teacher_training.png"
- "fig/e47_ii_lora_rank.png": "experiments/claim_tests/figures/e47_ii_lora_rank.png"
- "fig/e47_iii_keep_size.png": "experiments/claim_tests/figures/e47_iii_keep_size.png"
- "fig/e47_iv_remainder.png": "experiments/claim_tests/figures/e47_iv_remainder.png"
- "fig/e47_v_plain_sgd.png": "experiments/claim_tests/figures/e47_v_plain_sgd.png"
- "fig/e47_log_teacher_training.png": "experiments/claim_tests/figures/log_teacher_training.png"
- "fig/e47_log_plain_sgd.png": "experiments/claim_tests/figures/log_plain_sgd.png"
- "fig/e47_log_lora_rank.png": "experiments/claim_tests/figures/log_lora_rank.png"
- "fig/e47_log_remainder.png": "experiments/claim_tests/figures/log_remainder.png"
- "fig/e47_log_keep_size.png": "experiments/claim_tests/figures/log_keep_size.png"
- "fig/e50_shares.png": "experiments/restore_anchors/figures/shares.png"
- "fig/e50_amend1.png": "experiments/restore_anchors/figures/amend1.png"
