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
from the top changelog row.

## Pending decisions (section `decisions`, after the sub-questions; since 2026-10-22, user)

Every choice that waits on the user is a card in `parts/sections/decisions.html`, not a line buried in a report or
a chat message. A card says what the choice is about in plain words (no reliance on terms defined only in reports),
what the evidence says, the options with what each changes, and why it matters, then links the reports, notebooks and
cards. **Test before publishing (user, 2026-11-02, "as usual ... unreadable"):** a reader who has read no report can say what each
option would run, what it would tell them and what happens next. Plan labels (Q3, Stage D, anchor 1, D3, E-codes, option numbers from a report) do not appear in the question, evidence or options; name the thing instead (the template fine-tune). One small "Names in the reports" line at the end decodes them. Options get letters, a cost and their limits, and the card ends on one recommendation. The fleet/ask line and the decision report follow the same rule. Each card carries an urgency flag in its top right (`<span class="chip … urgency">`): Blocking (red, planned runs cannot go ahead), Soon (amber, decides one line's next step) or No rush (grey); re-rate the flags when the plans change. Under the flag, one line says what runs next on that line and when: "Running:
<what>", "Due to run: <what>, <when or on what condition>", or "Nothing runs until this is decided" (user,
2026-10-26). Starting, finishing or dropping a run changes that line, and the change is queued like any other. A card
whose choice waits on the user now has the label `Question for you · <lead> · new <date>` in its `<div class="k">`, `<lead>` being the role name
(`commute-lead`, `regating-lead`, `baseline-lead`, `qwen-decomp-lead`, ...), never a bare "commute": "commute" is also
the project's name, so a bare word misroutes the user's answer (user, 2026-11-02). These cards come first in the section; a card that waits on something else (a run, another decision) has no such
label and goes after them. When the user decides, the card is removed, the decision goes into `decisions.md` and the changelog row names
it. Adding or removing a card is a `PENDING.md` line like any other change.

## Main-page length (temporary, user 2026-10-26)

Until the user lifts this, a flush may not make the main page longer: what it adds, it takes out elsewhere, by rewriting
a superseded box or folding a card that a newer result replaced. The changelog, glossary, mistakes and experiments pages are exempt; they grow (the cards each
within the length below). Measure each flush with `python3 docs/research_log/build.py --words`: words per section, each section's
change since HEAD, and the total against the 20,000 limit. Plan the cuts once: `--words` says how much must go,
`--blocks` lists the 40 largest blocks; make them all, then re-measure once. Cut cheapest first:
what this flush supersedes; closed items in `next`, `gaps` and `timeline`; Techniques or Apparatus prose a card already
holds. Claim rows last: they are the record (friction 208, 368, 376).

## Batched updates (since 2026-10-21, user)

Updates are queued, not applied one by one. When any event above happens, append one line to `PENDING.md` as the
event's file is written (date, file, log sections, what changed). Flush the queue before the session ends, and at
least once a day:

1. Check nobody else has published since the last build: How to publish → step 2 below, before editing `parts/`. Each
   page "same": build on `parts/` as is. A diff: someone published without committing `parts/`; merge it in first.
2. `python3 docs/research_log/build.py --missing` lists reports dated on or after the newest changelog row, and E-codes
   without a card, that no part of the log or queue names. Add a `PENDING.md` line for each. A changelog wildcard
   (`analysis_qwen-*`) counts as naming.
3. Apply each line as "What to update" below says, grepping `parts/` for the places to change rather than reading
   whole sections. Write one changelog row for the flush, empty `PENDING.md` except its `HOLD` lines, build. The build warns on a removed line
   whose file, E-code or card id is in nothing the flush added: check it landed (friction 155). Then publish. A line
   appended meanwhile survives the merge and waits for the next flush; lines you removed stay gone (`fleet/merge-lines`).

## What to update

1. **Every experiment**: a new file `parts/experiments/eN.html` holding one `<article class="exp" id="eN">` card, or
   the fields of an existing one. Card format (user, 2026-10-29, after a red-team read by cold Sonnet readers; model
   cards: E16, E21, E34, E38, E39):
   - **Header**: code, name (the question), date, machine, directory; an outcome chip of one or two words (held,
     missed, mixed, surprise, no answer, ongoing, unscored). A card written before the independent check says
     provisional and has no reading above the verdict: E62's unchecked figure was live, and wrong (friction 362).
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
7. `experiments/TRIED.md` carries the same E-code. The README table is no longer kept up (friction 393).

**High level by default** (user, 2026-10-22): every section above the cards is for skimming, so it says what was learned or decided in plain words, not how a run went; numbers, arm names and gate results stay in the cards and reports. The timeline is one headline and two to four short sentences per day. Keep the house style: cards and labelled fields, not paragraphs; the current terms from `GOALS.md` → Terms; about two
numbers per sentence; E-codes link automatically, to whichever page holds the card. Links written as `href="#id"` are
fine anywhere: the build points them at the right page.

## How to publish

From a Claude Code session, with the Artifact tool:

0. `~/commute-agents/fleet/log-lock take <role> --wait` before claiming a version number or editing any part, and
   `~/commute-agents/fleet/log-lock release <role>` after the publish (two flushes both claimed v73, friction 37).
   `<role>` is the lead's role name; a clerk's brief gives it. `take --wait` waits 100 s a call; run it again on exit 1. A
   holder that has ended loses the lock to the next take (`log-lock status` says why). Every session takes it, for a
   one-off edit too. Nothing enforces that, and sessions have published past a held lock (friction 347, 149): step 2's
   diff shows it; merge and take the next version number. In the call before the send, `log-lock check <role>`: on exit
   1 do not publish; take it again and re-pull.
1. Commit `parts/` and the emptied queue (named paths, `git commit -m ... -- docs/research_log experiments/TRIED.md
   README.md`), since a pull refuses to merge into an uncommitted `PENDING.md`. Then `git pull --no-rebase` and
   `python3 docs/research_log/build.py`, so the build has every other session's changes. Commit what that changed and
   push **before** publishing: a published version whose `parts/` sat uncommitted or unpushed made the next flush merge it
   back from the live page (friction 138, 139, 323). If the pull brought in parts edits, rebuild before publishing, or
   the live page lags the repo (friction 153).
2. Save the live pages by path, which prints a line per page, not the page, and compare (on the dev node, not through
   `fleet/cpu-run`, whose worker cannot see `/ephemeral`).

   ```
   Artifact read    url: https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu   paths: ["nav.js"]
   python3 docs/research_log/build.py --changed <folder it saved to>      # the folder argument is required
   Artifact read    url: https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu   paths: [the pages --changed printed, index.html among them]
   python3 docs/research_log/build.py --check <folder>
   ```
   Read `nav.js` first: `--changed` and `--check` both need the live `nav.js` in the folder, and `--check` without it
   compares nothing useful (friction 565). `--check` lists the pages you did not save on one "not compared" line and
   needs no read of them (friction 561); it also prints `dead link:` for a `#anchor` (e.g. a changelog row's
   `#decision-...`) whose id exists in no built page: fix or drop the link (friction 588). `build.py --words` rebuilds
   `build/` too, so a publish after it is current (friction 519).
   `--changed` compares the hashes in `nav.js`, so a page nobody changed is neither read nor sent; rerun it after the
   last build and read any page newly named. A live `nav.js` missing or without hashes names every page: read those
   (friction 291). A file not yet live needs no read. Never `read` with `url` alone: ~50k tokens, and the first send was
   refused all the same (friction 115, 160). The saved index.html is for `--check`; it does not count as a view.
   Which check (friction 384): before editing `parts/`, `--check`, where any diff is someone else's. With uncommitted
   edits, `--check-head`, against `parts/` at HEAD. After your commit, `--check` diffs your own edits: read only
   index.html's place count, which predicts step 3. `--check` also flags a live page built from `parts/` in no commit
   (friction 323).
3. Publish (below). The first send is refused unless this session published the live version; the refusal's first
   sentence picks the case. One rule covers most of them: **if the refusal's diff shows only your own changes, your
   read was stale. Do not resend unchanged (that is refused "resent unchanged"). Re-read the live page with `Artifact
   read` and `path` (not `url` alone), then send once** (friction 517, 535, 553, 560, 585). If the diff shows a line that
   is not yours, another session published: pull, rebuild, merge, step 1.
   - **"same author content"**: send again; it passes.
   - **"shown in full below as a line diff"** (under about 10 places): read the diff. Only your own changes: the rule
     above. A line that is not yours: merge first, whatever the place count (friction 409, 451, 479).
   - **"not shown here"** (about 10 places or more): resending never passes (friction 315, 326, 359, 387). The refusal
     names a saved file: Read it whole, 130 lines a call (200 can overrun Read), through the last line it names (~45k
     tokens). Merge what yours lacks; the next send passes. Step 2's place count warns of this: split such a flush if you
     can (apply some lines, publish, then the rest).
   - **"touches files whose published content is not what you last saw"**: `Artifact read` each path named, `--check`,
     merge into `parts/` if it differs, rebuild, publish. A figure you are not changing comes out of `files`.
   Anything else: stop and report it with the refusal text. The version the tool reports counts every send and differs
   from the changelog row; the row is the log's version (friction 461).

```
Artifact publish
  url:        https://claude.ai/artifact/SvgkzUiESRkyE9GM65HCWu
  file_path:  docs/research_log/build/index.html
  files:      what --changed printed, except index.html, among
              {"nav.js":           "docs/research_log/build/nav.js",
               "experiments.html": "docs/research_log/build/experiments.html",
               "glossary.html":    "docs/research_log/build/glossary.html",
               "changelog.html":   "docs/research_log/build/changelog.html",
               "style.css":        "docs/research_log/build/style.css",
               "script.js":        "docs/research_log/build/script.js"}
              plus any new figure, e.g. "fig/e21_<name>.png": "experiments/<dir>/figures/<name>.png"
```

Files already published stay unless passed as `null`. Every new figure needs a line in the list below in the same
commit (friction 163). `Artifact list` with `scope: "files"` takes no `limit`
(friction 162, 169).

Figures already published (their sources, for a re-upload; grep `fig/eN_` here for a log figure's repo path):
- "fig/overview_component_training.png": "docs/research_log/component_training_tasks.png"
- "fig/e1_log_span.png": "experiments/transport_test/figures/log_span.png"
- "fig/e1_log_behaviour.png": "experiments/transport_test/figures/log_behaviour.png"
- "fig/e1_log_location.png": "experiments/transport_test/figures/log_location.png"
- "fig/e2_faithful.png": "experiments/lift_test/figures/inside_vs_outside.png"
- "fig/e2_behaviour.png": "experiments/lift_test/figures/pullback_behaviour.png"
- "fig/e2_log_adversarial.png": "experiments/lift_test/figures/log_adversarial.png"
- "fig/e2_log_behaviour.png": "experiments/lift_test/figures/log_behaviour.png"
- "fig/e2_log_outside.png": "experiments/lift_test/figures/log_outside.png"
- "fig/e2_log_parts.png": "experiments/lift_test/figures/log_parts.png"
- "fig/e3_log_concentration.png": "experiments/identity_test/figures/log_concentration.png"
- "fig/e3_log_drops.png": "experiments/identity_test/figures/log_drops.png"
- "fig/e3_log_identity.png": "experiments/identity_test/figures/log_identity.png"
- "fig/e6_overlap.png": "experiments/recruit_test/figures/overlap.png"
- "fig/e6_log_overlap.png": "experiments/recruit_test/figures/log_overlap.png"
- "fig/e6_log_share.png": "experiments/recruit_test/figures/log_share.png"
- "fig/e6_log_recovery.png": "experiments/recruit_test/figures/log_recovery.png"
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
- "fig/e7_log_faithful.png": "experiments/lora_commute/figures/log_faithful.png"
- "fig/e7_log_outside.png": "experiments/lora_commute/figures/log_outside.png"
- "fig/e7_log_left_in_delta.png": "experiments/lora_commute/figures/log_left_in_delta.png"
- "fig/e7_log_routes.png": "experiments/lora_commute/figures/log_routes.png"
- "fig/e8_headline.png": "experiments/lora_matched/figures/headline.png"
- "fig/e8_log_faithful.png": "experiments/lora_matched/figures/log_faithful.png"
- "fig/e8_log_delta.png": "experiments/lora_matched/figures/log_delta.png"
- "fig/e8_log_matched.png": "experiments/lora_matched/figures/log_matched.png"
- "fig/e9_pile.png": "experiments/lora_plateau/figures/pile.png"
- "fig/e9_log_readside.png": "experiments/lora_plateau/figures/log_readside.png"
- "fig/e9_log_where.png": "experiments/lora_plateau/figures/log_where.png"
- "fig/e9_log_adversary.png": "experiments/lora_plateau/figures/log_adversary.png"
- "fig/e9_log_delta.png": "experiments/lora_plateau/figures/log_delta.png"
- "fig/e10_reading_a.png": "experiments/kv_task/figures/reading_a.png"
- "fig/e11_ratios_per_task.png": "experiments/reading_a_coverage/figures/ratios_per_task.png"
- "fig/e11_log_adversarial.png": "experiments/reading_a_coverage/figures/log_adversarial.png"
- "fig/e11_log_unmasked.png": "experiments/reading_a_coverage/figures/log_unmasked.png"
- "fig/e11_log_offtarget.png": "experiments/reading_a_coverage/figures/log_offtarget.png"
- "fig/a1_log_checks.png": "experiments/reading_a_coverage/figures/log_a1_checks.png"
- "fig/e12_route_separation.png": "experiments/new_vs_existing/figures/route_separation.png"
- "fig/e12_separation.png": "experiments/new_vs_existing/figures/separation.png"
- "fig/e12_log_verdict.png": "experiments/new_vs_existing/figures/log_verdict.png"
- "fig/e12_log_routes.png": "experiments/new_vs_existing/figures/log_routes.png"
- "fig/e12_log_update.png": "experiments/new_vs_existing/figures/log_update.png"
- "fig/e12_log_switched_on.png": "experiments/new_vs_existing/figures/log_switched_on.png"
- "fig/e14_sides.png": "experiments/lora_split/figures/sides.png"
- "fig/e14_log_read.png": "experiments/lora_split/figures/log_read.png"
- "fig/e14_log_write.png": "experiments/lora_split/figures/log_write.png"
- "fig/e14_log_one_matrix.png": "experiments/lora_split/figures/log_one_matrix.png"
- "fig/e14_log_rescaling.png": "experiments/lora_split/figures/log_rescaling.png"
- "fig/e14_log_check.png": "experiments/lora_split/figures/log_check.png"
- "fig/e15_pgd_vs_step.png": "experiments/gamma_refit/figures/pgd_vs_step.png"
- "fig/e15_stage0_ablation.png": "experiments/gamma_refit/figures/stage0_ablation.png"
- "fig/e15_log_control.png": "experiments/gamma_refit/figures/log_control.png"
- "fig/e15_log_split.png": "experiments/gamma_refit/figures/log_split.png"
- "fig/e15_log_sparsity.png": "experiments/gamma_refit/figures/log_sparsity.png"
- "fig/e15_log_stage0.png": "experiments/gamma_refit/figures/log_stage0.png"
- "fig/e16_bars.png": "experiments/split_comparison/figures/bars.png"
- "fig/e16_routes.png": "experiments/split_comparison/figures/routes.png"
- "fig/e17_half_revert.png": "experiments/cancellation/figures/half_revert.png"
- "fig/e17_adversary_revert.png": "experiments/cancellation/figures/adversary_revert.png"
- "fig/e17_half_revert_toy.png": "experiments/cancellation/figures/half_revert_toy.png"
- "fig/e17_log_half_revert.png": "experiments/cancellation/figures/log_half_revert.png"
- "fig/e17_log_adversary_share.png": "experiments/cancellation/figures/log_adversary_share.png"
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
- "fig/e18_log_fixed.png": "experiments/regating_leak/figures/log_fixed.png"
- "fig/e18_log_where.png": "experiments/regating_leak/figures/log_where.png"
- "fig/e18_log_leftover.png": "experiments/regating_leak/figures/log_leftover.png"
- "fig/e18_log_natural.png": "experiments/regating_leak/figures/log_natural.png"
- "fig/e20_amend2_steps.png": "experiments/cancellation_complement/figures/amend2_steps.png"
- "fig/e20_log_excess.png": "experiments/cancellation_complement/figures/log_excess.png"
- "fig/e20_log_offset.png": "experiments/cancellation_complement/figures/log_offset.png"
- "fig/e20_log_steps.png": "experiments/cancellation_complement/figures/log_steps.png"
- "fig/e20_log_budget.png": "experiments/cancellation_complement/figures/log_budget.png"
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
- "fig/e23_log_parts.png": "experiments/cancellation_sgd/figures/log_parts.png"
- "fig/e23_log_offset.png": "experiments/cancellation_sgd/figures/log_offset.png"
- "fig/e23_log_budget.png": "experiments/cancellation_sgd/figures/log_budget.png"
- "fig/e24_r.png": "experiments/regating_small/figures/r.png"
- "fig/e24_shared.png": "experiments/regating_small/figures/shared.png"
- "fig/e24_log_switching.png": "experiments/regating_small/figures/log_switching.png"
- "fig/e24_log_fixed.png": "experiments/regating_small/figures/log_fixed.png"
- "fig/e24_log_trigger.png": "experiments/regating_small/figures/log_trigger.png"
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
- "fig/e27_log_outside.png": "experiments/lora_sgd/figures/log_outside.png"
- "fig/e27_log_readfactor.png": "experiments/lora_sgd/figures/log_readfactor.png"
- "fig/e27_log_faithful.png": "experiments/lora_sgd/figures/log_faithful.png"
- "fig/e27_log_behaviour.png": "experiments/lora_sgd/figures/log_behaviour.png"
- "fig/e28_fixed.png": "experiments/regating_tasks/figures/fixed.png"
- "fig/e28_shuffled.png": "experiments/regating_tasks/figures/shuffled.png"
- "fig/e28_other.png": "experiments/regating_tasks/figures/other.png"
- "fig/e28_log_fixed.png": "experiments/regating_tasks/figures/log_fixed.png"
- "fig/e28_log_shuffled.png": "experiments/regating_tasks/figures/log_shuffled.png"
- "fig/e28_log_other.png": "experiments/regating_tasks/figures/log_other.png"
- "fig/e29_overlap.png": "experiments/regating_down/figures/overlap.png"
- "fig/e29_function.png": "experiments/regating_down/figures/function.png"
- "fig/e29_log_overlap.png": "experiments/regating_down/figures/log_overlap.png"
- "fig/e29_log_function.png": "experiments/regating_down/figures/log_function.png"
- "fig/e29_log_where.png": "experiments/regating_down/figures/log_where.png"
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
- "fig/e32_log_per_key.png": "experiments/shift_anatomy/figures/log_per_key.png"
- "fig/e32_log_simple_forms.png": "experiments/shift_anatomy/figures/log_simple_forms.png"
- "fig/e32_log_blocks.png": "experiments/shift_anatomy/figures/log_blocks.png"
- "fig/e32_log_pile.png": "experiments/shift_anatomy/figures/log_pile.png"
- "fig/e33_log_cue_vs_answer.png": "experiments/many_cues/figures/log_cue_vs_answer.png"
- "fig/e33_log_donors.png": "experiments/many_cues/figures/log_donors.png"
- "fig/e33_log_overlap.png": "experiments/many_cues/figures/log_overlap.png"
- "fig/e33_log_strength.png": "experiments/many_cues/figures/log_strength.png"
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
- "fig/e35_log_rules.png": "experiments/regating_cuefree/figures/log_rules.png"
- "fig/e35_log_samecue.png": "experiments/regating_cuefree/figures/log_samecue.png"
- "fig/e35_log_groups.png": "experiments/regating_cuefree/figures/log_groups.png"
- "fig/e35_log_line.png": "experiments/regating_cuefree/figures/log_line.png"
- "fig/e36_log_switching.png": "experiments/regating_spread/figures/log_switching.png"
- "fig/e36_log_sampling.png": "experiments/regating_spread/figures/log_sampling.png"
- "fig/e36_log_space.png": "experiments/regating_spread/figures/log_space.png"
- "fig/e36_log_room.png": "experiments/regating_spread/figures/log_room.png"
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
- "fig/e42_log_check.png": "experiments/teacher_training/figures/log_e42_check.png"
- "fig/e42_log_trigger.png": "experiments/teacher_training/figures/log_e42_trigger.png"
- "fig/e42_log_routes.png": "experiments/teacher_training/figures/log_e42_routes.png"
- "fig/e42_log_pile.png": "experiments/teacher_training/figures/log_e42_pile.png"
- "fig/e42_log_controls.png": "experiments/teacher_training/figures/log_e42_controls.png"
- "fig/e39_moves.png": "experiments/broad_fragility/figures/moves.png"
- "fig/e39_reverts.png": "experiments/broad_fragility/figures/reverts.png"
- "fig/e39_shape.png": "experiments/broad_fragility/figures/shape.png"
- "fig/e39_amend1_rank.png": "experiments/broad_fragility/figures/amend1_rank.png"
- "fig/e39_amend3_rows.png": "experiments/broad_fragility/figures/amend3_rows.png"
- "fig/e40_log_control.png": "experiments/qwen_sft/figures/log_control.png"
- "fig/e40_log_firststep.png": "experiments/qwen_sft/figures/log_firststep.png"
- "fig/e40_log_accuracy.png": "experiments/qwen_sft/figures/log_accuracy.png"
- "fig/e40_log_donothing.png": "experiments/qwen_sft/figures/log_donothing.png"
- "fig/e44_log_masks.png": "experiments/qwen_maps/figures/log_masks.png"
- "fig/e44_log_kept.png": "experiments/qwen_maps/figures/log_kept.png"
- "fig/e44_log_ratios.png": "experiments/qwen_maps/figures/log_ratios.png"
- "fig/e45_log_anchor2.png": "experiments/qwen_updates/figures/log_anchor2.png"
- "fig/e45_log_step2.png": "experiments/qwen_updates/figures/log_step2.png"
- "fig/e45_log_anchor1.png": "experiments/qwen_updates/figures/log_anchor1.png"
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
- "fig/e48_s1_fidelity.png": "experiments/pass_reread/figures/e48_s1_fidelity.png"
- "fig/e48_log_ratio.png": "experiments/pass_reread/figures/log_ratio.png"
- "fig/e48_log_nats.png": "experiments/pass_reread/figures/log_nats.png"
- "fig/e48_log_carriers.png": "experiments/pass_reread/figures/log_carriers.png"
- "fig/e46_log_overview.png": "experiments/regating_table/figures/log_overview.png"
- "fig/e46_log_random.png": "experiments/regating_table/figures/log_random.png"
- "fig/e46_log_follow.png": "experiments/regating_table/figures/log_follow.png"
- "fig/e46_log_write.png": "experiments/regating_table/figures/log_write.png"
- "fig/e46_log_ratio.png": "experiments/regating_table/figures/log_ratio.png"
- "fig/e50_shares.png": "experiments/restore_anchors/figures/shares.png"
- "fig/e50_amend1.png": "experiments/restore_anchors/figures/amend1.png"
- "fig/e50_log_relay.png": "experiments/restore_anchors/figures/log_relay.png"
- "fig/e50_log_sets.png": "experiments/restore_anchors/figures/log_sets.png"
- "fig/e50_log_s_size.png": "experiments/restore_anchors/figures/log_s_size.png"
- "fig/e50_log_probe.png": "experiments/restore_anchors/figures/log_probe.png"
- "fig/e44_amend1_layer_cut.png": "experiments/qwen_maps/figures/amend1_layer_cut.png"
- "fig/e49_blocks.png": "experiments/row_blocks/figures/blocks.png"
- "fig/e49_hard_block.png": "experiments/row_blocks/figures/hard_block.png"
- "fig/e51_erank_vs_size.png": "experiments/baseline_weights/figures/erank_vs_size.png"
- "fig/e51_weights.png": "experiments/baseline_weights/figures/weights.png"
- "fig/e55_amend5_saves.png": "experiments/pcfg_anchors/figures/amend5_saves.png"
- "fig/e44_log_reading_b.png": "experiments/qwen_maps/figures/log_reading_b.png"
- "fig/e56_b2_curves.png": "experiments/keydependent_patch/figures/b2_curves.png"
- "fig/e58_a7_r.png": "experiments/open_pair/figures/a7_r.png"
- "fig/e55_c3_poles.png": "experiments/open_pair_instruments/figures/c3_poles.png"
- "fig/e55_c3_positions.png": "experiments/open_pair_instruments/figures/c3_positions.png"
- "fig/e55_ladder_steps.png": "experiments/open_pair_instruments/figures/ladder_steps.png"
- "fig/e45_amend2_anchor1.png": "experiments/qwen_updates/figures/amend2_anchor1.png"
- "fig/e45_amend2_matching.png": "experiments/qwen_updates/figures/amend2_matching.png"
- "fig/e45_amend2_own_null.png": "experiments/qwen_updates/figures/amend2_own_null.png"
- "fig/e54_size.png": "experiments/adjoint_sgd/figures/size.png"
- "fig/e55_c1b_readings.png": "experiments/pcfg_anchors/figures/c1b_readings.png"
- "fig/e55_c1b_scores.png": "experiments/pcfg_anchors/figures/c1b_scores.png"
- "fig/e55_labels.png": "experiments/pcfg_anchors/figures/labels.png"
- "fig/e58_synthesis.png": "experiments/open_pair/figures/synthesis.png"
- "fig/e58_log_prior.png": "experiments/open_pair/figures/log_prior.png"
- "fig/e52_neurons.png": "experiments/baseline_neurons/figures/p1_neuron_patch.png"
- "fig/e52_prompting.png": "experiments/baseline_neurons/figures/p4_prompting.png"
- "fig/e52_steering.png": "experiments/baseline_neurons/figures/p3_steering.png"
- "fig/e54_route_split.png": "experiments/keydependent_elicit/figures/p1_route_split.png"
- "fig/e54_steering.png": "experiments/keydependent_elicit/figures/p2_steering.png"
- "fig/e34_amend10_cosine.png": "experiments/teacher_training/figures/amend10_cosine.png"
- "fig/methodtable_shares.png": "experiments/method_table/figures/shares.png"
- "fig/e12_where_gain.png": "experiments/new_vs_existing/figures/amend1_where_gain.png"
- "fig/e61_log_size.png": "experiments/qwen_recipe/figures/log_size.png"
- "fig/e70_low.png": "experiments/pcfg_anchors/figures/e70_low.png"
- "fig/e70_ridge.png": "experiments/pcfg_anchors/figures/e70_ridge.png"
- "fig/e70_softmax.png": "experiments/pcfg_anchors/figures/e70_softmax.png"
- "fig/e70_a1_low.png": "experiments/pcfg_anchors/figures/e70_a1_low.png"
- "fig/e70_a1_tasks.png": "experiments/pcfg_anchors/figures/e70_a1_tasks.png"
- "fig/e70_a2_tasks.png": "experiments/pcfg_anchors/figures/e70_a2_tasks.png"
- "fig/e70_a2_low.png": "experiments/pcfg_anchors/figures/e70_a2_low.png"
- "fig/e71_check.png": "experiments/scorecard_ls/figures/check.png"
- "fig/e71_keep_delta.png": "experiments/scorecard_ls/figures/keep_delta.png"
- "fig/e68_gap_closed.png": "experiments/qwen_pass_test/figures/gap_closed.png"
- "fig/e68_opening_kl.png": "experiments/qwen_pass_test/figures/opening_kl.png"
- "fig/e82_gap_closed.png": "experiments/qwen_pass_test/figures/gap_closed_clean.png"
- "fig/e82_gain_kept.png": "experiments/qwen_pass_test/figures/gain_kept_clean.png"
- "fig/e72_t1_gain.png": "experiments/qwen_fixed_masks/figures/t1_gain.png"
- "fig/e72_t2_kl.png": "experiments/qwen_fixed_masks/figures/t2_kl.png"
- "fig/e72_t2_gain.png": "experiments/qwen_fixed_masks/figures/t2_gain.png"
- "fig/e73_kl_path.png": "experiments/qwen_diagnose/figures/kl_path.png"
- "fig/e73_gain_path.png": "experiments/qwen_diagnose/figures/gain_path.png"
- "fig/e73_by_turn.png": "experiments/qwen_diagnose/figures/by_turn.png"
- "fig/e74_by_block.png": "experiments/qwen_transmission/figures/by_block.png"
- "fig/e74_stochastic.png": "experiments/qwen_transmission/figures/stochastic.png"
- "fig/e74_direct.png": "experiments/qwen_transmission/figures/direct.png"
- "fig/e75_k_fraction.png": "experiments/qwen_rl_reads/figures/k_fraction.png"
- "fig/e75_beats_size.png": "experiments/qwen_rl_reads/figures/beats_size.png"
- "fig/e75_against_sft.png": "experiments/qwen_rl_reads/figures/against_sft.png"
- "fig/e77_own_by_block.png": "experiments/qwen_open_points/figures/own_by_block.png"
- "fig/e77_swap.png": "experiments/qwen_open_points/figures/swap.png"
- "fig/e77_routing.png": "experiments/qwen_open_points/figures/routing.png"
- "fig/e75_bad_placements.png": "experiments/qwen_rl_reads/figures/bad_placements.png"
- "fig/e76_two_terms.png": "experiments/qwen_masked_teacher/figures/two_terms.png"
- "fig/e76_size_vs_fall.png": "experiments/qwen_masked_teacher/figures/size_vs_fall.png"
- "fig/e76_by_block.png": "experiments/qwen_masked_teacher/figures/by_block.png"
- "fig/e78_gaps.png": "experiments/qwen_anchor1/figures/gaps.png"
- "fig/e75_r_vs_n.png": "experiments/qwen_rl_reads/figures/r_vs_n.png"
- "fig/e79_two_by_two.png": "experiments/qwen_masked_read/figures/two_by_two.png"
- "fig/e79_teeth.png": "experiments/qwen_masked_read/figures/teeth.png"
- "fig/e79_frozen.png": "experiments/qwen_masked_read/figures/frozen.png"
- "fig/e69_leads.png": "experiments/qwen_rl/figures/leads.png"
- "fig/e69_saved_steps.png": "experiments/qwen_rl/figures/saved_steps.png"
- "fig/e80_adversary_maths.png": "experiments/qwen_d1_adversary/figures/adversary_maths.png"
- "fig/e83_logit_shift.png": "experiments/logit_shift_67m/figures/logit_shift.png"
- "fig/e81_choice.png": "experiments/qwen_tofu/figures/choice.png"
- "fig/e81_recall.png": "experiments/qwen_tofu/figures/recall.png"
- "fig/e84_filed_test.png": "experiments/qwen_tofu_forms/figures/filed_test.png"
- "fig/e84_along_training.png": "experiments/qwen_tofu_forms/figures/along_training.png"
- "fig/e85_ratio_bars.png": "experiments/qwen_d1_teeth/figures/ratio_bars.png"
- "fig/e85_controls.png": "experiments/qwen_d1_teeth/figures/controls.png"
- "fig/e86_kept_vs_t.png": "experiments/qwen_d1_fractions/figures/kept_vs_t.png"
- "fig/e89_log_ratio.png": "experiments/qwen_tofu_adversary/figures/log_ratio.png"
- "fig/e99_net_share.png": "experiments/definition_verify/figures/net_share.png"
- "fig/e99_class_map.png": "experiments/definition_verify/figures/class_map.png"
- "fig/e100_reverts_sum.png": "experiments/fed_or_written/figures/reverts_sum.png"
- "fig/e93_clean.png": "experiments/qwen_d1_table/figures/clean.png"
- "fig/e93_share.png": "experiments/qwen_d1_table/figures/share.png"
- "fig/e93_kept_clean.png": "experiments/qwen_d1_table/figures/kept_clean.png"
- "fig/e94_outside.png": "experiments/qwen_d3_rows/figures/outside.png"
- "fig/e94_placement.png": "experiments/qwen_d3_rows/figures/placement.png"
- "fig/e94_think.png": "experiments/qwen_d3_rows/figures/think.png"
- "fig/e103_gain_curve.png": "experiments/known_answer_pair/figures/gain_curve.png"
- "fig/e103_pair_rules.png": "experiments/known_answer_pair/figures/pair_rules.png"
- "fig/e103_amend1_shared.png": "experiments/known_answer_pair/figures/amend1_shared.png"
- "fig/e103_amend2_overlap.png": "experiments/known_answer_pair/figures/amend2_overlap.png"
- "fig/e105_overlap.png": "experiments/rows_null/figures/overlap.png"
- "fig/e107_overlap.png": "experiments/label_vs_style/figures/overlap.png"
- "fig/e104_outside.png": "experiments/qwen_d3_gsm8k/figures/outside.png"
- "fig/e104_placement.png": "experiments/qwen_d3_gsm8k/figures/placement.png"
- "fig/e104_think.png": "experiments/qwen_d3_gsm8k/figures/think.png"
- "fig/e104_size.png": "experiments/qwen_d3_gsm8k/figures/size.png"
- "fig/e94_known.png": "experiments/qwen_d3_rows/figures/known.png"
- "fig/e109_seeds.png": "experiments/larger_size/figures/seeds.png"
- "fig/e109_verdicts.png": "experiments/neuron_vpd_table/figures/verdicts.png"
- "fig/e112_removal.png": "experiments/case_label/figures/removal.png"
- "fig/e112_amend1_removal.png": "experiments/case_label/figures/amend1_removal.png"
- "fig/e112_amend2_rank.png": "experiments/case_label/figures/amend2_rank.png"
- "fig/e113_readout.png": "experiments/case_patch/figures/readout.png"
- "fig/e113_flips.png": "experiments/case_patch/figures/flips.png"
- "fig/e114_removal.png": "experiments/case_locator/figures/removal.png"
- "fig/e114_reverts.png": "experiments/case_locator/figures/reverts.png"
- "fig/e116_possession.png": "experiments/definition_table/figures/possession.png"
- "fig/e116_carrying.png": "experiments/definition_table/figures/carrying.png"
- "fig/e117_seed.png": "experiments/neuron_vpd_table/figures/full_67m_seed.png"
- "fig/e117_set.png": "experiments/neuron_vpd_table/figures/full_67m_set.png"
- "fig/e117_width.png": "experiments/neuron_vpd_table/figures/full_67m_width.png"
- "fig/e118_e93.png": "experiments/neuron_vpd_table/figures/full_qwen_e93.png"
- "fig/e118_pairs.png": "experiments/neuron_vpd_table/figures/full_qwen_pairs.png"
- "fig/e118_known.png": "experiments/neuron_vpd_table/figures/full_qwen_known.png"
- "fig/e119_route_labels.png": "experiments/teacher_route_split/figures/route_labels.png"
- "fig/e124_p0_blocks.png": "experiments/natural_tasks/figures/p0_blocks.png"
