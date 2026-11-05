# Literature review

This is the project's one place for literature. Start with the theme notes, which say what the papers mean for us. The rest is
the record they draw on.

## Theme notes

- [`elicitation_vs_sharpening.md`](elicitation_vs_sharpening.md): how our mechanistic "elicitation" relates to the literature's
  "distribution sharpening"; 16 papers (2026-11-05).
- [`top_10_report.md`](top_10_report.md): the October sweep's overview: decomposition methods, what fine-tuning does, tracking
  features through training (2026-10-15).
- [`../theory/reveal_vs_create_literature.md`](../theory/reveal_vs_create_literature.md): does post-training reveal or create; persona
  selection, shard theory, emergent misalignment (2026-10-14).
- [`2026-10-29_baselines_notes.md`](2026-10-29_baselines_notes.md): the non-VPD methods a reviewer would ask us to compare with.

## The record

- `catalog.md`: every paper, one entry each, with its score and a link to its summary.
- `reading_list.md`: the same papers by score.
- `summaries/`: one structured summary per paper.
- PDFs: `~/commute-data/lit_review_pdfs/`, not in git. The older files say "pdfs/" for this folder.
- `research_proposal.md`, `search_terms*.json`, `stage1_report.md`, `progress.json`: how the October sweep was run.

Scores are the summarising agents' own and have not been checked. They were given against the brief of their day, so a score only
compares with others from the same batch. Summaries were read by agents, so check a number against the PDF before quoting it.

## Adding to it

1. Put the summary in `summaries/`, in the format the `lit-review:summarizer` agent writes, and the PDF in the PDF folder.
2. Add an entry to `catalog.md` and `reading_list.md`.
3. Add it to the theme note it bears on, or start a new note if it opens a new question. A new theme note gets a line above.

A paper is not in the review until a theme note says why it matters. For a new search rather than a paper in hand, run
`/lit-review:lit-review`.
