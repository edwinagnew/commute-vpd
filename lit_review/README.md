# Literature review

This is the project's one place for literature, shared by every lead through the repo. Start with the overview table,
which says what each note concluded for us. The rest is the record the notes draw on.

*Overview last updated 2026-11-11.*

## Where the literature leaves us (2026-11-11)

- **No one carries a weight decomposition across a training update.** That gap is the project's question
  (`top_10_report.md`).
- **The field's default is that small fine-tunes reuse existing machinery.** It holds that on activation evidence and
  attack tests (`2026-11-11_alignment_applications.md`).
- **No instrument tests reuse at the level of mechanism against a known label.** A fixed SAE can only report existing
  features. Crosscoders can report "new", but as an activation direction in one layer, and they can mislabel an old
  feature used in new contexts as new. No diffing method has been scored on a fine-tune whose new-against-reused label is
  known by construction (`crosscoders_and_novelty.md`).
- **So the contribution to aim for** is a reading from the weights that can return either answer, validated on built
  controls, with crosscoders scored on the same controls.
- **No established definition of reuse gives a verdict on our tasks.** The definition work is in `definition_survey/`.

## Theme notes

Each note answers one question. Read the bottom line first; open the note for sources and quotes.

| Note | Question | Bottom line | Date |
|---|---|---|---|
| [`top_10_report.md`](top_10_report.md) | What exists on decomposition methods, what fine-tuning does, and tracking features through training? | No paper carries a weight-space decomposition across an update. 91 papers read | 2026-10-15 |
| [`../theory/reveal_vs_create_literature.md`](../theory/reveal_vs_create_literature.md) | Does post-training reveal or create? | A spectrum from "only selects" (persona selection) to "builds"; abstracts and posts only | 2026-10-14 |
| [`2026-10-29_baselines_notes.md`](2026-10-29_baselines_notes.md) | Which non-VPD methods would a reviewer ask us to compare with? | Neurons, crosscoders, SAE transfer, steering and others, with a verdict on each | 2026-10-29 |
| [`elicitation_vs_sharpening.md`](elicitation_vs_sharpening.md) | How does our "elicitation" relate to "distribution sharpening"? | Different questions: none of the 16 papers looks inside the model, so none tests our definition | 2026-11-05 |
| [`definition_survey/`](definition_survey/) | What do existing papers define as mechanism, reuse and capability? | Blocks by field, listed below. No definition found that gives a verdict on our tasks | 2026-11-09 |
| [`2026-11-11_alignment_applications.md`](2026-11-11_alignment_applications.md) | Is there an alignment application that uses our readings unchanged? | Reuse is already the field's default in reward hacking, emergent misalignment and unlearning. No experiment adopted; two open questions named | 2026-11-11 |
| [`crosscoders_and_novelty.md`](crosscoders_and_novelty.md) | Can crosscoders already measure whether a fine-tune built something new? | Partly: they flag activation directions specific to one model, not mechanisms, can read reuse as new, and have never been scored against a known new-against-reused label | 2026-11-11 |

### Definition survey blocks

Verbatim definitions from each field, read from PDF text.

- [`definition_survey/mechanism.md`](definition_survey/mechanism.md): what "mechanism" and "reuse" mean outside the
  fine-tuning papers, and in the project's own catalogue.
- [`definition_survey/finetune.md`](definition_survey/finetune.md): mechanistic accounts of fine-tuning.
- [`definition_survey/causal.md`](definition_survey/causal.md): causal abstraction and circuits.
- [`definition_survey/transfer.md`](definition_survey/transfer.md): feature reuse in transfer learning.
- Four more blocks exist only in `definition-lead`'s clone and are not pushed as of 2026-11-11:
  `abstraction_between_networks.md`, `elicitation_formation.md` (includes the unlearning definitions), `information.md`,
  `regime.md`.

## The record

- `catalog.md`: every summarised paper (106), one entry each, with its score and a link to its summary.
- `reading_list.md`: the same papers by score.
- `summaries/`: one structured summary per paper.
- PDFs: `~/commute-data/lit_review_pdfs/`, not in git. The older files say "pdfs/" for this folder.
- `research_proposal.md`, `search_terms*.json`, `stage1_report.md`, `progress.json`: how the October sweep was run.

## How far to trust it

- Scores are the summarising agents' own and have not been checked. They were given against the brief of their day, so
  a score only compares with others from the same batch.
- Summaries were read by agents, so check a number against the PDF before quoting it.
- Each theme note states at its top how it was read. The two notes of 2026-11-11 were read mostly from web pages
  through an extraction model: their quotes are unchecked against raw text unless marked, and their new papers have no
  entry in `catalog.md` or `summaries/` yet.

## The standard

A theme note:

1. Is named for its question; add the date as a prefix when it records a dated decision or pass.
2. Opens with a dated line: who read it, how (PDF text, web extraction, abstracts only), and what was not checked.
3. Gives a "Bottom line" of a few sentences before any detail.
4. Gives a URL for every paper, and marks any quote not checked against the raw text.
5. Ends with "Not verified".
6. Gets a row in the table above, with its bottom line in one sentence.

## Adding to it

1. For a paper read in full: put the summary in `summaries/`, in the format the `lit-review:summarizer` agent writes,
   and the PDF in the PDF folder. Add an entry to `catalog.md` and `reading_list.md`.
2. Add it to the theme note it bears on, or start a new note if it opens a new question.
3. Add or update the note's row in the table above, and push, so the other leads see it at their next pull.

A paper is not in the review until a theme note says why it matters. For a new search rather than a paper in hand, run
`/lit-review:lit-review`.
