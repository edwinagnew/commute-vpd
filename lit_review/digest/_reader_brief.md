# Brief for a paper reader (cell P2.71, due 11-25)

You read ~10 papers properly and write one entry per paper. Writing only: no runs, no edits outside your output files,
no commits, no paper edits.

## Read first
1. `/mnt/nw/home/e.agnew/commute-agents/litreview-lead/lit_review/digest/_paper_context.md` — our paper (claims,
   definition of reuse, methods, baselines, results). Everything you judge is relative to it.
2. `/mnt/nw/home/e.agnew/commute-agents/litreview-lead/lit_review/digest/_template.md` — the entry format. Follow it
   exactly, frontmatter included; tags only from its fixed list.
3. `/mnt/nw/home/e.agnew/commute-agents/litreview-lead/lit_review/digest/_paper_refs.tsv` — our draft's bibliography,
   for the `in_our_bib` / `cited_in_draft` fields (match by arXiv ID or title).

## Your papers
Your batch file (path in your prompt) has tab-separated rows: slug, title, authors, year, id_or_url, summary_file,
source_files. Write each entry to `/mnt/nw/home/e.agnew/commute-agents/litreview-lead/lit_review/digest/papers/<slug>.md`.

## How to read
- Read the actual paper, not the old summary: at least the abstract, the method and the results. For arXiv use
  `https://arxiv.org/abs/<id>` for the abstract and `https://arxiv.org/html/<id>` (or `https://arxiv.org/pdf/<id>`) for
  the body; LessWrong / blog URLs directly. WebFetch truncates at 100k characters: for a long paper ask it about the
  method and results sections specifically, or `curl -sL` the HTML into your scratch dir and `grep -nF` / `sed -n` it.
  Local PDFs for some papers: `~/commute-data/lit_review_pdfs/` (text via `~/commute-agents/fleet/pdf-text FILE.pdf 2-6`;
  the Read tool cannot open PDFs).
- An existing summary in `lit_review/summaries/<summary_file>` may help you orient, but it is not a substitute and may be
  wrong; correct it in your entry where the paper disagrees.
- Rows with id "?" or "(title not given in files)": find the paper (WebSearch the title/authors given; check the
  source_files named in the row under `lit_review/` for context). If you truly cannot identify it, write the entry
  with `read_from: not found (what you tried)` and importance 0.
- Say in `read_from` what you actually read. Never write a result you did not read.
- Quotes are verbatim and short.

## What we most need from you (the user's complaint: a close known-answer case sat as a one-line note)
Look hard for:
1. **Methods we should compare against but don't** (tools that say what a fine-tune changed: diffing, attribution,
   weight analysis, probes, steering) — say whether it could run on our setting.
2. **Realistic fine-tunes with independent evidence of reuse or of something new**: the paper shows, by its own
   evidence (ablation, patching, steering, a mechanism traced in both models), that the fine-tuned behaviour runs
   through a base-model mechanism, or that it does not. These are known-answer candidates for us: name the model, the
   fine-tune, and the evidence.
3. **Definitions** of reuse / elicitation / mechanism / capability to cite or contrast with ours.
4. **Findings that support or contradict ours**, especially: do fine-tunes rescale existing components or rotate /
   replace them; does post-training only re-weight existing capabilities.
5. **Citation problems**: our draft should cite it and does not, or cites it for something it does not say.

Be calibrated with `importance`: most papers are 2-5. 8+ only if our paper is incomplete without engaging it.

## Hand back (keep it short: it goes into the lead's context)
One line per paper, nothing else but a final Friction line:
`<slug> | <primary_tag> | <importance> | <buckets or -> | <one line: what it means for us>`
then `Friction: ...` (or `Friction: none`).
