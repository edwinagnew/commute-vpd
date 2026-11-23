# E276 blind rater prompt (cell P2.66, confirmatory text panel)

Give the text between the two lines below, unchanged, to each fresh rater (one Opus agent with no tools, one GPT agent
with no tools), in this fixed order for both (a seeded shuffle, `shuf --random-source=<(yes 276)`). Save each answer
verbatim as `results/raters/rater_opus.md` and `results/raters/rater_gpt.md`, then transcribe the tier column into
predictions.json `raters` (form in `raters_form`; one-liner number to task name below) before any train job.
Wording: the base-model line and the instructions are E272's (`~/commute-data/scratch/regating-lead/e272-wide/tasklist.md`
section 2); each one-liner follows E272's form ("German: German Wikipedia text.").

One-liner number -> task: 1 dutch, 2 vietnamese, 3 basque, 4 pubmed, 5 books, 6 math, 7 welsh, 8 italian, 9 turkish,
10 stackexchange, 11 indonesian, 12 portuguese.

---

Base model: a small GPT-style language model with 67M parameters, pretrained only on the Pile (English web text, books,
Wikipedia, papers, code and more). It has no instruction tuning.

Each task below is a short fine-tune of this base model on the text described (plain next-token training on that
text).

Rank the tasks with ties, an uncertainty and a written reason, from "expected reuse" (the base model already does most
of this and the fine-tune strengthens it) through "middle" to "new" (the base cannot do it and the fine-tune must build
it).

Answer with one table, one row per task: task (number and name) | rank (1 = most reuse; ties share a rank) | tier
(expected reuse, middle or new) | confidence (high, medium or low) | reason (one sentence). Then one line naming the
tasks you are least sure of.

1. Dutch: Dutch Wikipedia text.
2. Vietnamese: Vietnamese Wikipedia text.
3. Basque: Basque Wikipedia text.
4. PubMed abstracts: abstracts of biomedical research papers.
5. Books: the full text of public-domain English books from Project Gutenberg, each with its Project Gutenberg licence header.
6. Math web pages: web pages with mathematical content (blog posts, forum threads, lecture notes), with their LaTeX kept.
7. Welsh: Welsh Wikipedia text.
8. Italian: Italian Wikipedia text.
9. Turkish: Turkish Wikipedia text.
10. StackExchange questions: questions posted on StackExchange question-and-answer sites, in their raw HTML markup.
11. Indonesian: Indonesian Wikipedia text.
12. Portuguese: Portuguese Wikipedia text.

---
