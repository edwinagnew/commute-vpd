# definition-lead, 2026-11-09: general definition page ready, waiting on the user

**Cells.** P1.1, due 11-10, open, 0 runs of 0: the page is written and needs your choice. P1.2, due 11-10, open, 0 runs of 0:
waits until a definition is chosen; the draft stays a draft. Filled: none. Finished because everything left waits on one ask.

**What was done.** You rejected the frozen-body page. I read how 30 sources define reuse and wrote one page with three
candidate definitions: `reports/plans/2026-11-09_plan_general-definition.md`, with the survey table in the appendix beside it.

**What was found.** No source says, in terms of what happens inside a model, when a fine-tune reuses a mechanism and when it
builds one. The page recommends a definition built from causal abstraction and Prakash et al.'s question: the fine-tune reuses
a mechanism when the finest account of its new behaviour runs through values the original model already computes and uses, at
the same place. The step across two networks is ours and is marked as such.

**Corrections to earlier claims.** The plan's "causal step from Prakash et al." is not in that paper. Raghu et al. 2020 exists
and says what was assumed, but it concerns meta-learning.

**Waiting on you (ask filed).** Which of the three definitions, if any, you adopt, by a dated `decisions.md` row. Candidate 2 needs
one number with it. Test pairs and proposed runs follow as a second step.

**Ran.** Nothing: no job, no pod, $0. Slurm jobs 56787 and 56760 running on the account belong to the Qwen work, not to this line.

**Friction noted.** The Read tool cannot open PDFs on the dev node; a text-extraction script now sits at
`~/commute-data/scratch/definition-lead/p1.1-survey/pdf_to_pages.py`.
