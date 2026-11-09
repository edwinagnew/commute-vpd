# P1.1, due 11-10: the second wording of reuse has a proposed first test, waiting for you

**Cells.** P1.1 open, 0 runs of 0: page written, not approved. P1.2 open, 0 runs of 0: not started, because today's
`decisions.md` row says "No verdicts on saved runs under either wording yet".

**Why I stopped.** Everything left waits on one ask only you can answer: whether to add the proposed run cells.

**What was done.** The second wording is recorded as adopted for testing. Its claims about Geiger et al. 2025, Prakash
et al. 2024 and Jain et al. 2024 were checked against the full papers. One sentence is wrong as written: Geiger et al.
do call adapter-tuning an intervention on model parameters. The rest hold with gaps, all listed in the appendix.

**What is proposed.** Four pairs on the 67M model, each with one added piece, so the label is known by construction:
a fixed push on an existing variable, a new wire from a cue, a table, and an XOR the model cannot read linearly. The
XOR is the only candidate for built. It also tests the first wording's fault, since that wording calls it reuse. The
search is not told the answers, and must find them on the three pairs where the answer is exactly known.

**Findings in the wording, left unchanged.** "M acts on φ" may exclude nothing. G and H read only the fine-tune's
outputs. The first wording's exact checks cannot hold on an ordinary fine-tune, so it gives no call there.

**Asks waiting.** Add run cells R1 and R2? Page: `reports/plans/2026-11-09_plan_reuse-variable-first-test.md`.

**Spent.** $0. No job, no pod, no model loaded.

**Friction.** The PDF page renderer is refused on the dev node, so two figures are known from captions only.
