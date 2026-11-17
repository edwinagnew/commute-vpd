# Digest 2026-11-07 (qwen-decomp-lead): the second Qwen decomposition

**Why finished:** the user's task (download, check and compare Goodfire's second Qwen3-1.7B decomposition) is done, and
the next work, the teacher-training runs E111, waits on two decisions only the user can make.

**Answer.** The new decomposition loads and works in our code, and every check against the authors' library passed. On
GSM8K maths text it is about seven times harder for VPD's mask adversary to break than the old one, with thinking on or
off. With its own masks it is about as faithful on maths and less faithful on web text, which lies outside its training
data. It also switches on about 30% fewer subcomponents per token. The reading is provisional: one adversary seed and
128 rows per set.

**Ran.** E131, MATS debug L40 jobs 55508-55542, $0. The filed predictions scored P1 a miss (own-mask error on maths,
thinking off: new/old 1.07) and P2-P4 passes (adversary on maths 0.14 and 0.15 of the old; own-mask error with thinking
on 0.88). Report: `reports/analyses/2026-11-07_analysis_e131-new-decomposition.md`. Notebook:
`notebooks/E131_new_decomposition.ipynb`. Log v284 has the card.

**Decisions for the user** (log card "Should the faithful-map work on Qwen move to the new decomposition?"):
- Ask 38: move E111 and the faithful-map deliverable (D1: a map on Qwen3-1.7B that keeps the decomposition faithful
  after an update) to the new decomposition. Recommended: yes, which means redoing E111 stage 1 on it, about 1 h.
- Ask 36: the go-ahead for E111 stages 2-5, the remaining 10 updates, about $10-15 of RunPod.

**Spend.** $0 this session (cluster only); no pods.

**Friction noted.** sacct was refused while slurmdbd was down. The log flush needed 8 rounds of trimming to fit the
page's length rule, and a publish refusal printed a 70k-token diff.
