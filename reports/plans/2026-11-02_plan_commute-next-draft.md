# Plan (DRAFT: Fable's advice, not approved work): what the commute line does next, if anything

*2026-11-02. For commute-lead. Read: `GOALS.md`; log cards `next-commute`, Q11, `decision-chain-size`,
`decision-e47-split-keep`; `experiments/TRIED.md` rows E41-E43, E47-E49, E34 Amendments 6-8c, E12 Amendments 1-3;
`reports/analyses/2026-11-02_analysis_e34-amend8-base-control.md` and its recheck; `backlog.md` 2026-10-29 and 2026-11-02 items.
Terms as in `GOALS.md`. "Student" = the decomposition after teacher training, Γ and Δ frozen.*

## Where the line stands, in plain words
On the 67M, teacher training is the one map that keeps a narrow fine-tune's behaviour and passes the adversarial check
with Δ untouched, at four adversary starts (E34, E42, E47). It fails on the broad Adam fine-tune and passes its rank-8
LoRA (E39, E49). Putting the rest of the fine-tune into Δ to get exact weights fails (E43). Plain SGD works at smaller
rates (E47). The route split on students still parts copy tasks from random maps; one table loses a fifth of the
re-switching term, not from precision (E12 Amendments 1-3). A student's move lies only 7-13% along its own fine-tune;
87-93% is along neither teacher, reproduces across batch orders, and is unidentified (E34 Amendment 8c). The two cards
waiting on the user both recommend wording only. The planned tree is used up; the log's "Next" names nothing unrun.

## Options
**A. Nothing runs; the line waits on the user.** The two no-rush cards close as the user reads them; the write-up
(writeup-lead, ~2026-11-11) takes the results as they stand. Next: the user says whether the weak version on the 67M
is settled. If yes, the line's remaining question is the strong version (`GOALS.md`, postponed 2026-10-14: a
re-decomposition of one fine-tuned 67M, a gated VPD run) or the line ends on this model; if no, the user names what is
missing.

**B. Teacher training held near the fine-tuned weights** (backlog 2026-10-29, Fable's advice taken by the user; new
method). The teacher loss plus a weight-space penalty toward W₀ + δW, on the emoticon fine-tune at step 1,000, with
do-nothing, plain teacher training and the least-squares split on the same rows; one cluster `debug` job, $0, ~25 min.
Tells: whether a state whose weights are the fine-tuned model's can pass the check, where E43's remainder route read
2.5 and 1.9 times the student's ratio. Pass: an exact-weights map exists and E43's "own state stays the map" is
revised. Fail: E43 stands with a second route tried. Not in an approved class: the user's confirmation first.

**C. The LoRA rank sweep** (backlog 2026-10-29, deferred by the user at `decision-e47-lora-rank`). LoRA fine-tunes of
the lowercase task at ranks 1, 4, 16, 32, each teacher-trained by E39's recipe and read on the same Pile blocks; two
cluster jobs, $0. Tells: whether the broad task's only pass is bound to a small rank (passes stop at some rank) or the
Adam full-matrix fail is about the update's kind (all pass). Re-raising a deferred item is the user's call.

**D. What the student's move is, read from saved offsets** ($0, class 1: analysis of what exists; named unrun in
E12 Amendment 2 and E34 Amendment 8's "What next"). The student's change is dU Vᵀ + U dVᵀ + dU dVᵀ, so it can lie only
in the subcomponents' span; the fair reference is the inside part of δW (δW minus the least-squares remainder), not δW.
For E12 Amendment 2's six students, Amendment 8's real and null students and E42's: cosine to the inside part, cosine
to δW (8c's number, re-read), and the inside share of δW, which bounds the first. Control first: the least-squares
split state reads cosine 1 to the inside part, by construction. Expected: real students near the bound if the move is
a copy of what the pieces can express, far below it if the unidentified part is the method's own; null student near 0
on both. One cluster `debug` job, minutes. It changes the wording of the line's headline claim ("copies the fine-tune
into the subcomponents" against "reproduces its behaviour by a change the pieces can express") and the chain card's
reading of size, and gives the Qwen lead the reference its teacher-training read lacks. It runs no training.

## Recommendation
**A, with D run once before the line closes.** D is a $0 read two reports already name, needs no sign-off, and the
headline claim reads differently depending on it; nothing else is worth starting without the user. After D, the line
has nothing to run: say so plainly and wait for the user's three decisions (the two cards; settled or not).

**What would change it.** The user opens the strong version: a plan for re-decomposing one fine-tuned 67M (gated).
Qwen's teacher training (qwen-decomp-lead) fails or needs exact weights: B, shown on the 67M first (user, 2026-10-28).
The write-up finds the rank question load-bearing for C4: C. Any other lead's reading of the 67M students is reused,
not repeated.

## Owners checked
Comparisons with cheaper methods, the method table, reuse, copying check: baseline-lead (none of A-D). Re-gating and
elicitation: regating-lead (D reads no r). Qwen: qwen-decomp-lead (D's reference is handed over, not run on Qwen).
Red team, write-up, cards: untouched. Grepped `reports/plans`, `reports/decisions`, `reports/analyses` since 2026-10-28.
