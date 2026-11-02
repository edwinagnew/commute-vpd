# Plan: what training does when there is only the base model to learn from (E34 Amendment 8)

*2026-11-02. The user's task, restated in the Terms' words: run teacher training and the other methods with the base
model W₀ as the only thing to learn from, to see whether a gap between W₀ and the decomposition, rather than the gap
between W₀ and the fine-tuned model, is what they learn. Add this as an amendment and say what it breaks. Evidence:
`reports/reviews/2026-11-02_review_base-control-audit.md`, Fable's `reports/plans/2026-11-02_advice_base-control-FABLE.md`.*

## Answer so far (nothing run yet)

No standing result breaks on what the code and saved files show. Teacher training cannot see the feared gap at its
start: its student is the stored W₀ plus a hook adding only the change, so with teacher = W₀ the two are the same bytes.
That is also why the control we have, E34's C2, could never have failed. Its gradient is exactly 0, so it took no real
step. The recipes that do contain the gap (masked teacher training, E37 Amendment 1, E42 Amendments 1-2) ran this
control and failed it at about 2 to 27 times the fail line, so your worry was real there and it is why that line closed.

What is unprotected is the claim that teacher training's move *is* the update. Student size barely follows the
teacher: 0.014, 0.017, 0.018 for teachers of 0.011, 0.022, 0.032 (E41, emoticon). Adam takes steps of about its learning
rate from any nonzero gradient. Nothing has measured how much of the move would happen anyway.

## Why else this control is needed

1. **Adam ignores gradient size.** A tiny gap drives steps as large as a real one, so the move may be mostly the recipe.
2. **Rounding feeds the optimiser.** Off W₀ the hook path reads 0.0005-0.0007 nats against merged weights (bf16; "hook
   floor"), so a student away from W₀ always has a gradient. Every E41 chain link starts away from W₀.
3. **Matching outputs does not fix (U, V).** Reachable teacher: gap closed to 0.5%, offsets recovered only 0.76. Whether
   the place training ends is a faithful decomposition has been read only with a teacher present.
4. **Readings below the base are unexplained.** Check ratios of 0.89-0.96 and a falling Δ-out ratio (0.72-0.97; the base's
   own true gap, 0.012 nats, in no hook-path loss) have no reference. Likely "at the base", untested.
5. **Labelled training has no reference at all.** E39 Amendment 2's labelled Adam 7.24 and SGD 1.28 have no reading of the
   same recipe on labels the base already predicts.
6. **Data.** W₀'s masked error is larger on task windows than on Pile (CI-masked 1.13 against 0.35 per position), so
   masked recipes learn something from text alone.
7. **Card.** The base's floor differs by card (0.0748 H100, 0.209 L40), so references must share the arm's card.

## Design check

- **Goal.** `GOALS.md` weak version: D′ scores on W + δW as D does on W. A map only earns this if its move is caused by δW.
- **Nearest tried.** E34 C2 (0 by construction); E41 C2 (lr 0); E37 A1 M0/M0b and E42 A1-A2 (masked, nothing to copy,
  failed; coefficient 0 never run); E39 A2's random move (does not train); E45's self-distillation null (Qwen). New here:
  a control that really moves, on the unmasked recipe every standing pass uses.
- **Patch or step.** A control on standing claims, in approved class 2, at the user's request.
- **Decisions relied on.** Γ, Δ frozen; teacher training = Adam 1e-5 on alive U, V (`decisions.md`). No departure.
- **Control first.** W₀ through the Pile harness in each job, alone, stop on a miss. Each job reads its own W₀ and
  do-nothing on the L40; L40 lines 1.24 / 1.81, floor 0.209 (E42). No cross-card comparison.

## Runs (MATS cluster `debug`, L40, ≤ 35 min each, $0; cheapest first)

E34's arm A recipe unless stated: emoticon task windows, Adam 1e-5, batch 32, 1,000 steps, batch-order seed 0, Γ and Δ fixed,
read at 0, 100, 1,000. Emoticon is chosen because E41 and E43 regenerated it, so references exist.

| Run | What runs | Expected | If not |
|---|---|---|---|
| N1 | Teacher = W₀, start = random offsets on the alive U, V matched to E41's one-shot size (0.018) | Gap to the hook floor; check ≤ 1.24 at 1,000 | Check above the fail line with the gap closed: training itself wrecks faithfulness away from W₀. Check not failing with the gap not closed: see the Tree; likely, since a small gap gives Adam a walk (code review, 2026-11-02) |
| N2 | Teacher = W₀, start = E41's teacher-trained T1000 state (regenerated in-job): the round trip | Size falls ≥ 0.76 of the way; check ≤ 1.24 | E41's chains and any pass reached from a moved start carry a caveat; E37 A1 arm 3's rise reads as the method's drift |
| N3a | The null fine-tune alone (Adam 3e-5, batch 32, each label drawn from W₀'s own distribution), read at 1, 3, 10, 30, 100, 300, 1,000: size, KL to W₀, P(target) | Walks as the square root of the steps; chooses the checkpoint nearest the same-job T1000's size (near step 100) | Decides n3b's teacher and its expected values, filed by dated note before n3b runs |
| N3b | The student from W₀ on that checkpoint, beside the same-job real student at batch orders 0 and 1 | Not "the recipe"; check ≤ 1.24 | Comparable size and a direction shared with the real student: the move is largely the recipe, and "copies the update" weakens, though the passes stand. Comparable size in another direction: a walk that happens anyway, but not the real student's move |
| N4 | Labelled training on the same self-labels: alive U, V, Adam and SGD at E39 Amendment 2's rates, on its lowercase rows | Near the base (check ≤ 1.24) | Near 7.24: E39's labelled fail belongs to the recipe, not the broad update (P10's reading changes) |
| N5 | Masked recipe at coefficient 0: component path, E42's rand32 windows, teacher = W₀ | Step-1 KL 0.00163; check ≤ 1.24 | Check ≥ 1.81: "the masked terms did it" becomes "path plus Adam; masked terms worsen it" |

Numbers read, no others: check ratio, stochastic, CI-masked and Δ-out ratios, unmasked KL to W₀, size, P(target) where the
task has one, return fraction (N1, N2). N1 and N2 run first, alone; N3 reads next. The rates for N4 are copied from
`experiments/broad_fragility/amend2.py`. Noise: W₀'s ratio moves 0.21 between adversary seeds on the L40, so readings are
class calls on the pass and fail lines, not differences. N3b's thresholds on size and direction are fixed in the predictions
before the run, with their edge cases (`.claude/CLAUDE.md` → Rules). N3b also reads the cosine of the weight changes, the one
number added here: size alone would read "about" for a reason unrelated to the question (code review, 2026-11-02).

**Blind to.** One task, one batch order, one adversary start, 67M, 20-step adversary. Qwen's teacher training has the same
shape (`src/commute/qwen_teacher.py`), and the re-gating readings start from the base's CI-masked error; both go as notes
to their leads, not runs.

**Not run, and why.** More teacher = W₀ from W₀: 0 by construction. The splits and step split: at δW = 0 they return 0
and take no training. E15: done, failed, read correctly. H100: references exist on the L40 and it is free.

## Tree

- All as expected: claims stand, wording changes only ("moves nothing" becomes "0 by construction"); ratios below 1 read as "at the base".
- N1 or N2 fails on the check, or N2 does not return: E41's chains get a card; the review says what else starts from a moved state.
- N1 "fails, gap not closed" (check not failing, KL to W₀ above the line): no card by itself. Teacher training does not restore W₀'s outputs from a small random gap; it walks while the check holds. Say so in the result message, and read N2's return with that walk in mind.
- N2 gap closed with P(target) above 0.04: the behaviour is still there, so it counts as not back.
- N3b "the recipe": a pending-decision card on what "teacher training copies the update" may claim, before Qwen uses it. N3b "partly": goes to the user as a surprise, no automatic card. N3b "a walk elsewhere": stated in the claim's wording, no card.
- N4 fails: card on E39 Amendment 2's reading. N5 fails: reopening the masked line becomes a card; nothing reopens by itself.
- Any qualitative surprise goes to the user before the next run.

## Gates and cost

Not a gated run: $0, under 3 h. One fresh Opus review of the code before running.
