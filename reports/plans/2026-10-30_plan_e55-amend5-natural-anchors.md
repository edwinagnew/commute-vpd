# Plan: E55 Amendment 5, reuse anchors trained on natural text (2026-10-30)

**Status:** approved class. The user chose option C of card `decision-e55-truth-set` (chat, 2026-10-30, "go with your
recommendations"; `decisions.md`). This plan says how. Free, on the MATS cluster's debug queue. Not a gated run.

## In plain words

Stage C needs fine-tunes that, by the Stage A checks, reuse LOW's existing count of a. Only two qualify today. Both
trained on letter-swapped rows that LOW itself miscounts (Amendment 3), and they answer such rows through a count of a
that got stronger (Amendment 4). We add fine-tunes where reusing the old count needs no repair:
- the slow all-spurious recipe trained on natural texts that already have count(b) = count(a) + 1, in four seeds;
- a fast copy of the same recipe;
- a fine-tune that asks for the old count under a prompt LOW has never seen, as a positive control for "reuse".

## Restated in the Terms

- **The model:** the base LOW, a minGPT pretrained on Jain et al.'s PCFG with count b rare.
- **The capability C:** the count of a, f_C = count_a. LOW possesses it (Stage A, Def 1).
- **The new task:** answer `C b 40` with count(b).
- **The training rows** are natural texts on which count(b) = count(a) + 1. So g(f_C) = count_a + 1 gives every
  training label through LOW's unchanged count.
- **Read with:** Stage A's behaviour check (B) and patch check (P), with the filed bars from `predictions.json` →
  `a2_bars`, on natural test rows where count(b) ≠ count(a) + 1. Amendment 4's off-label swap-built rows are read too,
  for description only.

## Arms (all from LOW; Stage A recipe otherwise: Adam, weight decay 0.1, batch 96)

| arm | training rows | lr | seeds | saves |
|---|---|---|---|---|
| WN | natural, count(b) = count(a) + 1, answer count(b) on `C b 40` | 1e-5 (W1's) | 0, 1, 2, 3 | 500, 1K, 2K |
| WNf | same | 1e-4 (W2's) | 0 | 500, 1K, 2K |
| ID | natural texts, answer count(a) on a prompt LOW never saw in pretraining | 1e-5 | 0, 1 | 500, 1K, 2K |

For ID, the prompt is a family or terminal code absent from LOW's pretraining stream; the code verifies it is absent.
ID's checks read count a at the new prompt with g = identity: B_a is the share answering count_a. Its P_a patch uses
pairs that differ in count a and share count b, at the new prompt.

## Expected values (filed in `predictions_amend5.json` before the run)

- **W₀ control, run first and alone:** LOW reproduces its Stage A natural checks, B_a and P_a, within 0.01.
- **LOW's coverage of the new training rows:** count_a + 1 is right on ≥ 0.99 of them. Amendment 3 measured 760 of
  760.
- **WN at 1K, each seed:** count a "through", count b "not through". So the state reads "reuses the old count".
  - Lean: 70%. The recipe needs no repair, and W1 read this way on natural text at 1K.
  - The main alternative is "neither". The narrow labels allow a near-constant answer: 14 is right on 67% of
    training rows.
  - **Shortcut check:** the share of training-distribution rows answered 14, against the 67% of rows whose label is
    14. A state reading "neither" with ≥ 0.6 of its answers equal to 14 is labelled "shortcut", not "new".
- **WN at 2K:** the same reading. Lean 60%; drift past 1K is not known on this recipe.
- **WNf at 1K:** a lean of 40% "reuses", 40% "neither". W2 at 10K answered a rule of two counts.
- **ID at 1K:** count a "through". Lean 85%. This is the positive control for the "reuses" side. If it fails, the
  checks or the recipe cannot show reuse under a new prompt, and WN's readings are not interpreted.
- **Off-label swap-built rows, described only:** WN should answer like LOW there, because it never saw swapped text.
  That means B_a well under W1's 0.67.

Edge cases to run against each rule before filing:
- void pairs (under 500);
- B and P straddling a bar;
- all four seeds split two and two;
- a shortcut state that also passes count a;
- ID's prompt colliding with a pretraining code.

## Design check (`/design`)

- **Goal:** the reuse line's ground truth (`GOALS.md`: does post-training strengthen existing capabilities or build
  new ones), defined without VPD.
- **Nearest tried:** E55 Stage A and Amendments 1-4 (`TRIED.md`).
- **What is new:** natural-only rows remove the premise Amendment 3 broke. ID was Fable's suggestion, recorded on the
  truth-set card.
- **What it could change:** WN reading "reuses" in most seeds brings the "reuses" side to roughly six to ten states,
  enough to score instruments in Stage C. WN reading "neither" or "new" would mean that on this grammar even a
  no-repair recipe does not keep the old route. That is a finding for the card, and Stage C goes ahead as option A
  with option B, with the imbalance reported.

## Order and stop rules

1. **Model-free build.** The rows have the filed counts, ID's prompt is unused, and the label histogram is recorded.
   It runs at tiny size on the cluster first: under a minute.
2. **The W₀ control, alone.** On a miss, stop.
3. **ID seed 0, then the WN seeds, then WNf.** Each state's readings are written as it finishes, and each arm resumes
   from its saves. Each arm takes about 1 min of fine-tuning plus its readings; Stage A's 10K-step arms took ~3 min.
4. **Then:** the write-up, an independent check, a log update, and Stage C as option A with option B on the labelled
   set.

## Departures from the literature

- **Bounded g (user decision, 2026-10-30).** Jain, Kirk, Lubana et al. 2024, §3, Def 2, calls a fine-tune reuse when
  the post-trained map is g∘f_C, for any g. We add that g, learnt on the fine-tuning examples, must also work on
  held-out values. Here g is +1 (WN) or the identity (ID), and both generalise, so the clause changes no reading in
  this amendment. It does change the cipher and copy arms (Stage B, planned separately).
- **Natural rows with count(b) = count(a) + 1, not the paper's spurious construction.** Jain et al. make the
  spurious correlation by editing texts. We sample texts that already carry it, because Amendment 3 showed the edit
  breaks the base model's count. The cost is narrow labels, 13 to 15, which the shortcut check above addresses.
- **The new-prompt (identity) arm** is not in the paper. It is our positive control.
- **The causal checks** (behaviour B and patch P), the bars and the "neither" class were added by this project to
  Def 2 in the Stage 1 plan (`reports/plans/2026-10-29_plan_reuse-stage1.md`). They are unchanged here.
- **Saves at 500 to 2K steps,** where the paper reads its fine-tunes at the end of training. Stage A found the
  reading changes between 1K and 10K steps.

## Then: Stage B with bounded g (next plan, after this one runs)

Held-out readings of the 67M cipher, copy and lowercase fine-tunes are one free cluster job. Does g, learnt on the
fine-tuning words, work on held-out words? A lookup-table cipher is expected to fail, which makes it "new" under the
bound. That plan goes through `/design` once this amendment is read.
