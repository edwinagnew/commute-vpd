# Framing check 1: the copy worked example and the built edits' site (redteam-lead, 2026-11-24)

regating-lead is right: the base model scores 0 on copy24, the task on which Fig 2's built edits are read. This does not break "built edits pass" or the copy example, because the kept clause has no clause requiring the base model to do the task. It does break the word "strengthening" for the built edits, and the text must name the site. The 0.10-0.14 head kept share survives the P1.62 conventions on Fig 2's own trained states, but its random bound does not.

## (a) M's top-1 at copy24 is 0: confirmed (SHOULD)
- **Built-edit windows.** `built_circuits/results/planning/measure.json`: `.copy_top1_heldout` = 0 (75 held-out windows) and `.copy.top1` = 0 (301 training windows), checked. The eight layer-1 neurons that Fig 2's built edits scale have a net signed attribution of −67.9 (`.copy.signed_sum_L1`, checked).
- **Other copy tasks.** cmap e58-copy reads W0 0/279 on each seed, and c3-copy 0/236. On E62's copy-at-marker M scores 0.064 (checked). On plain repeats it scores 0.825 (`kstar_levels/results/stage0.json .induction.top1_second`).
- **Where the edits are read.** x2, x10 and rebuilt are read on copy24's held-out windows at every position (`kstar_levels/run.py:222`), never on E62's windows. So the built block's K is F's top token at positions of a task M gets 0 on.
- **Not checked.** M's top-1 on E147's 96-key copy windows.

## (b) What it does to the claims: SHOULD (wording, not a reversal)
- **"Built edits pass" still tests the implementation.** The kept clause compares each part's mechanism with M's, and nothing in it requires M to get the task right. So x2 and x10 are kept by construction.
- **"Strengthening" breaks.** The eight neurons are not a behaviour M has, and their net attribution is negative. Say "rescaling base parts", and name the site in the text and the Fig 2 caption ("eight layer-1 neurons picked on a copy task the base model scores 0 on"). Say it is one site until P1.63 reports.
- **The copy example stands.** Head (2,4) is M's induction head.
- **Sentences on origin/v2 to change:**
  - main.tex 52-53 and 696-697
  - sec7_discussion.tex 9-11
  - sec5_results.tex 16, and the Fig 2 caption at 54-56
- **Sentences that can stay, with a label.** main.tex 700-702 and sec5 26-35, but they should say the copy fine-tune here is E62's copy-at-marker, not Fig 2's trained copy.

## (c) 0.10-0.14 kept: OK on source and convention, SHOULD on two numbers
- **Source.** E277, `needed_neurons/results/summary.json [.items["e62/copy_s0","e62/copy_s1","e62/copy_s2"].h24_kept_share]` = 0.131, 0.095, 0.141 (checked). E295 reproduces them (`rotation_kept/results/headnull/summary.json`). "At most 0.005" is E295's random maximum, 0.0047.
- **Convention.** The value is one variable's share (head (2,4) explained over dd at the neuron featurization), so the leftover rule does not enter it. K is by the swap test. The states are E62's full fine-tunes, seeds 0-2, unscaled.
- **Under the P1.62 protocol.** Fig 2's own copy states (E147 full) and E290 randoms, from `ia3_control/results/large/items_full__*.npz`:
  - The head's kept share reads 0.103, 0.133 and 0.112, so it survives.
  - The head is in K on every state, at 76-92 of 256 pairs (0.30-0.36), against E62's 0.50-0.55.
  - The copy randoms read 0.002-0.009, so "at most 0.005" fails here. On the other four tasks the randoms reach 0.034.
  - Check: the same arrays' pooled E_b_K gives back `fig2_standard/results/top_with_remainder_rows.csv` exactly (copy_s0 0.01168).
- **Fixes.**
  - Use one rounding: main.tex:719 says 0.09-0.14, while sec5:30 and appI:64 say 0.10-0.14.
  - Label the E62 numbers as E62's copy-at-marker. If the paper cites Fig 2's copy states instead, use 0.10-0.13, random at most 0.009, and 76-92 of 256 pairs.

## Not run
- M's top-1 on E147's 96-key copy windows.
- No GPU re-read. The P1.62 recomputation is from saved npz arrays, by the checking agent's CPU script.
