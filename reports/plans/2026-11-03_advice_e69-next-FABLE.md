# Advice: what runs next on E69's shuffled-reward null (Fable, 2026-11-03)

*For `qwen-rl-lead`. Read: the E69 analysis, the E69 and reorient plans, `experiments/qwen_rl/{run,job}.py`,
`src/commute/qwen_rl.py`, E75's `score_N.json`, the decomp lead's handoff folder.*

**Recommendation.** Run a second seed of both arms, RL and null, now, on one H100 (option d plus the RL seed), as
Amendment 1 of E69. Alongside, for $0: greedy accuracy at the saved steps 10, 20 and 50 of both arms (option b,
widened), then forced openings (a). Defer (c): it bears on SFT's small KL, not on RL against the null, and would spend
the second amendment. Nothing blocks the decomp lead from mapping: E75 has already read N10 and N100, and they take the
same labels as R10, R100 and the SFT updates on every row (deaf, holds, holds, beats size). What is missing is the
label on each state, not the map.

## Order, and what each result changes for the decomp lead

1. **Second seed, R2 then N2, one H100, about 4.9 h, about $20.** From the run's own seconds: R 1.9 h, N 2.45 h, each
   with 7 readouts and a 13-min accuracy pass, set-up 0.5 h. Under the gate; $44 of $64 would remain. No F2.
   *Reading.* N2 gains about 5 points again: the gain is a property of the loop, the decomp lead reads the null δW as a
   real behaviour change and RL's signal as the difference, with a two-seed spread. N2 inside P3's band: N1 was one walk,
   RL's lead is its whole gain, and N2's δW is read beside N1's. R2 is the other half of the spread; without it no
   seed-to-seed spread of the lead can be stated.
   *Code change needed (no seed flag exists).* `Config.seed` is 69 and feeds three things: `split` (H and the prompt
   order), `step_seed` (every arm's draws) and `null_weights`'s shuffle; the folder carries only the rate tag, so a
   second seed would overwrite `lr1e-6w10/`. Add `draw_seed` to `Config`, used by `step_seed` and `null_weights` only,
   a `--seed` flag, and folder and W&B prefix `lr1e-6w10_s<draw_seed>` when it differs. Keeping `split` on `seed` keeps
   H, the prompt order and A0 (fetch `rollouts/H.jsonl` and `acc/W0_*.jsonl` to the pod as `RUN_BRIEF.md`'s "Pod lost"),
   so ΔR and KL stay on the same samples and the test gains stay paired. About 15 lines and a test row. The step-1
   identity check compares N with R's `rollout_sha_1` in the same folder, so R2 runs first, same card.
   *Not the L40.* Rows come from the card's memory (301 against 512), so an L40 N2 would not match an H100 R2 at step
   1; the probes ran out of memory at step 11 there with 41.7 GiB held, so `reserved_gb` would go to about 36 at 1.4×
   slower sampling, about 6 h under `normal` QoS; and the queue holds anchor 1, E75's reads and a masked job of ours.
2. **Saved-step accuracy, $0, one `debug` job per two states (about 25 min each on an L40), or 13 min a state on the pod
   after N2.** R and N at steps 10, 20, 50 from the δW files, both sets. The decomp lead has read R10 and N10, and at
   step 10 RL's format-ignored sampled gain is 0.1 with its reward gain all answer line: those two states differ in
   format, not maths, and a gain at each read state says so. Script: `Job` plus `qwen_sft.load`, about 40 lines.
3. **Forced openings, $0, one `debug` job, about 1 h.** `qwen_eval.generate` has no prefill: add one argument that
   appends the opening to `prompt()`'s text and grades prefill plus continuation. W0 prefilled "We are" and "We need";
   N100 prefilled with W0's own first two words per problem (`acc/W0_gsm8k.jsonl`). If the opening carries the gain,
   the decomp lead can ask where a first-token style change lands; if not, the null's δW is a diffuse change.

## Filed rules (one `predictions_amend1.json`; P1-P7 untouched)

- **Seed job.** Early checks as filed; identity check N2 against R2. *S0, control:* R2 learns by P1's rule (≥ 2
  points and ≥ 2 SE, GSM8K test, drop rule, strict); else S1 and S2 are reported as read, not scored. *S1, the lead
  replicates:* R2 − N2 paired over problems kept at both ≥ 2 points and ≥ 2 SE → replicates; < 1 → no lead; between →
  ambiguous. *S2, the null's gain is systematic:* |N2's gain| ≥ 1.5 (P3's band) with N1's sign → systematic; < 1.5 →
  one-seed walk. P1's and P3's bands are reused so seed 2 is scored as seed 1 was. The two-seed spread of the lead,
  and the whole-answer and capped columns, are reported, unfiled.
- **Saved steps.** *Control, first stage:* W0 rescored on this card against A0's 78.17 and 78.89; the gap is the card
  floor f. N100 rebuilt from its δW must match `N.json` within f (exactly if f is 0), else stop. *B1:* N's whole-answer
  strict gain at step 50, GSM8K test, ≥ 2.4 (half its 4.85) → the gain arrives with the "We" flip (83% by steps 21-50);
  < 1.0 → after step 50; between → no call. Expected at R10: lenient gain within 1 point of 0.
- **Openings.** *Control:* W0 prefilled with its own first two words reproduces its greedy answers token for token;
  else a code error, stop. *A1:* W0 prefilled "We are", whole-answer strict gain over W0 ≥ 2.4 → the opening carries
  the null's gain; ≤ 0 → it does not; between → part. Expected ≤ 0 (the sampled test read −2.5). *A2:* N100 prefilled
  with W0's openings loses ≥ half its 4.85 → the opening carries it.

## Timeline, and what would change this

The reorient plan put the second seed on 11-08; running it 11-04 moves nothing else, since anchor 1 is on the cluster
at $0 and anchor 2 waits for the user. A lost pod costs the arm in flight, at most $10. I would change the order if the
independent check finds the headline lead a grading or pairing artefact (fix first, no seeds); if the user wants the
budget kept for anchor 2 (then N2 alone on the L40 with the memory change, R2 on 11-08); or if (b) reads the null's
gain already ≥ 2.4 at step 20 (then (a) is the more telling question, though the pod is still cheapest started now).
