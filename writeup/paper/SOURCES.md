# Paper skeleton: sources

`main.tex` is the skeleton: headings, [bracketed suggestions], planned figures. Numbers in it are provisional
(no independent check) and come from the summaries below, each of which cites its report and figure paths.

- `sources/part1_definition.md`: definitions of reuse tried, what each read (cells P1.1-P1.25).
- `sources/part2_67m_comparison.md`: the 67M scorecard, what the share measures (P2.1-P2.30).
- `sources/part3_qwen_and_carrying.md`: Qwen, RL vs SFT, and the earlier carrying line (appendix A).

Summaries written 2026-11-17 by read-only agents; check a number against its cited report before using it.
Twin numbers: cite the 2026-11-16 gate page, not the log's claims page (which still says 1 of 4).

## Framing advice (Fable, 2026-11-17), not yet decided by the user
- Write section 4 as the plan's fallback ("why existing definitions fail") unless E196 separates all five pairs.
- Head section 7 "Limits at 1.7B"; RL vs SFT one line in main text, details in an appendix.
- State the re-weighting hypothesis as motivation, untested: the paper tests instruments.

## Experiments a reviewer would most ask for (Fable's ranking)
1. Pairs where the prior and steering disagree (last attempt stopped at a prior of 2.32e-4 vs 2.35e-4 needed).
2. A crosscoder / SAE-diffing baseline on the same pairs.
3. The share on the hand-made edits (E137-E139) as a positive control of known kind.
4. A cause for Qwen saturation (MLP-only reading on the new decomposition; stable random band).
5. A second decomposition seed or second small model.
6. Third seed, independent checks of headline numbers, significance tests.
7. Size matching of cheap readings by retraining, not rescaling.
