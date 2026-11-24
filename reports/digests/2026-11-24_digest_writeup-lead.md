# writeup-lead digest, 2026-11-24

No cell of mine (writing only); draft due 11-25, final 11-29. Results is in your 22:00 order and ready to accept on
branch v2, pushed through 7766b93 (all overnight results; your push OK at 08:10).

## Cells
None owned. Writing follows the task file.

## Ready to accept (pushed on v2)
- **Results in your order.** 5.1 the definition around one figure (kept share at VPD, top neurons and SAE; built,
  trained and random points; copy head as panel b), gain-only control in one sentence, one sentence reconciling the
  0.999 of §3. 5.2 known answers: copy-heads-off label check first, then four panels (re-gating, steering, effective
  rank, prior) plus prior by mechanism; the other readings in an appendix figure. 5.3 realistic: the mirror figure of
  the same four readings against the raters' tier. 5.4 the tools table, blank filled.
- **Redteam number checks applied** (gain edits, SAE trained points, effective rank partialled on base loss, raters
  named as language models, random-head removal).
- **Discussion** cut per blind read; **Conclusion** three bullets answering the intro's questions (amount of
  post-training and RL against SFT left open); **Related work** citation fixes; **Appendix** 28 to 10 pages, cut text
  kept verbatim in `paper/cut/appendix_cut_2026-11-24.tex`, including your own paragraphs that moved there.

## Waiting on you
- **Pushed 08:10 on your OK:** nine overnight commits 62494d3..7766b93 (old easier tasks and difficulty figure,
  diff-SAE, E297 Limitations sentence, copy-head random null, redteam recheck, WeightWatch, Delta-Crosscoder run 1).
- **Flags in your sections (not edited):** "write share" in §4 and the abstract against "kept share" elsewhere; §4
  overclaims Qwen at lines 7-8; `main.tex:677` "Section" should be "Appendix"; §3 at `main.tex:260` states the neurons'
  0.999 without saying it is over K*; typos in §4 and the intro; leftover notes ("[cite - write share?]", "werent there
  any others???", "[cite]", "[one more point?]"); §4 still names a cut score. Abstract and intro redteam list:
  `~/commute-data/agents/writeup-lead/notes-read/note-2026-11-24-redteam-abstract-intro.md` (five must-fix items).
- **One reading for you:** the copy fine-tune reads as reused on the cheap readings (re-gating 1.09) while the
  definition reads its copy head as mostly changed (kept share 0.09 to 0.14). Which the paper leads with is your call.

## Still open in the text
Placeholders wait on P2.72 run 2 (your size choice via whip-lead);
the K* note, the RL line and the baselines claim stay open by your instruction.

## Money
$0.

Days left to the final draft (11-29): 5. Still skeleton: nothing structural; the abstract is yours.
