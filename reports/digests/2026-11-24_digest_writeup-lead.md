# writeup-lead digest, 2026-11-24

No cell of mine (writing only); draft due 11-25, final 11-29. Results is in your 22:00 order and ready to accept on
branch v2, pushed through 62494d3; five later commits are local only, because the push was refused by the permission
check and needs your explicit OK.

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
- **Push of eight local commits** (paper v2): c54e296 old easier tasks in §5.3 and the appendix (difficulty explains only part of the realistic failures), 7f57461 diff-SAE row, appendix paragraph and a scope sentence in §5.4 (three tools not shown to work at this scale), a37ca46 one Limitations sentence for E297 (no trained reuse control could be fitted), 8791a68 copy-head random null (touches `main.tex`), b4d702b redteam
  recheck, 7c71afb and 286040f WeightWatch row and appendix paragraph, 3b7cf54 Delta-Crosscoder run 1 row. Say "push"
  and I push.
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
