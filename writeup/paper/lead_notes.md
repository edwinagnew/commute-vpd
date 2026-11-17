# Writeup-lead notes for main.tex (the user edits main.tex; the lead never does)

New results to fold in when you choose. Each says where it would go.

## 2026-11-17
- §4 (v), held-model retention (E197, E201): refit only the base model's output weights per site. Built new head S=0.13,
  rebuilt circuit 0.56, every trained fine-tune ≥0.99 except the fact side (0.65). Reuse-minus-new gaps <0.02 on four
  pairs, 0.25 on twin/fact. Separates built mechanisms; barely separates trained ones.
- §6, disagreeing pair (E199): on the one built pair where prior and steering disagree, the share follows steering
  (share gap 0.18, prior gap −0.8 decades). Not just the prior; nothing shown beyond steering. One pair.
- §5 scorecard: crosscoder row (E198) and the share on hand-made edits (E200) not in yet.

## 2026-11-17 (late), after the user's "update draft" push
Checked: the push kept every lead line (E197/E201 at §4, E199 at §6, E198/E200 open markers at §5). Nothing to restore.

Small things in the new §5.1 text, for you to take or leave:
- The share is written $r := (M_{01}-M_{00})/G$ but the scores are defined as $S_{00}, S_{01}$; probably $(S_{01}-S_{00})/G$.
- "Clearly, $0 \leq r \leq 1$": not guaranteed. $S_{01}$ can sit below $S_{00}$ or above $S_1$ (the masked base model
  with the fine-tune's gates can beat or miss the gain), so $r$ can leave $[0,1]$. Suggest "typically lies in [0,1]".
- "with VPD subcomponents $C_0$" fine; the next sentence's "masked according to $g_0$" could say "the base model's
  subcomponents $C_0$, masked by ..." so it is clear $C_1$ is never needed (that is the point of the line after).
- "does not require updating the components of re-training" → "or re-training".
- "by As shown in fig ??" → sentence was cut mid-way.
- The baselines list you removed from §5.2 (neuron version of the share; steering, i.e. mean activation shift; update
  size; effective rank; neurons for half the gain; Goodfire-style neuron job retention; base-model prior on the
  answer; untrained-SVD share as an ablation) could be the body of App. "Baseline implementations", which now reads
  "List of baselines".

§4, definition adopted (user, 2026-11-18, decisions.md row "Working definition of reuse ... held-model wording"):
- Line 78's open marker ("which wording is adopted") is now answered: the held-model wording of causal-rule retention,
  framed as a cross-model interchange intervention (Geiger et al.; Prakash et al. 2024). Every unit of F is swapped for
  the base model's matching unit held to its rule; reuse is the share of the change that survives. Main reading: a
  fitted linear map per site; one gain per unit is the strict end. A linear re-mix counting as reuse is a stipulation.
- Numbers (reports/analyses/2026-11-18_analysis_held-model-retention-run1.md): one gain per unit keeps 0.01-0.06 of
  any change; the per-site map keeps ≥0.98 on all eight old-task fine-tunes, 0.56-0.999 on trained or rebuilt items,
  0.13 on a head grafted onto the frozen model (new by construction, one build, one site). Matched pairs differ by
  under 0.02, so no claim on them. No cut-offs, no labels.
- Suggest §4 becomes: requirements → the failed wordings as "what a definition must do" → the adopted wording with
  the graft as its one built case. The (v) held-model \sugg line at 94 would move up into the adopted subsection.
