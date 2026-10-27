# Decision card draft: a decomposition trained on a broad fine-tune stays twice as fragile (E37 Amendment 1)

*2026-10-27, analysis agent for the lead. Draft of the pending-decision card that E37 Amendment 1's filed decision tree
calls for in its last branch: "neither → card to the user with option (d) and the anchor question for broad updates"
(`reports/plans/2026-10-26_plan_teacher-spread.md` → "Amendment 1" → "Decision tree"). Option (d) is the earlier card
`decision-broad-copy`'s: test whether the frozen Γ limits the trained decomposition, by E15's refit, or whether the
subcomponents' span does. The anchor question is card `decision-pass-anchor`'s: judge the adversarial check against a
fixed line or against the same update left in Δ. Evidence: `reports/analyses/2026-10-26_analysis_teacher-spread.md` →
"Amendment 1". Format per `docs/research_log/README.md` → "Pending decisions". The lead places it in
`docs/research_log/parts/sections/decisions.html` at the next flush; this file does not touch `parts/`.*

**The question in one sentence.** Teacher training of a broad fine-tune stays about twice as fragile as the original
decomposition under the paper's check, however long it trains and wherever it starts: should we first test whether the
frozen Γ is the limit, change how a broad update's check is judged, or take teacher training to the larger model as it
is?

```html
  <div class="card key decision" id="decision-broad-fragile"><span class="chip refuted urgency">Blocking</span><div class="k">Commute line · new 2026-10-27</div><p class="small">Nothing runs on the commute line until this is decided.</p><h3>Training on a broad fine-tune leaves the decomposition twice as fragile, however long it trains: what next?</h3>
  <p><strong>What it is about.</strong> Teacher training retrains the subcomponents so the decomposition imitates a fine-tuned model, with Γ and Δ frozen. It is the commute line's working map for the larger model. On fine-tunes that act at one trigger token it gave decompositions that keep the behaviour and pass the paper's adversarial check. E37 tried a fine-tune that changes every position: Pile text lowercased. The decomposition after teacher training kept all of its behaviour, but the adversary did about twice the damage it does on the original decomposition, above the fail line. The user asked for longer training and other fixes in one run (E37 Amendment 1).</p>
  <p><strong>What the evidence says.</strong> Provisional, one run, not yet independently checked. Training four times longer left the check where it was, 2.20 times the original at step 1,000 and 2.23 at step 4,000. Meanwhile the decomposition kept getting closer to the fine-tuned model. Starting from the adjoint split, which passes at 1.05, the check climbed to 1.84 within 2,000 steps, level with the longer run. In both runs the check rose once the decomposition changed ordinary text as much as the fine-tuned model does; whether that is the cause is untested. The third fix, adding VPD's own masked losses to the training, could not be tested. Its control, the same recipe with nothing to copy, read 3.84 at the original model, and 3.14 with the losses weighted ten times less. That is more fragile than any decomposition trained on the fine-tuned model. Its own losses rose during training. E15's attempt to retrain Γ against the same losses at the original model failed the same way, twice. Judged against the fine-tuned model left in Δ, which reads 4.87, every trained decomposition here reads under half of it.</p>
  <p><strong>The options.</strong> (a) <em>Test whether the frozen Γ is the limit</em>, the first half of the earlier card's option (d): retrain Γ on the decomposition after teacher training, E15's refit, about $9 per state. Training against VPD's losses at the original model has failed three times, so first a control of about $1: the same recipe with the masked losses switched off. It says whether our trainer or the losses move the check, and whether a refit can be read at all. (b) <em>Test whether the subcomponents' span is the limit</em>, the second half of option (d): whether the alive subcomponents can express this fine-tune's change on ordinary text at all. No design exists yet; it goes through <code>/design</code> first. (c) <em>Settle the anchor question for broad updates</em>, card <code>decision-pass-anchor</code>: count a state as passing when it is no worse than the same update left in Δ. Every decomposition here then passes, at 0.36 to 0.46 of it, but that reference is inflated by construction, since the adversary can switch off Δ. (d) <em>Take teacher training to the larger model as it is</em>, reporting the check against both references. Another reading of about twice there would say this is what broad updates do, not a quirk of a toy fine-tune.</p>
  <p><strong>Why it matters.</strong> Real post-training is a broad update like this one. If teacher training is faithful only on trigger-like updates, the larger-model run needs a different map, a repaired Γ, or a pass rule that fits broad updates. Options (a) and (b) ask why the decomposition is fragile; (c) and (d) decide what to do while that is open.</p>
  <div class="links"><a href="#e37">E37</a><a href="#e34">E34</a><a href="#e15">E15</a><a href="#decision-pass-anchor">Anchor card</a></div>
  <p class="small">Plan <code>reports/plans/2026-10-26_plan_teacher-spread.md</code> → Amendment 1; analysis <code>reports/analyses/2026-10-26_analysis_teacher-spread.md</code> → Amendment 1; results <code>experiments/teacher_spread/results/amend1.json</code>; this draft <code>reports/decisions/2026-10-27_decision_e37-amend1-card-draft.md</code></p></div>
```

**Notes for the lead.**

- The card's id and title are new. Card `decision-broad-copy` closed on 2026-10-27 with option (a), and this card
  replaces it for the next step. Card `decision-pass-anchor` stays as it is, marked No rush; option (c) here points to
  it. Consider re-rating it to Soon, since this card now waits partly on it.
- "Running" line: nothing runs on the commute line until this is decided. The masked control at weight zero in option
  (a) is cheap enough to run first under class 4 if the lead reads it as the tree's next step. It is not filed anywhere
  yet.
- Check the links: `#e15` and `#e34` must exist as card anchors on the experiments page.
- The main-page length cap holds: if this card is added, fold or shorten something of equal length elsewhere.
