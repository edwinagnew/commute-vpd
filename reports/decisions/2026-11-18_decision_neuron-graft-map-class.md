# Decided (tentatively): a lookup built on fresh neurons reading as mostly kept is a stated consequence of the linear re-mix rule

*2026-11-18, definition-lead. Record of a definition question and the user's answer in chat (2026-11-17, about 23:55).
`decisions.md` and `GOALS.md` are unchanged.*

**The question.** On 2026-11-18 the user adopted the held-model definition and stipulated that "a linear re-mix of M's
units counts as reuse". Under it, 256 fresh MLP neurons trained to do a random lookup on the current token, which the
original model M cannot do, read as mostly kept. Should the main reading's map be narrower, so that this reads lower?

**What was seen** (`reports/analyses/2026-11-18_analysis_graft-sites.md`,
`reports/analyses/2026-11-18_analysis_held-model-size-matched.md`).
- Old-task fine-tunes keep 0.98 or more of their change under the main reading (one linear map per site from M's units).
- Four new attention heads, new by construction, keep 0.14 to 0.22.
- The 256 new neurons keep 0.954, just under the lowest old-task fine-tune (0.984). The map from M's 3,072 last-layer
  neurons fits almost all of what they write. Our hypothesis, not measured, is that M's neurons carry the current
  token linearly, so any lookup on it is a linear re-mix.
- Size moves the score too: shrinking an update along its own direction changes survival by up to 0.030, more than the
  pair gaps read.
- The filed rules held; the near-tie was not expected. Both premise checks rated the readings doubtful.

**The answer.** The user: "im not sure ... but intuitively yes, linear mixing is probably more reuse than new."
Taken as a tentative acceptance of the option that keeps the definition as it is: a lookup on the current token, built
on new neurons, reading as mostly kept is a stated consequence of the linear re-mix rule, not a finding against it.
Recorded as one caveat in the definition's box in the log. No run.

**Not chosen, still open.** A narrower map for the main reading (a few of M's units per unit of F, or low rank), which
would be a new cell and a definition change; and a built control whose label does not depend on linear readability
(the ladder's open rung). Either can be raised again.

**Links.** `reports/analyses/2026-11-18_analysis_graft-sites.md`,
`reports/analyses/2026-11-18_analysis_held-model-size-matched.md`, `experiments/graft_sites/results/summary.json`,
premise checks `~/commute-data/premise/P1.28.md` and `~/commute-data/premise/P1.27.md`.
