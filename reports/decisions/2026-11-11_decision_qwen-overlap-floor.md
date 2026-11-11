# Decision: the random update's neuron overlap on Qwen moves more between two draws than the filed tolerance allows. How should the second run read the floor? (due 11-13)

**Blocking for the second run only.** Owner: compare-lead. Cell P3.6, due 11-13.

**Question.** On Qwen3-1.7B, two random updates of the same size disagree on the neuron overlap by about twice what we filed as acceptable. May the second run read the floor with more draws and a band measured from their spread, instead of the filed two draws and the filed tolerance?

**Background.** For 19 saved Qwen3-1.7B updates, the cell compares how many of each block's most active MLP neurons (top 480 of 6144) stay the most active after the update (the "top-set overlap"), against a random update of the same size per matrix (the "floor"). The first run (cluster job 59127, $0) checked the floor before reading any real update. It reads no real update's overlap and labels no update reuse or new.

**What held.**
- The unchanged model reads overlap exactly 1.0 against itself.
- The known-answer checks for the singular-value reading passed (equal-rank changes read effective rank 10 and 1; a zero change reads undefined).
- The second overlap reading, the share of the firing change that falls on neurons in both sets, was reproducible across draws at both sizes (differences 0.0004 and 0.0065).

**What missed.** The filed rule said two random draws must agree within 0.0158 x (1 - mean overlap). That tolerance was borrowed from the steadiest real side's seed range on the 67M model. The two draws differ by 7.5e-5 at the smallest update (overlap 0.9978, allowed about 3.6e-5) and by 0.0023 at the largest (overlap 0.905, allowed about 0.0015): about twice the tolerance at both sizes. In plain words: on Qwen the random floor's own draw noise is about as large as a real fine-tune's seed spread was on the 67M, so a single draw cannot stand for the floor at the precision the rule asked.

**Also from the first run (no decision needed).** Every one of the 19 updates has a lower effective rank than its matched random update: 1057 to 1443 against floors of about 1812 to 1825 (the filed call "the updates are concentrated, not noise-like" held, 19 of 19, with both random draws reproducible for all 19). The RL arm reads 1355 and 1368 and its shuffled-reward twin 1374 and 1386; the two maths pools read 1311 and 1314 (one pool) and 1356 and 1356 (the other). This describes the updates; it labels none of them reuse or new.

**Options.**
- **(a) Read the floor with four draws. Recommended.** The second run draws four random updates at each update's own size, takes the floor's band from their measured spread, and files that band in its first stage, before any real update is scored. It tells us whether each real update's overlap sits outside the floor's own draw noise. Cost: the same single cluster job, about an hour, $0.
- **(b) Run the second job as filed, two draws.** The miss is stated as a limit on every real-versus-floor comparison. It is cheaper by a few minutes, but a real-versus-floor gap under about 0.005 could not be read.
- **(c) Close the cell on the first run's effective-rank column only.** No overlap reading on Qwen. Nothing further runs.

**Why it is a question.** The rule for this job is that a filed control that failed is not loosened without the user, and (a) changes what the floor is read from.

**Recommendation: (a).** It costs the same, replaces a borrowed tolerance with a measured one, and the band is filed before any real update is scored.

**Evidence.**
- `experiments/qwen_cheap_readings/results/run1_stage1.json` (floor draws, refs F_s0 and a2_s42)
- `experiments/qwen_cheap_readings/results/run1_stage2.json` (effective rank, floors)
- `experiments/qwen_cheap_readings/results/predictions_run1.json` (filed rules)
- `experiments/qwen_cheap_readings/card_run1.md`
- `reports/plans/2026-11-11_plan_qwen-cheap-readings.md`

**Not run.** The overlap of the 19 real updates; the three contrasts against seed spread.
