# Fable's advice on the E46 stage 0 decision card (2026-10-29)

Advice, not a decision. Requested by the regating-lead after Amendment 2 (Slurm 42791). Earlier advice (recommendation B, raise the cap) is recorded in the plan's Amendment 2.


## Pending decision: is the gate gain g good enough to take to the Qwen line?

**What the result means.** A random update scaled to the fine-tune's side effects (11-26 times its size) is not inert: it moves the full model's target log-odds by 0.01-0.4 nats, and Γ passes almost exactly that through the old subcomponents (g ≈ G on five of seven rows; r1 and r2 differ in sign, within noise). Γ is not a detector that stays silent under noise; it reports whatever a change does to the target. A learned update is told from a random one only because the random one does so little: g separates them 8-100 times (0.04-0.49 nats against 3.5-11.6). That is enough for stage 1; the absolute floor (0.1 nats) was the wrong null, not the instrument. The weak point is in the already-passed P0.4: the hand write, matched on G, takes 48-65 % of a random-map fine-tune's g while using no old machinery by construction. A G-matched null is what stage 1's claim ("distilled ≥ 0.7, invented ≤ 0.4") must beat, and on the 67M it sits near 0.5. Also, stage 0 scored g, but stage 1's first number is gated piece gain / piece gain after a least-squares split; the random and write nulls were never split, so that ratio has no null yet.

**Options.**
1. Adopt g; file stage 1 with the random null as a relative bar (random g ≤ 0.15 of the arm's g, the 67M's worst case) and the G-matched write as the second null. ~$5 on the Qwen states. Risk: on Qwen the real SFT's g/G is low (rand32: 0.27) and the nulls crowd it.
2. Stage 1 with every null matched on G. The write already is; a random direction cannot reach 10-20 nats of real gain without wrecking the model (kl_other 0.7 bought 0.4 nats), so this collapses into option 1.
3. One more $0 stage-0 job first: split the random δW and the write with least squares, as the fine-tunes were, and read stage 1's own number (gated piece gain / piece gain) on them, with g/G for every state on one axis. One debug job, half a day. Rules out a null near 1 for stage 1's ratio (every arm would then read "distilled") before any pod bills.
4. Retire. Not supported: P0.2 replicated and the separation is 8-100 times.

**Recommendation: 3, then 1.** Stage 1 should spend money on a number whose null has been read once. Split nulls' ratio ≤ 0.4: option 1 unchanged. Ratio ≥ 0.7: the ratio does not separate stage 1's classes and the claim needs a different number before a pod starts. What would change this: a Qwen δW already saved and idle, where $5 is cheaper than the half day.

Friction: P0.3's "inconclusive" and P0.4's "thin pass" were judged under different nulls (KL-matched vs G-matched); say so once, so the user is not asked to weigh an inconclusive that is mainly a mis-set floor while the real question is the write.

Source numbers (Fable): write g / fine-tune g = ident32 5.17/11.2, ident32b 6.71/11.6, rand32 1.80/3.75, rand32b
3.92/6.0; fine-tune g/G = emoticon 0.60, ident32 0.80, ident32b 0.79, r1 0.75, r2 0.28, rand32 0.27, rand32b 0.29;
random rows have piece gain 0 (in Δ, unsplit), so stage 1's ratio is undefined on them as run.
