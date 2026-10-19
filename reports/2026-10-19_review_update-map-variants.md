# Review: the map variants on hand, and what (a) and (b) would add

2026-10-19. Written before any new design, to compare the map variants on hand. Nothing here was run; expectations
are marked as such.

## Definitions

- $W_0$: the base model's weights, one matrix per module. $\delta W$: the fine-tune's weight change; the fine-tuned
  weights are $W_0 + \delta W$.
- $d_0$: the released decomposition of $W_0$: rank-1 subcomponents $u_c v_c^\top$, of which the alive ones form the
  columns of $U_A$ and $V_A$; $\Gamma$: the causal-importance function that gates them.
- $\sigma(d) = \sum_c u_c v_c^\top$, the sum of a decomposition's subcomponents.
- $\Delta_0 = W_0 - \sigma(d_0)$, the released residual. $\Delta' = W_0 + \delta W - \sigma(d')$, the residual of a
  new decomposition $d'$ of the fine-tuned weights. "$\Delta$ fixed" means $\Delta' = \Delta_0$: the residual does not
  move, but the weights do, by $\sigma(d') - \sigma(d_0)$, a change that lies entirely inside what the subcomponents
  can express. It need not equal the fine-tune's $\delta W$ or its expressible part $\delta W - \mathcal O(\delta W)$.
- $\mathcal O(\delta W)$: the obstruction, the part of $\delta W$ that no first-order change of $(U_A, V_A)$ can
  express (`theory/commuting_updates.md` §5); $\omega = \|\mathcal O(\delta W)\|^2 / \|\delta W\|^2$, its share.
  $P_U$, $P_V$: the orthogonal projectors onto the column spans of $U_A$ and $V_A$.
- $K(G) = U_A U_A^\top G + G V_A V_A^\top$, the weight change one SGD step on $(U_A, V_A)$ makes from a gradient $G$
  on $W$.
- Route shares (`reports/2026-10-16_prediction_ladder-routes-amendment.md`): the trigger-local log-odds gain split
  into direct $d$, re-gating $r$, interaction $i$, masked-off $e_\text{gate}$ and residual $e_\Delta$, summing to 1.
- (a), (b): the two $\Delta$-moving pull-backs sketched in `backlog.md` ("Pull-back variant: let Δ take what the
  pieces cannot"). (a): $\Delta$ trainable on the obstruction subspace only. (b): $\Delta$ takes everything the
  components' step leaves.

## Two axes sort every variant

1. **Who decides the components/$\Delta$ split.** By fiat (do-nothing: all $\Delta$; pull-back: all components), by
   a linear map of the finished or per-step $\delta W$ (lift, (b)), or by the task loss (pull-back's components, (a)).
2. **Do the weights end up equal to the fine-tune's?** Yes for do-nothing, lift, (b). No for pull-back, (a): their
   weights change too, but by their own training on the task, not by the fine-tune's $\delta W$. (The model
   $W_0 + \delta W - \mathcal O(\delta W)$, the fine-tune with its obstruction removed, is a fourth thing: round 2's
   projected arm, a weight edit, not a decomposition.)

| Map | Components | $\Delta'$ | Weights = fine-tune? | Split decided by | Run? | What it showed |
|---|---|---|---|---|---|---|
| do-nothing | unchanged | $\Delta_0 + \delta W$ | yes | fiat | yes (rounds 1–3) | reference; $e_\Delta = 1$ by construction; PGD ratio drifts to 2.2 by step 10 |
| Euclidean lift | min-norm preimage of $\delta W - \mathcal O(\delta W)$ | $\Delta_0 + \mathcal O(\delta W) - \delta U\,\delta V^\top$ | yes | geometry (gauge-dependent) | yes (round 2) | carries 88–93 % of the unmasked gap; PGD ratio 1.14 → 5.4 over steps 1–10; $\|\Delta' - \Delta_0\|^2 / \|\delta W\|^2 = 0.35$; cause of fragility open |
| pull-back (Adam, $\Delta$ and $\Gamma$ fixed) | trained by the task loss | $\Delta_0$ | no | fiat | yes (rounds 2–3, concentration, identity) | learns ($P(o)$ 0.95 by step 3), ⅓ of the fine-tune's off-target KL, Pile ratios ≤ 1.08; the behaviour runs through re-gating ($d \le 0.08$ on all 7 ladder tasks, $r$ 0.28–1.07; `reports/2026-10-19_analysis_task-ladder.md`), update diffuse |
| sparse pull-backs (top-1, top-10, siblings, gated-off, same-matrix) | as above, on a subset | $\Delta_0$ | no | fiat | partly (concentration test; ladder's gated-off control; same-matrix/same-rank LoRA arms not run) | subsets learn as well as all; also through re-gating |
| (b) $\Delta$ takes everything left | per step $-\eta K(G)$ (SGD) or an Adam step on $(U, V)$ from the fine-tune's gradient | $\Delta_0 + \delta W - (\sigma(d') - \sigma(d_0))$ | yes | per-step linear map (gauge-dependent) | no | — |
| (a) $\Delta$ trainable on the obstruction only | trained by the task loss | $\Delta_0 + (I - P_U)\, D\, (I - P_V)$, $D$ trained | no | task loss | no | — |

Same family, same answer: do-nothing, lift and (b) are "fix the weights to the fine-tune, split $\delta W$ by a
$\Gamma$-blind linear rule"; pull-back and (a) are "train inside the decomposition, let the weights land where they
land". Within a family the variants differ in the rule, not in the kind of evidence they give. In the weights-equal
family the components can carry at most the expressible part, $1 - \omega$, to first order, whatever preimage is used:
"update the subcomponents as much as possible" has a ceiling that round 2 already measured ($\omega \approx 0.3$
input-weighted). Lift and (b) differ only in *which* components take that part, and neither chooses within the kernel
by anything task-related (theory §5.1's $Q$ functional is where such a choice would live).

## (b): what it is and why it adds little

- **It is a per-step lift with a different preimage.** For SGD, components get $-\eta K(G_t)$ and $\Delta$ gets
  $-\eta\,(G_t - K(G_t))$ at every step, so the trajectory is the fine-tune's. The end state is the lift's family
  (weights = fine-tune) with $K$'s preimage in place of the min-norm one. $K$ is not a projection: with the released
  (non-orthonormal) $U_A$, $V_A$ it rescales expressible directions, and the doubly-expressible block $P_U G P_V$ is
  counted twice. So $\Delta' - \Delta_0$ holds the obstruction *plus* expressible directions with whatever sign makes
  the sum come out, i.e. expressible-but-reweighted mass (the backlog's point). Its $\Delta$ share is gauge-dependent:
  rescaling $u_c \to s u_c$, $v_c \to v_c / s$ changes $K$ and the split, with the same decomposition. The lift's
  min-norm preimage has the same defect (filed as "gauge-dependent: released gauge"); (a)'s obstruction subspace does
  not (spans are gauge-invariant).
- **Adam.** The fine-tune is Adam. "$\Delta$ takes what is left" is then not derived from anything: components take an
  Adam step on $(U, V)$ from the fine-tune's gradient at the fine-tune's weights, and $\Delta$ is defined as the
  difference. Defined and cheap (replay the 10-step fine-tune with `pullback.Offsets` receiving its gradients), but a
  definition. The SGD version is practically blocked: the concentration test found no SGD lr for the pull-back within
  the off-target limit.
- **Expected numbers (emoticon task, step 10).** Behaviour and off-target KL: the fine-tune's, by construction, so
  uninformative. Pile stochastic ratio ≈ 1.0 (blind). PGD ratio: between do-nothing's 2.2 and the lift's 5.4, because
  the component part is $\Gamma$-blind like the lift's; nearer the lift if the fragility comes from the component
  change, nearer do-nothing if it comes from $\Delta$. Route split: $e_\Delta$ is the *marginal* residual share,
  $(T_1 - U_1) - (T_0 - U_0)$ with $U$ the components-only unmasked model, so it reads ≈ 0 if the component part alone
  reproduces the behaviour, which the expressible part did in the obstruction review
  (`2026-10-15_review_obstruction-claim.md`: the behaviour is stored redundantly, $\mathcal O(\delta W)$ alone gives
  $P(o)$ 0.4–0.9 too). Meanwhile the Frobenius share $\|\Delta' - \Delta_0\|^2 / \|\delta W\|^2$ sits near the lift's
  0.35. The two will disagree for the whole weights-equal family, and that disagreement is itself why (b) is hard to
  read: a large weight share in $\Delta$ with a small functional share says nothing about whether $\Delta$ was needed.
  Components' direct share $d$ still small.
- **Verdict.** Under `GOALS.md` the weak version says $\Delta$ may not absorb $\delta W$; (b) lets it, by
  construction, by an amount set by the gauge and the optimiser rather than by the task. It answers "how much of the
  fine-tune does *this preimage rule* leave in $\Delta$", which the lift already answers with a cleaner rule (0.35).
  Not worth a run. If a number is wanted, compute it as a reported quantity on the existing fine-tune, not as an arm.

## (a): what it is and what it would add

- **It is the pull-back with one extra, principled, parameter block.** The components are trained by the loss as now;
  $\Delta$ gets a trainable term confined to the obstruction subspace of the released alive columns (fixed at $W_0$;
  moving it with the components is a second-order change). Nothing expressible by the components can go into
  $\Delta$, so $\Delta$ cannot take the cheaper expressible route; it can only supply capacity no subcomponent has.
- **The question it answers is different from the pull-back's.** The pull-back asks "can the components carry it when
  forced to?" (answered: yes, for every rung run so far). (a) asks "given a choice, does training *prefer* the
  components, and how much of the behaviour needs capacity outside them?" On a task the $\Delta$-fixed pull-back
  learns, the answer measures Adam's appetite, not the task's need: the free fine-tune put ~30 % (input-weighted) of
  $\delta W$ into the obstruction on a task that did not need it. On a rung where the $\Delta$-fixed pull-back fails,
  $e_\Delta$ for (a) is a direct reading of "new capability" vs "brittle": a large $e_\Delta$ with the components still
  faithful says the model needed capacity the decomposition does not have; a small $e_\Delta$ with (a) also failing
  says the obstacle is not capacity.
- **Expected numbers (emoticon task).** Learns at least as fast as the pull-back ($P(o) \ge 0.95$ by step 3).
  Off-target KL between the pull-back's 0.017 and the fine-tune's 0.055. The Frobenius share
  $\|\Delta' - \Delta_0\|^2 / \|\delta W_a\|^2$ of order the lift's 0.35 at step 10 (the free fine-tune's obstruction
  is ~30 % input-weighted, 13 % Frobenius; the Frobenius measure is uninformative on $W_0$, so report the
  input-weighted one beside it) and rising with steps (the accumulation the backlog warns of; this number along
  training is the check). Pile ratios ≈ the pull-back's (≤ 1.1, stochastic and PGD): the $\Delta$ term is trained on
  trigger windows and Pile rows hold ~3 trigger positions in 131k, so Pile metrics are blind to it (the 2026-10-15
  scope review applies to $\Delta$ as much as to the components). The discriminating numbers are trigger-local:
  $e_\Delta$ in 0.1–0.5 at step 10, $d$ no larger than the pull-back's (≈ 0.03), re-gating $r$ still dominant. If
  $e_\Delta$ comes out near 0 with the obstruction term free, that is the interesting outcome: Adam does not use
  $\Delta$ when the components suffice.
- **Risks.** (i) The obstruction block is large, $(d_\text{out} - r_U) \times (d_\text{in} - r_V)$ per matrix, so it
  is a lot of free capacity; its own Adam state keeps the step sizes comparable but not the capacity. (ii) It makes
  every pull-back number depend on one more lr. (iii) It is one more arm in the route split's five-way table;
  `GOALS.md` already assigns "create" to $e_\Delta$ dominating *when $\Delta$ may move*, so (a) is the arm that
  hypothesis was written for, and without it that hypothesis has no arm.

## What this does not change

Every pull-back so far carries the behaviour by re-gating, not directly (ladder, 7 tasks: $d \le 0.08$, $r$
0.28–1.07, with the masked-off share dominant at rung 2; the gated-off control shows the same shape;
`reports/2026-10-19_analysis_task-ladder.md`). Moving the weights' split between components and $\Delta$, which is all (a) and (b) do, does
not touch that finding. "The components carry $\delta W$" (weights) and "the components carry the behaviour at the
old gates" (route $d$) have come apart, and the second is the one the goal's strengthen / reveal / create reading uses.

## Recommendation

- Keep one representative per family: the lift (weights fixed, linear split) and the pull-back (loss-driven, $\Delta$
  fixed). Both are run. Drop (b) as an arm; it is the lift with a worse preimage.
- Add (a) only as the backlog already says: as a reported arm at the first rung where the $\Delta$-fixed pull-back
  fails to learn, with $\|\Delta' - \Delta_0\|^2 / \|\delta W\|^2$ along training and $e_\Delta$ as its two numbers. No
  rung has failed that way: every ladder pull-back learned its task (held-out $P \ge 0.986$, rungs 0–2); rungs 3–4
  are not run. What failed at rung 2 is the CI-masked model keeping the behaviour (masked-off share 0.34–0.66), which
  is a placement question, not a capacity one, and (a) does not bear on it.
- The ladder run is finished and analysed (`reports/2026-10-19_analysis_task-ladder.md`, 2026-10-19). Its hand-off is
  the PGD check, then a `/design` choice between $\Gamma$ updating and constrained placement; the $\Delta$ variants here
  sit behind both, since the direct-vs-re-gating finding is what any of them would be read against. The six ladder
  tasks' trained states were never saved, so any offline number on them would need a rerun.

## Not run

(a), (b), (b) as an offline number on the existing fine-tune, LoRA through the pull-back (same-matrix, same-rank),
any $\Gamma$ update, rungs 3–4 of the ladder, the lift's fragility cause.
