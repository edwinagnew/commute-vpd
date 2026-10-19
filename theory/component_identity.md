# What "updating a component" means: keeping track of component identity

*2026-10-14, coding agent, jotted at the user's request at the end of the day. A note, not a proposal: nothing here
has been through `/design`. Notation follows `commuting_updates.md` (component $c$ = $u_c v_c^\top$, Γ gives
$g_c(x)$).*

## The problem

A lift writes δW as a change to existing components: $u_c v_c^\top \to u'_c v'^\top_c$. Calling the result "the
same component, updated" is a claim, not bookkeeping. It splits into separate claims, and each one can fail on its
own:

| # | Claim | What could break it | Candidate measurement |
|---|---|---|---|
| 1 | **Same contexts.** It is active on the same inputs. | $v_c$ rotates; upstream weights move, so $a(x;W)$ moves even when Γ is frozen. | Overlap of the firing set $\{x: g_c(x)>\tau\}$ before and after (Jaccard), on held-out data *and* on the fine-tuning data. |
| 2 | **Same causal importance.** Where it is active, it matters the same amount. | Γ's $g_c$ is a *prediction* of importance. The real effect can move while $g_c$ stays put, especially with Γ frozen. | Ablation effect (KL when $c$ is masked) before and after, on the firing set; compare with $g_c$. |
| 3 | **Same meaning.** It does the same thing when it fires. | $u_c$ rotates, so the trigger is the same but the action is new. | Angle of $u_c$; next-token or downstream effect of writing $u_c$; top-activating examples; label stability. |
| 4 | **Same interactions.** Downstream components respond to it as before. | Other components move too, so $c$ keeps its own role but its edges change. | Which downstream $g_{c'}$ depend on $c$ (attribution or ablation edges), before vs after. |

Claims 1 and 2 are about the **read side** ($v_c$, Γ). Claims 3 and 4 are about the **write side** ($u_c$) and the
rest of the network. "Same component" needs all four. "Strengthened existing capability" (the reveal hypothesis) is
roughly: 1, 3 and 4 hold, and only the scale in 2 changes.

## A coarse taxonomy of outcomes

- **Rescaled:** only $\|u_c\|\|v_c\|$ changes. All four claims hold apart from magnitude. This is the cleanest
  "strengthen".
- **New action, same trigger:** 1 holds, 3 fails.
- **New trigger, same action:** 3 holds, 1 fails.
- **Effectively a new component:** 1 and 3 both fail. Reusing the old index is then misleading. Count it as created,
  not updated.
- **Relational drift:** 1–3 hold and 4 fails. The component is fine and its neighbourhood has moved.

## Rules for rigour (proposed)

1. **Keep a per-component ledger for every touched component.** Record $\Delta u$, $\Delta v$, scale change, both
   angles, and the four measurements above, before and after. A lift without a ledger is not a result.
2. **Remove gauge first.** Rank-1 scale ($u\cdot s,\ v/s$) and sign, and permutation when Γ is refit (§2 of
   `commuting_updates.md`). Report angles and the product norm, not raw $\|\Delta u\|$.
3. **Fix the reference inputs in advance.** Use held-out text plus the fine-tuning data, kept separate. A component
   can keep its contexts on one and lose them on the other.
4. **Compare against a null.** Run the same measurements on (a) untouched components and (b) a δW = 0 warm-start
   refit. That shows how much "change" is noise. Thresholds go in `predictions.json` before the run.
5. **Count, don't assert.** Report the fraction of δW (by Frobenius norm and by behaviour) carried by each outcome
   class above, not just "the lift worked".
6. **Watch the untouched components.** Under claims 1 and 4, they can change role when upstream weights move, even
   with their own $u, v$ fixed.

## Open questions

- Is "meaning" (3) anything beyond write direction plus downstream effect? If so, it needs autointerp. That is
  expensive and noisy, so it should come last.
- What threshold makes a component "the same"? It probably has to be relative to the warm-start null (rule 4), not an
  absolute angle.
- Frozen Γ makes claim 1 partly true by construction on the read side. A refit Γ makes it testable, but then
  permutation matching is needed.
