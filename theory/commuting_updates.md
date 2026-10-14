# Commuting updates past a parameter decomposition: a formal statement

*2026-10-14, coding agent, at the user's request ("write a theory note … be precise and formal and state clearly
what assumptions you are making"). This note formalises the question in `GOALS.md`. It proposes no experiment. The
quantities in §7 are candidates for round 2 and go through `/design` before anything is run.*

**Design check, shown with the note rather than before it (the user asked for the note directly).**
- **Goal served:** `GOALS.md`, "The question made precise". This note turns the weak and strong versions into
  definitions and derives what round 1 did and did not test.
- **Nearest prior:** `TRIED.md` row 1 (transport test round 1), plus the user's framing as a lift or a commuting
  square.
- **Patch or step:** a step. Round 2's measurements should be derived from definitions, not from the last
  experiment's flaws.
- **Decisions:** relies on the weak version as target, Γ frozen, and LoRA among the updates (`decisions.md`,
  2026-10-14). No departures.
- **What would change next:** Proposition 1 implies that round 1 tested only scale-only lifts, a subspace of
  dimension at most $C$ inside a first-order lift space of dimension up to $C(d_\text{in}+d_\text{out})$. If the
  user accepts that, round 2 measures the obstruction (§7a) rather than repeating the projection test.

---

## 1. Objects

Fix a network architecture and a finite set $\mathcal M$ of decomposed linear maps (here the 24 matrices
`h.ℓ.{attn.{q,k,v,o}_proj, mlp.{c_fc,down_proj}}`, ℓ = 0..3).

- **Weights.** $\mathcal W = \prod_{m\in\mathcal M}\mathbb R^{d^m_\text{out}\times d^m_\text{in}}$, with the
  Frobenius inner product $\langle A,B\rangle=\sum_m \operatorname{tr}(A_m^\top B_m)$. $f_W$ is the network with
  these matrices, all other parameters held at their released values.
- **Components.** For each $m$, $C_m$ rank-1 subcomponents $u_c v_c^\top$ with $u_c\in\mathbb R^{d_\text{out}}$
  (write vector) and $v_c\in\mathbb R^{d_\text{in}}$ (read vector). Collect them as
  $U_m=[u_1\cdots u_{C_m}]\in\mathbb R^{d_\text{out}\times C_m}$ and $V_m\in\mathbb R^{d_\text{in}\times C_m}$.
  (In code `U` is stored as $U_m^\top$.)
- **Causal importance.** $\Gamma_\varphi$ maps the pre-weight activations of all decomposed layers at a position,
  $a(x;W)=(a_m(x;W))_m$, to $g(x)\in[0,1]^{\sum_m C_m}$.
- **Decomposition space.** $\mathcal D=\prod_m(\mathbb R^{d_\text{out}\times C_m}\times\mathbb R^{d_\text{in}\times C_m})\times\Phi$,
  where $\Phi$ is the parameter space of Γ. A point is $d=(U,V,\varphi)$.
- **Component sum.** $\sigma:\mathcal D\to\mathcal W$, $\sigma(d)_m=U_mV_m^\top=\sum_c u_cv_c^\top$. It ignores
  $\varphi$.
- **Residual.** As in param-decomp, Δ is not a free parameter. For $d$ as a decomposition of $W$ it is
  $\Delta(d,W)=W-\sigma(d)$.
- **Update.** $\tau_{\delta W}:\mathcal W\to\mathcal W$, $W\mapsto W+\delta W$.

## 2. Standing assumptions

- **A1 (what moves).** Only the matrices in $\mathcal M$ change. Embeddings, unembedding and norms are frozen. A LoRA
  of rank $r$ on matrix $m$ is the case $\delta W_m=B_mA_m$ with $B_m\in\mathbb R^{d_\text{out}\times r}$.
- **A2 (small updates).** $\|\delta W_m\|_F\le\varepsilon\|W_m\|_F$ for an ε small enough that first-order
  expansions hold. This applies both to $\sigma$ (the term $\delta U\,\delta V^\top$ is neglected) and to the
  network (the activations $a(x;W+\delta W)=a(x;W)+O(\varepsilon)$). Every statement marked "first order" is exact
  only as $\varepsilon\to0$.
- **A3 (Γ frozen).** $\varphi$ is fixed: $\varphi'=\varphi_0$ (`decisions.md`, 2026-10-14). Because Γ reads
  activations, $g$ still changes with $W$, by $O(\varepsilon)$.
- **A4 (faithfulness means VPD's own objective).** The data distribution μ (Pile) is fixed. "$d$ is a faithful
  decomposition of $W$" is defined by the loss terms VPD trains on, each evaluated at $(d,W)$ on μ:
  - $\mathcal L_F(d,W)=\frac1N\sum_m\|\Delta_m(d,W)\|_F^2$: weight faithfulness, coefficient $10^7$ in the
    released run;
  - $\mathcal L_S$: stochastic masked reconstruction KL;
  - $\mathcal L_P$: adversarial (PGD) masked reconstruction KL;
  - $\mathcal L_I$: importance minimality, $\sum(g^{\text{upper}}_c+\epsilon)^p$.
- **A5 (alive restriction).** A map may change only alive components, $\mathcal A_m\subseteq[C_m]$, those with mean
  $g_c$ on μ above $10^{-6}$. $U_{\mathcal A},V_{\mathcal A}$ are the corresponding columns, with ranks
  $r_U=\operatorname{rank}U_{\mathcal A}$ and $r_V=\operatorname{rank}V_{\mathcal A}$. Dead components have
  $g\approx0$, and with Γ frozen they stay dead.
- **A6 (metrics).** Unless stated otherwise, all inner products on $\mathcal W$ and on the $(U,V)$ factor of
  $T\mathcal D$ are Euclidean (Frobenius). §5 shows that the choice of metric is the substantive choice, so this is a
  default, not a claim.
- **A7 (post-training data).** For the gradient map (§5.3) only, the post-training loss ℓ and its data ν are
  available. For the other maps only δW is.

## 3. The weak version: a lift

**Definition 1 (relative faithfulness).** Fix the released pair $(d_0,W_0)$ and a tolerance $\kappa\ge0$. A pair
$(d',W')$ is *κ-faithful relative to* $(d_0,W_0)$ if
$\mathcal L_\bullet(d',W')\le(1+\kappa)\,\mathcal L_\bullet(d_0,W_0)$ for every $\bullet\in\{F,S,P,I\}$. This is the
relation $R_\kappa\subseteq\mathcal D\times\mathcal W$. Prediction 3 of round 1 used this ratio form for
$\mathcal L_S,\mathcal L_P$ only.

**Definition 2 (map, lift).** A *map* is a rule $\Phi:(d,\delta W)\mapsto d'$ that does not re-fit by optimising the
VPD objective. It is a *strict lift* of $\tau_{\delta W}$ at $d_0$ if the components carry the whole update:
$$\sigma(\Phi(d_0,\delta W))=\sigma(d_0)+\delta W,\qquad\text{equivalently}\qquad\Delta(\Phi(d_0,\delta W),W_0+\delta W)=\Delta(d_0,W_0).$$
That is, the square below commutes:

```
        Φ(·, δW)
   𝒟 ───────────▶ 𝒟
   │σ              │σ
   ▼               ▼
   𝒲 ───────────▶ 𝒲
        τ_δW
```

**Weak version.** For small δW of the kinds in A1, find a map Φ such that $(\Phi(d_0,\delta W),W_0+\delta W)\in R_\kappa$
for small κ. A strict lift keeps $\mathcal L_F$ unchanged. The remaining terms ($\mathcal L_S$, $\mathcal L_P$,
$\mathcal L_I$) are what make the problem non-trivial.

**Remark 1 (why Δ must be constrained).** The map $\Phi(d,\delta W)=d$ is always defined, and it gives
$\Delta'=\Delta_0+\delta W$. The reconstruction losses include Δ as an extra maskable component (`use_delta_component`),
so $\mathcal L_S$ and $\mathcal L_P$ can stay near their $W_0$ values while $\mathcal L_F$ grows by
$\|\delta W\|^2/N$. Round 1's faithfulness measure checked only $\mathcal L_S$ and $\mathcal L_P$, so it could not
reject this map. Definition 1 includes $\mathcal L_F$, which does reject it.

## 4. Symmetries

Let $G_m=(\mathbb R^\times)^{C_m}\rtimes S_{C_m}$.
- $s\in(\mathbb R^\times)^{C_m}$ acts by $u_c\mapsto s_cu_c$, $v_c\mapsto v_c/s_c$.
- $\pi\in S_{C_m}$ permutes components, and permutes Γ's output coordinates for $m$ at the same time (an action on
  φ).

**Lemma 1.** $G=\prod_mG_m$ preserves $\sigma$, every masked forward (a mask multiplies $u_cv_c^\top a$, which is
invariant under the rescaling), $g$ (Γ reads activations, not $V^\top a$), and therefore every $\mathcal L_\bullet$.

*Proof.* Direct substitution. Invariance of $g$ under rescaling uses the fact that Γ's input is $a(x;W)$
(param-decomp `calc_causal_importances(pre_weight_acts)`), and $a$ depends on $W$, not on $(U,V)$. ∎

So decompositions are defined only up to $G$. Statements about "the same decomposition" (§6) live on $\mathcal D/G$.

## 5. First-order analysis

At $d_0$, for one matrix (drop $m$), the differential of σ on the alive columns is
$$d\sigma(\delta U,\delta V)=\delta U\,V_{\mathcal A}^\top+U_{\mathcal A}\,\delta V^\top .$$
By A2, a first-order strict lift exists iff $\delta W\in\operatorname{Im}d\sigma$. Let $P_U$ and $P_V$ be the
orthogonal projections onto $\operatorname{col}U_{\mathcal A}\subseteq\mathbb R^{d_\text{out}}$ and
$\operatorname{col}V_{\mathcal A}\subseteq\mathbb R^{d_\text{in}}$, and let $P^\perp=I-P$.

**Proposition 1 (image of the differential).**
$$\operatorname{Im}d\sigma=\{M:\;P_U^\perp MP_V^\perp=0\},\qquad\dim=d_\text{in}d_\text{out}-(d_\text{out}-r_U)(d_\text{in}-r_V).$$
In particular σ is a submersion at $d_0$ (all small δW liftable) iff $r_U=d_\text{out}$ or $r_V=d_\text{in}$.

*Proof.*
- (⊆) $P_U^\perp U_{\mathcal A}=0$ and $V_{\mathcal A}^\top P_V^\perp=0$, so both terms vanish under
  $P_U^\perp(\cdot)P_V^\perp$.
- (⊇) Suppose $P_U^\perp MP_V^\perp=0$. Write $M=P_UM+P_U^\perp MP_V$.
  - $P_UM=U_{\mathcal A}(U_{\mathcal A}^+M)$, which is of the form $U_{\mathcal A}\delta V^\top$.
  - $P_V=(V_{\mathcal A}^+)^\top V_{\mathcal A}^\top$, so $P_U^\perp MP_V=(P_U^\perp M(V_{\mathcal A}^+)^\top)V_{\mathcal A}^\top$,
    which is of the form $\delta U\,V_{\mathcal A}^\top$.
- The dimension is that of the kernel of the surjection $M\mapsto P_U^\perp MP_V^\perp$ onto
  $\operatorname{col}(P_U^\perp)\otimes\operatorname{col}(P_V^\perp)$. ∎

**Corollary 1 (the obstruction).** The orthogonal complement of $\operatorname{Im}d\sigma$ is
$\{P_U^\perp NP_V^\perp\}$, with orthogonal projection $\mathcal O(M)=P_U^\perp MP_V^\perp$. To first order, any map
that changes only $(U_{\mathcal A},V_{\mathcal A})$ leaves at least $\|\mathcal O(\delta W)\|_F$ in Δ:
$$\min_{\delta d}\|\Delta'-\Delta_0\|_F=\|P_U^\perp\,\delta W\,P_V^\perp\|_F .$$
This is the part of the update that maps inputs outside every alive read direction to outputs outside every alive
write direction. Define the **obstruction fraction** $\omega=\sum_m\|\mathcal O(\delta W_m)\|_F^2/\sum_m\|\delta W_m\|_F^2$.

**Corollary 2 (nulls).**
- For δW with i.i.d. isotropic Gaussian entries,
  $\mathbb E\,\|\mathcal O(\delta W)\|_F^2/\|\delta W\|_F^2=(1-r_U/d_\text{out})(1-r_V/d_\text{in})$, exactly in
  expectation.
- For a gradient-shaped update $\delta W=\sum_t e_t\,a_t^\top$ (output-side error $e_t$ times layer input $a_t$), ω
  is zero whenever the inputs lie in $\operatorname{col}V_{\mathcal A}$ or the errors lie in
  $\operatorname{col}U_{\mathcal A}$.
- VPD trains $V$ to read the activations μ produces. So the Gaussian null overstates the obstruction a real update
  meets, and a fair null keeps δW's row and column spaces (e.g. randomise the coefficients of $\sum_te_ta_t^\top$).

**Corollary 3 (LoRA).** If $\delta W=BA$ has rank $r$, then $\operatorname{rank}\mathcal O(\delta W)\le r$, and
$\mathcal O(\delta W)=(P_U^\perp B)(AP_V^\perp)$.

**Remark 2 (what round 1 tested).** The two transport families in round 1 have these first-order lift spaces:
- **Scale-only (T0):** $\delta u_c=\alpha_cu_c$, $\delta V=0$, with image $\operatorname{span}\{u_cv_c^\top\}$ of
  dimension at most $|\mathcal A|$. The Frobenius fraction measured the projection onto this subspace.
- **Write edits on a set $S$ (T1):** $\delta V=0$, with image $\{\delta U\,V_S^\top\}$, the matrices whose row space
  lies in $\operatorname{col}V_S$.

Both are proper subspaces of $\operatorname{Im}d\sigma$. So prediction 2's reversal (fraction 1.5× random) says
scale-only lifts fail. It says nothing about whether a full first-order lift exists. That question is ω.

**Lemma 2 (where the update is placed matters: locality).** Take a strict first-order lift with per-component
changes $\delta B_c=\delta u_cv_c^\top+u_c\delta v_c^\top$, so $\sum_c\delta B_c=\delta W$. Under A2 and A3, fix a
layer input $a$ and Γ's output $g=g(a)$, ignoring the $O(\varepsilon)$ change of $g$. With masks $m_c\in[0,1]$, the
change in the layer's masked output is $\sum_cm_c\,\delta B_c\,a$. The fine-tuned layer's change is
$\delta W a=\sum_c\delta B_c\,a$. The difference is
$$r(a)=\sum_c(1-m_c)\,\delta B_c\,a .$$
- **CI-masked forward** ($m=g$): $r(a)=\sum_c(1-g_c(a))\,\delta B_c\,a$.
- **Continuous stochastic sampling** ($m_c=g_c+(1-g_c)\xi_c$, $\xi_c\sim U[0,1]$ i.i.d.):
  $\mathbb E_\xi\,r(a)=\tfrac12\sum_c(1-g_c(a))\,\delta B_c\,a$.

*Proof.* Linearity of the layer in its weights. ∎

So a strict lift also keeps the masked forwards faithful only if the update is placed on components that are
causally important on the inputs $a$ where $\delta W a\neq0$. Define the **locality residual**
$$\rho(\delta d)=\frac{\mathbb E_{x\sim\nu\cup\mu}\sum_m\big\|\sum_{c}(1-g_c)\,\delta B_c\,a_m\big\|^2}{\mathbb E\sum_m\|\delta W_ma_m\|^2}.$$
Lemma 2 covers one layer. Effects propagated through later layers are $O(\varepsilon)$ per layer, but they compound;
the end-to-end statement is $\mathcal L_S,\mathcal L_P$ in Definition 1.

### 5.1 Maps as choices of lift

When $\delta W\in\operatorname{Im}d\sigma$, the first-order lifts form an affine space,
$\delta d^\ast+\ker d\sigma$. The kernel contains the gauge directions of Lemma 1
($\delta u_c=su_c$, $\delta v_c=-sv_c$), and more when the $u_c$ or $v_c$ are linearly dependent. A map must pick
one element.

**Definition 3.** Given a positive-definite quadratic form $Q_d$ on the $(U,V)$ part of $T_d\mathcal D$, the
*$Q$-horizontal lift* is
$$L^Q_d(\xi)=\arg\min\{Q_d(\delta d):\,d\sigma(\delta d)=\xi\}.$$
For $\xi\in\operatorname{Im}d\sigma$ this is linear in ξ and a right inverse of $d\sigma$. Its image $H_d=\operatorname{Im}L^Q_d$ is the $Q$-orthogonal complement of the vertical space
$\ker d\sigma$. When σ is a submersion, $d\mapsto H_d$ is an Ehresmann connection on σ. Off $\operatorname{Im}d\sigma$, apply $L^Q$ to the projection of ξ onto the image; the
remainder $\mathcal O(\xi)$ goes to Δ.

Instances:
1. **Euclidean** $Q=\|\delta U\|^2+\|\delta V\|^2$: the minimum-norm lift, $L=d\sigma^+$ (Moore–Penrose).
2. **Locality** $Q=\rho+\lambda(\|\delta U\|^2+\|\delta V\|^2)$, λ > 0 (data-dependent): the first-order lift that
   best preserves the masked forwards (Lemma 2), in closed form per matrix. It is a candidate "simple operation":
   one linear solve, no VPD re-fit.
3. **Round 1's T0 and T1** are not of this form. They restrict the lift to the subspaces of Remark 2.

### 5.2 Size of the lift

$\|L_d(\xi)\|\le\|\xi\|/s_{\min}$, where $s_{\min}$ is the smallest non-zero singular value of $d\sigma$. If the
alive $u_c$ or $v_c$ are nearly collinear, a small δW may need a large component change. That would break A2 in
$\mathcal D$ even though it holds in $\mathcal W$. $s_{\min}$ is a property of $d_0$ alone and can be computed once.

### 5.3 The gradient pull-back (what "pull back" meant in chat)

Given A7, let ℓ be the post-training loss on $\mathcal W$. The covector $d\ell$ pulls back along
$W=\sigma(d)+\Delta_0$ to $\sigma^*d\ell$ on $\mathcal D$; this is the differential-geometric pull-back, not the
categorical one. One Euclidean gradient step on the pulled-back loss is
$$\delta d=-\eta\,d\sigma^*\nabla\ell,\qquad d\sigma^*(M)=(MV_{\mathcal A},\,M^\top U_{\mathcal A}).$$

**Proposition 2.** The induced weight change is
$$d\sigma(\delta d)=-\eta\,K(\nabla\ell),\qquad K(M)=U_{\mathcal A}U_{\mathcal A}^\top M+MV_{\mathcal A}V_{\mathcal A}^\top .$$
It is not a lift of the step $-\eta\nabla\ell$ taken in $\mathcal W$. It is a lift of a different, preconditioned
step. $K$ is positive semi-definite on $\mathbb R^{d_\text{out}\times d_\text{in}}$, with eigenvalues
$s_i^2+t_j^2$, where $s_i$ and $t_j$ are the singular values of $U_{\mathcal A}$ and $V_{\mathcal A}$ (padded with
zeros). Its kernel is exactly the obstruction subspace $\{P_U^\perp NP_V^\perp\}$ of Corollary 1.

*Proof.* $d\sigma(MV_{\mathcal A},M^\top U_{\mathcal A})=MV_{\mathcal A}V_{\mathcal A}^\top+U_{\mathcal A}U_{\mathcal A}^\top M$.
As an operator, $K=U_{\mathcal A}U_{\mathcal A}^\top\otimes I+I\otimes V_{\mathcal A}V_{\mathcal A}^\top$,
a sum of commuting positive semi-definite operators. Its eigenvectors are the products of left and right singular
vectors, with eigenvalues $s_i^2+t_j^2$. An eigenvalue is zero iff $s_i=t_j=0$, i.e. on
$\operatorname{col}P_U^\perp\otimes\operatorname{col}P_V^\perp$. ∎

**Consequences.**
- The pull-back does not map a given δW (e.g. a LoRA's $BA$) to a component update. It replays the post-training
  problem on $\mathcal D$, so it needs ℓ and ν.
- It is weight-faithful by construction, since Δ is fixed. Whether it reaches the same behaviour as the
  $\mathcal W$-trained model is an empirical question. Its trajectory differs from the fine-tune's even to first
  order, because $K\ne\mathrm{id}$.
- It is the natural Φ when the question is "can post-training be done in the decomposition", rather than "can a
  finished update be translated into the decomposition".

## 6. Paths, the categorical pull-back, and the strong version

**6.1 Path lifting (the "lift" in the user's sense).** Let $\gamma:[0,1]\to\mathcal W$ be a post-training trajectory
with $\gamma(0)=W_0$. The weak version along training asks for a diagonal filler $\tilde\gamma$ in the square below,
staying in $R_\kappa$ for all $t$:

```
   {0} ──── d₀ ────▶ 𝒟
    │                │ σ + Δ₀
    ▼                ▼
  [0,1] ──── γ ────▶ 𝒲
```

This is the path-lifting property of $W(d)=\sigma(d)+\Delta_0$. The categorical pull-back appears here: the
pull-back bundle $\gamma^*\mathcal D=[0,1]\times_{\mathcal W}\mathcal D$, the fibre product of $\gamma$ and
$\sigma+\Delta_0$. A strict lift along γ is exactly a section of $\gamma^*\mathcal D$ through $(0,d_0)$.
- Where σ is a submersion (Proposition 1), short lifts exist by the implicit function theorem. Global lifts are not
  guaranteed: σ is not proper, so Ehresmann's fibration theorem does not apply.
- Given a connection $L^Q$ (Definition 3), the horizontal lift solves $\dot d=L^Q_d(\dot\gamma)$. This is parallel
  transport. "Continuous interpretability" is the statement that this ODE's solution stays in $R_\kappa$.
- **Holonomy.** Transport around a closed loop in $\mathcal W$ (e.g. $+\delta W_1,+\delta W_2,-\delta W_1,-\delta W_2$)
  returns to the starting fibre at a possibly different point. The displacement, measured on $\mathcal D/G$, is the
  path dependence of the transported decomposition. It is second order in the loop size and governed by the
  curvature of $L^Q$.

**6.2 The strong version: an approximately commuting square.** Let $\mathsf D$ denote VPD training. It is not a
function (non-convex, seed-dependent), so model it as a set-valued map:
- $\mathsf D(W)\subseteq\mathcal D$, the outcomes over seeds; or
- $\mathsf D(W;d_\text{init})$, warm-started.

Let $\delta_G$ be a metric on $\mathcal D/G$, e.g. $1-\text{MMCS}$, the paper's mean max cosine similarity of
matched rank-1 subcomponents (0.28 across seeds; 0.01 at random init). The strong version asks that the following
square commute up to the noise floor:

```
          τ_δW
   W₀ ───────────▶ W₀+δW
   │ 𝖣(·; d₀)       │ 𝖣(·; d₀)
   ▼                ▼
   d₀ ───────────▶ d′ = Φ(d₀, δW)
          Φ
```

Formally:
$$\delta_G\big([\Phi(d_0,\delta W)],\,[\mathsf D(W_0+\delta W;d_0)]\big)\;\le\;\delta_G\big([d_0],\,[\mathsf D(W_0;d_0)]\big)+\eta .$$
The right-hand side's first term is the δW = 0 warm-start control, i.e. how far re-fitting the *same* model moves the
decomposition. The square is up to $G$ and up to that floor; it cannot be strict. Warm-started re-fitting is itself a
map, but it is excluded by Definition 2 ("simple"). The strong version asks whether a simple map approximates it.
Postponed (`decisions.md`).

## 7. Quantities this note defines (candidates for round 2, not yet designed)

- **(a) Obstruction fraction ω** (Corollary 1). Per matrix and pooled, for each update: full fine-tune, LoRA, the
  gradient-shaped null of Corollary 2, and the Gaussian null. Needs only $U_{\mathcal A},V_{\mathcal A}$ and δW,
  with no forward passes. Small ω means a first-order strict lift exists; large ω means part of the update is
  outside what the alive components can express, whichever map is used.
- **(b) Locality residual ρ** of the Euclidean and locality lifts (Lemma 2, Definition 3), and then the end-to-end
  ratios of Definition 1, with Δ computed honestly (the $\mathcal L_F$ term included).
- **(c) Conditioning** $s_{\min}$ of $d\sigma$ at $d_0$ per matrix (§5.2).
- **(d) Pull-back replay** (§5.3): post-train on $(U_{\mathcal A},V_{\mathcal A})$ with ℓ, ν, compare behaviour
  with the $\mathcal W$-trained model, and check Definition 1.

## 8. What the formalism does not cover

- **Γ.** A3 freezes it. If $g$ is wrong for the new model (a new behaviour that no component is important for), no
  choice of $(U,V)$ lift fixes $\mathcal L_S$ or $\mathcal L_I$. The locality residual ρ detects this only as an
  irreducible remainder. Updating Γ is in `backlog.md`.
- **Dead components and new components.** A5 excludes reviving dead components or adding new ones. Both change $G$
  and the fibre structure.
- **Second-order effects.** The $\delta U\,\delta V^\top$ term and the change of $g$ through activations are
  neglected (A2). Lemma 2 is per layer.
- **"Simple".** It is defined negatively (no VPD re-fit). The lifts in Definition 3 are linear solves. The pull-back
  is a short optimisation on a different loss. Whether the latter counts as simple is a judgement for the user.
