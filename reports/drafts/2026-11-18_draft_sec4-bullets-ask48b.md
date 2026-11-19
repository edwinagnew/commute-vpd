# §4: the definition of reuse (bullets to write up)

*definition-lead, 2026-11-18. Part 1 is the body of §4: one bullet per definition, in Geiger's terms and notation
(Geiger et al. 2025, JMLR numbering; the arXiv HTML numbering differs). Everything else is in Part 2, for the
discussion or the appendix. **[open]** marks a clause waiting on the user.*

*Part 1 was revised on 2026-11-18 in chat with the user: the featurization is exact (the remainder is mandatory), the
fit errors are gone, and a featurization may be built on weights so that VPD fits. This revision is not yet in
`decisions.md` or in the research log's box "The working definition" (v393), which still carry the earlier wording.*

## Part 1. Definitions (body of §4)

1. **Networks.** We treat the base network $\mathcal{N}_0$ and the fine-tune $\mathcal{N}_1$ as deterministic causal
   models (Geiger Def. 5) on one signature $\Sigma$, with mechanisms $F^{(i)} = \{F^{(i)}_X\}_{X}$.
   - *Choice:* none. The two networks share an architecture, so they differ only in their mechanisms (the weights).

2. **Featurized pair.** From $\mathcal{N}_0$ alone, we fix a signature $\tilde\Sigma$ and a map
   $\tau_0 : \mathrm{Val}_{\tilde\Sigma} \to \mathrm{Val}_{\Sigma}$ that is the identity on $X_{In}$ and $X_{Out}$. The
   intermediate variables $U$ of $\tilde\Sigma$ are the *features*, one *remainder* $e$ per featurized site, and one
   remainder for the rest of the network (see *Coverage* below). The
   featurized models $\tilde{\mathcal{N}}_0, \tilde{\mathcal{N}}_1$ are causal models on $\tilde\Sigma$, with
   mechanisms $\tilde F^{(i)}_u$, such that each $\mathcal{N}_i$ is an exact transformation of $\tilde{\mathcal{N}}_i$
   under $\tau_0$ (Geiger Def. 25). There are two ways to build the pair.
   - **On activations.** Fix $\sigma_0 : \mathrm{Val}_{\Sigma} \to \mathrm{Val}_{\tilde\Sigma}$ with
     $\tau_0 \circ \sigma_0 = \mathrm{id}$, so $\tau_0$ is a bijective translation (Geiger Def. 28) and the features
     are modular features (Geiger §3.6). Then
     $$\tilde F^{(i)}_u(\tilde v) = \mathrm{Proj}_u\, \sigma_0\big(F^{(i)}(\tau_0(\tilde v))\big),$$
     which depends only on $u$'s parents $\mathrm{Pa}(u)$ (Geiger Remark 6). The remainder is whatever the features
     miss, e.g. $e = h - \mathrm{dec}_0(\mathrm{enc}_0(h))$ for an SAE. Units and rotations need none at the site.
   - **On weights (VPD).** Fix the decomposition $W_0 = \sum_c u_c v_c^\top + \Delta$ and a split rule that writes any
     update as $\delta W = \sum_c \delta W_c + \delta W_{\text{out}}$, where $\delta W_{\text{out}}$ is everything the
     rule does not assign to a component. Each component's contribution to the layer output is a variable, and
     $\tau_0$ sums them. For input $x$ to the layer,
     $$\tilde F^{(0)}_c = u_c v_c^\top x, \quad \tilde F^{(1)}_c = (u_c v_c^\top + \delta W_c)\, x, \qquad
       \tilde F^{(0)}_e = \Delta x, \quad \tilde F^{(1)}_e = (\Delta + \delta W_{\text{out}})\, x.$$
   - *Choice:* one featurization, fixed from the base model and used for both networks, so a feature is the same
     variable in both. It is exact, so there is no fit error to report.
   - *Coverage:* every mechanism the fine-tune changed has a variable (user, 2026-11-18, ask 50). One remainder
     variable, "the rest of the network", covers what the features do not. With an SAE at one site, that is the other
     layers. At the unit level, it is the embeddings, unembedding and layer-norm gains.
   - *Departure:* Geiger never relates two trained networks and has no remainder variable (§3.6.1 treats lossy SAEs
     as approximate transformations, Def. 41). His only sentence on parameters as variables is in §3.7.

3. **Behavior.** Following Geiger §3.5.2, a behavior is a partial function
   $B : \mathrm{Val}_{X_{In}} \rightharpoonup \mathrm{Val}_{X_{Out}}$. We take $B = \mathcal{N}_1|_D$, where $D$ is a
   set of inputs on which the fine-tune departs from the base model:
   $$\Delta_B = \mathbb{E}_{x \in D}\, \mathrm{KL}\big(\mathcal{N}_1(x) \,\|\, \mathcal{N}_0(x)\big) > 0.$$
   - *Choice:* $D$ is declared with every reading.

4. **Mean ablation.** For each variable $u \in U$, let $\mu_u = \mathbb{E}_{x \in D}\, \mathrm{Proj}_u\, \mathrm{Solve}(\tilde{\mathcal{N}}_{1,x})$
   be its mean value on $D$ in the fine-tune. For $K \subseteq U$ we define the hard intervention (Geiger Def. 9)
   $$\mathbf{i}_K = \{u \leftarrow \mu_u\}_{u \in U \setminus K},$$
   which fixes every variable outside $K$ to its mean.
   - *Choice:* the mean over the behavior's inputs, as the user chose (2026-11-18). This is the standard ablation of
     sub-circuit analysis, and the one place a value, rather than a mechanism, is fixed.

5. **The circuit the behavior runs through.** Following sub-circuit analysis (Geiger §3.5.2), $K^\ast$ is a smallest
   $K \subseteq U$ such that $(\tilde{\mathcal{N}}_1)_{\mathbf{i}_K}$ preserves $B$:
   $$\mathbb{E}_{x \in D}\, \mathrm{KL}\big(\mathcal{N}_1(x) \,\|\, (\tilde{\mathcal{N}}_1)_{\mathbf{i}_K}(x)\big) \le (1 - \alpha)\; \mathbb{E}_{x \in D}\, \mathrm{KL}\big(\mathcal{N}_1(x) \,\|\, (\tilde{\mathcal{N}}_1)_{\mathbf{i}_U}(x)\big).$$
   - *Choice:* every variable the behavior runs through counts, whether or not the fine-tune changed its mechanism
     (user, 2026-11-18, ask 48). The circuit is a set of variables rather than connections. $\alpha$ is declared, and
     the reference is full ablation, as in standard faithfulness scores.

6. **Reuse.** Let
   $$\mathcal{V}_u = \{\mathrm{Proj}_{\mathrm{Pa}(u)}\, \mathrm{Solve}(\tilde{\mathcal{N}}_{i,x}) : x \in D,\ i \in \{0, 1\}\}$$
   be the settings of $u$'s parents that occur on $D$, and let $s_u \in [0, 1]$ grade how far
   $\tilde F^{(1)}_u = \tilde F^{(0)}_u$ on $\mathcal{V}_u$ ($s_u = 1$ exactly when they agree there up to an output
   gain). Reuse is the
   share of the circuit whose mechanisms are still the base model's:
   $$R = \frac{1}{|K^\ast|} \sum_{u \in K^\ast} s_u.$$
   The **remainder's share** of $K^\ast$ is stated beside it.
   - *Choice:* graded, not binary. The similarity behind $s_u$ is declared with every reading.
   - *Gain:* mechanisms are compared by direction only, by squared cosine (user, 2026-11-18, ask 51). So a mechanism
     that only scales the base model's output is kept.
   - *Graft:* a variable the fine-tune has and the base model lacks, such as an added head, scores $s_u = 0$. The
     base model's mechanism for it outputs zero (user, 2026-11-18, ask 51).
   - In words: a fine-tune reuses the base model, for a behavior, to the extent that the features the behavior runs
     through still have the base model's mechanisms, on the settings of their parents that occur.
   - *Reference:* this is the claim of Prakash et al. (2024), that fine-tuning enhances existing mechanisms, stated
     for mechanisms rather than for patched values.

7. **Properties.** These follow from the definition; none of them is a measurement caveat.
   - Reuse is relative to the featurization. There is no featurization-free reuse that a reading falls short of.
   - A re-used circuit on a new cue reads as partial reuse. The untouched circuit counts as kept, and the new front end
     counts as changed.
   - A behavior the base model already has reads as full reuse, because nothing in its circuit changed.
   - Cost: a memorised fact also reads as mostly reused when it is stored by small changes in an otherwise old circuit
     (redteam-lead's cold read, 2026-11-18).
   - It is not invariant to the fine-tune re-coordinatizing an old computation. On activations, a moved feature is a
     different variable, and the move shows in the remainder. On weights, a component whose read or write vector moved
     is still that component, with a changed mechanism.
   - $K^\ast$ need not be unique.

## Part 2. Supporting material (discussion and appendix)

### A. Why compare mechanisms, not values (one paragraph in the body)

Reuse compares each variable's mechanism in the two featurized models on the same parent settings (item 6). An
interchange intervention (Geiger Def. 44) would instead fix a variable to its value in the base model's run, which is
cross-model activation patching (Prakash et al. 2024). The behavior is chosen where the two outputs differ, so every
variable it depends on takes different values in the two runs. Comparing values would read every circuit as changed.

Example: the fine-tune makes a new cue A′ look like A, and the base model's induction heads, untouched, then copy B.
- Compared by value, the heads read "changed": in the base model's run they saw plain A′.
- Compared by mechanism, they read "kept": their base-model mechanism, given what they receive in the fine-tune,
  still copies B. The example reads as partial reuse, as intended.

### B. Why one featurization (two sentences in the body)

Anything the fine-tune computes outside the base model's vocabulary lands in the remainder. When the behavior needs
it, the remainder enters $K^\ast$ with a changed mechanism.

A second, separately fitted featurizer would need a fitted pairing between the two sets of features. That freedom is
what makes abstraction vacuous (Sutter et al. 2025, by analogy).

The cost is that new structure is found but not named. The remainder's share of $K^\ast$ says how much of it there
is.

### C. Cases of the featurization (one line each in the body; the table goes in the appendix)

| Case | Built on | Features | Remainder |
|---|---|---|---|
| Units: neurons, heads, MLP layers, blocks (variable merge, Def. 37) | activations | the grouped units | none at the site; the rest of the network |
| Rotation $R$: PCA, DAS (Def. 28) | activations | $R^\top h$ | none |
| SAE | activations | $\mathrm{enc}_0(h)$ | $h - \mathrm{dec}_0(\mathrm{enc}_0(h))$ |
| VPD | weights | component contributions $u_c v_c^\top x$ | $\Delta x$, plus $\delta W_{\text{out}}\, x$ in the fine-tune |

- The split rule (least-squares, adjoint or rescaling) is part of the VPD featurization and is declared, like
  $\sigma_0$.
- Transcoders are not yet a case. Built on activations, their feature mechanisms are the base encoder in both
  networks, so every change lands in the remainder. They would need a construction on weights.
- The earlier VPD row ($a_c = v_c^\top x$ with the base read vector) had the same fault and is replaced.
- Every reading so far uses the units case.
- Attribution graphs are not a case. They freeze attention and error terms per prompt, so a feature has no mechanism
  to compare.

### D. Remarks (one sentence each in the body)

- **Two extreme choices.** Featurizations are not ordered from coarse to fine, so write "two extreme choices".
  - One variable for the whole hidden computation reads every change as new.
  - Neurons saturate: mechanism agreement is 0.999 or more on every fine-tune (E208). The user: "the neuron level is
    not a high-enough level abstraction to detect new".
- **Moved computation.** Computation the fine-tune moves elsewhere reads as changed mechanisms in the variables it
  moved into (Part 1, item 7).

### E. Open questions (cards in the research log)

- **Ask 48.** Closed 2026-11-18: every variable the behavior runs through counts (items 4-5). The user: the
  new-trigger example is "too contrived to be taken seriously, but yes intuitively that is partial reuse".
- **Ask 49.** Closed by the revision: with an exact featurization $\varepsilon_1 = 0$, so the question does not arise.
- **Ask 50.** Closed 2026-11-18: the coverage condition (item 2) replaces the repair "variables outside $U$ take
  base-model mechanisms". The user: "yes, go with the coverage condition".
- **Ask 51.** Closed 2026-11-18: a pure output-gain change is kept, and a graft scores 0 (item 6).

### F. Measurement (not the definition)

Estimates and bounds belong here, not in Part 1.
- The search for $K^\ast$ returns a set that preserves $B$; its size bounds the minimal size from above.
- The expectation over $D$ and the settings $\mathcal{V}_u$ are sampled.
- The hybrid's own parent settings may leave $\mathcal{V}_u$; report how far.
- Readings tested so far (their numbers stand):
  - $\rho = \frac{1}{|K|}\sum_{u \in K} q_u$, with
    $q_u = 1 - \mathbb{E}_{v}\, d(\tilde F^{(1)}_u(v), \tilde F^{(0)}_u(v))$ and $d = 1 - \cos^2$. Here $K$ is E208's
    circuit of the change, not $K^\ast$: the changed parts kept under swaps to the base model's mechanisms, scored
    against $\mathrm{KL}(\mathcal{N}_1 \,\|\, \mathcal{N}_0)$. It saturates at neurons (see D).
  - The held-model survival $S_{\text{held}}$. It separates only items that are new by construction (a graft, 0.13)
    from old-task fine-tunes (0.98 or more).

### G. What is ours, not Geiger's (for the "Departures" list)

- One featurization, fixed from the base model and applied to two trained networks. Geiger never relates two trained
  networks.
- A soft intervention whose replacement mechanisms come from a second trained network. The closest precedent is
  grafting (Panigrahi et al. 2023): it finds a smallest set of fine-tuned weights that preserves the task, which
  localises the change but does not compare mechanisms.
- Sub-circuit analysis over variables rather than connections, with base-model mechanisms in place of ablation values.
- The mandatory remainder variable. Geiger has no error node; §3.6.1 calls lossy SAEs "approximate transformations".
- A featurization built on weights (VPD), with a declared split of $\delta W$. Geiger's only sentence on parameters
  is in §3.7.

### H. Design choice for the discussion

The user asked on 2026-11-18 to "note this down as a design we had to make". §5 scores the re-gating share against the
definition, like the baselines. It does not present the share as the definition read at the VPD featurization, for
three reasons:
- no run has checked that the two match;
- scoring the definition against itself would say nothing;
- that reading depends on a declared split of $\delta W$.

The revision does not change this. The re-gating share is a share of the log-odds gain in VPD's masked model (the
norm share of $\delta W$ is the outside share); the definition read at VPD is
behavioral. They share the split.

### Notation

Each choice avoids a clash with another section.
- $\mathcal{N}_0$, $\mathcal{N}_1$, not the log's $M$, $F$.
- $\alpha$, not $\tau$, for the tolerance.
- $\Delta_B$ for the behavioral gap, since $\Delta$ is the VPD remainder matrix (`GOALS.md` → Terms).
- $q_u$, not $r_u$.
- $S_{\text{held}}$, not $S$.

### Not checked

- Whether $\alpha = 0.9$ in E195/E196 means $\mathrm{KL} \le 0.1\,\Delta_B$.
- The error-node precedents (Marks et al. 2024; Mueller et al. 2025; Ameisen et al. 2025), named from memory.
- The Geiger definition and section numbers, carried over from the earlier version of this file.
- Where `split.py` puts the second-order term $du_c\, dv_c^\top$; item 2 assigns anything unassigned to
  $\delta W_{\text{out}}$.
- No reading has been run at the VPD or SAE featurization.
