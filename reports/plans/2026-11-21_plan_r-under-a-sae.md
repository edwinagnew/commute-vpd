# Plan: R under wording A at the SAE featurization (E245, P1.40 run 2)

*2026-11-21, sae-lead. Cell P1.40, run 2 of 2 (the last), due 11-23. Card
`reports/plans/2026-11-21_plan_r-under-a-sae-card.md`; rules `experiments/sae_r_a/results/predictions.json`; code
`experiments/sae_r_a/run.py`, `src/commute/sae_feat.py` (to be written); SAEs from E234
(`experiments/sae_train/results/large/scored/`). Reuses `commute.stand_in` (E242, P1.46) and `commute.kstar_vpd.search`
(E226, P1.37) unchanged.*

**The request** (whip-lead 2026-11-20 ~20:35, user-approved; decisions.md "Definition wording A adopted" and the
2026-11-20 relative-bar row). Wording A: "A fine-tune F reuses the base model M, for a behavior, to the extent that M's
mechanisms, each up to an output gain, can stand in for those of the variables the behavior runs through and still
preserve the behavior." As operations: at the SAE featurization below, with a circuit K* found by E226's search, R_A is
the largest share of K* whose variables can all take M's mechanism at once, each times one positive gain, inside F's
full forward, with KL(F || F_S) <= 0.1 KL(F || M) at the read positions, mean over D (0.05 and 0.2 reported beside it).
Not the cell's done_when grade (agreement up to one gain), which wording A replaced.

## The featurization

E234's base-only BatchTopK SAEs (seed 0) at the 8 block-output sites: attn0-3 and mlp0-3 outputs. At site l with module
input y (the module's own input, after its layer norm), F's output is o_F(y) = sum_u f_u(o_F(y)) d_u + e_F(y), f the
SAE's thresholded encoder, d_u its decoder column, e_F = o_F - sum_u f_u d_u (b_dec in it). Variables per site: its
16,384 latents and one remainder e: 131,080 in all. The residual SAEs are not used (E234: under x10 the resid_post3 SAE's
FVE is -15.9, 1% of positions holding 99.6% of its error, `check_x10.py`). Embeddings, layer norms and the unembedding
are outside every variable and always keep F's mechanism (the rest-of-network remainder, main.tex ~214-216); an
overcomplete dictionary makes the translation non-bijective (~205), stated with every reading.

- **Mechanisms.** Latent u of site l in F: f_u(o_F(y)); in M: f_u(o_M(y)), M's module applied to the same input y. The
  remainder: e_F(y) in F, e_M(y) in M. Standing in (`stand_in.Site`): correction = sum_u t_u (g_u f_u(o_M) - f_u(o_F))
  d_u + t_e (g_e e_M - e_F); gain sums: num_u = sum f_u(o_F) f_u(o_M) |d_u|^2, den_u = sum f_u(o_M)^2 |d_u|^2, the
  remainder's <e_F, e_M> and |e_M|^2 over every row. At a site whose module is unchanged, o_M = o_F, so every gain is 1
  and the correction is exactly zero.
- **Ablation (K*).** F_K: each ablated latent's write replaced by its mean write, (f_u - mean f_u) d_u subtracted, the
  mean per position over D in F's unablated run (E226's convention); the remainder likewise. Keep-all is F bit for bit.
- **K*.** `kstar_vpd.search` with E226's settings (bar (1 - alpha) KL(F || F_all_ablated), alpha 0.9, its reranks,
  eval cap and chunk divisor), on attribution d KL / d keep.

## Stages (cluster debug, L40, $0; each a separate job, each starting with the do-nothing)

0. **Do-nothing** (F = M clone), first and alone in every job: every gain exactly 1, every variable stood in gives KL
   exactly 0; R_A = 1 by definition (the relative bar is 0 and the stand-in is exact). A miss stops the job.
1. **Six built edits**, at their own size, E214's World (`kstar_vpd.World`): x2, x10, moved, rebuilt (reuse); new_few,
   new_thin (new). Per item, before its search: (i) the positive control, every variable stood in with gain 1 gives
   M's logits (rel. error under the floor measured on the do-nothing); (ii) unchanged sites' gains exactly 1. Then K*,
   gains, R_A at the three bars, and the remainder share (below). Split-half band: K* and R_A on D's two halves.
2. **Stop rule** (whip-lead): if not every reuse item reads R_A above both new edits, stop; report, ask whip-lead.
3. **E237's five 96-key tasks x 3 seeds** at E237's common size (15 items), K* searched here. Descriptive.
4. **E187's nine pairs** at matched size (E232's items), only if stages 1-3 end by 11-22 18:00; else dropped. Descriptive.

**Remainder share** (the cell's "remainder share of the behaviour"): 1 - KL(F || F with only the 8 remainders stood
in, gains fitted) / KL(F || M); nan where KL(F || M) = 0. Its control is the planted change new_thin and new_few already
are: appended mlp0 neurons the base SAE cannot write (E234: their mlp0 task FVE drops 1.14-1.15), so their remainder
share should be high, x10's low (its mlp1 FVE drop is 0.02).

## What each number should read

| item | changed sites | expected R_A (bar 0.1) | expected remainder share |
|---|---|---|---|
| do-nothing | none | 1 exactly | nan |
| x2, x10 | mlp1 | near 1: mlp1's SAE writes the rescaled neurons (FVE drop <= 0.02) and per-latent gains undo a rescale | under 0.3 |
| moved, rebuilt | mlp1 (moved also its grafts) | near 1; moved's grafts keep F's mechanism | under 0.3 |
| new_few, new_thin | mlp0 | rival a: below every reuse item by the band; rival b: near 1, the whole new write sitting in one variable, mlp0's remainder | over 0.5 |

The worry for this featurization is rival b: R_A counts variables, and an SAE puts what its dictionary lacks into one
remainder per site, so the new edits can lose only one variable of K*. The remainder share is read beside R_A for that
reason.

## Runtime

One K* evaluation on the SAE forward is 8 encodes of (tokens x 16,384) beside F's forward; at 3,000-4,000 tokens per
item that is a few tens of ms on an L40 (to be measured in the rehearsal and written here). E226's cap is 2,000
evaluations per search, three searches per item (all of D, two halves), plus up to 2,000 stand-in evaluations: at
0.1 s each about 13 min per item, so stage 1 about 80 min in one job or 30 min in three; stage 3 about 15 x 13 min in
four jobs. Not gated (cluster, $0, each job under 2 h).

## Rehearsal ($0, cluster debug)

The do-nothing and x10 at full read sizes with an eval cap of 50: measures time per evaluation, peak GPU memory, the
positive-control floor (do-nothing and x10), and that the gains at unchanged sites are exactly 1.

## Not run

The residual-stream SAEs; standing in inside the circuit; gains fitted on behaviour; a random-update control (E242 runs
it at VPD); a second SAE seed; the nine pairs if late; an uncapped search.
