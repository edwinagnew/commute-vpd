"""E62 E_b diagnosis (cell P1.57, $0, CPU): is head (2,4) in K, is its update QK or OV, and is its write change kept.

Reads only saved files: the decomposition (vpd.checkpoint_weights), E62's dW (induction_cue/results/large),
E257's per-variable num, den, dd, changed (rb_states/results/large/items_e62-*.npz), E275's in_K, flips
(claim2_needed/results/large/items_e62-*.npz) and E275's item JSON (E_b_K, size_K, min_margin).
Writes one JSON (argv[1]).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from commute import split as SP
from commute import vpd

REPO = Path.cwd()
RB = REPO / "experiments/rb_states/results/large"
C2 = REPO / "experiments/claim2_needed/results"
E62 = REPO / "experiments/induction_cue/results/large"
HEAD, HD = 4, 128
H4 = slice(HEAD * HD, (HEAD + 1) * HD)
L2 = [f"h.2.attn.{m}_proj" for m in "qkvo"]
THRESH = (0.3, 0.5)  # no layer-2 subcomponent has over 0.74 of its weight in head 4 (shares.py); uniform is 1/6
SPLIT_TOL = 1e-6  # experiments/kstar_vpd/results/predictions.json settings.split_tol
STATES = [f"{m}_s{s}" for m in ("copy", "lookup") for s in range(3)]

W0, uv = vpd.checkpoint_weights()
keys = sorted(uv)
off, sites = 1, {}
for k in keys:
    C = uv[k][0].shape[0]
    sites[k] = (off, off + C)  # subcomponents off..off+C-1, remainder off+C
    off += C + 1
n = off + 1
assert n == 38938, n


def head4_share(k: str) -> np.ndarray:
    """Per subcomponent of matrix k: share of |u_c v_c^T|_F^2 in head 4's slice (rows of q/k/v, input columns of o)."""
    U, V = uv[k][0].double(), uv[k][1].double()  # U (C, d_out), V (d_in, C)
    if k.endswith("o_proj"):
        x = V.T  # (C, d_in): o_proj's input columns are the heads' outputs
    else:
        x = U  # (C, d_out): q/k/v rows are the heads
    return ((x[:, H4] ** 2).sum(1) / (x**2).sum(1).clamp(min=1e-300)).numpy()


share = {k: head4_share(k) for k in L2}
head_vars = {t: np.concatenate([np.arange(*sites[k])[share[k] > t] for k in L2]) for t in THRESH}
var_mat = {}
for k in L2:
    for i, c in enumerate(range(*sites[k])):
        var_mat[c] = (k, i)

bases = {k: SP.basis(uv[k][0].T.double(), uv[k][1].double(), SPLIT_TOL) for k in L2}


def site_of(v: int) -> str:
    for k, (a, r) in sites.items():
        if a <= v <= r:
            return k
    return "embed" if v == 0 else "unembed"


owner = np.array([site_of(v) for v in range(n)])
is_rem = np.zeros(n, bool)
is_rem[[r for _, r in sites.values()]] = True


def dw_parts(dW: dict[str, torch.Tensor]) -> dict:
    out = {}
    for k in L2:
        d = dW[k].double()
        hs = d[:, H4] if k.endswith("o_proj") else d[H4, :]
        tot = float((d**2).sum())
        # least-squares split of this matrix's update (as kstar_vpd._site): write side, read side, outside part,
        # each restricted to head 4's slice
        Ua, Va = uv[k][0].T.double(), uv[k][1].double()
        dUa, dVa = SP.least_squares_split(d, bases[k], Ua, Va)
        wr, rd = dUa @ Va.T, Ua @ dVa.T
        outp = d - wr - rd
        sl = (lambda m: m[:, H4]) if k.endswith("o_proj") else (lambda m: m[H4, :])
        # per subcomponent: its update du_c v_c^T + u_c dv_c^T, share of its squared norm in head 4's slice
        out[k] = {"frob2": tot, "frob2_head4": float((hs**2).sum()), "head4_share": float((hs**2).sum()) / tot,
                  "head4_outside_share": float((sl(outp) ** 2).sum()) / max(float((hs**2).sum()), 1e-300),
                  "outside_share_all": float((outp**2).sum()) / tot}  # fmt: skip
    return out


res: dict = {"head_rule": "layer-2 q/k/v (rows 512:640) or o (input columns 512:640) subcomponent with share of |u v^T|^2 in head 4's slice over threshold",
             "n_head_vars": {str(t): int(len(head_vars[t])) for t in THRESH},
             "head_vars_by_matrix": {str(t): {k: int((share[k] > t).sum()) for k in L2} for t in THRESH},
             "n_subcomponents_L2": {k: int(len(share[k])) for k in L2}, "states": {}}  # fmt: skip
for st in STATES:
    a, b = np.load(RB / f"items_e62-{st}.npz"), np.load(C2 / "large" / f"items_e62-{st}.npz")
    item = json.loads((C2 / "items" / f"e62-{st}.json").read_text())
    num, den, dd, ch = a["num"], a["den"], a["dd"], a["changed"]
    K, flips = b["in_K"].astype(bool), b["flips"]
    pos = (den > 0) & (num > 0)
    expl = np.where(pos, (num - den) ** 2 / np.where(pos, den, 1.0), 0.0)
    sel = K & ch
    tot = dd[sel].sum()
    eb = expl[sel].sum() / tot
    r: dict = {"E_b_K_item": item["E_b_K"], "E_b_K_recomputed": float(eb), "size_K": int(K.sum()), "n_vars": int(n),
               "K_frac": float(K.sum() / n), "min_margin": item["min_margin"], "n_pairs": item["n_pairs"],
               "E_b_all_changed": float(expl[ch].sum() / dd[ch].sum()),
               "flips_median_all": float(np.median(flips[1:-1])), "flips_median_K": float(np.median(flips[K])),
               "flips_max_all": int(flips.max())}  # fmt: skip
    for t in THRESH:
        hv = head_vars[t]
        hK = hv[K[hv]]
        hsel = np.zeros(n, bool)
        hsel[hv] = True
        hsel &= sel
        r[f"head_{t}"] = {
            "n": int(len(hv)), "n_in_K": int(K[hv].sum()), "in_K_frac": float(K[hv].mean()),
            "flips_median": float(np.median(flips[hv])), "flips_max": int(flips[hv].max()),
            "flips_pct_rank_median": float(np.median([(flips[1:-1] < f).mean() for f in flips[hv]])),
            "dd_share_of_K": float(dd[hsel].sum() / tot), "contrib_to_E_b": float(expl[hsel].sum() / tot),
            "kept_share_pooled": float(expl[hsel].sum() / dd[hsel].sum()) if dd[hsel].sum() > 0 else None,
            "E_b_without_head": float((expl[sel & ~hsel].sum()) / dd[sel & ~hsel].sum()),
        }  # fmt: skip
    # soft assignment: every layer-2 q/k/v/o subcomponent weighted by its head-4 weight share s_u
    s_all = np.zeros(n)
    for k in L2:
        s_all[sites[k][0] : sites[k][1]] = share[k]
    sw = s_all * sel
    r["soft"] = {"dd_share_of_K": float((sw * dd).sum() / tot), "contrib_to_E_b": float((sw * expl).sum() / tot),
                 "kept_share": float((sw * expl).sum() / (sw * dd).sum()),
                 "in_K_weighted_frac": float((s_all * K).sum() / s_all.sum())}  # fmt: skip
    for k in L2:
        a_, r_ = sites[k]
        r["soft"][k] = {"n_in_K": int(K[a_:r_].sum()), "n": int(r_ - a_), "flips_median": float(np.median(flips[a_:r_]))}
    # the top head-4 subcomponents by dd (t = 0.3)
    hv = head_vars[0.3]
    order = hv[np.argsort(-dd[hv])][:8]
    r["head_top_dd"] = [{"var": int(v), "matrix": var_mat[v][0], "c": var_mat[v][1], "head4_share": float(share[var_mat[v][0]][var_mat[v][1]]),
                         "in_K": bool(K[v]), "flips": int(flips[v]), "dd_weight_K": float(dd[v] / tot) if K[v] else None,
                         "kept": float(expl[v] / dd[v]) if dd[v] > 0 else None} for v in order]  # fmt: skip
    # remainders of the four layer-2 attention matrices
    r["L2_remainders"] = {k: {"in_K": bool(K[sites[k][1]]), "flips": int(flips[sites[k][1]]), "dd_weight_K": float(dd[sites[k][1]] / tot) if K[sites[k][1]] else 0.0,
                              "kept": float(expl[sites[k][1]] / dd[sites[k][1]]) if dd[sites[k][1]] > 0 else None} for k in L2}  # fmt: skip
    # dd mass over K & changed by matrix, subcomponents vs remainder, and by layer and kind
    by = {}
    for k in keys:
        m = (owner == k) & sel
        by[k] = {"dd_share_sub": float(dd[m & ~is_rem].sum() / tot), "dd_share_rem": float(dd[m & is_rem].sum() / tot),
                 "kept_sub": float(expl[m & ~is_rem].sum() / max(dd[m & ~is_rem].sum(), 1e-300)),
                 "kept_rem": float(expl[m & is_rem].sum() / max(dd[m & is_rem].sum(), 1e-300))}  # fmt: skip
    r["by_matrix"] = by
    r["dd_share_remainders"] = float(dd[sel & is_rem].sum() / tot)
    r["kept_remainders"] = float(expl[sel & is_rem].sum() / dd[sel & is_rem].sum())
    r["kept_subcomponents"] = float(expl[sel & ~is_rem].sum() / dd[sel & ~is_rem].sum())
    r["dd_share_L2_attn"] = float(sum(by[k]["dd_share_sub"] + by[k]["dd_share_rem"] for k in L2))
    r["dd_share_qk_all_layers"] = float(sum(v["dd_share_sub"] + v["dd_share_rem"] for k, v in by.items() if k.endswith(("q_proj", "k_proj"))))
    r["dd_share_mlp"] = float(sum(v["dd_share_sub"] + v["dd_share_rem"] for k, v in by.items() if ".mlp." in k))
    # per-variable kept share distribution over K & changed (dd-weighted quartiles)
    ks = np.where(dd[sel] > 0, expl[sel] / np.where(dd[sel] > 0, dd[sel], 1), 0)
    w = dd[sel] / tot
    o = np.argsort(ks)
    cw = np.cumsum(w[o])
    r["kept_dd_weighted_median"] = float(ks[o][np.searchsorted(cw, 0.5)])
    r["kept_frac_vars_over_0.5"] = float((ks > 0.5).mean())
    r["dd_share_top10_vars"] = float(np.sort(w)[::-1][:10].sum())
    r["dd_share_top100_vars"] = float(np.sort(w)[::-1][:100].sum())
    dW = torch.load(E62 / f"{st}.pt", map_location="cpu", weights_only=True)
    r["dW_L2"] = dw_parts(dW)
    allf = sum(float((v.double() ** 2).sum()) for v in dW.values())
    r["dW_L2_attn_share_of_all"] = sum(r["dW_L2"][k]["frob2"] for k in L2) / allf
    h4 = sum(r["dW_L2"][k]["frob2_head4"] for k in L2)
    r["dW_head4_share_of_all"] = h4 / allf
    r["dW_head4_share_of_L2_attn"] = h4 / sum(r["dW_L2"][k]["frob2"] for k in L2)
    r["dW_head4_qk_share"] = sum(r["dW_L2"][k]["frob2_head4"] for k in L2[:2]) / h4
    r["dW_by_layer_kind"] = {f"{l}.{kind}": sum(float((dW[k].double() ** 2).sum()) for k in keys if k.startswith(f"h.{l}.") and kind in k) / allf
                             for l in range(4) for kind in ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")}  # fmt: skip
    res["states"][st] = r
    print(st, f"E_b {eb:.4f} (item {item['E_b_K']:.4f})", f"head0.5 inK {r['head_0.5']['n_in_K']}/{r['head_0.5']['n']}",
          f"dd share {r['head_0.5']['dd_share_of_K']:.4f}", f"QK {r['dW_head4_qk_share']:.3f}", flush=True)  # fmt: skip
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
