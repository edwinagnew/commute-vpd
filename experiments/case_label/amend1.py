"""E112 (case_label) Amendment 1, 2026-11-06 (`decisions.md` row "E112 Amendment 1", user-approved): stage 2's causal check
P3 re-read on the four saved states with a stronger removal. Nothing else is re-read: possession comes from stage0.json,
P1 and P2 from stage2.json. Rules: results/predictions.json's, unchanged; this amendment's settings and expected values in
results/predictions_amend1.json (the run refuses unless both are committed unchanged). One cluster debug job, every
scored number printed and results/amend1.json written after every step:

  1  bases per block b = l*..3 (block b's output residual, readout.RESID): (a) stage 0's previous-token case probe
     (probes.pt), (b) an own-token case probe, the same ridge on the same Pile windows read at the word's own position
     HALF - 1; the pair orthonormalised per block. A seeded random rank-2 basis per block for the do-nothing. Beside: the
     own-token probe's held-out accuracy per block and the cosine of the two directions.
  2  W0 under each removal (every position, every block from l*): next-token loss change, label margin change per task.
  3  the do-nothing share on every state first and alone (exit 1 if one is over the bar), then P3 (both directions) and,
     beside, P3 with (a) alone per state; the verdict by E112's filed rule with the new P3.

    sbatch cluster/case_label_amend1.sbatch
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import Tensor

from commute import case_label as CL
from commute import keydep as KD
from commute import readout as RO
from commute import removal as RM
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

E = load_experiment("case_label", "run")  # E112's reads (label_diff, margin, ys, next_ce, say) and paths
RESULTS, LARGE, HALF, L, log = E.RESULTS, E.LARGE, E.HALF, E.L, E.log
PRED, PRED_A, OUT = RESULTS / "predictions.json", RESULTS / "predictions_amend1.json", RESULTS / "amend1.json"
Bases = dict[int, Tensor]  # block -> orthonormal basis (d, r)


def dump(res: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(jsonable(res), indent=1))


@torch.no_grad()
def resid_pos(net: ST.Net, toks: Tensor, pos: int, batch: int = 64) -> Tensor:
    """(4, N, d) fp32 CPU: block outputs at position `pos`, W0."""
    return torch.cat([RO.resid(net, toks[s : s + batch, :L])[1][:, :, pos].cpu() for s in range(0, len(toks), batch)], 1)


def strip(bases: Bases) -> ST.Edit:
    """Remove each block's basis from that block's output residual at every position."""
    return {RO.RESID[b]: (lambda x, u=u: RM.remove(x, u.to(x.device, x.dtype))) for b, u in bases.items()}


def bases(net: ST.Net, t: dict[str, Any], l_star: int, lam: float, seed: int) -> tuple[dict[str, Bases], dict[str, Any]]:
    """The three removals per block from l* ("both", "prev" = (a) alone, "random") and, beside, the own-token probe's
    held-out accuracy at every block (score > 0.5, as stage 0 reads its probe) and the two directions' cosine."""
    pr = torch.load(LARGE / "probes.pt", weights_only=False)
    ptoks, W = pr["pile_tokens"], pr["case"][0]
    y = torch.tensor([t["y"]["real"][int(w)] for w in ptoks[:, HALF - 1]])  # the word itself is the capitalised member
    h, hh, yh = resid_pos(net, ptoks, HALF - 1), resid_pos(net, t["held"], HALF - 1), E.ys(t, "real", "held")
    own = [RM.fit_ridge(h[i], y, lam) for i in range(len(h))]
    blocks = range(l_star, len(RO.RESID))
    out = {"both": {b: CL.pair_basis(W[b], own[b][0]) for b in blocks}, "prev": {b: RM.unit_basis(W[b]) for b in blocks},
           "random": {b: RM.random_basis(W.shape[1], 2, [seed, b]) for b in blocks}}  # fmt: skip
    info = {"blocks": list(blocks), "own_pile_share": float(y.double().mean()),
            "own_acc_heldout": [float((((hh[i].double() @ w) + b > 0.5).long() == yh).double().mean()) for i, (w, b) in enumerate(own)],
            "cos_prev_own": [float(W[i] @ own[i][0] / (W[i].norm() * own[i][0].norm())) for i in range(len(own))]}  # fmt: skip
    return out, info


def w0_reads(net: ST.Net, t: dict[str, Any], edits: dict[str, ST.Edit]) -> dict[str, Any]:
    """W0 under each removal: next-token loss at HALF on the probe's Pile windows (as stage 0), label margin on the fit windows."""
    ptoks = torch.load(LARGE / "probes.pt", weights_only=False)["pile_tokens"]
    ce0, d0 = E.next_ce(net, ptoks), E.label_diff(net, t["fit"], t["ab"])
    out: dict[str, Any] = {"next_ce_w0": ce0}
    for name, e in edits.items():
        de = E.label_diff(net, t["fit"], t["ab"], edit=e)
        out[name] = {"next_ce_change": E.next_ce(net, ptoks, e) - ce0,
                     **{f"margin_change_{k}": float((E.margin(de, E.ys(t, k, "fit")) - E.margin(d0, E.ys(t, k, "fit"))).mean()) for k in E.TASKS}}  # fmt: skip
    return out


def share(net: ST.Net, t: dict[str, Any], task: str, dW: ST.DW, edit: ST.Edit, floor: float) -> dict[str, float]:
    """Stage 2's P3 with another removal: on the state's own task's fit windows, the gain G = mean(m_state - m_W0) and the
    share mean(m_state - m_removed) / G (nan when G <= floor; not clipped, so > 1 or < 0 is reported as is)."""
    yf = E.ys(t, task, "fit")
    m0, ms = E.margin(E.label_diff(net, t["fit"], t["ab"]), yf), E.margin(E.label_diff(net, t["fit"], t["ab"], dW), yf)
    mr = E.margin(E.label_diff(net, t["fit"], t["ab"], dW, edit), yf)
    return {"gain": float((ms - m0).mean()), "share": CL.removed_share(ms, mr, m0, floor)}


def main() -> None:
    require_filed(PRED)
    require_filed(PRED_A)
    P, A = json.loads(PRED.read_text()), json.loads(PRED_A.read_text())
    rules, s2 = P["rules"], json.loads((RESULTS / "stage2.json").read_text())["states"]
    l_star = json.loads((RESULTS / "stage0.json").read_text())["l_star"]
    states = json.loads((RESULTS / "stage1.json").read_text())["states"]
    assert l_star is not None and sorted(states) == sorted(s2), "stages 0-2 are not complete"
    for name, s in states.items():
        assert KD.sha256(E.HERE / s["path"]) == s["sha256"], f"{name}: the saved state's sha256 differs from stage1.json"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    t = torch.load(LARGE / "task.pt", weights_only=False)
    B, info = bases(net, t, l_star, P["settings"]["ridge_lambda"], A["settings"]["random_seed"])
    res: dict[str, Any] = {"l_star": l_star, "bases": info}
    E.say("bases", info)
    dump(res)
    edits = {k: strip(b) for k, b in B.items()}
    res["w0"] = w0_reads(net, t, edits)
    E.say("W0 under each removal", res["w0"])
    dump(res)
    dWs = {n: {k: v.float() for k, v in torch.load(E.HERE / s["path"], weights_only=False).items()} for n, s in states.items()}
    res["do_nothing"] = {}
    for name, dW in dWs.items():  # control first and alone
        res["do_nothing"][name] = share(net, t, name.split("_")[0], dW, edits["random"], rules["gain_floor"])
        E.say(f"do-nothing {name}", res["do_nothing"][name])
        dump(res)
    if not CL.do_nothing_ok([s["share"] for s in res["do_nothing"].values()], rules["do_nothing_max"]):
        res["stop"] = f"the random rank-2 removal's share is over {rules['do_nothing_max']} on a state: no state is read"
        log(f"[stop] {res['stop']}")
        dump(res)
        hard_exit(1)
    res["states"] = {}
    for name, dW in dWs.items():
        task, p1, p2 = name.split("_")[0], s2[name]["P1"], s2[name]["P2"]
        both, prev = (share(net, t, task, dW, edits[k], rules["gain_floor"]) for k in ("both", "prev"))
        s = res["states"][name] = {"P1": p1, "P2": p2, "gain": both["gain"], "P3": both["share"], "P3_no_gain": not both["gain"] > rules["gain_floor"],
                                   "P3_prev_alone": prev["share"], "P3_stage2": s2[name]["P3"]}  # fmt: skip
        s["meets"] = CL.meets(p1, p2, s["P3"], rules["real"])
        s["verdict"] = CL.verdict(l_star, p1, p2, s["P3"], rules["real"])
        E.say(f"reads {name}", s)
        dump(res)
    log("=== amend1 done, exit 0 ===")
    hard_exit(0)


if __name__ == "__main__":
    main()
