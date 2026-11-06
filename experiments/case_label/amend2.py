"""E112 (case_label) Amendment 2, the last, 2026-11-06 (`decisions.md` row "E112 Amendment 2", user-approved): stage 2's
causal check P3 re-read on the four saved states with full linear concept erasure in place of Amendment 1's rank-2
removal. Nothing else is re-read: possession comes from stage0.json, P1 and P2 from stage2.json. Rules:
results/predictions.json's, unchanged; settings and expected values in results/predictions_amend2.json (the run refuses
unless it, predictions.json and predictions_amend1.json are committed unchanged). One cluster debug job, every number
printed and results/amend2.json written after every step:

  1  INLP (Ravfogel et al. 2020) per block 0..3 and per label: the previous-token case at the trigger HALF (stage 0's
     probe) and the own-token case at the word HALF - 1 (Amendment 1's); stage 0's ridge fitted on the even half of the
     saved Pile windows, read on the held-out-pair task windows, until accuracy <= the target or the cap. The two labels'
     bases merged per block (QR, near-dependent columns dropped). A seeded random basis of the same rank per block.
  2  Beside, no rule: a refit at block 3 (both labels, odd half of the Pile windows) at W0 and under the erasure.
  3  Beside, no rule: W0 under each removal (erasure at blocks l*..3, the same with block 0, random).
  4  The do-nothing share on every state first and alone (exit 1 if one is over the bar), then P3 per state under the
     erasure (beside: with block 0 too) and the verdict by E112's filed rule.

    sbatch cluster/case_label_amend2.sbatch
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

A1 = load_experiment("case_label", "amend1")  # Amendment 1's reads (resid_pos, strip, w0_reads, share)
E = A1.E  # E112's run.py (paths, ys, say)
RESULTS, LARGE, HALF, L, log = E.RESULTS, E.LARGE, E.HALF, E.L, E.log
PRED, PRED_A1, PRED_A2 = RESULTS / "predictions.json", RESULTS / "predictions_amend1.json", RESULTS / "predictions_amend2.json"
OUT = RESULTS / "amend2.json"
LABELS = {"prev": HALF, "own": HALF - 1}  # label -> window position read; both label the word at HALF - 1's case
Bases = dict[int, Tensor]  # block -> orthonormal basis (d, r)


def dump(res: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(jsonable(res), indent=1))


def pile(t: dict[str, Any]) -> tuple[Tensor, Tensor]:
    """The saved Pile windows (stage 0's probe windows) and their label: the word at HALF - 1 is the capitalised member."""
    ptoks = torch.load(LARGE / "probes.pt", weights_only=False)["pile_tokens"]
    return ptoks, torch.tensor([t["y"]["real"][int(w)] for w in ptoks[:, HALF - 1]])


def erasure(net: ST.Net, t: dict[str, Any], lam: float, S: dict[str, Any]) -> tuple[Bases, dict[str, Any]]:
    """INLP per block and label (fit: even Pile windows; read: held-out-pair windows), merged per block."""
    ptoks, y = pile(t)
    fit, yh = torch.arange(0, len(ptoks), 2), E.ys(t, "real", "held")
    info: dict[str, Any] = {"fit_case_share": float(y[fit].double().mean()), "held_case_share": float(yh.double().mean()), "blocks": {}, "curves": {}}
    per: dict[int, list[Tensor]] = {b: [] for b in range(len(RO.RESID))}
    for lab, pos in LABELS.items():
        h, hh = A1.resid_pos(net, ptoks[fit], pos).to(net.device), A1.resid_pos(net, t["held"], pos).to(net.device)
        for b, qs in per.items():
            q, accs, ok = CL.inlp(h[b], y[fit], hh[b], yh, lam, S["target_accuracy"], S["max_directions"])
            qs.append(q.cpu())
            info["blocks"].setdefault(b, {})[lab] = {"n_directions": q.shape[1], "acc_before": accs[0], "final_acc": accs[-1],
                                                     "status": "reached" if ok else "erasure incomplete"}  # fmt: skip
            info["curves"].setdefault(b, {})[lab] = accs
    B = {b: CL.merge_bases(qs, S["merge_tol"]) for b, qs in per.items()}
    for b, u in B.items():
        info["blocks"][b] |= {"rank": u.shape[1], "dropped": sum(q.shape[1] for q in per[b]) - u.shape[1]}
    return B, info


def removals(B: Bases, l_star: int, seed: int) -> dict[str, ST.Edit]:
    """The erasure at blocks l*..3 (scored), the same with block 0 (beside) and a seeded random basis of the same rank
    per block at l*..3 (do-nothing). A block of rank 0 is not edited."""
    on = {b: u for b, u in B.items() if b >= l_star and u.shape[1]}
    return {"erase": A1.strip(on), "erase_with_block0": A1.strip({b: u for b, u in B.items() if u.shape[1]}),
            "random": A1.strip({b: RM.random_basis(u.shape[0], u.shape[1], [seed, b]) for b, u in on.items()})}  # fmt: skip


@torch.no_grad()
def last_block(net: ST.Net, toks: Tensor, pos: int, edit: ST.Edit | None, batch: int = 64) -> Tensor:
    """(N, d) fp32 CPU: block 3's output at `pos`, captured after the edit."""
    return torch.cat([net.run(toks[s : s + batch, :L], edit, capture=True)[1][RO.RESID[-1]][:, pos].cpu() for s in range(0, len(toks), batch)])


def refit_block3(net: ST.Net, t: dict[str, Any], edit: ST.Edit | None, lam: float) -> dict[str, float]:
    """Beside: a fresh ridge per label at block 3 on the odd Pile windows, read on the held-out-pair windows (score > 0.5)."""
    ptoks, y = pile(t)
    odd, yh, out = torch.arange(1, len(ptoks), 2), E.ys(t, "real", "held"), {}
    for lab, pos in LABELS.items():
        w, b = RM.fit_ridge(last_block(net, ptoks[odd], pos, edit), y[odd], lam)
        out[lab] = float((((last_block(net, t["held"], pos, edit).double() @ w + b) > 0.5).long() == yh).double().mean())
    return out


def main() -> None:
    for p in (PRED, PRED_A1, PRED_A2):
        require_filed(p)
    P, A = json.loads(PRED.read_text()), json.loads(PRED_A2.read_text())
    rules, S, lam = P["rules"], A["settings"], P["settings"]["ridge_lambda"]
    s2 = json.loads((RESULTS / "stage2.json").read_text())["states"]
    a1 = json.loads((RESULTS / "amend1.json").read_text())["states"]
    l_star = json.loads((RESULTS / "stage0.json").read_text())["l_star"]
    states = json.loads((RESULTS / "stage1.json").read_text())["states"]
    assert l_star is not None and sorted(states) == sorted(s2) == sorted(a1), "stages 0-2 or Amendment 1 are not complete"
    for name, s in states.items():
        assert KD.sha256(E.HERE / s["path"]) == s["sha256"], f"{name}: the saved state's sha256 differs from stage1.json"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    t = torch.load(LARGE / "task.pt", weights_only=False)
    B, info = erasure(net, t, lam, S)
    torch.save(B, LARGE / "amend2_bases.pt")
    res: dict[str, Any] = {"l_star": l_star, "erasure": info}
    E.say("case shares (fit half, held-out windows)", {k: info[k] for k in ("fit_case_share", "held_case_share")})
    for b in info["blocks"]:
        E.say(f"erasure block {b}", info["blocks"][b])
        E.say(f"INLP held-out accuracy after 0, 1, ... removals, block {b}", info["curves"][b])
    dump(res)
    edits = removals(B, l_star, S["random_seed"])
    res["refit_block3"] = {"w0": refit_block3(net, t, None, lam), "erased": refit_block3(net, t, edits["erase"], lam)}
    E.say("refit at block 3, held-out accuracy", res["refit_block3"])
    dump(res)
    res["w0"] = A1.w0_reads(net, t, edits)
    E.say("W0 under each removal", res["w0"])
    dump(res)
    dWs = {n: {k: v.float() for k, v in torch.load(E.HERE / s["path"], weights_only=False).items()} for n, s in states.items()}
    res["do_nothing"] = {}
    for name, dW in dWs.items():  # control first and alone
        res["do_nothing"][name] = A1.share(net, t, name.split("_")[0], dW, edits["random"], rules["gain_floor"])
        E.say(f"do-nothing {name}", res["do_nothing"][name])
        dump(res)
    if not CL.do_nothing_ok([s["share"] for s in res["do_nothing"].values()], rules["do_nothing_max"]):
        res["stop"] = f"the same-rank random removal's share is over {rules['do_nothing_max']} on a state: no state is read"
        log(f"[stop] {res['stop']}")
        dump(res)
        hard_exit(1)
    res["states"] = {}
    for name, dW in dWs.items():
        task, p1, p2 = name.split("_")[0], s2[name]["P1"], s2[name]["P2"]
        er, er0 = (A1.share(net, t, task, dW, edits[k], rules["gain_floor"]) for k in ("erase", "erase_with_block0"))
        s = res["states"][name] = {"P1": p1, "P2": p2, "gain": er["gain"], "P3": er["share"], "P3_no_gain": not er["gain"] > rules["gain_floor"],
                                   "P3_with_block0": er0["share"], "P3_amend1": a1[name]["P3"], "P3_stage2": s2[name]["P3"]}  # fmt: skip
        s["meets"] = CL.meets(p1, p2, s["P3"], rules["real"])
        s["verdict"] = CL.verdict(l_star, p1, p2, s["P3"], rules["real"])
        E.say(f"reads {name}", s)
        dump(res)
    log("=== amend2 done, exit 0 ===")
    hard_exit(0)


if __name__ == "__main__":
    main()
