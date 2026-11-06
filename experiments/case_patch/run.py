"""E113 (case_patch), 2026-11-06: does patching the case readout flip the case-label fine-tune's answer? E112's causal
check read by patching (`decisions.md` row 81, "ablating or patching"), on E112's four saved states. Plan:
reports/plans/2026-11-06_plan_e113-case-patch.md. Rules: results/predictions.json (the run refuses unless it and E112's
predictions.json are committed unchanged); pure pieces in src/commute/case_patch.py. One cluster debug job, every number
printed and results/run.json written after every step:

  1  The two directions at block 1 (E112's previous-token probe u_p; the own-token probe u_o refitted as Amendment 1
     does) and their class means on E112's saved Pile windows at W0.
  2  Q0 on W0, first and alone: block 2's previous-token probe reads the patched case. Exit 1 if under the bar.
     Then Q2 (W0's next-token loss change under the patch), printed.
  3  The random-direction control on every state first (exit 1 if a draw flips more than the bar), then Q1 per state
     and the verdict by E112's filed rule with Q1 as the causal check; the numbers beside, no rule, last.

    sbatch cluster/case_patch.sbatch
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from commute import case_label as CL
from commute import case_patch as CP
from commute import keydep as KD
from commute import readout as RO
from commute import removal as RM
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.open_pair import jsonable

A1 = load_experiment("case_label", "amend1")  # E112 Amendment 1: resid_pos, and through it E112's run module
E = A1.E  # E112's run.py: ys, label reads, paths
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "run.json"
HALF, L, log = E.HALF, E.L, E.log
TRIG, WORD = HALF, HALF - 1  # the trigger on the task windows (the token after the word on the Pile windows); the word
STATES = ("real_s0", "real_s1", "twin_s0", "twin_s1")


@dataclass(frozen=True)
class Config:
    block: int
    q0_block: int
    n_random: int
    random_seed: int
    batch: int

    def __post_init__(self) -> None:
        assert self.q0_block == self.block + 1 and self.n_random >= 1 and self.batch >= 1


def dump(res: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(jsonable(res), indent=1))


# ----------------------------------------------------------------------------- model-side reads


def edit(C: Config, sites: list[CP.Site], rows: slice) -> ST.Edit | None:
    return {RO.RESID[C.block]: CP.edit_fn(sites, rows)} if sites else None


@torch.no_grad()
def diff(net: ST.Net, C: Config, toks: Tensor, ab: tuple[int, int], sites: list[CP.Site], dW: ST.DW | None = None) -> Tensor:
    """(N,) float64: logit A - logit B at the trigger, with the sites patched (none: unpatched)."""
    out = []
    with net.at(dW):
        for s in range(0, len(toks), C.batch):
            rows = slice(s, s + C.batch)
            lg = net.run(toks[rows, :L], edit(C, sites, rows))[0][:, HALF]
            out.append((lg[:, ab[0]] - lg[:, ab[1]]).double().cpu())
    return torch.cat(out)


@torch.no_grad()
def probe_class(net: ST.Net, C: Config, toks: Tensor, sites: list[CP.Site], w: Tensor, b: Tensor) -> Tensor:
    """(N,) long: E112's probe (w, b) read on block q0_block's output at the trigger, W0, sites patched (score > 0.5 = 1)."""
    out = []
    for s in range(0, len(toks), C.batch):
        rows = slice(s, s + C.batch)
        h = net.run(toks[rows, :L], edit(C, sites, rows), capture=True)[1][RO.RESID[C.q0_block]][:, HALF]
        out.append((h.double().cpu() @ w + b > 0.5).long())
    return torch.cat(out)


@torch.no_grad()
def next_ce(net: ST.Net, C: Config, toks: Tensor, sites: list[CP.Site]) -> float:
    """W0's mean next-token loss at HALF (the true token at HALF + 1), sites patched."""
    tot = 0.0
    for s in range(0, len(toks), C.batch):
        rows = slice(s, s + C.batch)
        lg = net.run(toks[rows], edit(C, sites, rows))[0][:, HALF]
        tot += float(F.cross_entropy(lg, toks[rows, HALF + 1].to(net.device), reduction="sum"))
    return tot / len(toks)


# ----------------------------------------------------------------------------- the directions and the sites


def directions(net: ST.Net, C: Config, t: dict[str, Any], lam: float) -> tuple[dict[int, Tensor], dict[int, tuple[float, float]], dict[str, Any]]:
    """u per position (TRIG: E112's previous-token probe; WORD: the own-token probe, refitted as Amendment 1 does) at
    block C.block, unit, and its class means over the saved Pile windows at W0; beside, the numbers that describe them."""
    pr = torch.load(E.LARGE / "probes.pt", weights_only=False)
    ptoks, W = pr["pile_tokens"], pr["case"][0]
    y = torch.tensor([t["y"]["real"][int(w)] for w in ptoks[:, WORD]])
    h = {p: A1.resid_pos(net, ptoks, p)[C.block] for p in (TRIG, WORD)}
    w_o, b_o = RM.fit_ridge(h[WORD], y, lam)
    hh = A1.resid_pos(net, t["held"], WORD)[C.block]
    u = {TRIG: RM.unit_basis(W[C.block])[:, 0], WORD: RM.unit_basis(w_o)[:, 0]}
    proj = {p: h[p].double() @ u[p].double() for p in u}
    mu = {p: CP.class_means(proj[p], y) for p in u}
    info = {"n_pile": len(y), "pile_case_share": float(y.double().mean()), "cos_up_uo": float(u[TRIG] @ u[WORD]),
            "own_acc_heldout": float((((hh.double() @ w_o) + b_o > 0.5).long() == E.ys(t, "real", "held")).double().mean()),
            **{f"{n}_{k}": v for n, p in (("u_p", TRIG), ("u_o", WORD)) for k, v in
               (("mu0", mu[p][0]), ("mu1", mu[p][1]), ("gap", mu[p][1] - mu[p][0]), ("pile_std", float(proj[p].std())))}}  # fmt: skip
    return u, mu, info


def sites(u: dict[int, Tensor], mu: dict[int, tuple[float, float]], y: Tensor, scale: float = 1.0, at: tuple[int, ...] = (TRIG, WORD),
          rand: dict[int, Tensor] | None = None) -> list[CP.Site]:  # fmt: skip
    """The patch at the positions `at` for windows of case y: targets mu_opp (scale 1) or past it; with `rand`, the
    control's step of the same size along each position's random direction."""
    return [CP.Site(p, u[p], CP.targets(y, mu[p], scale), None if rand is None else rand[p]) for p in at]


def random_dirs(C: Config, d: int, k: int) -> dict[int, Tensor]:
    return {p: RM.random_basis(d, 1, [C.random_seed, k, p])[:, 0] for p in (TRIG, WORD)}


# ----------------------------------------------------------------------------- the steps


def q0(net: ST.Net, C: Config, t: dict[str, Any], u: dict[int, Tensor], mu: dict[int, tuple[float, float]]) -> dict[str, Any]:
    """Q0 on W0's held-out windows: block q0_block's probe against the true and the flipped case, unpatched, patched, and
    under each random draw."""
    W, B = torch.load(E.LARGE / "probes.pt", weights_only=False)["case"]
    w, b, y, toks = W[C.q0_block], B[C.q0_block], E.ys(t, "real", "held"), t["held"]
    acc = lambda c: {"true": float((c == y).double().mean()), "flipped": float((c == 1 - y).double().mean())}
    rand = [acc(probe_class(net, C, toks, sites(u, mu, y, rand=random_dirs(C, u[TRIG].shape[0], k)), w, b)) for k in range(C.n_random)]
    return {"unpatched": acc(probe_class(net, C, toks, [], w, b)), "patched": acc(probe_class(net, C, toks, sites(u, mu, y), w, b)), "random": rand}


def q2(net: ST.Net, C: Config, t: dict[str, Any], u: dict[int, Tensor], mu: dict[int, tuple[float, float]]) -> dict[str, float]:
    ptoks = torch.load(E.LARGE / "probes.pt", weights_only=False)["pile_tokens"]
    y = torch.tensor([t["y"]["real"][int(w)] for w in ptoks[:, WORD]])
    ce0 = next_ce(net, C, ptoks, [])
    return {"next_ce_w0": ce0, "Q2": next_ce(net, C, ptoks, sites(u, mu, y)) - ce0,
            "random0": next_ce(net, C, ptoks, sites(u, mu, y, rand=random_dirs(C, u[TRIG].shape[0], 0))) - ce0}  # fmt: skip


def beside(net: ST.Net, C: Config, t: dict[str, Any], task: str, dW: ST.DW | None, u: dict[int, Tensor], mu: dict[int, tuple[float, float]],
           d0_fit: Tensor) -> dict[str, Any]:  # fmt: skip
    """No rule: unpatched accuracy, the held-out flip rate, each edit alone and the 2x shift."""
    ab, yf, yh = t["ab"], E.ys(t, "real", "fit"), E.ys(t, "real", "held")
    d0_held = diff(net, C, t["held"], ab, [], dW)
    fit = lambda ss: CP.flip_rate(d0_fit, diff(net, C, t["fit"], ab, ss, dW))
    return {"acc_fit": E.accuracy(d0_fit, E.ys(t, task, "fit")), "acc_held": E.accuracy(d0_held, E.ys(t, task, "held")),
            "flip_held": CP.flip_rate(d0_held, diff(net, C, t["held"], ab, sites(u, mu, yh), dW)),
            "flip_trigger_only": fit(sites(u, mu, yf, at=(TRIG,))), "flip_word_only": fit(sites(u, mu, yf, at=(WORD,))),
            "flip_2x": fit(sites(u, mu, yf, scale=2.0))}  # fmt: skip


def main() -> None:
    require_filed(PRED)
    require_filed(E.PRED)
    P, R112 = json.loads(PRED.read_text()), json.loads(E.PRED.read_text())
    C, R = Config(**P["settings"]), P["rules"]
    l_star = json.loads((E.RESULTS / "stage0.json").read_text())["l_star"]
    s1 = json.loads((E.RESULTS / "stage1.json").read_text())["states"]
    s2 = json.loads((E.RESULTS / "stage2.json").read_text())["states"]
    assert l_star == C.block and sorted(s1) == sorted(s2) == sorted(STATES), "E112 stages 0-2 are not as the plan read them"
    for name, s in s1.items():
        assert KD.sha256(E.HERE / s["path"]) == s["sha256"], f"{name}: the saved state's sha256 differs from stage1.json"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    t = torch.load(E.LARGE / "task.pt", weights_only=False)
    u, mu, info = directions(net, C, t, R112["settings"]["ridge_lambda"])
    res: dict[str, Any] = {"directions": info}
    E.say("directions", info)
    dump(res)
    res["Q0"] = q0(net, C, t, u, mu)
    E.say("Q0", res["Q0"])
    dump(res)
    if not res["Q0"]["patched"]["flipped"] >= R["Q0"]["min"]:
        res["stop"] = f"Q0 {res['Q0']['patched']['flipped']:.4f} < {R['Q0']['min']}: the patch does not reach block {C.q0_block}; no reading"
        log(f"[stop] {res['stop']}")
        dump(res)
        hard_exit(1)
    res["Q2"] = q2(net, C, t, u, mu)
    E.say("Q2", res["Q2"])
    dump(res)
    dWs = {n: {k: v.float() for k, v in torch.load(E.HERE / s1[n]["path"], weights_only=False).items()} for n in STATES}
    yf, ab, d0 = E.ys(t, "real", "fit"), t["ab"], {}
    res["control"] = {}
    for name, dW in dWs.items():  # the control first and alone, on every state
        d0[name] = diff(net, C, t["fit"], ab, [], dW)
        res["control"][name] = [CP.flip_rate(d0[name], diff(net, C, t["fit"], ab, sites(u, mu, yf, rand=random_dirs(C, u[TRIG].shape[0], k)), dW))
                                for k in range(C.n_random)]  # fmt: skip
        E.say(f"control {name}", res["control"][name])
        dump(res)
    if not CL.do_nothing_ok([r for v in res["control"].values() for r in v], R["Q1"]["control_max"]):
        res["stop"] = f"a random-direction patch flips more than {R['Q1']['control_max']} of a state's labels: no state is read"
        log(f"[stop] {res['stop']}")
        dump(res)
        hard_exit(1)
    bars = {**R112["rules"]["real"], "P3": R["Q1"]["real"]}
    res["states"] = {}
    for name, dW in dWs.items():
        q1 = CP.flip_rate(d0[name], diff(net, C, t["fit"], ab, sites(u, mu, yf), dW))
        p1, p2 = s2[name]["P1"], s2[name]["P2"]
        m = CL.meets(p1, p2, q1, bars)
        s = res["states"][name] = {"P1": p1, "P2": p2, "Q1": q1, "meets": {"P1": m["P1"], "P2": m["P2"], "Q1": m["P3"]},
                                   "verdict": CL.verdict(l_star, p1, p2, q1, bars)}  # fmt: skip
        E.say(f"reads {name}", s)
        dump(res)
    res["beside"] = {}
    for name, dW in (("W0", None), *dWs.items()):
        d = d0[name] if name in d0 else diff(net, C, t["fit"], ab, [], None)
        res["beside"][name] = beside(net, C, t, "real" if name == "W0" else name.split("_")[0], dW, u, mu, d)
        E.say(f"beside {name}", res["beside"][name])
        dump(res)
    log("=== E113 done, exit 0 ===")
    hard_exit(0)


if __name__ == "__main__":
    main()
