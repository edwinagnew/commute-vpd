"""E124 stage 0, alone, W0 only (run.py's docstring has the whole run). In this order, each step written to
results/stage0.json as it ends:

  a  the windows from the cached rows; the kept counts per gender are asserted before the model loads (exit 1);
  b  apparatus: E54 elicit seed 0's r through the data.LAST path within e54_r_tol of e54_r_ref, and each member's zero
     update with route cells exactly W0's (exit 1 on a miss);
  c  W0's label accuracy per member, part and gender; the two headroom gates; the flipped twin fit-window count;
  d  the possession probe and its control per block (P0, l*), its held-out accuracy per gender;
  e  W0 under the probe direction's and a random direction's removal at l*, with each direction's variance share.

`go` is true only if every gate passed; else exit 1 after the reads that can still be made.

    sbatch cluster/natural_tasks.sbatch 0
"""

from __future__ import annotations

from typing import Any

import torch
from run import (
    HALF,
    HERE,
    LARGE,
    MEMBERS,
    Config,
    L,
    ab_of,
    build,
    dump,
    finish,
    labels,
    log,
    say,
    setup,
)

from commute import case_label as CL
from commute import definition_reads as DFR
from commute import keydep as KD
from commute import label_reads as LR
from commute import natural_tasks as NT
from commute import removal as RM
from commute.copying import with_target
from commute.data import LAST
from commute.local_faith import route_cells

E54 = HERE.parent / "keydependent_elicit" / "results" / "large"
ALIVE = HERE.parent / "transport_test" / "results" / "large" / "alive.pt"


class Stop(Exception):
    pass


def stop(res: dict[str, Any], why: str) -> None:
    res["stop"] = why
    log(f"[stop] {why}")
    dump(0, res)
    raise Stop


def count_gate(C: Config, info: dict[str, Any], res: dict[str, Any]) -> None:
    need = {"train": C.n_train_windows, "fit": C.n_fit_windows, "held": C.n_heldout_windows}
    res["counts"] = {k: {**info["kept"][k], "need_per_gender": n // 2, "available": info["available"][k]} for k, n in need.items()}
    say("counts", res["counts"])
    short = [k for k, n in need.items() if not info["kept"][k]["female"] == info["kept"][k]["male"] == n // 2]
    if short:
        stop(res, f"kept windows under the filed counts in {short}")


def e54_control(ctx: KD.Ctx, P: dict[str, Any], res: dict[str, Any]) -> None:
    """E60's re-read of E54 elicit seed 0 through the per-window target path (E62's `path_control`, its `now` read)."""
    c = P["controls"]
    for f, sha in c["e54_sha256"].items():
        if KD.sha256(E54 / f"{f}.pt") != sha:
            stop(res, f"E54's {f} is not the one its run recorded")
    w = torch.load(E54 / "windows_elicit.pt", weights_only=False)
    n = w["heldout"].tokens.shape[1]
    toks = with_target(w["heldout"], w["table"][w["heldout"].tokens[:, HALF]]).tokens
    base = route_cells(ctx.net.model, toks, HALF, LAST, KD.BATCH, n)
    r = KD.route_windows(ctx, toks, HALF, LAST, torch.load(E54 / "elicit_s0.pt", weights_only=True), base, n)["r"]
    res["e54"] = {"r": r, "ref": c["e54_r_ref"], "tol": c["e54_r_tol"], "pass": abs(r - c["e54_r_ref"]) <= c["e54_r_tol"]}
    say("E54 elicit s0 r", res["e54"])
    if not res["e54"]["pass"]:
        stop(res, "E54 elicit seed 0's r is off its reference")


def route_x(t: dict[str, Any], member: str) -> torch.Tensor:
    """The member's fit windows with its target as a last column (data.LAST, ci_len L)."""
    toks, y = labels(t, member, "fit")
    return torch.cat([toks, LR.targets(y, ab_of(t, member))[:, None]], 1)


def zero_controls(ctx: KD.Ctx, t: dict[str, Any], res: dict[str, Any]) -> None:
    zero = {k: torch.zeros_like(x).cpu() for k, x in ctx.net.W0.items()}
    res["route_w0"], res["zero_update"] = {}, {}
    for m in MEMBERS:
        x = route_x(t, m)
        base = route_cells(ctx.net.model, x, HALF, LAST, KD.BATCH, L)
        res["route_w0"][m] = {k: base[k] for k in ("T", "U", "L", "p_T")}
        c = KD.route_windows(ctx, x, HALF, LAST, zero, base, L)["cells"]
        res["zero_update"][m] = {"L01-L00": c["L01"] - c["L00"], "L10-L00": c["L10"] - c["L00"], "T1-T0": c["T1"] - c["T0"]}
        say(f"route W0 / zero update {m}", {**res["route_w0"][m], **res["zero_update"][m]})
        dump(0, res)
        if any(v != 0.0 for v in res["zero_update"][m].values()):
            stop(res, f"the zero update's route cells are not W0's ({m})")


def w0_reads(net: Any, t: dict[str, Any], res: dict[str, Any]) -> list[str]:
    """W0's accuracy per member, part and gender (c: 1 male, 0 female); returns the headroom gates missed."""
    acc: dict[str, Any] = {}
    for m in MEMBERS:
        for part in ("train", "fit", "held"):
            toks, y = labels(t, m, part)
            right = ((DFR.label_diff(net, toks, ab_of(t, m), HALF) > 0).long() == y).double()
            c = t[part]["c"]
            acc.setdefault(m, {})[part] = {"all": float(right.mean()), "female": float(right[c == 0].mean()), "male": float(right[c == 1].mean())}
    res["w0_accuracy"] = acc
    res["flipped_twin_fit"] = int((labels(t, "twin", "fit")[1] != t["fit"]["c"]).sum())
    nat, new = acc["natural"]["train"]["all"], acc["new_word"]["train"]["all"]
    res["headroom"] = {"natural_train": nat, "new_word_train": new, "under_learn_bar": nat < 0.9, "over_new_word": nat > new}
    say("W0 accuracy", acc)
    say("headroom", {**res["headroom"], "flipped_twin_fit": res["flipped_twin_fit"]})
    return [k for k in ("under_learn_bar", "over_new_word") if not res["headroom"][k]]


def probes(net: Any, C: Config, t: dict[str, Any], prows: torch.Tensor, res: dict[str, Any]) -> dict[str, Any]:
    """P0: ridge on probe-row positions 3-15 tokens after a train-pool name (any next token; at most probe_cap per name;
    genders balanced), scored on the held-out-name task windows at the prediction position; control = a seeded bit
    per name (Hewitt-Liang)."""
    m = NT.pronoun_matches(prows, torch.tensor(list(t["pool"])), C.near, C.far, HALF, None)
    rec = NT.pronoun_records(m, prows, t["pool"])
    rec = NT.subset(rec, torch.isin(rec["content"], torch.tensor(t["train_names"])))
    rec = NT.cap_and_balance(rec, rec["content"], C.probe_cap, [(0, 0), (1, 1)], C.probe_seed)
    ptoks = NT.windows(prows, rec, HALF, extra=1)
    h, hh = DFR.resid_at(net, ptoks[:, :L], HALF), DFR.resid_at(net, t["held"]["toks"], HALF)
    ctrl = lambda r: torch.tensor([t["control"][int(n)] for n in r["content"]])
    out: dict[str, Any] = {"n_pile": len(ptoks), "pile_male_share": float(rec["c"].double().mean())}
    fitted: dict[str, Any] = {}
    for k, y, yh in (("case", rec["c"], t["held"]["c"]), ("control", ctrl(rec), ctrl(t["held"]))):
        W, B, out[f"acc_{k}"] = CL.value_probes(h, y, hh, yh, 2, C.ridge_lambda)
        fitted[k] = (W[:, 1].cpu(), B[:, 1].cpu())
    out["selectivity"] = [a - c for a, c in zip(out["acc_case"], out["acc_control"], strict=True)]
    pred = [((hh[i].double() @ fitted["case"][0][i]) + fitted["case"][1][i] > 0.5).long() for i in range(len(hh))]
    g = t["held"]["c"]
    out["acc_case_by_gender"] = {name: [float((p[g == v] == v).double().mean()) for p in pred] for name, v in (("female", 0), ("male", 1))}
    torch.save({**fitted, "pile_tokens": ptoks}, LARGE / "probes_pronoun.pt")
    return {**out, "h_probe": h}


def w0_removals(net: Any, C: Config, t: dict[str, Any], h: torch.Tensor, w: torch.Tensor, block: int) -> dict[str, Any]:
    """W0 under each removal at `block`: next-token loss on the probe's Pile windows, label margin per member on the fit
    windows, and each direction's share of the probe positions' residual variance at `block`."""
    ptoks = torch.load(LARGE / "probes_pronoun.pt", weights_only=False)["pile_tokens"]
    dirs = {"probe": RM.unit_basis(w), "random": RM.random_basis(w.shape[0], 1, [C.random_dir_seed])}
    ce0, out = DFR.next_ce(net, ptoks, HALF), {}
    for name, u in dirs.items():
        e = DFR.removal_edit(block, u, HALF)
        rec: dict[str, Any] = {"next_ce_change": DFR.next_ce(net, ptoks, HALF, e) - ce0, "variance_share": DFR.variance_share(h[block], u)}
        for m in MEMBERS:
            toks, y = labels(t, m, "fit")
            d0, de = DFR.label_diff(net, toks, ab_of(t, m), HALF), DFR.label_diff(net, toks, ab_of(t, m), HALF, edit=e)
            rec[f"margin_change_{m}"] = float((DFR.margin(de, y) - DFR.margin(d0, y)).mean())
        out[name] = rec
    return {"next_ce_w0": ce0, **out}


def stage0(C: Config, R: dict[str, Any], P: dict[str, Any]) -> int:
    res: dict[str, Any] = {"go": False}
    b = build(C)
    t, info = b["t"], b["info"]
    res["data"] = info
    say("data", {k: v for k, v in info.items() if k != "pool_words"})
    count_gate(C, info, res)  # before the model loads
    dump(0, res)
    ctx = KD.make_ctx("cuda", ALIVE, P, LARGE)
    e54_control(ctx, P, res)
    zero_controls(ctx, t, res)
    missed = [f"headroom {k}" for k in w0_reads(ctx.net, t, res)]
    dump(0, res)
    p0 = probes(ctx.net, C, t, b["probe_rows"], res)
    h = p0.pop("h_probe")
    res["P0"] = p0
    res["l_star"] = CL.first_passing_block(p0["acc_case"], p0["selectivity"], R["P0"]["accuracy"], R["P0"]["selectivity"])
    say("P0", {**p0, "l_star": res["l_star"]})
    dump(0, res)
    if res["l_star"] is None:
        res["verdict"] = "unreadable"
        missed.append("P0 fails at every block before the last: 'unreadable'")
    else:
        w = torch.load(LARGE / "probes_pronoun.pt", weights_only=False)["case"][0][res["l_star"]]
        res["w0_removal"] = w0_removals(ctx.net, C, t, h, w, res["l_star"])
        say("W0 removal at l*", res["w0_removal"])
    res["go"], res["missed"] = not missed, missed
    dump(0, res)
    if missed:
        log(f"[stop] {missed}: stages 1-3 do not run")
    return 0 if res["go"] else 1


def main() -> None:
    C, R, P = setup()
    try:
        code = stage0(C, R, P)
    except Stop:
        code = 1
    finish(0, code)


if __name__ == "__main__":
    main()
