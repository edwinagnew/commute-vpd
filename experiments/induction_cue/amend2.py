"""E62 Amendment 2: is the mask's gain on the copy map carried by the same old head as the target model's gain? A read of
E62's saved step-30 states on its 200 read windows and the same masks; no training, nothing in results/large written or
deleted. Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md, "Amendment 2 (2026-11-02)". Rules and expected values:
results/predictions_amend2.json (committed first; the run refuses otherwise; never edited), `commute.copying`.

  1 control, alone: repeated-sequence top-1 with (2,4) and (3,5) off through the masked forward, all gates on. Exit 1 on a miss.
  2 target-model share under (2,4) equals run.json, the zero update reads 0. Exit 1 on a miss.
  3 share of the masked gain L01 - L00 lost under (2,4) and (3,5), the masks read with every head on. Exit 1 if the cells
    with the head on miss run.json or the zero update's masked gain under the head is not 0.
  4 W0's own fall under (2,4), the state's own fall as a share of its gain.
  5 target-model share and W0's top-1 under (2,4) + (1,1).
  6 the pair-broken gain lost under (2,4), target model.
  7 r with the update scaled to the other map's size, same seed; alpha = 1 reproduces run.json.
Means: heads.pt everywhere, W0's target-model means (plan). results/amend2.json after every row; every scored number printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor

from commute import anchors as AN
from commute import copying as CP
from commute import elicit as EL
from commute import keydep as KD
from commute import steering as ST
from commute.data import LAST
from commute.faithfulness import model_state
from commute.local_faith import masked_lo_windows, route_cells
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, BASE, RUN, OUT = RESULTS / "predictions_amend2.json", RESULTS / "predictions.json", RESULTS / "run.json", RESULTS / "amend2.json"
log = KD.log
Heads = list[CP.Head]
NAMES = [f"{m}_s{s}" for m in ("copy", "lookup") for s in (0, 1, 2)]


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))


def stop(out: dict[str, Any], why: str) -> NoReturn:
    out["stopped"] = why
    dump(out)
    sys.exit(f"STOP: {why}: tell the lead")


def note(out: dict[str, Any], key: str, rec: dict[str, Any]) -> None:
    out[key] = rec
    dump(out)
    log(f"[{key}] {json.dumps(jsonable(rec))}")


@dataclass
class Reads:
    ctx: KD.Ctx
    half: int
    n: int
    means: Tensor
    n_boot: int
    W: dict[str, Tensor]  # map -> read windows, target in the last column
    broken: Tensor
    dws: dict[str, ST.DW]
    run: dict[str, Any]  # run.json

    @property
    def zero(self) -> ST.DW:
        return {k: torch.zeros_like(x).cpu() for k, x in self.ctx.net.W0.items()}

    def lo(self, w: Tensor, heads: Heads, dW: ST.DW | None = None) -> Tensor:
        """Target model, per window."""
        with CP.ablated(self.ctx.net, heads, self.means):
            return CP.read_lo(self.ctx.net, w, self.half, self.n, dW)

    def gates(self, w: Tensor, dW: ST.DW | None = None) -> list[dict[str, Tensor]]:
        """Γ's masks with every head on, at W0 or at the state as `route_split` reads them."""
        m = self.ctx.net.model
        if dW is None:
            return route_cells(m, w, self.half, LAST, KD.BATCH, self.n)["gates"]
        target = {k: x + dW[k].to(x.device, x.dtype) for k, x in self.ctx.net.W0.items()}
        with model_state(m, target=target, components=AN.lifted(m, self.ctx.lift, dW)):
            return route_cells(m, w, self.half, LAST, KD.BATCH, self.n)["gates"]

    def masked(self, w: Tensor, gates: list[dict[str, Tensor]], heads: Heads) -> Tensor:
        """Masked forward, original subcomponents, Delta out, per window. All 24 matrices are replaced, so the target weights
        are not read (row 3's re-read against L01 checks it)."""
        with CP.ablated(self.ctx.net, heads, self.means):
            return masked_lo_windows(self.ctx.net.model, w, self.half, LAST, gates, KD.BATCH, self.n)

    def share(self, d: Tensor, d_abl: Tensor) -> tuple[float, float, float, float]:
        G, G_abl = float(d.mean()), float(d_abl.mean())
        return G, G_abl, CP.share_lost(G, G_abl), CP.share_sd(d, d_abl, self.n_boot, 0)


def map_of(name: str) -> str:
    return name.split("_")[0]


def row1(R: Reads, S: dict[str, Any], H: Heads, ref: Heads, th: dict[str, Any], out: dict[str, Any]) -> None:
    nets = {"target": R.ctx.net, "masked": CP.AllGatesOn(R.ctx.net.model)}
    rec: dict[str, Any] = {}
    for fwd, net in nets.items():
        rec[fwd] = {}
        for label, heads in (("base", []), ("H", H), ("ref", ref)):
            with CP.ablated(net, heads, R.means):
                rec[fwd][label] = CP.induction(net, S["induction"])["top1_second"]
    rec["status"] = CP.control_masked(rec["masked"]["base"], rec["masked"]["H"], rec["masked"]["ref"], th)
    note(out, "row1_control", rec)
    if rec["status"] != "pass":
        stop(out, f"row 1: {rec['status']}")


def row2(R: Reads, H: Heads, th: dict[str, Any], out: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
    lo0 = {m: {"on": R.lo(w, []), "off": R.lo(w, H)} for m, w in R.W.items()}
    rec: dict[str, Any] = {"zero_max_abs": {m: float((R.lo(w, H, R.zero) - lo0[m]["off"]).abs().max()) for m, w in R.W.items()}, "states": {}}
    for name, dW in R.dws.items():
        m = map_of(name)
        d, d_abl = R.lo(R.W[m], [], dW) - lo0[m]["on"], R.lo(R.W[m], H, dW) - lo0[m]["off"]
        G, G_abl, s, sd = R.share(d, d_abl)
        rec["states"][name] = {"G": G, "G_abl": G_abl, "share": s, "share_sd": sd, "diff_run": abs(s - R.run["states"][name]["share"])}
    note(out, "row2_reread", rec)
    if any(v != 0.0 for v in rec["zero_max_abs"].values()) or not all(r["diff_run"] <= th["reread_tol"] for r in rec["states"].values()):
        stop(out, "row 2: the target model's share does not re-read or the zero update is not 0")
    return lo0


def row3(R: Reads, H: Heads, ref: Heads, th: dict[str, Any], out: dict[str, Any]) -> None:
    g0 = {m: R.gates(w) for m, w in R.W.items()}
    L00 = {m: {"on": R.masked(w, g0[m], []), "H": R.masked(w, g0[m], H), "ref": R.masked(w, g0[m], ref)} for m, w in R.W.items()}
    gz = R.gates(R.W["copy"], R.zero)
    rec: dict[str, Any] = {"L00": {m: {k: float(v.mean()) for k, v in x.items()} for m, x in L00.items()}, "states": {},
                           "zero_masked_gain_H_max_abs": float((R.masked(R.W["copy"], gz, H) - L00["copy"]["H"]).abs().max())}
    b0 = R.gates(R.broken)
    rec["lean_masked"] = {k: float((L00["copy"][k] - R.masked(R.broken, b0, h)).mean()) for k, h in (("on", []), ("H", H))}
    bad = rec["zero_masked_gain_H_max_abs"] != 0.0
    for name, dW in R.dws.items():
        m, cells = map_of(name), R.run["states"][name]["cells"]
        g1 = R.gates(R.W[m], dW)
        L01 = {"on": R.masked(R.W[m], g1, []), "H": R.masked(R.W[m], g1, H), "ref": R.masked(R.W[m], g1, ref)}
        d = L01["on"] - L00[m]["on"]
        M, M_abl, s, sd = R.share(d, L01["H"] - L00[m]["H"])
        t = out["row2_reread"]["states"][name]["share"]
        rec["states"][name] = {"L01": float(L01["on"].mean()), "L01_H": float(L01["H"].mean()), "M": M, "M_abl": M_abl, "share": s, "share_sd": sd, "target_share": t,
                               "own_fall": float((L01["on"] - L01["H"]).mean()) / M if M > 0 else float("nan"),
                               "ref_share": R.share(d, L01["ref"] - L00[m]["ref"])[2], "label": CP.masked_class(M, s, t, th)}
        bad = bad or max(abs(rec["states"][name]["L01"] - cells["L01"]), abs(rec["L00"][m]["on"] - cells["L00"])) > th["reread_tol"]
        log(f"[row3 {name}] {json.dumps(jsonable(rec['states'][name]))}")
    rec["copy_reading"] = CP.masked_reading([rec["states"][n]["label"] for n in NAMES[:3]], [rec["states"][n]["ref_share"] for n in NAMES[:3]], th)
    rec["lookup_reading"] = CP.agree([rec["states"][n]["label"] for n in NAMES[3:]], th)
    note(out, "row3_masked", rec)
    if bad:
        stop(out, "row 3: the cells with the head on do not re-read or the zero update's masked gain under the head is not 0")


def row4(R: Reads, H: Heads, lo0: dict[str, dict[str, Tensor]], th: dict[str, Any], out: dict[str, Any]) -> None:
    base = out["row2_reread"]["states"]
    falls = {m: float((v["on"] - v["off"]).mean()) for m, v in lo0.items()}
    rec: dict[str, Any] = {"lo0": {m: float(v["on"].mean()) for m, v in lo0.items()}, "lo0_H": {m: float(v["off"].mean()) for m, v in lo0.items()}, "w0_fall": falls, "states": {}}
    rec["bracket"] = CP.bracket_class(falls["copy"], th)
    br = R.broken
    rec["lean_target"] = {"on": float((lo0["copy"]["on"] - R.lo(br, [])).mean()), "H": float((lo0["copy"]["off"] - R.lo(br, H)).mean())}
    for name, b in base.items():
        own = (b["G"] - b["G_abl"] + falls[map_of(name)]) / b["G"] if b["G"] > 0 else float("nan")
        rec["states"][name] = {"filed_share": b["share"], "own_fall": own, "label": CP.own_fall_class(b["G"], own, b["share"], th)}
    rec["copy_reading"] = CP.agree([rec["states"][n]["label"] for n in NAMES[:3]], th)
    note(out, "row4_w0_fall", rec)


def row5(R: Reads, S: dict[str, Any], both: Heads, lo0: dict[str, dict[str, Tensor]], th: dict[str, Any], out: dict[str, Any]) -> None:
    with CP.ablated(R.ctx.net, both, R.means):
        top1 = CP.induction(R.ctx.net, S["induction"])["top1_second"]
    off0 = {m: R.lo(w, both) for m, w in R.W.items()}
    rec: dict[str, Any] = {"top1_both": top1, "top1_both_ok": top1 <= th["top1_both_max"], "states": {}}
    for name, dW in R.dws.items():
        m = map_of(name)
        G, G_abl, s, sd = R.share(R.lo(R.W[m], [], dW) - lo0[m]["on"], R.lo(R.W[m], both, dW) - off0[m])
        single = out["row2_reread"]["states"][name]["share"]
        lab = CP.two_head_class(G, s, single, th) if m == "copy" else CP.lookup_two_head(G, s, th)
        rec["states"][name] = {"G_abl": G_abl, "share": s, "share_sd": sd, "single": single, "label": lab}
    rec["copy_reading"] = CP.agree([rec["states"][n]["label"] for n in NAMES[:3]], th)
    rec["lookup_reading"] = CP.agree([rec["states"][n]["label"] for n in NAMES[3:]], th)
    note(out, "row5_two_heads", rec)


def row6(R: Reads, H: Heads, th: dict[str, Any], out: dict[str, Any]) -> None:
    on0, off0 = R.lo(R.broken, []), R.lo(R.broken, H)
    rec: dict[str, Any] = {"states": {}}
    for name in NAMES[:3]:
        dW = R.dws[name]
        G, G_abl, s, sd = R.share(R.lo(R.broken, [], dW) - on0, R.lo(R.broken, H, dW) - off0)
        t = out["row2_reread"]["states"][name]
        rec["states"][name] = {"G_broken": G, "G_broken_abl": G_abl, "share": s, "share_sd": sd, "intact_share": t["share"], "surviving_intact": t["G_abl"],
                               "diff_run": abs(G - R.run["states"][name]["gains_by_step"]["30"]["broken"]), "label": CP.broken_class(G, s, t["share"], th)}
    rec["reading"] = CP.agree([r["label"] for r in rec["states"].values()], th)
    rec["reread_ok"] = all(r["diff_run"] <= th["reread_tol"] for r in rec["states"].values())
    note(out, "row6_broken", rec)


def row7(R: Reads, th: dict[str, Any], out: dict[str, Any]) -> None:
    ctx, stored = R.ctx, R.run["states"]
    base = {m: route_cells(ctx.net.model, w, R.half, LAST, KD.BATCH, R.n) for m, w in R.W.items()}
    route = lambda name, a: KD.route_windows(ctx, R.W[map_of(name)], R.half, LAST, {k: a * v for k, v in R.dws[name].items()}, base[map_of(name)], R.n)
    rec: dict[str, Any] = {"states": {}, "repeat_floor": abs(route("copy_s0", 1.0)["r"] - route("copy_s0", 1.0)["r"])}
    for name in NAMES:
        m, s = name.split("_")
        size = EL.summed_norm(R.dws[f"{'lookup' if m == 'copy' else 'copy'}_{s}"]) / EL.summed_norm(R.dws[name])
        one, swapped = route(name, 1.0), route(name, size)
        r_sw = swapped["r"] if CP.gain_ok(swapped["G_logodds"], th) is None else float("nan")
        rec["states"][name] = {"alpha": size, "r_alpha1": one["r"], "r_stored": stored[name]["r"], "diff": abs(one["r"] - stored[name]["r"]), "r_swapped": r_sw, "G_swapped": swapped["G_logodds"]}
        log(f"[row7 {name}] {json.dumps(jsonable(rec['states'][name]))}")
    mean = lambda key, names: sum(rec["states"][n][key] for n in names) / len(names)
    rec["gap"], rec["gap_swapped"] = mean("r_stored", NAMES[:3]) - mean("r_stored", NAMES[3:]), mean("r_swapped", NAMES[:3]) - mean("r_swapped", NAMES[3:])
    rec["reproduced"] = all(r["diff"] <= th["reread_tol"] for r in rec["states"].values())
    rec["reading"] = CP.size_reading(rec["gap"], rec["gap_swapped"], rec["reproduced"])
    rec["near_half"] = abs(rec["gap_swapped"] - rec["gap"] / 2) <= th["r_noise"]
    note(out, "row7_size", rec)


def load(a: argparse.Namespace) -> tuple[Reads, dict[str, Any], dict[str, Any], dict[str, Any]]:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    if OUT.exists():
        sys.exit(f"{OUT} exists: move it aside, this run deletes nothing")
    P, S, run = json.loads(PRED.read_text()), json.loads(BASE.read_text())["settings"], json.loads(RUN.read_text())
    large = a.large.resolve()
    files = {"windows": run["windows_sha256"], **{n: run["states"][n]["sha256"] for n in NAMES}}
    for f, sha in files.items():
        if KD.sha256(large / f"{f}.pt") != sha:
            sys.exit(f"{f}.pt is not the one run.json recorded")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, large)
    v = torch.load(large / "windows.pt", weights_only=False)
    half, n = S["half"], 2 * S["half"] + 1
    W = {m: CP.with_target(v["read"], v["target"][m]["read"]).tokens for m in ("copy", "lookup")}
    b = CP.broken_pair(v["read"].tokens, v["p_read"], v["pools"]["keys"], half, S["control_seed"])
    means = torch.load(large / "heads.pt", weights_only=False)["means"].to(ctx.net.device)
    dws = {name: torch.load(large / f"{name}.pt", weights_only=True, map_location="cpu") for name in NAMES}
    R = Reads(ctx, half, n, means, S["n_boot"], W, torch.cat([b, v["target"]["copy"]["read"][:, None]], 1), dws, run)
    head = git("rev-parse", "HEAD").stdout.strip()
    return R, P, S, {"git_head": head, "predictions_sha256": KD.sha256(PRED), "host": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large")
    a = ap.parse_args()
    R, P, S, out = load(a)
    th = P["thresholds"]
    as_heads = lambda key: [(int(x), int(y)) for x, y in P["settings"][key]]
    H, ref, both = as_heads("head"), as_heads("ref_head"), as_heads("two_heads")
    log(f"[host] {out['host']}")
    row1(R, S, H, ref, th, out)
    lo0 = row2(R, H, th, out)
    row3(R, H, ref, th, out)
    row4(R, H, lo0, th, out)
    row5(R, S, both, lo0, th, out)
    row6(R, H, th, out)
    row7(R, th, out)
    log(f"[verdict] copy {out['row3_masked']['copy_reading']} lookup {out['row3_masked']['lookup_reading']} own fall {out['row4_w0_fall']['copy_reading']} "
        f"two heads {out['row5_two_heads']['copy_reading']} broken {out['row6_broken']['reading']} size {out['row7_size']['reading']}")


if __name__ == "__main__":
    main()
