"""E62 Amendment 3: does the head act through the masks? Amendment 2 held Γ's masks as read with the head on; this reads Γ again
with the head off. A read of E62's saved step-30 states on its 200 read windows; no training, nothing in results/large written or
deleted. Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md, "Amendment 3 (2026-11-02)". Rules and expected values:
results/predictions_amend3.json (committed first; the run refuses otherwise; never edited), `commute.copying`.

  1 control, alone, W0: `ablated` changes what Γ reads, as filed; the frozen masks reproduce amend2.json; the zero update's gain with
    Γ recomputed is 0. W0 with the head off in the Γ pass and the masked forward is reported. Exit 1 on a miss.
  2 share of the masked gain L01 - L00 lost with (2,4) off in the Γ pass and the masked forward, (3,5) as reference, six states; the
    frozen path beside, against amend2.json (exit 1 on a miss); copying.gamma_reading.
  3 the mean gate at the marker on the alive subcomponents of block 2's o_proj that read each head's slot, W0 and each state.
results/amend3.json after every row; every scored number printed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import torch
from amend2 import NAMES, Reads, map_of
from torch import Tensor

from commute import copying as CP
from commute import keydep as KD
from commute import steering as ST
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, BASE, RUN, A2_OUT, OUT = (RESULTS / f for f in ("predictions_amend3.json", "predictions.json", "run.json", "amend2.json", "amend3.json"))
log = KD.log
Heads = list[CP.Head]
Gates = list[dict[str, Tensor]]


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


def gates_off(R: Reads, w: Tensor, heads: Heads, dW: ST.DW | None = None) -> Gates:
    """Γ's masks read in a forward with `heads` ablated (Amendment 2 forbade this on purpose; here it is the read)."""
    with CP.ablated(R.ctx.net, heads, R.means):
        return R.gates(w, dW)


def moved(a: Gates, b: Gates, half: int, alive: dict[str, Tensor]) -> float:
    """Mean |change| of the alive gates at the marker, over all 24 matrices."""
    parts = [(x[k][:, half, alive[k]] - y[k][:, half, alive[k]]).abs().flatten() for x, y in zip(a, b, strict=True) for k in x]
    return float(torch.cat(parts).double().mean())


def row1(R: Reads, H: Heads, ref: Heads, tol: float, A2: dict[str, Any], out: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
    alive = R.ctx.lift["alive"]
    batch = R.W["copy"][: KD.BATCH, : R.n].to(R.ctx.net.device)
    rec: dict[str, Any] = {"gamma_inputs": CP.gamma_inputs(R.ctx.net.model, R.ctx.net, batch, H[0], R.means), "tol": tol, "maps": {}}
    L00: dict[str, dict[str, Tensor]] = {}
    bad = rec["gamma_inputs"]["status"] != "pass"
    for m, w in R.W.items():
        g0, gH, gr = R.gates(w), gates_off(R, w, H), gates_off(R, w, ref)
        frozen = {"on": R.masked(w, g0, []), "H": R.masked(w, g0, H), "ref": R.masked(w, g0, ref)}
        L00[m] = {"on": frozen["on"], "frozen_H": frozen["H"], "H": R.masked(w, gH, H), "ref": R.masked(w, gr, ref)}
        diff = {k: abs(float(v.mean()) - A2["row3_masked"]["L00"][m][k]) for k, v in frozen.items()}
        zero = float((R.masked(w, gates_off(R, w, H, R.zero), H) - L00[m]["H"]).abs().max())
        rec["maps"][m] = {"L00": float(frozen["on"].mean()), "L00_frozen_H": float(frozen["H"].mean()), "L00_gamma_H": float(L00[m]["H"].mean()),
                          "L00_gamma_ref": float(L00[m]["ref"].mean()), "fall_frozen": float((frozen["on"] - frozen["H"]).mean()),
                          "fall_gamma": float((frozen["on"] - L00[m]["H"]).mean()), "fall_gamma_ref": float((frozen["on"] - L00[m]["ref"]).mean()),
                          "fall_target": A2["row4_w0_fall"]["w0_fall"][m], "diff_a2": diff, "zero_gain_max_abs": zero, "gates_moved": moved(g0, gH, R.half, alive)}
        bad = bad or max(diff.values()) > tol or zero != 0.0 or rec["maps"][m]["gates_moved"] == 0.0
    note(out, "row1_control", rec)
    if bad:
        stop(out, "row 1: the head off does not reach Γ as filed, the frozen masks do not reproduce amend2.json, or the zero update's gain is not 0")
    return L00


def row2(R: Reads, H: Heads, ref: Heads, L00: dict[str, dict[str, Tensor]], th: dict[str, Any], tol: float, A2: dict[str, Any], out: dict[str, Any]) -> None:
    rec: dict[str, Any] = {"states": {}}
    bad = False
    for name, dW in R.dws.items():
        m, a2 = map_of(name), A2["row3_masked"]["states"][name]
        w, z, t = R.W[m], L00[map_of(name)], a2["target_share"]
        g1, gH, gr = R.gates(w, dW), gates_off(R, w, H, dW), gates_off(R, w, ref, dW)
        on = R.masked(w, g1, [])
        d = on - z["on"]
        s_frozen = R.share(d, R.masked(w, g1, H) - z["frozen_H"])[2]
        L01_H = R.masked(w, gH, H)
        M, M_g, s, sd = R.share(d, L01_H - z["H"])
        r = {"L01": float(on.mean()), "L01_gamma_H": float(L01_H.mean()), "M": M, "M_gamma": M_g, "share": s, "share_sd": sd, "sd_weak": sd > th["sd_weak_over"], "target_share": t,
             "own_fall": float((on - L01_H).mean()) / M if M > 0 else math.nan, "share_frozen": s_frozen, "diff_a2": {"L01": abs(float(on.mean()) - a2["L01"]), "share_frozen": abs(s_frozen - a2["share"])},
             "ref_share": R.share(d, R.masked(w, gr, ref) - z["ref"])[2], "gates_moved": moved(g1, gH, R.half, R.ctx.lift["alive"]), "label": CP.masked_class(M, s, t, th)}
        rec["states"][name] = r
        bad = bad or max(r["diff_a2"].values()) > tol
        log(f"[row2 {name}] {json.dumps(jsonable(r))}")
    labels = [rec["states"][n]["label"] for n in NAMES]
    gain = {m: sum(rec["states"][n]["M"] for n in NAMES if map_of(n) == m) / 3 for m in R.W}
    w0 = out["row1_control"]["maps"]
    rec["w0_shift"] = {f"{m}_{k}": abs(w0[m][f"fall_gamma{s}"]) / gain[m] for m in R.W for k, s in (("H", ""), ("ref", "_ref"))}
    rec["references"] = [*(rec["states"][n]["ref_share"] for n in NAMES), *rec["w0_shift"].values()]
    rec["copy_reading"] = CP.gamma_reading(labels[:3], labels[3:], rec["references"], th)
    rec["lookup_reading"] = CP.agree(labels[3:], th)
    note(out, "row2_gamma", rec)
    if bad:
        stop(out, "row 2: the frozen path does not reproduce amend2.json at a state")


def row3(R: Reads, H: Heads, th: dict[str, Any], out: dict[str, Any]) -> None:
    lay, head = H[0]
    key = f"h.{lay}.attn.o_proj"
    sets = CP.slot_reads(R.ctx.comps0[key][1].cpu(), R.ctx.net.net.h[lay].attn.n_head, R.ctx.lift["alive"][key], th["slot_share_min"])

    def read(w: Tensor, dW: ST.DW | None = None) -> dict[str, Any]:
        g = torch.cat([b[key][:, R.half] for b in R.gates(w, dW)]).double()
        per = [float(g[:, s].mean()) if len(s) else math.nan for s in sets]
        return {"per_head": per, "gate": per[head], "label": CP.slot_class(len(sets[head]), per[head], th)}

    rec = {"n_per_head": [len(s) for s in sets], "W0": {m: read(w) for m, w in R.W.items()}, "states": {n: read(R.W[map_of(n)], dW) for n, dW in R.dws.items()}}
    note(out, "row3_slot", rec)


def load(a: argparse.Namespace) -> tuple[Reads, dict[str, Any], dict[str, Any], dict[str, Any]]:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    if OUT.exists():
        sys.exit(f"{OUT} exists: move it aside, this run deletes nothing")
    P, S, run, A2 = json.loads(PRED.read_text()), json.loads(BASE.read_text())["settings"], json.loads(RUN.read_text()), json.loads(A2_OUT.read_text())
    large = a.large.resolve()
    for f, sha in {"windows": run["windows_sha256"], **{n: run["states"][n]["sha256"] for n in NAMES}}.items():
        if KD.sha256(large / f"{f}.pt") != sha:
            sys.exit(f"{f}.pt is not the one run.json recorded")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, large)
    v = torch.load(large / "windows.pt", weights_only=False)
    half, n = S["half"], 2 * S["half"] + 1
    W = {m: CP.with_target(v["read"], v["target"][m]["read"]).tokens for m in ("copy", "lookup")}
    means = torch.load(large / "heads.pt", weights_only=False)["means"].to(ctx.net.device)
    dws = {name: torch.load(large / f"{name}.pt", weights_only=True, map_location="cpu") for name in NAMES}
    R = Reads(ctx, half, n, means, S["n_boot"], W, torch.empty(0), dws, run)
    head = git("rev-parse", "HEAD").stdout.strip()
    return R, P, A2, {"git_head": head, "predictions_sha256": KD.sha256(PRED), "host": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large")
    a = ap.parse_args()
    R, P, A2, out = load(a)
    th = P["thresholds"]
    as_heads = lambda key: [(int(x), int(y)) for x, y in P["settings"][key]]
    H, ref = as_heads("head"), as_heads("ref_head")
    tol = th["repeat_tol"] if out["host"] == A2["host"] else th["other_card_tol"]
    log(f"[host] {out['host']} repeat tolerance {tol}")
    L00 = row1(R, H, ref, tol, A2, out)
    row2(R, H, ref, L00, th, tol, A2, out)
    row3(R, H, th, out)
    r2 = out["row2_gamma"]
    log(f"[verdict] copy {r2['copy_reading']} lookup {r2['lookup_reading']} slot at W0 {out['row3_slot']['W0']['copy']['label']}")


if __name__ == "__main__":
    main()
