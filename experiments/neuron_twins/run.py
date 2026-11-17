"""neuron_twins (E158, cell P2.21): E98's neuron reading (`commute.neuron_r`: r_n whole width, r_half) on the two
real-against-shuffled-twin pairs, case label (E112) and repeat label (E125), the method table's four empty cells. Filed
before the run: results/predictions.json (the run refuses unless it is committed unchanged). Descriptive: no bar.

The states, rows and unit are E140's (experiments/real_vs_twin, loaded from its run.py): seeds 0-1 of each arm from the
source experiments, seed 2 from E140's results/large, each sha256-checked; the first 256 fit windows of the update's own
arm, its own labels; every cell read in the contrast between the two labels (logit of the window's own label minus the
other's, E140's scored unit), not log-odds of one target. r_n = mean(L(O0, A1) - T0) / mean(T1 - T0), r_half likewise
with Ah (cells defined in `neuron_r`'s docstring). Target model only, fp32, TF32 off, batches of 64.

  stage 0, alone (a miss writes, logs FAIL and exits 1):
    zero update on both pairs' rows, both arms: every cell equals T0 (max |difference| <= identity_tol)
    positive control: real s0's last-block down_proj change alone, per pair: nothing re-fires, so the r_n, r_half,
    i_n and e_n numerators are 0 and d_n's equals G (each within band), with |G| above min_control_gain
  stage 1: the twelve states (pair x arm x seed), each checked as it lands (identity, telescoping, and its G against
    E140's saved gain in this unit), written and printed; then each pair's reuse (real) and new (twin) arrays.
Output: results/cells.json (.stage0, .states.<pair>_<arm>_s<seed>, .cells.<method table pair>.<neuron|neuron_half>).

    sbatch --comment=E158 cluster/neuron_twins.sbatch
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import keydep as KD
from commute import neuron_r as NR
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.local_faith import MIN_GAIN
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
RVT = load_experiment("real_vs_twin", "run")  # E140: load_pair, state, the rows and the saved gains
E140_S1, E140_S3 = RVT.RESULTS / "stage1.json", RVT.RESULTS / "stage3.json"
TABLE = {"case": "e123_case_label", "repeat": "e125_repeat_label"}  # E140's pair -> the method table's pair
COLS = {"neuron": "r_n", "neuron_half": "r_half"}  # the method table's column -> neuron_r's share
LAST_DOWN = f"h.{NR.N_LAYERS - 1}.mlp.down_proj"
log = KD.log


class Stopped(Exception):
    """A filed stop: cells.json records why; the job exits 1."""


class Out:
    def __init__(self) -> None:
        self.out: dict[str, Any] = {"stage0": {}, "states": {}, "cells": {}}

    def write(self) -> None:
        tmp = RESULTS / "cells.tmp"  # a kill mid-write must not leave unreadable JSON
        tmp.write_text(json.dumps(jsonable(self.out), indent=1, sort_keys=True))
        tmp.replace(RESULTS / "cells.json")

    def put(self, group: str, key: str, rec: dict[str, Any], ok: bool | None = None) -> None:
        self.out[group][key] = {**rec, **({} if ok is None else {"ok": bool(ok)})}
        if ok is False:
            self.out["stopped"] = f"{group}/{key}"
        self.write()
        log(f"[{group} {key}] ok {ok} {json.dumps(jsonable(rec), sort_keys=True)}")
        if ok is False:
            raise Stopped(f"{group}/{key}")


def contrast(ab: tuple[int, int]) -> ST.Readout:
    """logit of the target label minus the other label's at `pos`: E140's contrast (masked_reads.units), as the target
    is the window's own label (A when y = 1, else B)."""

    def f(lg: Tensor, pos: int, tgt: Tensor) -> Tensor:
        x, t = lg[:, pos].float(), tgt.to(lg.device)[:, None]
        return (x.gather(1, t) - x.gather(1, ab[0] + ab[1] - t)).squeeze(1).double().cpu()

    return f


def reads(p: Any, arm: str, n: int) -> NR.Reads:
    toks, y = p.part(arm, "fit", n)
    return NR.Reads(f"{p.name}_{arm}", toks, torch.where(y == 1, p.ab[0], p.ab[1]), RVT.HALF)


def numerators(net: ST.Net, p: Any, arm: str, dW: ST.DW, S: dict[str, Any]) -> dict[str, Any]:
    return NR.shares(NR.state_cells(net, reads(p, arm, S["n_read"]), dW, S["batch"], contrast(p.ab)), S["gain_floor"])


def stage0(net: ST.Net, pairs: dict[str, Any], S: dict[str, Any], o: Out) -> None:
    zero = {k: torch.zeros_like(w) for k, w in net.W0.items()}
    for p in pairs.values():
        for arm in RVT.ARMS:
            r = reads(p, arm, S["n_read"])
            dev = NR.max_deviation(NR.state_cells(net, r, zero, S["batch"], contrast(p.ab)))
            o.put("stage0", f"zero/{r.name}", {"max_abs": dev, "n_windows": len(r.toks)}, dev <= S["identity_tol"])
    band = S["identity_tol"]
    for p in pairs.values():
        dW = RVT.state(p, "real", 0, {})
        last = {k: dW[k] if k == LAST_DOWN else torch.zeros_like(w) for k, w in net.W0.items()}
        rec = numerators(net, p, "real", last, S)
        n, G = rec["num"], rec["G"]
        conds = {f"|{k} num| <= band": abs(n[k]) <= band for k in ("r_n", "r_half", "i_n", "e_n")}
        conds |= {"|d_n num - G| <= band": abs(n["d_n"] - G) <= band, "|G| > min_control_gain": abs(G) > S["min_control_gain"]}
        failed = [c for c, ok in conds.items() if not ok]
        o.put("stage0", f"down_last/{p.name}_real_s0", {**rec, "band": band, "failed": failed}, not failed)


def side(o: Out, names: list[str], key: str) -> tuple[list[float], list[str]]:
    """The readable states' values of `key` (a share is nan when |G| < gain_floor) and the names left out."""
    st = o.out["states"]
    return [st[n][key] for n in names if not math.isnan(st[n][key])], [n for n in names if math.isnan(st[n][key])]


def stage1(net: ST.Net, pairs: dict[str, Any], S: dict[str, Any], o: Out) -> None:
    new, saved = json.loads(E140_S1.read_text())["states"], json.loads(E140_S3.read_text())["states"]
    for p in pairs.values():
        names = {arm: [f"{p.name}_{arm}_s{s}" for s in S["seeds"]] for arm in RVT.ARMS}
        for arm in RVT.ARMS:
            for s, n in zip(S["seeds"], names[arm], strict=True):
                rec = numerators(net, p, arm, RVT.state(p, arm, s, new), S)
                rec["G_e140"] = saved[n]["gain"]["contrast"]
                ok = rec["identity"] <= S["identity_tol"] and abs(rec["telescope"]) <= S["telescope_tol"] and abs(rec["G"] - rec["G_e140"]) <= S["gain_tol"]
                o.put("states", n, rec, ok)
        for col, key in COLS.items():
            (reuse, out_r), (twin, out_t) = side(o, names["real"], key), side(o, names["twin"], key)
            o.put("cells", f"{TABLE[p.name]}/{col}", {"reuse": reuse, "new": twin, "states": names, "left_out": out_r + out_t,
                                                      "source": f"experiments/neuron_twins/results/cells.json .states[<state>].{key}"})  # fmt: skip


def main() -> int:
    require_filed(PRED)
    S = json.loads(PRED.read_text())["settings"]
    rvt = json.loads(RVT.PRED.read_text())["settings"]
    assert S["n_read"] == rvt["n_read"] and S["gain_floor"] == MIN_GAIN and S["seeds"] == list(RVT.SEEDS), "predictions.json disagrees with the code"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net, pairs, o = ST.Net("cuda"), {n: RVT.load_pair(n) for n in RVT.PAIRS}, Out()
    try:
        stage0(net, pairs, S, o)
        print("STAGE 0 DONE", flush=True)
        stage1(net, pairs, S, o)
        print("STAGE 1 DONE", flush=True)
    except Stopped as e:
        print(f"STOP: {e}: tell the lead", flush=True)
        return 1
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    hard_exit(main())
