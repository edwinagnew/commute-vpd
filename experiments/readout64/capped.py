"""E64 capped, exploratory: the saved capped readout through the possession rule per block, W0 only. Reads provisional
reading 1 of E64's analysis; changes no verdict. Filing, run refused unless committed: results/predictions_capped.json."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from commute import keydep as KD
from commute import readout as RO
from commute import steering as ST
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RES, REPO, E54 = HERE / "results", HERE.parents[1], HERE.parent / "keydependent_elicit" / "results"
FILED = RES / "predictions_capped.json"


def reading(c_max: int, last: int, e0: int) -> tuple[str, str | None]:
    """The filed cases, each (C_max, L, E0) in exactly one."""
    tilt = "tilt_back" if c_max >= 6 else "tilt_partial" if c_max >= 3 else "tilt_out" if last >= 15 else "no_power"
    if tilt != "tilt_out":
        return tilt, None
    return tilt, "outruns_table" if e0 >= 15 else "between" if e0 >= 13 else "fits_table" if e0 >= 8 else "below_table"


def main() -> None:
    rel = str(FILED.relative_to(REPO))
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=False).stdout
    if not git("ls-files", rel).strip() or git("diff", "HEAD", "--", rel):
        sys.exit(f"{rel} is not committed unchanged")
    R, th = json.loads((RES / "run.json").read_text()), json.loads((RES / "predictions.json").read_text())["thresholds"]
    path = RES / "large" / "readouts_low_cap.pt"
    if KD.sha256(path) != R["stage0"]["fit"]["low_cap"]["sha256"]:
        sys.exit("readouts_low_cap.pt is not the run's: no reading")
    net = ST.Net("cuda")
    tasks = {n: ST.load_task(net, E54 / "large" / f"windows_{n}.pt", n, n) for n in ("elicit", "cipher")}
    vals = RO.pair_values(tasks)
    vidx = {n: {k: vals.tolist().index(int(t.table[k])) for k in t.keys} for n, t in tasks.items()}
    probes = RO.Probes(len(RO.RESID), 768, len(vals)).to(net.device)
    probes.load_state_dict(torch.load(path, weights_only=True))
    per_value = {int(k): v for k, v in R["stage0"]["positions"]["per_value"].items()}
    shift = RO.prior_shift(vals, per_value, 241).to(net.device)
    out: dict = {"blocks": {}, "W0_flat_prior": {}}
    for n, t in tasks.items():
        lg, unfit, out["blocks"][n] = RO.readout_logits64(net, probes, t), R["stage0"]["positions"]["unfit_keys"][n], {}
        w0 = net.run(t.toks)[0][:, t.half][:, vals.to(net.device)] + shift
        out["W0_flat_prior"][n] = RO.key_reading(w0, t.toks[:, t.half].tolist(), t.keys, vidx[n], th)
        for p, sl in {"all": slice(None), "fit": t.part("fit")}.items():
            keys, M = t.toks[sl, t.half].tolist(), R["W0"]["elicit"][p]["possessed"]
            rs = [RO.key_reading(lg[b, sl], keys, t.keys, vidx[n], th, unfit) for b in range(len(RO.RESID))]
            for r in rs:
                r["verdict"] = RO.relative_possession(r["possessed"], M, th["frac_num"], th["frac_den"]) if n == "elicit" else RO.cipher_possession(r["possessed"], th["min_M"], th["frac_num"], th["frac_den"])
                print(n, p, r["verdict"], r["possessed"], flush=True)
            out["blocks"][n][p] = rs
    e = [r["verdict"]["count"] for r in out["blocks"]["elicit"]["all"]]
    c_max = max(r["verdict"]["count"] for r in out["blocks"]["cipher"]["all"])
    out["C_max"], out["L"], out["E0"] = c_max, e[RO.LAST], e[0]
    out["tilt"], out["elicit"] = reading(c_max, e[RO.LAST], e[0])
    print({k: out[k] for k in ("C_max", "L", "E0", "tilt", "elicit")}, flush=True)
    (RES / "capped.json").write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
