"""E82: teacher training on E68's scorecard. Reads E76's saved clean-loss arm (A) through E68's reading on u1, so its row sits beside
do-nothing and the splits: 128 maths rows of 256, masks from Γ on W0's forward with the state's V′ in the readouts, Δ off, fine-tuned clean
as reference. Nothing is trained. Rules: results/predictions_teacher.json. The control runs first: unchanged components must give
E68's do-nothing sums on the first 8 rows, or the job stops.

Usage (repo root, L40): sbatch cluster/qwen_pass_teacher.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

from commute.jobs import load_experiment
from commute.qwen_sft import load
from commute.qwen_teacher import Offsets, change_hooks
from commute.qwen_vpd import (
    SITES,
    importances,
    load_components,
    load_gamma,
    load_target,
    masked_logits,
)
from commute.vpd import kl_per_position

job = load_experiment("qwen_maps", "job")
e76 = load_experiment("qwen_masked_teacher", "common")
HERE = Path(__file__).resolve().parent
RESULTS, DEV = HERE / "results", "cuda"
DW_FILE = Path.home() / "commute-data" / "results-large" / "qwen_sft" / "nowarmup_regen" / "dW" / "step_10.safetensors"
ROWS, ROW_LEN, CONTROL_ROWS, TOL = 128, 256, 8, 1e-3
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def read(model: Any, comps: Any, gamma: Any, dw: dict[str, Tensor], rows: Tensor) -> dict[str, list[float]]:
    """Per-row sums, as `PassReader.row`: K = KL(FT clean ‖ masked), nll of the masked forward. The model stays at W0; FT is W0 plus δW by hooks."""
    out: dict[str, list[float]] = {"K": [], "nll": []}
    for row in rows:
        t = row[None].to(DEV)
        _, g = importances(model, t, gamma, comps)
        ci = masked_logits(model, t, comps, g)
        with change_hooks(model, lambda s: dw[s]):
            ref = model(t, use_cache=False).logits
        out["K"].append(float(kl_per_position(ref, ci).double().sum()))
        ce = torch.nn.functional.cross_entropy(ci[:, :-1].flatten(0, 1).float(), t[:, 1:].flatten(), reduction="none")
        out["nll"].append(float(ce.double().sum()))
    return out


def write_columns(comps: Any, off: Offsets) -> None:
    for s in SITES:
        V, U = comps.vu(s)
        idx = off.alive[s].to(V.device)
        V[:, idx] = (off.V[s] + off.dV[s]).to(V.device, V.dtype)
        U[idx] = (off.U[s] + off.dU[s]).to(U.device, U.dtype)


def score(k: list[float], nll: list[float], w0: dict, dn: dict) -> dict[str, float]:
    n, T = len(k), w0["T"]
    K, Kd, K0 = sum(k) / (n * T), sum(dn["sums"]["K"]) / (n * T), sum(w0["sums"]["ci"]) / (n * T)
    per = n * (T - 1)
    gain = (sum(w0["sums"]["nll"]) - sum(nll)) / per / ((sum(w0["sums"]["clean"]) - sum(w0["sums"]["ft"])) / per)
    return {"K": K, "KD": Kd, "K0": K0, "share_closed": (Kd - K) / (Kd - K0), "faithful_line": 1 - 0.5 * K0 / (Kd - K0), "gain_kept": gain}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    a = ap.parse_args()
    if not (RESULTS / "predictions_teacher.json").is_file():
        sys.exit("refusing to start: results/predictions_teacher.json is missing")
    w0, dn = (json.loads((RESULTS / "u1" / f"{n}.json").read_text()) for n in ("W0", "do_nothing"))
    rows = torch.tensor([r[:ROW_LEN] for r in json.loads(job.MATHS.read_text())["rows"][:ROWS]])
    dw = {s: v.to(DEV) for s, v in load(DW_FILE).items() if s in SITES}
    with torch.no_grad():
        model = load_target(DEV, torch.float32)
        comps = load_components(Path(a.decomp), DEV, torch.float32)
        gamma = load_gamma(Path(a.decomp), DEV, torch.bfloat16)
        log(f"loaded, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        control = read(model, comps, gamma, dw, rows[:CONTROL_ROWS])
        gaps = {k: max(abs(x / y - 1) for x, y in zip(control[k], dn["sums"][k], strict=False)) for k in control}
        log(f"[control] largest relative gap to E68's do-nothing on {CONTROL_ROWS} rows: {gaps}")
        if max(gaps.values()) > TOL:
            sys.exit(f"STOP control: {gaps} over {TOL}")
        off = Offsets(comps, load_file(str(e76.LARGE / "cut.safetensors")), "cpu")
        off.load_from(e76.LARGE / "A_offsets.safetensors")
        write_columns(comps, off)
        got = read(model, comps, gamma, dw, rows)
    out = {"control_gaps": gaps, **score(got["K"], got["nll"], w0, dn), "sums": got}
    (RESULTS / "teacher_clean.json").write_text(json.dumps(out, indent=1))
    log(json.dumps({k: v for k, v in out.items() if k != "sums"}))


if __name__ == "__main__":
    main()
