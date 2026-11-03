"""E74 Amendment 1 ($0, one cluster L40). Plan: reports/plans/2026-11-03_plan_e74-transmission.md, Amendment 1; rules: results/predictions_amend1.json.
Job 50510 stopped at the filed control on u3 (KL ratio 0.447 against 0.5, amplitude 0.75, cosine 0.96). This job: u3's remaining groups under the
amended control, then on u1 and u3 which masks lose the change (upstream of it, or at and after it) and how far the masked forward's activations
sit from the clean ones. Each group writes its file as it ends, a rerun skips it. Usage as run.py, with this file.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import qwen_sft
from commute.jobs import hard_exit
from commute.qwen_vpd import (
    SITES,
    _hooks,
    _linear,
    fp32_matmuls,
    importances,
    load_target,
    parse_site,
    site_name,
)

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e74_run", HERE / "run.py")
assert spec is not None and spec.loader is not None
run = importlib.util.module_from_spec(spec)
sys.modules["e74_run"] = run
spec.loader.exec_module(run)
LAYERS, ID_TOL = (0, 7, 14, 21), 1e-3
RESULTS, Arm, C, log, rules = run.RESULTS, run.Arm, run.C, run.log, run.rules


def stream_gap(rd: Any, rows: Tensor) -> dict[str, list[float]]:
    """Per layer, squared distance of the masked forward's residual stream (input of the MLP's gate matrix) from the clean one's, and the clean norm."""
    sites = [site_name(i, "gate") for i in range(28)]
    gap, norm = [0.0] * 28, [0.0] * 28

    def taps(store: dict[str, Tensor]) -> Any:
        return _hooks([_linear(rd.model, s).register_forward_pre_hook(
            lambda _m, args, s=s: store.__setitem__(s, args[0].detach().float())) for s in sites])

    for tokens in rows[: C.rows]:
        t, clean, masked = tokens[None].to(run.DEV), {}, {}
        g = importances(rd.model, t, rd.gamma, rd.comps)[1]
        with taps(clean):
            rd.model(t, use_cache=False)
        with taps(masked):
            run.forward(rd.model, t, rd.comps, g, {})
        for i, s in enumerate(sites):
            gap[i] += float((masked[s] - clean[s]).double().pow(2).sum())
            norm[i] += float(clean[s].double().pow(2).sum())
        del clean, masked, g
    return {"gap2": gap, "norm2": norm, "relative": [(a / b) ** 0.5 for a, b in zip(gap, norm, strict=True)]}


def factor(rd: Any, name: str, rows: Tensor, dw: dict[str, Tensor]) -> None:
    arms = {"up_0": Arm({s: v for s, v in dw.items() if parse_site(s)[0] == 0}, -1)}  # no mask anywhere: must read 1
    for layer in LAYERS:
        blk = {s: v for s, v in dw.items() if parse_site(s)[0] == layer}
        gauss = rd.fit(rows, blk, layer)[0]
        if layer:
            arms[f"up_{layer}"] = Arm(blk, layer - 1)
            arms[f"gauss_up_{layer}"] = Arm(gauss, layer - 1)
        arms[f"xclean_{layer}"] = Arm(blk, None, True)
        arms[f"gauss_xclean_{layer}"] = Arm(gauss, None, True)
    s = rd.group(name, "factor", rows, arms, {})["summary"]
    if abs(s["up_0"]["tau"] - 1) > ID_TOL:
        run.stop("control_identity", f"{name}: up_0 {s['up_0']['tau']}")
    v = {f"dw_{i}": rules.factor(s["up_0" if not i else f"up_{i}"]["tau"], s[f"xclean_{i}"]["tau"]) for i in LAYERS}
    v |= {f"gauss_{i}": rules.factor(s[f"gauss_up_{i}"]["tau"], s[f"gauss_xclean_{i}"]["tau"]) for i in LAYERS if i}
    (RESULTS / name / "factor_verdict.json").write_text(json.dumps(v, indent=1))
    log(f"[factor {name}] {json.dumps(v)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    a = ap.parse_args()
    if not (RESULTS / "predictions_amend1.json").is_file() or (RESULTS / "stopped.json").is_file():
        sys.exit("refusing to start: predictions_amend1.json missing, or stopped.json exists")
    fp32_matmuls()
    pr = run.load_file("pass_amend1", HERE.parent / "qwen_pass_test" / "amend1.py").pr
    maths = torch.tensor([r[:256] for r in json.loads((HERE.parent / "qwen_sft/results/maths_rows.json").read_text())["rows"]])
    rows = {"maths": maths, "a1": pr.a1_rows()}
    updates = {u.name: u for u in pr.P.updates}
    model = load_target(run.DEV, torch.float32)
    with torch.no_grad():
        rd = run.Read1(pr, model, Path(a.decomp))
        for name in ("u1", "u3"):
            u = updates[name]
            dw = {s: v.float() for s, v in qwen_sft.load(u.path).items() if s in SITES}
            rd.update(u, rows[u.rows], dw)  # u1's and u3's control re-scored; what exists is skipped
            factor(rd, name, rows[u.rows], dw)
            path = RESULTS / name / "stream_gap.json"
            if not path.is_file():
                r = stream_gap(rd, rows[u.rows])
                path.write_text(json.dumps(r, indent=1))
                log(f"[stream_gap {name}] relative {[round(x, 4) for x in r['relative']]}")
    log("[done amend1]")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    hard_exit(code)
