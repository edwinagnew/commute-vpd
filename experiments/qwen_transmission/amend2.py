"""E74 Amendment 2, the last ($0, one cluster L40). Plan: reports/plans/2026-11-03_plan_e74-transmission.md, Amendment 2; rules: results/predictions_amend2.json.
Two reads on u1 and u3. Direct: in the clean model, every matrix after the change's block fixed at its clean output, so the change reaches the logits only
through the residual skip: is the real update's clean effect amplified by later blocks where a Gaussian's is not? Stochastic: transmission under VPD's own
stochastic masks and Δ share, one draw per row. Usage as run.py, with this file.
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
    load_target,
    parse_site,
)

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e74_run", HERE / "run.py")
assert spec is not None and spec.loader is not None
run = importlib.util.module_from_spec(spec)
sys.modules["e74_run"] = run
spec.loader.exec_module(run)
LAYERS, ID_TOL, STOCH_SEED = (0, 7, 14, 21), 1e-3, 74
RESULTS, Arm, C, log, rules, DOWN27 = run.RESULTS, run.Arm, run.C, run.log, run.rules, run.DOWN27


def direct_logits(model: Any, t: Tensor, dW: dict[str, Tensor], layer: int) -> tuple[Tensor, Tensor]:
    """Clean logits, and logits with every site after `layer` fixed at its clean output and x dWᵀ added in `layer`."""
    later = [s for s in SITES if parse_site(s)[0] > layer]
    stored: dict[str, Tensor] = {}

    def keep(site: str) -> Any:
        def hook(_m: Any, _a: tuple[Tensor, ...], out: Tensor) -> None:
            stored[site] = out.detach()

        return hook

    with _hooks([_linear(model, s).register_forward_hook(keep(s)) for s in later]):
        clean = model(t, use_cache=False).logits

    def fixed(site: str) -> Any:
        def hook(_m: Any, _a: tuple[Tensor, ...], _out: Tensor) -> Tensor:
            return stored[site]

        return hook

    def plus(site: str) -> Any:
        d = dW[site]

        def hook(_m: Any, a: tuple[Tensor, ...], out: Tensor) -> Tensor:
            return out + a[0] @ d.to(a[0]).T

        return hook

    handles = [_linear(model, s).register_forward_hook(fixed(s)) for s in later] + [_linear(model, s).register_forward_hook(plus(s)) for s in dW]
    with _hooks(handles):
        return clean, model(t, use_cache=False).logits


def direct(rd: Any, name: str, rows: Tensor, dw: dict[str, Tensor]) -> None:
    path = RESULTS / name / "direct.json"
    if path.is_file():
        return
    out: dict[str, dict[str, float]] = {}
    block = {i: {s: v for s, v in dw.items() if parse_site(s)[0] == i} for i in (*LAYERS, 27)}
    arms = {f"dw_{i}": (b, i) for i, b in block.items()} | {f"gauss_{i}": (rd.fit(rows, block[i], i)[0], i) for i in LAYERS}
    for key, (dW, layer) in arms.items():
        k = 0.0
        for tokens in rows[: C.rows]:
            t = tokens[None].to(run.DEV)
            k += rd.pr.kl(*direct_logits(rd.model, t, dW, layer))
        k /= C.rows * rows.shape[1]
        k_c = run.k_clean(rd.pr, rd.model, rows[: C.rows], rd.comps, dW)
        out[key] = {"K_direct": k, "K_clean": k_c, "share": k / k_c}
        if key == "dw_27" and abs(k / k_c - 1) > ID_TOL:  # nothing after block 27: the identity
            run.stop("control_identity_direct", f"{name}: {k} vs {k_c}")
    verdict = rules.direct_gap({str(i): (out[f"dw_{i}"]["share"], out[f"gauss_{i}"]["share"]) for i in rules.EARLY})
    path.write_text(json.dumps({"arms": out, "verdict": verdict}, indent=1))
    log(f"[direct {name}] {verdict} {json.dumps(out)}")


def stochastic(rd: Any, name: str, rows: Tensor, dw: dict[str, Tensor]) -> None:
    d27 = {DOWN27: dw[DOWN27]}
    arms = {"dw_down27": Arm(d27), "gauss_down27": Arm(rd.fit(rows, d27, 27_000)[0])}
    for layer in (*LAYERS, 27):
        blk = {s: v for s, v in dw.items() if parse_site(s)[0] == layer}
        arms[f"dw_{layer}"], arms[f"gauss_{layer}"] = Arm(blk), Arm(rd.fit(rows, blk, layer)[0])
    arms["all"] = Arm(dw)
    gen = torch.Generator(run.DEV).manual_seed(STOCH_SEED)
    s = rd.group(name, "stochastic", rows, arms, {}, gen)["summary"]
    a_d, a_g = s["dw_down27"]["amp"], s["gauss_down27"]["amp"]
    if not rules.control(a_d, a_g):
        run.stop("control_transmission_stochastic", f"{name}: amplitude dw {a_d}, gauss {a_g}")
    log(f"[stochastic verdict {name}] {rules.stochastic(s['all']['tau'])} tau_all {s['all']['tau']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    a = ap.parse_args()
    if not (RESULTS / "predictions_amend2.json").is_file() or (RESULTS / "stopped.json").is_file():
        sys.exit("refusing to start: predictions_amend2.json missing, or stopped.json exists")
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
            direct(rd, name, rows[u.rows], dw)
            stochastic(rd, name, rows[u.rows], dw)
    log("[done amend2]")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    hard_exit(code)
