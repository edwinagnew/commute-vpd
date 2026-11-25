"""E83: the logit-shift bars of E74 and E77 on the 67M ($0, one cluster L40, debug QoS). The user's one-off for Q3 of the update
notes; it does not reopen the re-gating line. Plan: reports/plans/2026-11-04_plan_e83-logit-shift-67m.md. Saved inputs only:
row_blocks' dW_adam.pt and dW_lora.pt, Pile val rows 0-255 (the slide 5 scorecard's rows, E39 and E71).

The bar: the update enters as an unmasked term x dWᵀ at each matrix. amp = sqrt(sum (masked shift)² / sum (clean shift)²), a shift
being the logit change from adding the term, centred per position. Every forward is fp32, as on Qwen; Γ runs under the authors'
bf16 autocast (`vpd.compute_ci`). Ported from experiments/qwen_transmission/run.py (`forward`, `read_group`) and
experiments/qwen_open_points/run.py (`uniform_k_routes`, `routed`), which stay as they ran.

Order, each update: the last-matrix control first, with a stop (E74's: every matrix masked with Γ on W0, the change at the last
matrix alone, the update's and a Gaussian scaled to its clean KL, both in E74's band); then the bars, each written as it ends.

Usage (repo root, a job that sources cluster/env.sh): python experiments/logit_shift_67m/run.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import vpd
from commute.data import batches, pile_rows
from commute.faithfulness import model_state
from commute.jobs import hard_exit

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
INPUTS = HERE.parent / "row_blocks" / "results"
DEV = "cuda"
LAST = "h.3.mlp.down_proj"
ROWS, BATCH, SITES_N = 256, 8, 24
BAND = (0.5, 1.5)  # E74's last-matrix control (qwen_transmission/rules.py)
ID_TOL = 1e-3  # E77's identity tolerance (qwen_open_points/rules.py)
HOOK_TOL = 0.01  # E74's: the term against writing W0 + dW into the weights
FIT_ROUNDS, FIT_TOL = 4, 0.1  # E74's
CI_REF, CI_TOL = 0.3481880286708474, 0.15  # E71's W0 CI-masked KL on these rows (bf16); row_blocks' w0_ci_tol
STOCH_SEED, ROUTE_SEED, GAUSS_SEED = 74, 77_000, 27_000  # E74's and E77's
# masks, sites routed per position (None: the library's draw), route draws per batch. Fixed masks routed everywhere have d = 0: Δ off.
POINTS = {"fixed_all": ("fixed", SITES_N, 1), "fixed_none": ("fixed", 0, 1), "random_all": ("random", SITES_N, 1),
          "random_subset": ("random", None, 3)}
T0 = time.time()

Change = dict[str, Tensor]


def log(msg: str) -> None:
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with (RESULTS / "run.log").open("a") as f:
        f.write(line + "\n")


def stop(rule: str, detail: str) -> None:
    (RESULTS / "stopped.json").write_text(json.dumps({"rule": rule, "detail": detail}))
    log(f"STOP {rule}: {detail}")
    hard_exit(1)


def kl(p: Tensor, q: Tensor) -> float:
    return float(vpd.kl_per_position(p, q).double().sum())


def forward(model: Any, t: Tensor, comps: dict[str, tuple[Tensor, Tensor]], m: dict[str, Tensor] | None, d: dict[str, Tensor] | None,
            dW: Change) -> Tensor:
    """Each matrix's output: ((x V) * (m - d)) U + d * (x Wᵀ) (m None: the model's own; d None: Δ off), plus x dWᵀ."""

    def hook_for(site: str) -> Any:
        U, V = comps[site]
        change = dW.get(site)

        def hook(_: Any, args: tuple[Tensor, ...], out: Tensor) -> Tensor:
            x = args[0]
            if m is None:
                base = out
            elif d is None:
                base = (x @ V * m[site]) @ U
            else:
                share = d[site][..., None]
                base = (x @ V * (m[site] - share)) @ U + share * out
            return base if change is None else base + x @ change.T

        return hook

    handles = [vpd.target_linear(model, s).register_forward_hook(hook_for(s)) for s in comps]
    try:
        return model(t)
    finally:
        for h in handles:
            h.remove()


def centred_shift(with_term: Tensor, without: Tensor) -> Tensor:
    s = with_term - without
    return s - s.mean(-1, keepdim=True)


def squared(a: Tensor) -> float:
    return float((a * a).double().sum())


def stochastic(g: dict[str, Tensor], gen: torch.Generator) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """m = g + (1 - g) u per (row, position, component), d ~ U(0, 1) per (row, position, site), as `qwen_vpd.stochastic_masks`."""
    m, d = {}, {}
    for s, v in g.items():
        m[s] = v + (1 - v) * torch.rand(v.shape, generator=gen, device=v.device)
        d[s] = torch.rand(v.shape[:-1], generator=gen, device=v.device)
    return m, d


def routes(sites: list[str], shape: tuple[int, ...], k: int | None, gen: torch.Generator) -> dict[str, Tensor]:
    """Per position exactly k sites take the masked branch; k None draws k uniform on 1..24 per position (`UniformKSubsetRouter`)."""
    kk = torch.randint(1, len(sites) + 1, shape, generator=gen, device=DEV) if k is None else torch.full(shape, k, device=DEV)
    on = torch.rand((len(sites), *shape), generator=gen, device=DEV).argsort(0) < kk
    return {s: on[j] for j, s in enumerate(sites)}


def routed(m: dict[str, Tensor], d: dict[str, Tensor] | None, route: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """As given where routed (d None: Δ off); m = 1 and d = 1 where not, which is x Wᵀ."""
    return ({s: torch.where(route[s][..., None], m[s], 1.0) for s in m},
            {s: torch.where(route[s], 0.0 if d is None else d[s], 1.0) for s in m})


def importances(model: Any, t: Tensor) -> dict[str, Tensor]:
    return {k: v.float() for k, v in vpd.compute_ci(model, t).g.items()}


def amp(sums: dict[str, list[float]]) -> float:
    aa, bb = sum(sums["aa"]), sum(sums["bb"])
    return (bb / aa) ** 0.5 if aa > 0 else float("nan")


def clean_kl(model: Any, rows: list[Tensor], comps: Any, change: Change) -> float:
    total = 0.0
    for t in rows:
        total += kl(model(t), forward(model, t, comps, None, None, change))
    return total / (ROWS * rows[0].shape[1])


def gaussian_like(model: Any, rows: list[Tensor], comps: Any, change: Change) -> tuple[Change, dict[str, float]]:
    """E74's `gaussian_like`: Gaussian in the same matrix, scaled to the change's clean KL by the quadratic rule."""
    gen = torch.Generator().manual_seed(GAUSS_SEED)
    unit = {}
    for s, v in change.items():
        g = torch.randn(v.shape, generator=gen).to(v)
        unit[s] = g * v.norm() / g.norm()
    target, a, got = clean_kl(model, rows, comps, change), 1.0, 0.0
    for _ in range(FIT_ROUNDS):
        got = clean_kl(model, rows, comps, {s: a * v for s, v in unit.items()})
        if abs(got / target - 1) <= FIT_TOL:
            break
        a *= (target / got) ** 0.5
    return {s: a * v for s, v in unit.items()}, {"scale": a, "K_clean_fit": got, "K_clean_target": target}


def control(model: Any, rows: list[Tensor], comps: Any, name: str, dW: Change) -> dict[str, Any]:
    """The term reads as the weights; Γ's masks read E71's KL; the last matrix's change survives every mask. Stops on a miss."""
    t = rows[0]
    clean = model(t)
    with model_state(model, target={k: vpd.target_linear(model, k).weight.detach() + v for k, v in dW.items()}):
        by_weights = kl(clean, model(t))
    by_term = kl(clean, forward(model, t, comps, None, None, dW))
    log(f"[control {name} term vs weights] clean KL {by_term:.9g} vs {by_weights:.9g}")
    if not abs(by_term / by_weights - 1) <= HOOK_TOL:
        stop("control_hook", f"{name}: {by_term} vs {by_weights}")
    last = {LAST: dW[LAST]}
    gauss, fit = gaussian_like(model, rows, comps, last)
    log(f"[control {name} Gaussian fit] {json.dumps(fit)}")
    sums: dict[str, dict[str, list[float]]] = {"dw": {"aa": [], "bb": []}, "gauss": {"aa": [], "bb": []}}
    k0 = 0.0
    for t in rows:
        clean = model(t)
        g = importances(model, t)
        masked = forward(model, t, comps, g, None, {})
        k0 += kl(clean, masked)
        for arm, change in (("dw", last), ("gauss", gauss)):
            sums[arm]["aa"].append(squared(centred_shift(forward(model, t, comps, None, None, change), clean)))
            sums[arm]["bb"].append(squared(centred_shift(forward(model, t, comps, g, None, change), masked)))
        del clean, g, masked
    k0 /= ROWS * rows[0].shape[1]
    out = {"amp_dw": amp(sums["dw"]), "amp_gauss": amp(sums["gauss"]), "K0": k0, "K0_ref": CI_REF, "fit": fit, "sums": sums,
           "by_term": by_term, "by_weights": by_weights}
    log(f"[control {name}] last-matrix amplitude: update {out['amp_dw']:.4f}, Gaussian {out['amp_gauss']:.4f}; masked KL {k0:.4f} vs E71 {CI_REF:.4f}")
    if not abs(k0 / CI_REF - 1) <= CI_TOL:
        stop("control_K0", f"{name}: {k0} vs {CI_REF}")
    if not all(BAND[0] <= out[x] <= BAND[1] for x in ("amp_dw", "amp_gauss")):
        stop("control_last_matrix", f"{name}: {out['amp_dw']}, {out['amp_gauss']}; band {BAND}")
    return out


def point(model: Any, rows: list[Tensor], comps: Any, sites: list[str], dW: Change, label: str) -> dict[str, Any]:
    kind, k, draws = POINTS[label]
    stoch = torch.Generator(DEV).manual_seed(STOCH_SEED)
    gen = torch.Generator(DEV).manual_seed(ROUTE_SEED + (SITES_N + 1 if k is None else k))
    sums: dict[str, list[float]] = {"aa": [], "bb": []}
    for t in rows:
        clean = model(t)
        g = importances(model, t)
        d = None
        if kind == "random":
            g, d = stochastic(g, stoch)
        sums["aa"].append(squared(centred_shift(forward(model, t, comps, None, None, dW), clean)))
        del clean
        bb = 0.0
        for _ in range(draws):
            m_r, d_r = routed(g, d, routes(sites, tuple(t.shape), k, gen))
            bb += squared(centred_shift(forward(model, t, comps, m_r, d_r, dW), forward(model, t, comps, m_r, d_r, {}))) / draws
            del m_r, d_r
        sums["bb"].append(bb)
        del g, d
    return {"amp": amp(sums), "sums": sums, "k": k, "draws": draws, "masks": kind}


def cached(path: Path, fn: Any) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text())
    r = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=1))
    return r


def load_update(name: str, sites: list[str], filed: dict[str, str]) -> Change:
    path = INPUTS / "large" / f"dW_{name}.pt"
    with path.open("rb") as f:
        got = hashlib.file_digest(f, "sha256").hexdigest()
    if got != filed[path.name]:
        stop("input_file", f"{path}: sha256 {got}, filed {filed[path.name]}")
    dW = {k: v.float().to(DEV) for k, v in torch.load(path, weights_only=True).items()}
    assert set(dW) == set(sites)
    return dW


def main() -> None:
    if not PRED.is_file() or (RESULTS / "stopped.json").is_file():
        sys.exit("refusing to start: predictions.json missing, or stopped.json exists")
    filed = json.loads((INPUTS / "regen.json").read_text())["files"]
    model = vpd.load_component_model(DEV)
    sites = vpd.module_keys(model)
    assert len(sites) == SITES_N and LAST in sites, sites
    comps = {k: (vpd.uv(model, k)[0].float(), vpd.uv(model, k)[1].float()) for k in sites}
    rows = [b.to(DEV) for b in batches(pile_rows("val", 0, ROWS, log), BATCH)]
    log(f"[load] {torch.cuda.get_device_name()}, {ROWS} rows, GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    with torch.no_grad():
        for name in ("adam", "lora"):
            dW = load_update(name, sites, filed)
            c = cached(RESULTS / name / "control.json", lambda n=name, d=dW: control(model, rows, comps, n, d))
            log(f"[{name} control] update {c['amp_dw']:.4f}, Gaussian {c['amp_gauss']:.4f}")
            for label in POINTS:
                r = cached(RESULTS / name / f"{label}.json", lambda lb=label, d=dW: point(model, rows, comps, sites, d, lb))
                log(f"[{name} {label}] amp {r['amp']:.6f}")
                if label == "fixed_none" and not abs(r["amp"] - 1) <= ID_TOL:
                    stop("control_identity", f"{name}: {r['amp']}")
            del dW
    log(f"[done] peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
    hard_exit(0)
