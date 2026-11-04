"""E85 Amendment 2: does Γ's bf16 rounding of V limit what it hears of a placement? One `debug` QoS L40, 5-10 min, $0. Plan: reports/plans/2026-11-04_plan_e85-adversary-teeth.md, "Amendment 2"; rules:
rules_amend.py, results/predictions_amend2.json (the job refuses to start without it, and while results/stopped_amend2.json exists). Nothing in qwen_vpd.py changes: this read calls its pieces.

On the first 8 of E85's web rows, g from the code's own `backbone` and `readout`, for: W0 as run; W0 again through `importances`, the code's whole path (the repeat floor, and the check that this read is
that path); W0 with the readout in fp32 (the rounding floor); the rarely-on placement as run, with the readout in fp32, and with V held at W0's (activations alone); E85's random change. "fp32" computes h
and the readout in fp32, from the fp32 V, so nothing rounds V or h; taps and the backbone stay bf16. Every comparison is reported on g as the masks read it (clipped) and on the sum before the clip, on the
alive columns and on the rarely-on half's. Also the clean KL against W0 of the fine-tune and of the random change at 1, 30 and 100 times, which tells Amendment 1 whether its state moves the output, for $0.

Memory. The card holds the target (6.9 GB), Γ in bf16 (17.1) and two more sets of weights (5.6 each): 35.2 GB, plus ~4.5 per row, so ~40 GB on a 46 GB L40. V stays on the host, a site at a time on the card.
Host ~46 GB at its peak: components 19.5, alive copies ~14, δW 2.8, the target's load 6.9.

Usage (repo root, L40): sbatch cluster/qwen_d1_teeth_amend2.sbatch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import rules_amend as rules
import run as e85
import torch
from torch import Tensor

from commute import qwen_e40_files, qwen_maps, qwen_vpd
from commute.qwen_sft import linears, load, mean_kl, plus, set_weights, sha256
from commute.qwen_vpd import SITES, parse_site
from commute.vpd import kl_per_position

e80, e76 = e85.e80, e85.e76
RESULTS = Path(__file__).resolve().parent / "results"
PRED, OUT, STOPPED, LOG = (RESULTS / f for f in ("predictions_amend2.json", "amend2.json", "stopped_amend2.json", "run_amend2.log"))
N_ROWS, GPU_MIN_GIB, SCALES = 8, 44, (1, 30, 100)
F32 = torch.float32
COMPARE = ("W0_level", "W0_repeat", "W0_fp32", "Rare", "Rare_fp32", "Rare_move_fp32", "Held", "Rare_V_route", "Rand")  # what each reads: `read_row`


def pool(ts: list[list[float]]) -> dict[str, float]:
    """Mean |a - b| clipped and before the clip, and the share of pairs over MOVE, from Tally's sums; NaN for no pairs."""
    size = sum(t[3] for t in ts)
    if not size:
        return {"g": rules.NAN, "before_clip": rules.NAN, "share_over": rules.NAN}
    return {"g": sum(t[0] for t in ts) / size, "before_clip": sum(t[1] for t in ts) / size, "share_over": sum(t[2] for t in ts) / size}


class Tally:
    """Per comparison, on `alive` and `half`: Σ|a - b| clipped and before the clip, pairs moving over MOVE, and the count, by site and by row."""

    def __init__(self, cols: dict[str, dict[str, Tensor]]):
        self.cols = cols
        self.sums: dict[tuple[str, str, str, int], list[float]] = {}

    def add(self, name: str, s: str, row: int, a: Tensor, b: Tensor) -> None:
        pre, clip = (a - b).abs(), (a.clamp(0, 1) - b.clamp(0, 1)).abs()
        for sub, cols in self.cols.items():
            d, p = clip[..., cols[s]], pre[..., cols[s]]
            t = self.sums.setdefault((name, sub, s, row), [0.0, 0.0, 0.0, 0.0])
            t[0] += float(d.sum(dtype=torch.float64))
            t[1] += float(p.sum(dtype=torch.float64))
            t[2] += float((d > rules.MOVE).sum())
            t[3] += d.numel()

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in sorted({k[0] for k in self.sums}):
            mine = {k: t for k, t in self.sums.items() if k[0] == name}
            for sub in self.cols:
                got = [(k[2], k[3], t) for k, t in mine.items() if k[1] == sub]
                rows = sorted({r for _, r, _ in got})
                out.setdefault(name, {})[sub] = {**pool([t for _, _, t in got]), "by_row": [pool([t for _, r, t in got if r == i])["g"] for i in rows]}
            got = [(k[2], t) for k, t in mine.items() if k[1] == "alive"]
            by_site: dict[str, list[list[float]]] = {}
            for s, t in got:
                by_site.setdefault(s, []).append(t)
            out[name]["by_site"] = {s: [round(v, 8) for v in pool(ts).values()] for s, ts in by_site.items()}
        return out


def fp32_readout(gamma: Any, s: str, x: Tensor, V: Tensor) -> Tensor:
    """`qwen_vpd.readout` itself with the dtype it casts to set to fp32, so h and the sum are fp32. `site_importance`'s sum then promotes to fp32 too, and g has no rounding on this route."""
    view: Any = SimpleNamespace(dtype=F32, readouts={s: gamma.readouts[s].to(F32)})
    return qwen_vpd.readout(view, s, x, V)


def before_clip(gamma: Any, x: Tensor, s: str, r: Tensor) -> Tensor:
    """`qwen_vpd.site_importance` without its clamp; g is this clipped to [0, 1]."""
    layer, kind = parse_site(s)
    return (x @ gamma.heads_w[kind][layer] + gamma.heads_b[kind][layer] + r).float()


@torch.no_grad()
def forward(J: Any, tokens: Tensor, weights: dict[str, Tensor] | None) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """The target's clean forward at `weights`: its logits, Γ's backbone output, and each site's input (`qwen_vpd.clean_forward`'s hooks, the readouts left to the caller)."""
    taps: list[Tensor] = []
    inputs: dict[str, Tensor] = {}
    handles = [J.model.model.embed_tokens.register_forward_hook(lambda m, a, o: taps.append(o))]
    handles += [layer.register_forward_hook(lambda m, a, o: taps.append(o)) for layer in J.model.model.layers]
    lins = linears(J.model)

    def capture(s: str) -> Any:
        return lambda m, a, o: inputs.__setitem__(s, a[0])

    handles += [lins[s].register_forward_hook(capture(s)) for s in SITES]
    try:
        with set_weights(J.model, weights):
            logits = J.model(tokens, use_cache=False).logits
    finally:
        for h in handles:
            h.remove()
    return logits, qwen_vpd.backbone(J.gamma, taps), inputs


def moved_v(J: Any, s: str) -> Tensor:
    """The site's V on the card with the half's columns moved, as `qwen_maps.applied` writes them."""
    V = J.view.vu(s)[0].clone()
    V[:, J.half[s]] = V[:, J.half[s]] + J.dV[s].to(V)
    return V


@torch.no_grad()
def read_row(J: Any, row: int, tokens: Tensor, T: Tally) -> dict[str, float]:
    """One row. Each weights set is one forward; the g's it gives are compared site by site and dropped, W0's two being held for the row."""
    gamma = J.gamma
    l0, x0, in0 = forward(J, tokens, None)
    repeat = qwen_vpd.importances(J.model, tokens, gamma, J.view)[1]
    g0, g0_32 = {}, {}
    for s in SITES:
        V = J.view.vu(s)[0]
        g0[s] = before_clip(gamma, x0, s, qwen_vpd.readout(gamma, s, in0[s], V))
        g0_32[s] = before_clip(gamma, x0, s, fp32_readout(gamma, s, in0[s], V))
        T.add("W0_level", s, row, g0[s].clamp(0, 1), torch.zeros_like(g0[s]))
        T.add("W0_repeat", s, row, g0[s].clamp(0, 1), repeat[s])
        T.add("W0_fp32", s, row, g0_32[s], g0[s])
    del repeat, x0, in0
    l1, x1, in1 = forward(J, tokens, J.ft)
    for s in SITES:
        V, Vr = J.view.vu(s)[0], moved_v(J, s)
        held = before_clip(gamma, x1, s, qwen_vpd.readout(gamma, s, in1[s], V))
        rare = before_clip(gamma, x1, s, qwen_vpd.readout(gamma, s, in1[s], Vr))
        rare_32 = before_clip(gamma, x1, s, fp32_readout(gamma, s, in1[s], Vr))
        T.add("Rare", s, row, rare, g0[s])
        T.add("Rare_fp32", s, row, rare_32, rare)
        T.add("Rare_move_fp32", s, row, rare_32, g0_32[s])
        T.add("Held", s, row, held, g0[s])
        T.add("Rare_V_route", s, row, rare, held)
    del x1, in1, g0_32
    l2, x2, in2 = forward(J, tokens, J.rand)
    for s in SITES:
        T.add("Rand", s, row, before_clip(gamma, x2, s, qwen_vpd.readout(gamma, s, in2[s], J.view.vu(s)[0])), g0[s])
    return {"FT": float(kl_per_position(l0, l1).double().mean()), "Rand": float(kl_per_position(l0, l2).double().mean())}


def set_up(decomp: Path, inputs: Path, work: Path) -> Any:
    assert torch.cuda.get_device_properties(0).total_memory > GPU_MIN_GIB * 2**30, "under 44 GiB on the card"
    got, want = sha256(e76.DW_FILE), qwen_e40_files.sums(e76.DW_SUMS)[e76.DW_FILE.name]
    e80.log(f"[dW] {e76.DW_FILE}: sha256 {got}, want {want}")
    assert got == want, "δW is not the file E85 used"
    qwen_vpd.fp32_matmuls()
    comps = qwen_vpd.load_components(decomp, "cpu", F32)
    alive = qwen_maps.Alive(comps, qwen_maps.alive_from_stats(load(inputs / "W0_maths.safetensors"), load(inputs / "W0_fineweb.safetensors")))
    assert alive.count() == e80.ALIVE_TOTAL, f"alive set {alive.count()}"
    dw = {s: v for s, v in load(e76.DW_FILE).items() if s in SITES}
    model = qwen_vpd.load_target(e85.DEV, F32)
    gamma = qwen_vpd.load_gamma(decomp, e85.DEV, torch.bfloat16)
    def v_only(s: str) -> tuple[Tensor, None]:  # build_states clears comps.U, so comps.vu cannot be called after it
        layer, kind = qwen_vpd.parse_site(s)
        return comps.V[kind][layer].to(e85.DEV), None

    view = SimpleNamespace(vu=v_only)
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    web = json.loads(e76.WEB.read_text())["rows"][:N_ROWS]
    J = SimpleNamespace(model=model, comps=comps, gamma=gamma, view=view, alive=alive, dw=dw, W0=W0, inputs=inputs, x_dir=work / "X", out={"checks": {}},
                        rows=[torch.tensor([r], device=e85.DEV) for r in web])
    e80.log(f"[load] {torch.cuda.get_device_name()}, {e80.peak()}")
    return J


def build_states(J: Any) -> None:
    """The half's columns and moves on the host, from E85's own `rare_change`; the fine-tune's and the random change's weights on the card; then the U and the alive copies go."""
    e85.make_rare(J)
    J.half, J.dV, ft_change, rand_change = {}, {}, {}, {}
    for s in SITES:
        M, pos, _, dV = e85.rare_change(J, s)
        assert M is not None and pos is not None and dV is not None
        J.half[s], J.dV[s] = J.alive.idx[s][pos.cpu()].to(e85.DEV), dV.float().cpu()
        G = e85.rand_change(J, s)[0]
        assert G is not None
        ft_change[s], rand_change[s] = M.float().cpu(), G.float().cpu()  # an F64 set on the card would not fit beside Γ
    J.ft, J.rand = plus(J.W0, ft_change), plus(J.W0, rand_change)
    J.cols = {"alive": {s: J.alive.idx[s].to(e85.DEV) for s in SITES}, "half": J.half}
    J.comps.U.clear()
    del J.alive
    torch.cuda.empty_cache()
    e80.log(f"[states] built, {e80.peak()}")


def stage_clean(J: Any) -> None:
    """Clean KL of the random change at each scale against W0, on the rows: the go or no-go for Amendment 1's state, whose start cannot fit this card."""
    J.rand = None
    for k in SCALES:
        w = {s: J.W0[s] + (k * qwen_maps.gaussian_at_norm(J.dw[s].to(e85.DEV, e85.F64), e85.rules.GAUSS_SEED + SITES.index(s))).to(J.W0[s]) for s in SITES}
        J.out.setdefault("clean_rand", {})[str(k)] = mean_kl(J.model, J.rows, J.W0, w)
        del w
        e80.log(f"[clean Rand x{k}] KL against W0 {J.out['clean_rand'][str(k)]:.6f}, {e80.peak()}")
    e80.save(J)


def file_active_share(J: Any, inputs: Path) -> float:
    """W0_fineweb's pooled active share over the alive columns, against which `W0_level`'s share over MOVE checks that g reads as the file does."""
    st = load(inputs / "W0_fineweb.safetensors")
    total = sum(float(st[f"{s}.active"][J.cols["alive"][s].cpu()].double().sum()) for s in SITES)
    return total / sum(len(J.cols["alive"][s]) for s in SITES)


def stage_checks(J: Any, kl_ft: float, file_share: float) -> None:
    s = J.out["summary"]
    rows, half = s["W0_level"]["alive"]["share_over"], s["W0_level"]["half"]["share_over"]
    e80.log(f"[active share] rows {rows}, the half {half}, file {file_share}")
    e80.check(J, "active_share", rows / file_share, rules.ACTIVE_LOW <= rows / file_share <= rules.ACTIVE_HIGH, f"{rules.ACTIVE_LOW} to {rules.ACTIVE_HIGH}: g reads as the file")
    e80.check(J, "half_is_rarer", half / rows, half < rows, "under 1")
    f1, f2, b = s["W0_repeat"]["half"]["g"], s["W0_fp32"]["half"]["g"], s["Rare_fp32"]["half"]["g"]
    e80.check(J, "floors", f2, rules.floors_ok(f1, f2), f"W0 with the fp32 readout differs from as run, and at least as much as W0 read twice ({f1})")
    e80.check(J, "FT_clean", kl_ft, rules.FT_KL_LOW <= kl_ft <= rules.FT_KL_HIGH, f"{rules.FT_KL_LOW} to {rules.FT_KL_HIGH}, E79's rows read 0.040")
    e80.log(f"[reading] {rules.reading(b, f1, f2)}: Rare_fp32 {b}, floors {f1}, {f2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    a = ap.parse_args()
    if not PRED.is_file() or STOPPED.is_file():
        sys.exit("refusing to start: predictions_amend2.json missing, or stopped_amend2.json exists")
    t0 = time.time()
    J = set_up(a.decomp, a.inputs, a.work)
    build_states(J)
    T = Tally(J.cols)
    kls = []
    for i, tokens in enumerate(J.rows):
        kls.append(read_row(J, i, tokens, T))
        e80.log(f"[row {i}] {json.dumps(kls[-1])}, {time.time() - t0:.0f} s, {e80.peak()}")
    J.out["summary"], J.out["clean_rows"] = T.summary(), kls
    e80.save(J)
    e80.log(f"[summary] {json.dumps({n: {u: {k: v for k, v in x.items() if k != 'by_row'} for u, x in J.out['summary'][n].items() if u != 'by_site'} for n in COMPARE})}")
    stage_clean(J)
    J.out["peak_gb"], J.out["s"] = torch.cuda.max_memory_allocated() / 1e9, time.time() - t0
    e80.save(J)
    stage_checks(J, sum(r["FT"] for r in kls) / len(kls), file_active_share(J, a.inputs))
    e80.log(f"[done] {e80.peak()}")


def score() -> None:
    """From what amend2.json holds: a stop or a crash still reads what ran."""
    out = e76.read_json(OUT)
    if out is None or "summary" not in out:
        return
    s = out["summary"]
    out["score"] = {"reading": rules.reading(s["Rare_fp32"]["half"]["g"], s["W0_repeat"]["half"]["g"], s["W0_fp32"]["half"]["g"]),
                    "half": {n: s[n]["half"]["g"] for n in COMPARE}, "alive": {n: s[n]["alive"]["g"] for n in COMPARE}, "share_over_half": {n: s[n]["half"]["share_over"] for n in COMPARE},
                    "clean_rand": out.get("clean_rand", {}), "clean_ft_rows": [r["FT"] for r in out["clean_rows"]], "peak_gb": out.get("peak_gb"), "checks": out["checks"]}
    e76.write_json(OUT, out)
    e80.log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    e80.OUT, e80.STOPPED, e80.LOG = OUT, STOPPED, LOG  # E80's save, stop, check and log read these globals, so no earlier job's files are written
    e80.guarded(main, score)
