"""E74: where does the masked forward lose a weight change, and how much of KL(FT ‖ W0) is one temperature? ($0, one cluster L40.)
Plan: reports/plans/2026-11-03_plan_e74-transmission.md; rules: rules.py, results/predictions.json.

Read 2 first (clean, no decomposition), then read 1's control with a stop, then read 1. A change enters as an unmasked term x dWᵀ
at its matrix. Transmission = KL(masked W0 ‖ masked W0 + change) / KL(W0 ‖ W0 + change), W0's masks fixed. Each group writes
results/<update>/<group>.json as it ends (a rerun skips it); every number goes to results/run.log.

Usage (repo root; -P and PYTHONPATH as cluster/qwen_transmission.sbatch):
  .venv/bin/python -P experiments/qwen_transmission/run.py --decomp DIR
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor

from commute import qwen_sft
from commute.jobs import hard_exit
from commute.qwen_vpd import (
    SITES,
    Components,
    _hooks,
    _linear,
    fp32_matmuls,
    importances,
    load_components,
    load_gamma,
    load_target,
    parse_site,
)

HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
DEV = "cuda"
DOWN27 = "layers.27.mlp.down_proj"
Change = dict[str, Tensor]  # site -> fp32 matrix, on the host


class Arm(NamedTuple):
    change: Change
    cut: int | None = None  # masks at layers up to this one, the model's own after it
    clean_x: bool = False  # Amendment 1: the added term reads the clean forward's inputs, not the masked forward's


@dataclass(frozen=True)
class Cfg:
    rows: int = 32  # read 1
    hook_rows: int = 64  # as E73's block read
    batch: int = 8  # clean forwards
    layers: tuple[int, ...] = (0, 7, 14, 21, 27)
    cuts: tuple[int, ...] = (0, 7, 14, 21)
    temps: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
    fit_rounds: int = 4
    fit_tol: float = 0.1
    hook_tol: float = 0.01  # E68's hook against the weights read 6e-10 of KL(FT ‖ W0) (amend1/u1_hook.json)
    same_tol: float = 1e-3  # a number another job read through the same path
    updates_read2: tuple[str, ...] = ("u1", "u2", "u3", "u4")
    updates_read1: tuple[str, ...] = ("u1", "u3")


C = Cfg()
T0 = time.time()


def load_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rules = load_file("transmission_rules", HERE / "rules.py")


def log(msg: str) -> None:
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with (RESULTS / "run.log").open("a") as f:
        f.write(line + "\n")


def stop(rule: str, detail: str) -> None:
    (RESULTS / "stopped.json").write_text(json.dumps({"rule": rule, "detail": detail}))
    sys.exit(f"STOP {rule}: {detail}")


def write(key: str, value: dict[str, Any]) -> None:
    path = RESULTS / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1))
    log(f"[{key}] written, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


# ----------------------------------------------------------------------------- read 2: the ruler at temperature


def read2_update(rd: Any, e73: Any, rows: Tensor, dw: Change) -> dict[str, Any]:
    """Per temperature: KL(FT ‖ W0 at T) over every position and by the target's turn, W0's NLL and entropy at T, KL(FT ‖ FT at T)."""
    gpu = {s: v.to(DEV) for s, v in dw.items()}
    s: dict[str, dict[str, float]] = {str(T): {} for T in C.temps}
    n: dict[str, float] = {}

    def add(d: dict[str, float], k: str, v: Tensor | float) -> None:
        d[k] = d.get(k, 0.0) + float(v)

    for i in range(0, len(rows), C.batch):
        t = rows[i: i + C.batch].to(DEV)
        l0, lf = rd.logp(t, None), rd.logp(t, lambda site: gpu[site])
        rl = e73.roles(t)
        add(n, "all", t.numel())
        add(n, "n1", t[:, 1:].numel())
        for j, name in enumerate(("user", "assistant")):
            add(n, name, (rl == j).sum())
        for T in C.temps:
            d, q = s[str(T)], (l0 / T).log_softmax(-1)
            k = e73.kl_sum(lf, q)
            add(d, "K", k.double().sum())
            for j, name in enumerate(("user", "assistant")):
                add(d, f"K_{name}", k[:, :-1][rl == j].double().sum())
            base = e73.base_numbers(q, t)
            add(d, "nll", base["nll"])
            add(d, "H", base["H"])
            add(d, "K_self", e73.kl_sum(lf, (lf / T).log_softmax(-1)).double().sum())
            del q, k
        del l0, lf
    out = {}
    for T, d in s.items():
        out[T] = {"K": d["K"] / n["all"], "K_user": d["K_user"] / n["user"], "K_assistant": d["K_assistant"] / n["assistant"],
                  "nll": d["nll"] / n["n1"], "H": d["H"] / n["all"], "K_self": d["K_self"] / n["all"]}
    return {"temps": out, "n": n}


def read2(e73: Any, model: Any, rows: dict[str, Tensor], updates: dict[str, Any]) -> None:
    rd = e73.Reader(model)
    for name in C.updates_read2:
        u, path = updates[name], RESULTS / name / "temperature.json"
        if path.is_file():
            continue
        dw = {s: v.float() for s, v in qwen_sft.load(u.path).items() if s in SITES}
        r = read2_update(rd, e73, rows[u.rows], dw)
        write(f"{name}/temperature", r)
        log(f"[read2 {name}] " + json.dumps(r["temps"]))
        if name == "u1":  # W0 at T = 1 is E72's clean read; FT at T = 1 is 0, above it not
            want = sum(json.loads((HERE.parent / "qwen_fixed_masks/results/u1/clean_W0.json").read_text())["sums"]["K"]) / (128 * 256)
            got, self1 = r["temps"]["1.0"]["K"], r["temps"]["1.0"]["K_self"]
            log(f"[control temperature] K {got:.9g} vs E72 {want:.9g}; K_self at 1 {self1:.3g}, at 2 {r['temps']['2.0']['K_self']:.4g}")
            if abs(got / want - 1) > C.same_tol or self1 > 1e-6 or r["temps"]["2.0"]["K_self"] <= 0:
                stop("control_temperature", f"{got} vs {want}, K_self {self1}")
    rd.set(None)
    del rd
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- read 1: transmission


def forward(model: Any, tokens: Tensor, comps: Components, g: dict[str, Tensor] | None, dW: Change, cut: int | None = None,
            tap: dict[str, Tensor] | None = None, term_x: dict[str, Tensor] | None = None) -> Tensor:
    """Each matrix's output: masked by g at layers up to `cut` (None: all; g None: none), the model's own elsewhere; plus x dWᵀ.
    `tap` collects the changed matrices' inputs; `term_x` replaces the input the term reads."""

    def replace(site: str) -> Any:
        V, U = comps.vu(site)
        masked = g is not None and (cut is None or parse_site(site)[0] <= cut)
        d = dW.get(site)

        def hook(m: Any, args: tuple[Tensor, ...], out: Tensor) -> Tensor:
            x = args[0].to(V.dtype)
            base = (x @ V * g[site].to(V.dtype)) @ U if masked and g is not None else out.to(V.dtype)
            if d is None:
                return base.to(out.dtype)
            if tap is not None:
                tap[site] = x
            return (base + (x if term_x is None else term_x[site]) @ d.to(x).T).to(out.dtype)

        return hook

    with _hooks([_linear(model, s).register_forward_hook(replace(s)) for s in SITES]):
        return model(tokens, use_cache=False).logits


def entropy(logits: Tensor) -> float:
    lp = logits.float().log_softmax(-1)
    return float(-(lp.exp() * lp).sum(-1).double().sum())


def k_clean(pr: Any, model: Any, rows: Tensor, comps: Components, change: Change) -> float:
    total = 0.0
    for tokens in rows:  # one row: beside the decomposition the L40 has ~1 GB free, a batch of 8 rows' logits is 1.16 GiB (job 50507)
        t = tokens[None].to(DEV)
        total += pr.kl(model(t, use_cache=False).logits, forward(model, t, comps, None, change))
    return total / rows.numel()


def gaussian_like(pr: Any, model: Any, rows: Tensor, comps: Components, change: Change, seed: int) -> tuple[Change, dict[str, float]]:
    """Gaussian in the same matrices with each matrix's norm in proportion to the change's, scaled to its clean KL (quadratic rule)."""
    gen = torch.Generator().manual_seed(seed)
    unit = {}
    for s, v in change.items():
        g = torch.randn(v.shape, generator=gen)
        unit[s] = g * v.norm() / g.norm()
    target, a, got = k_clean(pr, model, rows, comps, change), 1.0, 0.0
    for _ in range(C.fit_rounds):
        got = k_clean(pr, model, rows, comps, {s: a * v for s, v in unit.items()})
        if abs(got / target - 1) <= C.fit_tol:
            break
        a *= (target / got) ** 0.5
    return {s: a * v for s, v in unit.items()}, {"scale": a, "K_clean_fit": got, "K_clean_target": target}


def read_group(pr: Any, model: Any, gamma: Any, comps: Components, rows: Tensor, arms: dict[str, Arm]) -> dict[str, Any]:
    """Row by row, W0's masks fixed: KL clean and masked for each arm, and the logit shifts' dot and squared norms (centred per position)."""
    sums: dict[str, dict[str, list[float]]] = {a: {} for a in arms}
    base_sums: dict[str, list[float]] = {"K0": [], "H_clean": [], "H_masked": []}
    for tokens in rows[: C.rows]:
        t = tokens[None].to(DEV)
        clean, g = importances(model, t, gamma, comps)
        masked = forward(model, t, comps, g, {})
        base_sums["K0"].append(pr.kl(clean, masked))
        base_sums["H_clean"].append(entropy(clean))
        base_sums["H_masked"].append(entropy(masked))
        for name, (dW, cut, clean_x) in arms.items():
            base = masked if cut is None else forward(model, t, comps, g, {}, cut)  # each cut is one arm's
            tap: dict[str, Tensor] | None = {} if clean_x else None
            a = forward(model, t, comps, None, dW, tap=tap)
            one = {"K_clean": pr.kl(clean, a)}
            a.sub_(clean)  # in place: the reader alone peaks at 42.7 of the L40's 44.3 GiB
            a.sub_(a.mean(-1, keepdim=True))
            b = forward(model, t, comps, g, dW, cut, term_x=tap)
            one["K_masked"] = pr.kl(base, b)
            b.sub_(base)
            b.sub_(b.mean(-1, keepdim=True))
            del base
            one |= {"dot": float((a * b).double().sum()), "aa": float((a * a).double().sum()), "bb": float((b * b).double().sum())}
            for k, v in one.items():
                sums[name].setdefault(k, []).append(v)
            del a, b
        del clean, g, masked
    return {"T": rows.shape[1], "base": base_sums, "arms": sums}


def summary(r: dict[str, Any]) -> dict[str, dict[str, float]]:
    out = {}
    n = len(r["base"]["K0"]) * r["T"]
    for name, s in r["arms"].items():
        kc, km = sum(s["K_clean"]) / n, sum(s["K_masked"]) / n
        aa, bb = sum(s["aa"]), sum(s["bb"])
        out[name] = {"K_clean": kc, "K_masked": km, "tau": rules.tau(km, kc), "cos": sum(s["dot"]) / (aa * bb) ** 0.5 if aa * bb > 0 else float("nan"),
                     "amp": (bb / aa) ** 0.5 if aa > 0 else float("nan")}
    return out


class Read1:
    def __init__(self, pr: Any, model: Any, decomp: Path):
        self.pr, self.model = pr, model
        self.comps = load_components(decomp, DEV, torch.float32)
        self.gamma = load_gamma(decomp, DEV, torch.bfloat16)
        log(f"[load] decomposition, GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def group(self, name: str, key: str, rows: Tensor, arms: dict[str, Arm], extra: dict[str, Any]) -> dict[str, Any]:
        path = RESULTS / name / f"{key}.json"
        if path.is_file():
            return json.loads(path.read_text())
        t0 = time.time()
        r = read_group(self.pr, self.model, self.gamma, self.comps, rows, arms) | extra
        r["summary"] = summary(r)
        write(f"{name}/{key}", r)
        n = len(r["base"]["K0"]) * r["T"]
        log(f"[{name}/{key}] K0 {sum(r['base']['K0']) / n:.6g} H_clean {sum(r['base']['H_clean']) / n:.4g} H_masked {sum(r['base']['H_masked']) / n:.4g} "
            f"{time.time() - t0:.0f} s {json.dumps(r['summary'])}")
        return r

    def fit(self, rows: Tensor, change: Change, seed: int) -> tuple[Change, dict[str, float]]:
        return gaussian_like(self.pr, self.model, rows[: C.rows], self.comps, change, seed)

    def control(self, u: Any, rows: Tensor, dw: Change) -> None:
        """A change entering just before the final norm must pass; the hook reads as E73's weights; K0 as E68's. Stops here."""
        if u.name == "u1":
            blk = {s: v for s, v in dw.items() if parse_site(s)[0] == 27}
            got = k_clean(self.pr, self.model, rows[: C.hook_rows], self.comps, blk)
            want = json.loads((HERE.parent / "qwen_diagnose/results/u1/block_27.json").read_text())["K_W0"]
            log(f"[control hook] block 27 clean KL {got:.9g} vs E73 {want:.9g}")
            if abs(got / want - 1) > C.hook_tol:
                stop("control_hook", f"{got} vs {want}")
        d27 = {DOWN27: dw[DOWN27]}
        gauss, fit = self.fit(rows, d27, 27_000)
        r = self.group(u.name, "control", rows, {"dw_down27": Arm(d27), "gauss_down27": Arm(gauss)}, {"fit": {"gauss_down27": fit}})
        ref = sum(json.loads((self.pr.RESULTS / u.name / "W0.json").read_text())["sums"]["ci"][: C.rows])
        got = sum(r["base"]["K0"])
        log(f"[control K0] {got:.9g} vs E68 {ref:.9g}")
        if abs(got / ref - 1) > C.same_tol:
            stop("control_K0", f"{got} vs {ref}")
        a_d, a_g = r["summary"]["dw_down27"]["amp"], r["summary"]["gauss_down27"]["amp"]
        log(f"[control transmission] {u.name} amplitude: dw {a_d:.4g} gauss {a_g:.4g}")
        if not rules.control(a_d, a_g):
            stop("control_transmission", f"{u.name}: amplitude dw {a_d}, gauss {a_g}: band {rules.BAND}")

    def update(self, u: Any, rows: Tensor, dw: Change) -> None:
        self.control(u, rows, dw)
        arms: dict[str, Arm] = {}
        fits = {}
        for layer in C.layers:
            blk = {s: v for s, v in dw.items() if parse_site(s)[0] == layer}
            gauss, fits[f"gauss_{layer}"] = self.fit(rows, blk, layer)
            log(f"[fit {u.name} {layer}] {fits[f'gauss_{layer}']}")
            arms[f"dw_{layer}"], arms[f"gauss_{layer}"] = Arm(blk), Arm(gauss)
        r = self.group(u.name, "blocks", rows, arms, {"fit": fits})
        cuts = {f"cut_{c}": Arm({s: v for s, v in dw.items() if parse_site(s)[0] == c}, c) for c in C.cuts}
        rc = self.group(u.name, "cuts", rows, cuts, {})
        ra = self.group(u.name, "all", rows, {"all": Arm(dw)}, {})
        s, sc = r["summary"], rc["summary"]
        verdict = {"where_lost": rules.where_lost(s["dw_0"]["tau"], sc["cut_0"]["tau"]),
                   "against_gaussian": rules.against_gaussian({
                       str(a): (s[f"dw_{a}"]["tau"], s[f"gauss_{a}"]["tau"], s[f"dw_{a}"]["K_clean"], s[f"gauss_{a}"]["K_clean"]) for a in rules.EARLY}),
                   "tau_all": ra["summary"]["all"]["tau"]}
        (RESULTS / u.name / "verdict.json").write_text(json.dumps(verdict, indent=1))
        log(f"[verdict {u.name}] {json.dumps(verdict)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    a = ap.parse_args()
    if not PRED.is_file() or (RESULTS / "stopped.json").is_file():
        sys.exit("refusing to start: predictions.json missing, or stopped.json exists")
    fp32_matmuls()
    e73 = load_file("e73_run", HERE.parent / "qwen_diagnose" / "run.py")
    pr = load_file("pass_amend1", HERE.parent / "qwen_pass_test" / "amend1.py").pr
    rows256 = torch.tensor([r[:256] for r in json.loads((HERE.parent / "qwen_sft/results/maths_rows.json").read_text())["rows"]])
    rows = {"maths": rows256, "a1": pr.a1_rows()}
    updates = {u.name: u for u in pr.P.updates}
    model = load_target(DEV, torch.float32)
    log(f"[load] {torch.cuda.get_device_name()}; rows { {k: tuple(v.shape) for k, v in rows.items()} }")
    with torch.no_grad():
        read2(e73, model, rows, updates)
        for name in ("u1", "u2", "u3", "u4"):
            temps = json.loads((RESULTS / name / "temperature.json").read_text())["temps"]
            s = rules.sharpness({k.removesuffix(".0"): v["K"] for k, v in temps.items()}, template=name == "u3")
            log(f"[sharpness {name}] {json.dumps(s)}")
            (RESULTS / name / "sharpness.json").write_text(json.dumps(s, indent=1))
        rd1 = Read1(pr, model, Path(a.decomp))
        for name in C.updates_read1:
            u = updates[name]
            rd1.update(u, rows[u.rows], {s: v.float() for s, v in qwen_sft.load(u.path).items() if s in SITES})
    log("[done]")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    hard_exit(code)
