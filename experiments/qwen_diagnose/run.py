"""E73: why does a small update move Qwen3-1.7B so far, and is W0 itself the odd one? ($0, one cluster L40, debug QoS.)
Plan: reports/plans/2026-11-03_plan_e73-diagnose.md; rules: results/predictions.json. Clean model only, no decomposition.
Every read writes results/<read>.json as it ends (a rerun skips it) and logs its numbers to results/run.log.

Usage (repo root; -P and PYTHONPATH as cluster/qwen_diagnose.sbatch): .venv/bin/python -P experiments/qwen_diagnose/run.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

from commute import qwen_eval, qwen_sft
from commute.qwen_vpd import SITES, fp32_matmuls, load_target, parse_site

HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
DEV, F32 = "cuda", torch.float32
IM_START, USER, ASSISTANT = 151644, 872, 77091
Delta = Callable[[str], Tensor | None]


@dataclass(frozen=True)
class Cfg:
    full_rows: int = 128  # the control, as E72's
    rows: int = 64  # every other read; u3's set has 64
    batch: int = 8
    t_path: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 0.6, 1.0)
    t_neg: tuple[float, ...] = (-0.1, -1.0)
    t_null: tuple[float, ...] = (0.1, 0.3, 1.0)
    rel_noise: float = 0.002  # about half of bf16's largest relative rounding (1/256)
    top_positions: int = 30
    control_rel: float = 1e-3


C = Cfg()
E72 = {"nll_W0_minus_FT": 0.171, "K": 1.6351103354567915, "H_W0": 0.6102411561108624, "H_FT": 1.5081335160990383, "nll_FT": 1.89778957}


def load_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with (RESULTS / "run.log").open("a") as f:
        f.write(line + "\n")


def kl_sum(lp: Tensor, lq: Tensor) -> Tensor:
    """KL(p ‖ q) per position, from log-probabilities, [batch, T]."""
    return (lp.exp() * (lp - lq)).sum(-1)


def base_numbers(lq: Tensor, tokens: Tensor) -> dict[str, float]:
    """Sums: next-token NLL and accuracy over T - 1 positions, entropy and top probability over T, as E72's `entropy`."""
    nll = -lq[:, :-1].gather(-1, tokens[:, 1:, None]).squeeze(-1)
    top_p, top = lq.exp().max(-1)
    return {"nll": float(nll.double().sum()), "H": float(-(lq.exp() * lq).sum(-1).double().sum()),
            "acc": float((top[:, :-1] == tokens[:, 1:]).sum()), "top_p": float(top_p.double().sum()),
            "n": float(tokens.numel()), "n1": float(tokens[:, 1:].numel())}


def roles(tokens: Tensor) -> Tensor:
    """For each target token (position i + 1): 0 user turn, 1 assistant turn, 2 anything else (system, text outside chat)."""
    out = torch.full_like(tokens, 2)
    for r in range(tokens.shape[0]):
        cur = 2
        for i, tok in enumerate(tokens[r].tolist()):
            if i and tokens[r, i - 1] == IM_START:
                cur = {USER: 0, ASSISTANT: 1}.get(tok, 2)
            out[r, i] = cur
    return out[:, 1:]


class Reader:
    def __init__(self, model: Any):
        self.model, self.lins = model, qwen_sft.linears(model)
        self.W0 = {s: lin.weight.data.clone() for s, lin in self.lins.items()}

    def set(self, d: Delta | None, full: dict[str, Tensor] | None = None) -> None:
        for s, lin in self.lins.items():
            p = None if d is None else d(s)
            lin.weight.data.copy_(full[s] if full is not None else self.W0[s] if p is None else self.W0[s] + p)

    def logp(self, tokens: Tensor, d: Delta | None, autocast: bool = False, full: dict[str, Tensor] | None = None) -> Tensor:
        self.set(d, full)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            return self.model(tokens, use_cache=False).logits.float().log_softmax(-1)

    def read(self, rows: Tensor, d: Delta | None, ft: Delta | None = None, **kw: Any) -> dict[str, float]:
        """Means of the state's numbers; KL(W0 ‖ state), its agreement with W0, and the same against FT when `ft` is given."""
        s: dict[str, float] = {}
        for i in range(0, len(rows), C.batch):
            t = rows[i: i + C.batch].to(DEV)
            lq = self.logp(t, d, **kw)
            one = base_numbers(lq, t)
            for name, ref in (("W0", None), ("FT", ft)):
                if name == "FT" and ft is None:
                    continue
                lr = self.logp(t, ref)
                one[f"K_{name}"] = float(kl_sum(lr, lq).double().sum())
                one[f"agree_{name}"] = float((lr.argmax(-1) == lq.argmax(-1)).sum())
                del lr
            for k, v in one.items():
                s[k] = s.get(k, 0.0) + v
            del lq
        return {"nll": s["nll"] / s["n1"], "acc": s["acc"] / s["n1"]} | {
            k: s[k] / s["n"] for k in s if k not in ("nll", "acc", "n", "n1")}

    def detail(self, rows: Tensor, ft: Delta, tok: Any) -> dict[str, Any]:
        """Where W0 and FT differ: by turn (target's role), by W0's top probability; and the positions that move most."""
        keys: list[str] = ["nll_W0", "nll_FT", "H_W0", "H_FT", "acc_W0", "acc_FT", "top_p_W0", "K", "n"]
        by_role: dict[str, dict[str, float]] = {r: dict.fromkeys(keys, 0.0) for r in ("user", "assistant", "other")}
        by_conf: dict[str, dict[str, float]] = {b: dict.fromkeys(keys, 0.0) for b in ("p<0.5", "0.5-0.9", "0.9-0.99", ">0.99")}
        tops: list[tuple[float, str]] = []
        for i in range(0, len(rows), C.batch):
            t = rows[i: i + C.batch].to(DEV)
            l0, lf = self.logp(t, None)[:, :-1], self.logp(t, ft)[:, :-1]
            tgt = t[:, 1:]
            p0, top0 = l0.exp().max(-1)
            pf, topf = lf.exp().max(-1)
            per = {"nll_W0": -l0.gather(-1, tgt[..., None]).squeeze(-1), "nll_FT": -lf.gather(-1, tgt[..., None]).squeeze(-1),
                   "H_W0": -(l0.exp() * l0).sum(-1), "H_FT": -(lf.exp() * lf).sum(-1), "acc_W0": (top0 == tgt).float(),
                   "acc_FT": (topf == tgt).float(), "top_p_W0": p0, "K": kl_sum(lf, l0), "n": torch.ones_like(p0)}
            rl = roles(t)
            conf = torch.bucketize(p0, torch.tensor([0.5, 0.9, 0.99], device=DEV))
            for out, labels, names in ((by_role, rl, ("user", "assistant", "other")), (by_conf, conf, tuple(by_conf))):
                for j, name in enumerate(names):
                    m = labels == j
                    for k, v in per.items():
                        out[name][k] += float(v[m].double().sum())
            vals, idx = per["K"].flatten().topk(C.top_positions)
            for v, ix in zip(vals.tolist(), idx.tolist(), strict=True):
                r, p = ix // tgt.shape[1], ix % tgt.shape[1]
                ctx = tok.decode(t[r, max(0, p - 12): p + 1])
                tops.append((v, (f"K {v:.2f} | …{ctx!r} | target {tok.decode(tgt[r, p])!r} | W0 {tok.decode(top0[r, p])!r} {p0[r, p]:.3f} | "
                                f"FT {tok.decode(topf[r, p])!r} {pf[r, p]:.3f}")))
            del l0, lf
        tops = sorted(tops, reverse=True)[: C.top_positions]
        def mean(d: dict[str, float]) -> dict[str, float | None]:
            return {k: (v / d["n"] if d["n"] else None) for k, v in d.items() if k != "n"} | {"n": d["n"]}

        return {"by_role": {k: mean(v) for k, v in by_role.items()}, "by_conf": {k: mean(v) for k, v in by_conf.items()},
                "top_positions": [x[1] for x in tops]}


class Job:
    def __init__(self, rd: Reader, tok: Any, rows: dict[str, Tensor], updates: dict[str, Any]):
        self.rd, self.tok, self.rows, self.updates = rd, tok, rows, updates

    def step(self, key: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        f = RESULTS / f"{key}.json"
        if f.is_file():
            return json.loads(f.read_text())
        t0 = time.time()
        with torch.no_grad():
            r = fn()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r, indent=1))
        peak = torch.cuda.max_memory_allocated() / 1e9
        log(f"[{key}] {json.dumps(r)[:900]}  {time.time() - t0:.0f} s, peak GPU {peak:.1f} GB")
        return r

    def dw_of(self, name: str) -> dict[str, Tensor]:
        return {s: v.to(DEV, F32) for s, v in qwen_sft.load(self.updates[name].path).items() if s in SITES}

    def control(self) -> None:
        """u1's FT and W0 against E72's own numbers, the full rows."""
        dw, rows = self.dw_of("u1"), self.rows["maths"]
        ft = self.step("control", lambda: self.rd.read(rows, lambda s: dw[s]))
        w0 = self.step("control_W0", lambda: self.rd.read(rows, None))
        gap = {"H_FT": abs(ft["H"] / E72["H_FT"] - 1), "H_W0": abs(w0["H"] / E72["H_W0"] - 1),
               "nll_FT": abs(ft["nll"] * 255 / 256 / E72["nll_FT"] - 1)}  # E72 divides the T - 1 positions by T
        log(f"[control gaps] {gap}")
        if max(gap.values()) > C.control_rel:
            (RESULTS / "stopped.json").write_text(json.dumps(gap))
            sys.exit(f"control missed {gap}")

    def texts(self) -> None:
        for name, rows in self.rows.items():
            self.step(f"W0_on_{name}", lambda rows=rows: self.rd.read(rows[: C.rows], None))

    def precision(self) -> None:
        rd, rows = self.rd, self.rows["maths"][: C.rows]
        rounded = {s: v.to(torch.bfloat16).to(F32) for s, v in rd.W0.items()}
        self.step("W0_rounded_bf16", lambda: rd.read(rows, None, full=rounded))
        self.step("W0_autocast_bf16", lambda: rd.read(rows, None, autocast=True))
        gen = torch.Generator(DEV).manual_seed(0)
        noise = {s: v * C.rel_noise * torch.randn(v.shape, generator=gen, device=DEV, dtype=F32)
                 for s, v in rd.W0.items()}
        self.step("W0_rel_noise", lambda: rd.read(rows, lambda s: noise[s]))

    def nulls(self, name: str, dw: dict[str, Tensor], rows: Tensor) -> None:
        """Two changes with each matrix's norm: Gaussian, and δW's own entries shuffled within the matrix."""
        rd, gen = self.rd, torch.Generator(DEV).manual_seed(0)
        gauss = {}
        for s, v in dw.items():
            g = torch.randn(v.shape, generator=gen, device=DEV, dtype=F32)
            gauss[s] = g * v.norm() / g.norm()
        perm = {s: v.flatten()[torch.randperm(v.numel(), generator=gen, device=DEV)].view(v.shape) for s, v in dw.items()}
        for t in C.t_null:
            self.step(f"{name}/gauss_{t}", lambda t=t: rd.read(rows, lambda s, t=t: t * gauss[s], ft=lambda s: dw[s]))
            self.step(f"{name}/perm_{t}", lambda t=t: rd.read(rows, lambda s, t=t: t * perm[s], ft=lambda s: dw[s]))

    def blocks(self, name: str, dw: dict[str, Tensor], rows: Tensor) -> None:
        """Which matrices: one block at a time, attention and MLP apart."""
        rd = self.rd

        def only(keep: Callable[[str], bool]) -> dict[str, float]:
            return rd.read(rows, lambda s: dw[s] if keep(s) else None, ft=lambda s: dw[s])

        for layer in range(28):
            self.step(f"{name}/block_{layer}", lambda layer=layer: only(lambda s: parse_site(s)[0] == layer))
        for group, kinds in (("attn", ("q", "k", "v", "o")), ("mlp", ("gate", "up", "down"))):
            self.step(f"{name}/{group}", lambda kinds=kinds: only(lambda s: parse_site(s)[1] in kinds))

    def update(self, name: str) -> None:
        rd, dw, u = self.rd, self.dw_of(name), self.updates[name]
        rows = self.rows[u.rows][: C.rows]
        self.step(f"{name}/detail", lambda: rd.detail(rows, lambda s: dw[s], self.tok))
        for t in C.t_path + C.t_neg:
            self.step(f"{name}/dw_{t}", lambda t=t: rd.read(rows, lambda s, t=t: t * dw[s], ft=lambda s: dw[s]))
        self.nulls(name, dw, rows)
        if name == "u1":
            self.blocks(name, dw, rows)


def main() -> None:
    assert PRED.is_file(), "predictions first"
    fp32_matmuls()
    pr = load_file("pass_amend1", HERE.parent / "qwen_pass_test" / "amend1.py").pr
    rows256 = lambda p: torch.tensor([r[:256] for r in json.loads(p.read_text())["rows"]]) 
    rows = {"maths": rows256(HERE.parent / "qwen_sft" / "results" / "maths_rows.json"),
            "web": rows256(HERE.parent / "qwen_adversary" / "results" / "tokens.json"), "a1": pr.a1_rows()}
    tok = AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV)
    updates = {u.name: u for u in pr.P.updates}
    for u in updates.values():
        assert u.path.is_file(), u.path
    model = load_target(DEV, F32)
    log(f"[load] {torch.cuda.get_device_name()}; rows { {k: tuple(v.shape) for k, v in rows.items()} }")
    job = Job(Reader(model), tok, rows, updates)
    job.control()
    job.texts()
    job.precision()
    for name in ("u1", "u2", "u3", "u4"):
        job.update(name)
    log("[done]")


if __name__ == "__main__":
    main()
