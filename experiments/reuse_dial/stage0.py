"""E### stage 0 (run.py's docstring lists (a)-(f)): the controls every pod runs first and alone, on its first task."""

from __future__ import annotations

import math
from typing import Any

import torch

from commute import dial as DL
from commute import finetune as FT
from commute import kstar_vpd as KV
from commute import needed as IX
from commute import realistic_reads as RR
from commute import stand_in as SI
from commute import vpd_gains as VG
from commute.jobs import load_experiment
from commute.realistic import update_size

KN: Any = load_experiment("kstar_needed", "run")  # E268: featurized (the do-nothing item)
S1: Any = load_experiment("rl_67m", "run_stage1")  # kl_from_base


def random_dial(W: Any, gen: torch.Generator, residual: bool) -> DL.Dial:
    """Gains 1 + sd N(0, 1) on every subcomponent and remainder; with residual, a Gaussian residual of size sd / 10 of W0's."""
    sd, b = W.cfg.rand_sd, W.b
    g = VG.VGains({k: 1 + sd * torch.randn(b.U[k].shape[0], generator=gen).to(W.net.device) for k in b.keys},
                  {k: 1 + sd * torch.randn((), generator=gen).to(W.net.device) for k in b.keys})  # fmt: skip
    if not residual:
        return DL.Dial(g, None)
    res = {k: torch.randn(b.R[k].shape, generator=gen).to(W.net.device) for k in b.keys}
    scale = 0.1 * sd * math.sqrt(W.w0sq / sum(float(r.double().pow(2).sum()) for r in res.values()))
    return DL.Dial(g, {k: r * scale for k, r in res.items()})


@torch.no_grad()
def bits(W: Any, t: Any) -> dict[str, Any]:
    """(a): the start of the full dial (gains and residual) against the base model, fp32 and bf16 autocast."""
    d, x = DL.Dial.start(W.b, True, True), t.read.toks[: W.cfg.gain_batch, : t.read.pos + 1].to(W.net.device)
    base, ac = W.net.run(x)[0], FT.logits_of(W.net.net, x)
    with DL.applied(W.net.net, W.b, d):
        same, same_ac = torch.equal(W.net.run(x)[0], base), torch.equal(FT.logits_of(W.net.net, x), ac)
    return {"fp32_equal": same, "autocast_equal": same_ac, "delta_nonzero": sum(int(v.count_nonzero()) for v in DL.delta(W.b, d).values())}


@torch.no_grad()
def export(W: Any, t: Any, gen: torch.Generator) -> dict[str, float]:
    """(b): a random dial's exported dW gives its hooked logit change at the read position; the gains shuffled do not."""
    d, x, p = random_dial(W, gen, True), t.read.toks[: W.cfg.gain_batch, : t.read.pos + 1].to(W.net.device), t.read.pos
    assert d.g is not None
    base = W.net.run(x)[0][:, p]
    with DL.applied(W.net.net, W.b, d):
        hooked = W.net.run(x)[0][:, p] - base
    shuf = DL.Dial(VG.VGains({k: v[torch.randperm(len(v), generator=gen).to(v.device)] for k, v in d.g.comp.items()}, d.g.rem), d.res)

    def rel(dd: DL.Dial) -> float:
        with W.net.at(DL.delta(W.b, dd)):
            return float((W.net.run(x)[0][:, p] - base - hooked).norm() / hooked.norm())

    return {"equiv": rel(d), "shuffled": rel(shuf), "hooked_norm": float(hooked.norm())}


def zeros(W: Any, t: Any) -> dict[str, Any]:
    """(c): a zero update: every reader numerator and gain exactly 0 (E211's zero check), the KL exactly 0."""
    z = {k: torch.zeros_like(w).cpu() for k, w in W.net.W0.items()}
    rec = t.reader.state(z, 0)
    return {"nonzero_numerators": {k: v for k, v in RR.numerators(rec).items() if v != 0}, "kl": S1.kl_from_base(W.net, W.kl_rows, z, W.cfg.kl_batch)}


def do_nothing(W: Any) -> dict[str, Any]:
    """(d): F = M (E268's do-nothing item): no changed variable, every dd exactly 0, E_b exactly 1."""
    F, feat, toks, _ = KN.featurized(W.w, "do_nothing")
    num, den, dd = SI.write_change(F.net, SI.vpd_sites(feat), feat.n, list(toks.split(W.kc.cache_batch)))
    ch = torch.zeros(feat.n, dtype=torch.bool)
    ch[sorted(feat.changed)] = True
    return {"n_changed": len(feat.changed), "dd_max": float(dd.abs().max()), "E_b": SI.explained_share(num, den, dd, ch)}


def construction(W: Any, t: Any, gen: torch.Generator, eb_state: Any, save_dw: Any, tag: str) -> dict[str, Any]:
    """(e), (f): a random gains-only state and a Gaussian at its norms, saved in bf16 as every state is, read on the
    task's D: E_b over every changed variable; then E268's identity scan on the gains-only F (floor and rate)."""
    g = DL.delta(W.b, random_dial(W, gen, False))
    out: dict[str, Any] = {}
    for key, dW in (("gains", g), ("gauss", FT.random_direction(g, W.cfg.random_seed))):
        rec = save_dw(W, f"stage0_{tag}_{key}", dW)
        out[key] = eb_state(W, t, rec, f"stage0_{tag}_{key}", scan=False) | {"size": update_size(dW, W.net.W0)}
    F, M, toks, at, _ = W.w.from_dw(W.large / "states" / f"stage0_{tag}_gains.pt", t.D, t.at)
    feat = KV.featurize(M, F, W.w.uv, W.w.delta, W.w.bases)
    kc = W.kc
    idn = IX.identity_scan(F, feat, toks, at, kc.examples, kc.ident_vars, kc.rows, kc.logit_rows, kc.cache_batch, print)
    est = feat.n * kc.derangements * min(kc.examples, len(toks)) / idn["cases_per_s"]
    return out | {"identity": idn, "est_scan_seconds": est}


def run(W: Any, task: str, tag: str, eb_state: Any, save_dw: Any) -> dict[str, Any]:
    t, gen = W.task(task), torch.Generator().manual_seed(W.cfg.random_seed)
    return {"task": t.name, "card": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "bits": bits(W, t), "export": export(W, t, gen),
            "zeros": zeros(W, t), "do_nothing": do_nothing(W), "construction": construction(W, t, gen, eb_state, save_dw, tag)}  # fmt: skip


def misses(o: dict[str, Any], th: dict[str, Any]) -> list[str]:
    b, e, z, dn, c = o["bits"], o["export"], o["zeros"], o["do_nothing"], o["construction"]
    m = [f"(a) the dial's start changes the model: {b}"] if not (b["fp32_equal"] and b["autocast_equal"] and b["delta_nonzero"] == 0) else []
    m += [f"(b) the export is not the dial: {e}"] if not (e["equiv"] <= th["equiv_tol"] and e["shuffled"] >= th["perm_min"]) else []
    m += [f"(c) a zero update reads non-zero: {z}"] if z["nonzero_numerators"] or z["kl"] != 0 else []
    m += [f"(d) the do-nothing: {dn}"] if not (dn["n_changed"] == 0 and dn["dd_max"] == 0 and dn["E_b"] == 1) else []
    m += [f"(e) gains only reads E_b {c['gains']['E_b_changed']}"] if not c["gains"]["E_b_changed"] >= th["eb_gain_min"] else []
    m += [f"(e) the Gaussian reads E_b {c['gauss']['E_b_changed']}"] if not c["gauss"]["E_b_changed"] <= th["eb_random_max"] else []
    m += [f"(f) identity scan: {c['identity']['flips']} argmax changes"] if c["identity"]["flips"] else []
    return m
