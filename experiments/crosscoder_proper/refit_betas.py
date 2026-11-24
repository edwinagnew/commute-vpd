"""Cell P2.70 follow-up: refit latent scaling on the ten planted controls' saved crosscoders at the paper's token count
(Minder et al. 2025 fit on 50M tokens; the runs used 100,000) and read them under App. E.2's literal exclusion. Plan
reports/plans/2026-11-23_plan_p2-70-betas-refit.md; settings, filed expectations and rules
results/refit/predictions.json (refuses unless committed unchanged). The crosscoders are E235's and E243's
results/large files and E212's planted one; each side and its stream are rebuilt by the run's own code (run.py `Job`,
run2.py `Job`, realistic_tasks/run3.py `Job`), and the fit is `commute.crosscoder.beta_sums`.

  dry       CPU (fleet/cpu-run with INPUT_MANIFEST): build every side and load every input, a two-row fit only
  0         stage 0, per control: the side's energy against the run's; the first chunk's latent-scaling tokens; the
            saved betas reproduced within a floor measured by jittering the weights by half a bfloat16 step (stops the
            job if not); a bootstrap over 512-token blocks of each dropped latent's base beta -> results/refit/stage0.json
  1         stage 1, per control (refuses one stage 0 has not passed): fits A and B on disjoint stream chunks, their sum
            the full fit; a bootstrap over chunks for its standard error; the literal reading and the "near-zero kept"
            reading (ours) -> results/refit/stage1.json, per-latent arrays results/large/refit/<item>.pt
  summary   CPU: the filed expectations read on stage0.json and stage1.json -> results/refit/summary.json

    PRED_SHA=<sha> python experiments/crosscoder_proper/refit_betas.py 0 2>&1 | tee -a experiments/crosscoder_proper/results/refit/stage0.log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import crosscoder as CC
from commute.jobs import (
    EXPERIMENTS,
    config_from_settings,
    hard_exit,
    load_experiment,
    require_filed,
)
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
PRED, RES = HERE / "results/refit/predictions.json", HERE / "results/refit"
LARGE = EXPERIMENTS  # each run's results/large (a link on the dev node; staged as a real folder on a pod)
RUN: Any = load_experiment("crosscoder_proper", "run")
BASE = ("eps_base", "r_base")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


@dataclass(frozen=True)
class Config:  # results/refit/predictions.json "settings"
    items: tuple[str, ...]  # "E235:head.B.c0", "E243:head-r5.A.c0", "E212:planted"
    ls_tokens: int  # the runs' latent-scaling tokens, the first chunk's first
    block: int  # tokens per bootstrap block in stage 0
    boot: int  # bootstrap resamples
    boot_seed: int
    chunks_a: tuple[int, int]  # stream chunk indices [lo, hi) of fit A; B likewise, disjoint, both in epoch 0
    chunks_b: tuple[int, int]
    jitter_q: float  # the quantile of |beta gap| over the latents that the reproduction check reads
    jitter_factor: float  # the reproduction gap may be this many jitter floors
    jitter_seed: int
    energy_rel_tol: float  # the side's activation change against the run's, as run2.py checks run 1's head
    plant_min_s: float  # a dropped latent is a plant latent when it carried at least this share of the change
    near_nu: float  # ... and both its ratios are above this: a base beta near zero
    se_z: float  # an exclusion is real when both halves are negative and the full fit is under -se_z standard errors
    projected_limit_min: float  # stage 1 stops if the first chunk projects an item past this

    def __post_init__(self) -> None:
        assert self.chunks_a[0] >= 1 and self.chunks_a[1] <= self.chunks_b[0] < self.chunks_b[1], self  # chunk 0 is stage 0's
        assert self.chunks_a[1] - self.chunks_a[0] == self.chunks_b[1] - self.chunks_b[0] > 0
        assert all(i.split(":")[0] in ("E235", "E243", "E212") for i in self.items) and self.block < self.ls_tokens


@dataclass
class Control:
    item: str
    cc: CC.Cfg
    saved: dict[str, Any]  # the saved reading: relative_norm, alive, s, scaled, groups, betas
    ae: Any
    chunk: Callable[[int], Tensor]  # stream chunk i as (m, 2, d) on the GPU
    energy: Callable[[], float] | None  # the side's activation change per token on its read rows
    energy_run: float | None


def cc_cfg(S: dict[str, Any]) -> CC.Cfg:
    return config_from_settings(CC.Cfg, S | {"shared": tuple(S["shared"]), "snapshots": tuple(S["snapshots"])})


def load_ae(cc: CC.Cfg, d: dict[str, Any]) -> Any:
    st = d["ae"]
    ae = CC.trainer(cc, st["activation_mean"].float(), st["activation_std"].float(), 0).ae
    ae.load_state_dict(st)  # the big tensors were saved in bfloat16; they load into the float32 parameters
    return ae.eval()


class Builders:
    """One run Job per experiment, built on first use (each loads the model and its Pile rows)."""

    def __init__(self) -> None:
        self.jobs: dict[str, Any] = {}
        self.P1 = json.loads(RUN.PRED.read_text())

    def run_job(self, code: str) -> Any:
        if code not in self.jobs:
            P1, cc = self.P1, cc_cfg(self.P1["settings"]["cc"]["full"])
            cfg1 = config_from_settings(RUN.Config, {k: tuple(v) if isinstance(v, list) else v for k, v in P1["settings"]["run"].items()})
            th = P1["thresholds"]["full"]
            if code == "E235":
                self.jobs[code] = RUN.Job(cfg1, cc, th)
            else:
                R2: Any = load_experiment("crosscoder_proper", "run2")
                self.jobs[code] = R2.Job(cfg1, cc, th, R2.settings(json.loads(R2.PRED.read_text())))
        return self.jobs[code]

    def control(self, item: str, dry: bool) -> Control:
        code, key = item.split(":")
        if code == "E212":
            return self.e212(item, dry)
        job, run = self.run_job(code), "run1" if code == "E235" else "run2"
        rec = json.loads((HERE / f"results/{run}/items/{key}.json").read_text())
        d = torch.load(LARGE / f"crosscoder_proper/results/large/{run}/{key}.pt", map_location="cpu", weights_only=True)
        s = job.side(RUN.Item.parse(key))
        if code == "E243":
            s.apply(rec["match"]["x"])  # the rung's saved scale, not a new search
        _, make, _ = job.stream(s)

        def chunk(i: int) -> Tensor:
            with RUN.exact():
                return job.pair(s.read[:2], s, True) if dry else make(i)

        return Control(item, job.cc, d, load_ae(job.cc, d), chunk, lambda: job.energy(s, s.dW), float(rec["energy"]))

    def e212(self, item: str, dry: bool) -> Control:
        R3: Any = load_experiment("realistic_tasks", "run3")
        P = json.loads(R3.PRED.read_text())
        cc = cc_cfg(P["settings"]["full"])
        job = R3.Job(P, cc, P["thresholds"]["full"], R3.Out(RES / "e212_job.json", "refit", P["experiment"]), RES, None)
        v = torch.randn(cc.d, generator=torch.Generator().manual_seed(job.S["plant_seed"]))  # run3.py `planted`, as run
        L = job.reads["L1"]
        h = CC.acts(job.net, L, None, job.layer).reshape(len(L), L.shape[1] - 1, -1)[:, -1]
        side = R3.Side("planted", job.zero, ("L1",), (v / v.norm() * h.norm(dim=1).median().cpu()).cuda())
        d = torch.load(LARGE / "realistic_tasks/results/large/run3/planted.pt", map_location="cpu", weights_only=True)
        chunk = (lambda i: job.h_read(side, "L1")[:2]) if dry else (lambda i: job.chunk(i, side))
        return Control(item, cc, d["reads"]["L1"], load_ae(cc, d), chunk, None, None)


# ---- the fit ----------------------------------------------------------------------------------------------------------


def stacked(sums: list[dict[str, Tensor]]) -> Tensor:
    """(n, 5, |J|): each part's four numerators and denominator."""
    return torch.stack([torch.stack([s[k] for k in (*CC.BETA_KEYS, "den")]) for s in sums])


def unstack(x: Tensor) -> dict[str, Tensor]:
    return dict(zip((*CC.BETA_KEYS, "den"), x.unbind(-2), strict=True))


def boot(parts: Tensor, n: int, seed: int) -> dict[str, Tensor]:
    """Betas of n resamples (with replacement) of the parts' sums: each (n, |J|)."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.stack([torch.bincount(torch.randint(len(parts), (len(parts),), generator=g), minlength=len(parts)) for _ in range(n)])
    flat = counts.double() @ parts.reshape(len(parts), -1)
    return CC.ratios(unstack(flat.reshape(n, *parts.shape[1:])))


def jittered(ae: Any, seed: int) -> Any:
    """A copy of ae with every bfloat16-saved weight moved by up to half a bfloat16 step: the size of the rounding the
    saved weights went through, so the betas it changes give the reproduction check's floor."""
    import copy

    out, g = copy.deepcopy(ae), torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in out.parameters():
            if p.numel() > 1_000_000:  # the tensors run.py saved in bfloat16
                w = p.detach().cpu()
                _, e = torch.frexp(w)
                u = torch.rand(p.shape, generator=g) - 0.5
                p.add_((u * torch.ldexp(torch.ones_like(u), e - 8) * (w != 0)).to(p.device))  # a zero stays zero
    return out


def gap(a: dict[str, Tensor], b: dict[str, Tensor], q: float) -> float:
    x = torch.cat([(a[k].double() - b[k].double()).abs() for k in CC.BETA_KEYS])
    return float(torch.quantile(x[torch.isfinite(x)], q))


def plants(c: Control, cfg: Config) -> dict[str, Tensor]:
    """Over J: the latents the literal rule drops at the runs' fit (old specific, a base beta < 0), split into plant
    latents (both ratios above near_nu, carrying at least plant_min_s) and clearly negative ones; the triggering base
    beta of each (the target with the lower ratio)."""
    b, J = c.saved["betas"], c.saved["scaled"]
    old = c.saved["groups"]["ft_specific"][J]
    dropped = old & ((b["eps_base"] < 0) | (b["r_base"] < 0))
    near, big = torch.minimum(b["nu_r"], b["nu_eps"]) > cfg.near_nu, c.saved["s"][J].nan_to_num(0) >= cfg.plant_min_s
    return {"dropped": dropped, "plant": dropped & near & big, "clear": dropped & ~near & big, "trig_r": b["nu_r"] < b["nu_eps"]}


def trig(b: dict[str, Tensor], use_r: Tensor) -> Tensor:
    return torch.where(use_r, b["r_base"].double(), b["eps_base"].double())


# ---- stages -----------------------------------------------------------------------------------------------------------


class Stop(Exception):
    pass


def write(path: Path, d: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jsonable(d), indent=1) + "\n")
    tmp.replace(path)


def stage0(c: Control, cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if c.energy is not None:
        e = c.energy()
        out["energy"] = {"got": e, "run": c.energy_run, "pass": math.isclose(e, c.energy_run or 0, rel_tol=cfg.energy_rel_tol)}
        log(f"[{c.item}] energy {e:.6g} against the run's {c.energy_run:.6g}")
        if not out["energy"]["pass"]:
            raise Stop(f"{c.item}: the rebuilt side's energy is not the run's")
    J = c.saved["scaled"]
    h = c.chunk(0)[: cfg.ls_tokens].clone()
    blocks = [CC.beta_sums(c.ae, x, J) for x in h.split(cfg.block)]
    parts = stacked(blocks)
    ref = CC.ratios(unstack(parts.sum(0)))
    floor = gap(CC.betas(jittered(c.ae, cfg.jitter_seed), h, J), ref, cfg.jitter_q)
    got = gap(ref, c.saved["betas"], cfg.jitter_q)
    out["reproduce"] = {"tokens": len(h), "gap": got, "floor": floor, "bar": cfg.jitter_factor * floor, "pass": got <= cfg.jitter_factor * floor}
    log(f"[{c.item}] saved betas reproduced: q{cfg.jitter_q} gap {got:.3g}, jitter floor {floor:.3g}, bar {cfg.jitter_factor * floor:.3g}")
    if not out["reproduce"]["pass"]:
        raise Stop(f"{c.item}: the saved betas are not reproduced within {cfg.jitter_factor} jitter floors")
    m = plants(c, cfg)
    bs = boot(parts, cfg.boot, cfg.boot_seed)
    t_ref, t_bs = trig(ref, m["trig_r"]), trig(bs, m["trig_r"])
    se = t_bs.std(0)
    for name in ("plant", "clear"):
        idx = torch.nonzero(m[name]).squeeze(1)
        out[name] = [{"latent": int(J[i]), "s": float(c.saved["s"][J[i]]), "target": "r" if m["trig_r"][i] else "eps",
                      "beta": float(t_ref[i]), "se": float(se[i]), "z": float(t_ref[i] / se[i]),
                      "frac_negative": float((t_bs[:, i] < 0).double().mean())} for i in idx.tolist()]  # fmt: skip
        for r in out[name]:
            log(f"[{c.item} {name}] {json.dumps(r)}")
    out |= {"dropped": int(m["dropped"].sum()), "pass": True}
    return out


def fit(c: Control, cfg: Config) -> tuple[list[dict[str, Tensor]], dict[str, Any]]:
    J, sums, t0, info = c.saved["scaled"], [], time.time(), {}
    lo, total = cfg.chunks_a[0], 2 * (cfg.chunks_a[1] - cfg.chunks_a[0])
    for i in (*range(*cfg.chunks_a), *range(*cfg.chunks_b)):
        x = c.chunk(i)
        sums.append(CC.beta_sums(c.ae, x, J))
        if i == lo:
            proj = (time.time() - t0) * total / 60
            info = {"chunk_tokens": len(x), "projected_min": proj}
            log(f"[{c.item}] first chunk {len(x)} tokens, projected {proj:.1f} min")
            if proj > cfg.projected_limit_min:
                raise Stop(f"{c.item}: projected {proj:.1f} min, over {cfg.projected_limit_min}")
        del x
    return sums, info | {"tokens_each_half": info["chunk_tokens"] * (cfg.chunks_a[1] - cfg.chunks_a[0]), "minutes": (time.time() - t0) / 60}


def specific_part(c: Control, spec_J: Tensor) -> dict[str, Any]:
    """The specific part with the loaded crosscoder's top N as candidates and the saved carried parts."""
    rn, sv = CC.relative_norm(c.ae), c.saved
    spec = torch.zeros(len(rn), dtype=torch.bool)
    spec[sv["scaled"]] = spec_J
    g = CC.groups(c.cc, rn, sv["alive"], CC.top(rn, c.cc.top_n), spec)
    s = sv["s"].double()
    lead = int(torch.where(sv["alive"], s.nan_to_num(-math.inf), -math.inf).argmax())
    return {"specific": float(s[g["ft_specific"]].sum()), "count": int(g["ft_specific"].sum()), "lead_specific": bool(g["ft_specific"][lead])}


def stage1(c: Control, cfg: Config) -> dict[str, Any]:
    sums, info = fit(c, cfg)
    parts = stacked(sums)
    n = cfg.chunks_a[1] - cfg.chunks_a[0]
    A, B, F = (CC.ratios(unstack(x)) for x in (parts[:n].sum(0), parts[n:].sum(0), parts.sum(0)))
    se = {k: v.std(0) for k, v in boot(parts, cfg.boot, cfg.boot_seed).items() if k in BASE}
    real = torch.zeros(len(c.saved["scaled"]), dtype=torch.bool)
    for k in BASE:  # redteam-lead's rule, filed: both halves negative and the full fit below zero by more than se_z SE
        real |= (A[k] < 0) & (B[k] < 0) & (F[k] < -cfg.se_z * se[k])
    literal = CC.specific(c.cc, F)
    kept = (F["nu_r"] < c.cc.nu_r) & (F["nu_eps"] < c.cc.nu_eps) & ~real  # ours: coin-flip exclusions kept, never the reading
    m = plants(c, cfg)
    old = c.saved["betas"]
    out: dict[str, Any] = {"fit": info, "literal": specific_part(c, literal), "near_zero_kept_ours": specific_part(c, kept),
                           "literal_100k": specific_part(c, CC.specific(c.cc, old))}  # fmt: skip
    for name in ("plant", "clear"):
        rows = []
        for i in torch.nonzero(m[name]).squeeze(1).tolist():
            k = "r_base" if m["trig_r"][i] else "eps_base"
            rows.append({"latent": int(c.saved["scaled"][i]), "target": k, "beta_100k": float(old[k][i]), "a": float(A[k][i]),
                         "b": float(B[k][i]), "full": float(F[k][i]), "se": float(se[k][i]), "real": bool(real[i]),
                         "shrink": abs(float(F[k][i])) / abs(float(old[k][i])), "literal_specific": bool(literal[i])})  # fmt: skip
        out[name] = rows
        for r in rows:
            log(f"[{c.item} {name}] {json.dumps(r)}")
    for k in ("literal", "near_zero_kept_ours", "literal_100k"):
        log(f"[{c.item} read] {k}: {json.dumps(out[k])}")
    big = {"scaled": c.saved["scaled"], "a": {k: A[k].float() for k in CC.BETA_KEYS}, "b": {k: B[k].float() for k in CC.BETA_KEYS},
           "full": {k: F[k].float() for k in CC.BETA_KEYS}, "se": {k: v.float() for k, v in se.items()}, "real": real}  # fmt: skip
    path = HERE / "results/large/refit" / f"{c.item.replace(':', '_')}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(big, path)
    return out


def run(stage: str, cfg: Config, experiment: str) -> int:
    path = RES / f"stage{stage}.json"
    res = json.loads(path.read_text()) if path.is_file() else {"experiment": experiment, "items": {}}
    s0 = RES / "stage0.json"
    passed = json.loads(s0.read_text())["items"] if stage == "1" and s0.is_file() else {}
    build, t0 = Builders(), time.time()
    for item in cfg.items:
        if res["items"].get(item, {}).get("pass"):
            log(f"[{item}] done before, skipped")
            continue
        if stage == "1" and not passed.get(item, {}).get("pass"):
            log(f"STOP: {item} has not passed stage 0")
            return 1
        try:
            c = build.control(item, False)
            res["items"][item] = (stage0 if stage == "0" else stage1)(c, cfg) | {"pass": True}
        except Stop as e:
            res["items"][item] = {"pass": False, "stopped": str(e)}
            write(path, res)
            log(f"STOP: {e}: tell the lead")
            return 1
        write(path, res)
        log(f"ITEM {item} DONE at {(time.time() - t0) / 60:.1f} min")
        torch.cuda.empty_cache()
    log(f"STAGE {stage} DONE in {(time.time() - t0) / 60:.1f} min; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    return 0


def dry(cfg: Config) -> int:
    """Every input opened, two read rows fitted per control (INPUT_MANIFEST's dry run; its numbers mean nothing)."""
    build = Builders()
    for item in cfg.items:
        c = build.control(item, True)
        b = CC.betas(c.ae, c.chunk(0), c.saved["scaled"][:8])
        log(f"[dry {item}] ok, {len(c.saved['scaled'])} latents, a beta {float(b['r_ft'][0]):.3g}")
    return 0


def median(xs: list[float]) -> float:
    return float(torch.tensor(xs, dtype=torch.float64).median()) if xs else math.nan


def summary(P: dict[str, Any]) -> int:
    """The filed expectations (predictions.json "rules") read on the stage files, pooled over the controls."""
    s0, s1 = (json.loads((RES / f"stage{k}.json").read_text())["items"] for k in "01")
    pool = lambda st, name: [r for v in st.values() for r in v.get(name, [])]
    p0, c0, p1 = pool(s0, "plant"), pool(s0, "clear"), pool(s1, "plant")
    out = {"n_plant": len(p0), "n_clear": len(c0),
           "P1_plant_frac_negative_median": median([r["frac_negative"] for r in p0]),
           "P1_plant_over_095": sum(r["frac_negative"] > 0.95 for r in p0),
           "P2_clear_frac_negative_median": median([r["frac_negative"] for r in c0]),
           "P3_plant_shrink_median": median([r["shrink"] for r in p1]),
           "P4_plant_both_halves_negative": sum(r["a"] < 0 and r["b"] < 0 for r in p1),
           "P4_plant_real": sum(r["real"] for r in p1),
           "readings": {k: {w: v[w] for w in ("literal", "near_zero_kept_ours", "literal_100k")} for k, v in s1.items()}}  # fmt: skip
    write(RES / "summary.json", out)
    log(json.dumps(jsonable(out)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("dry", "0", "1", "summary"))
    a = ap.parse_args()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    cfg = config_from_settings(Config, {k: tuple(v) if isinstance(v, list) else v for k, v in P["settings"].items()})
    if a.job == "dry":
        return dry(cfg)
    if a.job == "summary":
        return summary(P)
    assert str(P["experiment"])[1:].isdigit() and os.environ.get("INPUT_MANIFEST_CPU") != "1", "claim the E-number and file it first"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32, as the runs scored
    log(f"[{P['experiment']}] stage {a.job}: {cfg}")
    return run(a.job, cfg, P["experiment"])


if __name__ == "__main__":
    hard_exit(main())
