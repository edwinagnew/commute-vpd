"""E195 (cell P1.25, run 1): causal-rule retention under the fixed choices on the 67M (plan
reports/plans/2026-11-17_plan_rule-retention-fixed-choices.md; card card.md; filed settings, thresholds and expected
readings in results/predictions.json; the run refuses unless that file is committed unchanged). The measurement lives in
commute.rule_retention_fixed; data, the circuit's attribution and the turned-up items are P1.24's code
(experiments/rule_retention/run.py, called, not changed). Every reading is on copy24's 75 held-out windows.

Stages, cheapest first; each writes results/stage<n>.json when it finishes, prints `STAGE n DONE`, and is skipped on a
rerun once its JSON says done:
    0  the code checks, alone: C recomputed by P1.24's attribution (exit 1 unless it is layer 1's filed 8 neurons), then
       do-nothing, C turned up x2 and x10, and the moved copy (C's output columns zeroed, copied exactly onto 8
       appended neurons). Exit 1 on any miss of the filed checks (predictions.json "stops").
    1  the rebuilt item: C zeroed; only the MLP weights of layers 2 and 3 trained on KL(x2 || F) for `steps` steps;
       exit 1 unless the held-out KL at `check_step` is below its value at step 0. Trained weights, fp32:
       results/large/rebuilt_mlp.pt.
    2  the rebuilt item read at both levels; per-unit r, gain, partner for every unit in results/units_rebuilt.npz.
Inputs: artifacts/vpd (the 67M), artifacts/sets/train_0_4096.npy and val_0_256.npy (Pile rows),
experiments/open_pair/results/large/amend1/windows_copy_trained.pt (copy24) and, for stage 2 alone,
results/large/rebuilt_mlp.pt. Exit 0 done, 1 a stop or refusal. The log is standard output; keep it with tee:

    python experiments/rule_retention_fixed/run.py 2>&1 | tee experiments/rule_retention_fixed/results/run.log
    (on a pod: E120_PRED_SHA=<sha of the predictions file> python ...)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from commute import rule_retention as R
from commute import rule_retention_fixed as RF
from commute.jobs import hard_exit, load_experiment, require_filed

P: Any = load_experiment("rule_retention", "run")  # P1.24: load_data, attribution, clone, rescaled, mean_kl, chunks, log, dump, stop
CELL = "P1.25"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, LARGE = RESULTS / "predictions.json", RESULTS / "large"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
log, dump, stop = P.log, P.dump, P.stop


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    layer: int  # C's layer
    C: tuple[int, ...]  # C's neurons, P1.24's circuit
    factors: tuple[float, ...]  # the turned-up items
    target: float  # the factor of the item the rebuilt item is fitted to
    trained: tuple[int, ...]  # layers whose MLP weights the rebuilt item trains
    window: int  # tokens per window, copy24's
    pile_train: int  # Pile windows cut from train rows 0..4096 (fit only)
    pile_held: int  # Pile windows cut from val rows 0..256 (fit ratio only)
    data_seed: int
    fit_seed: int
    batch: int  # per step, half Pile, half copy24
    steps: int
    lr: float
    warmup: int
    check_step: int
    log_every: int
    eval_batch: int
    profile_batch: int

    def __post_init__(self) -> None:
        assert self.batch % 2 == 0 and self.warmup < self.check_step < self.steps and self.target in self.factors
        assert self.layer not in self.trained and len(set(self.C)) == len(self.C) and max(self.C) < R.D_MLP


# ---- reading one item ---------------------------------------------------------------------------------------------


def read(F: R.Model, M: R.Model, d: Any, cfg: Config, th: dict[str, Any], item: str) -> tuple[RF.Profile, dict[str, Any]]:
    """Profile, hybrid gate and the contrast at both levels."""
    prof = RF.profile(F, M, P.chunks(d.task_held, cfg.profile_batch))
    out: dict[str, Any] = {"gate": RF.gate(F, M, d.task_held, prof)}
    log(f"{item}: hybrid gate (max relative logit error, nothing put back) {out['gate']:.3g}")
    for lv in "AB":
        us, ch = RF.units(prof, lv), RF.changed(prof, lv, th["r_tol"])
        r = [RF.get(prof, u, "r") for u in us]
        zero = sum(int(p.zero.sum()) for (k, _), p in prof.items() if k in RF.KINDS[lv])
        log(f"{item} level {lv}: {len(us)} non-zero units, {zero} zero, {len(ch)} changed, min r {min(r):.8f}, mean r {sum(r) / len(r):.6f}; changed "
            + ", ".join(f"{R.name(u)} r {RF.get(prof, u, 'r'):.6f} g {RF.get(prof, u, 'gain'):.4f}" for u in ch[:24]))
        c = RF.contrast(F, M, d.task_held, prof, lv, th, log)
        log(f"{item} level {lv}: contrast {c['reading']}, KL(F || M) {c['kl']:.6g}, retention {c.get('retention')}")
        out[lv] = {"n_units": len(us), "n_zero": zero, "min_r": min(r), "mean_r": sum(r) / len(r), "contrast": c,
                   "changed": {R.name(u): {k: RF.get(prof, u, k) for k in ("r", "gain", "same", "partner")} for u in ch[:64]}}  # fmt: skip
    return prof, out


# ---- stage 0: the code checks -------------------------------------------------------------------------------------


def moved(M: R.Model, layer: int, C: list[int]) -> R.Model:
    """C's output columns zeroed and C copied exactly onto len(C) appended neurons: same input rows and output columns."""
    F, mlp = P.clone(M), M.net.h[layer].mlp
    assert mlp.c_fc.bias is None and mlp.down_proj.bias is None
    e = R.Extra(len(C), mlp.c_fc.weight.shape[1], F.net.h[layer].mlp.gelu, 0.0, 0.0, 0).to(mlp.c_fc.weight.device)
    with torch.no_grad():
        e.w_in.copy_(mlp.c_fc.weight[C])
        e.w_out.copy_(mlp.down_proj.weight[:, C])
        F.net.h[layer].mlp.down_proj.weight[:, C] = 0
    F.append(layer, e.requires_grad_(False))
    return F


def expected(cfg: Config, item: str, factor: float, n: int, l: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Level A, layer l with n units in F: the expected partner, zero mask and gain."""
    partner, zero, gain = torch.arange(n), torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.float64)
    C = torch.tensor(cfg.C)
    if l == cfg.layer and item == "moved_copy":
        partner[C], zero[C], partner[R.D_MLP :] = -1, True, C
    elif l == cfg.layer:
        gain[C] = factor
    return partner, zero, gain


def misses(prof: RF.Profile, rd: dict[str, Any], cfg: Config, th: dict[str, Any], item: str, factor: float) -> list[str]:
    """Every filed stage-0 check the item misses (predictions.json "stops")."""
    out = [f"gate {rd['gate']:.3g}"] if not rd["gate"] <= th["gate_tol"] else []
    for (kind, l), p in prof.items():
        if kind != "neuron":
            continue
        partner, zero, gain = expected(cfg, item, factor, len(p.partner), l)
        nz = ~zero
        out += [f"L{l} zero units differ"] if not torch.equal(p.zero, zero) else []
        out += [f"L{l} pairing differs"] if not torch.equal(p.partner, partner) else []
        out += [f"L{l} max |1 - r| {float((1 - p.r[nz]).abs().max()):.3g}"] if not (1 - p.r[nz]).abs().max() <= th["r_tol"] else []
        out += [f"L{l} max |gain - expected| {float((p.gain[nz] - gain[nz]).abs().max()):.3g}"] if not (p.gain[nz] - gain[nz]).abs().max() <= th["gain_tol"] else []
    c = rd["A"]["contrast"]
    if item == "do_nothing":
        out += [f"level B min r {rd['B']['min_r']:.8f}"] if not rd["B"]["min_r"] >= 1 - th["r_tol"] else []
        out += [f"level {lv} contrast {rd[lv]['contrast']['reading']}" for lv in "AB" if rd[lv]["contrast"]["reading"] != "no change"]
    elif c["reading"] == "read" or item != "moved_copy":
        out += [f"level A contrast {c['reading']} {c.get('retention')}"] if c["reading"] != "read" or not all(abs(1 - v) <= th["r_tol"] for v in c["retention"]) else []
    return out


def stage0(M: R.Model, d: Any, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    A = P.attribution(M, d, cfg.eval_batch)
    mass = A.abs().sum(1)
    layer = int(mass.argmax())
    C = sorted(A[layer].abs().topk(len(cfg.C)).indices.tolist())
    out: dict[str, Any] = {"experiment": EXPERIMENT, "mass": mass.tolist(), "layer": layer, "C": C, "items": {}}
    log(f"attribution mass per layer {[round(m, 4) for m in mass.tolist()]}; layer {layer}; C {C}")
    if layer != cfg.layer or C != list(cfg.C):
        stop(RESULTS / "stage0.json", out, f"stage 0: C is layer {layer} {C}, not the filed layer {cfg.layer} {list(cfg.C)}")
    items: dict[str, tuple[Callable[[], R.Model], float]] = {"do_nothing": (lambda: P.clone(M), 1.0), "moved_copy": (lambda: moved(M, layer, C), 1.0)}
    items |= {f"x{f:g}": (lambda f=f: P.rescaled(M, layer, C, f), f) for f in cfg.factors}
    bad = []
    for item, (make, factor) in items.items():
        prof, rd = read(make(), M, d, cfg, th, item)
        rd["misses"] = misses(prof, rd, cfg, th, item, factor)
        out["items"][item] = rd
        log(f"{item}: {'checks hold' if not rd['misses'] else 'MISSES ' + '; '.join(rd['misses'])}")
        bad += [f"{item}: {m}" for m in rd["misses"]]
    if bad:
        stop(RESULTS / "stage0.json", out, "stage 0: " + "; ".join(bad))
    return out


# ---- stage 1: the rebuilt item ------------------------------------------------------------------------------------


def zeroed(M: R.Model, cfg: Config) -> R.Model:
    F = P.clone(M)
    with torch.no_grad():
        F.net.h[cfg.layer].mlp.down_proj.weight[:, list(cfg.C)] = 0
    return F


def trained(F: R.Model, cfg: Config) -> dict[str, torch.nn.Parameter]:
    return {f"h.{l}.mlp.{n}": p for l in cfg.trained for n, p in F.net.h[l].mlp.named_parameters() if n.split(".")[0] in ("c_fc", "down_proj")}


def fit(F: R.Model, T: R.Model, d: Any, cfg: Config) -> list[dict[str, float]]:
    """Adam with linear warm-up on the trained layers' MLP weights; loss KL(T || F), mean over positions, on batches of
    half Pile, half copy24 training windows. Exit 1 unless the held-out copy24 KL(T || F) at check_step is below step 0's."""
    params = list(trained(F, cfg).values())
    opt = torch.optim.Adam([p.requires_grad_(True) for p in params], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))
    gen, half = torch.Generator().manual_seed(cfg.fit_seed), cfg.batch // 2
    curve = [{"step": 0, "held_kl": P.mean_kl(T, F, d.task_held, cfg.eval_batch)}]
    log(f"  fit step 0: held-out copy24 KL(x{cfg.target:g} || F) {curve[0]['held_kl']:.6g}")
    for step in range(1, cfg.steps + 1):
        i, j = torch.randint(0, len(d.pile_train), (half,), generator=gen), torch.randint(0, len(d.task_train), (half,), generator=gen)
        toks = torch.cat([d.pile_train[i.to(d.pile_train.device)], d.task_train[j.to(d.task_train.device)]])
        with torch.no_grad():
            target = T.logits(toks)
        loss = R.kl(target, F.net(toks)[0]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg.log_every == 0 or step in (1, cfg.check_step):
            curve.append({"step": step, "batch_kl": float(loss.detach()), "held_kl": P.mean_kl(T, F, d.task_held, cfg.eval_batch)})
            log(f"  fit step {step}: batch KL {curve[-1]['batch_kl']:.6g}, held-out copy24 KL(x{cfg.target:g} || F) {curve[-1]['held_kl']:.6g}")
        if step == cfg.check_step and not curve[-1]["held_kl"] < curve[0]["held_kl"]:
            stop(RESULTS / "stage1.json", {"experiment": EXPERIMENT, "curve": curve}, f"stage 1: held-out KL at step {step} is not below step 0's")
    for p in params:
        p.requires_grad_(False)
    return curve


def stage1(M: R.Model, d: Any, cfg: Config) -> dict[str, Any]:
    T, F = P.rescaled(M, cfg.layer, list(cfg.C), cfg.target), zeroed(M, cfg)
    out: dict[str, Any] = {"experiment": EXPERIMENT, "curve": fit(F, T, d, cfg)}
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save({k: p.detach().cpu() for k, p in trained(F, cfg).items()}, LARGE / "rebuilt_mlp.pt")
    for name, toks in (("task", d.task_held), ("pile", d.pile_held)):
        out[f"kl_fit_{name}"], out[f"kl_target_{name}"] = P.mean_kl(F, T, toks, cfg.eval_batch), P.mean_kl(T, M, toks, cfg.eval_batch)
        out[f"fit_ratio_{name}"] = out[f"kl_fit_{name}"] / out[f"kl_target_{name}"]
        out[f"kl_rebuilt_M_{name}"] = P.mean_kl(F, M, toks, cfg.eval_batch)
    log(f"rebuilt: fit ratio KL(F || x{cfg.target:g}) / KL(x{cfg.target:g} || M) copy24 held-out {out['fit_ratio_task']:.4g}, Pile held-out {out['fit_ratio_pile']:.4g}; "
        f"KL(F || M) copy24 {out['kl_rebuilt_M_task']:.6g}, Pile {out['kl_rebuilt_M_pile']:.6g}")
    return out


# ---- stage 2: the rebuilt reading ---------------------------------------------------------------------------------


def stage2(M: R.Model, d: Any, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    F = zeroed(M, cfg)
    state = torch.load(LARGE / "rebuilt_mlp.pt", weights_only=True)
    with torch.no_grad():
        for k, p in trained(F, cfg).items():
            p.copy_(state[k])
    prof, rd = read(F, M, d, cfg, th, "rebuilt")
    out = {"experiment": EXPERIMENT, "readings": rd}
    if not rd["gate"] <= th["gate_tol"]:
        stop(RESULTS / "stage2.json", out, f"stage 2: hybrid gate {rd['gate']:.3g}")
    np.savez_compressed(RESULTS / "units_rebuilt.npz", **{f"{k}{l}_{f}": getattr(p, f).numpy() for (k, l), p in prof.items() for f in ("partner", "zero", "r", "gain", "same")})
    for lv in "AB":
        c = rd[lv]["contrast"]
        log(f"rebuilt level {lv}: contrast {c['reading']}, retention {c.get('retention')} (range {c.get('min')}..{c.get('max')})")
    return out


def staged(n: int, make: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    path = RESULTS / f"stage{n}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        log(f"stage {n}: done before, skipped")
        return json.loads(path.read_text())
    out = make() | {"done": True}
    dump(path, out)
    log(f"STAGE {n} DONE")
    return out


def main() -> int:
    from commute.steering import Net

    pred = json.loads(PRED.read_text())
    settings: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    cfg = Config(**settings)
    th = pred["thresholds"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout
    log(f"{EXPERIMENT} ({CELL}) start; settings {asdict(cfg)}; thresholds {th}")
    M = R.Model(Net("cuda").net)
    d = P.load_data(cfg, "cuda")
    staged(0, lambda: stage0(M, d, cfg, th))
    staged(1, lambda: stage1(M, d, cfg))
    staged(2, lambda: stage2(M, d, cfg, th))
    log(f"{EXPERIMENT} DONE")
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
