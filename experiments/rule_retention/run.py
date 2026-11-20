"""E193 (cell P1.24) and its amendments: calibrating causal-rule retention on three items of the 67M built with known
machinery (plan reports/plans/2026-11-17_plan_causal-rule-retention.md; the definition is tested, not adopted; no trained
fine-tune is read). Card: card.md (E193), card_<amend>.md. Filed rules and settings: results/predictions.json (E193) or
results/<amend>/predictions.json with `--amend <amend>`; the run refuses unless that file is committed unchanged, and its
"experiment" names the run. Outputs go to results/ (E193) or results/<amend>/, appended weights to results/large/ (E193) or
results/large/<amend>/. Levels, rules, distance and the hybrid: commute.rule_retention. Every reading is on copy24's
held-out windows.

Every item's faithfulness gate is read at both levels before its readings (exit 1 on a miss). Stages, cheapest first;
each writes stage<n>.json when it finishes, prints `STAGE n DONE`, and is skipped on a rerun:
    0  do-nothing (F a copy of M) and the faithfulness gate at both levels, on M and F. Exit 1 if any r differs from 1 by
       more than r_tol, a gate reads over gate_tol, or the contrast is not "no change".
    1  the circuit C (k neurons of the layer with the most attribution mass on copy24's training windows) and the
       rescaled item (C's down_proj columns times alpha); exit 1 if KL(rescaled || M) is under kl_floor. Both levels read.
    2  per fit seed, the replaced item: C's columns zeroed, n_new neurons appended to that layer and trained alone by
       KL(replaced || rescaled) + lambda * mean corr^2 with C's neurons on Pile and copy24 training windows, lambda
       through the filed grid in ascending order; the first fit passing the fit and not-a-copy checks on held-out windows
       is read at both levels and the run goes on to the next seed; exit 1 only if no lambda passes on a seed. Appended
       weights of every try in <large>/replaced_s<seed>_lambda<lambda>.pt.
    3  rule.score -> score.json.
Inputs: artifacts/vpd (the 67M), artifacts/sets/train_0_4096.npy and val_0_256.npy (Pile rows), and
experiments/open_pair/results/large/amend1/windows_copy_trained.pt (copy24). Exit 0 done, 1 a stop or refusal.
The log is standard output only (no file is written by the run); keep it with tee:

    python experiments/rule_retention/run.py --amend amend1 2>&1 | tee experiments/rule_retention/results/amend1/run.log
    (on a pod: E120_PRED_SHA=<sha of the predictions file> python ...)
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import rule_retention as R
from commute.data import pile_rows
from commute.jobs import (
    EXPERIMENTS,
    hard_exit,
    load_experiment,
    require_filed,
    write_stage,
)
from commute.jobs import stop as jobs_stop

RU: Any = load_experiment("rule_retention", "rule")
CELL = "P1.24"
HERE = Path(__file__).resolve().parent
AMEND = sys.argv[sys.argv.index("--amend") + 1] if "--amend" in sys.argv else ""  # "" is E193 itself
RESULTS = HERE / "results" / AMEND if AMEND else HERE / "results"
PRED, LARGE = RESULTS / "predictions.json", HERE / "results" / "large" / AMEND if AMEND else HERE / "results" / "large"
EXPERIMENT = json.loads(PRED.read_text())["experiment"]
COPY24 = EXPERIMENTS / "open_pair" / "results" / "large" / "amend1" / "windows_copy_trained.pt"


@dataclass(frozen=True)
class Config:  # filed in predictions.json "settings"
    k: int  # neurons in C
    alpha: float  # the rescaled item's gain on C
    n_new: int  # appended neurons of the replaced item
    fit_seeds: tuple[int, ...]
    lambdas: tuple[float, ...]  # the corr^2 penalty weights tried in this order per seed (amendment 2026-11-17)
    window: int  # tokens per window, copy24's
    pile_train: int  # Pile windows cut from train rows 0..4096 (fit only)
    pile_held: int  # Pile windows cut from val rows 0..256 (fit checks only)
    data_seed: int
    batch: int  # per step, half Pile, half copy24
    steps: int
    lr: float
    warmup: int
    log_every: int
    eval_batch: int

    def __post_init__(self) -> None:
        assert list(self.lambdas) == sorted(self.lambdas) and self.batch % 2 == 0 and self.k <= R.D_MLP and self.steps > self.warmup and len(set(self.fit_seeds)) == len(self.fit_seeds)


@dataclass
class Data:
    task_train: Tensor  # copy24's 301 training windows: attribution and fit
    task_held: Tensor  # its 75 held-out windows: every reading, and the fit checks
    answer: Tensor  # (V,) key -> value
    half: int  # the answer position
    pile_train: Tensor
    pile_held: Tensor


def log(msg: str) -> None:
    print(msg, flush=True)  # the job's standard output is the log (tee it to <results>/run.log)


def dump(path: Path, obj: dict[str, Any]) -> None:
    write_stage(path, obj)  # also keeps the .stopped marker beside it in step (friction 861)


def stop(path: Path, out: dict[str, Any], why: str) -> None:
    jobs_stop(path, out, why, log)  # the JSON's "stopped" key and a .stopped marker file beside it, exit 1


def cut(rows: Tensor, n: int, length: int, gen: torch.Generator) -> Tensor:
    r = torch.randint(0, rows.shape[0], (n, 1), generator=gen)
    p = torch.randint(0, rows.shape[1] - length + 1, (n, 1), generator=gen)
    return rows[r, p + torch.arange(length)]


def load_data(cfg: Config, device: str) -> Data:
    w = torch.load(COPY24, weights_only=False)
    assert w["train"].tokens.shape[1] == cfg.window and w["train"].half == w["heldout"].half
    gen = torch.Generator().manual_seed(cfg.data_seed)
    pt, ph = cut(pile_rows("train", 0, 4096), cfg.pile_train, cfg.window, gen), cut(pile_rows("val", 0, 256), cfg.pile_held, cfg.window, gen)
    return Data(w["train"].tokens.to(device), w["heldout"].tokens.to(device), w["table"].to(device), w["train"].half, pt.to(device), ph.to(device))


def chunks(toks: Tensor, size: int) -> list[Tensor]:
    return [toks[i : i + size] for i in range(0, toks.shape[0], size)]


def mean_kl(P: R.Model, Q: R.Model, toks: Tensor, size: int) -> float:
    with torch.no_grad():
        return sum(float(R.kl(P.logits(t), Q.logits(t)).sum()) for t in chunks(toks, size)) / toks.shape[0]


def clone(M: R.Model) -> R.Model:
    return R.Model(copy.deepcopy(M.net).requires_grad_(False))


def read(F: R.Model, M: R.Model, d: Data, th: dict[str, Any], item: str) -> dict[str, Any]:
    """Both levels on the held-out copy24 windows: F's faithfulness gate, the profile (every unit not at r = 1, by name)
    and the contrast. Exit 1 if F's gate misses at either level."""
    gate, path = R.gate(F, d.task_held), RESULTS / f"gate_{item}.json"
    log(f"{item} gate (max relative logit error, fp32, copy24 held-out): {gate}")
    if not all(v <= th["gate_tol"] for v in gate.values()):
        stop(path, {"experiment": EXPERIMENT, "gate": gate}, f"{item}: the faithfulness gate misses")
    prof, out = R.profile(F, M, chunks(d.task_held, 25)), dict[str, Any](gate=gate)
    for lv in "AB":
        r = prof[lv]
        ch = R.changed(r, th["r_tol"])
        log(f"{item} level {lv}: {len(r)} units, {len(r) - len(ch)} at r = 1 within {th['r_tol']}, min r {min(r.values()):.8f}; changed "
            + ", ".join(f"{R.name(u)} {r[u]:.6f}" for u in ch[:40]))
        c = R.contrast(F, M, d.task_held, r, th, log)
        log(f"{item} level {lv}: contrast {c['reading']}, KL(F || M) {c['kl']:.6g}, retention {c.get('retention')}, closure {c.get('closure')}")
        out[lv] = {"n_units": len(r), "max_dev": max(1 - v for v in r.values()), "changed": {R.name(u): r[u] for u in ch}, "contrast": c}
    return out


# ---- stage 0: do-nothing and the gate -----------------------------------------------------------------------------


def stage0(M: R.Model, d: Data, th: dict[str, Any]) -> dict[str, Any]:
    F, path = clone(M), RESULTS / "stage0.json"
    windows = [d.task_held, d.pile_held[:128]]
    gate = {n: {lv: max(R.gate(N, t)[lv] for t in windows) for lv in "AB"} for n, N in (("M", M), ("do_nothing", F))}
    log(f"gate (max relative logit error, fp32, copy24 held-out and 128 Pile windows): {gate}")
    out = {"experiment": EXPERIMENT, "gate": gate, "readings": read(F, M, d, th, "do_nothing")}
    misses = [f"gate {n} level {lv} {v:.3g}" for n, g in gate.items() for lv, v in g.items() if not v <= th["gate_tol"]]
    rd = out["readings"]
    misses += [f"level {lv} max |1 - r| {rd[lv]['max_dev']:.3g}" for lv in "AB" if not rd[lv]["max_dev"] <= th["r_tol"]]
    misses += [f"level {lv} contrast {rd[lv]['contrast']['reading']}" for lv in "AB" if rd[lv]["contrast"]["reading"] != "no change"]
    if misses:
        stop(path, out, "stage 0: " + "; ".join(misses))
    return out


# ---- stage 1: the circuit and the rescaled item -------------------------------------------------------------------


def attribution(M: R.Model, d: Data, size: int) -> Tensor:
    """(layers, 3072) per neuron: activation times the gradient of the answer logit at the answer position, summed over
    copy24's training windows and positions."""
    seen: list[Tensor] = []

    def keep(_m: Any, a: tuple[Tensor, ...]) -> None:
        a[0].retain_grad()
        seen.append(a[0])

    hooks = [M.net.wte.register_forward_hook(lambda _m, _a, o: o.detach().requires_grad_())]
    hooks += [b.mlp.down_proj.register_forward_pre_hook(keep) for b in M.net.h]
    A = torch.zeros(len(M.net.h), R.D_MLP, dtype=torch.float64, device=d.task_train.device)
    try:
        for t in chunks(d.task_train, size):
            seen.clear()
            with torch.enable_grad():
                logits = M.net(t)[0]
                logits[torch.arange(len(t), device=t.device), d.half, d.answer[t[:, d.half]]].sum().backward()
            A += torch.stack([(a * a.grad).double().sum((0, 1)) for a in seen])  # type: ignore[operator]
    finally:
        for h in hooks:
            h.remove()
    return A


def rescaled(M: R.Model, layer: int, C: list[int], alpha: float) -> R.Model:
    F = clone(M)
    with torch.no_grad():
        F.net.h[layer].mlp.down_proj.weight[:, C] *= alpha
    return F


def share(M: R.Model, d: Data, layer: int, C: list[int], alpha: float) -> dict[str, float]:
    """C's share of the block's output on M's inputs, q = |c|^2 / |f|^2 and rho = <f, c> / |f|^2, and the block's r that
    follows from them, 1 - (a-1)^2 q / (2 + 2 (a-1) rho + (a-1)^2 q)."""
    with torch.no_grad():
        y = R.m_inputs(M, d.task_held)[layer][1]
        f = R.block(M, layer, y).double()
        c = (R.acts(M, layer, y)[..., C] @ R.cols(M, layer)[:, C].T).double()
    q, rho, g = float(c.pow(2).sum() / f.pow(2).sum()), float((f * c).sum() / f.pow(2).sum()), alpha - 1
    return {"q": q, "rho": rho, "block_r": 1 - g * g * q / (2 + 2 * g * rho + g * g * q)}


def stage1(M: R.Model, d: Data, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    path = RESULTS / "stage1.json"
    A = attribution(M, d, cfg.eval_batch)
    mass = A.abs().sum(1)
    layer = int(mass.argmax())
    C = sorted(A[layer].abs().topk(cfg.k).indices.tolist())
    out = {"experiment": EXPERIMENT, "mass": mass.tolist(), "layer": layer, "C": C, "attribution_C": A[layer, C].tolist()}
    log(f"attribution mass per layer {[round(m, 4) for m in mass.tolist()]}; layer {layer}; C {C}, attribution {[round(a, 4) for a in A[layer, C].tolist()]}")
    F = rescaled(M, layer, C, cfg.alpha)
    out |= {"kl_task": mean_kl(F, M, d.task_held, cfg.eval_batch), "kl_pile": mean_kl(F, M, d.pile_held, cfg.eval_batch), "share": share(M, d, layer, C, cfg.alpha)}
    log(f"rescaled: KL(rescaled || M) copy24 held-out {out['kl_task']:.6g}, Pile held-out {out['kl_pile']:.6g}; C's share {out['share']}")
    if not out["kl_task"] > th["kl_floor"]:
        stop(path, out, f"stage 1: KL(rescaled || M) {out['kl_task']:.3g} is not above kl_floor; the item has no contrast")
    out["readings"] = read(F, M, d, th, "rescaled")
    return out


# ---- stage 2: the replaced item -----------------------------------------------------------------------------------


def batch_corr2(a: Tensor, b: Tensor) -> Tensor:
    """Mean over (new, C) pairs of the squared Pearson correlation over the positions of one batch; a (N, k), b (N, n)."""
    a, b = a - a.mean(0), b - b.mean(0)
    return ((b.T @ a) / torch.outer(b.norm(dim=0), a.norm(dim=0)).clamp(min=1e-12)).pow(2).mean()


def fit(F: R.Model, T: R.Model, e: R.Extra, d: Data, cfg: Config, s1: dict[str, Any], lam: float, seed: int) -> list[dict[str, float]]:
    """Train only e's weights by KL(F || T), mean over positions, plus lam times `batch_corr2` between e's and C's
    activations on the layer's normed input, on batches of half Pile, half copy24 training windows."""
    opt = torch.optim.Adam(e.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))
    gen, half, curve, seen = torch.Generator().manual_seed(10_000 + seed), cfg.batch // 2, [], []
    blk = F.net.h[s1["layer"]].mlp
    hook = blk.register_forward_pre_hook(lambda _m, a: seen.append(a[0]))
    try:
        for step in range(1, cfg.steps + 1):
            idx = torch.randint(0, len(d.pile_train), (half,), generator=gen), torch.randint(0, len(d.task_train), (half,), generator=gen)
            toks = torch.cat([d.pile_train[idx[0].to(d.pile_train.device)], d.task_train[idx[1].to(d.task_train.device)]])
            with torch.no_grad():
                target = T.logits(toks)
            seen.clear()
            kl = R.kl(F.net(toks)[0], target).mean()
            h = seen[0].flatten(0, 1)
            pen = batch_corr2(blk.gelu(blk.c_fc(h))[:, s1["C"]].detach(), e.acts(h))
            opt.zero_grad(set_to_none=True)
            (kl + lam * pen).backward()
            opt.step()
            sched.step()
            if step % cfg.log_every == 0 or step == 1:
                curve.append({"step": step, "kl": float(kl), "corr2": float(pen)})
                log(f"  fit s{seed} lambda {lam} step {step}: batch KL(replaced || rescaled) {float(kl):.6g}, mean corr^2 with C {float(pen):.4g}")
    finally:
        hook.remove()
    return curve


def corr(M: R.Model, e: R.Extra, layer: int, C: list[int], toks: Tensor, size: int) -> Tensor:
    """(n_new, k) Pearson correlation over every position of `toks` between appended neurons and C's neurons, on M's input."""
    s: dict[str, Tensor] = {}
    with torch.no_grad():
        for t in chunks(toks, size):
            y = R.m_inputs(M, t)[layer][1]
            a = R.acts(M, layer, y)[..., C].double().flatten(0, 1)
            b = e.acts(M.net.h[layer].rms_2(y)).double().flatten(0, 1)
            for k, v in {"n": torch.tensor(float(len(a)), dtype=torch.float64), "a": a.sum(0), "b": b.sum(0), "aa": (a * a).sum(0),
                         "bb": (b * b).sum(0), "ab": b.T @ a}.items():  # fmt: skip
                s[k] = s[k] + v.to(a.device) if k in s else v.to(a.device)
    n = s["n"]
    cov = s["ab"] / n - torch.outer(s["b"], s["a"]) / n**2
    return cov / torch.outer((s["bb"] / n - (s["b"] / n) ** 2).sqrt(), (s["aa"] / n - (s["a"] / n) ** 2).sqrt())


def replaced(M: R.Model, d: Data, cfg: Config, th: dict[str, Any], s1: dict[str, Any], seed: int) -> tuple[R.Model, dict[str, Any]]:
    """The replaced item at one fit seed (amendment 2026-11-17): a fit per lambda of cfg.lambdas in ascending order, each
    from the same start, keeping the first that passes both held-out checks. rec["tries"] holds every lambda tried;
    rec["misses"] is empty when one passed, else the last try's misses."""
    tries: list[dict[str, Any]] = []
    F, rec = attempt(M, d, cfg, th, s1, seed, cfg.lambdas[0])
    tries.append(rec)
    for lam in cfg.lambdas[1:]:
        if not rec["misses"]:
            break
        F, rec = attempt(M, d, cfg, th, s1, seed, lam)
        tries.append(rec)
    return F, {"lambda": rec["lambda"], "tries": tries, "misses": rec["misses"]}


def attempt(M: R.Model, d: Data, cfg: Config, th: dict[str, Any], s1: dict[str, Any], seed: int, lam: float) -> tuple[R.Model, dict[str, Any]]:
    """One fit at one lambda and its two checks on held-out windows; rec["misses"] lists the checks missed."""
    layer, C = s1["layer"], s1["C"]
    T, F = rescaled(M, layer, C, cfg.alpha), clone(M)
    blk = F.net.h[layer].mlp
    std_in, std_out = float(blk.c_fc.weight.std()), float(blk.down_proj.weight.std())  # the layer's own weight scales
    with torch.no_grad():
        blk.down_proj.weight[:, C] = 0
    e = R.Extra(cfg.n_new, blk.c_fc.weight.shape[1], blk.gelu, std_in, std_out, seed).to(d.task_held.device)
    F.append(layer, e)
    rec: dict[str, Any] = {"lambda": lam, "std_in": std_in, "std_out": std_out, "curve": fit(F, T, e, d, cfg, s1, lam, seed)}
    e.requires_grad_(False)
    for name, toks in (("task", d.task_held), ("pile", d.pile_held)):
        rec[f"kl_fit_{name}"], rec[f"kl_rescaled_{name}"] = mean_kl(F, T, toks, cfg.eval_batch), mean_kl(T, M, toks, cfg.eval_batch)
        rec[f"fit_ratio_{name}"] = rec[f"kl_fit_{name}"] / rec[f"kl_rescaled_{name}"]
    cc = corr(M, e, layer, C, torch.cat([d.pile_held, d.task_held]), cfg.eval_batch)
    rec |= {"corr": cc.tolist(), "max_abs_corr": float(cc.abs().max())}
    log(f"replaced s{seed} lambda {lam}: fit ratio copy24 {rec['fit_ratio_task']:.4g}, Pile {rec['fit_ratio_pile']:.4g} (bar {th['fit_ratio']}); "
        f"max |corr| with C {rec['max_abs_corr']:.4f} (bar {th['copy_corr']})")
    rec["misses"] = [f"fit ratio {n} {rec[f'fit_ratio_{n}']:.4g}" for n in ("task", "pile") if not rec[f"fit_ratio_{n}"] <= th["fit_ratio"]]
    if not rec["max_abs_corr"] <= th["copy_corr"]:
        rec["misses"].append(f"max |corr| {rec['max_abs_corr']:.4f}")
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in e.state_dict().items()}, LARGE / f"replaced_s{seed}_lambda{lam:g}.pt")
    return F, rec


def stage2(M: R.Model, d: Data, cfg: Config, th: dict[str, Any], s1: dict[str, Any]) -> dict[str, Any]:
    path = RESULTS / "stage2.json"
    out = json.loads(path.read_text()) if path.is_file() else {"experiment": EXPERIMENT, "seeds": {}}
    out.pop("stopped", None)
    for seed in (s for s in cfg.fit_seeds if "readings" not in out["seeds"].get(str(s), {})):
        F, rec = replaced(M, d, cfg, th, s1, seed)
        out["seeds"][str(seed)] = rec
        if rec["misses"]:
            stop(path, out, f"stage 2 seed {seed}: no lambda passes; last: " + "; ".join(rec["misses"]))
        rec["readings"] = read(F, M, d, th, f"replaced_s{seed}")
        dump(path, out)
        print(f"STAGE 2 seed {seed} DONE", flush=True)
    return out


def staged(n: int, make: Any) -> dict[str, Any]:
    path = RESULTS / f"stage{n}.json"
    if path.is_file() and json.loads(path.read_text()).get("done"):
        return json.loads(path.read_text())
    out = make() | {"done": True}
    dump(path, out)
    print(f"STAGE {n} DONE", flush=True)
    return out


def main() -> int:
    from commute.steering import Net

    pred = json.loads(PRED.read_text())
    settings: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in pred["settings"].items()}
    th, cfg = pred["thresholds"], Config(**settings)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, for the gate
    log(f"{EXPERIMENT} ({CELL}) start; settings {asdict(cfg)}; thresholds {th}")
    M = R.Model(Net("cuda").net)
    d = load_data(cfg, "cuda")
    s0 = staged(0, lambda: stage0(M, d, th))
    s1 = staged(1, lambda: stage1(M, d, cfg, th))
    s2 = staged(2, lambda: stage2(M, d, cfg, th, s1))
    score = RU.score(s0, s1, s2, th)
    dump(RESULTS / "score.json", {"experiment": EXPERIMENT, "cell": CELL, **score})
    for lv, v in score["levels"].items():
        log(f"level {lv}: {v['verdict']}; rescaled {v['rescaled']}, replaced {v['replaced']}")
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
