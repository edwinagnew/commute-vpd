"""E148 (cell P1.9): the built control on XOR2. Plan: reports/plans/2026-11-10_plan_built-control-xor2.md; design:
~/commute-data/scratch/definition-lead/built-control/fable_advice_xor2.md (approved by the user). Filed rules:
results/predictions.json (the run refuses unless it is committed unchanged).

XOR2: 128 key tokens with a balanced seeded bit each; F = M + a push at block 2's input, last position, on the windows whose
last two tokens are keys with different bits. The push is E137's XOR push (its pieces' Q b) times alpha, measured in stage 1.
Two originals are fine-tuned toward the same F and read with E139's searches: the plain 67M, and the 67M taught XOR2 at a
third of the push. Only the original differs.

    run.py job1    stage 0, the controls (zero-step fine-tune of M; the moved capacity read repeats E137's), then stage 1:
                   alpha, both target gaps, the capacity read on the plain 67M, the taught original (300 steps, else 3000)
                   and its capacity read, the plain seed-0 fine-tune and its learned check on held-out key pairs (300
                   steps, else 3000). Writes results/stage0.json, results/stage1.json, results/large/{taught_original,
                   plain_s0}.pt. Any stop exits 1; a plain fine-tune not learned at 3000 steps closes the cell.
    run.py job2    needs a passed stage 1. H for seeds 0 and 1 of each arm, then G for seed 0 of each arm (all four sites,
                   or in2 only if stage 1 needed the 3000-step rung). Writes results/{taught,plain}_s{0,1}.json after
                   each search. Exit 3: the wall limit (105 min) was reached with what was read saved.
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute.hidden_maps import add_last, chunks, kl, run, tv
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.steering import Net

R: Any = load_experiment("reuse_variable_finetune", "run")  # E139: Config, clone_net, distances, last, log, dump, stop, _floor, MoreToDo
A1: Any = load_experiment("reuse_variable_finetune", "amend1")  # setup: rows, E137's task and pieces
A2: Any = load_experiment("reuse_variable_finetune", "amend2")  # search_g over a list of sites, search_h
EXPERIMENT = "E148"
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
E137 = HERE.parent / "reuse_variable_test" / "results" / "stage0.json"
RUNGS = (VR.Schedule(300, 3e-5, 30, (1, 2, 3, 5, 10, 30, 100, 300)),  # E139's recipe
         VR.Schedule(3000, 1e-5, 300, (1, 2, 3, 5, 10, 30, 100, 300, 1000, 2000, 3000)))  # E139 Amendment 2's
log, dump, stop = R.log, R.dump, R.stop


@dataclass(frozen=True)
class Config(R.Config):
    results: Path = HERE / "results"
    n_keys2: int = 128  # Fable's design: 16,384 ordered pairs, so a held-out pair needs a mechanism, not pair memory
    held_share: float = 0.25  # of the ordered pairs, kept for the read half only
    alphas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)  # push sizes tried, in units of E137's XOR push
    calib_n: int = 20_000  # read-half windows for choosing alpha
    taught_share: float = 1 / 3  # the taught original's push, as a share of F's
    seeds: tuple[int, ...] = (0, 1)
    g_sites: tuple[tuple[str, ...], ...] = (VR.SITES, ("in2",))  # per rung: in2 only after 3000 steps, to fit the debug limit

    def __post_init__(self) -> None:
        super().__post_init__()
        assert list(self.alphas) == sorted(self.alphas) and self.calib_n <= self.D // 2 and self.n_keys2 % 2 == 0
        assert len(self.g_sites) == len(RUNGS) and self.seeds[0] == 0 and 0 < self.held_share < 1


def seed_of(item: str) -> int:
    return 300_000 + zlib.crc32(item.encode()) % 65_536  # E139's batch-draw seed rule


def mean_tv(A: VR.Target, B: VR.Target, windows: Tensor, cfg: Config) -> dict[str, float]:
    d = torch.cat([tv(A(t)[0][:, -1], B(t)[0][:, -1]) for t in chunks(windows, cfg.read_batch)])
    return {"mean": float(d.mean()), "max": float(d.max())}


def zero_step(base: Net, read: Tensor, cfg: Config, th: dict[str, Any], floor: float, path: Path, out: dict[str, Any], name: str) -> None:
    """E139's W0 control: a fine-tune at zero steps of `base` is `base`: its gap at the float floor and under floor_gap."""
    d = mean_tv(VR.net_target(R.clone_net(base)), VR.net_target(base), read, cfg)
    out[f"zero_step_{name}"] = d
    log(f"zero-step {name}: gap {d['mean']:.3e}, max TV {d['max']:.3e}, floor {floor:.3e}")
    if d["max"] > floor or d["mean"] >= th["floor_gap"]:
        stop(path, out, f"zero-step fine-tune of the {name} original is off the float floor or reads")


def capacity(net: Net, x: VR.Xor2, fit: Tensor, read: Tensor, cfg: Config) -> dict[str, Any]:
    """E137's capacity read of XOR2 on the keyed windows only (a far window has no XOR2, and labelling it 0 would let key
    presence, which is linear, read above chance). Held-out windows carry key pairs never in the fit half."""
    cap = VR.capacity(net, fit[VR.xor2_keyed(x, fit)], read[VR.xor2_keyed(x, read)], lambda t: VR.xor2_f(x, t), cfg, log)
    return {"held_out_real_by_site": cap["held_out_real_by_site"], "max": max(cap["held_out_real_by_site"].values()),
            "shuffled_by_site": {s: cap[s]["shuf"]["held_out"] for s in VR.SITES}}


def learned(m: Net, F: VR.Target, ft: Net, push: Tensor, check: Tensor, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    """E139's `xor_learned` on keyed windows of held-out pairs: KL(F || ft) against KL(F || M + half the push), the best a model
    blind to the bits can do on keyed windows. Learned if the ratio is at most the filed bar."""
    lf = R.last(F, check, cfg.read_batch)
    lhalf = torch.cat([run(m, t, edit={"in2": add_last(0.5 * push)})[0][:, -1] for t in chunks(check, cfg.read_batch)])
    k_ft, k_half = float(kl(lf, R.last(VR.net_target(ft), check, cfg.read_batch)).mean()), float(kl(lf, lhalf).mean())
    k_m = float(kl(lf, R.last(VR.net_target(m), check, cfg.read_batch)).mean())
    assert k_half > 0, "a zero push"
    out = {"kl_ft": k_ft, "kl_half_push": k_half, "kl_M": k_m, "ratio": k_ft / k_half, "learned": k_ft / k_half <= th["learned_kl_ratio"]}
    log(f"  learned check: {out}")
    return out


def save(ft: Net, path: Path) -> None:
    torch.save(ft.net.state_dict(), path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


# ---- job 1 --------------------------------------------------------------------------------------------------------------


def stage0(net: Net, cfg: Config, th: dict[str, Any], sets: VR.Sets, task: VR.Task, p: VR.Pieces, read: Tensor) -> float:
    path = cfg.results / "stage0.json"
    if path.is_file() and json.loads(path.read_text()).get("passed"):
        return json.loads(path.read_text())["floor_runs_tv"]
    out: dict[str, Any] = {"passed": False, "floor_runs_tv": R._floor(net, read[:8192], cfg)}
    zero_step(net, read, cfg, th, out["floor_runs_tv"], path, out, "plain")
    efit, eread = VR.pair_windows(cfg, "XOR", sets, task)
    cap = VR.capacity(net, efit, eread, lambda t: VR.xor_f(p, t), cfg, log)["held_out_real_by_site"]
    e137 = json.loads(E137.read_text())["capacity"]["held_out_real_by_site"]
    out["capacity_replay"] = {"read": cap, "e137": e137}
    if any(abs(cap[s] - e137[s]) > th["capacity_replay_tol"] for s in VR.SITES):
        stop(path, out, "the moved capacity read does not repeat E137's")
    out["passed"] = True
    dump(path, out)
    print("STAGE 0 DONE", flush=True)
    return out["floor_runs_tv"]


def calibrate(net: Net, x: VR.Xor2, unit: Tensor, read: Tensor, cfg: Config, th: dict[str, Any]) -> dict[str, Any]:
    """Per alpha: F's gap TV(F, M) and the gap a perfect taught original would leave, TV(F, M + push / 3). Chosen: the smallest
    alpha whose second gap is at least `taught_gap_aim`; if none, the alpha with the largest second gap (ties: the smaller)."""
    sums = torch.zeros(len(cfg.alphas), 2, dtype=torch.float64)
    w = read[: cfg.calib_n]
    for t in chunks(w, cfg.read_batch):
        lm = run(net, t)[0][:, -1]
        for i, a in enumerate(cfg.alphas):
            lf = VR.xor2_target(net, x, a * unit)(t)[0][:, -1]
            lt = VR.xor2_target(net, x, a * cfg.taught_share * unit)(t)[0][:, -1]
            sums[i, 0] += tv(lf, lm).sum().cpu()
            sums[i, 1] += tv(lf, lt).sum().cpu()
    rows = [{"alpha": a, "gap": float(s[0]) / w.shape[0], "ideal_taught_gap": float(s[1]) / w.shape[0]} for a, s in zip(cfg.alphas, sums, strict=True)]
    for r in rows:
        log(f"  alpha {r['alpha']}: gap {r['gap']:.4f}, ideal taught gap {r['ideal_taught_gap']:.4f}")
    ok = [r for r in rows if r["ideal_taught_gap"] >= th["taught_gap_aim"]]
    return {"rows": rows, "alpha": (ok[0] if ok else max(rows, key=lambda r: r["ideal_taught_gap"]))["alpha"]}


def ladder(base: Net, F: VR.Target, item: str, fit: Tensor, check: Tensor, cfg: Config, path: Path, out: dict[str, Any],
           passes: Any) -> tuple[Net, int]:
    """Fine-tune a copy of `base` toward F at 300 steps, then from scratch at 3000 steps if `passes(ft, rec)` is false.
    Returns the first copy that passes and its rung; (copy, -1) if neither passes. Each rung's record goes in out[item]."""
    ft = base
    for r, sch in enumerate(RUNGS):
        log(f"{item}: {sch.steps} steps, peak {sch.lr}")
        ft = R.clone_net(base)
        rec = VR.finetune(base, ft, F, fit, check, sch, cfg.batch, seed_of(item), log)
        out.setdefault(item, {})[str(sch.steps)] = rec
        if rec["stopped"]:
            stop(path, out, f"{item}: {rec['stopped']}")
        ok = passes(ft, rec)
        dump(path, out)
        if ok:
            return ft, r
    return ft, -1


def stage1(net: Net, cfg: Config, th: dict[str, Any], x: VR.Xor2, unit: Tensor, fit: Tensor, read: Tensor, floor: float) -> None:
    path = cfg.results / "stage1.json"
    if path.is_file() and json.loads(path.read_text()).get("passed"):
        return
    out: dict[str, Any] = {"passed": False, "keys": x.keys.tolist(), "n_held_pairs": int(x.held.sum())}
    out["calibration"] = calibrate(net, x, unit, read, cfg, th)
    push = out["calibration"]["alpha"] * unit
    F = VR.xor2_target(net, x, push)
    out["plain_gap"] = mean_tv(F, VR.net_target(net), read, cfg)["mean"]
    log(f"alpha {out['calibration']['alpha']}, plain target gap {out['plain_gap']:.4f}")
    if out["plain_gap"] <= th["target_gap_min"]:
        stop(path, out, "the plain arm's target gap is not above the bar")
    out["capacity_plain"] = capacity(net, x, fit, read, cfg)
    dump(path, out)
    if out["capacity_plain"]["max"] >= th["capacity_bar"]:
        stop(path, out, "the plain 67M already reads XOR2 affinely: no built premise")
    caps: dict[str, Any] = out.setdefault("capacity_taught", {})

    def has_it(ft: Net, rec: dict[str, Any]) -> bool:
        caps[str(rec["steps"])] = capacity(ft, x, fit, read, cfg)
        return caps[str(rec["steps"])]["max"] >= th["capacity_bar"]

    F3 = VR.xor2_target(net, x, cfg.taught_share * push)
    taught, r = ladder(net, F3, "taught_original", fit, read[: cfg.check_n], cfg, path, out, has_it)
    if r < 0:
        stop(path, out, "the taught original does not read XOR2 affinely at 300 or 3000 steps")
    out["taught_rung"] = r
    save(taught, cfg.results / "large" / "taught_original.pt")
    out["taught_gap"] = mean_tv(F, VR.net_target(taught), read, cfg)["mean"]
    log(f"taught target gap {out['taught_gap']:.4f}")
    if out["taught_gap"] <= th["target_gap_min"]:
        stop(path, out, "the taught arm's target gap is not above the bar")
    zero_step(taught, read, cfg, th, floor, path, out, "taught")
    check = read[VR.xor2_keyed(x, read)][: cfg.check_n]
    checks: dict[str, Any] = out.setdefault("learned_plain_s0", {})

    def learns(ft: Net, rec: dict[str, Any]) -> bool:
        checks[str(rec["steps"])] = learned(net, F, ft, push, check, cfg, th)
        return checks[str(rec["steps"])]["learned"]

    plain, r = ladder(net, F, "plain_s0", fit, read[: cfg.check_n], cfg, path, out, learns)
    if r < 0:
        out["not_learned"] = True
        stop(path, out, "CELL CLOSES: the plain fine-tune did not learn XOR2 on held-out key pairs at 300 or 3000 steps")
    save(plain, cfg.results / "large" / "plain_s0.pt")
    out.update(rung=r, passed=True)
    dump(path, out)
    print("STAGE 1 DONE", flush=True)


# ---- job 2 --------------------------------------------------------------------------------------------------------------


def item_h(m: Net, base: Net, arm: str, seed: int, F: VR.Target, push: Tensor, st1: dict[str, Any], cfg: Config,
           th: dict[str, Any], fit: Tensor, read: Tensor, check: Tensor) -> Net | None:
    """One fine-tune of `base` toward F (the plain seed 0 read from job 1), its distances and learned check, then H at five
    sites. Returns the fine-tune, or None when it has no reading or its H was read before (seed 1)."""
    item = f"{arm}_s{seed}"
    path = cfg.results / f"{item}.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"item": item, "arm": arm, "seed": seed, "G": {}, "H": {}}
    if out.get("no_reading") or (out.get("h_done") and seed != 0):
        return None
    ft = R.clone_net(base)
    if item == "plain_s0":
        ft.net.load_state_dict(torch.load(cfg.results / "large" / "plain_s0.pt", map_location=base.device))
        out["train"] = "stage1.json, plain_s0"
    else:
        log(f"{item}: {RUNGS[st1['rung']].steps} steps")
        out["train"] = VR.finetune(base, ft, F, fit, read[: cfg.check_n], RUNGS[st1["rung"]], cfg.batch, seed_of(item), log)
        if out["train"]["stopped"]:
            stop(path, out, f"{item}: {out['train']['stopped']}")
    out["dist"] = R.distances(base, F, ft, read, cfg)
    out["learned"] = learned(m, F, ft, push, check, cfg, th)
    log(f"{item}: {out['dist']}")
    if not out["learned"]["learned"] or out["dist"]["gap"] < th["floor_gap"]:
        out["no_reading"] = "not learned" if not out["learned"]["learned"] else "gap under the floor"
        dump(path, out)
        log(f"{item}: no reading ({out['no_reading']})")
        return None
    dump(path, out)
    A2.search_h(base, cfg, f"{EXPERIMENT}/{item}", fit, read, VR.net_target(ft), out["dist"]["gap"], out, lambda: dump(path, out))
    out["h_done"] = True
    dump(path, out)
    print(f"STAGE 2 {item} H DONE", flush=True)
    return ft


def item_g(base: Net, ft: Net, item: str, sites: tuple[str, ...], cfg: Config, fit: Tensor, read: Tensor) -> None:
    path = cfg.results / f"{item}.json"
    out = json.loads(path.read_text())
    A2.search_g(base, cfg, f"{EXPERIMENT}/{item}", sites, fit, read, VR.net_target(ft), out["dist"]["gap"], out, lambda: dump(path, out))
    out["g_done"] = True
    dump(path, out)
    print(f"STAGE 3 {item} G DONE", flush=True)


def job2(net: Net, cfg: Config, th: dict[str, Any], x: VR.Xor2, unit: Tensor, fit: Tensor, read: Tensor, st1: dict[str, Any]) -> None:
    push = st1["calibration"]["alpha"] * unit
    F = VR.xor2_target(net, x, push)
    taught = R.clone_net(net)
    taught.net.load_state_dict(torch.load(cfg.results / "large" / "taught_original.pt", map_location=net.device))
    bases = {"taught": taught, "plain": net}
    check = read[VR.xor2_keyed(x, read)][: cfg.check_n]
    seed0: dict[str, Net | None] = {}
    for seed in cfg.seeds:  # H first, both arms, seed by seed: the headline reading before the slower G
        for arm, base in bases.items():
            ft = item_h(net, base, arm, seed, F, push, st1, cfg, th, fit, read, check)
            if seed == 0:
                seed0[arm] = ft
            torch.cuda.empty_cache()
    for arm, base in bases.items():
        ft = seed0[arm]
        if ft is not None:
            item_g(base, ft, f"{arm}_s0", cfg.g_sites[st1["rung"]], cfg, fit, read)


def main(job: str, net: Net) -> int:
    cfg, th = Config(), json.loads(PRED.read_text())["thresholds"]
    sets, task, p = A1.setup(net, cfg)
    from commute import vpd

    x = VR.make_xor2(task, vpd.tokenizer(), int(net.net.lm_head.weight.shape[0]), cfg.n_keys2, cfg.held_share)  # type: ignore[union-attr]
    fit, read = VR.xor2_windows(cfg, sets, x)
    unit = (p.Q @ p.b).to(net.device)
    (cfg.results / "large").mkdir(parents=True, exist_ok=True)
    log(f"{EXPERIMENT} {job}: rows ready, {x.keys.numel()} keys, {int(x.held.sum())} held-out pairs, |unit push| {float(unit.norm()):.3f}")
    if job == "job1":
        floor = stage0(net, cfg, th, sets, task, p, read)
        stage1(net, cfg, th, x, unit, fit, read, floor)
        return 0
    st1_path = cfg.results / "stage1.json"
    st1 = json.loads(st1_path.read_text()) if st1_path.is_file() else {}
    if not st1.get("passed"):
        log("job 2 refused: stage 1 has not passed (run job1 first)")
        return 1
    try:
        job2(net, cfg, th, x, unit, fit, read, st1)
    except R.MoreToDo:
        log("more to do: wall limit reached; what was read is saved")
        return 3
    log(f"{EXPERIMENT} JOB 2 DONE")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("job1", "job2"))
    job_ = ap.parse_args().job
    require_filed(PRED)
    code = 1
    try:
        code = main(job_, Net("cuda"))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
