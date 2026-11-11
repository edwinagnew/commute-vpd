"""E150 (cell P1.10): the built control on the parity of a sum. Plan: reports/plans/2026-11-11_plan_built-control-sum.md;
design: ~/commute-data/scratch/definition-lead/built-control/fable_advice_natural_2026-11-11.md (option A and its addendum,
chosen by the user). Filed rules: results/predictions.json (the run refuses unless it is committed unchanged).

Windows end in two numbers a, b (0 to 99, each one token with a leading space; a quarter of the ordered pairs held out for
reading). Each arm fine-tunes its original toward that original + a push at block 2's input, last position, wherever its bit
is 1. The push is E137's XOR push (its pieces' Q b) times one alpha for all arms, chosen in stage 1. Arms:
    control  plain 67M, bit "b is odd" (a function of one token: the positive control, read first)
    plain    plain 67M, bit "a + b is odd" (the built case)
    taught   the taught original (the 67M fine-tuned toward 67M + alpha Q b' on the same bit, b' another direction of the
             pieces' span: the same skill with other answers), bit "a + b is odd"; runs only if a ridge read of the bit from
             the taught original is at least 0.9 on held-out pairs
Each arm: seed 0 at 300 steps, else 3000 (learned check), seed 1 on seed 0's rung; H on both seeds, G on seed 0.

    run.py job1   stage 0 (zero-step control, E148's saved plain fine-tune read for XOR2, single-number reads on the 67M),
                  stage 1 (alpha and target gaps), the control arm, then (unless the control reads below 0.5 on both
                  seeds) the taught original and its gate; exit 3 before the taught original if under 20 min are left
    run.py job2   the plain arm; refused unless the control arm read and is not below 0.5 on both seeds
    run.py job3   the taught arm; refused as job2, and exits 0 at once if the taught original was dropped
Writes results/stage0.json, stage1.json, taught_original.json, {arm}_s{seed}.json after each step; results/large/
taught_original.pt and {arm}_s0.pt. Exit 0 done, 1 a stop or refusal, 3 the wall limit (resubmit the same job).
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute.hidden_maps import chunks, run, tv
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.steering import Net

R: Any = load_experiment("reuse_variable_finetune", "run")  # E139: Config, clone_net, distances, log, dump, stop, _floor, MoreToDo
A1: Any = load_experiment("reuse_variable_finetune", "amend1")  # setup: rows, E137's task and pieces
A2: Any = load_experiment("reuse_variable_finetune", "amend2")  # search_g over a list of sites, search_h
P: Any = load_experiment("built_control_sum", "plot")  # best, side: the filed reading of an arm
EXPERIMENT = "E150"
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
E148 = HERE.parent / "built_control" / "results"  # its stage1.json (keys) and large/plain_s0.pt, read only
log, dump, stop = R.log, R.dump, R.stop


@dataclass(frozen=True)
class Config(R.Config):
    results: Path = HERE / "results"
    n_max: int = 100  # 0 to 99, all one token as " n": fewer numbers than the model's width, so a bit of b is affine in b's embedding
    held_share: float = 0.25  # as E148
    alphas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)  # E148's grid, units of E137's XOR push
    calib_n: int = 20_000  # as E148
    seeds: tuple[int, ...] = (0, 1)
    rungs: tuple[VR.Schedule, ...] = (VR.Schedule(300, 3e-5, 30, (1, 2, 3, 5, 10, 30, 100, 300)),  # E139's recipe
                                      VR.Schedule(3000, 1e-5, 300, (1, 2, 3, 5, 10, 30, 100, 300, 1000, 2000, 3000)))  # E139 A2's
    g_sites: tuple[tuple[str, ...], ...] = (VR.SITES, ("in2",))  # per rung, as E148: in2 only after 3000 steps
    e148_keys: int = 128  # E148's Config.n_keys2 and held_share: its XOR2 rebuilt for the stage-0 read
    e148_held: float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()
        assert list(self.alphas) == sorted(self.alphas) and self.calib_n <= self.D // 2 and self.seeds == (0, 1)
        assert len(self.g_sites) == len(self.rungs) and 0 < self.held_share < 1


@dataclass
class Ctx:
    net: Net
    cfg: Config
    th: dict[str, Any]
    x: VR.Xor2
    fit: Tensor
    read: Tensor
    unit: Tensor  # E137's XOR push, Q b
    unit2: Tensor  # Q b', the taught original's push direction

    def bit(self, arm: str) -> Callable[[Tensor], Tensor]:
        if arm == "control":
            return lambda t: VR.odd_at(self.x, t, -1)
        return lambda t: VR.xor2_f(self.x, t)

    def check(self) -> Tensor:
        """Keyed windows of held-out pairs, for the learned check."""
        return self.read[VR.xor2_keyed(self.x, self.read)][: self.cfg.check_n]

    def path(self, name: str) -> Path:
        return self.cfg.results / f"{name}.json"

    def load(self, name: str) -> dict[str, Any]:
        return json.loads(self.path(name).read_text()) if self.path(name).is_file() else {}


def seed_of(item: str) -> int:
    return 300_000 + zlib.crc32(item.encode()) % 65_536  # E139's batch-draw seed rule


def other_direction(b: Tensor) -> Tensor:
    """A seeded vector orthogonal to b with b's length: Q b' is then a push as large as Q b, at right angles to it."""
    g = torch.randn(b.shape, generator=torch.Generator().manual_seed(150_002)).to(b)
    g = g - (g @ b) / (b @ b) * b
    return g * b.norm() / g.norm()


def zero_step(c: Ctx, base: Net, name: str, floor: float, path: Path, out: dict[str, Any]) -> None:
    """E139's W0 control: a fine-tune of `base` at zero steps is `base`: max TV at the float floor, gap under floor_gap."""
    d = VR.mean_tv(VR.net_target(R.clone_net(base)), VR.net_target(base), c.read, c.cfg.read_batch)
    out[f"zero_step_{name}"] = d
    log(f"zero-step {name}: gap {d['mean']:.3e}, max TV {d['max']:.3e}, floor {floor:.3e}")
    if d["max"] > floor or d["mean"] >= c.th["floor_gap"]:
        stop(path, out, f"zero-step fine-tune of the {name} original is off the float floor or reads")


def save(ft: Net, path: Path) -> None:
    torch.save(ft.net.state_dict(), path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def load_net(c: Ctx, path: Path) -> Net:
    ft = R.clone_net(c.net)
    ft.net.load_state_dict(torch.load(path, map_location=c.net.device))
    return ft


# ---- job 1: stage 0, stage 1, the taught original, the control arm ---------------------------------------------------


def e148_read(c: Ctx, sets: VR.Sets, task: VR.Task, tok: Any, path: Path, out: dict[str, Any]) -> dict[str, Any]:
    """The XOR2 capacity read on E148's saved plain seed-0 fine-tune (300 steps toward the full push), on E148's held-out pairs."""
    x = VR.make_xor2(task, tok, c.x.bit.numel(), c.cfg.e148_keys, c.cfg.e148_held)
    if x.keys.tolist() != json.loads((E148 / "stage1.json").read_text())["keys"]:
        stop(path, out, "E148's XOR2 keys did not rebuild")
    fit, read = VR.xor2_windows(c.cfg, sets, x)
    cap = VR.keyed_capacity(load_net(c, E148 / "large" / "plain_s0.pt"), x, lambda t: VR.xor2_f(x, t), fit, read, c.cfg, log)
    return {**cap, "linear": cap["max"] >= c.th["linear_bar"]}


def stage0(c: Ctx, sets: VR.Sets, task: VR.Task, tok: Any) -> float:
    path = c.path("stage0")
    if c.load("stage0").get("passed"):
        return c.load("stage0")["floor_runs_tv"]
    out: dict[str, Any] = {"passed": False, "number_tokens": c.x.keys.tolist(), "n_held_pairs": int(c.x.held.sum()),
                           "floor_runs_tv": R._floor(c.net, c.read[:8192], c.cfg)}
    zero_step(c, c.net, "plain", out["floor_runs_tv"], path, out)
    out["e148_plain_s0_xor2"] = e148_read(c, sets, task, tok, path, out)
    dump(path, out)
    labels = {"a_odd": lambda t: VR.odd_at(c.x, t, -2), "b_odd": c.bit("control"), "sum_odd": c.bit("plain")}
    reads = out.setdefault("single_reads", {})
    for name, label in labels.items():
        reads[name] = VR.keyed_capacity(c.net, c.x, label, c.fit, c.read, c.cfg, log)
        dump(path, out)
    out["a_linear_at_last"] = reads["a_odd"]["max"] >= c.th["linear_bar"]
    if not out["a_linear_at_last"]:
        out["note"] = "a's parity is not linearly read at the last position: the searches, which read the last position only, cannot take it from there"
    if reads["sum_odd"]["max"] >= c.th["capacity_bar"]:
        stop(path, out, "the plain 67M already reads the sum's parity affinely: no built premise")
    if reads["b_odd"]["max"] < c.th["linear_bar"]:
        stop(path, out, "the control's bit (b odd) is not linearly read from the plain 67M: no positive control")
    out["passed"] = True
    dump(path, out)
    print("STAGE 0 DONE", flush=True)
    return out["floor_runs_tv"]


def calibrate(c: Ctx) -> dict[str, Any]:
    """Per alpha, on read windows: the gaps TV(F, M) of the control, plain and taught-original targets. Chosen: the smallest
    alpha whose control and plain gaps are both at least gap_aim; if none, the one with the largest smaller gap (ties: smaller)."""
    arms = {"control": (c.bit("control"), c.unit), "plain": (c.bit("plain"), c.unit), "taught_original": (c.bit("plain"), c.unit2)}
    sums = torch.zeros(len(c.cfg.alphas), len(arms), dtype=torch.float64)
    w = c.read[: c.cfg.calib_n]
    for t in chunks(w, c.cfg.read_batch):
        lm = run(c.net, t)[0][:, -1]
        for i, a in enumerate(c.cfg.alphas):
            for j, (label, u) in enumerate(arms.values()):
                sums[i, j] += tv(VR.label_target(c.net, label, a * u)(t)[0][:, -1], lm).sum().cpu()
    rows = [{"alpha": a, **{k: float(v) / w.shape[0] for k, v in zip(arms, s, strict=True)}} for a, s in zip(c.cfg.alphas, sums, strict=True)]
    for r in rows:
        log(f"  alpha {r['alpha']}: gaps control {r['control']:.4f}, plain {r['plain']:.4f}, taught original {r['taught_original']:.4f}")
    ok = [r for r in rows if min(r["control"], r["plain"]) >= c.th["gap_aim"]]
    return {"rows": rows, "alpha": (ok[0] if ok else max(rows, key=lambda r: min(r["control"], r["plain"])))["alpha"]}


def stage1(c: Ctx) -> float:
    path = c.path("stage1")
    if c.load("stage1").get("passed"):
        return c.load("stage1")["calibration"]["alpha"]
    out: dict[str, Any] = {"passed": False, "calibration": calibrate(c)}
    alpha = out["calibration"]["alpha"]
    out["gaps"] = {arm: VR.mean_tv(VR.label_target(c.net, c.bit(arm), alpha * c.unit), VR.net_target(c.net), c.read,
                                   c.cfg.read_batch)["mean"] for arm in ("control", "plain")}
    log(f"alpha {alpha}, target gaps {out['gaps']}")
    if min(out["gaps"].values()) <= c.th["target_gap_min"]:
        stop(path, out, "a target gap is not above the bar")
    out["passed"] = True
    dump(path, out)
    print("STAGE 1 DONE", flush=True)
    return alpha


def drop(path: Path, out: dict[str, Any], why: str) -> None:
    out["dropped"] = why
    dump(path, out)
    log(f"TAUGHT ARM DROPPED: {why}")
    print("TAUGHT ORIGINAL DONE", flush=True)


def taught_original(c: Ctx, alpha: float, floor: float) -> None:
    """The 67M fine-tuned toward 67M + alpha Q b' on the sum's bit, 300 steps, else 3000; the gate: a ridge read of the bit on
    held-out pairs at least linear_bar. Then its own target's gap and its zero-step control. A miss drops the taught arm."""
    path, out = c.path("taught_original"), c.load("taught_original")
    if out.get("passed") or out.get("dropped"):
        return
    caps: dict[str, Any] = out.setdefault("capacity", {})

    def has_it(ft: Net, rec: dict[str, Any]) -> bool:
        caps[str(rec["steps"])] = VR.keyed_capacity(ft, c.x, c.bit("plain"), c.fit, c.read, c.cfg, log)
        dump(path, out)
        return caps[str(rec["steps"])]["max"] >= c.th["linear_bar"]

    T0 = VR.label_target(c.net, c.bit("plain"), alpha * c.unit2)
    log("taught original")
    ft, r, out["train"] = VR.ladder(c.net, T0, c.fit, c.read[: c.cfg.check_n], c.cfg.rungs, c.cfg.batch,
                                    seed_of("taught_original"), R.clone_net, has_it, log)
    if ft is None:
        return drop(path, out, "the step-10 check missed" if r >= 0 else "the gate missed: the bit's read stayed under linear_bar at 300 and 3000 steps")
    save(ft, c.cfg.results / "large" / "taught_original.pt")
    out["rung"] = r
    out["gap"] = VR.mean_tv(VR.label_target(ft, c.bit("plain"), alpha * c.unit), VR.net_target(ft), c.read, c.cfg.read_batch)["mean"]
    log(f"taught target gap {out['gap']:.4f}")
    if out["gap"] <= c.th["target_gap_min"]:
        return drop(path, out, "the taught arm's target gap is not above target_gap_min")
    zero_step(c, ft, "taught", floor, path, out)
    out["passed"] = True
    dump(path, out)
    print("TAUGHT ORIGINAL DONE", flush=True)


# ---- an arm: fine-tunes, H on both seeds, G on seed 0 ------------------------------------------------------------------


def fine_tune(c: Ctx, base: Net, F: VR.Target, push: Tensor, item: str, rungs: tuple[VR.Schedule, ...], path: Path,
              out: dict[str, Any]) -> Net | None:
    """Train `item` on `rungs` (the ladder for seed 0, seed 0's rung for seed 1) until learned; a step-10 miss stops the job.
    Saved, for seed 0's G search and a resume. None, with `no_reading` set, if not learned."""
    pt = c.cfg.results / "large" / f"{item}.pt"
    if "rung" in out and pt.is_file():
        return load_net(c, pt)
    check = c.check()
    learned: dict[str, Any] = out.setdefault("learned", {})

    def learns(ft: Net, rec: dict[str, Any]) -> bool:
        lr = VR.learned_ratio(base, F, ft, push, check, c.cfg.read_batch)
        learned[str(rec["steps"])] = {**lr, "learned": lr["ratio"] <= c.th["learned_kl_ratio"]}
        log(f"  learned check: {learned[str(rec['steps'])]}")
        return learned[str(rec["steps"])]["learned"]

    ft, r, out["train"] = VR.ladder(base, F, c.fit, c.read[: c.cfg.check_n], rungs, c.cfg.batch, seed_of(item), R.clone_net, learns, log)
    if ft is None and r >= 0:
        stop(path, out, f"{item}: step-10 check missed")
    if ft is None:
        out["no_reading"] = "not learned"
        dump(path, out)
        return None
    out["rung"] = c.cfg.rungs.index(rungs[r])
    save(ft, pt)
    dump(path, out)
    return ft


def item_h(c: Ctx, arm: str, seed: int, base: Net, F: VR.Target, push: Tensor) -> Net | None:
    """One fine-tune of the arm, its distances, then H at five sites. Returns it, or None without a reading."""
    item = f"{arm}_s{seed}"
    path = c.path(item)
    out = c.load(item) or {"item": item, "arm": arm, "seed": seed, "G": {}, "H": {}}
    if out.get("no_reading"):
        return None
    if out.get("h_done"):
        return load_net(c, c.cfg.results / "large" / f"{item}.pt")
    rungs = c.cfg.rungs if seed == 0 else (c.cfg.rungs[c.load(f"{arm}_s0")["rung"]],)
    log(f"{item}")
    ft = fine_tune(c, base, F, push, item, rungs, path, out)
    if ft is None:
        return None
    out["dist"] = R.distances(base, F, ft, c.read, c.cfg)
    log(f"{item}: {out['dist']}")
    if out["dist"]["gap"] < c.th["floor_gap"]:
        out["no_reading"] = "gap under the floor"
        dump(path, out)
        return None
    dump(path, out)
    A2.search_h(base, c.cfg, f"{EXPERIMENT}/{item}", c.fit, c.read, VR.net_target(ft), out["dist"]["gap"], out, lambda: dump(path, out))
    out["h_done"] = True
    dump(path, out)
    print(f"ARM {item} H DONE", flush=True)
    return ft


def do_arm(c: Ctx, arm: str, base: Net, push: Tensor) -> None:
    F = VR.label_target(base, c.bit(arm), push)
    ft0 = item_h(c, arm, 0, base, F, push)
    if ft0 is None:
        log(f"{arm}: seed 0 has no reading, so the arm is 'no call'; seed 1 and G are not run")
        return
    torch.cuda.empty_cache()
    item_h(c, arm, 1, base, F, push)
    torch.cuda.empty_cache()
    path, out = c.path(f"{arm}_s0"), c.load(f"{arm}_s0")
    if not out.get("g_done"):
        A2.search_g(base, c.cfg, f"{EXPERIMENT}/{arm}_s0", c.cfg.g_sites[out["rung"]], c.fit, c.read, VR.net_target(ft0),
                    out["dist"]["gap"], out, lambda: dump(path, out))
        out["g_done"] = True
        dump(path, out)
    print(f"ARM {arm} DONE", flush=True)


def control_side(c: Ctx) -> str | None:
    """The control arm's filed side of the H cut-off; None while it has not been read on both seeds."""
    recs = [c.load(f"control_s{s}") for s in c.cfg.seeds]
    if recs[0].get("no_reading") or recs[1].get("no_reading"):
        return "no call"
    if not all(r.get("h_done") for r in recs):
        return None
    return P.side([P.best(r)["H"] for r in recs])


def main(job: str, net: Net) -> int:
    cfg, th = Config(), json.loads(PRED.read_text())["thresholds"]
    sets, task, p = A1.setup(net, cfg)
    from commute import vpd

    tok = vpd.tokenizer()
    x = VR.make_numbers(tok, int(net.net.lm_head.weight.shape[0]), cfg.n_max, cfg.held_share)  # type: ignore[union-attr]
    fit, read = VR.xor2_windows(cfg, sets, x)
    c = Ctx(net, cfg, th, x, fit, read, (p.Q @ p.b).to(net.device), (p.Q @ other_direction(p.b)).to(net.device))
    (cfg.results / "large").mkdir(parents=True, exist_ok=True)
    log(f"{EXPERIMENT} {job}: {x.keys.numel()} numbers, {int(x.held.sum())} held-out pairs, |Q b| {float(c.unit.norm()):.3f}, "
        f"|Q b'| {float(c.unit2.norm()):.3f}, cos {float(c.unit @ c.unit2 / c.unit.norm() / c.unit2.norm()):.1e}")
    if job == "job1":
        floor = stage0(c, sets, task, tok)
        alpha = stage1(c)
        do_arm(c, "control", net, alpha * c.unit)
        if control_side(c) == "below":
            log("the control arm reads below 0.5 on both seeds: rival b, the built control is uninformative, nothing further runs")
            return 0
        if time.monotonic() - R._t0 > cfg.wall_limit_s - 1200:  # its worst case, 3000 steps and two reads, is ~18 min on an L40
            raise R.MoreToDo
        taught_original(c, alpha, floor)
        return 0
    side = control_side(c)
    if side in (None, "below"):
        log(f"{job} refused: the control arm reads {side or 'not yet'} (below on both seeds: rival b, nothing further runs)")
        return 1
    alpha = c.load("stage1")["calibration"]["alpha"]
    if job == "job2":
        do_arm(c, "plain", net, alpha * c.unit)
        return 0
    t0 = c.load("taught_original")
    if t0.get("dropped"):
        log(f"job3: the taught arm was dropped ({t0['dropped']}); nothing to run")
        return 0
    if not t0.get("passed"):
        log("job3 refused: the taught original has not been trained (job1 not finished)")
        return 1
    do_arm(c, "taught", load_net(c, cfg.results / "large" / "taught_original.pt"), alpha * c.unit)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("job1", "job2", "job3"))
    job_ = ap.parse_args().job
    require_filed(PRED)
    code = 1
    try:
        code = main(job_, Net("cuda"))
        log(f"{EXPERIMENT} {job_} DONE" if code == 0 else f"{EXPERIMENT} {job_} exit {code}")
    except R.MoreToDo:
        log("more to do: wall limit reached; what was read is saved (resubmit the same job)")
        code = 3
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
