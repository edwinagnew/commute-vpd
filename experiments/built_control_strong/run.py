"""E153 (cell P1.11): E148's built control on XOR2 with a strong positive control. Plan:
reports/plans/2026-11-11_plan_built-control-strong.md; card: card.txt. Filed rules: results/predictions.json (the run refuses
unless it is committed unchanged).

E148's taught original read XOR2 at only 0.588. Its plain seed-0 fine-tune (67M fine-tuned 300 steps toward 67M + push on
XOR2 windows; saved at built_control/results/large/plain_s0.pt) reads XOR2 linearly at 0.9997 (E150's stage 0). That model
is the original M' here. It is fine-tuned toward its own target F' = M' + push (E148's push: E137's Q b times E148's alpha,
added at block 2's input, last position, where the two keys' bits differ), with E148's keys, windows, held-out pairs, recipe
and learned check, on 2 seeds; H is read on both seeds and G on seed 0, against M'. The built side is E148's plain arm,
read already (H 0.080, 0.087); nothing of it reruns.

Confound: M' was trained toward 67M + push, so F' is the push applied twice. Stage 1 checks that the second push still moves
M' (gap above E148's target_gap_min) and reads how far a perfect M' would move, TV(67M + 2 push, 67M + push).

    run.py    stage 0: the float floor on M', the zero-step fine-tune of M' (W0), the XOR2 capacity read on M' (stop under
              0.9); stage 1: the target gap TV(F', M') (stop unless above 0.15); stage 2: seed 0 (300 steps, else 3000, until
              the learned check passes), H; seed 1 on seed 0's rung, H; G on seed 0 (all four sites, in2 only after 3000).
Writes results/stage0.json, stage1.json, strong_s{0,1}.json after each step, results/large/strong_s{0,1}.pt. Exit 0 done,
1 a stop, 3 the wall limit (105 min; resubmit the same command, it resumes from the saved files).
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.variable_read as VR
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.steering import Net

R: Any = load_experiment("reuse_variable_finetune", "run")  # E139: Config, clone_net, distances, log, dump, stop, _floor, budget, MoreToDo
A1: Any = load_experiment("reuse_variable_finetune", "amend1")  # setup: rows, E137's task and pieces
A2: Any = load_experiment("reuse_variable_finetune", "amend2")  # search_g over a list of sites, search_h
EXPERIMENT = "E153"
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
E148 = HERE.parent / "built_control" / "results"  # stage1.json (keys, alpha, plain gap), large/plain_s0.pt; read only
ARM = "strong"
MATCHED = "plain"  # E148's plain arm: each seed's batch draws use its item name, so only the original differs
log, dump, stop = R.log, R.dump, R.stop


@dataclass(frozen=True)
class Config(R.Config):
    results: Path = HERE / "results"
    n_keys2: int = 128  # E148's Config.n_keys2 and held_share: its XOR2 rebuilt
    held_share: float = 0.25
    seeds: tuple[int, ...] = (0, 1)
    rungs: tuple[VR.Schedule, ...] = (VR.Schedule(300, 3e-5, 30, (1, 2, 3, 5, 10, 30, 100, 300)),  # E139's recipe, E148's rung 0
                                      VR.Schedule(3000, 1e-5, 300, (1, 2, 3, 5, 10, 30, 100, 300, 1000, 2000, 3000)))  # E148's rung 1
    g_sites: tuple[tuple[str, ...], ...] = (VR.SITES, ("in2",))  # per rung, as E148: in2 only after 3000 steps

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.seeds == (0, 1) and len(self.g_sites) == len(self.rungs) and 0 < self.held_share < 1


@dataclass
class Ctx:
    net: Net  # the plain 67M
    orig: Net  # M', E148's plain seed-0 fine-tune
    cfg: Config
    th: dict[str, Any]
    x: VR.Xor2
    fit: Tensor
    read: Tensor
    push: Tensor  # E148's push, alpha Q b
    F: VR.Target  # F' = M' + push where xor2_f is 1

    def check(self) -> Tensor:
        """Keyed windows of held-out pairs, for the learned check (E148's)."""
        return self.read[VR.xor2_keyed(self.x, self.read)][: self.cfg.check_n]

    def path(self, name: str) -> Path:
        return self.cfg.results / f"{name}.json"

    def load(self, name: str) -> dict[str, Any]:
        return json.loads(self.path(name).read_text()) if self.path(name).is_file() else {}


def seed_of(item: str) -> int:
    return 300_000 + zlib.crc32(item.encode()) % 65_536  # E139's batch-draw seed rule


# ---- stage 0: controls on M' --------------------------------------------------------------------------------------------


def stage0(c: Ctx) -> None:
    """W0: a fine-tune of M' at zero steps is M' (gap 0, max TV under the float floor measured on M'). Then the XOR2 capacity
    read on M', which E150's stage 0 read at 0.9997; under strong_capacity_bar M' is no strong positive control."""
    path = c.path("stage0")
    if c.load("stage0").get("passed"):
        return
    out: dict[str, Any] = {"passed": False, "floor_runs_tv": R._floor(c.orig, c.read[:8192], c.cfg)}
    d = VR.mean_tv(VR.net_target(R.clone_net(c.orig)), VR.net_target(c.orig), c.read, c.cfg.read_batch)
    out["zero_step"] = d
    log(f"zero-step M': gap {d['mean']:.3e}, max TV {d['max']:.3e}, floor {out['floor_runs_tv']:.3e}")
    if d["max"] > out["floor_runs_tv"] or d["mean"] >= c.th["floor_gap"]:
        stop(path, out, "the zero-step fine-tune of M' is off the float floor or reads")
    cap = VR.keyed_capacity(c.orig, c.x, lambda t: VR.xor2_f(c.x, t), c.fit, c.read, c.cfg, log)
    out["capacity"] = cap  # E150 read 0.99968 on the same model, windows and code (filed in predictions.json, S0b)
    dump(path, out)
    if cap["max"] < c.th["strong_capacity_bar"]:
        stop(path, out, "M' does not read XOR2 linearly: no strong positive control")
    out["passed"] = True
    dump(path, out)
    print("STAGE 0 DONE", flush=True)


# ---- stage 1: the target gap and the push applied twice ---------------------------------------------------------------


def stage1(c: Ctx) -> None:
    """gap = TV(F', M'). Beside it: E148's plain gap TV(67M + push, 67M), and the gap a perfect M' would leave,
    TV(67M + 2 push, 67M + push). A second push that moves much less than the first says the target sits where the push
    saturates."""
    path = c.path("stage1")
    if c.load("stage1").get("passed"):
        return
    b = c.cfg.read_batch
    out: dict[str, Any] = {"passed": False, "gap": VR.mean_tv(c.F, VR.net_target(c.orig), c.read, b)["mean"],
                           "e148_plain_gap": json.loads((E148 / "stage1.json").read_text())["plain_gap"]}
    once, twice = VR.xor2_target(c.net, c.x, c.push), VR.xor2_target(c.net, c.x, 2 * c.push)
    out["ideal_second_gap"] = VR.mean_tv(twice, once, c.read, b)["mean"]
    out["second_over_first"] = out["gap"] / out["e148_plain_gap"]
    log(f"target gap TV(F', M') {out['gap']:.4f}; E148's plain gap {out['e148_plain_gap']:.4f}; "
        f"TV(67M + 2 push, 67M + push) {out['ideal_second_gap']:.4f}")
    if out["gap"] <= c.th["target_gap_min"]:
        out["saturated"] = out["ideal_second_gap"] <= c.th["target_gap_min"]
        stop(path, out, "the target gap TV(F', M') is not above target_gap_min" +
             (": the push saturates (a perfect M' would leave no more)" if out["saturated"] else ""))
    out["passed"] = True
    dump(path, out)
    print("STAGE 1 DONE", flush=True)


# ---- stage 2: the strong arm, H on both seeds, G on seed 0 -------------------------------------------------------------


def fine_tune(c: Ctx, item: str, seed: int, rungs: tuple[VR.Schedule, ...], path: Path, out: dict[str, Any]) -> Net | None:
    """Fine-tune M' toward F' on `rungs` (the ladder for seed 0, seed 0's rung for seed 1) until E148's learned check passes;
    a step-10 miss stops the job. Saved, for G and a resume. None, with `no_reading` set, if not learned."""
    pt = c.cfg.results / "large" / f"{item}.pt"
    if "rung" in out and pt.is_file():
        return VR.load_state(c.net, pt, R.clone_net)
    R.budget(c.cfg)
    check = c.check()
    learned: dict[str, Any] = out.setdefault("learned", {})

    def learns(ft: Net, rec: dict[str, Any]) -> bool:
        lr = VR.learned_ratio(c.orig, c.F, ft, c.push, check, c.cfg.read_batch)
        learned[str(rec["steps"])] = {**lr, "learned": lr["ratio"] <= c.th["learned_kl_ratio"]}
        log(f"  learned check: {learned[str(rec['steps'])]}")
        return learned[str(rec["steps"])]["learned"]

    ft, r, out["train"] = VR.ladder(c.orig, c.F, c.fit, c.read[: c.cfg.check_n], rungs, c.cfg.batch,
                                    seed_of(f"{MATCHED}_s{seed}"), R.clone_net, learns, log)
    if ft is None and r >= 0:
        stop(path, out, f"{item}: the step-10 check missed")
    if ft is None:
        out["no_reading"] = "not learned"
        dump(path, out)
        return None
    out["rung"] = c.cfg.rungs.index(rungs[r])
    VR.save_state(ft, pt)
    dump(path, out)
    return ft


def item_h(c: Ctx, seed: int) -> Net | None:
    """One fine-tune of M', its distances, then H at five sites against M'. Returns it, or None without a reading."""
    item = f"{ARM}_s{seed}"
    path = c.path(item)
    out = c.load(item) or {"item": item, "arm": ARM, "seed": seed, "G": {}, "H": {}}
    if out.get("no_reading"):
        return None
    if out.get("h_done"):
        return VR.load_state(c.net, c.cfg.results / "large" / f"{item}.pt", R.clone_net)
    rungs = c.cfg.rungs if seed == 0 else (c.cfg.rungs[c.load(f"{ARM}_s0")["rung"]],)
    log(item)
    ft = fine_tune(c, item, seed, rungs, path, out)
    if ft is None:
        return None
    out["dist"] = R.distances(c.orig, c.F, ft, c.read, c.cfg)
    log(f"{item}: {out['dist']}")
    if out["dist"]["gap"] < c.th["floor_gap"]:
        out["no_reading"] = "gap under the floor"
        dump(path, out)
        return None
    dump(path, out)
    A2.search_h(c.orig, c.cfg, f"{EXPERIMENT}/{item}", c.fit, c.read, VR.net_target(ft), out["dist"]["gap"], out,
                lambda: dump(path, out))
    out["h_done"] = True
    dump(path, out)
    print(f"STAGE 2 {item} H DONE", flush=True)
    return ft


def stage2(c: Ctx) -> None:
    ft0 = item_h(c, 0)
    if ft0 is None:
        log(f"{ARM}: seed 0 has no reading, so the arm is 'no call'; seed 1 and G are not run")
        return
    torch.cuda.empty_cache()
    item_h(c, 1)
    torch.cuda.empty_cache()
    path, out = c.path(f"{ARM}_s0"), c.load(f"{ARM}_s0")
    if not out.get("g_done"):
        A2.search_g(c.orig, c.cfg, f"{EXPERIMENT}/{ARM}_s0", c.cfg.g_sites[out["rung"]], c.fit, c.read, VR.net_target(ft0),
                    out["dist"]["gap"], out, lambda: dump(path, out))
        out["g_done"] = True
        dump(path, out)
    print("STAGE 2 DONE", flush=True)


def main(net: Net) -> int:
    cfg, th = Config(), json.loads(PRED.read_text())["thresholds"]
    sets, task, p = A1.setup(net, cfg)
    from commute import vpd

    x = VR.make_xor2(task, vpd.tokenizer(), int(net.net.lm_head.weight.shape[0]), cfg.n_keys2, cfg.held_share)  # type: ignore[union-attr]
    e148 = json.loads((E148 / "stage1.json").read_text())
    if x.keys.tolist() != e148["keys"]:
        raise SystemExit("E148's XOR2 keys did not rebuild")
    fit, read = VR.xor2_windows(cfg, sets, x)
    push = e148["calibration"]["alpha"] * (p.Q @ p.b).to(net.device)
    orig = VR.load_state(net, E148 / "large" / "plain_s0.pt", R.clone_net)
    c = Ctx(net, orig, cfg, th, x, fit, read, push, VR.xor2_target(orig, x, push))
    (cfg.results / "large").mkdir(parents=True, exist_ok=True)
    log(f"{EXPERIMENT}: {x.keys.numel()} keys, {int(x.held.sum())} held-out pairs, alpha {e148['calibration']['alpha']}, "
        f"|push| {float(push.norm()):.3f}")
    stage0(c)
    stage1(c)
    stage2(c)
    return 0


if __name__ == "__main__":
    require_filed(PRED)
    code = 1
    try:
        code = main(Net("cuda"))
        log(f"{EXPERIMENT} DONE" if code == 0 else f"{EXPERIMENT} exit {code}")
    except R.MoreToDo:
        log("more to do: wall limit reached; what was read is saved (resubmit the same command)")
        code = 3
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
