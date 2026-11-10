"""Old-task labels (cell P1.7): each of the 17 old 67M tasks, two saved fine-tunes each (rand1 has one), read with E137's blind
search exactly as E139 reads a fine-tune, and labelled by E139's filed rule with E137's cut-offs. Labels are provisional: no
wording is adopted. Plan: reports/plans/2026-11-10_plan_old-task-labels.md. Rules and predictions: results/predictions.json
(the run refuses unless it is committed unchanged). F is M plus the task's saved delta-W (`commute.old_tasks.old_f`); D is the
task's 8-token windows (`old_tasks.task_windows`); the searches are E139's own `search_g` and `search_h`, loaded from its run.py.

A job runs the control or a list of items, so several jobs run side by side, each writing its own files:
    run.py control                       stage 0, alone, first (results/control.json; exit 1 if it fails)
    run.py copy24:0 copy24:1 rank1:0     items task:seed (seeds 0, 1; rand1 0 only); refuses without a passed control
    run.py all                           every item not finished

Stage 0, alone: (a) the do-nothing state, a delta of zeros through the old tasks' loading path, on every task's D: its gap must
be at the float floor and under the 0.02 floor (no reading); (b) the loading path against `Net.at`, the path older experiments
read these files with, on one fp32 and one bf16 file: the same outputs to the float floor; (c) the positive control, a finished
E139 fine-tune with a reading, expressed as a delta and loaded by the same `with_delta`: its gap, best G and best H repeat E139's
within E139's filed tolerances. Any miss stops the job (exit 1).
An item loads F, reads its gap on D's read half (under the floor: no reading, done), then the G search at k = 4 (4 sites, 3
starts) and H (5 sites), writing results/<task>_s<seed>.json after each search. Exit 0 done, 1 a stop or refusal, 3 more to do.

    sbatch cluster/old_task_labels.sbatch control
    sbatch cluster/old_task_labels.sbatch copy24:0 copy24:1 rank1:0
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import commute.old_tasks as OT
import commute.variable_read as VR
from commute.hidden_maps import chunks, run, tv
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed
from commute.keydep import sha256
from commute.steering import Net

EXPERIMENT = "E###"  # the lead fills this in when the cell is claimed (`fleet/ecode`); only the log line reads it
HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
E139 = load_experiment("reuse_variable_finetune", "run")  # its search_g, search_h, dump, log, budget and MoreToDo, unchanged
E139_ITEMS = ("always_on_s0", "always_on_s1", "always_on_s2", "cue_s0", "cue_s1", "cue_s2", "table_s0", "table_s1", "table_s2")
log, dump = E139.log, E139.dump


@dataclass(frozen=True)
class Config(OT.DSizes):
    results: Path = HERE / "results"
    wall_limit_s: int = 6300  # E139's: exit 3 before the 1:55 limit, leaving one H site's time


def stop(path: Path, out: dict[str, Any], msg: str) -> None:
    out["stopped"] = msg
    dump(path, out)
    log(f"STOPPED: {msg}")
    raise SystemExit(1)


def gap(net: Net, ft: Net, windows: Tensor, cfg: Config) -> tuple[float, float]:
    """(mean, max) over `windows` of TV(F, M) at the last position: E139's `distances` gap."""
    d = torch.cat([tv(run(ft, t)[0][:, -1], run(net, t)[0][:, -1]) for t in chunks(windows, cfg.read_batch)])
    return float(d.mean()), float(d.max())


# ---- stage 0, alone -------------------------------------------------------------------------------------------------


def do_nothing(net: Net, cfg: Config, th: dict[str, Any], sets: VR.Sets, path: Path, out: dict[str, Any]) -> None:
    """(a): a zero delta through `with_delta`, on the read half of every group's D. Also builds every D once."""
    ft0 = OT.with_delta(net, {f"{k}.weight": torch.zeros_like(w) for k, w in net.W0.items()})
    for task in {t.group: t for t in OT.TASKS.values()}.values():
        _fit, read, sizes = OT.task_windows(task, sets, cfg)
        if "floor_runs_tv" not in out:
            out["floor_runs_tv"] = E139._floor(net, read[:8192], cfg)
        mean, top = gap(net, ft0, read, cfg)
        out["do_nothing"][task.group] = {"gap": mean, "max_tv": top, "reading": mean >= th["floor_gap"], **sizes}
        log(f"do-nothing {task.group}: gap {mean:.3e} (max TV {top:.3e}), D {sizes}")
        if top > out["floor_runs_tv"] or mean >= th["floor_gap"]:
            stop(path, out, f"{task.group}: the zero delta is off the float floor or reads at or above the gap floor")


def loader_check(net: Net, cfg: Config, sets: VR.Sets, path: Path, out: dict[str, Any]) -> None:
    """(b): `old_f` against `Net.at` (W0 + delta-W written into M, E97's and E116's path), one fp32 and one bf16 file."""
    _fit, read, _ = OT.task_windows(OT.TASKS["case_real"], sets, cfg)
    for name in ("e62_copy", "case_real"):
        state = OT.TASKS[name].states[0]
        ft, toks = OT.old_f(net, state), read[:8192]
        dw = {k: v.float() for k, v in torch.load(state, weights_only=True, map_location="cpu").items()}
        with net.at(dw):
            at = torch.cat([run(net, t)[0][:, -1] for t in chunks(toks, cfg.read_batch)])
        mine = torch.cat([run(ft, t)[0][:, -1] for t in chunks(toks, cfg.read_batch)])
        out["loader"][name] = float(tv(mine, at).max())
        log(f"loader {name}: max TV against Net.at {out['loader'][name]:.3e}")
        if out["loader"][name] > out["floor_runs_tv"]:
            stop(path, out, f"{name}: the loading path differs from Net.at beyond the float floor")


def _e139_item() -> tuple[str, dict[str, Any]]:
    for item in E139_ITEMS:
        p = E139.HERE / "results" / f"{item}.json"
        rec = json.loads(p.read_text()) if p.is_file() else {}
        if rec.get("complete") and not rec.get("no_reading") and rec.get("G") and rec.get("H"):
            return item, rec
    raise FileNotFoundError("no finished E139 fine-tune with a reading yet: the positive control needs one (wait for E139)")


def replay(net: Net, cfg: Config, th: dict[str, Any], sets: VR.Sets, tok: Any, path: Path, out: dict[str, Any]) -> None:
    """(c): a finished E139 fine-tune as a delta through `with_delta`; its best G key and best H site re-read with E139's seeds."""
    item, rec = _e139_item()
    vocab = int(net.net.lm_head.weight.shape[0])  # type: ignore[union-attr]
    fit, read = VR.pair_windows(cfg, rec["pair"], sets, VR.make_task(sets, tok, vocab, cfg))
    S = torch.load(E139.HERE / "results" / "large" / f"{item}.pt", map_location=net.device)
    ft = OT.with_delta(net, {k: S[k].float() - p.detach() for k, p in net.net.named_parameters()})
    F, (g, _) = VR.net_target(ft), gap(net, ft, read, cfg)
    key = max(rec["G"], key=lambda k: rec["G"][k]["G"])
    _, site, start = key.split("/")
    found = VR.g_search(net, fit, site, cfg.k, int(start[1:]), F, cfg, 200_000 + zlib.crc32(f"{item}/{key}".encode()) % 65_536)
    A, c = found["A"], found["c"]
    G = 1 - VR.read_pair(net, read, site, found["Q"], lambda v, _s, A=A, c=c: v @ A + c, F, cfg)["d"] / g
    hs = max(rec["H"], key=lambda s: rec["H"][s]["H"])
    ridge, X_val, Y_val, ps = VR.h_fit(net, fit, F, cfg, log)
    beta, _lam, _mse = VR.ridge_select(ridge, hs, ps[hs], X_val, Y_val[hs], cfg.ridge_grid)
    n_val = fit.shape[0] * 4 // 5
    b, y, _ = VR.h_refine(net, fit[:n_val], fit[n_val:], hs, ridge, beta, F, cfg, 137_800 + VR.H_SITES.index(hs))
    H = 1 - VR.read_pair(net, read, hs, None, VR.ridge_s(ridge, hs, b, y), F, cfg)["d"] / g
    out["replay"] = {"item": item, "gap": g, "gap_e139": rec["dist"]["gap"], "G_key": key, "G": G, "G_e139": rec["G"][key]["G"],
                     "H_site": hs, "H": H, "H_e139": rec["H"][hs]["H"]}  # fmt: skip
    log(f"replay: {out['replay']}")
    r = out["replay"]
    if abs(r["gap"] - r["gap_e139"]) > th["gap_replay_tol"] or abs(G - r["G_e139"]) > th["replay_tol"] or abs(H - r["H_e139"]) > th["replay_tol"]:
        stop(path, out, "the loading path does not repeat E139's reading of its own fine-tune")


def control(net: Net, cfg: Config, th: dict[str, Any], sets: VR.Sets, tok: Any) -> None:
    path = cfg.results / "control.json"
    if path.is_file() and json.loads(path.read_text()).get("passed"):
        log("control already passed")
        return
    out: dict[str, Any] = {"do_nothing": {}, "loader": {}, "passed": False}
    do_nothing(net, cfg, th, sets, path, out)
    dump(path, out)
    loader_check(net, cfg, sets, path, out)
    dump(path, out)
    replay(net, cfg, th, sets, tok, path, out)
    out["passed"] = True
    dump(path, out)
    print("STAGE 0 DONE", flush=True)


# ---- one item -------------------------------------------------------------------------------------------------------


def do_item(net: Net, cfg: Config, th: dict[str, Any], sets: VR.Sets, name: str, seed: int) -> None:
    task, item = OT.TASKS[name], f"{name}_s{seed}"
    path, state = cfg.results / f"{item}.json", OT.TASKS[name].states[seed]
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"item": item, "task": name, "seed": seed, "G": {}, "H": {}, "complete": False}
    if out.get("complete"):
        log(f"{item} already complete")
        return
    log(f"item {item}")
    fit, read, sizes = OT.task_windows(task, sets, cfg)
    ft = OT.old_f(net, state)
    if "dist" not in out:
        mean, top = gap(net, ft, read, cfg)
        out.update(state=str(state.relative_to(EXPERIMENTS.parent)), sha256=sha256(state), D=sizes, dist={"gap": mean, "max_tv": top})
        dump(path, out)
        log(f"{item}: gap {mean:.5f} (max TV {top:.4f}), D {sizes}")
    if out["dist"]["gap"] < th["floor_gap"]:
        out.update(no_reading="gap under the floor", complete=True)
        dump(path, out)
        log(f"{item}: no reading (gap under the floor)")
        return
    F, g = VR.net_target(ft), out["dist"]["gap"]
    E139.search_g(net, cfg, item, fit, read, F, g, path, out)
    E139.search_h(net, cfg, item, fit, read, F, g, path, out)
    out["complete"] = True
    dump(path, out)
    log(f"ITEM {item} DONE")


def parse(args: list[str]) -> list[tuple[str, int]] | None:
    """`control` -> None. Otherwise the (task, seed) items of `all` or the listed task:seed."""
    if args == ["control"]:
        return None
    every = [(t.name, s) for t in OT.TASKS.values() for s in t.states]
    if args == ["all"]:
        return every
    out = []
    for a in args:
        name, _, s = a.partition(":")
        if not s.isdigit() or (name, int(s)) not in every:
            sys.exit(f"bad item {a!r}: task:seed with task in {sorted(OT.TASKS)} and a saved seed, or `control`, or `all`")
        out.append((name, int(s)))
    return out


def main(items: list[tuple[str, int]] | None, cfg: Config, net: Net) -> int:
    th = json.loads(PRED.read_text())["thresholds"]
    net.net.requires_grad_(False)
    from commute import vpd

    tok = vpd.tokenizer()
    sets = VR.make_sets(cfg, tok, log)
    cfg.results.mkdir(parents=True, exist_ok=True)
    log(f"{EXPERIMENT}: rows ready")
    if items is None:
        control(net, cfg, th, sets, tok)
        return 0
    path = cfg.results / "control.json"
    if not (path.is_file() and json.loads(path.read_text()).get("passed")):
        log("no passed control: run `control` first and alone")
        return 1
    try:
        for name, seed in items:
            do_item(net, cfg, th, sets, name, seed)
            torch.cuda.empty_cache()
    except E139.MoreToDo:
        log("more to do: wall limit reached, saved; rerun to resume")
        return 3
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+")
    chosen = parse(ap.parse_args().what)
    require_filed(PRED)
    code = 1
    try:
        code = main(chosen, Config(), Net("cuda"))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
