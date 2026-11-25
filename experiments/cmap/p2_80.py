"""E305, cell P2.80: E288's whole-group head CMAP read on the five seed-2 states of Fig 4 that E288 did not read
(E299's files in experiments/known_3seeds/results/large/), by E288's job.py and stage0.py with its settings, gates and
rules unchanged. Each state is read as its seed-0 sibling's item in E288's manifest (task, scale, pair side). Card
p2-80/card.md; states, sha256s and expected values p2-80/predictions.json (the run refuses unless it is committed
unchanged). E288's files are not touched: everything goes to results/p2-80/.

  0  the controls again, alone and first, by E288's functions and miss rules -> results/p2-80/stage0.json: every built
     task's rows equal E288's record, the hook floor per format, the do-nothing, the two-layer path and the planted head
     (value rows) through the whole pipeline. Taken from E288's stage0.json instead: the gates (the five face the bar
     their siblings faced) and each item's estimate (its sibling's). Not re-run: E237's random items, the output-weights
     plant, the timed item (none guards these items).
  1  the five items, each written as it finishes -> results/p2-80/items/; stops after 4 items if they run over the
     estimate by rerun_over. A rerun skips a passed stage 0 and finished items.

    sbatch --comment=E305 --export=ALL,CMAP_ENTRY=p2_80.py cluster/cmap.sbatch
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import torch
from job import (
    RESULTS,
    Item,
    Job,
    dump,
    gate_key,
    item_name,
    load_config,
    log,
    run_item,
    stop,
)
from stage0 import ALL, _sites_ok, hook_floor, plant, two_layer

from commute.jobs import EXPERIMENTS, hard_exit, require_filed, run_items
from commute.keydep import sha256

HERE = Path(__file__).resolve().parent
PRED, OUT = HERE / "p2-80" / "predictions.json", RESULTS / "p2-80"
STAGE0, ITEMS, E288 = OUT / "stage0.json", OUT / "items", RESULTS / "stage0.json"
FILED = json.loads(PRED.read_text())
META = {"experiment": FILED["experiment"]}


def seed2_items(job: Job) -> dict[str, Item]:
    """Each filed state as its sibling's item in E288's manifest (one item, the same as seed 1's), with E299's file,
    whose sha256 must be the filed one."""
    out = {}
    for st, f in FILED["states"].items():
        sibs = [it for it in job.items.values() if it.state in (f["sibling"], f["sibling"].replace("_s0", "_s1"))]
        assert len(sibs) == 2 and len({(it.task, it.c, it.roles) for it in sibs}) == 1, f"{st}: siblings {sibs}"
        got = sha256(EXPERIMENTS.parent / f["file"])
        assert got == f["sha256"], f"{st}: {f['file']} sha256 {got}, filed {f['sha256']}"
        s = sibs[0] if sibs[0].state == f["sibling"] else sibs[1]
        it = Item(item_name(st, s.c), st, s.task, f["file"], s.c, s.roles)
        out[it.name] = it
        log(f"[item {it.name}] task {it.task}, c {it.c:.4f}, roles {it.roles}, as {s.name}; sha256 ok")
    return out


def stage0(job: Job, e288: dict[str, Any], items: dict[str, Item]) -> None:
    """E288's stage0 controls that guard these items, in its order, with its miss rules."""
    cfg, miss = job.cfg, list[str]()
    out: dict[str, Any] = {"experiment": META["experiment"], "gates_from": str(E288.relative_to(EXPERIMENTS.parent)), "tasks": {}}
    save = lambda: dump(STAGE0, out | {"misses": miss})
    tasks = {it.task for it in items.values()} | {f"e58/{cfg.key_tasks[0]}"} | set(cfg.hook_floor_tasks.values())
    for name in sorted(tasks):
        out["tasks"][name] = rec = job.task(name).record
        miss += [] if rec == e288["tasks"][name] else [f"{name}: rows {rec} not E288's {e288['tasks'][name]}"]
    out["hook_floor"] = {fmt: hook_floor(job, task) for fmt, task in cfg.hook_floor_tasks.items()}
    miss += [f"hook floor {f}: {r}" for f, r in out["hook_floor"].items() if r["n_flipped"] or not r["sites"]]
    tol = {f: r["tol"] for f, r in out["hook_floor"].items()}
    log(f"[hook floor] {out['hook_floor']}")
    save()
    ctrl = next(it for it in job.items.values() if it.state == cfg.control_state and it.c == 1.0)
    copy, zero = ctrl.task, job.dW(dataclasses.replace(ctrl, c=0.0))  # E237's control state times 0
    floor = job.gates[gate_key(copy, 1.0)]
    dn = run_item(job, "do_nothing", copy, zero, floor, OUT / "stage0_do_nothing.json", lambda p: {"all": ALL}, large_ok=True, meta=META)
    out["do_nothing"] = {k: dn[k] for k in ("W0", "ft", "gain", "faith", "groups_pruned")}
    out["do_nothing"]["same_as_E288"] = {k: dn[k] == e288["do_nothing"][k] for k in ("W0", "ft", "gain", "groups_pruned")}
    miss += _sites_ok(dn, tol["kv"], "vs_W0", list(dn["cmap"]))
    save()
    out["two_layer"] = two_layer(job)
    miss += [f"two-layer: {k}" for k, ok in out["two_layer"]["checks"].items() if not ok]
    log(f"[two-layer] {out['two_layer']}")
    save()
    pv = plant(job, "h.3.attn.v_proj", floor, True)
    out["plant_v"] = {k: v for k, v in pv.items() if k != "dW"}
    if pv["step"] is None:
        miss.append("plant (value rows): no checkpoint clears the gate")
    else:
        hn, other = (3, pv["head"][1], 0), (*pv["other"], 0)
        extra = lambda p: {"A_minus_h": [n for n in p["A"] if n != hn], "other_head": [other], "h_before": [(3, hn[1], 1)]}
        rec = run_item(job, "plant_v", copy, pv["dW"], floor, OUT / "stage0_plant_v.json", extra, large_ok=True, meta=META)
        out["plant_v"] |= {"groups_pruned": rec["groups_pruned"], "faith": rec["faith"], "cmap_A": rec["cmap"]["A"]["share"]}
        out["plant_v"]["same_as_E288"] = {k: out["plant_v"][k] == e288["plant_v"][k] for k in ("head", "step", "groups_pruned", "cmap_A")}
        miss += [] if hn in rec["groups_pruned"]["A"] else [f"plant (value rows): {hn} not in A after pruning: {rec['groups_pruned']}"]
        miss += _sites_ok(rec, tol["kv"], "vs_ft", ["A"]) + _sites_ok(rec, tol["kv"], "vs_W0", ["A_minus_h", "other_head", "h_before"])
    log(f"[vs E288] do-nothing {out['do_nothing']['same_as_E288']}; plant {out['plant_v'].get('same_as_E288')}")
    log(f"stage 0: {'controls hold' if not miss else 'MISSES ' + '; '.join(miss)}")
    if miss:
        stop(STAGE0, out | {"misses": miss}, "stage 0: " + "; ".join(miss))
    dump(STAGE0, out | {"misses": [], "done": True})
    log("STAGE 0 DONE")


def stage1(job: Job, e288: dict[str, Any], items: dict[str, Item], large_ok: bool) -> int:
    """E288's stage 1 on the five items: each at its task and scale's gate, its sibling's estimate for the overrun rule."""
    est = {it.name: e288["estimate"]["items"][item_name(FILED["states"][it.state]["sibling"], it.c)] for it in items.values()}
    ran = {"n": 0, "seconds": 0.0, "estimate": 0.0}

    def one(name: str) -> None:
        it, path = items[name], ITEMS / f"{name.replace('/', '-')}.json"
        if path.is_file() and not {"waits", "stopped"} & set(json.loads(path.read_text())):
            log(f"{name}: done before, skipped")
            return
        t0 = time.time()
        meta = META | {"state": it.state, "c": it.c, "roles": list(it.roles), "file": it.file, "sibling": FILED["states"][it.state]["sibling"]}
        rec = run_item(job, it.name, it.task, job.dW(it), job.gates[gate_key(it.task, it.c)], path, large_ok=large_ok, meta=meta)
        if "waits" in rec:
            return
        ran["n"] += 1
        ran["seconds"] += time.time() - t0
        ran["estimate"] += est[name]
        log(f"ITEM {name} DONE in {time.time() - t0:.0f} s (estimate {est[name]:.0f} s)")

    stopped = []
    for name in items:
        stopped += [name] if run_items([name], one, log) else []  # an item's stop ends that item only
        if ran["n"] == 4 and ran["seconds"] > job.cfg.rerun_over * ran["estimate"]:
            stop(ITEMS / "overrun.json", ran, f"the first 4 items took {ran['seconds']:.0f} s against {ran['estimate']:.0f} s estimated")
    log(f"STAGE 1 DONE ({ran}; stopped items {stopped})")
    return 1 if stopped else 0


def main(authors: Path, large_ok: bool) -> int:
    cfg = load_config()
    assert FILED["settings"] == json.loads((RESULTS / "predictions.json").read_text())["settings"], "settings differ from E288's"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # fp32 throughout, as E288
    log(f"P2.80 CMAP ({META['experiment']}) start, authors' checkout {authors}; E288's settings")
    job, e288 = Job(cfg, authors), json.loads(E288.read_text())
    assert e288.get("done"), "E288's stage 0 did not pass"
    job.gates = e288["gates"]
    items = seed2_items(job)
    if not (STAGE0.is_file() and json.loads(STAGE0.read_text()).get("done")):
        stage0(job, e288, items)
    code = stage1(job, e288, items, large_ok)
    log(f"DONE; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", type=Path, required=True, help="the read-only checkout of Nix07/finetuning at 6d6c098")
    ap.add_argument("--large-pools", action="store_true", help="run items whose pool exceeds max_pool (after pricing)")
    args = ap.parse_args()
    require_filed(PRED)
    code = 1
    try:
        code = main(args.authors, args.large_pools)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    hard_exit(code)
