"""E100: is each 67M fine-tune's gain fed through unchanged parts or written into changed ones? No training. Plan:
reports/plans/2026-11-05_plan_e100-fed-or-written.md. Rules and expected values: results/predictions.json (committed first;
the run refuses otherwise; never edited) and `commute.fed_rules`. Reads: `commute.locator.split`, on E97's sites and
states (experiments/locator_table/{job,sites}.py, sha256-checked).

  stage 0, alone: E97's two one-matrix plants (blocks 3 and 1) on the 24-key copy table, exact by construction; the
           block-1 plant's largest inherited share downstream is the floor. Exit 1 on any miss.
  stage 1, the poles: E99's forced and forbidden copy states (definition-lead's, read only, if saved), E62's copy and
           lookup, E53's cipher at step 300 on whole rows. The bars are written to stage1.json before stage 2 reads
           them; exit 1 unless the copy bar is below the cipher bar.
  stage 2: every other saved state: lowercase on whole rows, the 24- and 32-key tables; the constant-answer tasks beside
           only, with their same-windows null.
  table:   results/table.json, one row per (task, seed), from the stage files alone.

Every state reads 28 parts (24 heads, 4 MLPs) and the filed pairs, three reads each; its stage file is written as each
state finishes and a resubmit skips the states already there. "STAGE <n> DONE" is printed when a stage file is complete.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute import fed_rules as FR
from commute import locator as L
from commute import steering as ST
from commute.jobs import hard_exit, load_experiment
from commute.keydep import log, sha256

JB = load_experiment("locator_table", "job")
sites = JB.sites
HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, TABLE = RESULTS / "predictions.json", RESULTS / "table.json"
STAGE = {k: RESULTS / f"stage{k}.json" for k in range(3)}
E97 = sites.EXP / "locator_table" / "results"
E99 = sites.EXP / "definition_verify" / "results"
POLES = {"E99 forced copy": ("forced", "stage1.json"), "E99 forbidden copy": ("forbidden", "stage2.json")}
STAGES = {1: ["E99 forced copy", "E99 forbidden copy", "T5 copy", "T5 lookup", "T8 cipher"],
          2: ["T7 lowercase", "T12 rank-1", "T11 elicit", "T13 copy-24", "T11 cipher", "T4 ident32", "T6 rand32",
              "T1 emoticon", "T1 emoticon null", "T2 r1_t534_v369", "T2 r1_t534_v369 null", "T2 r2_t359_v816", "T2 r2_t359_v816 null"]}  # fmt: skip
dump, say, load, stop = JB.dump, JB.say, JB.load, JB.stop


# ----------------------------------------------------------------------------- states


def e99_states(J: Any, task: str) -> dict[int, tuple[str, Path]]:
    """E99's saved copy states of one mode, with the sha256 its stage file recorded; {} when not saved."""
    mode, stage = POLES[task]
    rec = load(E99 / stage).get(f"{mode} copy", {})
    large = Path(J.P["settings"]["e99_large"]).expanduser()
    return {int(s[1:]): (r["sha256"], large / r["file"]) for s, r in rec.items() if r.get("file") and (large / r["file"]).is_file()}


def e99_dW(sha: str, path: Path) -> ST.DW:
    if sha256(path) != sha:
        raise ValueError(f"{path}: sha256 differs from E99's record {sha}")
    return {k: v.float() for k, v in torch.load(path, weights_only=True, map_location="cpu").items()}


def states_of(J: Any, task: str) -> dict[int, tuple[str, Any]]:
    """seed -> (site key, a loader of dW). A null task's two seeds share one file: read once, as seed 0."""
    if task in POLES:
        return {s: ("e62_copy", lambda x=x: e99_dW(*x)) for s, x in e99_states(J, task).items()}
    T = sites.TASKS[task]
    seeds = {0: T.states[0]} if not T.in_table else T.states
    return {s: (T.site, lambda p=p: J.dW(p)) for s, p in seeds.items() if p.is_file()}


def parts_for(J: Any, task: str) -> list[str]:
    return [L.name(p) for p in L.PARTS] + J.P["settings"]["extra_parts"].get(task, [])


def read_state(J: Any, task: str, key: str, dW: ST.DW) -> dict[str, Any]:
    t0 = time.monotonic()
    st = L.state(J.net, J.base(key), dW)
    rec: dict[str, Any] = {"G": L.units(st.G), "parts": {n: L.split(J.net, st, n) for n in parts_for(J, task)}}
    rec["G_all"] = L.units((L.read(J.net, st.b.site, L.reverted(st.dW, list(st.dW))) - st.b.lo0).mean(0))
    rec["seconds"] = time.monotonic() - t0
    return rec


def read_stage(J: Any, k: int, out: dict[str, Any]) -> dict[str, Any]:
    """Every state of the stage's tasks, written as each finishes; a state already in the file is not read again. All 24
    reverted must read G 0 on every state, or the job stops. A task with no saved state is skipped, logged."""
    out.setdefault("states", {})
    out.setdefault("skipped", {})
    for task in STAGES[k]:
        got = states_of(J, task)
        if not got:
            out["skipped"][task] = "no saved state found"
            log(f"skip {task}: no saved state found")
        for s, (key, dW) in sorted(got.items()):
            name = f"{task} s{s}"
            if name in out["states"]:
                continue
            rec = read_state(J, task, key, dW()) | {"task": task, "seed": s, "site": key}
            out["states"][name] = rec
            dump(STAGE[k], out)
            say(f"state {name}", rec)
            if not (abs(rec["G_all"]["lo"]) <= J.th["zero_tol"]):
                stop(STAGE[k], out, f"{name}: all 24 reverted reads G {rec['G_all']['lo']}")
    return out


def lo(rec: dict[str, Any]) -> FR.Split:
    return {n: {k: v[k]["lo"] for k in ("carry", "own", "graft")} for n, v in rec["parts"].items()}


def at_of(J: Any, rec: dict[str, Any]) -> str | None:
    return FR.read_at(lo(rec), J.P["settings"]["read_at"].get(rec["task"]))


def seeds_of(out: dict[str, Any], task: str) -> list[dict[str, Any]]:
    return [r for r in out["states"].values() if r["task"] == task]


# ----------------------------------------------------------------------------- stages


def stage0(J: Any) -> None:
    """The two plants, as E97's stage 0 (`locator.plant`), scaled to copy-24 seed 0's G."""
    th, rec = J.th, {"plants": {}, "files": J.F.seen}
    b, (_, w) = J.base("kv_copy24"), sites.kv(J.F, sites.COPY24)
    Gt = float(J.state("T13 copy-24", 0).G[0])
    for block in J.P["settings"]["plant_blocks"]:
        dW, fit = L.plant(J.net, b.site, w["train"].tokens, w["train"].half, w["table"], block, Gt, th["plant_rel_tol"])
        r = read_state(J, "plant", "kv_copy24", dW)
        sp = lo(r)
        r |= {"fit": fit, "misses": FR.plant_misses(sp, block, r["G_all"]["lo"], th), "floor": FR.floor_of(sp, block)}
        rec["plants"][f"block{block}"] = r
        say(f"stage0 plant block {block}", r)
        dump(STAGE[0], rec)
    floor = rec["floor"] = rec["plants"]["block1"]["floor"]
    ref = th["floor_ref"]
    if not abs(floor - ref) <= th["floor_tol"]:
        rec["plants"]["block1"]["misses"].append(f"floor {floor} off E97's {ref} by more than {th['floor_tol']}")
    misses = [f"{k}: {m}" for k, p in rec["plants"].items() for m in p["misses"]]
    rec |= {"status": "miss" if misses else "pass", "missed": misses}
    dump(STAGE[0], rec)
    if misses:
        stop(STAGE[0], rec, f"stage 0: {misses[:5]}")


def reread(out: dict[str, Any]) -> dict[str, Any]:
    """Carry and own against E97's saved carry and single-part write on the same states: the largest difference (report)."""
    e97 = {**load(E97 / "stage1.json").get("states", {}), **load(E97 / "stage2.json").get("states", {})}
    res = {}
    for name, r in out["states"].items():
        if r["task"] not in sites.TASKS or (k := JB.sid(r["task"], r["seed"])) not in e97:
            continue
        old = e97[k]
        own = {**old["write"]["head"], **old["write"]["mlp"]}
        d = [abs(r["parts"][n]["carry"]["lo"] - old["carry"][n]["lo"]) for n in old["carry"]] + [abs(r["parts"][n]["own"]["lo"] - own[n]["lo"]) for n in own]
        res[name] = max(d)
    return res


def stage1(J: Any) -> dict[str, Any]:
    out = read_stage(J, 1, load(STAGE[1]))
    out["poles_e99"] = {n: FR.pole_misses(lo(r), J.P["settings"]["read_at"][r["task"]], POLES[r["task"]][0], J.th)
                        for n, r in out["states"].items() if r["task"] in POLES}  # fmt: skip
    out["reread_e97_max_abs"] = reread(out)
    ratio = lambda r: FR.ratio(*(lo(r).get(at_of(J, r) or "", {}).get(k, math.nan) for k in ("carry", "own")))
    out["bars"] = FR.bars([ratio(r) for r in seeds_of(out, "T5 copy")], [ratio(r) for r in seeds_of(out, "T8 cipher")])
    out["bars"]["read_at"] = {n: at_of(J, r) for n, r in out["states"].items()}
    dump(STAGE[1], out)
    say("stage1 poles", {k: out[k] for k in ("poles_e99", "reread_e97_max_abs", "bars", "skipped")})
    if not out["bars"]["ok"]:
        stop(STAGE[1], out, f"stage 1: {out['bars']['why']}; the split does not separate the poles (the plan's tree: drop the column)")
    return out


# ----------------------------------------------------------------------------- the table


def write_table(J: Any) -> None:
    s0, s1, s2 = (load(STAGE[k]) for k in range(3))
    floor, b = s0.get("floor", math.nan), s1.get("bars", {})
    leans = J.P["expected"]["leans"]
    rows, tasks = [], {}
    for k, out in ((1, s1), (2, s2)):
        for task in STAGES[k]:
            recs = seeds_of(out, task) if out else []
            if not recs or task.endswith(" null"):
                continue
            T = sites.TASKS.get(task)
            got = [FR.row(task, r["seed"], lo(r), r["G"]["lo"], at_of(J, r), floor, b, bool(T and T.constant)) | {"stage": k} for r in recs]
            if T and T.constant:
                null = seeds_of(out, f"{task} null")
                for g in got:
                    g["G_null"] = null[0]["G"]["lo"] if null else math.nan
            rows += got
            word = FR.task_reading([str(g["reading_rule"]) for g in got])
            tasks[task] = {"reading": word, "lean": leans.get(task), "lean_met": FR.lean_met(word, leans.get(task))}
    rec = {"table": rows, "tasks": tasks, "floor": floor, "bars": b, "outcome": FR.outcome({t: v["reading"] for t, v in tasks.items()}),
           "limit": "a new circuit built from old parts still reads as reuse: 'fed' says the gain runs through a part whose own weights barely changed"}  # fmt: skip
    dump(TABLE, rec)
    for r in rows:
        say("table", r)
    say("tasks", tasks)
    say("outcome", rec["outcome"])


# ----------------------------------------------------------------------------- main


def filed() -> dict[str, Any]:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    return json.loads(PRED.read_text())


def timed(k: int, f: Any, *a: Any) -> None:
    t0 = time.monotonic()
    f(*a)
    log(f"[stage {k} done] {time.monotonic() - t0:.0f} s")
    print(f"STAGE {k} DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stages", default="0,1,2")
    args = ap.parse_args()
    want = {int(x) for x in args.stages.split(",")}
    P = filed()
    if load(STAGE[0]).get("status") == "miss":
        sys.exit(f"{STAGE[0]} holds a missed stage 0: move it aside; this run deletes nothing")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as E96, E97
    net = ST.Net(args.device)
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    log(f"[host] {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'} git {git} predictions {sha256(PRED)} stages {sorted(want)}")
    J = JB.Job(net, sites.Files(), P, {})
    if 0 in want and load(STAGE[0]).get("status") != "pass":
        timed(0, stage0, J)
    if load(STAGE[0]).get("status") != "pass":
        sys.exit("stage 0 has not passed: run it first")
    if 1 in want:
        J.clear()
        timed(1, stage1, J)
    if 2 in want:
        if not load(STAGE[1]).get("bars", {}).get("ok"):
            sys.exit("stage 1's bars are not ok or not read: stage 1 first")
        J.clear()
        timed(2, lambda: read_stage(J, 2, load(STAGE[2])))
    write_table(J)
    hard_exit(0)


if __name__ == "__main__":
    main()
