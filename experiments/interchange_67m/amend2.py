"""E66 Amendment 2: the patch on the saved fine-tunes, read as exploratory, after job 48971 stopped at stage 1. The stop stands; no
label and no verdict on any state. Plan: reports/plans/2026-11-02_plan_e66-interchange-67m.md, "Amendment 2". Only what can stop or
void the run: results/predictions_amend2.json (committed first; the run refuses otherwise). Calls run.py and `commute.ctxpatch`
unchanged; results/run.json and results/predictions.json are read, never written. Neither the neuron reading nor any control runs.

  0 W0 alone, as run.py's stage 0 without K: pairs, transplant check, R(W0), l*, and that these reproduce run.json. Exit 1 on a miss.
  1 the write and sharpening readings copied from run.json, for reference. Nothing is rebuilt.
  2 E62's six states, 3 the eight copy and random maps: G re-read (exit 1 on a miss), steering, T, T_off, Q at blocks 1-3.
  4 E62 Amendment 2's size-swapped copies: the same three at blocks 1-3.
results/amend2.json after every state; every number printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import run as R
import torch

from commute import anchors as AN
from commute import ctxpatch as X
from commute import elicit as EL
from commute import keydep as KD
from commute import steering as ST
from commute.ctxpatch import Map
from commute.data import pile_rows
from commute.open_pair import jsonable
from commute.steering import DW

RESULTS = Path(__file__).resolve().parent / "results"
PRED, BASE, FIRST, OUT = RESULTS / "predictions_amend2.json", RESULTS / "predictions.json", RESULTS / "run.json", RESULTS / "amend2.json"
Rec = dict[str, Any]
log = KD.log


def note(out: Rec, key: str, rec: Rec) -> None:
    out[key] = rec
    OUT.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
    log(f"[{key}] {json.dumps(jsonable(rec))}")


def stop(out: Rec, why: str) -> NoReturn:
    note(out, "stopped", {"why": why})
    sys.exit(f"STOP: {why}: tell the lead")


def stage0(net: ST.Net, maps: dict[str, Map], first: Rec, S: Rec, th: Rec, out: Rec) -> None:
    half = S["half"]
    sd = lambda r: float(r[X.boot_index(len(r), th["n_boot"], S["seed"])].mean(1).std())
    rec: Rec = {"maps": {}}
    for m in maps.values():
        old = first["maps"][m.name]
        diffs = [abs(float(a.R0[l].mean()) - old[k][str(l)][0]) for a, k in ((m.on, "R0"), (m.off, "R0_off")) for l in X.BLOCKS]
        rec["maps"][m.name] = {"lstar": m.lstar, "lstar_first": old["lstar"], "pairs": len(m.on.pairs.tgt), "pairs_off": len(m.off.pairs.tgt),
                               "R0": {l: [float(m.on.R0[l].mean()), sd(m.on.R0[l])] for l in X.BLOCKS}, "R0_off": {l: [float(m.off.R0[l].mean()), sd(m.off.R0[l])] for l in X.BLOCKS},
                               "transplant_max": X.nan_max([X.transplant_max(net, m.on, half), X.transplant_max(net, m.off, half)]), "R0_diff_first": X.nan_max(diffs)}
    r = rec["maps"]
    rec["status"] = X.stage0_status(X.nan_max([v["transplant_max"] for v in r.values()]), {n: v["lstar"] for n, v in r.items()}, th)
    worst = X.nan_max([v["R0_diff_first"] for v in r.values()])
    rec["reproduces_first"] = worst <= th["r0_tol"] and all(v["lstar"] == v["lstar_first"] for v in r.values())
    note(out, "stage0", rec)
    if rec["status"] != "pass":
        stop(out, f"stage 0: {rec['status']}")
    if not rec["reproduces_first"]:
        stop(out, f"stage 0: the pairs do not reproduce job 48971's, so its write readings are no reference (worst {worst})")


def references(first: Rec) -> Rec:
    """Stage 1's readings copied from run.json, per map: the write (the lookup's for the copy map) and sharpening. Pure. ident32b's
    block 1 was void there, 0.003 nats under the floor; its T from the two means is given beside, with no sd."""
    maps = first["stage1"]["maps"]
    keep = lambda c: {"row": c["row"], "info": c["info"]}
    out: Rec = {n: {"write_of": "lookup" if n == "copy" else n, "write": keep(maps["lookup" if n == "copy" else n]["controls"]["write"]), "sharpen": keep(v["controls"]["sharpen"])} for n, v in maps.items()}
    out["ident32b"]["write_block1_T_under_floor"] = maps["ident32b"]["controls"]["write"]["row"]["1"]["R"] / first["stage0"]["maps"]["ident32b"]["R0"]["1"][0]
    return out


def row_of(net: ST.Net, m: Map, dW: DW, A: Rec, th: Rec, seed: int) -> dict[int, dict[str, Any]]:
    """T, T_off, Q per block as run.py reads them. Only `read_floor`'s map takes a lower floor, and every block is marked against the filed one."""
    row: dict[int, dict[str, Any]] = X.patch_row(net, m, dW, {**th, "floor": A["read_floor"].get(m.name, th["floor"])}, seed)
    for l in X.BLOCKS:
        row[l]["under_floor"] = float(m.on.R0[l].mean()) < th["floor"]
    return row


def read_map(net: ST.Net, m: Map, run: Rec, A: Rec, S: Rec, th: Rec, out: Rec, key: str) -> None:
    half, filed = S["half"], R.filed_steering()
    e52 = R.filed_json("baseline_neurons", "baseline_neurons.json")
    for s in R.SEEDS[m.name]:
        name = f"{m.name}_s{s}"
        G, diff = R.reproduce(net, name, m, run, e52, half)
        if diff > th["g_tol"]:
            stop(out, f"{key}: {name} does not reproduce its filed gain: {G} ({diff} off)")
        dW, shares = R.state(name), filed.get(name)
        if shares is None:
            v = R.e62_read()
            shares = X.steer_shares(net, v["read"].tokens, v["target"][m.name]["read"], half, dW)
        rec = {"exploratory": True, "G": G, "G_diff": diff, "norm": EL.summed_norm(dW), "lstar": m.lstar, "steering": shares, "steering_measured": name not in filed,
               "mark_by_block": {l: X.blind_mark(shares, l, th) for l in X.BLOCKS}, "row": row_of(net, m, dW, A, th, S["seed"])}
        out.setdefault(key, {})[name] = rec
        OUT.write_text(json.dumps(jsonable(out), indent=1, sort_keys=True))
        log(f"[{key} {name}] {json.dumps(jsonable(rec))}")


def stage4(net: ST.Net, maps: dict[str, Map], A: Rec, S: Rec, th: Rec, out: Rec) -> None:
    """Each E62 state times its filed alpha, the other map's size at its seed. T against size is computed in the analysis."""
    rec: Rec = {"exploratory": True, "scaled": {}}
    for name, st in R.filed_json(R.E62, "amend2.json")["row7_size"]["states"].items():
        dW = {k: st["alpha"] * v for k, v in R.state(name).items()}
        rec["scaled"][name] = {"alpha": st["alpha"], "norm": EL.summed_norm(dW), "row": row_of(net, maps[name.split("_")[0]], dW, A, th, S["seed"])}
    note(out, "stage4", rec)


def load() -> tuple[Rec, Rec, Rec, Rec, Rec]:
    git = lambda *x: subprocess.run(["git", "-C", str(R.REPO), *x], capture_output=True, text=True, check=False)
    for p in (PRED, BASE, FIRST):
        rel = str(p.relative_to(R.REPO))
        if not p.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
            sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    if OUT.exists():
        sys.exit(f"{OUT} exists: move it aside, this run deletes nothing")
    A, first = json.loads(PRED.read_text()), json.loads(FIRST.read_text())
    head = {"git_head": git("rev-parse", "HEAD").stdout.strip(), "predictions_sha256": KD.sha256(PRED), "first_run_sha256": KD.sha256(FIRST)}
    return A, json.loads(BASE.read_text()), first, R.filed_json(R.E62, "run.json"), head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    A, base, first, run, out = load()
    S, th = base["settings"], {**base["thresholds"], **A["thresholds"]}
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net(a.device)
    rows = pile_rows("train", 0, AN.ROWS, log)
    held = torch.arange(len(rows)) % S["heldout_every"] == 0
    maps = {m.name: m for m in [*R.cue_maps(net, rows[held], run, S, th), *(R.key_map(net, rows[held], n, S, th) for n in list(R.SEEDS)[2:])]}
    stage0(net, maps, first["stage0"], S, th, out)
    note(out, "references", references(first))
    for key, names in (("stage2", ("copy", "lookup")), ("stage3", list(R.SEEDS)[2:])):
        for n in names:
            read_map(net, maps[n], run, A, S, th, out, key)
    stage4(net, maps, A, S, th, out)


if __name__ == "__main__":
    main()
