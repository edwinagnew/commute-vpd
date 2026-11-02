"""regating_truth_table: the re-gating side of the step-2 table of reports/plans/2026-10-30_plan_regating-ground-truth.md
(stand-in ratio, r and steering share S for the 18 saved 67M runs), $0, saved states only, no labels. Rules, tolerances
and expected values: results/predictions.json (filed before the run, read here, never edited).

Only one column of the plan's table is empty on disk: S on E53's six step-300 states (lowercase, cipher, broad cipher;
seeds 0, 1). Their target is each window's own next token (`data.NEXT`), so steering's Task is read with that target
(`NextTask`). Every other cell comes from the file predictions.json 'table_sources' names (`table`).

  run.py   stage 0, alone: sha256 of every state and windows file (before the model loads), the zero update on each task,
           E53 lowercase_s0's stand-in ratio, G and r = g / G, E54 elicit_s0's and E46 ident32_s0's r and S, and the
           NEXT path's log-odds gain against the split read's G -> results/stage0.json; a miss logs FAIL and exits 1.
           Stage 1: S on the six E53 states, each written as it finishes -> results/stage1.json.
           Table: the 18 runs' ratio, r and S, each with its source -> results/table.json.
Every number is printed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor

from commute import keydep as KD
from commute import steering as ST
from commute.data import NEXT, target_tokens

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
EXP = REPO / "experiments"
PRED = RESULTS / "predictions.json"
E46, E52, E53, E54 = (EXP / n / "results" for n in ("regating_table", "baseline_neurons", "performed_anchor", "keydependent_elicit"))
ALIVE = EXP / "transport_test" / "results" / "large" / "alive.pt"
E46_TASKS, E53_TASKS, E54_TASKS = ("ident32", "ident32b", "rand32", "rand32b"), ("lowercase", "cipher", "broad_cipher"), ("elicit", "cipher")
SEEDS, DEVICE = (0, 1), "cuda"
log = KD.log


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def clean(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    return [clean(v) for v in o] if isinstance(o, list | tuple) else o


def dump(name: str, obj: dict[str, Any]) -> None:
    (RESULTS / name).write_text(json.dumps(clean(obj), indent=1, sort_keys=True))


def git_head() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()


def performed_anchor() -> ModuleType:
    """E53's run.py, loaded by path (`Run`, `Task`, `split_row`, `e53_pi`, `a1_maps`): called, not copied."""
    spec = importlib.util.spec_from_file_location("performed_anchor_run", EXP / "performed_anchor" / "run.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses looks the defining module up in sys.modules
    spec.loader.exec_module(mod)
    return mod


class NextTask(ST.Task):
    """steering's Task whose target is each window's own next token (E53's Pile windows), not a key -> value table."""

    def tgt(self, toks: Tensor) -> Tensor:
        return target_tokens(NEXT, toks, self.half)


def next_task(net: ST.Net, name: str, held: Tensor, half: int) -> NextTask:
    t = NextTask(name, "pile", held, torch.empty(0), (), -1, half, torch.empty(0))
    t.lp0 = ST.logp_at(net.run(t.toks)[0], t.half, t.tgt(t.toks))
    return t


class Out:
    """One results file: each record written and logged as it lands; a miss (ok False) writes, logs FAIL and exits 1."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.out: dict[str, Any] = {"commit": git_head(), "rows": {}}

    def put(self, key: str, rec: dict[str, Any], ok: bool | None = None) -> dict[str, Any]:
        self.out["rows"][key] = {**rec, **({} if ok is None else {"ok": bool(ok)})}
        dump(self.name, self.out)
        log(f"[{self.name} {key}] ok {ok} {json.dumps(clean(rec), sort_keys=True)}")
        if ok is False:
            self.out["stopped"] = key
            dump(self.name, self.out)
            sys.exit(f"FAIL: {key}: stop")
        return rec


# ----------------------------------------------------------------------------- files


def files() -> dict[str, tuple[Path, str]]:
    """The 18 runs' state files and the two windows files read, each with the sha256 its source run recorded."""
    s0a = load(E46 / "stage0a.json")["groups"]
    out = {"e46/windows_ident32": (E46 / "large" / "windows_ident32.pt", s0a["e12/build"]["sha256"]["ident32"]),
           "e54/windows_elicit": (E54 / "large" / "windows_elicit.pt", load(E54 / "stage1.json")["sha256"]["elicit"])}
    out |= {f"e46/{t}_s{s}": (E46 / "large" / f"{t}_s{s}.pt", s0a[f"e12/{t}"]["states"][f"s{s}"]["sha256"]) for t in E46_TASKS for s in SEEDS}
    ft = {**load(E53 / "finetune.json")["states"], **load(E53 / "finetune_amend1b.json")["states"]}
    out |= {f"e53/{t}_s{s}": (E53 / "large" / ft[f"{t}/s{s}/step300"]["file"], ft[f"{t}/s{s}/step300"]["sha256"]) for t in E53_TASKS for s in SEEDS}
    s2 = load(E54 / "stage2.json")["finetunes"]
    out |= {f"e54/{t}_s{s}": (E54 / "large" / f"{t}_s{s}.pt", s2[f"{t}_s{s}"]["sha256"]) for t in E54_TASKS for s in SEEDS}
    return out


def check_files(o: Out, F: dict[str, tuple[Path, str]]) -> None:
    got = {k: KD.sha256(p) if p.is_file() else "missing" for k, (p, _) in F.items()}
    o.put("sha256", {"n_files": len(F), "mismatch": sorted(k for k, (_, h) in F.items() if got[k] != h)},
          all(got[k] == h for k, (_, h) in F.items()))


def state(F: dict[str, tuple[Path, str]], key: str) -> ST.DW:
    return {k: v.float() for k, v in torch.load(F[key][0], weights_only=True).items()}


# ----------------------------------------------------------------------------- stage 0


def e53_tasks(pa: ModuleType, run: Any) -> dict[str, Any]:
    """E53's scored windows per task (`Run.scored`), the broad cipher's map checked against check_amend1.json."""
    broad = pa.a1_maps()["broad_cipher"]
    assert [[x, y] for x, y in broad.items()] == [r[:2] for r in load(E53 / "check_amend1.json")["maps"]["broad_cipher"]]
    return {n: run.scored(n, pi) for n, pi in (("lowercase", {}), ("cipher", pa.e53_pi()), ("broad_cipher", broad))}


def ref_reading(ctx: KD.Ctx, o: Out, key: str, t: ST.Task, dW: ST.DW, c: dict[str, Any]) -> None:
    """r and S of a reference state by keydep.readings (E54's path), against the filed values."""
    g = ST.gains(ctx.net, t, dW)
    rec = {**g, **KD.readings(ctx, t, dW, g, KD.base_of(ctx, t))}
    rec.pop("cells")
    ok = abs(rec["r"] - c["r_ref"]) <= c["r_tol"] and abs(rec["S"] - c["S_ref"]) <= c["S_tol"]
    if "block0_tol" in c:
        ok = ok and abs(rec["S_block0"]) <= c["block0_tol"]
    o.put(key, {**rec, "r_ref": c["r_ref"], "S_ref": c["S_ref"]}, ok)


def e53_reference(pa: ModuleType, run: Any, o: Out, low: Any, dW: ST.DW, c: dict[str, Any]) -> float:
    """E53 lowercase_s0 split and read by E53's code: stand-in ratio, G and r = g / G against read_amend1b.json."""
    d = pa.split_row(run, low, dW)
    ratio, r = d["masked_gain"] / d["piece_gain"], d["g"] / d["G"]
    ok = abs(ratio - c["ratio_ref"]) <= c["ratio_tol"] and abs(d["G"] - c["G_ref"]) <= c["G_tol"] and abs(r - c["r_ref"]) <= c["r_tol"]
    o.put("e53/lowercase_s0", {k: d[k] for k in ("G", "g", "piece_gain", "masked_gain", "cells", "boot")} | {"ratio": ratio, "r": r}, ok)
    return d["G"]


def next_path(ctx: KD.Ctx, o: Out, t: NextTask, dW: ST.DW, G_split: float, P: dict[str, Any]) -> None:
    """The NEXT path on lowercase_s0: full-model log-odds gain against the split read's G, and block 0's share in nats."""
    lo = ST.read(ctx.net, t, t.toks, dW, readout=ST.logodds_at) - ST.read(ctx.net, t, t.toks, readout=ST.logodds_at)
    g = ST.gains(ctx.net, t, dW)
    s = KD.steer_reading(ctx, t, dW, g)
    rec = {**g, **s, "G_logodds": float(lo.mean()), "G_split": G_split, "block0_nats": abs(s["S_block0"] * g["G_score"])}
    c = P["stage0"]["next_token_path"]
    o.put("next_path/lowercase_s0", rec, abs(rec["G_logodds"] - G_split) <= c["G_logodds_tol"]
          and rec["block0_nats"] <= P["settings"]["block0_nats_tol"])


def stage0(ctx: KD.Ctx, pa: ModuleType, run: Any, F: dict[str, tuple[Path, str]], P: dict[str, Any], o: Out) -> dict[str, NextTask]:
    c = P["stage0"]
    e46, e54 = (ST.load_task(ctx.net, F[k][0], n, kind) for k, n, kind in
                (("e46/windows_ident32", "ident32", "copy"), ("e54/windows_elicit", "elicit", "elicit")))
    pile = e53_tasks(pa, run)
    nt = {n: next_task(ctx.net, n, w.held, pa.CFG.half) for n, w in pile.items()}
    zero = {k: torch.zeros_like(w) for k, w in ctx.net.W0.items()}
    for n, t in {"e46/ident32": e46, "e54/elicit": e54, **{f"e53/{k}": v for k, v in nt.items()}}.items():
        G = ST.gains(ctx.net, t, zero)["G"]
        o.put(f"zero/{n}", {"G": G, "n_windows": len(t.toks)}, G == 0.0)
    G_split = e53_reference(pa, run, o, pile["lowercase"], state(F, "e53/lowercase_s0"), c["e53_lowercase_s0"])
    ref_reading(ctx, o, "e54/elicit_s0", e54, state(F, "e54/elicit_s0"), c["e54_elicit_s0"])
    ref_reading(ctx, o, "e46/ident32_s0", e46, state(F, "e46/ident32_s0"), c["e46_ident32_s0"])
    next_path(ctx, o, nt["lowercase"], state(F, "e53/lowercase_s0"), G_split, P)
    return nt


# ----------------------------------------------------------------------------- stage 1


def steer_row(ctx: KD.Ctx, t: NextTask, dW: ST.DW, P: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """S of one E53 state; 'undefined' under the floor, 'outside_expected' against the filed range (reported only)."""
    S = P["settings"]
    g = ST.gains(ctx.net, t, dW)
    rec: dict[str, Any] = {**g, **KD.steer_reading(ctx, t, dW, g)}
    rec["block0_nats"] = abs(rec["S_block0"] * g["G_score"])
    rec["defined"] = abs(g["G_score"]) >= S["S_floor"] and math.isfinite(rec["S"])
    lo, hi = P["stage1"]["expected"][t.name]["range"]
    rec["outside_expected"] = bool(rec["defined"] and not lo <= rec["S"] <= hi)
    return rec, rec["block0_nats"] <= S["block0_nats_tol"]


def stage1(ctx: KD.Ctx, nt: dict[str, NextTask], F: dict[str, tuple[Path, str]], P: dict[str, Any]) -> dict[str, Any]:
    o = Out("stage1.json")
    for n, s in ((n, s) for n in E53_TASKS for s in SEEDS):
        rec, ok = steer_row(ctx, nt[n], state(F, f"e53/{n}_s{s}"), P)
        o.put(f"e53/{n}_s{s}", rec, ok)
        if rec["outside_expected"]:
            log(f"SURPRISE: e53/{n}_s{s} S {rec['S']:.4f} outside the filed range {P['stage1']['expected'][n]['range']}")
    rows = o.out["rows"]
    if all(r["defined"] for r in rows.values()):
        low = min(rows[f"e53/lowercase_s{s}"]["S"] for s in SEEDS)
        high = max(rows[f"e53/{n}_s{s}"]["S"] for n in ("cipher", "broad_cipher") for s in SEEDS)
        o.out["order_as_expected"] = bool(low > high)
        log(f"[stage1] min lowercase S {low:.4f} vs max cipher S {high:.4f}: order as expected {low > high}")
    else:
        o.out["order_as_expected"] = "undefined: a state's S is undefined"
    dump(o.name, o.out)
    return rows


# ----------------------------------------------------------------------------- the table (saved JSON only)


def cell(value: float, source: str, defined: bool = True) -> dict[str, Any]:
    return {"value": value if defined else None, "parts_value": value, "source": source, "defined": defined}


def split_cells(num: float, den: float, g: float, G: float, src: str, th: dict[str, Any]) -> dict[str, Any]:
    """Stand-in ratio num / den (subcomponent gain den) and r = g / G, each 'undefined' under its floor."""
    return {"ratio": cell(num / den, src, den >= th["stand_in_ratio_floor"]), "r": cell(g / G if G else math.nan, src, abs(G) >= 1e-6)}


def table(s1: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    th, rows = P["settings"], {}
    a3, s0b, b3 = load(E46 / "amend3.json")["rows"], load(E46 / "stage0b.json")["states"], load(E52 / "baseline_neurons.json")["b3"]
    for t, s in ((t, s) for t in E46_TASKS for s in SEEDS):
        x = a3[f"{t}/s{s}"]
        rows[f"e46/{t}_s{s}"] = {"ratio": cell(x["gated_piece_gain"] / x["piece_gain"], f"regating_table amend3.json {t}/s{s}"),
                                 "r": cell(s0b[f"{t}/s{s}"]["r"], f"regating_table stage0b.json {t}/s{s}"),
                                 "S": cell(b3[f"{t}_s{s}"]["S"], f"baseline_neurons.json b3 {t}_s{s}")}
    rd = {**load(E53 / "read.json")["rows"], **load(E53 / "read_amend1b.json")["rows"]}
    for t, s in ((t, s) for t in E53_TASKS for s in SEEDS):
        x, k = rd[f"{t}/s{s}/step300"], f"e53/{t}_s{s}"
        src = f"performed_anchor {'read_amend1b' if t == 'broad_cipher' else 'read'}.json {t}/s{s}/step300"
        rows[k] = {**split_cells(x["masked_gain"], x["piece_gain"], x["g"], x["G"], src, th),
                   "S": cell(s1[k]["S"], f"stage1.json {k}", s1[k]["defined"])}
    s2 = load(E54 / "stage2.json")["states"]
    for t, s in ((t, s) for t in E54_TASKS for s in SEEDS):
        x, c = s2[f"{t}_s{s}"], s2[f"{t}_s{s}"]["cells"]
        src = f"keydependent_elicit stage2.json {t}_s{s}"
        rows[f"e54/{t}_s{s}"] = {"ratio": cell((c["L11"] - c["L00"]) / (c["U1"] - c["U0"]), src + " cells", c["U1"] - c["U0"] >= th["stand_in_ratio_floor"]),
                                 "r": cell(x["r"], src), "S": cell(x["S"], src)}
    for k, v in rows.items():
        log(f"[table {k}] " + " ".join(f"{q} {v[q]['parts_value']:.4f}{'' if v[q]['defined'] else ' (undefined)'}" for q in ("ratio", "r", "S")))
    return {"commit": git_head(), "runs": rows, "labels": "none yet (predictions.json label_predictions)"}


# ----------------------------------------------------------------------------- main


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    P = load(PRED)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    F, o = files(), Out("stage0.json")
    check_files(o, F)  # before the model loads
    pa = performed_anchor()
    ctx = KD.make_ctx(DEVICE, ALIVE, P, RESULTS / "large")
    run = pa.Run({"thresholds": load(E53 / "predictions_amend1b.json")["thresholds"]})  # `cells`' bootstrap floor (reported)
    nt = stage0(ctx, pa, run, F, P, o)
    o.out["pass"] = True
    dump(o.name, o.out)
    log("[stage0] pass")
    dump("table.json", table(stage1(ctx, nt, F, P), P))
    peak = torch.cuda.max_memory_allocated() / 2**30
    log(f"[done] peak GPU {peak:.1f} GB")


if __name__ == "__main__":
    main()
