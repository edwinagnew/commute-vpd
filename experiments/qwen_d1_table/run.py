"""E93 (qwen_d1_table), D1's first table: does the least-squares split keep the decomposition faithful on every saved
full update? Plan: reports/plans/2026-11-05_plan_e93-d1-table.md; lines: results/predictions.json (the run refuses to
start without it, and while results/stopped.json exists); rules and verdicts: rules.py; shared reads: `commute.qwen_d3`.
No adversary runs here (decisions.md, 2026-11-05, ask 31).

  --stage control     stage 0, alone and first: E40's step 10 and E81's TOFU update through every path, against the
                      numbers E86 and E68 stored
  --states a,b,...    names from rules.STATES, in the order given; refuses before stage 0 has passed; a state whose
                      state.json says done is skipped

Per state, cheapest first, every read logged and results/<state>/state.json rewritten as it ends: file and norm;
fractions (the loss of W0 + t δW on the arm's gain rows, no masks); clean W0 (the ruler), FT, random, signs; the
least-squares parts (share left in Δ, squared size); clean moved part and remainder, and the loss at W0 + moved part;
Γ: W0 masked and do-nothing (base), least squares under its own masks and under the W0 weights' masks (own); on a
graded arm the ceiling and a random under masks. Exit 1: a stop (results/stopped.json); exit 3: the time left under
the job's limit is under the longest state so far, resubmit the same command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rules
import torch
from transformers import AutoTokenizer

from commute import qwen_d3 as d3
from commute import qwen_eval, qwen_sets
from commute.jobs import EXPERIMENTS, hard_exit
from commute.qwen_sft import Seq, gold_loss, load, sha256

RESULTS = rules.RESULTS
PRED, STOPPED, CONTROL = (RESULTS / f for f in ("predictions.json", "stopped.json", "control.json"))
DW = EXPERIMENTS / "qwen_sft" / "results" / "large" / "nowarmup_regen" / "dW"
E40_10, U1_FP32 = DW / "step_10.safetensors", DW / "step_100_fp32.safetensors"  # bf16; u1's step 100 kept in fp32
E81 = EXPERIMENTS / "qwen_tofu" / "results" / "large" / "dW" / "a2_2e-5" / "step_1250.safetensors"
OLD_FORM = ("a2_s42",)  # E72's in-span (dσ alone) read beside the moved part
T0 = time.time()


@dataclass(frozen=True)
class Config:
    gain_batch: int = 8  # E80 Amendment 2's and E86's GOLD_BATCH
    limit_s: float = 2 * 3600 - 180  # debug QoS, less a margin for the last write
    first_guess_s: float = 40 * 60  # the plan's anchor-2 state, 41 min, before one is measured

    def __post_init__(self) -> None:
        assert rules.TS[0] == 1.0 and 0.1 in rules.TS and 0.5 in rules.TS and self.limit_s > self.first_guess_s


C = Config()


def log(msg: str) -> None:
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with (RESULTS / "run.log").open("a") as f:
        f.write(line + "\n")


def write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float) + "\n")
    tmp.replace(path)


def check(name: str, holds: bool, detail: dict[str, Any]) -> dict[str, Any]:
    log(f"[check] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail, default=float)}")
    if not holds:
        write(STOPPED, {"check": name, "detail": detail, "at_s": time.time() - T0})
        log(f"STOP {name}")
        hard_exit(1)
    return detail | {"holds": True}


def rel(x: float, ref: float) -> float:
    return abs(x / ref - 1) if ref != 0 else float("inf")


def kmean(s: d3.Sums, rs: d3.RowSet, key: str = "K", c: int = d3.POOLED) -> float:
    return d3.total(s, key, c) / rs.counts()["kl"][c]


# ----------------------------------------------------------------------------- the reads


def gain(d: d3.Decomp, key: str, seqs: list[Seq]) -> float:
    """Mean response-token loss of weight set `key`, no masks, fp32 (E86's gain read)."""
    t0 = time.time()
    d.use(key)
    x = gold_loss(d.model, seqs, C.gain_batch, d3.DEV)
    log(f"[gain {key}] {x:.12f} nats per response token, {len(seqs)} rows, {time.time() - t0:.0f} s, {d3.peaks()}")
    return x


def gain_at(d: d3.Decomp, change: dict[str, torch.Tensor], seqs: list[Seq], scale: float = 1.0) -> float:
    d.weights.put("state", change, scale)
    x = gain(d, "state", seqs)
    d.weights.drop("state")
    return x


def clean_at(d: d3.Decomp, rs: d3.RowSet, name: str, change: dict[str, torch.Tensor] | None = None) -> d3.Sums:
    """The clean read of W0 + `change` (None: `name` is a weight set already held), logged."""
    t0 = time.time()
    if change is not None:
        d.weights.put("state", change)
    s = d3.clean(d, rs, "state" if change is not None else name)
    if change is not None:
        d.weights.drop("state")
    log(f"[clean {name}] K {kmean(s, rs):.9g} nll {d3.total(s, 'nll') / rs.counts()['nll'][d3.POOLED]:.9g}; "
        f"rows {len(rs.tokens)}, {time.time() - t0:.0f} s, {d3.peaks()}")
    return s


def cells_timed(d: d3.Decomp, rs: d3.RowSet, tag: str, **kw: Any) -> d3.Sums:
    t0 = time.time()
    s = d3.cells(d, rs, **kw)
    log(f"[cells {tag}] " + " ".join(f"{k} {kmean(s, rs, k) if k.startswith('K') else d3.total(s, k):.9g}" for k in s)
        + f"; rows {len(rs.tokens)}, {time.time() - t0:.0f} s, {d3.peaks()}")
    return s


def memory(name: str, F: dict[str, Any]) -> None:
    gpu, host = torch.cuda.max_memory_allocated() / d3.GIB, d3.host_gb()
    check(f"{name}_memory", gpu <= F["gpu_max_gib"] and host <= F["host_max_gb"], {"gpu_gib": gpu, "host_gb": host})


# ----------------------------------------------------------------------------- stage 0, the control


def control_u1(d: d3.Decomp, m: d3.RowSet, seqs: list[Seq], P: dict[str, Any], out: dict[str, Any]) -> None:
    """E40's step 10 (E86's u1): gains, clean W0 twice and FT, the share left in Δ and its null, E72's in-span; then
    W0 masked and do-nothing (E68) and least squares under its own and the W0 weights' masks (E68, E94's stage 0)."""
    R, F = P["controls"], P["first_minutes"]
    dW = load(E40_10)
    d.weights.put("FT", dW)
    t0 = time.time()
    g = {"W0": gain(d, "W0", seqs), "FT": gain(d, "FT", seqs), "t0.1": gain_at(d, dW, seqs, 0.1)}
    out["seconds"]["gain"] = (time.time() - t0) / 3
    out["gain"] = check("u1_gain_E86", all(rel(g[k], R["u1_gain"][k]) <= R["rep"] for k in g), g)
    t0 = time.time()
    k = {n: kmean(clean_at(d, m, key), m) for n, key in (("W0", "W0"), ("W0_again", "W0"), ("FT", "FT"))}
    out["seconds"]["clean"] = (time.time() - t0) / 3
    out["clean"] = check("u1_clean_E86", all(rel(k[n], R["u1_ruler"]) <= R["id"] for n in ("W0", "W0_again"))
                         and k["FT"] <= R["ft_zero"], k)
    p = d3.ls_parts(d, dW, old=True)
    out["seconds"]["split"] = p.seconds
    sh, null, size = p.sums["left"] / p.sums["size"], p.sums["null"] / p.sums["size"], p.sums["size"]
    log(f"[split u1 step 10] {json.dumps(p.sums)}, {p.seconds:.0f} s, {d3.peaks()}")
    out["share"] = check("u1_share_E86", rel(sh, R["u1_share"]) <= R["share_rel"] and sh < null and rel(size, R["u1_size"]) <= R["id"],
                         {"share": sh, "null": null, "size": size})
    assert p.old is not None
    ins = kmean(clean_at(d, m, "inspan_old", p.old), m)
    out["inspan_old"] = check("u1_inspan_old_E86", rel(ins, R["u1_inspan_old"]) <= R["rep"], {"K": ins})
    b = cells_timed(d, m, "u1 base")
    memory("control", F)
    o = cells_timed(d, m, "u1 own", change=d3.ls_change(d, p.X), w0_v=False)
    got = {"K00": kmean(b, m, "K00"), "K01": kmean(b, m, "K01"), "K10": kmean(o, m, "K10"), "K11": kmean(o, m, "K11")}
    out["masked"] = check("u1_masked_E68", all(rel(got[x], R["u1_masked"][x]) <= R["rep"] for x in got), got)
    d.weights.drop("FT")


def control_e81(d: d3.Decomp, rs: d3.RowSet, seqs: list[Seq], R: dict[str, Any], out: dict[str, Any]) -> None:
    """E81's TOFU update on E86's rows: W0's loss, the tenth's kept, and the ruler on the 128 TOFU rows."""
    dW = load(E81)
    d.weights.put("FT", dW)
    w0, ft, t1 = gain(d, "W0", seqs), gain(d, "FT", seqs), gain_at(d, dW, seqs, 0.1)
    tenth = rules.kept(t1, w0, ft)
    ruler = kmean(clean_at(d, rs, "W0"), rs)
    got = {"W0": w0, "tenth": float("nan") if tenth is None else tenth, "ruler": ruler}
    out["e81"] = check("e81_tofu_E86", all(rel(got[x], R["e81"][x]) <= R["rep"] for x in got), got)
    d.weights.drop("FT")


def stage_control(d: d3.Decomp, rows: dict[str, d3.RowSet], seqs: dict[str, list[Seq]], P: dict[str, Any]) -> None:
    out: dict[str, Any] = {"seconds": {}}
    control_u1(d, rows["maths"], seqs["maths"], P, out)
    control_e81(d, rows["tofu"], seqs["tofu_all"], P["controls"], out)
    s = out["seconds"]  # a maths state: 7 gains, 7 clean reads, the split, two masked passes (E68: about 2.6 min each)
    out["projection_min"] = {"maths_state": (7 * s["gain"] + 7 * s["clean"] + s["split"] + 2 * 160) / 60,
                             "graded_extra": (s["split"] + 3 * 160) / 60}
    write(CONTROL, out | {"passed": True})
    log(f"[control] passed; projection {json.dumps(out['projection_min'])} min, {d3.peaks()}")


# ----------------------------------------------------------------------------- one state


def fractions(d: d3.Decomp, dW: dict[str, torch.Tensor], seqs: list[Seq], rec: dict[str, Any], F: dict[str, Any]) -> None:
    fr = rec["fractions"] = {"W0": gain(d, "W0", seqs), "FT": gain(d, "FT", seqs)}
    for t in rules.TS:
        fr[f"t{t}"] = gain_at(d, dW, seqs, t)
    fr["W0_end"] = gain(d, "W0", seqs)
    got = {"t1_vs_FT": rel(fr["t1.0"], fr["FT"]), "W0_end_vs_W0": rel(fr["W0_end"], fr["W0"])}  # the same sums twice
    check(f"{rec['name']}_fractions", max(got.values()) <= F["t1_rel"], got | rules.saturation(fr))


def clean_reads(d: d3.Decomp, rs: d3.RowSet, dW: dict[str, torch.Tensor], rec: dict[str, Any], F: dict[str, Any]) -> None:
    cl = rec["clean"] = {"W0": clean_at(d, rs, "W0"), "FT": clean_at(d, rs, "FT")}
    check(f"{rec['name']}_clean_FT", kmean(cl["FT"], rs) <= F["ft_zero"], {"K": kmean(cl["FT"], rs)})
    memory(rec["name"], F)
    write(RESULTS / rec["name"] / "state.json", rec)
    for kind in ("random", "signs"):
        cl[kind] = clean_at(d, rs, kind, d3.wrong_change(dW, kind))


def parts_reads(d: d3.Decomp, rs: d3.RowSet, dW: dict[str, torch.Tensor], seqs: list[Seq], rec: dict[str, Any]) -> dict[str, torch.Tensor]:
    """The split; the clean reads of the moved part and the remainder (E72's in-span on OLD_FORM); the gain at the moved part."""
    p = d3.ls_parts(d, dW, old=rec["name"] in OLD_FORM)
    rec["split"] = p.sums | {"seconds": p.seconds}
    log(f"[split {rec['name']}] share {p.sums['left'] / p.sums['size']:.9g} null {p.sums['null'] / p.sums['size']:.9g} "
        f"size {p.sums['moved'] / p.sums['size']:.9g}; {p.seconds:.0f} s, {d3.peaks()}")
    write(RESULTS / rec["name"] / "state.json", rec)
    rec["clean"]["moved"] = clean_at(d, rs, "moved", p.moved)
    rec["fractions"]["moved"] = gain_at(d, p.moved, seqs)
    rec["clean"]["remainder"] = clean_at(d, rs, "remainder", {s: dW[s].float() - p.moved[s] for s in dW})
    if p.old is not None:
        rec["clean"]["inspan_old"] = clean_at(d, rs, "inspan_old", p.old)
    write(RESULTS / rec["name"] / "state.json", rec)
    return p.X


def graded_reads(d: d3.Decomp, rs: d3.RowSet, dW: dict[str, torch.Tensor], rec: dict[str, Any]) -> None:
    """E72's ceiling (do-nothing's masks plus the whole δW); a random under masks: a Gaussian at δW's norm per matrix
    (the clean table's random) placed by least squares, Γ on its own weights and moved V′. Leaves "FT" as the random's."""
    t0 = time.time()
    rec["ceiling"] = d3.ceiling(d, rs, {s: d3.pinned(w.float()) for s, w in dW.items()})
    log(f"[ceiling {rec['name']}] K_c {kmean(rec['ceiling'], rs, 'K_c'):.9g}; {time.time() - t0:.0f} s, {d3.peaks()}")
    write(RESULTS / rec["name"] / "state.json", rec)
    G = d3.wrong_change(dW, "random")
    X = d3.ls_parts(d, G).X
    d.weights.put("FT", G)
    del G
    rnd = cells_timed(d, rs, f"{rec['name']} random under masks", change=d3.ls_change(d, X), w0_v=False)
    rec["random"] = {"nll_m11": rnd["nll_m11"], "K11_against_its_own_clean": rnd["K11"]}


def read_state(d: d3.Decomp, rows: dict[str, d3.RowSet], seqs: dict[str, list[Seq]], name: str, P: dict[str, Any]) -> None:
    arm, rk, label = rules.STATES[name]
    g = d3.grid_state(arm, "full")
    if name == "u1":  # E91 read the bf16 file; the fp32 one is scored (plan), its norm against the same record
        g |= {"path": U1_FP32, "file": str(U1_FP32.relative_to(EXPERIMENTS)), "dtype": "torch.float32"}
    dW, F, rs = d3.load_state(g), P["first_minutes"], rows[rk]
    nrm = d3.norm(dW)
    check(f"{name}_norm", rel(nrm, g["norm"]) <= F["norm_rel"], {"norm": nrm, "e91": g["norm"]})
    rec = {"name": name, "grid_arm": arm, "rows": rk, "arm": label, "file": g["file"], "dtype": g["dtype"], "norm": nrm,
           "sha256": sha256(g["path"]), "n": row_counts(rs), "seconds": {}}
    d.weights.put("FT", dW)
    t0 = time.time()
    fractions(d, dW, seqs[rk], rec, F)
    rec["seconds"]["fractions"], t0 = time.time() - t0, time.time()
    clean_reads(d, rs, dW, rec, F)
    rec["seconds"]["clean"], t0 = time.time() - t0, time.time()
    X = parts_reads(d, rs, dW, seqs[rk], rec)
    rec["seconds"]["parts"] = time.time() - t0
    rec["base"] = cells_timed(d, rs, f"{name} base")
    rec["own"] = cells_timed(d, rs, f"{name} own", change=d3.ls_change(d, X), w0_v=False)
    del X
    write(RESULTS / name / "state.json", rec)
    if label in rules.BEHAVIOUR and rules.saturation(rec["fractions"])["band"] == "graded":
        graded_reads(d, rs, dW, rec)
    if name == "u1":  # the bf16 floor of the share: the same split on the bf16 file E91 read
        rec["split_bf16"] = d3.ls_parts(d, d3.load_state(d3.grid_state(arm, "full"))).sums
    d.weights.drop("FT")
    write(RESULTS / name / "state.json", rec | {"done": True})
    t = rules.score()
    log(f"[{name}] {json.dumps({k: t['states'][name][k] for k in ('verdict', 'note') if k in t['states'][name]})} "
        f"rules {json.dumps({k: v['result'] for k, v in t['states'][name]['rules'].items()})}")


def row_counts(rs: d3.RowSet) -> dict[str, list[list[float]]]:
    """Each row's positions per class: every position ("kl"), and those with a next token ("nll")."""
    L = rs.tokens.shape[1]
    kl, nll = torch.ones(L), torch.ones(L)
    nll[-1] = 0
    return {"kl": [d3.by_class(kl, c) for c in rs.cls], "nll": [d3.by_class(nll, c) for c in rs.cls]}


# ----------------------------------------------------------------------------- the job


def build(P: dict[str, Any]) -> tuple[dict[str, d3.RowSet], dict[str, list[Seq]]]:
    tok = AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV)
    sets = qwen_sets.load()
    rows = {"maths": d3.maths_rows(), "a1": d3.own_rows("a1", tok, sets), "tofu": d3.own_rows("tofu", tok, sets)}
    seqs = {"maths": d3.gold_held(tok), "a1": d3.a1_seqs(tok, sets["test"]), "tofu": d3.tofu_seqs(tok, d3.tofu_kept()),
            "tofu_all": d3.tofu_seqs(tok, None)}
    got = {k: rows[k].counts()["kl"][0] for k in ("a1", "tofu")} | {f"seqs_{k}": len(v) for k, v in seqs.items()}
    check("rows_and_seqs", got == P["controls"]["rows_and_seqs"], got)
    return rows, seqs


def main() -> None:
    ap = argparse.ArgumentParser(description="E93: D1's first table")
    ap.add_argument("--stage", choices=("control",))
    ap.add_argument("--states", help="comma-separated names from rules.STATES")
    ap.add_argument("--decomp", default=os.environ.get("COMMUTE_QWEN_DIR", ""))
    a = ap.parse_args()
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
    if STOPPED.is_file():
        sys.exit(f"refusing to start: {STOPPED} exists (a check missed; the lead decides)")
    names = a.states.split(",") if a.states else []
    if bool(a.stage) == bool(names) or any(n not in rules.STATES for n in names):
        sys.exit("give --stage control or --states from rules.STATES, not both")
    if names and not json.loads(CONTROL.read_text() if CONTROL.is_file() else "{}").get("passed"):
        sys.exit("refusing: stage 0 (--stage control) has not passed")
    P = json.loads(PRED.read_text())
    log(f"[start] {vars(a)}; config {C}")
    rows, seqs = build(P)
    with torch.no_grad():
        d = d3.Decomp(Path(a.decomp), log)
        if a.stage:
            stage_control(d, rows, seqs, P)
            hard_exit(0)
        took: list[float] = []
        for n in names:
            if rules.load(n) is not None:
                log(f"[{n}] already written")
                continue
            longest = max(took, default=C.first_guess_s)
            if time.time() - T0 + longest > C.limit_s:
                log(f"cut short before {n}: {time.time() - T0:.0f} s used, longest state {longest:.0f} s; resubmit")
                hard_exit(3)
            t0 = time.time()
            read_state(d, rows, seqs, n, P)
            took.append(time.time() - t0)
            log(f"[{n}] done in {time.time() - t0:.0f} s, {d3.peaks()}")
    log(f"[done] {names}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)
