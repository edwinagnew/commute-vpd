"""E93 Amendment 2 (qwen_d1_table): E93's table and Amendment 1's adjoint split, re-read on the NEW decomposition
p-9707d6ff, with E86's least-squares rows and E75's wrong-change rows. Plans: reports/plans/2026-11-09_plan_new-decomposition-reruns.md
(line 1) and reports/plans/2026-11-09_plan_e93-new-decomposition.md; lines: results/predictions_amend2.json (the run
refuses to start unless it is committed unchanged, and while results/amend2/stopped.json exists). Everything goes in
results/amend2/, every file stamped `decomp`. Scoring, old beside new, and the CPU dry run: amend2_score.py.

  --stage control            stage 0, alone and first, on E40's step 10: the W0 and do-nothing references on new
  --states a,b --map ls      E93's reads of a state on new; --map adjoint needs that state's ls record
  --extras R10,u3,tofu_E81   E86's and E75's rows on updates outside the table (no Γ reads)

What a state reads. Carried unchanged from E93's results/<state>/state.json, because no decomposition enters them
(`experiments/DECOMPOSITIONS.md`: the δW, behaviour and clean-model reads): the gain rows' losses (W0, FT, t), the
clean reads of W0 (the ruler), FT, random and signs; the file, its sha256 and norm. Read on new: the split and share,
clean moved and remainder (δW − moved; on OLD_FORM states also E72's in-span and its remainder), the gain at the moved
part, W0 masked and do-nothing ("base"), the state's own-mask read, and on a graded arm the ceiling and random under
masks. --map adjoint carries the new ls record's base, ceiling and random as Amendment 1 carried E93's.
Exit 1: a stop (stopped.json); 3: resubmit the same command (time left under the longest state so far).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from commute import qwen_d3 as d3
from commute import qwen_maps, split
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed
from commute.qwen_sft import load, sha256
from commute.qwen_vpd import SITES, decomp_id, importances, masked_logits
from commute.vpd import kl_per_position

sc = load_experiment("qwen_d1_table", "amend2_score")
run, am1, rules = sc.run, sc.am1, sc.rules
NEW, OUT, PRED = sc.NEW, sc.OUT, sc.PRED
CONTROL = OUT / "control.json"
W0_REFS = EXPERIMENTS / "qwen_new_decomp" / "results" / "w0_refs_stage1.json"
E40_FILE = "qwen_sft/results/large/nowarmup_regen/dW/step_10.safetensors"  # stage 0's update, run.E40_10
OLD_FORM = ("a2_s42", "u2", "R_s0")  # E72's in-span and remainder: a2_s42 as E93, u2 and R_s0 for E86's and E75's rows
EXTRA_OF = {"u2": "u2", "R_s0": "R_s0"}  # a table state that also writes an extras record, under this name


# ----------------------------------------------------------------------------- stage 0


def w0_masked_kl(d: d3.Decomp, n: int = d3.MATHS_ROWS, length: int = 512) -> float:
    """Mean KL(W0 clean ‖ W0 masked, Δ off) on the rows E131's w0_refs read: its `masked_kl`, per row then averaged."""
    d.use("W0")
    rows = torch.tensor([r[:length] for r in json.loads(d3.MATHS.read_text())["rows"][:n]])
    total = 0.0
    for t in rows:
        t = t[None].to(d3.DEV)
        clean, g = importances(d.model, t, d.gamma, d.comps)
        total += float(kl_per_position(clean, masked_logits(d.model, t, d.comps, g)).mean())
    return total / n


def left_share(A: torch.Tensor, b: Any, U_A: torch.Tensor, V_A: torch.Tensor) -> float:
    """‖A − dσ(the split of A)‖² / ‖A‖², through the pieces the forward applies."""
    return float(((A - d3.moved_of(split.least_squares_x(A, b).float(), U_A, V_A)[0]) ** 2).sum() / (A * A).sum())


def split_control(d: d3.Decomp, dW: dict[str, torch.Tensor]) -> dict[str, Any]:
    """On three sites of the new basis: a matrix in the split's span leaves ~0 (the positive control), and a Gaussian
    leaves the exact null `split.gaussian_null` (the null)."""
    out: dict[str, Any] = {}
    for i in (0, len(SITES) // 2, len(SITES) - 1):
        U_A, V_A, b = d3.basis(d, SITES[i])
        M = dW[SITES[i]].to(d3.DEV, d3.F64)
        inspan = d3.moved_of(d3.unit_draw(tuple(M.shape), d3.WRONG_SEED + i).to(d3.DEV).float(), U_A, V_A)[0]
        out[SITES[i]] = {"in_span_left": left_share(inspan, b, U_A, V_A),
                         "gaussian_left": left_share(qwen_maps.gaussian_at_norm(M, d3.WRONG_SEED + i), b, U_A, V_A),
                         "null": split.gaussian_null(b, M.shape[0], M.shape[1])}
        del U_A, V_A, b, M
    return out


def do_nothing_two_paths(d: d3.Decomp, m: d3.RowSet, out: dict[str, Any], R: dict[str, Any], F: dict[str, Any]) -> None:
    """Base cells at u1 step 10, then the same cells through the moved-components path with nothing moved."""
    t0 = time.time()
    b = run.cells_timed(d, m, "u1 step 10 base")
    out["seconds"]["cells"] = time.time() - t0
    run.memory("control", F)
    again = run.cells_timed(d, m, "u1 step 10 do-nothing, second path", change=d3.ls_change(d, None), w0_v=False)
    base = {k: run.kmean(b, m, k) for k in ("K0", "K00", "K01")}
    second = {"K00": run.kmean(again, m, "K10"), "K01": run.kmean(again, m, "K11")}
    out["do_nothing"] = run.check("do_nothing_two_paths", all(run.rel(second[x], base[x]) <= R["rep"] for x in second)
                                  and base["K01"] < base["K00"], {"base": base, "second": second})


def stage_control(d: d3.Decomp, rows: dict[str, d3.RowSet], seqs: dict[str, Any], P: dict[str, Any]) -> None:
    R, F, m, out = P["controls"], P["first_minutes"], rows["maths"], {"decomp": NEW, "seconds": {}}
    refs = json.loads(W0_REFS.read_text())
    assert refs.get("pass") is True and refs.get("decomp") == NEW, f"{W0_REFS}: not a passed read of {NEW}"
    out["alive"] = run.check("alive_E131", d.alive.count() == refs["alive"]["total"], {"got": d.alive.count(), "ref": refs["alive"]["total"]})
    kl, ref = w0_masked_kl(d), refs["blocks"]["maths"]["masked_kl"]
    out["w0_masked_kl"] = run.check("w0_masked_kl_E131", run.rel(kl, ref) <= R["rep"], {"K": kl, "ref": ref})
    torch.cuda.reset_peak_memory_stats()  # these 512-token rows peak above the 256-token rows the memory line was filed for
    dW = load(run.E40_10)
    d.weights.put("FT", dW)  # the decomposition-free reads E93 stored, which this job carries: the same numbers or stop
    g = {"W0": run.gain(d, "W0", seqs["maths"]), "FT": run.gain(d, "FT", seqs["maths"]), "t0.1": run.gain_at(d, dW, seqs["maths"], 0.1)}
    out["gain"] = run.check("carried_gain", all(run.rel(g[k], R["u1_gain"][k]) <= R["rep"] for k in g), g)
    k = {n: run.kmean(run.clean_at(d, m, key), m) for n, key in (("W0", "W0"), ("W0_again", "W0"), ("FT", "FT"))}
    out["clean"] = run.check("carried_clean", all(run.rel(k[n], R["u1_ruler"]) <= R["id"] for n in ("W0", "W0_again")) and k["FT"] <= R["ft_zero"], k)
    do_nothing_two_paths(d, m, out, R, F)
    p = d3.ls_parts(d, dW, old=True)
    out["seconds"]["split"] = p.seconds
    sh, null, size = p.sums["left"] / p.sums["size"], p.sums["null"] / p.sums["size"], p.sums["size"]
    out["share"] = run.check("ls_share", sh < null and run.rel(size, R["u1_size"]) <= R["id"], {"share": sh, "null": null, "size": size})
    c = split_control(d, dW)
    holds = all(x["in_span_left"] <= R["split_pos_max"] and run.rel(x["gaussian_left"], x["null"]) <= R["split_null_rel"] for x in c.values())
    out["split_control"] = run.check("split_control", holds, c)
    moved = run.kmean(run.clean_at(d, m, "moved", p.moved), m)
    extra = old_form_reads(d, m, dW, p)
    out["clean_moved"] = run.check("moved_under_ruler", moved < k["W0"], {"K": moved, "ruler": k["W0"]})
    write_extra("u1_10", "maths", E40_FILE, sha256(run.E40_10), p, extra)
    o = run.cells_timed(d, m, "u1 step 10 ls own", change=d3.ls_change(d, p.X), w0_v=False)
    out["ls_own"] = {"K10": run.kmean(o, m, "K10"), "K11": run.kmean(o, m, "K11")}
    del p, o
    d.weights.put("FT", dW)  # the put's drop returns the freed pinned pages (X and the moved part) before Σ's 11 GB
    a = d3.adjoint_parts(d, dW)
    out["seconds"]["adjoint_parts"] = a.seconds
    out["adjoint"] = {"share": a.sums["left"] / a.sums["size"], "null": a.sums["null"] / a.sums["size"], "size": a.sums["moved"] / a.sums["size"],
                      "clean_moved_K": run.kmean(run.clean_at(d, m, "adjoint moved", a.moved), m)}
    run.log(f"[adjoint u1 step 10] {json.dumps(out['adjoint'])}; {a.seconds:.0f} s, {d3.peaks()}")
    del a
    d.weights.drop("FT")
    run.control_e81(d, rows["tofu"], seqs["tofu_all"], R, out)
    s = out["seconds"]
    out["projection_min"] = {"ls_state": (s["split"] + 3 * 60 + 2 * s["cells"]) / 60, "adjoint_state": (s["adjoint_parts"] + 3 * 60 + 200) / 60}
    run.write(CONTROL, out | {"passed": True})
    run.log(f"[control] passed; projection {json.dumps(out['projection_min'])}, {d3.peaks()}")
    print("STAGE 0 DONE", flush=True)


# ----------------------------------------------------------------------------- the reads


def old_form_reads(d: d3.Decomp, rs: d3.RowSet, dW: dict[str, torch.Tensor], p: Any) -> dict[str, d3.Sums]:
    """E72's in-span (dσ alone, `LsParts.old`) and its remainder δW − dσ, on the clean model."""
    return {"inspan": run.clean_at(d, rs, "inspan_old", p.old),
            "remainder": run.clean_at(d, rs, "remainder_old", {s: dW[s].float() - p.old[s] for s in dW})}


def k_read(s: d3.Sums) -> dict[str, Any]:
    """A clean read in E75's and E86's format: per-row K over every position of 256."""
    return {"T": d3.ROW_LEN, "sums": {"K": [r[d3.POOLED] for r in s["K"]]}}


def write_extra(name: str, rk: str, file: str, sha: str, p: Any, clean: dict[str, d3.Sums]) -> None:
    run.write(OUT / "extras" / f"{name}.json", {"name": name, "rows": rk, "file": file, "sha256": sha, "split": p.sums,
              "clean": {k: k_read(v) for k, v in clean.items()}, "decomp": NEW, "done": True})
    print(f"STAGE extras/{name} DONE", flush=True)


def load_checked(rec: dict[str, Any], F: dict[str, Any]) -> dict[str, torch.Tensor]:
    """A table state's δW, the same bytes E93 read (sha256), its norm as a second stop."""
    path, name = EXPERIMENTS / rec["file"], rec["name"]
    run.check(f"{name}_file", sha256(path) == rec["sha256"], {"file": rec["file"]})
    dW = d3.load_state({"path": path, "dtype": rec["dtype"]})
    nrm = d3.norm(dW)
    run.check(f"{name}_norm", run.rel(nrm, rec["norm"]) <= F["norm_rel"], {"norm": nrm, "e93": rec["norm"]})
    return dW


def ls_parts_reads(d: d3.Decomp, rs: d3.RowSet, dW: dict[str, torch.Tensor], seqs: list[Any], rec: dict[str, Any]) -> dict[str, torch.Tensor]:
    """E93's `parts_reads` with the old-form reads on OLD_FORM states and an extras record: split, moved, remainder, gain."""
    name, p = rec["name"], d3.ls_parts(d, dW, old=rec["name"] in OLD_FORM)
    rec["split"] = p.sums | {"seconds": p.seconds}
    run.log(f"[split {name}] share {p.sums['left'] / p.sums['size']:.9g} null {p.sums['null'] / p.sums['size']:.9g} "
            f"size {p.sums['moved'] / p.sums['size']:.9g}; {p.seconds:.0f} s, {d3.peaks()}")
    run.write(OUT / name / "state.json", rec)
    rec["clean"]["moved"] = run.clean_at(d, rs, "moved", p.moved)
    rec["fractions"]["moved"] = run.gain_at(d, p.moved, seqs)
    rec["clean"]["remainder"] = run.clean_at(d, rs, "remainder", {s: dW[s].float() - p.moved[s] for s in dW})
    if p.old is not None:
        old = old_form_reads(d, rs, dW, p)
        rec["clean"]["inspan_old"] = old["inspan"]
        if name in EXTRA_OF:
            write_extra(EXTRA_OF[name], rec["rows"], rec["file"], rec["sha256"], p, old)
    run.write(OUT / name / "state.json", rec)
    return p.X


def read_ls(d: d3.Decomp, rows: dict[str, d3.RowSet], seqs: dict[str, Any], name: str, P: dict[str, Any]) -> None:
    rec, F = sc.carried(name), P["first_minutes"]
    rs, sq, label = rows[rec["rows"]], seqs[rec["rows"]], rec["arm"]
    dW = load_checked(rec, F)
    d.weights.put("FT", dW)
    t0 = time.time()
    X = ls_parts_reads(d, rs, dW, sq, rec)
    run.memory(name, F)
    rec["seconds"]["parts"], t0 = time.time() - t0, time.time()
    rec["base"] = run.cells_timed(d, rs, f"{name} base")
    rec["own"] = run.cells_timed(d, rs, f"{name} own", change=d3.ls_change(d, X), w0_v=False)
    rec["seconds"]["cells"] = time.time() - t0
    del X
    run.write(OUT / name / "state.json", rec)
    if label in rules.BEHAVIOUR and rules.saturation(rec["fractions"])["band"] == "graded":
        run.graded_reads(d, rs, dW, rec)
    if name == "u1":  # E93's printed bf16 floor of the share (the bf16 file E91 read)
        rec["split_bf16"] = d3.ls_parts(d, d3.load_state(d3.grid_state(rec["grid_arm"], "full"))).sums
    d.weights.drop("FT")
    run.write(OUT / name / "state.json", rec | {"done": True})
    print(f"STAGE ls/{name} DONE", flush=True)


def read_adjoint(d: d3.Decomp, rows: dict[str, d3.RowSet], seqs: dict[str, Any], name: str, P: dict[str, Any]) -> None:
    """Amendment 1's read on new; base, ceiling, random and the carried reads come from the new ls record."""
    ls, F = sc.load_rec(name), P["first_minutes"]
    assert ls is not None, f"{name}: --map ls has not been run"
    rec, rs, sq = am1.skeleton(ls) | {"decomp": NEW}, rows[ls["rows"]], seqs[ls["rows"]]
    dW = load_checked(rec, F)
    d.weights.put("FT", dW)
    p = d3.adjoint_parts(d, dW, old=name in run.OLD_FORM)
    rec["split"] = p.sums | {"seconds": p.seconds, "c_zero": sum(c == 0 for c in p.c.values())}
    run.log(f"[adjoint {name}] share {p.sums['left'] / p.sums['size']:.9g} null {p.sums['null'] / p.sums['size']:.9g} "
            f"size {p.sums['moved'] / p.sums['size']:.9g}; {p.seconds:.0f} s, {d3.peaks()}")
    d.weights.put("adjoint", p.moved)
    rec["clean"]["moved"] = run.clean_at(d, rs, "adjoint")
    run.memory(name, F)
    rec["fractions"]["moved"] = run.gain(d, "adjoint", sq)
    run.write(OUT / name / "adjoint.json", rec)
    rec["own"] = run.cells_timed(d, rs, f"{name} adjoint own", change=d3.adjoint_change(d, dW, p.c), w0_v=False, state="adjoint")
    d.weights.drop("adjoint")
    for s, w in dW.items():  # the remainder δW − moved part, in place
        p.moved[s].neg_().add_(w.float())
    rec["clean"]["remainder"] = run.clean_at(d, rs, "remainder", p.moved)
    if p.old is not None:
        rec["clean"]["adjoint_old"] = run.clean_at(d, rs, "adjoint_old", p.old)
    d.weights.drop("FT")
    run.write(OUT / name / "adjoint.json", rec | {"done": True})
    print(f"STAGE adjoint/{name} DONE", flush=True)


def read_extra(d: d3.Decomp, rows: dict[str, d3.RowSet], name: str, P: dict[str, Any]) -> None:
    """E86's and E75's rows on an update outside the table: split, share, in-span and remainder (no Γ read)."""
    x, rs = P["extras"][name], rows[P["extras"][name]["rows"]]
    path = EXPERIMENTS / x["file"]
    run.check(f"{name}_file", sha256(path) == x["sha256"], {"file": x["file"]})
    dW = d3.load_state({"path": path, "dtype": x["dtype"]})
    d.weights.put("FT", dW)
    p = d3.ls_parts(d, dW, old=True)
    run.log(f"[split {name}] share {p.sums['left'] / p.sums['size']:.9g} null {p.sums['null'] / p.sums['size']:.9g}; {p.seconds:.0f} s, {d3.peaks()}")
    write_extra(name, x["rows"], x["file"], x["sha256"], p, old_form_reads(d, rs, dW, p))
    d.weights.drop("FT")


# ----------------------------------------------------------------------------- the job


def main() -> None:
    ap = argparse.ArgumentParser(description="E93 Amendment 2: the table on the new decomposition")
    ap.add_argument("--stage", choices=("control",))
    ap.add_argument("--states")
    ap.add_argument("--map", choices=("ls", "adjoint"))
    ap.add_argument("--extras")
    ap.add_argument("--decomp", type=Path, required=True)
    a = ap.parse_args()
    P = json.loads(PRED.read_text())
    require_filed(PRED)
    if run.STOPPED.is_file():
        sys.exit(f"refusing to start: {run.STOPPED} exists (a check missed; the lead decides)")
    names = a.states.split(",") if a.states else []
    extras = a.extras.split(",") if a.extras else []
    if sum(map(bool, (a.stage, names, extras))) != 1 or bool(names) != bool(a.map) or any(n not in rules.STATES for n in names) or any(x not in P["extras"] for x in extras):
        sys.exit("give one of --stage control, --states ... --map {ls,adjoint}, --extras ...")
    if (names or extras) and not json.loads(CONTROL.read_text() if CONTROL.is_file() else "{}").get("passed"):
        sys.exit("refusing: stage 0 (--stage control) has not passed")
    assert decomp_id(a.decomp) == NEW, f"--decomp is {decomp_id(a.decomp)}, this amendment reads {NEW}"
    OUT.mkdir(parents=True, exist_ok=True)
    run.log(f"[start] amend2 {vars(a)}; config {run.C}")
    rows, seqs = run.build(P)
    with torch.no_grad():
        d = d3.Decomp(a.decomp, run.log)
        if a.stage:
            stage_control(d, rows, seqs, P)
            hard_exit(0)
        for n in extras:
            if sc.load_extra(n) is None:
                read_extra(d, rows, n, P)
        kind, reader = ("state", read_ls) if a.map == "ls" else ("adjoint", read_adjoint)
        took: list[float] = []
        for n in names:
            if sc.load_rec(n, kind) is not None:
                run.log(f"[{n}] {a.map} already written")
                continue
            if time.time() - run.T0 + max(took, default=run.C.first_guess_s) > run.C.limit_s:
                run.log(f"cut short before {n}: {time.time() - run.T0:.0f} s used; resubmit")
                hard_exit(3)
            t0 = time.time()
            reader(d, rows, seqs, n, P)
            took.append(time.time() - t0)
            t = sc.score(P)[a.map]["states"].get(n, {})
            run.log(f"[{n}] {a.map} done in {took[-1]:.0f} s, {d3.peaks()}; verdict {t.get('verdict')}")
    run.log(f"[done] amend2 {names or extras}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)
