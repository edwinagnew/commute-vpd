"""E61 (qwen_recipe): five $0 checks from reports/reviews/2026-11-02_review_qwen-pipeline-audit.md. Rules and expected
values: results/predictions_checks.json. One L40 debug job each (cluster/qwen_recipe_checks.sbatch, which sets
PYTHONPATH for E57's lab.py), cheapest first: tiny (finding 4), k0 (5), sizes (2), gen (6), k10 --state (1). Each writes
results/checks_<stage>.json as it goes (k10: checks_k10_<state>.json, verdict in checks_k10.json), logs every number to
results/run.log and resumes from what is written. Exit 1: a stop (results/checks_stopped.json). Exit 3: the 25-min
budget, resubmit. X is saved in fp32 under results/large/checks/, 5.6 GB per state (δW's shape).
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from lab import CFG, DEV, E45, F64, REF, Lab, kl, null_seed, pinned, write_json
from lab import LARGE as E57_LARGE
from lab import RESULTS as E57
from safetensors.torch import load_file, save_file
from torch import Tensor
from transformers import AutoTokenizer, GenerationConfig

from commute import qwen_eval, qwen_maps, qwen_sets, qwen_sft, split
from commute import qwen_reading as qr
from commute.qwen_sft import linears
from commute.qwen_vpd import SITES, importances, masked_logits
from commute.vpd import kl_per_position

HERE = Path(__file__).resolve().parent
RESULTS, CHECKS = HERE / "results", HERE / "results" / "large" / "checks"
PRED, STOPPED = RESULTS / "predictions_checks.json", RESULTS / "checks_stopped.json"
BUDGET_S = 1500.0
PARK_OVER = 400  # tokens: E57's log has 40.65 GiB held plus 1.45 per 256 tokens, so 43 GiB here; the L40's rule is 44
SIZES = ("e40", "gold_s0", "a2long_s0", "a2long_s1", "gauss", "inspan")
K10_STATES = ("gold_s0", "e40", "a2long_s0", "gauss")
OWN = E45 / "large" / "dW" / "own_s0" / "step_100.safetensors"

# E57's run.py by path, for Text, inspan and total: this folder has its own run.py, so `import run` would find that.
_spec = importlib.util.spec_from_file_location("e57_run", E57.parent / "run.py")
assert _spec is not None and _spec.loader is not None
RUN: Any = importlib.util.module_from_spec(_spec)
sys.modules["e57_run"] = RUN
_spec.loader.exec_module(RUN)


class Checks(Lab):
    """E57's job, with its log, files and stops here: E57's results are only read."""

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")

    def get(self, name: str) -> dict[str, Any]:
        p = RESULTS / f"{name}.json"
        return json.loads(p.read_text()) if p.is_file() else {}

    def put(self, name: str, d: dict[str, Any]) -> None:
        write_json(RESULTS / f"{name}.json", d)
        self.log(f"[{name}] {json.dumps(d, default=float)}")

    def stop(self, name: str, why: str) -> None:
        write_json(STOPPED, {"name": name, "why": why, "stage": self.stage})
        self.log(f"STOP {name}: {why}")
        sys.exit(f"STOP {name}: {why}")

    def budget(self, need_s: float) -> None:
        if time.time() - self.t0 + need_s > BUDGET_S:
            self.log(f"BUDGET: {need_s:.0f} s more would pass {BUDGET_S:.0f} s; resubmit the same command")
            sys.exit(3)


def rules() -> dict[str, Any]:
    return json.loads(PRED.read_text())["rules"]


def e57_record(name: str) -> dict[str, Any]:
    """E57's record of a state: C_<state>.json, or its entry in C0.json for the two random changes."""
    if name in ("gauss", "inspan"):
        return json.loads((E57 / "C0.json").read_text())[name]
    return json.loads((E57 / f"C_{name}.json").read_text())


def state_dw(lab: Lab, name: str) -> dict[str, Tensor]:
    if name == "inspan":
        return RUN.inspan(lab)
    if name != "gauss":
        return lab.load_dw(name)
    e40, sig = lab.load_dw("e40"), lab.sigma()  # as stage_C0, at its scale; Σ needs Γ parked or not yet loaded
    b = json.loads((E57 / "C0.json").read_text())["search"]["b"]
    G = {s: pinned(split.random_like(e40[s].to(DEV, F64), sig[s].to(DEV), null_seed(0, s)).float().cpu()) for s in SITES}
    for g in G.values():
        g.mul_(b)
    return G


# ----------------------------------------------------------------------------- the filed rules


def tiny_verdict(gap: float, switched_on: int, R: dict[str, Any]) -> dict[str, Any]:
    v = qr.ok(gap) and qr.r9(gap) <= R["gap_max"] and switched_on < R["switched_on_max"]
    return {"no_floor": v, "reads": "Γ's bf16 cast has no floor worth reading" if v
            else "the fp32 Γ comparison must run (not written here)"}


def k0_verdict(unpacked: float, packed: float, R: dict[str, Any]) -> dict[str, Any]:
    v = qr.within(unpacked, packed, R["factor"])
    return {"same": v, "ratio": qr.div(unpacked, packed),
            "reads": "row format does not change the K0 reading" if v else "a card to the user"}


def paired(base: dict[int, Any], new: dict[int, Any], drop: bool) -> dict[str, Any]:
    """strict(new) - strict(base) per problem, mean and se; `drop`: only problems fitting the window in both."""
    d = [int(new[i]["strict"]) - int(base[i]["strict"]) for i in base
         if i in new and (not drop or (base[i]["fits"] and new[i]["fits"]))]
    return {"n": len(d), "gain": statistics.mean(d) if d else qr.NAN,
            "se": statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else qr.NAN}


def gen_verdict(p: dict[str, Any], R: dict[str, Any]) -> dict[str, Any]:
    g, se = p["gain"], p["se"]
    v = qr.ok(g) and qr.ok(se) and qr.r9(g) >= R["gain_min"] and qr.r9(g - R["se_multiple"] * se) >= 0
    return {"survives": v, "reads": f"the own run's gain {'survives' if v else 'misses the rule'} on fp32 weights"}


def same_order(a: dict[str, float], b: dict[str, float]) -> bool:
    """Every pair has the same sign of difference at 9 places: a tie in one list only is a different ordering."""
    sign = lambda x: (qr.r9(x) > 0) - (qr.r9(x) < 0)
    return all(sign(a[x] - a[y]) == sign(b[x] - b[y]) for x in a for y in a if x < y)


def k10_verdict(recs: dict[str, dict[str, Any]], R: dict[str, Any]) -> dict[str, Any]:
    """rules.k10: `waiting`, `no reading` (a state unscored on either path), `stands` or `superseded`."""
    missing = [n for n, r in recs.items() if "share" not in r]
    if missing:
        return {"status": "waiting", "missing": missing}
    df, dt = ({n: r["share"][k]["direct"] for n, r in recs.items()} for k in ("file", "true"))
    out = {"direct_file": df, "direct_true": dt, "gap": {n: dt[n] - df[n] for n in recs}}
    bad = [n for n, r in recs.items() if {r["share"]["file"]["status"], r["share"]["true"]["status"]} != {"scored"}]
    if bad:
        return out | {"status": "no reading", "unscored": bad, "reads": "a card to the user; neither branch fires"}
    near, order = all(qr.r9(abs(dt[n] - df[n])) <= R["tol"] for n in recs), same_order(df, dt)
    return out | {"within_tol": near, "same_order": order} | (
        {"status": "stands", "reads": "the mask path does not change the reading; E57's direct shares stand"}
        if near and order else
        {"status": "superseded", "reads": "E57's direct shares are superseded by direct_true, read on all 128 rows"})


# ----------------------------------------------------------------------------- tiny, k0


def stage_tiny(lab: Lab) -> None:
    if lab.get("checks_tiny"):
        return
    R, rows = rules()["tiny"], lab.maths()
    lab.set_weights("FT", lab.load_dw("e40"), R["scale"])
    c0, c1 = lab.counts(), lab.counts()
    m = {k: v / rows.numel() for k, v in RUN.total(lab.cells(rows, c0=c0, c1=c1, tag="E40 x 1e-3")).items()}
    w0 = load_file(str(E57_LARGE / "A/W0_maths.safetensors"))["active"]
    differ = int((c0.active.cpu() != w0).sum())
    lab.check("tiny_W0", differ == 0 and qr.close(m["K0"], REF["K0"], CFG.rep_tol), {"K0": m["K0"], "g0_differ": differ})
    on, e40 = int((c1.on() & ~c0.on()).sum()), e57_record("e40")
    lab.put("checks_tiny", {"means": m, "K01_minus_K0": m["K01"] - m["K0"], "switched_on": on,
                            "switched_off": int((c0.on() & ~c1.on()).sum()), "E40_switched_on": e40["S"],
                            "E40_K01_minus_K0": e40["maths"]["means"]["K01"] - e40["maths"]["means"]["K0"],
                            "verdict": tiny_verdict(m["K01"] - m["K0"], on, R)})


def k0_row(lab: Lab, ids: list[int]) -> Tensor:
    """KL(W0 ‖ masked W0) per position of one row, at any length: a long row's gates run with U on the host (they read
    V only, as stage_B2), its masked forward with Γ on the host."""
    t, comps, long = torch.tensor([ids], device=DEV), lab.comps, len(ids) > PARK_OVER
    if long:
        comps.U = {k: v.cpu() for k, v in comps.U.items()}
        torch.cuda.empty_cache()
    logits, g = importances(lab.model, t, lab.gamma, comps)
    if long:
        lab.park_gamma()
        comps.U = {k: v.to(DEV) for k, v in comps.U.items()}
    return kl_per_position(logits, masked_logits(lab.model, t, lab.comps, g))[0].double().cpu()


def stage_k0(lab: Lab) -> None:
    out = lab.get("checks_k0")
    if "verdict" in out:
        return
    p16 = sum(float(k0_row(lab, r.tolist()).sum()) for r in lab.maths(CFG.n16)) / (CFG.n16 * CFG.row_len)
    lab.check("k0_packed16", qr.close(p16, REF["K0_16"], CFG.rep_tol), {"K0_16": p16, "ref": REF["K0_16"]})
    k256, k512 = (k0_row(lab, lab.maths(1, n)[0].tolist()) for n in (CFG.row_len, 2 * CFG.row_len))
    out = {"packed16": p16, "packed_on_file": REF["K0"], "row0": {
        "k0_256": float(k256.mean()), "k0_512": float(k512.mean()), "k0_512_first256": float(k512[: CFG.row_len].mean())}}
    lab.put("checks_k0", out)
    plain = RUN.Text().plain
    sums, lens = [float(k0_row(lab, r).sum()) for r in plain], [len(r) for r in plain]
    out["unpacked"] = {"k0": sum(sums) / sum(lens), "positions": sum(lens), "per_row_sums": sums, "per_row_len": lens}
    lab.put("checks_k0", out | {"verdict": k0_verdict(out["unpacked"]["k0"], REF["K0"], rules()["k0"])})


# ----------------------------------------------------------------------------- sizes


def kl_both(lab: Lab, key: str, rows: Tensor) -> tuple[list[float], list[float]]:
    """Per-row unmasked KL sums, KL(W0 ‖ key) and KL(key ‖ W0), one weight copy per row (the order alternates)."""
    a, b = [], []
    for r in rows:
        t, out = r[None].to(DEV), {}
        for k in ((key, "W0") if lab.weights.now == key else ("W0", key)):
            lab.use(k)
            out[k] = lab.model(t, use_cache=False).logits
        a.append(kl(out["W0"], out[key]))
        b.append(kl(out[key], out["W0"]))
        del out
    lab.use("W0")
    return a, b


def stage_sizes(lab: Checks) -> None:
    note = "E57's size K00 - K0 and these two KLs are all reported from now on; this table chooses no axis."
    out = lab.get("checks_sizes") or {"note": note, "states": {}}
    rows, n16 = lab.maths(), CFG.n16 * CFG.row_len
    ref16 = {"e40": REF["kl_e40_16"], "inspan": REF["kl_inspan_16"],
             "gauss": json.loads((E57 / "C0.json").read_text())["search"]["kl"]}
    for name in (n for n in SIZES if n not in out["states"]):
        lab.budget(400)
        lab.set_weights("state", state_dw(lab, name))
        a, b = kl_both(lab, "state", rows[: CFG.n16])
        x16 = sum(a) / n16
        repro = qr.close(x16, ref16[name], CFG.rep_tol) if name in ref16 else None
        if name == "e40":  # the control: a miss stops; the Gaussian's and the alive pieces' are recorded
            lab.check("sizes_E40_16", bool(repro), {"kl_16": x16, "ref": ref16[name]})
        a2, b2 = kl_both(lab, "state", rows[CFG.n16 :])
        lab.drop_set("state")
        m = e57_record(name)["maths"]["means"]
        out["states"][name] = {"KL_W0_state": sum(a + a2) / rows.numel(), "KL_state_W0": sum(b + b2) / rows.numel(),
                               "K00_minus_K0": m["K00"] - m["K0"], "KL_W0_state_16": x16, "reproduces_16": repro}
        lab.put("checks_sizes", out)


# ----------------------------------------------------------------------------- gen


def answers(path: Path) -> dict[int, Any]:
    recs = (json.loads(x) for x in path.read_text().splitlines()) if path.is_file() else ()
    return {r["idx"]: r for r in recs}


def generate(lab: Checks, tok: Any, items: list[Any], key: str, batch: int, e45: dict[int, Any]) -> dict[int, Any]:
    """qwen_eval.score one batch at a time, the batches its own loop would cut, so a resume loses at most one."""
    sink, tmp, slowest = CHECKS / f"gen_{key}.jsonl", CHECKS / f"gen_{key}.batch", 60.0
    done = answers(sink)
    W0 = {s: x.weight.data for s, x in linears(lab.model).items()}
    W = None if key == "W0" else qwen_sft.plus(W0, qwen_sft.load(OWN, DEV))
    with qwen_sft.set_weights(lab.model, W):
        for s in range(len(done), len(items), batch):
            lab.budget(slowest)
            t0 = time.time()
            recs = qwen_eval.score(tok, lab.model, items[s : s + batch], math.inf, tmp, f"{key} {s}", batch)
            gap = sum(r["prompt_tokens"] != e45[r["idx"]]["prompt_tokens"] for r in recs)
            lab.check(f"gen_prompts_{key}_{s}", gap == 0, {"prompt_lengths_differ_from_E45": gap})
            with sink.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in recs)
            done |= {r["idx"]: r for r in recs}
            slowest = max(slowest, 1.5 * (time.time() - t0))
    return done


def stage_gen(lab: Checks) -> None:
    R, out = rules()["gen"], lab.get("checks_gen")
    if "verdict" in out:
        return
    CHECKS.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV)
    tok.padding_side = "left"
    items, eos = qwen_sets.load()["test"], lab.model.generation_config.eos_token_id
    lab.model.generation_config = GenerationConfig(eos_token_id=eos, pad_token_id=tok.pad_token_id)  # as `load_model`
    e45_w0, e45_own = answers(E45 / "large/W0_test.jsonl"), answers(E45 / "large/E/own_s0/step_100_test.jsonl")
    w0 = generate(lab, tok, items, "W0", R["batch"], e45_w0)
    out["W0"] = qwen_eval.summary(list(w0.values()))
    lab.put("checks_gen", out)
    own = generate(lab, tok, items, "own_s0", R["batch"], e45_w0)
    ids = [i for i in w0 if i in e45_w0]
    out |= {"own_s0_sha256": qwen_sft.sha256(OWN), "own_s0": qwen_eval.summary(list(own.values())),
            "paired": {"drop": paired(w0, own, True), "all": paired(w0, own, False)},
            "bf16_on_file": {"drop": paired(e45_w0, e45_own, True), "all": paired(e45_w0, e45_own, False)},
            "W0_fp32_vs_bf16": {
                "n": len(ids), "answers_differ": sum(w0[i]["response"] != e45_w0[i]["response"] for i in ids),
                "grades_flip": sum(w0[i]["strict"] != e45_w0[i]["strict"] for i in ids), "card_gap_on_file": 0.011,
                "fp32_minus_bf16": {"drop": paired(e45_w0, w0, True), "all": paired(e45_w0, w0, False)}}}
    lab.put("checks_gen", out | {"verdict": gen_verdict(out["paired"]["drop"], R)})


# ----------------------------------------------------------------------------- k10


def split_x(lab: Checks, name: str, dW: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """The fp32 X of E57's split, saved in fp32, so a resume reads the same X the first pass used. A bf16 copy (the four
    k10 files before 2026-11-02) is refused: X V and Xᵀ U amplify X's rounding (qwen_maps.py)."""
    path = CHECKS / f"{name}_X.safetensors"
    if path.is_file():
        X = load_file(str(path))
        if any(x.dtype != torch.float32 for x in X.values()):
            sys.exit(f"refusing: {path} is not fp32")
        return X, {"resumed": True}
    lab.budget(700)
    sp = lab.split(dW, nulls=False, halves=False)
    tmp = path.with_name(path.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    save_file(sp["X"], str(tmp))
    tmp.replace(path)
    return sp["X"], {"stored_error": sp["stored_error"], "split_s": sp["s"], "file_bytes": path.stat().st_size}


def true_kl(lab: Lab, rows: Tensor, refs: list[Tensor], key: str, change: Any) -> list[float]:
    """Per-row KL(FT's clean ‖ masked) at weights `key`, masks through the original subcomponents: every mask is computed
    and held BEFORE the context that applies `change`, and that context calls no `importances`, so no mask here can see
    V'. Masks wait on the host, about 1.2 GB a row."""
    lab.use(key)
    masks = [{s: v.cpu() for s, v in importances(lab.model, r[None].to(DEV), lab.gamma, lab.comps)[1].items()}
             for r in rows]
    keep = {s: x.weight.data for s, x in linears(lab.model).items()}
    out = []
    with nullcontext() if change is None else qwen_maps.applied(lab.model, lab.comps, lab.alive, change, keep):
        for r, g, ref in zip(rows, masks, refs, strict=True):
            m = masked_logits(lab.model, r[None].to(DEV), lab.comps, {s: v.to(DEV) for s, v in g.items()})
            out.append(kl(ref.to(DEV), m))
            del m
            lab.gpu_rule(f"true masks {key}")
    lab.use("W0")
    return out


def k10_read(lab: Checks, name: str, key: str, out: dict[str, Any]) -> None:
    """Base control, split, K10_file's control, the true pass; each written. The Gaussian's Σ comes before Γ loads."""
    rows, n = lab.maths(CFG.n16), CFG.n16 * CFG.row_len
    saved = {k: sum(v[: CFG.n16]) for k, v in e57_record(name)["rows"]["maths"].items()}
    dW = state_dw(lab, name)
    lab.set_weights("FT", dW)
    if "base" not in out:
        out["base"] = RUN.total(lab.cells(rows, tag=f"{name} 16 rows"))
        lab.check(f"k10_base_{name}", all(qr.close(out["base"][k], saved[k], CFG.rep_tol) for k in ("K0", "K00", "K01")),
                  {"this_job": out["base"], "file": saved})
        lab.put(key, out)
    X, rec = split_x(lab, name, dW)
    out["X"] = out.get("X", {}) | rec
    lab.put(key, out)
    ch = lab.change(dW, X)
    if "file" not in out:
        lab.budget(120)
        out["file"] = RUN.total(lab.cells(rows, ch, tag=f"{name} 16 rows, split, masks through V'"))
        lab.check(f"k10_control_{name}", qr.close(out["file"]["K10"], saved["K10"], CFG.rep_tol),
                  {"K10_file": out["file"]["K10"], "saved": saved["K10"]})
        lab.put(key, out)
    lab.budget(300)
    lab.use("FT")
    refs = [lab.model(r[None].to(DEV), use_cache=False).logits.cpu() for r in rows]
    k00 = true_kl(lab, rows, refs, "W0", None)
    lab.check(f"k10_K00_equals_the_files_{name}",
              all(qr.close(sum(k00), x, CFG.id_tol) for x in (out["base"]["K00"], saved["K00"])),
              {"K00_this_pass": sum(k00), "cells": out["base"]["K00"], "file": saved["K00"]})
    k10, k11 = true_kl(lab, rows, refs, "W0", ch), true_kl(lab, rows, refs, "FT", ch)
    out["true"] = {"K00": sum(k00), "K10": sum(k10), "K11": sum(k11), "per_row": {"K10": k10, "K11": k11}}
    out["share"] = {"file": qr.shares(out["base"] | out["file"], n), "true": qr.shares(out["base"] | out["true"], n)}
    lab.put(key, out)


def stage_k10(lab: Checks, name: str) -> None:
    key = f"checks_k10_{name}"
    out = lab.get(key)
    if "share" not in out:
        k10_read(lab, name, key, out)
    lab.put("checks_k10", k10_verdict({n: lab.get(f"checks_k10_{n}") for n in K10_STATES}, rules()["k10"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("tiny", "k0", "sizes", "gen", "k10"))
    ap.add_argument("--state", choices=K10_STATES, help="k10: the state of this job")
    ap.add_argument("--decomp", default=os.environ.get("COMMUTE_QWEN_DIR", ""))
    ap.add_argument("--e40", default=str(Path.home() / "commute-data/results-large/qwen_sft/nowarmup_regen"))
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"refusing to start: {PRED} is missing; rules are filed before any run")
    if STOPPED.is_file():
        sys.exit(f"refusing to start: {STOPPED} exists (a stop fired; the lead decides)")
    if a.stage == "k10" and a.state is None:
        sys.exit("k10 reads one state per job: --state")
    lab = Checks(a.stage, a.e40, a.decomp)
    with torch.no_grad():
        if a.stage == "k10":
            stage_k10(lab, a.state)
        else:
            {"tiny": stage_tiny, "k0": stage_k0, "sizes": stage_sizes, "gen": stage_gen}[a.stage](lab)
    lab.log(f"[{a.stage} done] {lab.peaks()}")


if __name__ == "__main__":
    main()
