"""E118 (qwen_comparator_table), 2026-11-06: the comparator cells E92 and E102 left empty beside the Qwen VPD rows, on
states other leads saved, with E92's and E102's code and settings unchanged. No fine-tune, no decomposition. Plan:
reports/plans/2026-11-06_plan_e118-qwen-comparator-table.md; controls: results/predictions.json; the readings:
`qwen_comparators.read_state` (E92: size, rank, steering) and `qwen_neurons.state_cells` + `set_reads` (E102: the
carrying set and its scores). The join with the VPD rows is build.py's, on CPU.

  run.py --part compare [--states u1,...]   E92's three readings on COMPARE's states (default all)
  run.py --part carry   [--states R_s0_10,...]   E102's carrying set on CARRY's states (default all)

Each part runs its controls first and alone (results/controls_<part>.json; skipped once written): C0 the do-nothing
file reads zero, then C1 (compare) or C2 (carry) re-reads anchor 1 seed 0 against E92's or E102's stored numbers. A miss
writes results/stopped.json and every run refuses while it exists. A state's numbers go to results/<part>/<name>.json
and the log as it ends, its per-position tensors to results/large/<part>/; a state already written is skipped, so a lost
or paused job resumes. Exit 0 done; 1 a stop or a refusal; 3 the time budget: resubmit the same command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from commute import known_facts as KF
from commute import qwen_comparators as qc
from commute import qwen_neurons as QN
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed
from commute.qwen_sft import load, plus

E92: Any = load_experiment("qwen_comparators", "run")  # its row builders (a1, tofu, gold) and, through it, E91 and E45's Job
E91 = E92.E91
DEV, GRID = E91.DEV, E91.ARMS
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE, PRED, STOPPED = RESULTS / "large", RESULTS / "predictions.json", RESULTS / "stopped.json"
P: dict[str, Any] = json.loads(PRED.read_text())
S, C = P["settings"], P["controls"]
CS = S["carrying"]
E102R, KAP = EXPERIMENTS / "qwen_carrying" / "results", EXPERIMENTS / "known_answer_pair" / "results"
KAP_DW, LVS_DW = KAP / "large" / "dW", EXPERIMENTS / "label_vs_style" / "results" / "large" / "dW"
ZERO, A1 = GRID["anchor1_s0"][1], GRID["anchor1_s0"][100]  # E92's do-nothing file and its control state
U1 = E91.U1_REGEN / "step_100_fp32.safetensors"  # E93 scored the fp32 file; E91 read the bf16 one


def past(seed: int, step: int) -> Path:  # E91's second stage, read by E94 as a2_s<seed>_past
    return E91.large("qwen_size_grid", "dW", f"a2_s{seed}", f"step_{step}.safetensors")


# name -> (file, rows). Rows: E92's rules, a1 = anchor 1's, tofu = TOFU's pair 0, gold = GSM8K test gold solutions (u1 and
# u2 trained on gold; the RL arms train no fixed token), known / wrong = pair 0 of what each arm trained (the sentences,
# true or with the wrong answer), real = E103's carrying probe (pair 0's bare answers), as the stored known sets.
COMPARE: dict[str, tuple[Path, str]] = {
    "u1": (U1, "gold"), "u2": (GRID["u2"][100], "gold"),
    "known_s42_matched": (KAP_DW / "amend1_matched.safetensors", "known"),
    "known_s142_matched": (KAP_DW / "amend2_s142_matched.safetensors", "known"),
    "known_s42_end": (KAP_DW / "amend1_end.safetensors", "known"),
    "known_s142_end": (KAP_DW / "amend2_s142_end.safetensors", "known"),
    "wrong_s42_matched": (LVS_DW / "WRONG42_matched.safetensors", "wrong"),
    "wrong_s142_matched": (LVS_DW / "WRONG142_matched.safetensors", "wrong"),
}
CARRY: dict[str, tuple[Path, str]] = {
    **{f"{a}_s{s}_{t}": (GRID[f"{a}_s{s}"][t], "gold") for a, s in (("R", 0), ("N", 0), ("F", 0), ("R", 70), ("N", 70)) for t in (10, 100)},
    "a2_s42_past": (past(42, 550), "tofu"), "a2_s142_past": (past(142, 500), "tofu"),
    "u1": COMPARE["u1"], "u2": COMPARE["u2"],
    "known_s42_end": (KAP_DW / "amend1_end.safetensors", "real"),
    "known_s142_end": (KAP_DW / "amend2_s142_end.safetensors", "real"),
}
PARTS = {"compare": COMPARE, "carry": CARRY}
STORED = [E92.RESULTS / "states" / "anchor1_s0_step100.json", E102R / "states" / "a1_s0.json", E102R / "w0.json",
          E102R / "score.json", E102R / "large" / "engaged.safetensors", E102R / "large" / "fineweb_rows.safetensors",
          KAP / "large" / "engaged_real.safetensors", KAP / "forms.json", KAP / "forms_amend1.jsonl", ZERO, A1]


class Lab(E91.JOB.Job):  # E45's tokenizer, problems and model (as E92 and E102); the log is this experiment's
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")


def put(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name + ".tmp").write_text(json.dumps(obj, indent=1, default=float) + "\n")
    path.with_name(path.name + ".tmp").replace(path)


def stop(lab: Lab, why: str, detail: Any) -> None:
    put(STOPPED, {"stopped": why, "detail": detail})
    lab.log(f"STOP: {why}: {json.dumps(detail, default=float)}")
    hard_exit(1)


def check(lab: Lab, out: dict[str, Any], name: str, holds: bool, detail: dict[str, Any]) -> None:
    out.setdefault("checks", {})[name] = {"holds": bool(holds), **detail}
    lab.log(f"[check] {name}: {'ok' if holds else 'MISSED'} {json.dumps(detail, default=float)}")
    if not holds:
        stop(lab, name, detail)


def rel(x: float | None, ref: float) -> float:
    return abs(x - ref) / abs(ref) if x is not None else float("inf")


# ----------------------------------------------------------------------------- rows, W0's reads, engaged values


class Book:
    """Each row set, W0's E92 read of it and its engaged values, built once per job."""

    def __init__(self, lab: Lab, model: Any) -> None:
        self.lab, self.model, self._rows, self._w0, self._eng = lab, model, {}, {}, {}

    def rows(self, name: str) -> qc.Rows:
        if name not in self._rows:
            lab = self.lab
            if name in ("a1", "tofu", "gold"):
                R = {"a1": E92.a1_rows, "tofu": E92.tofu_rows, "gold": E92.gold_rows}[name](lab)
            elif name == "real":
                R = KF.real_rows(lab.tok, lab.end_id, set(json.loads((KAP / "forms.json").read_text())["kept_rows"]))
            else:
                said = KF.sentences(KAP / "forms_amend1.jsonl")
                R = KF.said_rows(name, lab.tok, lab.end_id, said if name == "known" else KF.wrong_said(said))
            self._rows[name] = R
            lab.log(f"[rows {name}] {list(R.tokens.shape)}, {int(R.answer.sum())} answer positions, {int(R.first.sum())} first")
        return self._rows[name]

    def w0(self, name: str) -> qc.Read:
        if name not in self._w0:
            r = self._w0[name] = qc.read(self.model, self.rows(name))
            finite(self.lab, r, f"W0 {name}")
            put(RESULTS / "W0" / f"{name}.json", {"rows": name, "shape": list(self.rows(name).tokens.shape), **qc.summary(r)})
        return self._w0[name]

    def engaged(self, name: str) -> list[float]:
        """E102's engaged values (a1, tofu), E103's (real); the gold rows' read here on E102's corpus sample."""
        if name not in self._eng:
            if name in ("a1", "tofu"):
                e = load_file(str(E102R / "large" / "engaged.safetensors"))[name]
            elif name == "real":
                e = load_file(str(KAP / "large" / "engaged_real.safetensors"))["real"]
            else:
                e = self.gold_engaged()
            self._eng[name] = e.tolist()
        return self._eng[name]

    def gold_engaged(self) -> torch.Tensor:
        path = LARGE / "engaged_gold.safetensors"
        if not path.is_file():
            t0, (a, b) = time.time(), S["quantile_rows"]
            fw = load_file(str(E102R / "large" / "fineweb_rows.safetensors"))["rows"].long()
            sample = QN.sorted_sample(self.model, fw[a:b])
            e = QN.engaged(self.model, self.rows("gold"), sample, CS["batch"])
            del sample
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"gold": e.contiguous()}, str(path))
            self.lab.log(f"[engaged gold] median {float(e.median()):.4f}, {time.time() - t0:.0f} s, {self.lab.peaks()}")
        return load_file(str(path))["gold"]


def finite(lab: Lab, r: qc.Read, what: str) -> None:
    if not bool(r.logp.isfinite().all()):
        stop(lab, "non-finite log-probability", what)


# ----------------------------------------------------------------------------- the controls, first and alone


def control_compare(lab: Lab, model: Any, book: Book) -> None:
    out: dict[str, Any] = {}
    R, w0 = book.rows("a1"), book.w0("a1")
    rec, st = qc.read_state(model, lab.W0, load(ZERO, DEV), R, w0, S["steer_seed"])
    worst = float((st.logp - w0.logp).abs().max())
    check(lab, out, "C0_zero_compare", rec["size"]["total"]["norm"] == 0 and worst <= C["C0_zero"]["zero_logp_max"]
          and rec["steering"]["layer"] is None, {"norm": rec["size"]["total"]["norm"], "max_logp_diff": worst, "layer": rec["steering"]["layer"]})
    T, tol = C["C1_e92"]["targets"], C["C1_e92"]["rel"]
    for name, ref in T["W0_mean_logp"].items():
        got = qc.summary(book.w0(name))["all"]["mean_logp"]
        check(lab, out, f"C1_W0_{name}", rel(got, ref) <= tol, {"here": got, "e92": ref})
    rec, _ = qc.read_state(model, lab.W0, load(A1, DEV), R, w0, S["steer_seed"])
    sv = rec["steering"]
    s, r = (sv[k]["all"]["share"] if sv[k] else None for k in ("steer", "random"))
    detail = {"norm": rec["size"]["total"]["norm"], "gain": rec["gain"]["all"]["gain"], "layer": sv["layer"], "steer_share": s,
              "random_share": r, "targets": T}
    check(lab, out, "C1_e92_a1_s0", rel(detail["norm"], T["norm"]) <= C["C1_e92"]["norm_rel"] and rel(detail["gain"], T["gain_all"]) <= tol
          and sv["layer"] == T["layer"] and s is not None and abs(s - T["steer_share"]) <= T["steer_share_se"]
          and r is not None and abs(r - T["random_share"]) <= T["random_share_se"], detail)
    put(RESULTS / "controls_compare.json", out)


def control_carry(lab: Lab, model: Any, book: Book) -> None:
    out: dict[str, Any] = {}
    a1 = book.rows("a1")
    sub = qc.Rows("a1_16", a1.tokens[:16], a1.answer[:16], a1.first[:16])
    W = plus(lab.W0, load(ZERO, DEV))
    r = QN.rank_state(model, sub, W, CS["batch"])
    nums = QN.grid(model, sub, W, torch.arange(QN.N), [QN.N], CS["batch"])
    dev, band = float((r["T1"] - r["T0"]).abs().max()), C["C0_zero"]["zero_logp_max"]
    n_set = len(QN.top_set(r["score"].sum(0), CS["K"]))
    check(lab, out, "C0_zero_carry", dev <= band and abs(nums[QN.N]) <= band and n_set == 0,
          {"max_T1_T0": dev, "num_N": nums[QN.N], "set_size": n_set})
    W = plus(lab.W0, load(A1, DEV))
    rec, _ = QN.state_cells(model, a1, W, CS["K"], CS["ks"], CS["batch"], CS["carry_floor"])
    del W
    T, tol = C["C2_e102"]["targets"], C["C2_e102"]["rel"]
    same = QN.overlap(rec["carry"], json.loads((E102R / "states" / "a1_s0.json").read_text())["carry"])
    scores = QN.set_reads(rec["carry"], book.engaged("a1"), contrast(), CS["n_draws"], torch.Generator().manual_seed(CS["seed"] + 1 + 2))
    detail = {"gain": rec["gain"]["gain"], "nums_N": rec["nums"][str(QN.N)], "set_overlap": same, "k_half": rec["k_half"],
              "neuron_share": rec["neuron_share"], "prompt_overlap": scores["prompt_overlap"], "engaged": scores["engaged"], "targets": T}
    check(lab, out, "C2_e102_a1_s0", rel(detail["gain"], T["gain"]) <= tol and rel(detail["nums_N"], T["nums_N"]) <= tol
          and same >= C["C2_e102"]["min_set_overlap"], detail)
    put(RESULTS / "controls_carry.json", out)


def contrast() -> list[int]:
    return json.loads((E102R / "w0.json").read_text())["contrast"]["sets"]["all"]


# ----------------------------------------------------------------------------- one state


def compare_one(lab: Lab, model: Any, book: Book, name: str, _i: int) -> dict[str, Any]:
    path, rows = COMPARE[name]
    dW = load(path, DEV)
    out: dict[str, Any] = {"name": name, "file": str(path.relative_to(EXPERIMENTS)), "dtype": str(next(iter(dW.values())).dtype)}
    rec, st = qc.read_state(model, lab.W0, dW, book.rows(rows), book.w0(rows), S["steer_seed"])
    del dW
    finite(lab, st, name)
    E92.save(LARGE / "compare" / f"{name}.safetensors", st)
    return out | rec


def carry_one(lab: Lab, model: Any, book: Book, name: str, i: int) -> dict[str, Any]:
    path, rows = CARRY[name]
    W = plus(lab.W0, load(path, DEV))
    rec, r = QN.state_cells(model, book.rows(rows), W, CS["K"], CS["ks"], CS["batch"], CS["carry_floor"])
    del W
    gen = torch.Generator().manual_seed(CS["seed"] + 1 + i)
    (LARGE / "carry").mkdir(parents=True, exist_ok=True)
    save_file({k: r[k].contiguous() for k in ("T0", "T1", "row_of", "score")}, str(LARGE / "carry" / f"{name}.safetensors"))
    return {"name": name, "file": str(path.relative_to(EXPERIMENTS))} | rec | QN.set_reads(rec["carry"], book.engaged(rows), contrast(), CS["n_draws"], gen)


READ = {"compare": compare_one, "carry": carry_one}
CONTROL = {"compare": control_compare, "carry": control_carry}
FIRST_GUESS_S = {"compare": 150.0, "carry": 120.0}  # E92's TOFU state 105-111 s, E102's 82 s, both with the load (run.logs)


def states(lab: Lab, model: Any, book: Book, part: str, names: list[str], t_start: float) -> None:
    unit = FIRST_GUESS_S[part]
    order = list(PARTS[part])
    for name in names:
        f = RESULTS / part / f"{name}.json"
        if f.is_file():
            lab.log(f"[{part} {name}] already written")
            continue
        if time.time() - t_start + unit > S["budget_s"]:
            lab.log(f"BUDGET: {time.time() - t_start:.0f} s used, next unit about {unit:.0f} s; resubmit the same command")
            hard_exit(3)
        t0 = time.time()
        rec = READ[part](lab, model, book, name, order.index(name))
        rec["seconds"] = time.time() - t0
        unit = max(unit, rec["seconds"])
        put(f, rec)
        lab.log(f"[{part} {name}] {json.dumps({k: v for k, v in rec.items() if 'draws' not in k and k != 'carry'}, default=float)}, {lab.peaks()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="E118: E92's and E102's comparators on the states they left out")
    ap.add_argument("--part", required=True, choices=list(PARTS))
    ap.add_argument("--states", help="comma-separated names of the part (default: all)")
    a = ap.parse_args()
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    require_filed(PRED)
    names = a.states.split(",") if a.states else list(PARTS[a.part])
    bad = [n for n in names if n not in PARTS[a.part]] + [str(p) for p in (*STORED, *(PARTS[a.part][n][0] for n in names if n in PARTS[a.part])) if not p.is_file()]
    if bad:
        sys.exit(f"unknown state or missing file: {bad}")  # before the model loads
    lab = Lab(a.part, "E118", PRED)
    t_start, model = time.time(), lab.fp32()
    model.eval()
    book = Book(lab, model)
    if not (RESULTS / f"controls_{a.part}.json").is_file():
        CONTROL[a.part](lab, model, book)
    states(lab, model, book, a.part, names, t_start)
    lab.log(f"[done] {a.part}: {names}, {lab.peaks()}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)
