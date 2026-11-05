"""E92 (qwen_comparators), 2026-11-05: three cheap non-VPD readings on saved Qwen3-1.7B states, one row of them beside each D3
row: update size, the unchanged model's rank of the trained answer, one steering vector. Plan:
reports/plans/2026-11-05_plan_e92-comparators.md. Rules: results/predictions.json; the readings: `commute.qwen_comparators`.

  --states anchor1_s0:100,R_s0:10   arm and step, E91's names (`qwen_size_grid/run.py`, ARMS); a matched anchor 2 state of
                                    E91's second stage is found under `qwen_size_grid/results/large/dW/a2_s<seed>/`.

The controls run first and alone, in the first job only (results/controls.json): E91's own reads of anchor 1's zero state and
step 100 against its record, then W0 and both states through this path. A miss stops the job (results/stopped.json, every run
refuses until it is removed). Per state, cheapest first: size, rank, steering. One state's numbers go to results/states/ and
the log as it ends, its positions to results/large/; a state already written is skipped, so a lost job resumes. Lost before
controls.json is written, the controls run again whole.
Exit 0 done; 1 a stop or a refusal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from commute import qwen_comparators as qc
from commute.jobs import hard_exit, load_experiment
from commute.qwen_sft import load, plus, set_weights, sft_seq

E91: Any = load_experiment("qwen_size_grid", "run")  # ARMS, large, stored, read_state, control_miss; its Job is E45's
JOB, DEV = E91.JOB, E91.DEV

HERE = Path(__file__).resolve().parent
RESULTS, EXPERIMENTS = HERE / "results", HERE.parent
PRED, STOPPED, CONTROLS = RESULTS / "predictions.json", RESULTS / "stopped.json", RESULTS / "controls.json"
STATES, LARGE = RESULTS / "states", RESULTS / "large"
A1_ITEMS, FORMS = JOB.LARGE / "P" / "a1_rows.json", EXPERIMENTS / "qwen_tofu_forms" / "results" / "forms.jsonl"
MATCHED = {"anchor2_s0": "a2_s42", "anchor2_s142": "a2_s142"}  # E91's second stage (matched.py) names seeds by number
ZERO, CONTROL = ("anchor1_s0", 1), ("anchor1_s0", 100)
ROW_LEN, MATHS_ROWS, STEER_SEED = 256, 128, 92_000  # E68's row_len; E75's maths rows; the random direction's draw


class Lab(JOB.Job):  # E45's tokenizer, problems and model; the log is this experiment's
    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        with (RESULTS / "run.log").open("a") as f:
            f.write(line + "\n")


def stop(lab: Lab, why: str, detail: Any) -> None:
    STOPPED.write_text(json.dumps({"stopped": why, "detail": detail}, indent=1, default=float) + "\n")
    lab.log(f"STOP: {why}: {json.dumps(detail, default=float)}")
    hard_exit(1)


# ----------------------------------------------------------------------------- the rows


def a1_rows(lab: Lab) -> qc.Rows:  # moved to commute.qwen_comparators on 2026-11-05 (E94)
    return qc.a1_rows(lab.tok, lab.sets["test"], json.loads(A1_ITEMS.read_text())["items"])


def tofu_rows(lab: Lab) -> qc.Rows:  # moved to commute.qwen_comparators on 2026-11-05 (E94)
    return qc.tofu_rows(lab.tok, lab.end_id, {rec["row"] for rec in map(json.loads, FORMS.read_text().splitlines()) if rec["kept"]})


def gold_rows(lab: Lab) -> qc.Rows:
    """GSM8K test 0-255 with the gold solutions, packed and cut as the maths rows. A departure: the D3 maths rows hold W0's
    own greedy answers (`qwen_sft/run.py` stage_G), so W0 ranks them first by construction, and the GSM8K arms train no fixed token."""
    seqs = [sft_seq(lab.tok, it["question"], it["gold"], lab.end_id, 512) for it in lab.sets["test"][:256]]
    rows = qc.rows_of("gold", seqs, 512, ROW_LEN)
    return qc.Rows(rows.name, rows.tokens[:MATHS_ROWS], rows.answer[:MATHS_ROWS], rows.first[:MATHS_ROWS])


BUILD = {"anchor1": a1_rows, "anchor2": tofu_rows, "R": gold_rows, "N": gold_rows, "F": gold_rows}


def path_of(arm: str, step: int) -> Path:
    if step in E91.ARMS.get(arm, {}):
        return E91.ARMS[arm][step]
    if arm in MATCHED:
        return E91.large("qwen_size_grid", "dW", MATCHED[arm], f"step_{step}.safetensors")
    sys.exit(f"{arm}:{step} is not a state of E91's grid")


# ----------------------------------------------------------------------------- one state


class Reader:
    """The model at W0, each row set and W0's read of it, kept for the job."""

    def __init__(self, lab: Lab) -> None:
        self.lab, self.model, self.rows, self.w0 = lab, lab.fp32(), {}, {}

    def base(self, arm: str) -> tuple[qc.Rows, qc.Read]:
        name = BUILD[arm.split("_")[0]].__name__
        if name not in self.rows:
            rows = self.rows[name] = BUILD[arm.split("_")[0]](self.lab)
            self.w0[name] = qc.read(self.model, rows)
            self.finite(self.w0[name], f"W0 {name}")
            w0_summary = {"rows": rows.name, "shape": list(rows.tokens.shape), **qc.summary(self.w0[name])}
            self.lab.log(f"[W0 {rows.name}] {json.dumps(w0_summary)}")
            (RESULTS / f"W0_{rows.name}.json").write_text(json.dumps(w0_summary, indent=1) + "\n")
            save(LARGE / f"W0_{rows.name}.safetensors", self.w0[name])
        return self.rows[name], self.w0[name]

    def finite(self, r: qc.Read, what: str) -> None:
        if not bool(r.logp.isfinite().all()):
            stop(self.lab, "non-finite log-probability", what)

    def state(self, arm: str, step: int) -> tuple[dict[str, Any], qc.Read, qc.Read]:
        lab, t0, path = self.lab, time.time(), path_of(arm, step)
        dW = load(path, DEV)
        size = qc.update_size(dW, lab.W0)
        out: dict[str, Any] = {"arm": arm, "step": step, "file": str(path.relative_to(EXPERIMENTS)), "dtype": str(next(iter(dW.values())).dtype),
                               "size": size, "e91": check_e91(lab, arm, step, size["total"]["norm"])}
        lab.log(f"[size {arm}:{step}] {json.dumps(out['size']['total'])}, E91 {out['e91']}, {time.time() - t0:.0f} s")
        rows, w0 = self.base(arm)
        W = plus(lab.W0, dW)
        del dW
        with set_weights(self.model, W):
            st = qc.read(self.model, rows)
        self.finite(st, f"{arm}:{step}")
        d = st.logp - w0.logp
        f = st.is_first
        out["rows"], out["rank_W0"], out["rank_state"] = rows.name, qc.summary(w0), qc.summary(st)
        out["gain"] = {"all": qc.share(d, d, st.row_of), "first": qc.share(d[f], d[f], st.row_of[f])}
        lab.log(f"[rank {arm}:{step}] {json.dumps({k: out[k] for k in ('rows', 'rank_W0', 'rank_state', 'gain')})}, {time.time() - t0:.0f} s")
        out["steering"] = qc.steer(self.model, rows, w0, st, W, STEER_SEED)
        out["seconds"] = time.time() - t0
        lab.log(f"[steer {arm}:{step}] {json.dumps(out['steering'])}, {out['seconds']:.0f} s, {lab.peaks()}")
        return out, w0, st


def check_e91(lab: Lab, arm: str, step: int, norm: float) -> float | None:
    """The total against E91's stored read of this file, the only definition of the size: a miss stops. No E91 read yet: None."""
    rec = next((r for r in E91.stored() if r["arm"] == arm and r["step"] == step), None)
    if rec is None:
        return None
    tol = json.loads(PRED.read_text())["controls"]["e91_norm_rel"]
    if abs(norm - rec["norm"]) > tol * max(rec["norm"], 1e-30):
        stop(lab, "the size misses E91's", {"arm": arm, "step": step, "here": norm, "e91": rec["norm"]})
    return rec["norm"]


def save(path: Path, r: qc.Read) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"rank": r.rank.int(), "logp": r.logp.float(), "row_of": r.row_of.int(), "is_first": r.is_first.to(torch.uint8)}, str(path))


def state_file(arm: str, step: int) -> Path:
    return STATES / f"{arm}_step{step}.json"


def read_one(rd: Reader, arm: str, step: int) -> tuple[dict[str, Any], qc.Read, qc.Read]:
    out, w0, st = rd.state(arm, step)
    save(LARGE / f"{arm}_step{step}.safetensors", st)
    STATES.mkdir(parents=True, exist_ok=True)
    state_file(arm, step).write_text(json.dumps(out, indent=1) + "\n")
    return out, w0, st


# ----------------------------------------------------------------------------- the controls, first and alone


def controls(rd: Reader, P: dict[str, Any]) -> None:
    """E91's reads and record; the zero state must equal W0 through this path; anchor 1's own update must raise its own tokens."""
    lab, e91_rows, e91_pred = rd.lab, lab_rows(rd.lab), json.loads(E91.PRED.read_text())
    out: dict[str, Any] = {}
    for arm, step in (ZERO, CONTROL):
        r = E91.read_state(rd.lab, e91_rows, arm, step, E91.ARMS[arm][step])
        lab.log(f"[control E91 {arm}:{step}] norm {r['norm']}, KL {r['kl']}")
        out[f"e91_step{step}"] = r
        miss = E91.control_miss(r, e91_pred)
        if miss:
            stop(lab, miss, r)
    rec, w0, st = read_one(rd, *ZERO)
    worst = float((st.logp - w0.logp).abs().max())
    out["zero"] = {"max_logp_diff": worst, "ranks_equal": bool((st.rank == w0.rank).all()), "steering_layer": rec["steering"]["layer"]}
    lab.log(f"[control zero] {json.dumps(out['zero'])}; band {P['controls']['zero_logp_max']}")
    if worst > P["controls"]["zero_logp_max"] or rec["size"]["total"]["norm"] != 0 or rec["steering"]["layer"] is not None:
        stop(lab, "the zero state does not read W0", out["zero"])
    rec = read_one(rd, *CONTROL)[0]
    g = rec["gain"]["all"]
    out["learned"] = {"gain": g["gain"], "gain_se": g["gain_se"], "rank_W0": rec["rank_W0"]["all"], "rank_state": rec["rank_state"]["all"]}
    lab.log(f"[control learned] {json.dumps(out['learned'])}: must be defined and positive; {rec['seconds']:.0f} s a state of these rows")
    if g["share"] is None or g["gain"] <= 0:
        stop(lab, "anchor 1's update does not raise the tokens it was trained toward", out["learned"])
    CONTROLS.write_text(json.dumps(out, indent=1, default=float) + "\n")


def lab_rows(lab: Lab) -> list[torch.Tensor]:
    rows = lab.maths()[: E91.CFG.track_rows]
    assert len(rows) == 16
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="E92: the cheap comparators")
    p.add_argument("--states", required=True, help="arm:step,arm:step (E91's names)")
    a = p.parse_args()
    if STOPPED.is_file():
        sys.exit(f"{STOPPED} exists: read it, then remove it")
    wanted = [(s.split(":")[0], int(s.split(":")[1])) for s in a.states.split(",")]
    for arm, step in wanted:  # a bad list fails before the model loads
        if arm.split("_")[0] not in BUILD or not path_of(arm, step).is_file():
            sys.exit(f"{arm}:{step}: no rows for this arm or no file at {path_of(arm, step)}")
    P = json.loads(PRED.read_text())
    lab = Lab("C", "E92", PRED)
    rd = Reader(lab)
    if not CONTROLS.is_file():
        controls(rd, P)
    for arm, step in wanted:
        if state_file(arm, step).is_file():
            lab.log(f"[{arm}:{step}] already written")
        else:
            read_one(rd, arm, step)
    lab.log(f"[done] {[f'{a}:{s}' for a, s in wanted]}, {lab.peaks()}")
    print("FINISHED", flush=True)
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)
