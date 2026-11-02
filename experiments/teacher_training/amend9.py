"""teacher_training E34 Amendment 9 (2026-11-02): does a student's weight change point along what the subcomponents could
express of its fine-tune's update? Plan: reports/plans/2026-11-02_plan_e34-amend9-student-move.md. Rules, leans and expected
values: results/predictions_amend9.json (the run refuses without it, or if CM differs).

Tensor arithmetic on saved states, fp64, CPU, no model and no forward. The inside part of a fine-tune's dW is dW minus its
least-squares remainder O(dW), per matrix against W0's alive (U, V) (split.py). Every inner product is over the 24 matrices
together (E12 Amendment 2's change_stats). Controls first and alone, on each of the six fine-tunes, before any student;
each student's reads are written as it finishes. E42's students are left out: their fine-tunes were regenerated in-job and
not saved. A stop is `stopped` in the file, exit 1. Results: results/amend9.json.

Usage (repo root): PYTHONPATH=experiments/new_vs_existing ~/commute-agents/fleet/cpu-run -c 4 --mem 16G -t 00:45:00 --wait 1800 \
    python experiments/teacher_training/amend9.py > ~/commute-data/results-large/teacher_training/run_amend9.log 2>&1
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from run import dump, halt, log
from torch import Tensor

from commute import split
from commute.keydep import sha256
from commute.saved_states import offsets_state

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT = RESULTS / "predictions_amend9.json", RESULTS / "amend9.json"
A8C = RESULTS / "large" / "amend8" / "amend8c"
E12, E46 = HERE.parent / "new_vs_existing" / "results", HERE.parent / "regating_table" / "results" / "large"
TABLES = {"rand32": "rand32b", "rand32b": "rand32"}  # E12 Amendment 2's two map tables; each is the other's wrong-table reference
E12_STUDENTS = ("rand32_s0", "rand32_s1", "rand32b_s0", "rand32b_s0_o1", "rand32b_s1", "rand32b_s1_o1")
REAL_8C, NULL_8C, T_REAL, T_NULL = ("real_o0", "real_o1"), ("null_o0", "null_o1"), "real_teacher", "null_teacher"
_spec = importlib.util.spec_from_file_location("e12_amend2", HERE.parent / "new_vs_existing" / "amend2.py")  # teacher_training has its own amend2
assert _spec is not None and _spec.loader is not None
_e12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e12)
change_stats, agree = _e12.change_stats, _e12.agree


@dataclass(frozen=True)
class Amend9:
    tol: float = 1e-6  # a singular value below tol x max counts as zero: split.bases' default, every split so far
    ctrl_tol: float = 1e-6  # a reading equal by construction: E12 Amendment 2's zero_tol
    read_tol: float = 1e-3  # a cosine rebuilt from bf16 against fp32: predictions_amend8c's filed "within about 1e-3"
    half: float = 0.5  # cos^2 at or above it, most of the student's squared length: Amendment 8c's `share`, E12 Amendment 2's `half`
    along_min: float = 0.1  # a cosine under this is not along it: Amendment 8c's along_min
    min_share: float = 1e-4  # inside share under this may be rounding: bf16 rounding energy is at most 2^-16 = 1.5e-5 of dW's

    def __post_init__(self) -> None:
        assert 0 < self.along_min**2 < self.half < 1 and self.tol > 0 and self.min_share > 2.0**-16


CM = Amend9()


# ----------------------------------------------------------------------------- rules (pure)


def finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def along_class(c_in: float, share: float) -> str:
    """One student against one fine-tune's inside part. Both lines inclusive; a negative cosine is 'not along'."""
    if not finite(c_in, share) or share < CM.min_share:
        return "unscorable"
    if c_in > 0 and c_in * c_in >= CM.half:
        return "mostly along"
    return "mostly elsewhere" if c_in >= CM.along_min else "not along"


def wording(labels: list[str]) -> str:
    """P1, the headline claim. Every student 'mostly along': 'copies'. None: 'expressible change'. Otherwise 'mixed'."""
    if not labels or "unscorable" in labels:
        return "unscorable"
    n = labels.count("mostly along")
    return "copies" if n == len(labels) else "expressible change" if n == 0 else "mixed"


def separated(own: list[float], other: list[float]) -> str:
    """P2: students' cosines to their own fine-tune's inside part against a fine-tune they never saw."""
    if not own or not other or not finite(*own, *other):
        return "unscorable"
    if max(other) >= min(own):
        return "no power"
    return "reads its own" if max(other) < CM.along_min <= min(own) else "partly"


def span_class(c_all: float, c_in: float, share: float) -> str:
    """P3: does cos to dW = cos to the inside part x sqrt(inside share) hold? It does if the student has no outside overlap."""
    if not finite(c_all, c_in, share) or share < CM.min_share:
        return "unscorable"
    return "in the span" if abs(c_all - c_in * math.sqrt(share)) <= CM.read_tol else "outside counts"


def cell(out: dict[str, Any], s: str, t: str, q: str) -> float:
    return out["reads"].get(f"{s} | {t}", {}).get(q, math.nan)  # a missing state reads nan, which every rule refuses


def score(out: dict[str, Any]) -> dict[str, Any]:
    own = {s: s.removesuffix("_o1") for s in E12_STUDENTS} | dict.fromkeys(REAL_8C, T_REAL) | dict.fromkeys(NULL_8C, T_NULL)
    wrong = {s: f"{TABLES[s.split('_')[0]]}_{s.split('_')[1]}" for s in E12_STUDENTS}
    real = [*E12_STUDENTS, *REAL_8C]
    label = {s: along_class(cell(out, s, own[s], "cos_in"), cell(out, s, own[s], "share")) for s in own}
    span = {s: span_class(*(cell(out, s, own[s], q) for q in ("cos_all", "cos_in", "share"))) for s in own}
    return {"P1": {"verdict": wording([label[s] for s in real]), "E12": wording([label[s] for s in E12_STUDENTS]),
                   "E34": wording([label[s] for s in REAL_8C]), "per_state": label},
            "P2": {"null_student": separated([cell(out, s, T_REAL, "cos_in") for s in REAL_8C], [cell(out, s, T_REAL, "cos_in") for s in NULL_8C]),
                   "other_table": separated([cell(out, s, own[s], "cos_in") for s in E12_STUDENTS], [cell(out, s, wrong[s], "cos_in") for s in E12_STUDENTS])},
            "P3": {"verdict": agree(list(span.values())), "per_state": span}}


# ----------------------------------------------------------------------------- reads


def inside(M: dict[str, Tensor], b: dict[str, split.Basis]) -> dict[str, Tensor]:
    """M minus its least-squares remainder, per matrix, fp64: the part of M a first-order change of U and V can write."""
    return {k: M[k].double() - split.obstruction(M[k], b[k]) for k in M}


def inside_share(a: dict[str, Tensor], b: dict[str, split.Basis]) -> float:
    return change_stats(inside(a, b), a)["size_ratio"] ** 2


def null_inside(T: dict[str, Tensor], b: dict[str, split.Basis]) -> float:
    """Inside share of an isotropic update with T's per-matrix energies (split.gaussian_null, exact in expectation)."""
    e = {k: float(T[k].double().pow(2).sum()) for k in T}
    return 1 - sum(e[k] * split.gaussian_null(b[k], *T[k].shape) for k in T) / sum(e.values())


def ls_changes(T: dict[str, Tensor], uv: dict[str, tuple[Tensor, Tensor]], b: dict[str, split.Basis]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """The least-squares split's weight change in fp64: first order (dU V^T + U dV^T), and whole (dU dV^T added)."""
    first, whole = {}, {}
    for k, M in T.items():
        U, V = uv[k][0].double(), uv[k][1].double()
        dU, dV = split.least_squares_split(M, b[k], U, V)
        first[k] = split.dsigma(dU, dV, U, V)
        whole[k] = first[k] + dU @ dV.T
    return first, whole


def control(T: dict[str, Tensor], I: dict[str, Tensor], uv: dict[str, tuple[Tensor, Tensor]], b: dict[str, split.Basis]) -> dict[str, Any]:
    """On one fine-tune, before any student. `stop` is true on a miss. The whole state's reading is reported only: dU dV^T is not in I."""
    share = change_stats(I, T)["size_ratio"] ** 2
    first, whole = ls_changes(T, uv, b)
    rec = {"share": share, "null_inside": null_inside(T, b), "ls_first_order": change_stats(first, I)["cosine"], "do_nothing": change_stats(T, I)["cosine"],
           "ls_whole": change_stats(whole, I)["cosine"], "ls_whole_size": change_stats(whole, I)["size_ratio"]}
    rec["stop"] = not (abs(rec["ls_first_order"] - 1) <= CM.ctrl_tol and abs(rec["do_nothing"] - math.sqrt(share)) <= CM.ctrl_tol and share >= CM.min_share)
    return rec


def read(s: dict[str, Tensor], T: dict[str, Tensor], I: dict[str, Tensor]) -> dict[str, Any]:
    a, c = change_stats(s, T), change_stats(s, I)
    share = change_stats(I, T)["size_ratio"] ** 2
    return {"cos_all": a["cosine"], "cos_in": c["cosine"], "share": share, "size_to_all": a["size_ratio"], "size_to_inside": c["size_ratio"],
            "out_term": a["cosine"] - c["cosine"] * math.sqrt(share)}


def student(out: dict[str, Any], name: str, path: Path, teachers: list[tuple[str, float | None]], F: dict[str, Any], uv: dict[str, Any], b: dict[str, Any]) -> None:
    """All of one student's reads. A float beside a teacher is the cosine the earlier run filed from fp32: off by read_tol, stop."""
    if not path.is_file():
        out["missing"].append(name)
        return log(f"[missing] {name}: {path}")
    s = offsets_state(path, uv)[0]
    out["inside_of_student"][name] = inside_share(s, b)
    for t, saved in teachers:
        rec = read(s, F[t][0], F[t][1])
        rec["saved_cos_all"] = saved
        out["reads"][f"{name} | {t}"] = rec
        log(f"[read {name} | {t}] {json.dumps(rec)}")
        if saved is not None and abs(rec["cos_all"] - saved) > CM.read_tol:
            halt(OUT, out, f"{name} | {t}: cos to dW {rec['cos_all']:.6f} against the filed {saved:.6f}")
    log(f"[inside of student {name}] {out['inside_of_student'][name]}")
    dump(OUT, out)


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    assert json.loads(PRED.read_text())["amend9"] == json.loads(json.dumps(asdict(CM))), "predictions were filed for another config"
    if OUT.is_file():
        sys.exit(f"{OUT} exists. Move it aside to start again")
    head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    out: dict[str, Any] = {"commit": head, "predictions_sha256": sha256(PRED), "controls": {}, "reads": {}, "inside_of_student": {}, "missing": []}
    uv = torch.load(A8C / "uv0.pt", weights_only=False)
    b = split.bases(uv, CM.tol)
    out["ranks"] = {k: [v.r_u, v.r_v] for k, v in b.items()}
    log(f"[bases] {json.dumps(out['ranks'])}")
    files = {**{f"{t}_{s}": E46 / f"{t}_{s}.pt" for t in TABLES for s in ("s0", "s1")}, **{t: A8C / f"{t}_dW.pt" for t in (T_REAL, T_NULL)}}
    F: dict[str, Any] = {}
    for t, p in files.items():  # the control, alone, before any student
        T = torch.load(p, weights_only=False, map_location="cpu")
        assert set(T) == set(uv) and all(T[k].shape == (uv[k][0].shape[0], uv[k][1].shape[0]) for k in uv), t
        F[t] = (T, inside(T, b))
        out["controls"][t] = rec = control(T, F[t][1], uv, b)
        log(f"[control {t}] {json.dumps(rec)}")
        dump(OUT, out)
        if rec["stop"]:
            halt(OUT, out, f"control on {t}: {json.dumps(rec)}")
    e12 = json.loads((E12 / "new_vs_existing_amend2.json").read_text())["students"]
    for name in E12_STUDENTS:
        t, seed = name.split("_")[:2]
        student(out, name, E12 / "large" / "amend2" / f"{name}.pt", [(f"{t}_{seed}", e12[name]["cosine"]), (f"{TABLES[t]}_{seed}", None)], F, uv, b)
    c8 = json.loads((RESULTS / "teacher_training_amend8c.json").read_text())["cos"]
    for name in (*REAL_8C, *NULL_8C):
        student(out, name, A8C / f"{name}.pt", [(t, c8[f"{name} | {t}"]) for t in (T_REAL, T_NULL)], F, uv, b)
    out["score"], out["done"] = score(out), True
    dump(OUT, out)
    log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    main()
