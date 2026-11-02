"""teacher_training E34 Amendment 8c (2026-11-02), asked for by reports/reviews/2026-11-02_review_amend8-analysis-check.md,
items 1, 2, 5: does a null student reproduce a direction across batch orders as the real one does (0.927), and how much of
each student's weight change lies along its own teacher's and along the other teacher's? Rules, leans and expected values:
results/predictions_amend8c.json (the run refuses without it, or if CM or run.CFG differ).

One MATS `debug` job on an L40, results/teacher_training_amend8c.json. W0 under the Pile harness first (run.Job, 7 min), then:
T1000 and the real student at batch order 0 (amend8.real_pair) and order 1, whose cosine must reproduce n3b's; the null teacher
and the null student at order 0, whose cosines must reproduce n3b's; the null student at order 1. Every cosine is in delta-W,
over the 24 matrices concatenated, fp32, as trained. Students are saved as bf16 offsets, teachers as bf16 delta-W, and W0's
alive (U, V) once as uv0.pt, so saved_states.offsets_state gives any saved state's delta-W later, for $0, without the model.
No harness read but W0's.

Usage (repo root, project venv, GPU):  amend8c.py [--device cuda]; cluster/teacher_amend8c.sbatch
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from amend8 import (
    AM,
    STATES,
    check,
    deltas,
    keep_curve,
    near_one,
    null_fine_tune,
    real_pair,
    train_from,
)
from amend8b import gap_to, saved
from run import CFG, EMO, RESULTS, Job, dump, halt, load, log

PRED, OUT, DIR = RESULTS / "predictions_amend8c.json", RESULTS / "teacher_training_amend8c.json", STATES / "amend8c"
R0, R1, N0, N1, TR, TN = "real_o0", "real_o1", "null_o0", "null_o1", "real_teacher", "null_teacher"
VECTORS = (R0, R1, N0, N1, TR, TN)


@dataclass(frozen=True)
class Amend8c:
    saved_tol: float = 1e-3  # a regenerated cosine against n3b's, absolute: same card, the regeneration reads identically
    size_tol: float = 0.01  # the null teacher's size / n3b's
    null_step: int = 100  # Amendment 8's chosen checkpoint (n3a)
    order: int = 1  # the second batch order: n3b's control
    ctrl_min: float = 0.2  # the real students' cosine under this: 'no power'
    reproduces: float = 0.75  # null students' cosine / real students' at or above this: 'reproduces'
    not_reproduced: float = 0.25  # at or under this: 'does not'
    along_min: float = 0.1  # a cosine with a teacher under this is not along it (amend8's cos_none)
    along_ratio: float = 4.0  # one share at least this many times the other: that teacher alone (twice the cosine)

    def __post_init__(self) -> None:
        assert 0 < self.not_reproduced < self.reproduces <= 1 and self.along_ratio > 1 and self.order != 0


CM = Amend8c()


# ----------------------------------------------------------------------------- rules (pure)


def key(a: str, b: str) -> str:
    return " | ".join(sorted((a, b), key=VECTORS.index))


def finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def reproduces_class(null: float, real: float) -> tuple[str, float | None]:
    """cos(null o0, null o1) over cos(real o0, real o1). Both ends inclusive; a negative null cosine is 'does not'."""
    if not finite(null, real):
        return "unscorable", None
    if real < CM.ctrl_min:
        return "no power", None
    r = null / real
    return ("reproduces" if r >= CM.reproduces else "does not" if r <= CM.not_reproduced else "partly"), r


def share(cos: float) -> float:
    """The share of a vector's squared length along another's: cos^2, and 0 for an opposite direction."""
    return max(cos, 0.0) ** 2


def along_class(own: float, other: float) -> str:
    """A student's cosines with its own teacher and with the other, against the line and each other."""
    if not finite(own, other):
        return "unscorable"
    a, b = own >= CM.along_min, other >= CM.along_min
    if not a and not b:
        return "neither"
    if a and share(own) >= CM.along_ratio * share(other):
        return "own"
    if b and share(other) >= CM.along_ratio * share(own):
        return "other"
    return "both"


def plane_share(c1: float, c2: float, c12: float) -> float | None:
    """The share of a vector's squared length in the plane of two others, from its cosines with them and theirs with each
    other. None if the two are parallel, or a cosine is not finite."""
    if not finite(c1, c2, c12) or abs(c12) >= 1:
        return None
    return (c1 * c1 + c2 * c2 - 2 * c1 * c2 * c12) / (1 - c12 * c12)


def shared_of(cos: float, norm_a: float, norm_b: float) -> float | None:
    """The share of a's squared length shared with b if each is one shared vector plus an orthogonal walk (review item 2): dot / ||a||^2."""
    return cos * norm_b / norm_a if finite(cos, norm_a, norm_b) and norm_a > 0 else None


def inflation_class(same: tuple[float, float], other: tuple[float, float]) -> tuple[str, float | None]:
    """Mean cosine at a shared batch order less the mean at different orders, against the ranges within each pair."""
    if not finite(*same, *other):
        return "unscorable", None
    d = sum(same) / 2 - sum(other) / 2
    return ("inflated" if d > abs(same[0] - same[1]) + abs(other[0] - other[1]) else "not shown"), d


def student_row(c: dict[str, float], student: str, own: str, other: str) -> dict[str, Any]:
    co, cx, ct = c[key(student, own)], c[key(student, other)], c[key(TR, TN)]
    plane = plane_share(co, cx, ct)
    return {"class": along_class(co, cx), "cos_own": co, "cos_other": cx, "share_own": share(co), "share_other": share(cx),
            "plane": plane, "own_beyond_other": None if plane is None else plane - share(cx)}


def score(c: dict[str, float], norms: dict[str, float]) -> dict[str, Any]:
    cls, ratio = reproduces_class(c[key(N0, N1)], c[key(R0, R1)])
    same, cross = (c[key(R0, N0)], c[key(R1, N1)]), (c[key(R0, N1)], c[key(R1, N0)])
    rows = {R1: student_row(c, R1, TR, TN), N1: student_row(c, N1, TN, TR)}  # order 1 shares no batches with a teacher
    fine = rows[R1]["class"] == rows[N1]["class"] == "own"
    return {"reproduces": cls, "reproduces_ratio": ratio, "real_cos": c[key(R0, R1)], "null_cos": c[key(N0, N1)],
            "along": rows, "along_order0": {R0: student_row(c, R0, TR, TN), N0: student_row(c, N0, TN, TR)},
            "follows_own_teacher": None if "unscorable" in (rows[R1]["class"], rows[N1]["class"]) else fine,
            "students_inflation": inflation_class(same, cross),
            "teacher_inflation": {key(s0, t): c[key(s0, t)] - c[key(s1, t)] for s0, s1 in ((R0, R1), (N0, N1)) for t in (TR, TN)},
            "real_shared_with_null": {"same_order": shared_of(same[0], norms[R0], norms[N0]), "different_orders":
                                      [shared_of(cross[0], norms[R0], norms[N1]), shared_of(cross[1], norms[R1], norms[N0])]}}


# ----------------------------------------------------------------------------- the job


def remember(job: Job, out: dict[str, Any], path: Path, name: str, net: Any, st: Any, order: int, vec: dict[str, Any]) -> None:
    """A student's bf16 offsets saved, its weight change in `vec`, its size and gap to its teacher in the file."""
    from commute import arms

    tctx = job.task(EMO)[0]
    arms.save_offsets(DIR / f"{name}.pt", tctx.uv_cpu, st, {"read": name, "order": order, "step": CFG.steps[-1]})
    vec[name] = deltas(st)
    size = max(float(vec[name][k].norm() / tctx.W0[k].cpu().norm()) for k in vec[name])
    out["students"][name] = {"size": size, "gap": gap_to(tctx, vec[name], net)}
    dump(path, out)
    log(f"[{name}] {json.dumps(out['students'][name])}")


def student(job: Job, out: dict[str, Any], path: Path, name: str, net: Any, order: int, vec: dict[str, Any]) -> list[dict[str, Any]]:
    """Teacher training from W0 at a batch order."""
    curve, states = train_from(job.task(EMO)[0], net, None, CFG.steps[-1:], seed=order)
    keep_curve(out, path, name, curve)
    remember(job, out, path, name, net, states[CFG.steps[-1]], order, vec)
    return curve


def save_teacher(name: str, dW: dict[str, Any], vec: dict[str, Any]) -> None:
    vec[name] = dW
    torch.save({k: v.bfloat16() for k, v in dW.items()}, DIR / f"{name}_dW.pt")


def cos_of(vec: dict[str, Any], a: str, b: str) -> float:
    from commute.finetune import cosines

    return cosines(vec[a], vec[b])["all"]


def near_saved(out: dict[str, Any], path: Path, name: str, value: float, ref: float) -> None:
    check(out, path, name, value, (ref - CM.saved_tol, ref + CM.saved_tol))


def run_job(job: Job, out: dict[str, Any], path: Path) -> None:
    from commute import teacher
    from commute.saved_states import offsets_state

    ref = saved("n3b")
    d, vec = ref["direction"], {}
    DIR.mkdir(parents=True, exist_ok=True)
    p = real_pair(job, out, path)
    tctx = p.tctx
    torch.save(tctx.uv_cpu, DIR / "uv0.pt")
    save_teacher(TR, p.dW, vec)
    remember(job, out, path, R0, p.net, p.st, 0, vec)
    near_saved(out, path, "cos(real student, its teacher) against n3b", cos_of(vec, R0, TR), d["real_teacher"])
    # The control, alone before any null student
    curve = student(job, out, path, R1, p.net, CM.order, vec)
    check(out, path, "real order 1 P(target)", curve[-1]["heldout_p_o"], AM.real_p)
    check(out, path, "real order 1 size", curve[-1]["rel_norm_max"], AM.real_size)
    near_saved(out, path, "cos(real, real order 1) against n3b", cos_of(vec, R0, R1), d["control"])
    net0 = teacher.teacher_net(tctx.model, {})
    res = null_fine_tune(tctx, net0, (CM.null_step,))
    size = res.evals[CM.null_step]["rel_norm_max"]
    check(out, path, "null teacher size / n3b's", size / ref["null_teacher"]["rel_norm_max"], near_one(CM.size_tol))
    save_teacher(TN, res.deltas[CM.null_step], vec)
    net = teacher.teacher_net(tctx.model, vec[TN])
    student(job, out, path, N0, net, 0, vec)
    near_saved(out, path, "cos(null, real) against n3b", cos_of(vec, N0, R0), d["null_real"])
    near_saved(out, path, "cos(null, its teacher) against n3b", cos_of(vec, N0, TN), d["null_teacher"])
    curve = student(job, out, path, N1, net, CM.order, vec)
    check(out, path, "null order 1 size at step 1 / real", curve[0]["rel_norm_max"] / p.size1, near_one(AM.s1_tol))
    out["cos"] = {key(a, b): cos_of(vec, a, b) for a, b in combinations(VECTORS, 2)}
    out["norms"] = {n: sum(float(v.double().norm()) ** 2 for v in vec[n].values()) ** 0.5 for n in VECTORS}
    dump(path, out)
    log(f"[cos] {json.dumps(out['cos'])}")
    # What a later $0 reading from the offsets will differ by: reported, no stop
    later = {n: offsets_state(DIR / f"{n}.pt", tctx.uv_cpu)[0] for n in (R0, R1)}
    out["rebuilt_cos_real"] = cos_of(later, R0, R1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    filed = json.loads(PRED.read_text())
    for name, cfg in (("config", CFG), ("amend8c", CM)):
        assert filed[name] == json.loads(json.dumps(asdict(cfg))), f"predictions were filed for another {name}: {filed[name]}"
    job = Job(args.device)
    out = load(OUT, job, {"stage": "amend8c", "reads": {}, "students": {}})
    dump(OUT, out)
    floor = job.w0["pgd_floor"]
    if not job.w0["pass"] or abs(floor / AM.floor - 1) > AM.floor_tol:
        halt(OUT, out, f"W0 first: check {job.w0['pass']}, floor {floor:.4f} (L40 {AM.floor} within {AM.floor_tol:.0%}) on {job.card}")
    if out.get("done"):
        return log("done already")
    run_job(job, out, OUT)
    out["score"], out["done"] = score(out["cos"], out["norms"]), True
    dump(OUT, out)
    log(f"[score] {json.dumps(out['score'], default=float)}")


if __name__ == "__main__":
    main()
