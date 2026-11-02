"""teacher_training E34 Amendment 10 (2026-11-02): the Amendment 9 comparison, weighted by the inputs the model sees. Is a
student's weight change as close to its fine-tune's as a rerun of that fine-tune is, in angle and in length?
Plan: reports/plans/2026-11-02_plan_e34-amend10-input-weighted.md. Rules, expected values and what each outcome allows the claim
to say: results/predictions_amend10.json (the run refuses without it, or if CM differs).

For two weight changes A, B, summed over the 24 matrices: <A, B>_Sigma = sum_k tr(A_k Sigma_k B_k^T) = E sum_k <A_k x, B_k x>,
x the input matrix k sees in the unchanged model W0, read with no autocast and summed in fp64. Stages cheapest first, each writes
on finishing: stage 0 without a model, the layout control, the reruns (each recipe at its saved seed first, which must reproduce
the saved fine-tune), every input set's moments, every power control, and only then the reads, task before trigger. A stop is
`stopped` in the file, exit 1. Results: results/amend10.json; per-matrix inner products: results/large/amend10/gram_<set>.pt.

Usage (repo root): sbatch cluster/teacher_amend10.sbatch. Needs PYTHONPATH=experiments/new_vs_existing (amend9 imports E12's).
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from amend9 import (
    A8C,
    E12,
    E12_STUDENTS,
    E46,
    REAL_8C,
    T_NULL,
    T_REAL,
    TABLES,
    agree,
    finite,
)
from run import CFG, FT_ADAM_LR, TRANSPORT, dump, halt, log
from torch import Tensor

from commute import anchors, split, vpd
from commute.data import batches, pile_rows
from commute.faithfulness import model_state
from commute.finetune import ArmAConfig, finetune_arm_a
from commute.keydep import sha256
from commute.saved_states import offsets_state

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT, A9, LARGE = RESULTS / "predictions_amend10.json", RESULTS / "amend10.json", RESULTS / "amend9.json", RESULTS / "large" / "amend10"
EMO, ZERO, T_REAL_R, REAL_O1_R = "emoticon", "zero", f"{T_REAL}~", f"{REAL_8C[1]}~"  # the last two carry one bf16 rounding draw
AS_CLOSE, FURTHER, TELL = "as close as a rerun", "further than a rerun", "cannot tell"
OWN_RUN, TASK_ONLY = "follows its own run", "follows the task only"
FORMS, PARTS, KINDS = ("uncentred", "centred"), ("whole", "A", "B"), ("task", "pile", "trigger", "taught")
REAL_RERUNS = ("real_teacher_o1", "real_teacher_o2")  # batch orders 1 and 2, trained in this job
MAP_STEP = {"rand32": 10, "rand32b": 30}  # each table's scored step, both saved seeds (regating_table predictions, refs.e12)
LONG_TABLE, LONG_STEPS = "rand32", CFG.steps[-1]  # the reference for how a rerun moves with length: read by no rule
LONG_PAIR = tuple(f"{LONG_TABLE}_long_s{s}" for s in (0, 1))
TABLE_FT = tuple(f"{t}_{s}" for t in TABLES for s in ("s0", "s1"))  # saved, each with students
MAP_RUNS = {t: tuple(f"{t}_s{s}" for s in range(4)) for t in TABLES}  # the two saved seeds and two trained in this job
REAL_RUNS = (T_REAL, *REAL_RERUNS)
OWN = {f"S_{s}": s.removesuffix("_o1") for s in E12_STUDENTS} | dict.fromkeys(REAL_8C, T_REAL)  # student: its fine-tune
FAMILY = {"map": TABLE_FT, "real": (T_REAL,)}
FINE_TUNES = (*(r for t in TABLES for r in MAP_RUNS[t]), *LONG_PAIR, *REAL_RUNS, T_NULL)
REFS = {EMO: (*REAL_RUNS, T_REAL_R), **{t: MAP_RUNS[t] + (LONG_PAIR if t == LONG_TABLE else ()) for t in TABLES}}
POWER = {f"task:{EMO}": T_REAL, **{f"task:{t}": f"{t}_s0" for t in TABLES}, "pile": T_REAL}  # the fine-tune each power control is built from


@dataclass(frozen=True)
class Amend10:
    ctrl_tol: float = 1e-6  # a reading equal by construction: E12 Amendment 2's zero_tol, Amendment 9's ctrl_tol
    layout_tol: float = 1e-3  # module output against x (W0 + dW)^T, relative: fp32 forward
    saved_tol: float = 1e-3  # a recipe at its saved seed against the saved fine-tune, 1 - cosine: Amendment 8c's saved_tol
    t_p_min: float = 0.9946  # a 1,000-step real fine-tune's held-out P(target): Amendment 8's filed AM.t_p
    rerun_floor: float = 0.5  # under it a rerun of equal length is further from dW than no change: distance^2 = 2 - 2 cos
    gap_min: float = 0.05  # angle: two reads may each move by Amendment 9's largest batch-order change, 0.025
    gap_dist_min: float = 0.056  # distance: the same, on this distance by Frobenius on the saved states, 0.028
    excite_eps: float = 1e-3  # the power control's "not excited": an eigenvalue under this x their mean
    rho2: float = 8.0  # the power control's added squared size, against dW's, per matrix: Frobenius design value 1/3
    power_sigma_min: float = 0.99  # the brief's "near 1"; rho2 x excite_eps gives about 0.996
    round_rel: float = 2.0**-8  # bf16's largest relative rounding of an entry, Amendment 9's min_share source
    seed: int = 0  # the rounding draw and the power control's draw
    pile_rows: int = 256  # val rows 0..255: the harness's Pile rows (faithfulness.pile_blocks)
    batch_windows: int = 32
    batch_pile: int = 8

    def __post_init__(self) -> None:
        assert 0 < self.excite_eps < 1 and 0 < self.gap_min and 0 < self.gap_dist_min


CM = Amend10()


# ----------------------------------------------------------------------------- arithmetic (pure)


def gram(X: Tensor, sigma: Tensor | None) -> Tensor:
    """(N, N) of tr(A sigma B^T) over the rows of X (N, d_out, d_in), fp64; sigma None is the identity: Frobenius."""
    Y = X if sigma is None else X @ sigma
    return Y.flatten(1) @ X.flatten(1).T


def mean_part(X: Tensor, mu: Tensor) -> Tensor:
    """The mean's part of `gram`: Sigma = centred + mu mu^T, so the centred read is gram minus this."""
    out = X @ mu
    return out @ out.T


def pooled(grams: Iterable[Tensor]) -> Tensor:
    return torch.stack(list(grams)).sum(0)


def _lengths(G: Tensor) -> Tensor:
    d = G.diag().clone()
    d[d <= 0] = math.nan
    return d.sqrt()


def cosines(G: Tensor) -> Tensor:
    """Cosine under an inner product's Gram matrix. A zero or negative diagonal gives nan on its row and column."""
    n = _lengths(G)
    return G / (n[:, None] * n[None, :])


def sizes(G: Tensor) -> Tensor:
    """[a, b]: ||a|| / ||b|| under the Gram. nan as `cosines`."""
    n = _lengths(G)
    return n[:, None] / n[None, :]


def distances(G: Tensor) -> Tensor:
    """[a, b]: ||a - b|| / ||b|| under the Gram. A zero change is at exactly 1. nan as `cosines`."""
    n = _lengths(G)
    return (n[:, None] ** 2 - 2 * G + n[None, :] ** 2).clamp_min(0).sqrt() / n[None, :]


def power_change(T: Tensor, sigma: Tensor, eps: float, rho2: float, gen: torch.Generator) -> tuple[Tensor, float]:
    """The power control's synthetic change, and k/d: T, plus a random part of squared size rho2 ||T||^2 on the read directions
    sigma does not excite (eigenvalue under eps x their mean). It should read near 1 under sigma and low by Frobenius."""
    vals, vecs = torch.linalg.eigh(sigma.cpu())
    Q = vecs[:, vals < eps * vals.mean()].to(T.device)
    R = torch.randn(T.shape, generator=gen, dtype=T.dtype).to(T.device) @ Q @ Q.T
    n = R.norm()
    return T + (R * math.sqrt(rho2) * T.norm() / n if n > 0 else R), 1 - Q.shape[1] / sigma.shape[0]


def ceilings(vec: dict[str, dict[str, Tensor]], bases: dict[str, split.Basis], sigma: dict[str, Tensor], names: list[str]) -> dict[str, float]:
    """sqrt(1 - outside-both share) per fine-tune, the share pooled over the matrices on this sigma (GOALS.md, split.four_blocks).
    No first-order change of the alive subcomponents reads a higher cosine with that fine-tune: its (I - P_U) dW a2 is out of reach."""
    outside, total = dict.fromkeys(names, 0.0), dict.fromkeys(names, 0.0)
    for k, S in sigma.items():
        S2 = split.read_moments(bases[k], S)[1]
        Pu = bases[k].P[:, bases[k].s > 0]
        for n in names:
            M = vec[n][k].double().to(S.device)
            outside[n] += split.weighted_energy(M - Pu @ (Pu.T @ M), S2)
            total[n] += split.weighted_energy(M, S)
    return {n: math.sqrt(max(0.0, 1 - outside[n] / total[n])) if total[n] > 0 else math.nan for n in names}


# ----------------------------------------------------------------------------- rules (pure), as filed


def close(own: float, line: float, gap: float, m: float) -> str:
    """Larger is closer. Within gap of the line: as close, unless the noise is over twice gap, when a read within the band may
    be noise (split-half is about twice a whole read's noise). Under the line by more than m: further. Between: cannot tell."""
    if own >= line - gap:
        return AS_CLOSE if m <= 2 * gap else TELL
    return FURTHER if own < line - m else TELL


def angle_label(own: float, rerun: float, other: float, ceiling: float, m: float) -> str:
    """One student against its fine-tune, the reruns as the line. Refused: a rerun under the floor, references that do not
    separate, or a line no first-order change can reach (the as-close branch's positive control)."""
    if not finite(own, rerun, other, ceiling, m):
        return "unscorable"
    if rerun < CM.rerun_floor or rerun - other < 2 * m or ceiling < rerun - CM.gap_min:
        return TELL
    return close(own, rerun, CM.gap_min, m)


def length_label(dist: float, rerun_dist: float, ceiling: float, m: float) -> str:
    """The same by distance from the fine-tune, in its lengths. A zero change is at 1, and no first-order change gets nearer
    than sqrt(1 - ceiling^2). Given an angle read as close, a further distance comes from size alone."""
    if not finite(dist, rerun_dist, ceiling, m):
        return "unscorable"
    if rerun_dist > 1 or math.sqrt(max(0.0, 1 - ceiling**2)) > rerun_dist + CM.gap_dist_min:
        return TELL
    return close(-dist, -rerun_dist, CM.gap_dist_min, m)


def seed_class(own: float, others: list[float], m: float) -> str:
    """Against one fine-tune: is the student trained on it nearer than the same table's other seed's students?"""
    if not others or not finite(own, *others, m):
        return "unscorable"
    return {AS_CLOSE: TASK_ONLY, FURTHER: OWN_RUN, TELL: TELL}[close(max(others), own, CM.gap_min, m)]


def margin(noises: dict[str, list[float]], gap: float) -> float:
    """The gap two reads must clear: the largest of `gap` and every measured noise. Any nan makes it nan, and the rules refuse."""
    all_noise = [x for v in noises.values() for x in v]
    if not all_noise or not finite(*all_noise):
        return math.nan
    return max(gap, *all_noise)


def reading(task: str, pile: str) -> str:
    """One family on one axis: a label only if task windows and Pile both read it."""
    if "unscorable" in (task, pile):
        return "unscorable"
    if not {task, pile} <= {AS_CLOSE, FURTHER}:
        return TELL
    return task if task == pile else "depends on the inputs"


Cell = Callable[[str, str, str, str, str], float]


def task_of(r: str) -> str:
    return EMO if r in REAL_RUNS else r.split("_")[0]


def runs_of(r: str) -> tuple[str, ...]:
    return REAL_RUNS if r == T_REAL else MAP_RUNS[task_of(r)]


def other_tasks(r: str) -> tuple[str, ...]:
    return (*TABLE_FT, T_NULL) if r == T_REAL else (*(f"{TABLES[task_of(r)]}_{s}" for s in ("s0", "s1")), T_REAL, T_NULL)


def students_of(r: str) -> list[str]:
    return [s for s, x in OWN.items() if x == r]


def other_seed_students(r: str) -> list[str]:
    if r == T_REAL:  # no student of a real rerun exists
        return []
    t, s = r.split("_")
    return students_of(f"{t}_{'s1' if s == 's0' else 's0'}")


def rerun_pairs(r: str) -> list[tuple[str, str]]:
    return [(a, b) for a in runs_of(r) for b in runs_of(r) if a != b]


def input_set(kind: str, task: str) -> str:
    return "pile" if kind == "pile" else f"{kind}:{task}"


def noise(c: Cell, kind: str, form: str, family: str, pre: str) -> dict[str, list[float]]:
    """What a family's margin is made of on one kind of input, for cosines (pre "") or distances (pre "dist_"): split-half of every
    cell its rules read, batch order, one rounding draw, the range over its reruns."""
    fts, e = FAMILY[family], input_set(kind, EMO)
    cells = [(input_set(kind, task_of(r)), a, b) for r in fts for a, b in rerun_pairs(r)]
    cells += [(input_set(kind, task_of(r)), x, r) for r in fts for x in (*students_of(r), *other_tasks(r), *other_seed_students(r))]
    def whole(st: str, a: str, b: str) -> float:
        return c(st, form, f"{pre}whole", a, b)
    if family == "real":
        order = [(REAL_8C[0], REAL_8C[1], T_REAL, e)]
    else:
        order = [(f"S_rand32b_{s}", f"S_rand32b_{s}_o1", f"rand32b_{s}", input_set(kind, "rand32b")) for s in ("s0", "s1")]
    spread = []
    for r in fts:
        v = [whole(input_set(kind, task_of(r)), a, b) for a, b in rerun_pairs(r)]
        spread.append(max(v) - min(v) if finite(*v) else math.nan)
    return {"split_half": [abs(c(st, form, f"{pre}A", a, b) - c(st, form, f"{pre}B", a, b)) for st, a, b in cells],
            "batch_order": [abs(whole(st, a, r) - whole(st, o, r)) for a, o, r, st in order],
            "rounding": [abs(whole(e, REAL_O1_R, T_REAL_R) - whole(e, REAL_8C[1], T_REAL))],
            "rerun_spread": spread}


def rerun_line(c: Cell, st: str, form: str, part: str, r: str) -> float:
    """The mean over every ordered pair of r's runs."""
    return statistics.fmean(c(st, form, part, a, b) for a, b in rerun_pairs(r))


def student_read(c: Cell, kind: str, form: str, m: float, md: float, s: str) -> dict[str, Any]:
    """One student against its own fine-tune: the angle, the length beside it, and on the map tables P3."""
    r = OWN[s]
    st = input_set(kind, task_of(r))
    others = [c(st, form, "whole", f, r) for f in other_tasks(r)]
    other = max(others) if finite(*others) else math.nan
    own, rerun, ceiling = c(st, form, "whole", s, r), rerun_line(c, st, form, "whole", r), c(st, form, "ceiling", r, r)
    dist, rerun_dist = c(st, form, "dist_whole", s, r), rerun_line(c, st, form, "dist_whole", r)
    rec = {"own": own, "rerun": rerun, "other_task": other, "ceiling": ceiling, "angle": angle_label(own, rerun, other, ceiling, m),
           "size": c(st, form, "size_whole", s, r), "rerun_size": rerun_line(c, st, form, "size_whole", r),
           "dist": dist, "rerun_dist": rerun_dist, "length": length_label(dist, rerun_dist, ceiling, md)}
    if r != T_REAL:
        rec["other_seed_students"] = same = [c(st, form, "whole", x, r) for x in other_seed_students(r)]
        rec["seed"] = seed_class(own, same, m)
    return rec


def family_verdicts(rec: dict[str, Any], family: str) -> dict[str, Any]:
    """Over a family's students and both forms, on one kind of input; per table too, as a split between tables is a live outcome."""
    reads = [(s, x) for f in FORMS for s, x in rec[f][family]["students"].items()]
    out = {q: agree([x[q] for _, x in reads]) for q in ("angle", "length")}
    if family == "map":
        out["seed"] = agree([x["seed"] for _, x in reads])
        out["by_table"] = {t: {q: agree([x[q] for s, x in reads if task_of(OWN[s]) == t]) for q in ("angle", "length")} for t in TABLES}
    return out


def family_reading(b: dict[str, Any], family: str) -> dict[str, Any]:
    """What this job measured for one family. The wording of a claim is the filing's `what_each_outcome_allows`, not a code label."""
    v = {kind: b[kind][family] for kind in KINDS}
    return {q: reading(v["task"][q], v["pile"][q]) for q in ("angle", "length")} | {
        "by_input": v,
        "scope": "A weight change at each matrix's output, W0's inputs: every position of held-out task windows, and of Pile. Not about the "
                 f"trigger; read apart, the trigger alone: angle {v['trigger']['angle']}, length {v['trigger']['length']}; the training's own "
                 f"weighting, trigger + task: angle {v['taught']['angle']}, length {v['taught']['length']}. Not a read of behaviour."}


def largest(noises: dict[str, list[float]]) -> dict[str, float]:
    return {k: max(v) if finite(*v) else math.nan for k, v in noises.items()}


def score(reads: dict[str, Any]) -> dict[str, Any]:
    def c(st: str, form: str, part: str, a: str, b: str) -> float:
        return reads.get(st, {}).get(form, {}).get(part, {}).get(f"{a} | {b}", math.nan)

    by_input: dict[str, Any] = {}
    for kind in KINDS:
        rec: dict[str, Any] = {f: {} for f in FORMS}
        for form in FORMS:
            for family, fts in FAMILY.items():
                na, nd = noise(c, kind, form, family, ""), noise(c, kind, form, family, "dist_")
                m, md = margin(na, CM.gap_min), margin(nd, CM.gap_dist_min)
                rec[form][family] = {"margin": m, "margin_dist": md, "noise": largest(na), "noise_dist": largest(nd),
                                     "students": {s: student_read(c, kind, form, m, md, s) for s, r in OWN.items() if r in fts}}
        for family in FAMILY:
            rec[family] = family_verdicts(rec, family)
        by_input[kind] = rec
    long_st = {k: input_set(k, LONG_TABLE) for k in ("task", "pile")}
    return {"by_input": by_input, "P1_map": family_reading(by_input, "map"), "P2_real": family_reading(by_input, "real"),
            "P3_seed": {k: by_input[k]["map"]["seed"] for k in ("task", "pile")},
            "P4_trigger": {f: {q: by_input["trigger"][f][q] for q in ("angle", "length")} for f in FAMILY},
            "P5_taught": {f: {q: by_input["taught"][f][q] for q in ("angle", "length")} for f in FAMILY},
            "length_reference": {k: {f: {f"long, {LONG_STEPS} steps": c(st, f, "whole", *LONG_PAIR),
                                         f"short, {MAP_STEP[LONG_TABLE]} steps": rerun_line(c, st, f, "whole", f"{LONG_TABLE}_s0"),
                                         f"real task, {LONG_STEPS} steps": rerun_line(c, input_set(k, EMO), f, "whole", T_REAL)}
                                     for f in FORMS} for k, st in long_st.items()}}


# ----------------------------------------------------------------------------- the vectors


def rounded_offsets(path: Path, uv: dict[str, tuple[Tensor, Tensor]], gen: torch.Generator) -> dict[str, Tensor]:
    """A student's change rebuilt with each saved bf16 offset moved by one rounding draw."""
    def move(t: Tensor) -> Tensor:
        return t.double() * (1 + CM.round_rel * (2 * torch.rand(t.shape, generator=gen, dtype=torch.float64) - 1))

    dW = {}
    for k, (dU, dV) in torch.load(path, weights_only=False)["offsets"].items():
        U, V = uv[k][0].double(), uv[k][1].double()
        dW[k] = (U + move(dU)) @ (V + move(dV)).T - U @ V.T
    return dW


def load_vectors(out: dict[str, Any], uv: dict[str, tuple[Tensor, Tensor]]) -> dict[str, dict[str, Tensor]]:
    """Every saved fine-tune and student, and two with a rounding draw. A missing file is listed, and its cells read nan."""
    def ft(p: Path) -> dict[str, Tensor]:
        return torch.load(p, weights_only=False, map_location="cpu")

    files: dict[str, tuple[Path, Callable[[Path], dict[str, Tensor]]]] = {
        **{n: (E46 / f"{n}.pt", ft) for n in TABLE_FT}, **{t: (A8C / f"{t}_dW.pt", ft) for t in (T_REAL, T_NULL)},
        **{f"S_{s}": (E12 / "large" / "amend2" / f"{s}.pt", lambda p: offsets_state(p, uv)[0]) for s in E12_STUDENTS},
        **{s: (A8C / f"{s}.pt", lambda p: offsets_state(p, uv)[0]) for s in REAL_8C}}
    vec: dict[str, dict[str, Tensor]] = {}
    for name, (path, how) in files.items():
        if not path.is_file():
            out["missing"].append(name)
            log(f"[missing] {name}: {path}")
            continue
        vec[name] = how(path)
        assert set(vec[name]) == set(uv) and all(vec[name][k].shape == (uv[k][0].shape[0], uv[k][1].shape[0]) for k in uv), name
    gen = torch.Generator().manual_seed(CM.seed)
    if T_REAL in vec and REAL_8C[1] in vec:
        vec[T_REAL_R] = {k: v.double() * (1 + CM.round_rel * (2 * torch.rand(v.shape, generator=gen, dtype=torch.float64) - 1)) for k, v in vec[T_REAL].items()}
        vec[REAL_O1_R] = rounded_offsets(A8C / f"{REAL_8C[1]}.pt", uv, gen)
    return vec


def stacks(vec: dict[str, dict[str, Tensor]], k: str, device: str) -> Tensor:
    return torch.stack([v[k].double() for v in vec.values()]).to(device)


def named(C: Tensor, names: list[str], refs: tuple[str, ...]) -> dict[str, float]:
    return {f"{a} | {b}": round(float(C[i, names.index(b)]), 7) for i, a in enumerate(names) for b in refs if b in names}  # 7 places keep the file near 1 MB


def stage0(out: dict[str, Any], vec: dict[str, dict[str, Tensor]], device: str) -> None:
    """No model. Frobenius reads every vector against every fine-tune, and must reproduce Amendment 9's cos_all; self reads 1; zero reads nan."""
    with_zero = vec | {ZERO: {k: torch.zeros_like(v) for k, v in next(iter(vec.values())).items()}}
    names = list(with_zero)
    C = cosines(pooled(gram(stacks(with_zero, k, device), None).cpu() for k in with_zero[ZERO]))
    out["reads"]["frobenius"] = {"uncentred": {"whole": named(C, names[:-1], FINE_TUNES)}}
    log(f"[frobenius] {json.dumps(out['reads']['frobenius'])}")
    a9 = json.loads(A9.read_text())["reads"]
    pairs = [(key, *key.split(" | ")) for key in a9]
    pairs = [(key, f"S_{a}" if a in E12_STUDENTS else a, b) for key, a, b in pairs]
    worst = max(abs(float(C[names.index(a), names.index(b)]) - a9[key]["cos_all"]) for key, a, b in pairs if a in names and b in names)
    self_read = float((C.diag()[:-1] - 1).abs().max())
    zero_nan = bool(torch.isnan(C[-1]).all())
    out["controls"]["stage0"] = rec = {"frobenius_against_amend9": worst, "self_minus_1": self_read, "zero_reads_nan": zero_nan}
    log(f"[control stage0] {json.dumps(rec)}")
    dump(OUT, out)
    if worst > CM.ctrl_tol or self_read > CM.ctrl_tol or not zero_nan:
        halt(OUT, out, f"stage 0: {json.dumps(rec)}")


# ----------------------------------------------------------------------------- reads of the model


@torch.no_grad()
def layout_control(model: Any, rows: Tensor, T: dict[str, Tensor]) -> float:
    """At W0 + T: does the module's output equal x (W0 + T)^T with x from the input cache? Largest error over the matrices, relative."""
    device = next(model.parameters()).device
    W1 = {k: w + T[k].float().to(device) for k, w in vpd.target_weights(model).items()}
    seen: dict[str, Tensor] = {}

    def hook(_m: Any, _a: Any, o: Tensor, k: str) -> None:
        seen[k] = o

    handles = [vpd.target_linear(model, k).register_forward_hook(partial(hook, k=k)) for k in W1]
    try:
        with model_state(model, target=W1):
            cache = model(rows.to(device), cache_type="input").cache
    finally:
        for h in handles:
            h.remove()
    return max(float((cache[k] @ W1[k].T - seen[k]).abs().max() / seen[k].abs().max()) for k in W1)


@torch.no_grad()
def moments(model: Any, rows: Tensor, position: int | None, batch: int) -> tuple[dict[str, Tensor], dict[str, Tensor], int]:
    """Sums of x x^T and of x over every position of `rows`, or over `position` alone, fp64 from an fp32 forward: no autocast. W0's own inputs."""
    device = next(model.parameters()).device
    S: dict[str, Tensor] = {}
    mu: dict[str, Tensor] = {}
    n = 0
    for b in batches(rows, batch):
        cache = model(b.to(device), cache_type="input").cache
        for k, v in cache.items():
            x = (v.reshape(-1, v.shape[-1]) if position is None else v[:, position]).double()
            S[k] = S.get(k, 0) + x.T @ x
            mu[k] = mu.get(k, 0) + x.sum(0)
        n += b.numel() if position is None else b.shape[0]
    return S, mu, n


Half = tuple[dict[str, Tensor], dict[str, Tensor], int]


def halves(out: dict[str, Any], model: Any, st: str, rows: Tensor, row_id: Tensor, position: int | None, batch: int) -> tuple[Half, Half]:
    """Moments of alternate source rows: a row's windows stay in one half, so the halves share no text."""
    ids = row_id.unique()
    if ids.numel() < 2:
        halt(OUT, out, f"{st}: fewer than two source rows, no split-half")
    a = torch.isin(row_id, ids[::2])
    return moments(model, rows[a], position, batch), moments(model, rows[~a], position, batch)


def sigmas(h: tuple[Half, Half]) -> dict[str, dict[str, Tensor]]:
    """The whole set's second moment per matrix, by form."""
    (Sa, ma, na), (Sb, mb, nb) = h
    n = na + nb
    un = {k: (Sa[k] + Sb[k]) / n for k in Sa}
    return {"uncentred": un, "centred": {k: un[k] - torch.outer(ma[k] + mb[k], ma[k] + mb[k]) / n**2 for k in un}}


def read_set(out: dict[str, Any], vec: dict[str, dict[str, Tensor]], bases: dict[str, split.Basis], st: str, h: tuple[Half, Half], refs: tuple[str, ...],
             ) -> tuple[dict[str, dict[str, Tensor]], dict[str, dict[str, Tensor]]]:
    """One input set: one Gram per half, the whole their weighted sum. Saved, then read. Returns the Grams and Sigmas, for the sum of two sets."""
    device = next(iter(h[0][0].values())).device
    t0 = time.time()
    (Sa, ma, na), (Sb, mb, nb) = h
    n = na + nb
    grams: dict[str, dict[str, Tensor]] = {p: {} for p in ("A", "B", "whole", "mean_A", "mean_B", "mean_whole")}
    for k in Sa:
        X = stacks(vec, k, str(device))
        grams["A"][k], grams["B"][k] = gram(X, Sa[k] / na).cpu(), gram(X, Sb[k] / nb).cpu()
        grams["whole"][k] = (na * grams["A"][k] + nb * grams["B"][k]) / n
        grams["mean_A"][k], grams["mean_B"][k] = mean_part(X, ma[k] / na).cpu(), mean_part(X, mb[k] / nb).cpu()
        grams["mean_whole"][k] = mean_part(X, (ma[k] + mb[k]) / n).cpu()
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save({"names": list(vec), "n": [na, nb], **grams}, LARGE / f"gram_{st.replace(':', '_')}.pt")
    sigma = sigmas(h)
    write_reads(out, vec, bases, st, grams, sigma, refs)
    out["seconds"][st] = time.time() - t0
    dump(OUT, out)
    log(f"[seconds {st}] {out['seconds'][st]:.0f}")
    return grams, sigma


def write_reads(out: dict[str, Any], vec: dict[str, dict[str, Tensor]], bases: dict[str, split.Basis], st: str, grams: dict[str, dict[str, Tensor]],
                sigma: dict[str, dict[str, Tensor]], refs: tuple[str, ...]) -> None:
    """Cosine, distance and size per form and part, the ceilings, and the largest matrix's share of each fine-tune's energy. A change with no energy on this set stops."""
    names = list(vec)
    pool = {p: pooled(g.values()) for p, g in grams.items()}
    out["controls"][f"smallest diagonal {st}"] = smallest = float(pool["whole"].diag().min())
    if not smallest > 0:
        halt(OUT, out, f"{st}: a change has no energy here, {smallest}")
    with_students = [r for r in refs if r in (T_REAL, *TABLE_FT) and r in vec]
    rec: dict[str, Any] = {}
    for f in FORMS:
        G = {p: pool[p] - pool[f"mean_{p}"] if f == "centred" else pool[p] for p in PARTS}
        rec[f] = {p: named(cosines(G[p]), names, refs) for p in PARTS} | {f"dist_{p}": named(distances(G[p]), names, refs) for p in PARTS}
        rec[f]["size_whole"] = named(sizes(G["whole"]), names, refs)
        rec[f]["ceiling"] = {f"{r} | {r}": v for r, v in ceilings(vec, bases, sigma[f], with_students).items()}
    energy = {r: [float(g[names.index(r), names.index(r)]) for g in grams["whole"].values()] for r in with_students}
    rec["top_matrix_share"] = {r: max(e) / sum(e) for r, e in energy.items()}
    out["reads"][st] = rec
    dump(OUT, out)
    log(f"[reads {st}] {json.dumps(rec)}")


def power_control(out: dict[str, Any], T: dict[str, Tensor], sigma: dict[str, Tensor], st: str) -> None:
    """A change matching T on what sigma excites and large and random elsewhere: near 1 here, low by Frobenius, or the weighting has no power.
    Writes the spectrum's k/d per matrix. It says Sigma is ill-conditioned, no more; the ceilings are the control that bears on a verdict."""
    gen = torch.Generator().manual_seed(CM.seed)
    pair, kd = {}, {}
    for k, s in sigma.items():
        M = T[k].double().to(s.device)
        Z, kd[k] = power_change(M, s, CM.excite_eps, CM.rho2, gen)
        pair[k] = torch.stack([M, Z])
    rec = {"cos_sigma": float(cosines(pooled(gram(pair[k], sigma[k]) for k in sigma))[0, 1]),
           "cos_frobenius": float(cosines(pooled(gram(pair[k], None) for k in sigma))[0, 1]), "k_over_d": kd}
    out["controls"][f"power {st}"] = rec
    log(f"[control power {st}] {json.dumps(rec)}")
    dump(OUT, out)
    if not rec["cos_sigma"] >= CM.power_sigma_min:  # cos_frobenius is written, not a stop: amended 2026-11-02 after job 49463
        halt(OUT, out, f"power control on {st}: {json.dumps(rec)}")


def one_rerun(out: dict[str, Any], vec: dict[str, dict[str, Tensor]], name: str, train: Callable[[], tuple[dict[str, Tensor], dict[str, float]]],
              saved: str | None, p_min: float) -> None:
    """Train one fine-tune. With `saved`, it must reproduce that saved fine-tune and is not kept; otherwise it joins the vectors."""
    t0 = time.time()
    dW, ev = train()
    rec = {**ev, "seconds": time.time() - t0}
    if saved is not None:
        rec["cos_frobenius_to_saved"] = float(cosines(pooled(gram(torch.stack([dW[k].double(), vec[saved][k].double()]), None) for k in dW))[0, 1]) if saved in vec else math.nan
    out["controls"][f"rerun {name}"] = rec
    log(f"[rerun {name}] {json.dumps(rec)}")
    dump(OUT, out)
    if not ev["p_o"] >= p_min or (saved is not None and not 1 - rec["cos_frobenius_to_saved"] <= CM.saved_tol):
        halt(OUT, out, f"rerun {name}: {json.dumps(rec)}")
    if saved is None:
        torch.save(dW, LARGE / f"{name}_dW.pt")
        vec[name] = dW


def rerun_stage(out: dict[str, Any], model: Any, vec: dict[str, dict[str, Tensor]], tables: dict[str, dict[str, Any]], device: str) -> None:
    """Cheapest first: each recipe at its saved seed, the control, alone; then the same recipes' other seeds; then the length reference."""
    real = torch.load(TRANSPORT / "large" / "windows.pt", weights_only=False, map_location="cpu")
    token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]

    def real_run(order: int) -> tuple[dict[str, Tensor], dict[str, float]]:  # amend8.real_teacher's recipe at one batch order
        cfg = ArmAConfig(o_token=token, lr=FT_ADAM_LR, steps=LONG_STEPS, checkpoints=(1, LONG_STEPS), seed=order, optimizer="adam")
        res = finetune_arm_a(model, real["train"], real["heldout"], cfg, device, log)
        return {k: v.bfloat16() for k, v in res.deltas[LONG_STEPS].items()}, {**res.evals[LONG_STEPS], "size_1": res.evals[1]["rel_norm_max"]}

    def map_run(t: str, seed: int, steps: int) -> tuple[dict[str, Tensor], dict[str, float]]:
        return anchors.kv_finetune(model, tables[t], seed, steps, device, log)

    LARGE.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        one_rerun(out, vec, f"{t}_s0 again", partial(map_run, t, 0, MAP_STEP[t]), f"{t}_s0", 0.0)
    one_rerun(out, vec, "real order 0", partial(real_run, 0), T_REAL, CM.t_p_min)
    for t in TABLES:
        for seed in (2, 3):
            one_rerun(out, vec, f"{t}_s{seed}", partial(map_run, t, seed, MAP_STEP[t]), None, 0.0)
    for order, name in enumerate(REAL_RERUNS, start=1):
        one_rerun(out, vec, name, partial(real_run, order), None, CM.t_p_min)
    for seed, name in enumerate(LONG_PAIR):
        one_rerun(out, vec, name, partial(map_run, LONG_TABLE, seed, LONG_STEPS), None, 0.0)


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    assert json.loads(PRED.read_text())["amend10"] == json.loads(json.dumps(asdict(CM))), "predictions were filed for another config"
    if OUT.is_file():
        sys.exit(f"{OUT} exists. Move it aside to start again")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    out: dict[str, Any] = {"commit": head, "predictions_sha256": sha256(PRED), "card": torch.cuda.get_device_name() if device == "cuda" else "cpu",
                           "controls": {}, "reads": {}, "seconds": {}, "missing": []}
    uv = torch.load(A8C / "uv0.pt", weights_only=False, map_location="cpu")
    vec = load_vectors(out, uv)
    stage0(out, vec, device)
    t0 = time.time()
    model = vpd.load_component_model(device)
    tables = {t: torch.load(E46 / f"windows_{t}.pt", weights_only=False, map_location="cpu") for t in TABLES}
    windows = {EMO: torch.load(TRANSPORT / "large" / "windows.pt", weights_only=False, map_location="cpu")["heldout"], **{t: w["heldout"] for t, w in tables.items()}}
    out["controls"]["layout"] = err = layout_control(model, windows[EMO].tokens[:4], vec[T_REAL]) if T_REAL in vec else math.nan
    out["seconds"]["model and layout"] = time.time() - t0
    log(f"[control layout] {err}; {out['seconds']['model and layout']:.0f} s")
    dump(OUT, out)
    if not err <= CM.layout_tol:
        halt(OUT, out, f"layout control: {err}")
    rerun_stage(out, model, vec, tables, device)
    stage0(out, vec, device)  # Frobenius again, with the reruns
    t0 = time.time()
    bases = {k: replace(b, P=b.P.to(device), s=b.s.to(device), Q=b.Q.to(device), t=b.t.to(device)) for k, b in split.bases(uv).items()}
    sets = {}
    for t, w in windows.items():
        sets[f"task:{t}"] = halves(out, model, f"task:{t}", w.tokens, w.row, None, CM.batch_windows)
        sets[f"trigger:{t}"] = halves(out, model, f"trigger:{t}", w.tokens, w.row, w.half, CM.batch_windows)
    rows = pile_rows("val", 0, CM.pile_rows, log)
    sets["pile"] = halves(out, model, "pile", rows, torch.arange(rows.shape[0]), None, CM.batch_pile)
    out["seconds"]["forwards and bases"] = time.time() - t0
    t0 = time.time()
    for st, r in POWER.items():  # every control stop comes before any input-weighted read of a student
        if r not in vec:
            halt(OUT, out, f"power control on {st}: {r} is missing")
        power_control(out, vec[r], sigmas(sets[st])["uncentred"], st)
    out["seconds"]["power controls"] = time.time() - t0
    dump(OUT, out)
    for t in windows:
        task = read_set(out, vec, bases, f"task:{t}", sets.pop(f"task:{t}"), REFS[t])
        trigger = read_set(out, vec, bases, f"trigger:{t}", sets.pop(f"trigger:{t}"), REFS[t])
        grams = {p: {k: task[0][p][k] + trigger[0][p][k] for k in task[0][p]} for p in task[0]}
        sigma = {f: {k: task[1][f][k] + trigger[1][f][k] for k in task[1][f]} for f in FORMS}
        write_reads(out, vec, bases, f"taught:{t}", grams, sigma, REFS[t])
        del task, trigger, grams, sigma
    read_set(out, vec, bases, "pile", sets.pop("pile"), (*FINE_TUNES, T_REAL_R))
    out["score"], out["done"] = score(out["reads"]), True
    dump(OUT, out)
    log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    main()
