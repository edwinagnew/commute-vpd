"""teacher_training E34 Amendment 10 (2026-11-02): the Amendment 9 comparison, weighted by the inputs the model sees. Does the
headline read "copies the fine-tune's change on real inputs" or "reproduces its behaviour by a different change"?
Plan: reports/plans/2026-11-02_plan_e34-amend10-input-weighted.md. Rules, leans and expected values:
results/predictions_amend10.json (the run refuses without it, or if CM differs).

For two weight changes A, B, summed over the 24 matrices: <A, B>_Sigma = sum_k tr(A_k Sigma_k B_k^T) = E sum_k <A_k x, B_k x>,
x the input matrix k sees in the unchanged model W0, read with no autocast and summed in fp64. Stages cheapest first, each
writes on finishing: stage 0 without a model (files, Frobenius re-read of Amendment 9, self and zero reads), the layout
control, then per input set Sigma, its power control, its reads. Saved states only, no training. A stop is `stopped` in the
file, exit 1. Results: results/amend10.json; per-matrix inner products: results/large/amend10/gram_<set>.pt.

Usage (repo root): sbatch cluster/teacher_amend10.sbatch. Needs PYTHONPATH=experiments/new_vs_existing (amend9 imports E12's).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from amend9 import (
    A8C,
    E12,
    E12_STUDENTS,
    E46,
    NULL_8C,
    REAL_8C,
    T_NULL,
    T_REAL,
    TABLES,
    agree,
    finite,
)
from run import TRANSPORT, dump, halt, log
from torch import Tensor

from commute import vpd
from commute.data import batches, pile_rows
from commute.faithfulness import model_state
from commute.keydep import sha256
from commute.saved_states import offsets_state

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT, A9, LARGE = RESULTS / "predictions_amend10.json", RESULTS / "amend10.json", RESULTS / "amend9.json", RESULTS / "large" / "amend10"
E38 = HERE.parent / "recruitment_test" / "results" / "large" / "states"
E53 = HERE.parent / "performed_anchor" / "results" / "large"
EMO, EMO_RERUN, LADDER = "emoticon", ("emoticon_10_s0", "emoticon_10_s1"), (10, 100, 300)
FORMS, PARTS = ("uncentred", "centred"), ("whole", "A", "B")
ZERO, T_REAL_R, REAL_O1_R = "zero", f"{T_REAL}~", f"{REAL_8C[1]}~"  # a zero change, and two with one bf16 rounding draw
STUDENTS = tuple(f"S_{s}" for s in E12_STUDENTS)  # E12 named a student after its fine-tune; here they must differ
OWN = {f"S_{s}": s.removesuffix("_o1") for s in E12_STUDENTS}
TABLE_FT = tuple(f"{t}_{s}" for t in TABLES for s in ("s0", "s1"))
CIPHER = tuple(f"cipher_{n}_{s}" for n in LADDER for s in ("s0", "s1"))
FINE_TUNES = (*TABLE_FT, T_REAL, T_NULL, *EMO_RERUN, *CIPHER)
LONG = (f"cipher_{LADDER[-1]}_s0", f"cipher_{LADDER[-1]}_s1")  # the longest rerun saved
REFS = {EMO: (T_REAL, T_NULL, T_REAL_R, REAL_8C[1], *EMO_RERUN, LONG[0]), **{t: tuple(f"{t}_{s}" for s in ("s0", "s1")) for t in TABLES}}
POWER = {f"task:{EMO}": T_REAL, **{f"task:{t}": f"{t}_s0" for t in TABLES}, "pile": T_REAL}  # the fine-tune each power control is built from


@dataclass(frozen=True)
class Amend10:
    ctrl_tol: float = 1e-6  # a reading equal by construction: E12 Amendment 2's zero_tol, Amendment 9's ctrl_tol
    layout_tol: float = 1e-3  # module output against x (W0 + dW)^T, relative: fp32 forward
    along_min: float = 0.1  # a cosine under this is not along it: Amendment 8c's along_min, Amendment 9's
    gap_min: float = 0.05  # the least margin: twice Amendment 9's largest batch-order change, 0.025
    tail: float = 1e-3  # the power control's "not excited": the directions past this share of tr(Sigma), per matrix
    rho2: float = 8.0  # the power control's added squared size, against dW's, per matrix: Frobenius design value 1/3
    power_sigma_min: float = 0.99  # the brief's "near 1"
    power_frob_max: float = 0.5  # the brief's "low": design value 1/3, room for dW outside the excited directions
    round_rel: float = 2.0**-8  # bf16's largest relative rounding of an entry, Amendment 9's min_share source
    seed: int = 0  # the rounding draw and the power control's draw
    pile_rows: int = 256  # val rows 0..255: the harness's Pile rows (faithfulness.pile_blocks)
    batch_windows: int = 32
    batch_pile: int = 8

    def __post_init__(self) -> None:
        assert 0 < self.tail < 1 and 0 < self.along_min and 0 < self.gap_min and self.power_frob_max < self.power_sigma_min


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


def cosines(G: Tensor) -> Tensor:
    """Cosine under an inner product's Gram matrix. A zero or negative diagonal gives nan on its row and column."""
    d = G.diag().clone()
    d[d <= 0] = math.nan
    return G / (d.sqrt()[:, None] * d.sqrt()[None, :])


def excited(sigma: Tensor, tail: float) -> Tensor:
    """Orthonormal columns: the top eigendirections of sigma, the fewest whose eigenvalues reach (1 - tail) of its trace."""
    vals, vecs = torch.linalg.eigh(sigma)
    vals, vecs = vals.flip(0), vecs.flip(1)
    keep = int((vals.cumsum(0) < (1 - tail) * vals.sum()).sum()) + 1
    return vecs[:, :keep]


def power_change(T: Tensor, sigma: Tensor, tail: float, rho2: float, gen: torch.Generator) -> Tensor:
    """The power control's synthetic change: T on the directions sigma excites, plus a random part of squared size rho2 ||T||^2
    on those it does not. It should read near 1 under sigma and low by Frobenius."""
    Q = excited(sigma, tail)
    on = T @ Q @ Q.T
    R = torch.randn(T.shape, generator=gen, dtype=T.dtype).to(T.device)
    R = R - R @ Q @ Q.T
    n = R.norm()
    return on + (R * math.sqrt(rho2) * T.norm() / n if n > 0 else R)


# ----------------------------------------------------------------------------- rules (pure), as filed


def table_label(own: float, rerun: float, other_task: float, m: float) -> str:
    """One student against its own fine-tune, with that fine-tune's rerun as the reference. Both lines inclusive."""
    if not finite(own, rerun, other_task, m):
        return "unscorable"
    if rerun < CM.along_min or rerun - other_task < 2 * m:
        return "cannot tell"
    return "copies" if own >= rerun - m else "different change"


def real_label(own: list[float], band: list[float], other_task: float, m: float) -> str:
    """The real students, whose fine-tune has no 1,000-step rerun. `band` holds the reruns that exist of a task learned, 10 to 300 steps,
    the real task's own at 10 among them. Always limited: it assumes a 1,000-step rerun would fall in the band; see predictions_amend10.json."""
    if not own or not band or not finite(*own, *band, other_task, m):
        return "unscorable"
    lo, hi = min(band), max(band)
    if lo < CM.along_min or lo - other_task < 2 * m:
        return "cannot tell"
    if min(own) >= hi - m:
        return "copies, limited"
    return "different change, limited" if max(own) < lo - m else "cannot tell"


def seed_class(own: float, others: list[float], m: float) -> str:
    """Against one fine-tune: does the student trained on it read above the same table's other seed's students?"""
    if not others or not finite(own, *others, m):
        return "unscorable"
    return "follows its own run" if own - max(others) >= m else "follows the task only"


def margin(split_half: list[float], batch_order: list[float], rounding: list[float]) -> float:
    """The gap two reads must clear: the largest of gap_min and the three measured noises. Any nan makes it nan, and the rules refuse."""
    all_noise = [*split_half, *batch_order, *rounding]
    if not all_noise or not finite(*all_noise):
        return math.nan
    return max(CM.gap_min, *all_noise)


def wording(map_task: str, map_pile: str, real_task: str, real_pile: str) -> dict[str, str]:
    """The headline. The map tables decide, as they have reruns of the same length; the real task can only agree, limit or block."""
    labels = (map_task, map_pile, real_task, real_pile)
    words = {"copies": "copies the fine-tune's change on real inputs", "different change": "reproduces its behaviour by a different change"}
    if "unscorable" in labels:
        return {"wording": "unscorable", "scope": "a read is missing"}
    if map_task != map_pile or map_task not in words:
        split = map_task in words and map_pile in words
        return {"wording": "depends on the inputs" if split else "cannot tell", "scope": f"map tables: task {map_task}, Pile {map_pile}"}
    other = "different change" if map_task == "copies" else "copies"
    if any(r.startswith(other) for r in (real_task, real_pile)):
        return {"wording": "cannot tell", "scope": f"map tables read {map_task}; real task {real_task} on its windows, {real_pile} on Pile"}
    if real_task == real_pile == f"{map_task}, limited":
        return {"wording": words[map_task], "scope": "map tables; the real task agrees, limited by having no 1,000-step rerun"}
    return {"wording": words[map_task], "scope": f"map tables only; real task {real_task} on its windows, {real_pile} on Pile"}


Cell = Callable[[str, str, str, str, str], float]


def other_seed(r: str) -> str:
    t, s = r.split("_")
    return f"{t}_{'s1' if s == 's0' else 's0'}"


def input_set(kind: str, task: str) -> str:
    return "pile" if kind == "pile" else f"{kind}:{task}"


def noise(c: Cell, kind: str, form: str) -> dict[str, list[float]]:
    """What the margin is made of on one kind of input: split-half of every cell the rules read, batch order, one rounding draw."""
    cells = [(input_set(kind, r.split("_")[0]), s, r) for s, r in OWN.items()]
    cells += [(input_set(kind, t), f"{t}_s1", f"{t}_s0") for t in TABLES]
    cells += [(input_set(kind, EMO), s, T_REAL) for s in REAL_8C] + [(input_set(kind, EMO), EMO_RERUN[1], EMO_RERUN[0])]
    pairs = [(f"S_rand32b_{s}", f"S_rand32b_{s}_o1", f"rand32b_{s}", input_set(kind, "rand32b")) for s in ("s0", "s1")]
    pairs.append((REAL_8C[0], REAL_8C[1], T_REAL, input_set(kind, EMO)))
    e = input_set(kind, EMO)
    return {"split_half": [abs(c(st, form, "A", a, b) - c(st, form, "B", a, b)) for st, a, b in cells],
            "batch_order": [abs(c(st, form, "whole", a, r) - c(st, form, "whole", o, r)) for a, o, r, st in pairs],
            "rounding": [abs(c(e, form, "whole", REAL_O1_R, T_REAL_R) - c(e, form, "whole", REAL_8C[1], T_REAL))]}


def table_reads(c: Cell, kind: str, form: str, m: float) -> dict[str, dict[str, Any]]:
    """P1, P3, P4's reads per student against its own fine-tune."""
    rec = {}
    for s, r in OWN.items():
        t = r.split("_")[0]
        st, other_t = input_set(kind, t), TABLES[t]
        own = c(st, form, "whole", s, r)
        rerun = c(st, form, "whole", other_seed(r), r)
        others = [c(st, form, "whole", f, r) for f in (f"{other_t}_s0", f"{other_t}_s1", T_REAL, T_NULL)]
        same_table = [c(st, form, "whole", x, r) for x in STUDENTS if OWN[x] == other_seed(r)]
        rec[s] = {"own": own, "rerun": rerun, "other_task": max(others) if finite(*others) else math.nan, "other_seed_students": same_table,
                  "label": table_label(own, rerun, max(others) if finite(*others) else math.nan, m), "seed": seed_class(own, same_table, m)}
    return rec


def real_read(c: Cell, kind: str, form: str, m: float) -> dict[str, Any]:
    st = input_set(kind, EMO)
    own = [c(st, form, "whole", s, T_REAL) for s in REAL_8C]
    band = [c(st, form, "whole", EMO_RERUN[1], EMO_RERUN[0]), c(st, form, "whole", LONG[1], LONG[0])]
    band += [c(input_set(kind, t), form, "whole", f"{t}_s1", f"{t}_s0") for t in TABLES]
    others = [c(st, form, "whole", f, T_REAL) for f in (*TABLE_FT, T_NULL)]
    other = max(others) if finite(*others) else math.nan
    return {"own": own, "band": band, "other_task": other, "label": real_label(own, band, other, m),
            "null_students": [c(st, form, "whole", s, T_REAL) for s in NULL_8C], "null_own": [c(st, form, "whole", s, T_NULL) for s in NULL_8C],
            "batch_order_students": c(st, form, "whole", REAL_8C[0], REAL_8C[1])}


def score(reads: dict[str, Any]) -> dict[str, Any]:
    def c(st: str, form: str, part: str, a: str, b: str) -> float:
        return reads.get(st, {}).get(form, {}).get(part, {}).get(f"{a} | {b}", math.nan)

    out: dict[str, Any] = {"by_input": {}}
    for kind in ("task", "pile", "trigger"):
        rec: dict[str, Any] = {}
        for form in FORMS:
            nn = noise(c, kind, form)
            m = margin(nn["split_half"], nn["batch_order"], nn["rounding"])
            rec[form] = {"margin": m, "noise": {k: max(v) if finite(*v) else math.nan for k, v in nn.items()},
                         "tables": table_reads(c, kind, form, m), "real": real_read(c, kind, form, m)}
        tables = [rec[f]["tables"][s] for f in FORMS for s in OWN]
        rec["map_verdict"] = agree([x["label"] for x in tables])
        rec["real_verdict"] = agree([rec[f]["real"]["label"] for f in FORMS])
        rec["seed_verdict"] = agree([x["seed"] for x in tables])
        out["by_input"][kind] = rec
    b = out["by_input"]
    out["P1_wording"] = wording(b["task"]["map_verdict"], b["pile"]["map_verdict"], b["task"]["real_verdict"], b["pile"]["real_verdict"])
    out["ladder"] = {f: [c("pile", f, "whole", f"cipher_{n}_s1", f"cipher_{n}_s0") for n in LADDER] for f in FORMS}  # how a rerun moves with length, reported
    out["P3_seed"] = {kind: b[kind]["seed_verdict"] for kind in ("task", "pile")}
    out["P4_trigger"] = {"map": b["trigger"]["map_verdict"], "real": b["trigger"]["real_verdict"]}
    return out


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
    """Every fine-tune and student, a zero change and two with a rounding draw. A missing file is listed, and its cells read nan."""
    def ft(p: Path) -> dict[str, Tensor]:
        return torch.load(p, weights_only=False, map_location="cpu")

    files: dict[str, tuple[Path, Callable[[Path], dict[str, Tensor]]]] = {
        **{n: (E46 / f"{n}.pt", ft) for n in TABLE_FT}, **{t: (A8C / f"{t}_dW.pt", ft) for t in (T_REAL, T_NULL)},
        **{f"emoticon_10_{s}": (E38 / f"emoticon_{s}.pt", ft) for s in ("s0", "s1")},
        **{f"cipher_{n}_{s}": (E53 / f"dW_cipher_{s}_step{n}.pt", ft) for n in LADDER for s in ("s0", "s1")},
        **{f"S_{s}": (E12 / "large" / "amend2" / f"{s}.pt", lambda p: offsets_state(p, uv)[0]) for s in E12_STUDENTS},
        **{s: (A8C / f"{s}.pt", lambda p: offsets_state(p, uv)[0]) for s in (*REAL_8C, *NULL_8C)}}
    vec: dict[str, dict[str, Tensor]] = {}
    for name, (path, how) in files.items():
        if not path.is_file():
            out["missing"].append(name)
            log(f"[missing] {name}: {path}")
            continue
        vec[name] = how(path)
        assert set(vec[name]) == set(uv) and all(vec[name][k].shape == (uv[k][0].shape[0], uv[k][1].shape[0]) for k in uv), name
    gen = torch.Generator().manual_seed(CM.seed)
    vec[ZERO] = {k: torch.zeros(uv[k][0].shape[0], uv[k][1].shape[0]) for k in uv}
    if T_REAL in vec and REAL_8C[1] in vec:
        vec[T_REAL_R] = {k: v.double() * (1 + CM.round_rel * (2 * torch.rand(v.shape, generator=gen, dtype=torch.float64) - 1)) for k, v in vec[T_REAL].items()}
        vec[REAL_O1_R] = rounded_offsets(A8C / f"{REAL_8C[1]}.pt", uv, gen)
    return vec


def stacks(vec: dict[str, dict[str, Tensor]], k: str, device: str) -> Tensor:
    return torch.stack([v[k].double() for v in vec.values()]).to(device)


def named(C: Tensor, names: list[str], refs: tuple[str, ...]) -> dict[str, float]:
    return {f"{a} | {b}": float(C[i, names.index(b)]) for i, a in enumerate(names) for b in refs if b in names}


def stage0(out: dict[str, Any], vec: dict[str, dict[str, Tensor]], device: str) -> None:
    """No model. Frobenius reads every vector against every fine-tune, and must reproduce Amendment 9's cos_all; self reads 1; zero reads nan."""
    names = list(vec)
    G = sum(gram(stacks(vec, k, device), None).cpu() for k in vec[names[0]])
    assert isinstance(G, Tensor)
    C = cosines(G)
    out["reads"]["frobenius"] = {"uncentred": {"whole": named(C, names, FINE_TUNES)}}
    log(f"[frobenius] {json.dumps(out['reads']['frobenius'])}")
    a9 = json.loads(A9.read_text())["reads"]
    pairs = [(key, *key.split(" | ")) for key in a9]
    pairs = [(key, f"S_{a}" if a in E12_STUDENTS else a, b) for key, a, b in pairs]
    worst = max(abs(float(C[names.index(a), names.index(b)]) - a9[key]["cos_all"]) for key, a, b in pairs if a in names and b in names)
    nz = [i for i, n in enumerate(names) if n != ZERO]
    self_read = float((C.diag()[nz] - 1).abs().max())
    zero_nan = bool(torch.isnan(C[names.index(ZERO)]).all())
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


def read_set(out: dict[str, Any], model: Any, vec: dict[str, dict[str, Tensor]], st: str, rows: Tensor, position: int | None, batch: int, refs: tuple[str, ...]) -> None:
    """One input set. Halves by alternate window; the whole is their weighted sum, so one Gram per half. Controls before any read is written."""
    device = str(next(model.parameters()).device)
    t0 = time.time()
    if rows.shape[0] < 2:
        halt(OUT, out, f"{st}: fewer than two windows, no split-half")
    (Sa, ma, na), (Sb, mb, nb) = moments(model, rows[::2], position, batch), moments(model, rows[1::2], position, batch)
    n, names, keys = na + nb, list(vec), list(Sa)
    if st in POWER and POWER[st] in vec:
        power_control(out, vec[POWER[st]], {k: (Sa[k] + Sb[k]) / n for k in keys}, st, device)
    grams: dict[str, dict[str, Tensor]] = {"A": {}, "B": {}, "whole": {}, "mean_A": {}, "mean_B": {}, "mean_whole": {}}
    for k in keys:
        X = stacks(vec, k, device)
        grams["A"][k], grams["B"][k] = gram(X, Sa[k] / na), gram(X, Sb[k] / nb)
        grams["whole"][k] = (na * grams["A"][k] + nb * grams["B"][k]) / n
        grams["mean_A"][k], grams["mean_B"][k], grams["mean_whole"][k] = mean_part(X, ma[k] / na), mean_part(X, mb[k] / nb), mean_part(X, (ma[k] + mb[k]) / n)
    del Sa, Sb, ma, mb
    LARGE.mkdir(parents=True, exist_ok=True)
    torch.save({"names": names, "n": [na, nb], **{p: {k: v.cpu() for k, v in g.items()} for p, g in grams.items()}}, LARGE / f"gram_{st.replace(':', '_')}.pt")
    pool = {p: torch.stack(list(g.values())).sum(0).cpu() for p, g in grams.items()}
    C = {"uncentred": {p: cosines(pool[p]) for p in PARTS}, "centred": {p: cosines(pool[p] - pool[f"mean_{p}"]) for p in PARTS}}
    nz = [i for i, x in enumerate(names) if x != ZERO]
    whole = pool["whole"]
    rec = {"self_minus_1": float((C["uncentred"]["whole"].diag()[nz] - 1).abs().max()), "zero_reads_nan": bool(torch.isnan(C["uncentred"]["whole"][names.index(ZERO)]).all()),
           "asymmetry": float(((whole - whole.T).abs() / (whole.diag().clamp_min(1e-300).sqrt()[:, None] * whole.diag().clamp_min(1e-300).sqrt()[None, :]))[nz][:, nz].max()),
           "smallest_diagonal": float(whole.diag()[nz].min()), "windows_or_positions": n}
    out["controls"][f"self {st}"] = rec
    log(f"[control self {st}] {json.dumps(rec)}")
    if rec["self_minus_1"] > CM.ctrl_tol or rec["asymmetry"] > CM.ctrl_tol or not rec["zero_reads_nan"] or rec["smallest_diagonal"] <= 0:
        halt(OUT, out, f"control on {st}: {json.dumps(rec)}")
    out["reads"][st] = {f: {p: named(C[f][p], names, refs) for p in PARTS} for f in FORMS}
    diag = {x: {"top_matrix_share": max(float(grams["whole"][k][i, i]) for k in keys) / float(whole[i, i]), "mean_share": float(pool["mean_whole"][i, i] / whole[i, i])}
            for i, x in enumerate(names) if x in refs}
    out["concentration"][st] = diag
    out["seconds"][st] = time.time() - t0
    dump(OUT, out)
    log(f"[reads {st}] {json.dumps(out['reads'][st])}")
    log(f"[concentration {st}] {json.dumps(diag)} {out['seconds'][st]:.0f} s")


def power_control(out: dict[str, Any], T: dict[str, Tensor], sigma: dict[str, Tensor], st: str, device: str) -> None:
    """A change matching T on the directions sigma excites and random elsewhere: near 1 here, low by Frobenius, or the weighting has no power."""
    gen = torch.Generator().manual_seed(CM.seed)
    pair: dict[str, Tensor] = {}
    for k, s in sigma.items():
        M = T[k].double().to(device)
        pair[k] = torch.stack([M, power_change(M, s, CM.tail, CM.rho2, gen)])
    pooled = sum(gram(pair[k], sigma[k]) for k in sigma)
    frob = sum(gram(pair[k], None) for k in sigma)
    assert isinstance(pooled, Tensor) and isinstance(frob, Tensor)
    rec = {"cos_sigma": float(cosines(pooled)[0, 1]), "cos_frobenius": float(cosines(frob)[0, 1])}
    out["controls"][f"power {st}"] = rec
    log(f"[control power {st}] {json.dumps(rec)}")
    dump(OUT, out)
    if not rec["cos_sigma"] >= CM.power_sigma_min or not rec["cos_frobenius"] <= CM.power_frob_max:
        halt(OUT, out, f"power control on {st}: {json.dumps(rec)}")


def main() -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED}")
    assert json.loads(PRED.read_text())["amend10"] == json.loads(json.dumps(asdict(CM))), "predictions were filed for another config"
    if OUT.is_file():
        sys.exit(f"{OUT} exists. Move it aside to start again")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    out: dict[str, Any] = {"commit": head, "predictions_sha256": sha256(PRED), "card": torch.cuda.get_device_name() if device == "cuda" else "cpu",
                           "controls": {}, "reads": {}, "concentration": {}, "seconds": {}, "missing": []}
    uv = torch.load(A8C / "uv0.pt", weights_only=False, map_location="cpu")
    vec = load_vectors(out, uv)
    stage0(out, vec, device)
    t0 = time.time()
    model = vpd.load_component_model(device)
    windows = {EMO: torch.load(TRANSPORT / "large" / "windows.pt", weights_only=False, map_location="cpu")["heldout"],
               **{t: torch.load(E46 / f"windows_{t}.pt", weights_only=False, map_location="cpu")["heldout"] for t in TABLES}}
    out["controls"]["layout"] = err = layout_control(model, windows[EMO].tokens[:4], vec[T_REAL]) if T_REAL in vec else math.nan
    out["seconds"]["model and layout"] = time.time() - t0
    log(f"[control layout] {err}; {out['seconds']['model and layout']:.0f} s")
    dump(OUT, out)
    if not err <= CM.layout_tol:
        halt(OUT, out, f"layout control: {err}")
    for t, w in windows.items():
        read_set(out, model, vec, f"trigger:{t}", w.tokens, w.half, CM.batch_windows, REFS[t])
        read_set(out, model, vec, f"task:{t}", w.tokens, None, CM.batch_windows, REFS[t])
    read_set(out, model, vec, "pile", pile_rows("val", 0, CM.pile_rows, log), None, CM.batch_pile, (*FINE_TUNES, T_REAL_R, REAL_8C[1]))
    out["score"], out["done"] = score(out["reads"]), True
    dump(OUT, out)
    log(f"[score] {json.dumps(out['score'])}")


if __name__ == "__main__":
    main()
