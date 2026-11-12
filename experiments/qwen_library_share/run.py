"""Cell P3.8 (run 1 E170, run 2 E174): E132's re-gating share, on the new decomposition p-9707d6ff, of the pair
library's saved states, each side read on its own rows; per pair, side A's share minus side B's (copy − cipher,
known − invented, gold − shuffled gold) at the late and the matched-size save. A description: no reuse or new label.
Plan: reports/plans/2026-11-12_plan_p38-library-share.md. Rules: results/predictions.json (the job refuses to start
without it, while its run's code there is E174, or while results/stopped.json exists).

  run 1 (alone, first)  controls, then one random change per arm (seed 0's state) and save: iid Gaussian matrices, one
        draw shared by every state, each scaled to the library state's Frobenius norm at that matrix (<key>_norm, the
        user's control, read as it is), then the same change times one factor that brings its KL on 16 rows to the
        state's (<key>_kl, a stated departure that gives the band; Fable's advice 2026-11-12: at matched norms E132's
        Gaussian moved the output ~400x less than its update). Read on the arm's rows, in results/run1/.
  run 2  controls, then the library's states (3 seeds x 2 saves per arm), fetched from W&B while the previous one is
        read, sha256-checked against the library, read, deleted. One JSON per state in results/run2/. Refuses to start
        before run 1's twelve records are home. --arms splits it over pods.
Controls (a miss stops the job): W0 as the fine-tuned model on 16 rows of each row set (K00 = K01 = K0; under an
all-zero split K10 = K11 = K0); the do-nothing state through the split path on the job's first change (share 1); each
state's K0 over its 128 rows equals the library's floor for that side (results/floors.json): the rows are the
library's. A resubmitted job skips what is written. Every number also goes to results/run.log.
The share is E57's: (K00 - K01) / (K00 - K11) summed over the rows, status by `qwen_reading.shares`.

  python experiments/qwen_library_share/run.py --run 1 --decomp "$COMMUTE_QWEN_NEW_DIR"
  python experiments/qwen_library_share/run.py --run 2 --arms copy,cipher --decomp "$COMMUTE_QWEN_NEW_DIR"
  python experiments/qwen_library_share/run.py --summary      (no GPU: results/summary.json)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

from commute import library_states as LS
from commute import qwen_pairs as P
from commute import qwen_reading as qr
from commute.jobs import hard_exit, load_experiment
from commute.qwen_d3 import ALIVE_TOTAL_OF, GSTATS_OF, null_seed, pinned
from commute.qwen_eval import MODEL, MODEL_REV
from commute.qwen_sft import load
from commute.qwen_vpd import SITES, decomp_id

NEW = "p-9707d6ff"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
PAIRS = {"copy_cipher": ("copy", "cipher"), "facts": ("known", "invented"), "gold": ("gold", "gold_shuf")}
ARMS = tuple(a for pair in PAIRS.values() for a in pair)
RUN_OF = {a: "library" if a.startswith("gold") else "library_run3" for a in ARMS}  # run 2 (E164) or run 3 (E165)
ROWSET = {a: "gold" if a.startswith("gold") else a for a in ARMS}  # the library read both gold arms on gold's rows
KINDS = ("late", "matched")

# E57's Lab, pointed at the new decomposition and this folder. Its functions read these names when called (as E132).
L = load_experiment("qwen_reading", "lab")
L.RESULTS, L.PRED, L.LARGE, L.GSTATS = RESULTS, PRED, RESULTS / "large", GSTATS_OF[NEW]
L.CFG = replace(L.CFG, alive_total=ALIVE_TOTAL_OF[NEW])


def _saves() -> dict[str, dict[str, str]]:
    # The library save read as each kind. late: step 300 (the gold twin's last save, step 100, stands in); matched:
    # the first matched-size save, 5x, agreed with compare-lead's P3.7 (rules.py PAIRS) 2026-11-12.
    arms = ("copy", "cipher", "known", "invented", "gold", "gold_shuf")
    return {arm: {"late": "last", "matched": "5x"} for arm in arms}


@dataclass(frozen=True)
class Config:
    saves: dict[str, dict[str, str]] = field(default_factory=_saves)
    seeds: tuple[int, ...] = (0, 1, 2)  # run 2's library seeds
    norm_seed: int = 0  # the library seed whose states' norms run 1's random changes take
    draw_seed: int = 0  # the Gaussian draw, per matrix `null_seed(draw_seed, site)`
    n16: int = 16  # rows of the W0 and do-nothing controls
    k0_tol: float = 1e-3  # relative, a state's K0 against the library's floor (E57's rep_tol)

    def __post_init__(self) -> None:
        assert set(self.saves) == set(ARMS) and all(tuple(v) == KINDS for v in self.saves.values())
        assert self.norm_seed in self.seeds and self.n16 <= L.CFG.n_rows


C = Config()


# ----------------------------------------------------------------------------- rows, states, the random change


def rows_of(arm: str) -> Tensor:
    """The arm's own rows as library.py built them (128 x 256): a built side's packed trained cue sequences, and gold's
    rows (E40's held-out problems with gold solutions) for both gold arms."""
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    if ROWSET[arm] == "gold":
        return P.gold_rows(tok).tokens
    end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
    return P.own_rows(tok, end_id, arm, P.tasks(LS.LIBRARY / "words.json")[arm]["train"]).tokens


def states(arms: tuple[str, ...], seeds: tuple[int, ...]) -> list[LS.State]:
    """The library states read, the late saves first."""
    return [LS.state(RUN_OF[a], a, s, C.saves[a][k]) for k in KINDS for a in arms for s in seeds]


def streamed(todo: list[LS.State], root: Path, log: Any) -> Iterator[tuple[LS.State, Path]]:
    """Each state's file in order, the next one fetched while the caller reads this one; deleted once read."""
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(LS.fetch, todo[0], root, log) if todo else None
        for i, st in enumerate(todo):
            assert nxt is not None
            path = nxt.result()
            nxt = pool.submit(LS.fetch, todo[i + 1], root, log) if i + 1 < len(todo) else None
            yield st, path
            shutil.rmtree(path.parent)


def gaussian(norms: dict[str, float], shapes: dict[str, tuple[int, ...]]) -> dict[str, Tensor]:
    """iid Gaussian matrices, the same draw for every state, each scaled to `norms[site]` (Frobenius, float64)."""
    out = {}
    for s in SITES:
        G = torch.randn(shapes[s], generator=torch.Generator().manual_seed(null_seed(C.draw_seed, s)), dtype=torch.float64)
        out[s] = pinned((G * (norms[s] / float(G.norm()))).float())
    return out


# ----------------------------------------------------------------------------- controls and the reading


def total(rows: dict[str, list[float]]) -> dict[str, float]:
    return {k: sum(v) for k, v in rows.items()}


def w0_control(lab: Any, name: str, r16: Tensor) -> dict[str, Any]:
    """W0 as the fine-tuned model: K00 = K01 = K0, and under an all-zero split K10 = K11 = K0 (16 rows)."""
    lab.set_weights("FT", None)
    w0 = total(lab.cells(r16, tag=f"W0 as FT, {name}, 16 rows"))
    zero = total(lab.cells(r16, lab.change(None, "zero"), tag=f"W0, zero split, {name}, 16 rows"))
    lab.drop_set("FT")
    holds = all(qr.close(w0[k], w0["K0"], L.CFG.id_tol) for k in ("K00", "K01")) and all(
        qr.close(zero[k], w0["K0"], L.CFG.id_tol) for k in ("K10", "K11"))
    return lab.check(f"W0_{name}", holds, {"w0": w0, "zero_split": zero})


def do_nothing(lab: Any, dW: dict[str, Tensor], r16: Tensor) -> dict[str, Any]:
    """The do-nothing state through the split path: K10 = K00 and K11 = K01 by construction, so the share is 1."""
    lab.set_weights("FT", dW)
    cells = total(lab.cells(r16, tag="do-nothing, 16 rows")) | total(
        lab.cells(r16, lab.change(dW, None), tag="do-nothing split, 16 rows"))
    lab.drop_set("FT")
    share = qr.shares(cells, r16.numel())["regating"]
    return lab.check("do_nothing", qr.ok(share) and abs(share - 1) <= L.CFG.id_tol, {"share": share} | cells)


def controls(lab: Any, run: str, rows: dict[str, Tensor]) -> None:
    for name, r in rows.items():
        if not lab.get(f"{run}/W0_{name}"):
            lab.put(f"{run}/W0_{name}", w0_control(lab, name, r[: C.n16]))


def split(lab: Any, dW: dict[str, Tensor]) -> dict[str, Any]:
    assert set(dW) == set(SITES), "sites differ"
    return lab.split(dW, nulls=False, halves=False)


def read(lab: Any, name: str, st: LS.State, dW: dict[str, Tensor], sp: dict[str, Any], rows: Tensor,
         extra: dict[str, Any]) -> dict[str, Any]:
    """The four cells of one change on its arm's rows (split `sp` of it); K0 against the side's floor. Written to
    results/<name>.json."""
    floor = json.loads((LS.LIBRARY / "floors.json").read_text())["floors"][ROWSET[st.arm]]["K0"]
    lab.set_weights("FT", dW)
    cells = lab.cells(rows, tag=name)
    k0 = sum(cells["K0"]) / rows.numel()
    lab.check(f"K0_{name}", qr.close(k0, floor, C.k0_tol), {"K0": k0, "floor": floor})
    cells |= lab.cells(rows, lab.change(dW, sp["X"]), tag=f"{name}, split")
    lab.drop_set("FT")
    rec = {"decomp": NEW, "state": asdict(st), "rows": cells, "stored_error": sp["stored_error"], "split_s": sp["s"],
           "means": {k: sum(v) / rows.numel() for k, v in cells.items()}} | qr.whole(cells, L.CFG.row_len) | extra
    lab.put(name, rec)
    lab.log(f"[share] {name}: {rec['regating']:.9g} ({rec['status']}), {lab.peaks()}")
    return rec


# ----------------------------------------------------------------------------- the two runs


def kl16(lab: Any, dW: dict[str, Tensor], r16: Tensor, scale: float = 1.0) -> float:
    lab.set_weights("state", dW, scale)
    x = lab.kl_rows("state", r16)
    lab.drop_set("state")
    return x


def scaled(lab: Any, name: str, R: dict[str, Tensor], target: float, r16: Tensor) -> dict[str, Any]:
    """E57's search (as E132's `scaled`) for the factor b that brings R's KL on 16 rows within kl_match of target; R is
    multiplied by b in place."""

    def kl_at(b: float) -> float:
        x = kl16(lab, R, r16, b)
        lab.log(f"[{name} search] b {b:.6g}: KL {x:.9g}, target {target:.9g}")
        return x

    search = qr.scale_search(kl_at, target, L.CFG.kl_match, L.CFG.kl_evals)
    for m in R.values():
        m.mul_(search["b"])
    return search


def random_state(lab: Any, st: LS.State, path: Path, rows: Tensor, first: bool) -> None:
    """Stage 1, the user's control: the Gaussian at the state's per-matrix norms, read as it is. Stage 2 (a departure,
    stated in the plan): the same change times the one factor that brings its KL on 16 rows to the state's; the band
    is read from stage 2, which must be 'scored'. The split is linear in δW, so stage 2's is stage 1's times b."""
    r16 = rows[: C.n16]
    dW = load(path)
    norms = {s: float(m.double().norm()) for s, m in dW.items()}
    target = kl16(lab, dW, r16)
    R = gaussian(norms, {s: tuple(m.shape) for s, m in dW.items()})
    del dW
    if first:
        lab.put(f"run1/do_nothing_{st.key}", do_nothing(lab, R, r16))
    sp = split(lab, R)
    base = {"draw_seed": C.draw_seed, "norms": norms, "state_kl16": target}
    if not lab.get(f"run1/{st.key}_norm"):
        read(lab, f"run1/{st.key}_norm", st, R, sp, rows, {"random": base | {"scale": 1.0}})
    search = scaled(lab, st.key, R, target, r16)
    sp["X"] = {s: x.mul_(search["b"]) for s, x in sp["X"].items()}
    rec = read(lab, f"run1/{st.key}_kl", st, R, sp, rows, {"random": base | {"scale": search["b"], "search": search}})
    lab.check(f"scored_{st.key}_kl", rec["status"] == "scored", {"status": rec["status"], "den_mean": rec["den_mean"]})


def run1(lab: Any, rows: dict[str, Tensor], root: Path) -> None:
    todo = [st for st in states(ARMS, (C.norm_seed,)) if not lab.get(f"run1/{st.key}_kl")]
    for i, (st, path) in enumerate(streamed(todo, root, lab.log)):
        random_state(lab, st, path, rows[ROWSET[st.arm]], i == 0)


def run2(lab: Any, arms: tuple[str, ...], rows: dict[str, Tensor], root: Path) -> None:
    todo = [st for st in states(arms, C.seeds) if not lab.get(f"run2/{st.key}")]
    for i, (st, path) in enumerate(streamed(todo, root, lab.log)):
        dW, r = load(path), rows[ROWSET[st.arm]]
        if i == 0:
            lab.put(f"run2/do_nothing_{st.key}", do_nothing(lab, dW, r[: C.n16]))
        read(lab, f"run2/{st.key}", st, dW, split(lab, dW), r, {})
        del dW


# ----------------------------------------------------------------------------- the description (no GPU)


def stored(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def gap(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    """Side A's share minus side B's; None unless both are 'scored'."""
    if a is None or b is None or a["status"] != "scored" or b["status"] != "scored":
        return None
    return a["regating"] - b["regating"]


def cell(gaps: list[float | None], band: float | None, n: int) -> str:
    """The filed rule (predictions.json "description"): unreadable, beyond band, inside band or mixed."""
    if len(gaps) < n or band is None or any(g is None or not qr.ok(g) for g in gaps):
        return "unreadable"
    gs = [float(g) for g in gaps if g is not None]
    if all(qr.r9(abs(g)) > qr.r9(band) for g in gs) and (all(g > 0 for g in gs) or all(g < 0 for g in gs)):
        return "beyond band"
    return "inside band" if qr.r9(abs(sum(gs) / len(gs))) <= qr.r9(band) else "mixed"


def summarise(R: dict[str, Any]) -> dict[str, Any]:
    spread, out = R["description"]["spread"], {}
    for pair, (a, b) in PAIRS.items():
        for kind in KINDS:
            def rec(run: str, arm: str, seed: int, end: str = "", kind: str = kind) -> dict[str, Any] | None:
                return stored(RESULTS / run / f"{arm}_s{seed}_{C.saves[arm][kind]}{end}.json")

            rand = gap(rec("run1", a, C.norm_seed, "_kl"), rec("run1", b, C.norm_seed, "_kl"))
            band = abs(rand) + spread if rand is not None else None
            gaps = [gap(rec("run2", a, s), rec("run2", b, s)) for s in C.seeds]
            out[f"{pair}/{kind}"] = {"pair": pair, "kind": kind, "saves": [C.saves[a][kind], C.saves[b][kind]],
                                     "gaps": gaps, "random_gap": rand, "band": band, "cell": cell(gaps, band, len(C.seeds)),
                                     "random_gap_norm": gap(rec("run1", a, C.norm_seed, "_norm"),
                                                            rec("run1", b, C.norm_seed, "_norm"))}
            print(f"[summary] {pair} {kind}: {json.dumps(out[f'{pair}/{kind}'])}", flush=True)
    L.write_json(RESULTS / "summary.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="P3.8: the re-gating share on the pair library")
    ap.add_argument("--run", type=int, choices=(1, 2))
    ap.add_argument("--arms", default=",".join(ARMS), help="run 2 only: the arms this job reads")
    ap.add_argument("--decomp", help="the p-9707d6ff decomposition's directory")
    ap.add_argument("--fetch-to", default=str(RESULTS / "large" / "states"), help="where states land, two at a time")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if not PRED.is_file():
        sys.exit(f"refusing to start: {PRED} missing")
    R = json.loads(PRED.read_text())
    if a.summary:
        summarise(R)
        hard_exit(0)
    arms = tuple(a.arms.split(","))
    assert a.run is not None and set(arms) <= set(ARMS), (a.run, arms)
    if R[f"experiment_run{a.run}"] == "E???":
        sys.exit(f"refusing to start: run {a.run} has no experiment number in {PRED.name}")
    missing = [st.key for st in states(ARMS, (C.norm_seed,)) if not (RESULTS / "run1" / f"{st.key}_kl.json").is_file()]
    if a.run == 2 and missing:
        sys.exit(f"refusing to start: run 1 first, alone ({len(missing)} random states not home: {missing})")
    assert decomp_id(Path(a.decomp)) == NEW, a.decomp
    lab = L.Lab(R[f"experiment_run{a.run}"], "unused", a.decomp)
    arms = ARMS if a.run == 1 else arms
    rows = {ROWSET[x]: rows_of(x) for x in arms}
    assert all(tuple(r.shape) == (L.CFG.n_rows, L.CFG.row_len) for r in rows.values()), {k: r.shape for k, r in rows.items()}
    with torch.no_grad():
        assert lab.alive is not None  # the live set's count is checked here, before any reading
        run = f"run{a.run}"
        controls(lab, run, rows)
        if a.run == 1:
            run1(lab, rows, Path(a.fetch_to))
        else:
            run2(lab, arms, rows, Path(a.fetch_to))
    lab.log(f"[{run} done] arms {','.join(arms)}; {lab.peaks()}")
    hard_exit(0)


if __name__ == "__main__":
    main()
