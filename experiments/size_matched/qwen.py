"""P2.28 (E???), the Qwen half: the frozen-neuron share (p-015e0710) and the cheap readings of the pair library's
larger-side states, rescaled to the smaller side's size. Plan reports/plans/2026-11-16_plan_p2-28-size-matched.md; rules
results/predictions.json (the run refuses unless committed unchanged). Descriptive: no reuse or new label.

Per pair (copy - cipher, known - invented, gold - shuffled gold) and save (late "last", matched "5x"): size is the
library's |δW| (all 196 matrices; over the shared W0 this is P3.7's ||δW|| / ||W0||). The side with the larger mean size
over seeds 0-2 is the larger side, and each of its three states is multiplied by c = mean size of the smaller side / mean
size of the larger side. Only the larger side is read again; the smaller side keeps its saved readings (plot.py).
One pod per pair, two parts in order, each its own process (each loads its own model):

  --part share     (P3.9's reader, `qwen_library_share`, pointed at p-015e0710: the MLP matrices only, as P3.9)
      stage 0, alone (a miss writes stopped.json and exits): W0 as the fine-tune on 16 rows of each row set; the
               do-nothing state through the split path (share 1); c = 1 on seed 0's late larger-side state reproduces
               P3.9's saved share within share_tol. Then the random floor of each save, before any scaled state: P3.9's
               Gaussian draw at the scaled seed-0 state's per-matrix norms, times the factor that brings its KL on 16
               rows to that state's (P3.9's run 1), read; it must be "scored".
      stage 1: the six scaled states. Files are fetched in order in the background and kept for the next part.
  --part readings  (P3.7's five readings, `qwen_library_readings` Lab.readings, fp32 model)
      stage 2, alone: c = 1 on seed 0's late larger-side state reproduces P3.7's saved steering, overlap and effective
               rank within their tolerances (a miss stops).
      stage 3: the six scaled states; the effective rank must equal the unscaled one's (one scalar changes no rank);
               each file deleted once read.
Writes results/qwen/<pair>/share/*.json (the Lab's records) and results/qwen/<pair>/readings.json; prints "STAGE n DONE".

    E120_PRED_SHA=<sha> python experiments/size_matched/qwen.py --part share --pair facts --decomp "$COMMUTE_QWEN_FROZEN_DIR"
    E120_PRED_SHA=<sha> python experiments/size_matched/qwen.py --part readings --pair facts
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from commute import library_states as LS
from commute.jobs import hard_exit, load_experiment, require_filed
from commute.qwen_sft import load
from commute.qwen_vpd import SITES, decomp_id, decomp_kinds, parse_site

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED = RESULTS / "predictions.json"
SEEDS = (0, 1, 2)
SAVES = ("last", "5x")  # P3.7's labels; P3.9 calls them "late" and "matched"
P39 = HERE.parent / "qwen_library_share" / "results" / "p-015e0710"
P37 = HERE.parent / "qwen_library_readings" / "results"


@dataclass(frozen=True)
class Cell:
    """One pair at one save: which side is larger, the factor, and the larger side's three states."""

    pair: str
    save: str
    larger: str
    smaller: str
    c: float
    states: tuple[LS.State, ...]


def cells(pair: str) -> list[Cell]:
    p, out = load_experiment("qwen_library_readings", "rules").PAIRS[pair], []
    for save in SAVES:
        st = {arm: [LS.state(p.run, arm, s, save) for s in SEEDS] for arm in (p.a, p.b)}
        size = {arm: fmean(x.dw_norm for x in v) for arm, v in st.items()}
        big, small = (p.a, p.b) if size[p.a] > size[p.b] else (p.b, p.a)
        out.append(Cell(pair, save, big, small, size[small] / size[big], tuple(st[big])))
    return out


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_all(order: list[LS.State], root: Path, pool: ThreadPoolExecutor) -> dict[str, Future[Path]]:
    """Every state's file fetched in order in the background (LS.fetch keeps a file already whole and checks sha256)."""
    return {st.key: pool.submit(LS.fetch, st, root, log) for st in dict.fromkeys(order)}


# ----------------------------------------------------------------------------- the share (P3.9's reader)


def share_part(pair: str, decomp: str, root: Path, P: dict[str, Any]) -> None:
    QS = load_experiment("qwen_library_share", "run")
    QS.point(QS.FROZEN)
    assert decomp_id(Path(decomp)) == QS.FROZEN, decomp
    out_dir = RESULTS / "qwen" / pair / "share"
    out_dir.mkdir(parents=True, exist_ok=True)
    QS.L.RESULTS, QS.L.PRED = out_dir, PRED  # the Lab's records, log and stop file go here, never into P3.9's folder
    QS.OWN = frozenset(s for s in SITES if parse_site(s)[1] in decomp_kinds(Path(decomp)))
    T, cs = P["qwen"]["thresholds"], cells(pair)
    lab = QS.L.Lab(P["experiment"], "unused", decomp)
    rows = {QS.ROWSET[a]: QS.rows_of(a) for a in dict.fromkeys(c.larger for c in cs)}
    order = [cs[0].states[0], *(c.states[0] for c in cs), *(s for c in cs for s in c.states)]
    with torch.no_grad(), ThreadPoolExecutor(1) as pool:
        files = fetch_all(order, root, pool)
        assert lab.alive is not None  # the live set's count is checked here, before any reading
        QS.controls(lab, "ctl", rows)
        first, r = cs[0].states[0], rows[QS.ROWSET[cs[0].larger]]
        dW = QS.own_sites(load(files[first.key].result()))
        if not lab.get("ctl/do_nothing"):
            lab.put("ctl/do_nothing", QS.do_nothing(lab, dW, r[: QS.C.n16]))
        rec = lab.get(f"ctl/{first.key}_c1") or QS.read(lab, f"ctl/{first.key}_c1", first, dW, QS.split(lab, dW), r, {"c": 1.0})
        saved = json.loads((P39 / "run2" / f"{first.key}.json").read_text())["regating"]
        lab.check("reproduce_c1", abs(rec["regating"] - saved) <= T["share_tol"], {"got": rec["regating"], "saved": saved, "tol": T["share_tol"]})
        del dW
        for c in cs:  # the random floor of each save, before any scaled state
            if not lab.get(f"floor/{c.save}"):
                dW = scaled(QS.own_sites(load(files[c.states[0].key].result())), c.c)
                floor(QS, lab, c, dW, rows[QS.ROWSET[c.larger]])
        lab.log("STAGE 0 DONE")
        for c in cs:
            for st in c.states:
                name = f"{c.save}/{st.key}"
                if not lab.get(name):
                    dW = scaled(QS.own_sites(load(files[st.key].result())), c.c)
                    QS.read(lab, name, st, dW, QS.split(lab, dW), rows[QS.ROWSET[c.larger]], {"c": c.c, "smaller": c.smaller})
                    del dW
        lab.log(f"STAGE 1 DONE {pair}; {lab.peaks()}")


def scaled(dW: dict[str, torch.Tensor], c: float) -> dict[str, torch.Tensor]:
    for m in dW.values():
        m.mul_(c)
    return dW


def floor(QS: Any, lab: Any, c: Cell, dW: dict[str, torch.Tensor], rows: torch.Tensor) -> None:
    """P3.9's run-1 random change (its `random_state`, KL-matched form) at the scaled seed-0 state: the same Gaussian
    draw at this state's per-matrix norms, times the factor b that brings its KL on 16 rows to the scaled state's."""
    r16, st = rows[: QS.C.n16], c.states[0]
    norms = {s: float(m.double().norm()) for s, m in dW.items()}
    target = QS.kl16(lab, dW, r16)
    R = QS.gaussian(norms, {s: tuple(m.shape) for s, m in dW.items()})
    sp = QS.split(lab, R)
    search = QS.scaled(lab, f"floor {c.save}", R, target, r16)
    sp["X"] = {s: x.mul_(search["b"]) for s, x in sp["X"].items()}
    extra = {"random": {"draw_seed": QS.C.draw_seed, "norms": norms, "state_kl16": target, "scale": search["b"], "search": search},
             "c": c.c, "larger": c.larger, "smaller": c.smaller}  # fmt: skip
    rec = QS.read(lab, f"floor/{c.save}", st, R, sp, rows, extra)
    lab.check(f"scored floor {c.save}", rec["status"] == "scored", {"status": rec["status"], "den_mean": rec["den_mean"]})


# ----------------------------------------------------------------------------- the cheap readings (P3.7's Lab)


def readings_part(pair: str, root: Path, P: dict[str, Any]) -> None:
    QR = load_experiment("qwen_library_readings", "run")
    T, cs = P["qwen"]["thresholds"], cells(pair)
    path = RESULTS / "qwen" / pair / "readings.json"
    out: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {"states": {}}
    if "stopped" in out:
        raise SystemExit(f"{path} records a stop ({out['stopped']}): the lead decides")
    out["cells"] = {c.save: {"larger": c.larger, "smaller": c.smaller, "c": c.c} for c in cs}
    saved = json.loads((P37 / pair / "stage2.json").read_text())["states"]
    lab = QR.Lab()
    try:
        read_all(QR, lab, cs, path, out, saved, T, root)
    except QR.Stopped as e:  # the stage's file records why
        log(f"STOP: {e}: tell the lead")
        hard_exit(1)
    log(f"STAGE 3 DONE {pair}; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def read_all(QR: Any, lab: Any, cs: list[Cell], path: Path, out: dict[str, Any], saved: dict[str, Any], T: dict[str, Any], root: Path) -> None:
    """Stage 2 (c = 1 against P3.7's saved readings), then stage 3 (the scaled states)."""
    keep = ("steer", "overlap", "erank", "size", "gain", "layer", "on_both")
    order = [cs[0].states[0], *(s for c in cs for s in c.states)]
    with ThreadPoolExecutor(1) as pool:
        files = fetch_all(order, root, pool)
        first = cs[0].states[0]
        if "c1" not in out:
            r = lab.readings(load(files[first.key].result(), "cuda"), first.arm, first.seed)
            out["c1"] = {"state": first.key, **{k: r[k] for k in keep}}
            ref = saved[first.key]
            for k in ("steer", "overlap"):
                QR.check(path, out, f"reproduce_c1 {k}", abs(r[k] - ref[k]) <= T[f"{k}_tol"], {"got": r[k], "saved": ref[k], "tol": T[f"{k}_tol"]})
            QR.check(path, out, "reproduce_c1 erank", abs(r["erank"] / ref["erank"] - 1) <= T["erank_rel"], {"got": r["erank"], "saved": ref["erank"]})
            QR.write(path, out)
        log(f"STAGE 2 DONE {cs[0].pair}")
        for c in cs:
            for st in c.states:
                if st.key in out["states"]:
                    continue
                t0, file = time.time(), files[st.key].result()
                r = lab.readings(scaled(load(file, "cuda"), c.c), st.arm, st.seed)
                shutil.rmtree(file.parent)
                rec = {"save": c.save, "c": c.c, **{k: r[k] for k in keep}, "seconds": time.time() - t0}
                out["states"][st.key] = rec
                QR.write(path, out)
                log(f"[state {st.key}] {json.dumps(rec)}")
                er = saved[st.key]["erank"]
                QR.check(path, out, f"erank unchanged {st.key}", abs(r["erank"] / er - 1) <= T["erank_rel"], {"scaled": r["erank"], "saved": er})


def main() -> None:
    ap = argparse.ArgumentParser(description="P2.28, the Qwen half")
    ap.add_argument("--part", required=True, choices=("share", "readings"))
    ap.add_argument("--pair", required=True, choices=("copy_cipher", "facts", "gold"))
    ap.add_argument("--decomp", help="--part share: p-015e0710's directory ($COMMUTE_QWEN_FROZEN_DIR)")
    ap.add_argument("--states", type=Path, default=Path("/root/states"), help="where fetched states land and stay until read")
    a = ap.parse_args()
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    if P["experiment"] == "E???":
        raise SystemExit("refusing to start: predictions.json has no experiment number")
    if a.part == "share":
        assert a.decomp, "--part share needs --decomp"
        share_part(a.pair, a.decomp, a.states, P)
    else:
        readings_part(a.pair, a.states, P)
    hard_exit(0)


if __name__ == "__main__":
    main()
