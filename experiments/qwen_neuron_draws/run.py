"""Cell P3.15: two more Gaussian draws (draw seeds 1 and 2) of the random reference behind the frozen-neuron share's
size-matched comparison (E187, experiments/size_matched/qwen.py) and P3.9's unmatched picture. Descriptive: no reuse or
new label. Rules: results/predictions.json (the run refuses unless it is committed unchanged with a number).

The reference is P3.9 run 1's random change, KL-matched form only (no `_norm` read): iid Gaussian matrices, per matrix
`null_seed(draw_seed, site)`, at a state's per-matrix norms on the 84 MLP matrices, split, times the one factor that
brings its KL on 16 rows to the state's, read on the arm's rows (`size_matched.qwen.floor` does exactly this; it is
called for every read here). The draw is chosen by `QS.C = replace(QS.C, draw_seed=d)`: `QS.gaussian` reads the module's
`C` and `SM.floor` records `QS.C.draw_seed`, both at call time. One pod per pair, records in results/<pair>/:

  stage 0, alone (a miss writes results/<pair>/stopped.json and exits): W0 as the fine-tune on 16 rows of each row set
           (ctl/W0_<rows>); the do-nothing state through the split path on side A's seed-0 last state (ctl/do_nothing,
           share 1); draw 0 at that state reproduces P3.9's run1/<A>_s0_last_kl.json share within E187's share_tol
           (ctl/reproduce_d0). Prints "STAGE 0 DONE".
  stage 1: per draw d and save (last, 5x): each side's seed-0 state (d<d>/side/<arm>_s0_<save>_kl; in these records
           "larger" names the arm read and "c" is 1), then E187's floor, the larger side's seed-0 state times E187's c
           (d<d>/floor/<save>). Every read must be "scored". Prints "STAGE 1 DONE <pair>".
A resubmitted job skips what is written. Every share also goes to results/<pair>/run.log.

INPUT_MANIFEST (gitignored inputs a pod needs; everything else is committed):
  - p-015e0710's W0 g statistics, `qwen_d3.GSTATS_OF["p-015e0710"]` =
    experiments/qwen_library_share/results/large/p-015e0710/gstats/W0_{maths,fineweb}.safetensors, 4.0 MB (du -sh).
    results/large is a symlink: stage the two files by hard link in a flat root-level directory and put them at that
    path on the pod as a real directory.
  - the decomposition, $COMMUTE_QWEN_FROZEN_DIR (qwen3-1.7b-p-015e0710-torch, 30 GB): NOT sent. run.py fetches it from
    W&B into --decomp (`qwen_files.fetch_decomp`, kept if complete; P3.9's pod log shows ~8 min), needs WANDB_API_KEY.
  - the four seed-0 states (5.6 GB each, 22.5 GB): fetched from W&B into --fetch-to in the background
    (`SM.fetch_all`, sha256-checked) and kept. Pod disk: >= 60 GB free beyond the venv and HF caches.
  Committed and read: P3.9's floors.json and run1/*_kl.json, E187's predictions.json (share_tol) and summary inputs,
  the library's per-arm JSONs and words.json. The model and tokenizer come from the Hugging Face hub, gsm8k too.
Timing basis (P3.9 run1_pod.log, gold_s0_5x on an H100 NVL; the A100 80GB start was replaced): one state took 357 s
for both reads, of which the `_norm` read was 118 s, so a KL-only read is ~240 s (load and target ~55 s, split 42 s,
search 3-6 KL evaluations at 9 s, the read 116 s). Per pod: 13 reads x ~4 min = ~52 min, plus ~8 min for the
decomposition and ~3 min for loads: ~65 min on one H100. Peak GPU 38.6 GiB, host 84 GB.

    PRED_SHA=<sha> python experiments/qwen_neuron_draws/run.py --pair facts --draws 1,2 \
        --decomp "$COMMUTE_QWEN_FROZEN_DIR" --fetch-to /root/states
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from commute import library_states as LS
from commute import qwen_files
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment, require_filed
from commute.qwen_sft import load
from commute.qwen_vpd import SITES, decomp_id, decomp_kinds, parse_site

HERE = Path(__file__).resolve().parent
PRED = HERE / "results" / "predictions.json"
SHARE_TOL = EXPERIMENTS / "size_matched" / "results" / "predictions.json"  # .qwen.thresholds.share_tol
QS = load_experiment("qwen_library_share", "run")
SM = load_experiment("size_matched", "qwen")
SAVES = SM.SAVES


def point(pair: str, decomp: str) -> Path:
    """QS's Lab at p-015e0710, writing into results/<pair>/ (never into P3.9's or E187's folders)."""
    QS.point(QS.FROZEN)
    assert decomp_id(Path(decomp)) == QS.FROZEN, decomp
    out = HERE / "results" / pair
    out.mkdir(parents=True, exist_ok=True)
    QS.L.RESULTS, QS.L.PRED = out, PRED
    QS.OWN = frozenset(s for s in SITES if parse_site(s)[1] in decomp_kinds(Path(decomp)))
    return out


def random_read(lab: Any, name: str, cell: Any, dW: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, Any]:
    """`SM.floor` of `cell` at dW (its record is written as floor/<save>), moved to `name`."""
    SM.floor(QS, lab, cell, dW, rows)
    dest = QS.L.RESULTS / f"{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    (QS.L.RESULTS / "floor" / f"{cell.save}.json").replace(dest)
    lab.log(f"[moved] floor/{cell.save} -> {name}")
    return lab.get(name)


def side_cell(pair: str, save: str, arm: str, other: str) -> Any:
    """A Cell for one side's unscaled seed-0 state: c = 1, 'larger' is the arm read."""
    return SM.Cell(pair, save, arm, other, 1.0, (LS.state(QS.RUN_OF[arm], arm, 0, save),))


def stage0(lab: Any, pair: str, rows: dict[str, torch.Tensor], files: dict[str, Any], tol: float) -> None:
    a, b = QS.PAIRS[pair]
    QS.controls(lab, "ctl", rows)
    cell, r = side_cell(pair, "last", a, b), rows[QS.ROWSET[a]]
    st = cell.states[0]
    if not lab.get("ctl/do_nothing"):
        lab.put("ctl/do_nothing", QS.do_nothing(lab, QS.own_sites(load(files[st.key].result())), r[: QS.C.n16]))
    QS.C = replace(QS.C, draw_seed=0)
    rec = lab.get("ctl/reproduce_d0") or random_read(lab, "ctl/reproduce_d0", cell, QS.own_sites(load(files[st.key].result())), r)
    saved = json.loads((SM.P39 / "run1" / f"{st.key}_kl.json").read_text())["regating"]
    lab.check("reproduce_d0", abs(rec["regating"] - saved) <= tol, {"got": rec["regating"], "saved": saved, "tol": tol})
    lab.log("STAGE 0 DONE")


def stage1(lab: Any, pair: str, draws: list[int], rows: dict[str, torch.Tensor], files: dict[str, Any]) -> None:
    a, b = QS.PAIRS[pair]
    for d in draws:
        QS.C = replace(QS.C, draw_seed=d)
        for c in SM.cells(pair):
            for arm, other in ((a, b), (b, a)):
                name = f"d{d}/side/{arm}_s0_{c.save}_kl"
                if not lab.get(name):
                    cell = side_cell(pair, c.save, arm, other)
                    random_read(lab, name, cell, QS.own_sites(load(files[cell.states[0].key].result())), rows[QS.ROWSET[arm]])
            name = f"d{d}/floor/{c.save}"
            if not lab.get(name):
                dW = SM.scaled(QS.own_sites(load(files[c.states[0].key].result())), c.c)
                random_read(lab, name, c, dW, rows[QS.ROWSET[c.larger]])
    lab.log(f"STAGE 1 DONE {pair}; {lab.peaks()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="P3.15: more random draws for the frozen-neuron share")
    ap.add_argument("--pair", required=True, choices=tuple(QS.PAIRS))
    ap.add_argument("--draws", default="1,2", help="draw seeds read in stage 1 (draw 0 is P3.9's and E187's)")
    ap.add_argument("--decomp", required=True, help="p-015e0710's directory ($COMMUTE_QWEN_FROZEN_DIR), fetched if incomplete")
    ap.add_argument("--fetch-to", type=Path, default=Path("/root/states"), help="where the seed-0 states land and stay")
    a = ap.parse_args()
    draws = [int(x) for x in a.draws.split(",")]
    assert draws and 0 not in draws, draws
    require_filed(PRED)
    P = json.loads(PRED.read_text())
    if P["experiment"] == "E???":
        raise SystemExit("refusing to start: predictions.json has no experiment number")
    tol = json.loads(SHARE_TOL.read_text())["qwen"]["thresholds"]["share_tol"]
    A, B = QS.PAIRS[a.pair]
    order = [LS.state(QS.RUN_OF[arm], arm, 0, save) for save in SAVES for arm in (A, B)]
    with torch.no_grad(), ThreadPoolExecutor(1) as pool:
        files = SM.fetch_all(order, a.fetch_to, pool)  # the states download while the decomposition does
        qwen_files.fetch_decomp(Path(a.decomp), QS.FROZEN, SM.log)
        point(a.pair, a.decomp)
        lab = QS.L.Lab(P["experiment"], "unused", a.decomp)
        rows = {QS.ROWSET[x]: QS.rows_of(x) for x in (A, B)}
        assert lab.alive is not None  # the live set's count is checked here, before any reading
        stage0(lab, a.pair, rows, files, tol)
        stage1(lab, a.pair, draws, rows, files)
    hard_exit(0)


if __name__ == "__main__":
    main()
