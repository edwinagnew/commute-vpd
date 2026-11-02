"""E55 C3's paths, state inventory, file gates and results files, shared by run.py's three stages."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

import torch

from commute import keydep as KD
from commute import open_pair as OP
from commute import steering as ST
from commute.heldout_keys import PARTS
from commute.revival_kv import MAPS, Ref, make_ref

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, LARGE = RESULTS / "predictions.json", RESULTS / "large"
E58 = HERE.parent / "open_pair" / "results"
E58_RUN, E58_A1, E58_PRED_A1 = E58 / "run.json", E58 / "amend1.json", E58 / "predictions_amend1.json"
ALIVE = HERE.parent / "transport_test" / "results" / "large" / "alive.pt"
OLD_SEEDS, NEW_SEEDS = (0, 1, 2), (3, 4, 5, 6, 7)
EXIT_STOP, EXIT_ORDER, EXIT_RESUME = 1, 2, 3
log = KD.log
Tasks = dict[str, dict[str, ST.Task]]


def map_dir(m: str) -> Path:
    return E58 / "large" if m in ("elicit", "cipher") else E58 / "large" / "amend1"


def windows_path(m: str, part: str) -> Path:
    return map_dir(m) / f"windows_{m}_{part}.pt"


def state_path(m: str, s: int) -> Path:
    return map_dir(m) / f"{m}_s{s}.pt" if s in OLD_SEEDS else LARGE / f"{m}_s{s}.pt"


def e58_records() -> dict[str, dict[str, Any]]:
    """E58's and Amendment 1's records of seeds 0-2, by state name."""
    run, a1 = json.loads(E58_RUN.read_text()), json.loads(E58_A1.read_text())
    return {f"{m}_s{s}": (run["states"] if m in ("elicit", "cipher") else a1["control"])[f"{m}_s{s}"] for m in MAPS for s in OLD_SEEDS}


def e58_windows_sha() -> dict[str, dict[str, str]]:
    run, a1 = json.loads(E58_RUN.read_text()), json.loads(E58_A1.read_text())
    return {m: run["windows_sha256"][m] if m in ("elicit", "cipher") else {p: a1["c0"][m]["sha256"][p] for p in PARTS} for m in MAPS}


def e58_thresholds() -> dict[str, Any]:
    """E58's label and readout thresholds as Amendment 1 filed them (G_min = E58's gate as run), for every map."""
    return json.loads(E58_PRED_A1.read_text())["thresholds"]


def out_of_order(why: str) -> NoReturn:
    print(f"{why} (exit {EXIT_ORDER})", file=sys.stderr, flush=True)
    sys.exit(EXIT_ORDER)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False).stdout.strip()


def load_predictions(need_b: bool) -> dict[str, Any]:
    """predictions.json, committed unchanged, with filing A filed (and filing B, for the seeds and instruments
    stages); exit 2 otherwise."""
    rel = str(PRED.relative_to(REPO))
    if not PRED.is_file() or not git("ls-files", rel) or subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode:
        out_of_order(f"{rel} is not committed unchanged: file and commit it first")
    P = json.loads(PRED.read_text())
    if P.get("filing_A", {}).get("status") != "filed":
        out_of_order("filing A is not filed")
    if need_b and P.get("filing_B", {}).get("status") != "filed":
        out_of_order("filing B is not filed: append it from results/measure.json's filing_B_draft and commit it (plan § 5)")
    return P


def header(code: list[str]) -> dict[str, Any]:
    return {"git_head": git("rev-parse", "HEAD"), "git_status_code": git("status", "--porcelain", "--", *code), "predictions_sha256": KD.sha256(PRED)}


def dump(path: Path, out: dict[str, Any]) -> None:
    path.write_text(json.dumps(OP.jsonable(out), indent=1, sort_keys=True))


def stop(path: Path, out: dict[str, Any], why: str) -> int:
    out["stopped"] = why
    dump(path, out)
    log(f"STOP: {why}: tell the user")
    return EXIT_STOP


def load_tasks(net: ST.Net) -> Tasks:
    return {m: {p: ST.load_task(net, windows_path(m, p), f"{m}_{p}", m) for p in PARTS} for m in MAPS}


def rank1_ref(net: ST.Net) -> tuple[Ref, torch.Tensor]:
    """The reading windows (the rank-1 file's held-out windows; the four maps' tokens are identical, job 1 checks) and
    the reverse fine-tune's train windows' tokens."""
    w = torch.load(windows_path("rank1", "trained"), weights_only=False)
    return make_ref(net, w["heldout"].tokens, w["heldout"].half), w["train"].tokens


def load_state(m: str, s: int) -> ST.DW:
    return torch.load(state_path(m, s), weights_only=True)


class Clock:
    """The job's deadline: `left()` seconds remain."""

    def __init__(self, minutes: float) -> None:
        self.end = time.monotonic() + 60 * minutes

    def left(self) -> float:
        return self.end - time.monotonic()
