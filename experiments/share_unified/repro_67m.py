"""Cell P2.90: the 67M paper's re-gating shares (Fig known `known_main.csv` and Fig tools_known, "Re-gating share" rows:
the 48 states of known3.json's `share` cells) re-read with the one share for both models, `commute.share` in the
whole-answer log-odds ("ans", `share.answer_at`). On these rows every answer is one token, so "ans" is the old unit
`steering.logodds_at` ("lo") exactly; both are read from the same forwards and their largest difference is written.

Each state is read as E136 read it (`.trained`): its task's held rows (`windows.read_rows(task, "held")`), E136's batch
for the task (E53's tasks 32, the rest 64; E147's e58 maps 64), bf16 autocast, the gain from the same forwards, the
floor |gain| >= 1e-6 two-sided. Expected: new = known3's saved value within `floor_abs` (results/predictions.json).

  stage W0  first, alone: the zero update on `w0_task`'s rows; its numerator must be exactly 0 (E136's filed
            zero_numerator), or the job stops (exit 1).
  stage read  every state, one JSON line each in results/repro_67m.jsonl as it lands (old, new, new - old, the "lo"
            share, max |ans - lo| over the cells, numerator, gain, rows, batch, seconds); a rerun skips states written.
Prints "REPRO_67M DONE" at the end.

    sbatch --comment=E### experiments/share_unified/repro_67m.sbatch
    ~/commute-agents/fleet/cpu-run --mem 8G python3 experiments/share_unified/repro_67m.py --dry   (CPU: states, files, rows)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from commute import share as SH
from commute import steering as ST
from commute import world
from commute.items import Item, load_dw, resolve
from commute.jobs import EXPERIMENTS, config_from_settings, hard_exit, require_filed
from commute.windows import Rows, e136_tasks, read_rows

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRED, OUT = RESULTS / "predictions.json", RESULTS / "repro_67m.jsonl"
KNOWN3 = EXPERIMENTS / "known_3seeds" / "results" / "known3.json"
# E299's five seed-2 states (experiments/known_3seeds/run.py NEW): state -> file in its results/large
E299 = {"e46/ident32_s2": "ident32_s2.pt", "e46/rand32_s2": "rand32_s2.pt", "e53/lowercase_s2": "dW_lowercase_s2_step300.pt",
        "e53/cipher_s2": "dW_cipher_s2_step300.pt", "e54/cipher_own_s2": "cipher_own_s2.pt"}  # fmt: skip
E147_LARGE = "experiments/refit_96key/results/large"  # E147's e58 refits, dW_<map>_s<seed>.pt
UNITS: dict[str, ST.Readout] = {"ans": SH.answer_at, "lo": ST.logodds_at}


@dataclass(frozen=True)
class Config:
    floor_abs: float  # |new - old| expected under it (predictions.json settings, with its source)
    gain_floor: float  # E136's route_split floor, two-sided
    e58_batch: int  # E147's inventory batch for its e58 maps
    w0_task: str  # the zero update's rows
    zero_numerator: float  # E136's filed bar on the zero update's numerator

    def __post_init__(self) -> None:
        assert self.floor_abs > 0 and self.gain_floor > 0 and self.zero_numerator >= 0 and self.e58_batch > 0


@dataclass(frozen=True)
class State:
    name: str  # E136's name, <family>/<task>_s<seed>
    task: str  # the rows' task
    item: Item
    batch: int
    old: float  # known3's saved share
    old_source: str


def task_of(name: str) -> str:
    """e54/cipher_own_s2 -> e54/cipher (the own cipher reads E54's cipher rows), else the name without its seed."""
    task = re.sub(r"_s\d+$", "", name)
    return "e54/cipher" if task == "e54/cipher_own" else task


def states(c: Config) -> list[State]:
    """known3's share cells, each state once (c3/cipher sits in two pairs with the same values), in file order."""
    out: dict[str, State] = {}
    for cell in json.loads(KNOWN3.read_text())["cells"]:
        if cell["method"] != "share":
            continue
        for side in ("reuse", "new"):
            for name, old in zip(cell["states"][side], cell[side], strict=True):
                task = task_of(name)
                if name.startswith("e58/"):
                    item, batch = Item(name, f"{E147_LARGE}/dW_{task.removeprefix('e58/')}_s{name[-1]}.pt", None), c.e58_batch
                else:
                    batch = e136_tasks()[task]["batch"]
                    item = Item(name, f"experiments/known_3seeds/results/large/{E299[name]}", None) if name in E299 else resolve(name)
                prev = out.get(name)
                assert prev is None or prev.old == old, f"{name}: two saved shares, {prev.old if prev else None} and {old}"
                out[name] = State(name, task, item, batch, float(old), cell["source"])
    return list(out.values())


def read(w: world.World, rows: Rows, dW: dict[str, torch.Tensor], batch: int, c: Config) -> dict[str, Any]:
    t0 = time.time()
    cells = SH.share_cells(w.model, w.W0, dW, rows, UNITS, batch)
    floor = SH.Floor(c.gain_floor, two_sided=True)
    g = {u: SH.cells_gain(cells, u) for u in UNITS}
    diff = max(float((cells[k]["ans"] - cells[k]["lo"]).abs().max()) for k in SH.CELLS)
    return {"new": SH.share(cells, "ans", g["ans"], floor), "share_lo": SH.share(cells, "lo", g["lo"], floor),
            "numerator": SH.numerator(cells, "ans"), "gain": g["ans"], "max_abs_ans_minus_lo": diff,
            "rows": len(rows.toks), "batch": batch, "seconds": time.time() - t0}


def put(rec: dict[str, Any]) -> None:
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{rec['item']}] {json.dumps(rec)}", flush=True)


def done() -> set[str]:
    return {json.loads(ln)["item"] for ln in OUT.read_text().splitlines() if ln} if OUT.is_file() else set()


def main(dry: bool) -> int:
    c = config_from_settings(Config, json.loads(PRED.read_text())["settings"]["repro_67m"])
    todo = states(c)
    print(f"{len(todo)} states; {len(done())} already written", flush=True)
    if dry:
        for s in todo:
            r, d = read_rows(s.task, "held"), load_dw(s.item)
            print(f"{s.name}: {s.item.path} ({len(d)} matrices) rows {tuple(r.toks.shape)} pos {r.pos} batch {s.batch} old {s.old:.6f}", flush=True)
        return 0
    require_filed(PRED)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as every old run
    w = world.World.load("cuda")
    meta = {"card": torch.cuda.get_device_name(), "torch": torch.__version__}
    if "W0" not in done():
        rows = read_rows(c.w0_task, "held")
        rec = read(w, rows, {k: torch.zeros_like(v).cpu() for k, v in w.W0.items()}, e136_tasks()[c.w0_task]["batch"], c)
        put({"item": "W0", "task": c.w0_task, **rec, **meta})
        if abs(rec["numerator"]) > c.zero_numerator:
            print(f"STOP: the zero update's numerator is {rec['numerator']!r}, bar {c.zero_numerator}", flush=True)
            return 1
    for s in todo:
        if s.name in done():
            continue
        rec = read(w, read_rows(s.task, "held"), load_dw(s.item), s.batch, c)
        put({"item": s.name, "task": s.task, "file": s.item.path, "old": s.old, "old_source": s.old_source,
             "new_minus_old": rec["new"] - s.old, **rec, **meta})
    print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB; REPRO_67M DONE", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="CPU: list the states, their files and rows, and stop")
    hard_exit(main(ap.parse_args().dry))
