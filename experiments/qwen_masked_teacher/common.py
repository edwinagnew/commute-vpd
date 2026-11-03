"""E76: the config, paths, and the job's log and stop."""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from commute.qwen_vpd import SITES
from commute.vpd import kl_per_position

HERE = Path(__file__).resolve().parent
RESULTS, LARGE, PRED = HERE / "results", HERE / "results" / "large", HERE / "results" / "predictions.json"
EXPERIMENTS = HERE.parent
DW_FILE = Path.home() / "commute-data" / "results-large" / "qwen_sft" / "nowarmup_regen" / "dW" / "step_10.safetensors"  # u1, E40 step 10
DW_SUMS = EXPERIMENTS / "qwen_maps" / "results" / "regen_SHA256SUMS"
MATHS, WEB = EXPERIMENTS / "qwen_sft" / "results" / "maths_rows.json", EXPERIMENTS / "qwen_adversary" / "results" / "tokens.json"
HOOK_READ = EXPERIMENTS / "qwen_open_points" / "results" / "u1"  # E77's amend2_hook.json, amend1_own_last.json
LAST_SITE = "layers.27.mlp.down_proj"
DEV = "cuda"
T0 = time.time()


@dataclass(frozen=True)
class Cfg:
    columns: int = 200  # per site, the often-on ones (`qwen_maps.halves`'s order)
    row_len: int = 256
    e74_rows: int = 32  # stage 0's K0, E77's hook test
    fixed_rows: int = 8  # the readings' rows, and the dtype floor's: the first of E74's
    pool_each: int = 32  # training rows: maths and FineWeb, none of them a reading row
    lr: float = 1e-6  # the rate the rehearsals at 96 and 384 columns chose
    read_every: int = 50
    ckpt_every: int = 200
    target_kl: float = 0.01  # control C's size, amend1's B_rate target
    pc_iters: int = 4  # amend1's size search
    pc_seed: int = 0
    e74_seed: int = 74  # E74 Amendment 2's stochastic stream
    batch_seed: int = 0
    mask_seed: int = 76_000
    route_seed: int = 76_100
    read_seed: int = 76_200  # every reading draws the same masks per row
    check_seed: int = 76_300

    def __post_init__(self) -> None:
        assert self.fixed_rows <= self.e74_rows and self.ckpt_every % self.read_every == 0


class Arm(NamedTuple):
    name: str
    steps: int
    masked: str | None  # None: clean term alone; "all": all 196 sites masked; "library": the library's routing
    control: bool = False  # teacher W0 + random_change, else the update


C = Cfg()
ARMS = (Arm("C", 300, "all", True), Arm("A", 1200, None), Arm("B", 1200, "all"), Arm("L", 1200, "library"))


def log(msg: str) -> None:
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with (RESULTS / "run.log").open("a") as f:
        f.write(line + "\n")


def peak() -> str:
    return f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if torch.cuda.is_available() else "no GPU"


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def stop(rule: str, detail: str) -> None:
    """A plan stop: results/stopped.json, the reading in the log, exit 1. A rerun refuses while the file exists."""
    log(f"STOP {rule}: {detail}")
    write_json(RESULTS / "stopped.json", {"rule": rule, "detail": detail})
    sys.exit(f"STOP {rule}: {detail}")


def kl_sum(p: Tensor, q: Tensor) -> float:
    """KL(p ‖ q) summed over the positions: fp32 per position, summed in double."""
    return float(kl_per_position(p, q).double().sum())


def rows_all() -> tuple[Tensor, Tensor]:
    """E74's rows (maths, first 32), and the job's rows: the fixed reading rows, then the training pool, FineWeb after maths
    (maths rows past E74's 32, FineWeb evaluation rows: all on disk, and none a reading row)."""
    maths = json.loads(MATHS.read_text())["rows"]
    web = json.loads(WEB.read_text())["rows"]
    cut = [r[: C.row_len] for r in (*maths[: C.fixed_rows], *maths[C.e74_rows : C.e74_rows + C.pool_each], *web[: C.pool_each])]
    return torch.tensor([r[: C.row_len] for r in maths[: C.e74_rows]]), torch.tensor(cut)


def pools() -> tuple[range, range]:
    """Indices into `rows_all()[1]` of the maths and FineWeb training rows."""
    a, b = C.fixed_rows, C.fixed_rows + C.pool_each
    return range(a, b), range(b, b + C.pool_each)


def g_path(out: Path, i: int) -> Path:
    return out / f"g_{i}.safetensors"


def save_g(out: Path, i: int, g: dict[str, Tensor]) -> None:
    """One row's g, bf16, on the node's disk (0.35 GB; another node computes it again)."""
    out.mkdir(parents=True, exist_ok=True)
    tmp = g_path(out, i).with_suffix(".tmp")
    save_file({s: v.to("cpu", torch.bfloat16).contiguous() for s, v in g.items()}, str(tmp))
    os.replace(tmp, g_path(out, i))


def load_g(out: Path, i: int) -> dict[str, Tensor]:
    """In SITES order: the order the stochastic stream reads them."""
    t = load_file(str(g_path(out, i)))
    return {s: t[s] for s in SITES}
