"""E309 (cell P2.79): writer 2a's half of the NEW-code reproduction. Every reading of moves 2-4 that old.py computes
(size, erank, summed norm, prior in both forms, KL in both directions, every re-gating share variant, and the panel's
share, gain, KL, prior, size and erank), under the same keys, through the new shared functions only (`commute.world`,
`windows`, `items`, `share`, `kl`, `prior`; no `load_experiment`). Writes results/new_a.json, one record per key:
the value and the settings that tell its variant apart. compare.py holds it against old_A/old_B.json.

  --part data       CPU (`fleet/cpu-run --mem 16G`): erank, size, summed norm, log10 of a saved prior
  --part gpu        one L40 (new_a.sbatch), the W0 check inside `World.load` first
  --part gpu --dry  CPU: builds every GPU reading's rows and updates and stops before the model is loaded

    ~/commute-agents/fleet/cpu-run --mem 16G python3 experiments/reading_refactor/new_a.py --part data
    sbatch --comment=E309 experiments/reading_refactor/new_a.sbatch
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import elicit as EL
from commute import kl as KL
from commute import prior as PR
from commute import realistic as RZ
from commute import share as SH
from commute import steering as ST
from commute import update_spectrum as US
from commute import vpd, world
from commute.data import cached_rows
from commute.items import DW, Item, load_dw, resolve
from commute.jobs import EXPERIMENTS, hard_exit
from commute.local_faith import MIN_GAIN
from commute.windows import Rows, e136_tasks, read_rows

RESULTS = Path(__file__).resolve().parent / "results"
E62_COPY = "experiments/induction_cue/results/large/copy_s0.pt"
SEQ_RL = "experiments/procedural_rl/results/large/states/sequence_rl_s0.pt"
LOWER_128 = "experiments/convergence_panel/results/large/states/lower_s0_step128.pt"
ROUTE_FLOOR = SH.Floor(1e-6, two_sided=True)  # local_faith.route_split
LO = {"lo": ST.logodds_at}


def predictions(exp: str) -> dict[str, Any]:
    return json.loads((EXPERIMENTS / exp / "results" / "predictions.json").read_text())


def saved_file(path: str) -> Item:
    """A saved update outside the item families, read as saved."""
    return Item(path, path, None)


class Out:
    """results/new_a.json, rewritten after every reading (a part run later adds to what an earlier part wrote)."""

    def __init__(self) -> None:
        self.path = RESULTS / "new_a.json"
        self.d: dict[str, Any] = json.loads(self.path.read_text()) if self.path.is_file() else {}
        self.errors: list[str] = []

    def put(self, key: str, value: Any, **settings: Any) -> None:
        self.d[key] = {"value": value, "settings": settings}
        self.write()
        print(f"[{key}] {value!r}", flush=True)

    def write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1, default=str))
        tmp.replace(self.path)

    def run(self, name: str, f: Callable[[Out], None]) -> None:
        t0 = time.time()
        try:
            f(self)
        except Exception:  # noqa: BLE001 (one reading's failure must not lose the others: recorded, the job exits 1)
            self.errors.append(name)
            self.d[f"_error/{name}"] = traceback.format_exc()
            print(f"[{name}] ERROR\n{traceback.format_exc()}", flush=True)
        self.d.setdefault("_seconds", {})[name] = time.time() - t0
        self.write()


# ----------------------------------------------------------------------------- data part (CPU)


def spectra_and_sizes(o: Out) -> None:
    W0 = vpd.checkpoint_weights()[0]
    for name, path in (("e62/copy_s0", E62_COPY), ("sequence_rl_s0", SEQ_RL), ("lower_s0_step128", LOWER_128)):
        d = load_dw(saved_file(path))
        o.put(f"erank/{name}", US.state(d)["erank"], fn="update_spectrum.state", dtype="fp64 svdvals")
        o.put(f"size_rel/{name}", RZ.update_size(d, W0), fn="realistic.update_size", W0="checkpoint")
    o.put("summed_norm/e62/copy_s0", EL.summed_norm(load_dw(saved_file(E62_COPY))), fn="elicit.summed_norm (scorecard size)")


def saved_priors(o: Out) -> None:
    stage1 = json.loads((EXPERIMENTS / "prior_vs_share" / "results" / "stage1.json").read_text())
    for name in ("e62_copy", "e46_ident32"):
        o.put(f"prior_log10_geo/{name}@saved_P", PR.prior_log10_geo(stage1[name]["prior"]), fn="prior.prior_log10_geo", P="prior_vs_share stage1")


# ----------------------------------------------------------------------------- gpu part


@dataclass
class Inputs:
    """Every GPU reading's rows and updates, built on CPU before the model loads."""

    rows: dict[str, Rows]
    dw: dict[str, DW]
    kl_rows: dict[str, Tensor]  # procedural_rl's and the panel's cached val rows
    s_rl: dict[str, Any]
    case_floor: float


def inputs() -> Inputs:
    case_sha = json.loads((EXPERIMENTS / "case_label" / "results" / "stage1.json").read_text())["states"]["real_s0"]["sha256"]
    s_rl, s_cp = predictions("procedural_rl")["settings"], predictions("convergence_panel")["settings"]
    return Inputs(
        rows={t: read_rows(t, "held") for t in ("c3/cipher", "e53/lowercase", "case_label/real", "e46/ident32")},
        dw={"c3/cipher_s0": load_dw(resolve("c3/cipher_s0")), "e53/lowercase_s0": load_dw(resolve("e53/lowercase_s0")),
            "case_real_s0": load_dw(Item("case_label/real_s0", "experiments/case_label/results/large/dW_real_s0.pt", case_sha)),
            "sequence_rl_s0": load_dw(saved_file(SEQ_RL)), "lower_s0_step128": load_dw(saved_file(LOWER_128))},
        kl_rows={"procedural_rl": cached_rows("val", 0, s_rl["kl_rows"]), "panel": cached_rows("val", 0, s_cp["kl_rows"])},
        s_rl=s_rl,
        case_floor=predictions("real_vs_twin")["thresholds"]["gain_floor"],
    )  # fmt: skip


def share_e136(w: world.World, x: Inputs, o: Out) -> None:
    """E136 `.trained`: log-odds at the inventory's batch, over the same forwards' bf16 gain, floor two-sided 1e-6."""
    for state, task in (("c3/cipher_s0", "c3/cipher"), ("e53/lowercase_s0", "e53/lowercase")):
        batch = e136_tasks()[task]["batch"]
        c = SH.share_cells(w.model, w.W0, x.dw[state], x.rows[task], LO, batch)
        g = SH.cells_gain(c, "lo")
        o.put(f"share/{state}@b{batch}", SH.share(c, "lo", g, ROUTE_FLOOR), batch=batch, rows=f"{task} held ({len(x.rows[task].toks)})", unit="lo", gain="bf16 cells", floor="|gain| >= 1e-6", gain_value=g)


def share_e211(w: world.World, x: Inputs, o: Out) -> None:
    """E211's ctl: E53's scored windows at batch 64, floor MIN_GAIN one-sided."""
    c = SH.share_cells(w.model, w.W0, x.dw["e53/lowercase_s0"], x.rows["e53/lowercase"], LO, 64)
    g = SH.cells_gain(c, "lo")
    o.put("share/e53/lowercase_s0@b64", SH.share(c, "lo", g, SH.Floor(MIN_GAIN, two_sided=False)), batch=64, rows="e53/lowercase held (500)", unit="lo", gain="bf16 cells", floor="gain >= 1", gain_value=g)


def share_e140(w: world.World, x: Inputs, o: Out) -> None:
    """real_vs_twin stage 2: case real's first 256 fit windows, log-odds and the two-label contrast, batch 64."""
    r = x.rows["case_label/real"]
    assert r.ab is not None
    c = SH.share_cells(w.model, w.W0, x.dw["case_real_s0"], r, LO | {"contrast": ST.contrast_at(r.ab)}, 64)
    for u in ("lo", "contrast"):
        o.put(f"share/case_real_s0@{u}", SH.share(c, u, SH.cells_gain(c, u), SH.Floor(x.case_floor, two_sided=False)), batch=64, rows=f"case_label fit[:{len(r.toks)}]", unit=u, gain="bf16 cells", floor=f"gain >= {x.case_floor}")


def prior_forward(w: world.World, x: Inputs, o: Out) -> None:
    """E133 `forward_prior`: the geometric-mean P on a kv task's held-out windows, all rows in one forward."""
    r = x.rows["e46/ident32"]
    o.put("prior_geo/e46_ident32", PR.prior_geo(w.net, r, len(r.toks)), rows=f"e46/ident32 held ({len(r.toks)})", batch=len(r.toks), dtype="fp32", unit="P (geometric mean)")


def kl_procedural(w: world.World, x: Inputs, o: Out) -> None:
    s, toks, d = x.s_rl, x.kl_rows["procedural_rl"], x.dw["sequence_rl_s0"]
    rows = f"cached val 0..{s['kl_rows']}"
    o.put("kl_base_to_ft/sequence_rl_s0", KL.kl_base_to_ft(w.net, toks, d, s["kl_batch"]), batch=s["kl_batch"], rows=rows, positions="all", unit="nats per token")
    o.put("kl_ft_to_base/sequence_rl_s0", KL.kl_ft_to_base(w.net, toks, d, s["kl_batch"], slice(None)), batch=s["kl_batch"], rows=rows, positions="all", unit="nats per token")


def panel(w: world.World, x: Inputs, o: Out) -> None:
    """E272 `readings` on lower_s0_step128 (its read is E53's lowercase scored windows), the parts of moves 2-4."""
    r, d = x.rows["e53/lowercase"], x.dw["lower_s0_step128"]
    c = SH.share_cells(w.model, w.W0, d, r, LO, 64)
    g = SH.cells_gain(c, "lo")
    common = {"rows": f"lower read ({len(r.toks)})", "unit": "lo"}
    o.put("share/lower_s0_step128", SH.share(c, "lo", g, SH.Floor(MIN_GAIN, two_sided=False)), **common, batch=64, gain="bf16 cells", floor="gain >= 1")
    o.put("gain/lower_s0_step128", g, **common, batch=64, dtype="bf16 cells")
    o.put("kl_base_to_ft/lower_s0_step128", KL.kl_base_to_ft(w.net, x.kl_rows["panel"], d, 8), rows=f"cached val 0..{len(x.kl_rows['panel'])}", batch=8, unit="nats per token")
    o.put("prior_mean_lo/lower_s0_step128", PR.prior_mean_lo(w.net, r, ST.logodds_at, 64), **common, batch=64)
    o.put("size_rel/lower_s0_step128", RZ.update_size(d, w.W0), W0="live (checkpoint-checked)")
    o.put("erank/lower_s0_step128", US.state(d)["erank"])


DATA = {"spectra_and_sizes": spectra_and_sizes, "saved_priors": saved_priors}
GPU = {"share_e136": share_e136, "share_e211": share_e211, "share_e140": share_e140, "prior_forward": prior_forward, "kl_procedural": kl_procedural, "panel": panel}


def dry(x: Inputs) -> None:
    for t, r in x.rows.items():
        print(f"rows {t}: {tuple(r.toks.shape)} pos {r.pos} tgt {None if r.tgt is None else tuple(r.tgt.shape)} sha256 {r.sha256[:12]} ({r.source})", flush=True)
    for n, d in x.dw.items():
        print(f"dW {n}: {len(d)} matrices, {sum(v.numel() for v in d.values())} entries", flush=True)
    print(f"kl rows {[(k, tuple(v.shape)) for k, v in x.kl_rows.items()]}; case ab {x.rows['case_label/real'].ab}, floor {x.case_floor}; kl batch {x.s_rl['kl_batch']}", flush=True)


def main(part: str, only: list[str], is_dry: bool) -> int:
    o = Out()
    if part == "data":
        for n, f in DATA.items():
            if not only or n in only:
                o.run(n, f)
    else:
        x = inputs()
        if is_dry:
            dry(x)
            return 0
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # as every old run
        w = world.World.load("cuda")
        o.d["_meta"] = {"card": torch.cuda.get_device_name(), "torch": torch.__version__}
        for n, gf in GPU.items():
            if not only or n in only:
                o.run(n, lambda o, gf=gf: gf(w, x, o))
        print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    print(f"NEW_A {part} DONE; errors: {o.errors or 'none'}", flush=True)
    return 1 if o.errors else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("data", "gpu"), required=True)
    ap.add_argument("--only", nargs="*", default=[], help="run only these groups (names in DATA / GPU)")
    ap.add_argument("--dry", action="store_true", help="gpu part: build the inputs on CPU and stop before the model loads")
    a = ap.parse_args()
    hard_exit(main(a.part, a.only, a.dry))
