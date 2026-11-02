"""E67: the weight-side read, do-nothing side. On 128 Pile rows, every position: the change in Γ's values, and the re-switching term's overlap share S_R with the update's own output
change (`commute.weight_side`), on the learned-nothing fine-tune against the copy fine-tunes; then both on the 64 rows where r failed. No training.
Plan reports/plans/2026-11-02_plan_e67-weight-side-read.md; rules and sizes results/predictions.json (committed first; the run refuses otherwise). cluster/weight_side_read.sbatch.

  a the control, alone: W1 = W0 on one chunk reads exactly 0; a miss is exit 1 and no state is read
  b the states in the filed order, one read each, then the held rows; a dead or non-finite read is exit 1
  c the filed verdict. Results after every stage in results/run.json, every number printed. fp32 throughout: no autocast.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor

from commute import split, vpd
from commute import weight_side as WS
from commute.data import batches, pile_rows
from commute.keydep import sha256

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions.json", RESULTS / "run.json"
E46 = HERE.parent / "regating_table" / "results"
CODE_PATHS = ("experiments/weight_side_read/run.py", "src/commute/weight_side.py")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


@dataclass
class Job:
    model: Any
    W0: dict[str, Tensor]
    comps: dict[str, tuple[Tensor, Tensor]]
    chunks: list[Tensor]
    held: list[Tensor]  # the rows where r failed: E36's code_held.pt
    P: dict[str, Any]
    out: dict[str, Any]
    results: Path
    fine_tunes: Path

    def note(self, key: str, rec: dict[str, Any]) -> None:
        self.out[key] = rec
        tmp = self.results.with_suffix(".tmp")  # a kill mid-write must not leave unreadable JSON
        tmp.write_text(json.dumps(self.out, indent=1, sort_keys=True))
        tmp.replace(self.results)
        log(f"[{key}] {json.dumps(rec)}")

    def stop(self, verdict: str, why: str) -> NoReturn:
        self.note("stopped", {"why": why, "verdict": verdict})
        sys.exit(f"STOP {verdict}: {why}: tell the lead")

    def fine_tune(self, name: str) -> dict[str, Tensor]:
        return torch.load(self.fine_tunes / f"{name}.pt", weights_only=False, map_location="cpu")


def stage_zero(J: Job) -> None:
    ms = WS.read_state(J.model, J.W0, J.comps, {k: torch.zeros_like(v) for k, v in J.W0.items()}, J.chunks[:1])
    bad = [k for k, m in ms.items() if not m.is_zero()]
    J.note("stage_a_zero", {"matrices_not_exactly_zero": bad, "mean_abs_dg": WS.mean_abs_dg(ms)})
    if bad:
        J.stop("unreadable", f"zero: W1 = W0 reads non-zero on {bad}: a repeated Γ pass is not exact")


def stage_state(J: Job, name: str, dW: dict[str, Tensor], chunks: list[Tensor]) -> dict[str, Any]:
    t = time.time()
    ms = WS.read_state(J.model, J.W0, J.comps, dW, chunks)
    rec = {**WS.pooled(ms), "per_matrix": WS.per_matrix(ms), "seconds": time.time() - t}
    J.note(f"state_{name}", rec)
    if not math.isfinite(rec["S_R"]):
        J.stop("unreadable", f"{name}: S_R reads {rec['S_R']}")
    if rec["mean_abs_dg"] == 0:  # S_R's denominator is above 0 here, so the update is not zero: a g1 that equals g0 would read as the lean
        J.stop("dead_read", f"{name}: Γ's values did not move at all on a nonzero update")
    return rec


def run(J: Job) -> None:
    stage_zero(J)
    log("[control] passed; reading the states")
    states, reads = J.P["states"], {}
    for name in states["order"]:
        if name == "gaussian":
            sigma = split.input_second_moments(J.model, J.chunks, list(J.comps))
            dW = WS.matched_gaussian(J.fine_tune("ident32_s0"), sigma, J.P["config"]["gaussian_seed"])
        else:
            dW = J.fine_tune(name)
        reads[name] = stage_state(J, name, dW, J.chunks)
    for name in states["held_rows"]:
        reads[f"{name}_held"] = stage_state(J, f"{name}_held", J.fine_tune(name), J.held)
    s = {n: r["S_R"] for n, r in reads.items()}
    v = WS.verdict(s, J.P["thresholds"])
    v["mean_abs_dg"] = {n: r["mean_abs_dg"] for n, r in reads.items()}
    v["leave_one_chunk_out_spread"] = {n: r["leave_one_chunk_out"]["spread"] for n, r in reads.items()}
    J.note("verdict", v)
    log(f"*** VERDICT {v['verdict']}: S_R code_step10 {s['code_step10']:.4f}, code_step100 {s['code_step100']:.4f}, ident32 {s['ident32_s0']:.4f} {s['ident32_s1']:.4f}, gaussian {s['gaussian']:.4f}; "
        f"on the rows r failed: code_step10 {s['code_step10_held']:.4f}, ident32_s0 {s['ident32_s0_held']:.4f} (report-only)")
    if v["verdict"] != "drop":
        log(f"*** LOUD: {v['verdict']} is against the filed lean (drop): tell the lead before anything is built on it")
    if v["gaussian_like_copy"]:
        log("*** LOUD: the matched Gaussian reads like the copy fine-tunes")


def load(a: argparse.Namespace) -> Job:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    if OUT.exists():
        sys.exit(f"{OUT} exists: move it aside, this run deletes nothing")
    P, s0a = json.loads(PRED.read_text()), json.loads((E46 / "stage0a.json").read_text())["groups"]
    cfg, e36 = P["config"], s0a["e36/code"]
    want = {f"ident32_{s}": s0a["e12/ident32"]["states"][s]["sha256"] for s in ("s0", "s1")}
    want |= {f"code_step{k}": e36["states"][k]["sha256"] for k in ("10", "100")} | {"code_held": e36["held_sha256"]}
    for n, sha in want.items():
        if sha256(E46 / "large" / f"{n}.pt") != sha:
            sys.exit(f"{n}.pt is not the file stage0a.json recorded")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = vpd.load_component_model(a.device)
    rows = pile_rows(cfg["rows_split"], 0, cfg["rows_cached"])[: cfg["rows"]]
    held = torch.load(E46 / "large" / "code_held.pt", weights_only=False, map_location="cpu")
    out = {"git_head": git("rev-parse", "HEAD").stdout.strip(), "code_status": git("status", "--porcelain", "--", *CODE_PATHS).stdout, "predictions_sha256": sha256(PRED),
           "host": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}
    return Job(model, vpd.target_weights(model), vpd.component_weights(model), batches(rows, cfg["chunk"]), batches(held, cfg["chunk"]), P, out, OUT, E46 / "large")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    J = load(ap.parse_args())
    log(f"[host] {J.out['host']}: fp32, no autocast; {len(J.chunks)} chunks, {len(J.held)} on the rows r failed; code status {J.out['code_status']!r}")
    run(J)


if __name__ == "__main__":
    main()
