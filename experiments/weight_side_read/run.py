"""E67: the weight-side read. On 128 Pile rows, every position: the change in Γ's values, and each term's overlap share S_X with the update's own output change
(`commute.weight_side`), on the learned-nothing fine-tune against the copy fine-tunes. No training. Plan reports/plans/2026-11-02_plan_e67-weight-side-read.md;
rules and sizes results/predictions.json (committed first; the run refuses otherwise; never edited). cluster/weight_side_read.sbatch.

  a control, alone, cheapest first; any miss is exit 1 and no state is read:
      size      the students' rebuilt rel_norm_max against E12 Amendment 1's, and student / teacher against its table (no Γ, no a)
      zero      W1 = W0 on one chunk: every numerator and |g1 - g0| exactly 0
      ones      every gate on, one chunk: the least-squares state against 1 - omega - 2 cross + second, the students' S_D = 1
  b the states in ORDER, one read each: do-nothing and least squares share W1, so g1 and S_R are the same pass. The shares' sum is checked at every state.
  c the filed verdict. Results after every stage in results/run.json, every number printed. fp32 throughout: no autocast.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
from torch import Tensor

from commute import component_training, saved_states, split, vpd
from commute import weight_side as WS
from commute.data import batches, pile_rows
from commute.keydep import sha256

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions.json", RESULTS / "run.json"
E46, E12 = HERE.parent / "regating_table" / "results", HERE.parent / "new_vs_existing" / "results"
ORDER = ("ident32_s0", "code_step10", "ident32_s1", "code_step100", "rand32_s0", "rand32_s1", "gaussian")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


@dataclass
class Job:
    model: Any
    W0: dict[str, Tensor]
    comps: dict[str, tuple[Tensor, Tensor]]
    uv: dict[str, tuple[Tensor, Tensor]]  # alive columns, theory layout, CPU
    alive: dict[str, Tensor]
    bases: dict[str, split.Basis]
    chunks: list[Tensor]
    P: dict[str, Any]
    out: dict[str, Any]
    results: Path
    fine_tunes: Path
    students: Path
    filed: dict[str, Any]  # E12 Amendment 1's students

    def note(self, key: str, rec: dict[str, Any]) -> None:
        self.out[key] = rec
        tmp = self.results.with_suffix(".tmp")  # a kill mid-write must not leave unreadable JSON
        tmp.write_text(json.dumps(self.out, indent=1, sort_keys=True))
        tmp.replace(self.results)
        log(f"[{key}] {json.dumps(rec)}")

    def stop(self, why: str) -> NoReturn:
        self.note("stopped", {"why": why, "verdict": "unreadable"})
        sys.exit(f"STOP: {why}: tell the lead")

    def fine_tune(self, name: str) -> dict[str, Tensor]:
        return torch.load(self.fine_tunes / f"{name}.pt", weights_only=False, map_location="cpu")

    def least_squares(self, dW: dict[str, Tensor]) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
        return split.split_state(dW, self.uv, self.bases)

    def offsets(self, st: dict[str, tuple[Tensor, Tensor, Tensor]]) -> dict[str, WS.Offsets]:
        return WS.offsets_from({k: (v[1], v[2]) for k, v in st.items()}, self.uv, self.alive)


def stage_size(J: Job) -> None:
    th, rec = J.P["thresholds"], {}
    for name, table in J.P["size_ratio_table"].items():
        dWs, _ = saved_states.offsets_state(J.students / f"{name}.pt", J.uv)
        rel, teacher = WS.rel_norm_max(dWs, J.W0), WS.rel_norm_max(J.fine_tune(name), J.W0)
        filed = J.filed[name]["curve"][-1]["rel_norm_max"]
        rec[name] = {"rel": rel, "filed": filed, "teacher": teacher, "ratio": rel / teacher, "table": table}
        if not (abs(rel / filed - 1) <= th["size_rel_tol"] and abs(rel / teacher - table) <= th["size_ratio_tol"]):
            J.note("stage_a_size", rec)
            J.stop(f"size: {name} rebuilt {rel} against filed {filed}, student / teacher {rel / teacher} against the table's {table}")
    J.note("stage_a_size", rec)


def stage_zero(J: Job) -> None:
    ms = WS.read_state(J.model, J.W0, J.comps, {k: torch.zeros_like(v) for k, v in J.W0.items()}, {}, J.chunks[:1])
    bad = [k for k, m in ms.items() if not m.is_zero()]
    all_of = WS.pooled(ms)
    J.note("stage_a_zero", {"matrices_not_exactly_zero": bad, "mean_abs_dg": all_of["mean_abs_dg"], "den": all_of["den"]})
    if bad:
        J.stop(f"zero: W1 = W0 reads non-zero on {bad}: a repeated Γ pass is not exact, or a term is wrong")


def stage_ones(J: Job) -> None:
    """Every gate on, on one chunk's real a. The identities' tolerances come from floors measured on saved states with white inputs (plan)."""
    th, rec = J.P["thresholds"], {}
    a, _ = WS.inputs_and_gates(J.model, J.chunks[0])
    dW = J.fine_tune("ident32_s0")
    st, device = J.least_squares(dW), next(J.model.parameters()).device
    off = J.offsets(st)
    for k in st:
        x, (U, V) = a[k], J.comps[k]
        o = WS.Offsets(off[k].dU.to(device), off[k].dV.to(device))
        parts = WS.split_identity(dW[k], J.bases[k], st[k][1] - J.uv[k][0].float(), st[k][2] - J.uv[k][1].float(), (x.double().T @ x.double()).cpu() / len(x))
        D = WS.changed(x, torch.ones(len(x), V.shape[1], device=device), U, V, o)
        got = WS.dot(D, D) / len(x) / parts["den"]
        rec[k] = {**parts, "got": got, "miss": got - parts["predicted"]}
    worst = max(abs(r["miss"]) for r in rec.values())
    J.note("stage_a_ones_split", {"matrices": rec, "worst_miss": worst, "tol": th["identity_tol"]})
    if not worst <= th["identity_tol"]:
        J.stop(f"ones: the least-squares state misses its identity by {worst}")
    for name in J.P["config"]["ones_students"]:
        dWs, uv2 = saved_states.offsets_state(J.students / f"{name}.pt", J.uv)
        offs, num, den = WS.offsets_from(uv2, J.uv, J.alive), 0.0, 0.0
        for k, (U, V) in J.comps.items():
            o = WS.Offsets(offs[k].dU.to(device), offs[k].dV.to(device))
            Ma = a[k] @ dWs[k].to(device).T
            num, den = num + WS.dot(WS.changed(a[k], torch.ones(len(a[k]), V.shape[1], device=device), U, V, o), Ma), den + WS.dot(Ma, Ma)
        J.note(f"stage_a_ones_{name}", {"S_D": num / den, "tol": th["identity_tol"]})
        if not abs(num / den - 1) <= th["identity_tol"]:
            J.stop(f"ones: {name}'s S_D with every gate on reads {num / den}, not 1")


def stage_state(J: Job, name: str, dW: dict[str, Tensor]) -> dict[str, Any]:
    t = time.time()
    ms = WS.read_state(J.model, J.W0, J.comps, dW, J.offsets(J.least_squares(dW)), J.chunks)
    rec = {**WS.pooled(ms), "per_matrix": WS.per_matrix(ms), "seconds": time.time() - t}
    J.note(f"state_{name}", rec)
    for side in ("split", "nothing"):
        if not abs(rec[side]["sum"] - 1) <= J.P["thresholds"]["sum_tol"]:
            J.stop(f"{name} {side}: the shares sum to {rec[side]['sum']}")
    return rec


def run(J: Job) -> None:
    stage_size(J)
    stage_zero(J)
    stage_ones(J)
    log("[control] passed; reading the states")
    reads = {}
    for name in ORDER:
        if name == "gaussian":
            dW = WS.matched_gaussian(J.fine_tune("ident32_s0"), WS.second_moments(J.model, J.chunks), J.P["config"]["gaussian_seed"])
        else:
            dW = J.fine_tune(name)
        reads[name] = stage_state(J, name, dW)
    th, s_r = J.P["thresholds"], {n: r["nothing"]["S_R"] for n, r in reads.items()}
    v = WS.verdict(s_r, th)
    v["rand32_seed_spread"] = abs(s_r["rand32_s0"] - s_r["rand32_s1"])
    v["mean_abs_dg"] = {n: r["mean_abs_dg"] for n, r in reads.items()}
    J.note("verdict", v)
    log(f"*** VERDICT {v['verdict']}: S_R code_step10 {s_r['code_step10']:.4f}, code_step100 {s_r['code_step100']:.4f}, ident32 {s_r['ident32_s0']:.4f} {s_r['ident32_s1']:.4f}, gaussian {s_r['gaussian']:.4f}")
    if v["verdict"] in ("keep", "reversed", "code_negative", "seeds_disagree"):
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
    cfg = P["config"]
    want = {f"{t}_{s}": s0a[f"e12/{t}"]["states"][s]["sha256"] for t in ("ident32", "rand32") for s in ("s0", "s1")}
    want |= {f"code_step{k}": s0a["e36/code"]["states"][k]["sha256"] for k in ("10", "100")}
    for n, sha in want.items():
        if sha256(E46 / "large" / f"{n}.pt") != sha:
            sys.exit(f"{n}.pt is not the file stage0a.json recorded")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = vpd.load_component_model(a.device)
    alive = saved_states.load_alive(a.alive)
    uv = {k: (u.cpu(), v.cpu()) for k, (u, v) in component_training.alive_uv(model, alive).items()}
    rows = pile_rows(cfg["rows_split"], 0, cfg["rows_cached"])[: cfg["rows"]]
    out = {"git_head": git("rev-parse", "HEAD").stdout.strip(), "predictions_sha256": sha256(PRED), "host": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}
    return Job(model, vpd.target_weights(model), vpd.component_weights(model), uv, alive, split.bases(uv, cfg["split_tol"]), batches(rows, cfg["chunk"]), P, out, OUT,
               E46 / "large", E12 / "large" / "amend1", json.loads((E12 / "new_vs_existing_amend1.json").read_text())["students"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large")
    J = load(ap.parse_args())
    log(f"[host] {J.out['host']}: fp32, no autocast; {len(J.chunks)} chunks")
    run(J)


if __name__ == "__main__":
    main()
