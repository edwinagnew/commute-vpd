"""E86: does any saved update's gain build gradually, and does the clean table still work on it? ($0, one cluster L40 per update, debug QoS.) Plan:
reports/plans/2026-11-04_plan_e86-update-fractions.md; rules: rules.py, results/predictions.json (the run refuses to start without it, and while
results/stopped.json exists). Nothing is measured here that E75 or E80 did not define.

One update per job, u1 first and alone. In order, cheapest first, each read written to results/<update>/<read>.json as it ends (a rerun skips it),
results/score_<update>.json rescored, every number in results/run.log; a control's miss writes results/stopped.json, exit 1:
  gain   the mean response-token loss of W0 + t δW, no masks (E80 Amendment 2's path: `change_hooks`, `gold_loss`), before the components load.
  clean  E72's clean read of W0 twice (the floor), FT, then E75 Amendment 1's random and signs: no split needed.
  X      of E44's stage S, only the splits' X and the adjoint's c, by the same calls as `s_site` and `site_products`. The rest of stage S (piece
         energies, nulls), which nothing here reads, is most of its 23 min. On --out, gone when the update is done. u1's and gold's gap to E75 checks it.
  parts  in-span, remainder, often-on and rarely-on halves, adjoint; then the plain share left in Δ.
Past --wall-min, between reads: exit 0, resubmit to resume. No Γ is ever loaded.

Usage (repo root; -P and PYTHONPATH as cluster/qwen_d1_fractions.sbatch, so `run` means qwen_maps/run.py for the frozen reader):
  PYTHONPATH=experiments/qwen_maps:experiments/qwen_pass_test:src .venv/bin/python -P experiments/qwen_d1_fractions/run.py \
    --update u1 --decomp DIR --out /ephemeral/$USER/e86 --wall-min 26
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import Tensor
from transformers import AutoTokenizer

from commute import qwen_eval, qwen_maps, split
from commute import qwen_tofu as tofu
from commute.jobs import hard_exit, load_experiment
from commute.qwen_sft import Seq, gold_loss, load, pack, sha256
from commute.qwen_teacher import change_hooks
from commute.qwen_vpd import SITES

HERE = Path(__file__).resolve().parent
RESULTS, PRED, STOPPED = HERE / "results", HERE / "results" / "predictions.json", HERE / "results" / "stopped.json"
E80_OUT, E61_GOLD = HERE.parent / "qwen_d1_adversary" / "results" / "amend2.json", HERE.parent / "qwen_recipe" / "results" / "train_gold_1e-5.json"
E81 = HERE.parent / "qwen_tofu" / "results"
NEED_GB = 45.6  # E75's peak on this card, with Γ; this job loads none, so its need is below that, by an amount no file measures
GOLD_BATCH, TOFU_BATCH, TOFU_MICRO, TOFU_SEED = 8, 16, 8, 42  # E80 Amendment 2's; E81's batch, micro-batch and seed
N_ROWS, STORE_ERR_MAX, FILE_TOL, FIRST_TOL = 128, 1e-3, 1e-6, 1e-3  # qwen_maps/run.py's; E81's filed `files` and step-1 bands


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rules = _load("d1_fractions_rules", HERE / "rules.py")
e75 = _load("rl_reads_amend1", HERE.parent / "qwen_rl_reads" / "amend1.py")  # its Reader: E72's clean read, with random and signs
pr, a1, DEV, F64 = e75.pr, e75.a1, e75.DEV, e75.F64
e80 = load_experiment("qwen_d1_adversary", "amend2")  # gold_seqs, moved; only those: its log and stop write E80's files
UPDATES = {"u1": pr.P.updates[0], "u2": pr.P.updates[1],  # u2 is the gold fine-tune, E75's name for it
           "tofu": pr.Update("tofu", pr.LARGE / "qwen_tofu" / "dW" / "a2_2e-5" / "step_1250.safetensors", 1250, "tofu", N_ROWS)}


def tofu_seqs(tok: Any, end: int, name: str) -> list[Seq]:
    return [tofu.seq(tok, end, r["question"] + tofu.ASK, r["answer"]) for r in tofu.rows(name)]


def tofu_rows(tok: Any, end: int) -> tuple[list[Seq], Tensor]:
    """The 800 rows, each checked to be a row E81 trained on, in that wording; and 128 table rows of 256 evenly from what they pack into."""
    trained = {(r["question"], r["answer"]) for r in tofu.rows("full")}
    held = [r for f in tofu.HELD_FILES for r in tofu.rows(f)]
    assert len(held) == 800 and all((r["question"], r["answer"]) in trained for r in held), "a row E81 did not train on"
    seqs = [s for f in tofu.HELD_FILES for s in tofu_seqs(tok, end, f)]
    packed = pack([ids for ids, _ in seqs], None, pr.P.row_len)
    return seqs, torch.tensor([packed[i * len(packed) // N_ROWS] for i in range(N_ROWS)])


def check_tofu(job: Any, rd: Any, u: Any, tok: Any, end: int) -> None:
    """The file: fp32, the 196 sites' shapes, its norm E81's record at the step. The rows: E81's first batch at W0 reads stage 0's loss."""
    dW, W0 = load(u.path), rd.sets["W0"]
    if set(dW) != set(SITES) or any(dW[s].dtype != torch.float32 or dW[s].shape != W0[s].shape for s in SITES):
        job.stop("dW_file", f"{u.path}: not fp32 tensors of the 196 sites' shapes")
    norm = math.sqrt(sum(float(dW[s].norm()) ** 2 for s in SITES))
    record = json.loads((E81 / "train_2e-5.json").read_text())["steps"][u.step - 1]["dw_norm"]
    rel = norm / math.sqrt(sum(float(W0[s].norm()) ** 2 for s in SITES))
    job.log(f"[dW tofu] {u.path}: sha256 {sha256(u.path)}, ‖δW‖ {norm:.9g}, record {record:.9g}, ‖δW‖/‖W0‖ {rel:.6g}")
    if abs(norm / record - 1) > FILE_TOL:
        job.stop("dW_file", f"{u.path}: ‖δW‖ {norm} against the record's {record}")
    first = tofu.stream(tofu_seqs(tok, end, "full"), TOFU_BATCH, TOFU_SEED, 1)
    got, want = gold_loss(job.model, first, TOFU_MICRO, DEV), json.loads((E81 / "stage0.json").read_text())["first_batch_loss"]
    job.log(f"[rows tofu] E81's first batch at W0 {got:.9g}, stage 0's {want:.9g}")
    if abs(got / want - 1) > FIRST_TOL:
        job.stop("rows", f"E81's first batch at W0 reads {got}, stage 0 {want}")


def stage_x(job: Any, t: int, wall: Callable[[], None]) -> None:
    """Per site, on --out in the format `Job.change` reads: fp32 X of least squares and of each half (stage S's calls), the adjoint's c on E44's Σ."""
    how, tol, gstats, t0 = job.cfg.basis, job.cfg.tol, job.gstats("maths"), time.time()
    job.site_dir(t).mkdir(parents=True, exist_ok=True)
    worst = 0.0
    for s in SITES:
        path = job.site_dir(t) / f"{s}.safetensors"
        if path.with_suffix(".json").is_file():
            continue
        wall()
        U_A, V_A = job.alive.uv(s, DEV, F64)
        M, S, idx = job.dw(t)[s].to(DEV, F64), job.sigma()[s].to(DEV), job.alive.idx[s].cpu()
        X = split.least_squares_x(M, split.basis_from(qwen_maps.make_basis(U_A, how), qwen_maps.make_basis(V_A, how), tol))
        worst = max(worst, qwen_maps.stored_error(X, U_A, V_A, S, split.weighted_energy(M, S)))
        arrays = {"X_ls": X}
        for half, pos in zip(("lower", "upper"), qwen_maps.halves(gstats[f"{s}.active"][idx], gstats[f"{s}.mean_g"][idx]), strict=True):
            pos = pos.to(DEV)
            Uh, Vh = U_A[:, pos], V_A[:, pos]
            arrays[f"X_{half}"] = split.least_squares_x(M, split.basis_from(qwen_maps.make_basis(Uh, how), qwen_maps.make_basis(Vh, how), tol))
            arrays[f"pos_{half}"] = pos
        tmp = path.with_suffix(".tmp")
        save_file(qwen_maps.to_disk(arrays), str(tmp))
        tmp.replace(path)
        a1.write_json(path.with_suffix(".json"), {"c": split.adjoint_scale(M, U_A, V_A, S)[0], "a": 0.0})
    job.drop_sigma()
    job.log(f"[X {job.update.name}] {time.time() - t0:.0f} s, stored error at most {worst:.3g} of ‖δW‖_Σ, {job.peaks()}")
    if worst > STORE_ERR_MAX:
        job.stop("stored_error", f"stored pieces miss float64 by {worst:.3g}")


def left_in_delta(job: Any, t: int) -> dict[str, float]:
    """‖Δ′ − Δ‖² / ‖δW‖², plain, pooled over the matrices: E71's measure, E80 Amendment 2's `scales`. The adjoint's c is E75's."""
    plain, size, t0 = {"ls": 0.0, "adjoint": 0.0}, 0.0, time.time()
    change = {n: job.change(n, t) for n in plain}
    for s in SITES:
        U_A, V_A = job.alive.uv(s, DEV, F64)
        M = job.dw(t)[s].to(DEV, F64)
        size += float((M * M).sum())
        for n in plain:
            d = e80.moved(change[n](s), U_A, V_A)
            plain[n] += float((d * d).sum())
    out = {n: v / size for n, v in plain.items()} | {"size": size}
    job.log(f"[left in Δ {job.update.name}] {json.dumps(out)}, {time.time() - t0:.0f} s, {job.peaks()}")
    return out


class Runner(pr.Runner):
    """E68's Runner for `step` and `wall`; these files, these rules. `anchors` are the update's filed references."""

    def __init__(self, job: Any, rd: Any, res: dict[str, dict], root: Path, wall_s: float, anchors: dict[str, Any]):
        super().__init__(job, rd, res, root, wall_s)
        self.anchors = anchors

    def put(self, u: Any, key: str, value: dict[str, Any]) -> None:
        self.res[u.name][key] = value
        a1.write_json(RESULTS / u.name / f"{key}.json", value)
        self.rescore(u)

    def rescore(self, u: Any) -> None:
        sc = rules.score(self.res[u.name], **self.anchors)
        a1.write_json(RESULTS / f"score_{u.name}.json", sc)
        self.job.log(f"[score {u.name}] {json.dumps(sc, default=float)}")
        if miss := rules.missed(sc):
            self.job.stop(miss[0], json.dumps({k: sc[k] for k in miss}, default=float))

    def fractions(self, u: Any, seqs: list[Seq]) -> None:
        """W0, FT, t = 1 down to 0.1, W0 again; u1 also t = 0.1 multiplied in fp32, no rule. As E80: t times the file's dtype, on the card."""
        job, model = self.job, self.job.model
        dw = {s: v.to(DEV) for s, v in job.dw(u.step).items()}

        def read(name: str, change: Callable[[str], Tensor] | None) -> None:
            if name in self.res[u.name].setdefault("fractions", {}):
                return
            self.wall()
            t0 = time.time()
            with change_hooks(model, change):
                x = gold_loss(model, seqs, GOLD_BATCH, DEV)
            job.log(f"[gain {u.name} {name}] {x:.12f} nats per response token, {time.time() - t0:.0f} s, {job.peaks()}")
            self.put(u, "fractions", self.res[u.name]["fractions"] | {name: x})

        read("W0", None)
        read("FT", dw.__getitem__)
        for t in rules.TS[::-1]:
            read(f"t{t}", lambda s, t=t: t * dw[s])
        read("W0_end", None)
        if u.name == "u1":
            read("t0.1_fp32", lambda s: 0.1 * dw[s].float())

    def table(self, u: Any) -> None:
        job, rd, t = self.job, self.rd, u.step
        self.step(u, "clean_W0", lambda: rd.clean("W0", t, 1.0))
        self.step(u, "clean_W0_again", lambda: rd.clean("W0", t, 1.0))
        for name in ("FT", *rules.WRONG):
            self.step(u, f"clean_{name}", lambda name=name: rd.clean(name, t, 1.0))
        if any(k not in self.res[u.name] for k in (*(f"clean_{n}" for n in rules.PARTS), "left_in_delta")):
            stage_x(job, t, self.wall)
        for name in rules.PARTS:
            self.step(u, f"clean_{name}", lambda name=name: rd.clean(name, t, 1.0))
        self.step(u, "left_in_delta", lambda: left_in_delta(job, t))
        shutil.rmtree(job.out, ignore_errors=True)
        job.log(f"[{u.name}] done; {job.out} removed, {job.peaks()}")


def stored(name: str) -> dict[str, Any]:
    """E75's reads of an update: E75's or E72's clean files, Amendment 1's random and signs."""
    parts = {f"clean_{r}": e75.prior(name, f"clean_{r}") for r in ("W0", "FT", "inspan", "remainder", "upper", "lower", "adjoint")}
    return parts | {f"clean_{r}": json.loads((e75.OUT / name / f"{r}.json").read_text()) for r in rules.WRONG}


def anchors_of(name: str) -> dict[str, Any]:
    """The references `rules.score` reads: none on tofu; E75's table on u1 and gold; E80 Amendment 2 on u1; E61's two losses on gold."""
    if name == "tofu":
        return {"table_ref": None, "u1_ref": None, "gold_ref": None}
    if name == "u1":
        e80_out = json.loads(E80_OUT.read_text())
        return {"table_ref": stored(name), "gold_ref": None, "u1_ref": {"gold": e80_out["gold"], "ls": e80_out["left_in_delta"]["ls"]["plain_energy"]}}
    beside = json.loads(E61_GOLD.read_text())["verdict"]["beside"]
    return {"table_ref": stored(name), "u1_ref": None, "gold_ref": {"W0": beside["w0"]["held_gold"]["loss"], "FT": beside["reads"]["held_gold"]["loss"]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", required=True, choices=tuple(UPDATES))
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--out", required=True)  # the splits' X
    ap.add_argument("--wall-min", type=float, required=True)  # past this, between reads, exit 0
    a = ap.parse_args()
    if not PRED.is_file() or STOPPED.is_file():
        sys.exit(f"refusing to start: {PRED} missing, or {STOPPED} exists (the lead decides what next)")
    if a.update != "u1" and not (RESULTS / "u1" / "left_in_delta.json").is_file():
        sys.exit("refusing to start: u1's control has not finished")
    free = torch.cuda.mem_get_info(0)[0] / 1e9
    print(f"[card] {torch.cuda.get_device_name(0)}: {free:.1f} GB free, need {NEED_GB}", flush=True)
    if free < NEED_GB:
        sys.exit(f"refusing to start: this card has {free:.1f} GB free, under {NEED_GB}: someone holds it outside Slurm. Resubmit, and tell the lead")
    u = UPDATES[a.update]
    (RESULTS / u.name).mkdir(parents=True, exist_ok=True)
    job = pr.PassJob(a1.JobOpts(out=a.out, e40_large=str(a1.E40_RESULTS / "large" / "nowarmup"), decomp=a.decomp, results=str(RESULTS)), a1.CFG, lambda *_: {})
    job.select(u, Path(a.out))
    res = {u.name: {p.stem: json.loads(p.read_text()) for p in (RESULTS / u.name).glob("*.json")}}
    job.log(f"[config] {u}; holds {sorted(res[u.name])}; args {vars(a)}")
    with torch.no_grad():
        rd = e75.Reader(job)  # the model only; W0 pinned on the host
        tok = AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV)
        end = int(tok.convert_tokens_to_ids("<|im_end|>"))
        if u.name == "tofu":
            seqs, rows = tofu_rows(tok, end)
            check_tofu(job, rd, u, tok, end)
        else:
            seqs, rows = e80.gold_seqs()[0], job.rows("maths")[:, : pr.P.row_len]
            pr.check_file(job, rd, u)
        assert tuple(rows.shape) == (N_ROWS, pr.P.row_len), f"rows {rows.shape}"
        rd.rows["maths"] = rows  # "maths" is the reader's one block: this update's rows
        rd.drop_ft()
        runner = Runner(job, rd, res, Path(a.out), a.wall_min * 60, anchors_of(u.name))
        runner.rescore(u)  # a resume obeys the stops too
        runner.fractions(u, seqs)
        torch.cuda.empty_cache()  # its δW on the card went with the call
        runner.table(u)
    job.log(f"[done] {job.peaks()}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    hard_exit(code)
