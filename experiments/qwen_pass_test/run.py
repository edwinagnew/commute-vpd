"""E68, the D1 pass test on the saved updates ($0, one cluster L40, normal QoS). Plan:
reports/plans/2026-11-03_plan_e68-d1-pass-test.md; rules: results/predictions.json (the run refuses to start without it,
and while results/stopped.json exists), scored in rules.py. E44 Amendment 2's reader (experiments/qwen_maps/amend2.py) with
a next-token NLL beside each KL and reading A's masks; E44's stage S, one step at a time. No replays, no web rows.

Four updates in the filed order. Per update: W0 (K0, KD and the clean NLLs: one row loop), do-nothing, stage S, the often-on
half, the rarely-on half, the matched random change, least squares, adjoint; on u1 then reading A on do-nothing, lower, upper.
Each read writes results/<update>/<read>.json as it ends (a rerun skips what is written) and rescores results/score.json;
every number goes to results/run.log. control_reproduce missed: results/stopped.json, exit 1. Past --wall-min: exit 0 with
what is written; resubmit to resume. An update's stage S files on --out go when its reads finish.

Usage (repo root; sibling imports, and `-P` because `run` must mean qwen_maps/run.py for amend2.py, not this file):
  PYTHONPATH=experiments/qwen_maps:experiments/qwen_pass_test:src .venv/bin/python -P experiments/qwen_pass_test/run.py \
    --decomp DIR --out /ephemeral/$USER/e68 --wall-min 340
"""
# pyright: reportMissingImports=false, reportCallIssue=false, reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rules
import torch
from amend2 import Reader, kl
from job import CFG, E40_RESULTS, Job, JobOpts, write_json
from job import RESULTS as E44_RESULTS
from run import s_site, s_summary
from torch import Tensor
from transformers import AutoTokenizer

from commute import qwen_eval, qwen_sets
from commute import qwen_reading as qr
from commute.qwen_e40_files import sums
from commute.qwen_sft import free_bytes, load, sha256
from commute.qwen_vpd import SITES, importances, masked_logits, parse_site

HERE = Path(__file__).resolve().parent
RESULTS, PRED = HERE / "results", HERE / "results" / "predictions.json"
LARGE = Path.home() / "commute-data" / "results-large"
E40_DW, TRAIN = LARGE / "qwen_sft" / "nowarmup_regen" / "dW", LARGE / "qwen_recipe" / "train"
A1_ITEMS = HERE.parent / "qwen_updates" / "results" / "large" / "P" / "a1_rows.json"
GOLD_ID = HERE.parent / "qwen_recipe" / "results" / "size_state_gold_1e-5_100.json"
READING_A = ("do_nothing", "lower", "upper")


@dataclass(frozen=True)
class Update:
    name: str
    path: Path
    step: int
    rows: str  # the row set: "maths", or E57's "a1"
    n_rows: int  # filed "128 x 256"; qr.a1_rows makes 64 rows of 512, which "cut to 256" leaves at 64


@dataclass(frozen=True)
class PassCfg:
    updates: tuple[Update, ...] = (Update("u1", E40_DW / "step_10.safetensors", 10, "maths", 128),
                                   Update("u2", TRAIN / "gold_1e-5" / "step_100.safetensors", 100, "maths", 128),
                                   Update("u3", TRAIN / "a1_1e-6" / "step_100.safetensors", 100, "a1", 64),
                                   Update("u4", E40_DW / "step_100.safetensors", 100, "maths", 128))
    row_len: int = 256
    disk_min: float = 40e9  # S's files for one step, ~32 GB (amend2's 80 GB was for two steps)
    dw_rel_max: float = 0.1  # ‖file‖ / ‖W0‖: a weights file reads ~1, the δW files ~1e-3

    def __post_init__(self) -> None:
        assert [u.name for u in self.updates] == ["u1", "u2", "u3", "u4"] and self.updates[0].step == 10


P = PassCfg()


def nll(logits: Tensor, tokens: Tensor) -> float:
    """Summed next-token NLL of one batch, every position with a next token: fp32 per position, summed in double, as `kl`."""
    per = torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1).float(), tokens[:, 1:].flatten(), reduction="none")
    return float(per.double().sum())


class PassJob(Job):
    """Job with one update's δW file at a time and its stage S files under --out/<update>/."""

    update: Update

    def select(self, u: Update, root: Path) -> None:
        self.update, self.out = u, root / u.name
        self.forget()

    def dw(self, t: int, fp32: bool = False) -> dict[str, Tensor]:
        assert not fp32 and t == self.update.step
        if (t, False) not in self._dw:
            self._dw[(t, False)] = load(self.update.path)
        return self._dw[(t, False)]


class PassReader(Reader):
    """amend2's reader; its `read` calls this row. Keys: "K", "ft" (FT clean NLL) with "K" wanted; "ci", "clean" (W0's own)
    with "ci" wanted, on W0 only; "nll" always. "A" wanted: masks from Γ on the state's own weights."""

    def row(self, tokens: Tensor, own_key: str, want: set[str], srcs: dict[str, dict[str, Tensor]]) -> dict[str, float]:
        model, gamma, comps = self.job.model, self.job.gamma(), self.job.comps()
        if "A" in want:
            self.use(own_key)
        logits, g = importances(model, tokens, gamma, comps)  # both readings use the state's V′ in the readouts
        ci = masked_logits(model, tokens, comps, g)  # Δ off: the weights do not enter
        out = {"nll": nll(ci, tokens)}
        if "ci" in want:
            assert own_key == "W0" and "A" not in want
            out |= {"ci": kl(logits, ci), "clean": nll(logits, tokens)}
        del logits
        if "K" in want:
            self.use("FT")
            ref = model(tokens, use_cache=False).logits
            out |= {"K": kl(ref, ci), "ft": nll(ref, tokens)}
        self.use("W0")
        return out


def a1_rows() -> Tensor:
    """E57's anchor-1 rows (qwen_reading's stage D), cut to 256."""
    tok = AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV)
    rows = qr.a1_rows(tok, json.loads(A1_ITEMS.read_text())["items"], qwen_sets.load()["test"])
    return torch.tensor([r[: P.row_len] for r in rows])


def want_sha(u: Update) -> str | None:
    """E40's against the regen sums, gold's against E61's record; the template has no record, so it is only logged."""
    if u.path.parent == E40_DW:
        return sums(E44_RESULTS / "regen_SHA256SUMS").get(u.path.name)
    return json.loads(GOLD_ID.read_text())["identity"] if u.name == "u2" else None


def check_file(job: PassJob, rd: PassReader, u: Update) -> None:
    """The file is the one named, holds bf16 δW for exactly the 196 sites, and is small against W0 (not W itself)."""
    got, want = sha256(u.path) if u.path.is_file() else "absent", want_sha(u)
    job.log(f"[dW {u.name}] {u.path}: sha256 {got}, want {want}")
    if got == "absent" or (want is not None and got != want):
        job.stop("dW_file", f"{u.path}: sha256 {got} != {want}")
    dW, W0 = load(u.path), rd.sets["W0"]
    if set(dW) != set(SITES) or any(dW[s].dtype != torch.bfloat16 or dW[s].shape != W0[s].shape for s in SITES):
        job.stop("dW_file", f"{u.path}: not bf16 tensors of the 196 sites' shapes")
    rel = {s: float(dW[s].float().norm() / W0[s].norm()) for s in SITES}
    all_rel = math.sqrt(sum(float(dW[s].float().norm()) ** 2 for s in SITES) / sum(float(W0[s].norm()) ** 2 for s in SITES))
    job.log(f"[dW {u.name}] ‖δW‖/‖W0‖ {all_rel:.6g}, largest site {max(rel.values()):.6g}")
    if all_rel > P.dw_rel_max:
        job.stop("dW_file", f"{u.path}: ‖file‖/‖W0‖ = {all_rel:.3g} > {P.dw_rel_max}: this is not a δW")


class Runner:
    def __init__(self, job: PassJob, rd: PassReader, res: dict[str, dict], root: Path, wall_s: float):
        self.job, self.rd, self.res, self.root, self.wall_s = job, rd, res, root, wall_s

    def wall(self) -> None:
        if time.time() - self.job.t0 > self.wall_s:
            self.job.log(f"cut short: wall clock {time.time() - self.job.t0:.0f} s > {self.wall_s:.0f}; resubmit to resume")
            sys.exit(0)

    def rescore(self) -> None:
        sc = rules.score(self.res)
        write_json(RESULTS / "score.json", sc)
        self.job.log(f"[score] {json.dumps(sc, default=float)}")
        if sc["control_reproduce"]["missed"]:
            self.job.stop("control_reproduce", json.dumps(sc["control_reproduce"], default=float))

    def put(self, u: Update, key: str, value: dict[str, Any]) -> None:
        self.res[u.name][key] = value
        write_json(RESULTS / u.name / f"{key}.json", value)
        self.rescore()

    def step(self, u: Update, key: str, fn: Callable[[], dict[str, Any]]) -> None:
        if key not in self.res[u.name]:
            self.wall()
            self.job.stage = f"{u.name}/{key}"
            self.put(u, key, fn())

    def stage_s(self, u: Update) -> None:
        job, t = self.job, u.step
        files = all((job.site_dir(t) / f"{s}.safetensors").is_file() for s in job.sites)
        if "S" in self.res[u.name] and files:
            return
        job.stage = f"{u.name}/S"
        self.wall()
        job.out.mkdir(parents=True, exist_ok=True)
        if (free := free_bytes(job.out)) < P.disk_min:
            job.stop("disk", f"{free / 1e9:.0f} GB free on {job.out} before S < {P.disk_min / 1e9:.0f}")
        self.rd.drop_ft()
        job.drop_gamma()  # fp32 components + S's working set would pass 44 GiB beside Γ
        t0 = time.time()
        job.sigma()  # E44's maths rows at W0, for every update
        job.log(f"sigma built in {time.time() - t0:.0f} s")
        for layer in job.cfg.layers:
            for s in (x for x in job.sites if parse_site(x)[0] == layer):
                s_site(job, s, (t,), with_halves=True)
            job.log(f"[S {u.name}] layer {layer}: {time.time() - t0:.0f} s, {job.peaks()}")
            self.wall()
        job.forget()
        job.drop_sigma()  # 11 GB of host memory
        summary = s_summary(job, t)
        del summary["per_site"]  # 390 KB the rules do not read; the sites' records go with --out
        job.log(f"[S {u.name}] {json.dumps(summary, default=float)}")
        job.gamma()
        self.put(u, "S", summary)

    def update(self, u: Update, rows: Tensor) -> None:
        job, rd = self.job, self.rd
        job.select(u, self.root)
        (RESULTS / u.name).mkdir(parents=True, exist_ok=True)
        rd.rows["maths"], rd.w0_logits = rows, []  # "maths" is the reader's one block: this update's rows
        rd.drop_ft()
        d, t = self.res.setdefault(u.name, {}), u.step
        self.step(u, "W0", lambda: rd.read("W0", t, "maths", {"ci", "K"}))
        self.step(u, "do_nothing", lambda: rd.read("do_nothing", t, "maths", {"K"}))
        later: list[tuple[str, Callable[[], dict[str, Any]]]] = [
            ("upper", lambda: rd.read("upper", t, "maths", {"K"})),
            ("lower", lambda: rd.read("lower", t, "maths", {"K"})),
            ("search", lambda: rd.search(t)),
            ("random", lambda: rd.read("random", t, "maths", {"K"}, scale=d["search"]["a"])),
            ("ls", lambda: rd.read("ls", t, "maths", {"K"})),
            ("adjoint", lambda: rd.read("adjoint", t, "maths", {"K"}))]
        if u.name == "u1":
            later += [(f"A_{n}", lambda n=n: rd.read(n, t, "maths", {"K", "A"})) for n in READING_A]
        if any(key not in d for key, _ in later):
            self.stage_s(u)
        for key, fn in later:
            self.step(u, key, fn)
        shutil.rmtree(job.out, ignore_errors=True)
        job.log(f"[{u.name}] done; {job.out} removed, {job.peaks()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--out", required=True)  # stage S's files
    ap.add_argument("--wall-min", type=float, required=True)  # past this, between reads, exit 0
    a = ap.parse_args()
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
    if (RESULTS / "stopped.json").is_file():
        sys.exit(f"refusing to start: {RESULTS / 'stopped.json'} exists (a stop rule fired; the lead decides what next)")
    job = PassJob(JobOpts(out=a.out, e40_large=str(E40_RESULTS / "large" / "nowarmup"), decomp=a.decomp,
                          results=str(RESULTS)), CFG, lambda *_: {})
    res = {u.name: {p.stem: json.loads(p.read_text()) for p in (RESULTS / u.name).glob("*.json")} for u in P.updates}
    job.log(f"[config] {P}; {rules.R}; resumed { {k: sorted(v) for k, v in res.items()} }; args {vars(a)}")
    rows = {"maths": job.rows("maths")[:, : P.row_len], "a1": a1_rows()}  # built first: a failure shows in the first minute
    with torch.no_grad():
        job.gamma()
        rd = PassReader(job)
        runner = Runner(job, rd, res, Path(a.out), a.wall_min * 60)
        runner.rescore()  # a resume obeys the control too
        for u in P.updates:
            assert tuple(rows[u.rows].shape) == (u.n_rows, P.row_len), f"{u.name}: rows {rows[u.rows].shape}"
            check_file(job, rd, u)  # every file before any long stage
        for u in P.updates:
            runner.update(u, rows[u.rows])
    job.log(f"[done] {job.peaks()}")


if __name__ == "__main__":
    main()
