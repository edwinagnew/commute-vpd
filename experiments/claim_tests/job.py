"""E47's job: the model and harness, the C0 W0 control, the same-job lines, one state's reading, gates and the
results file (results/<stage>.json, rewritten after every state; a rerun on the same card with the same W0 skips the
finished states). Plan: reports/plans/2026-10-28_plan_claim-tests.md (section 3 and Amendment 1)."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from recipes import at
from rules import fail_line

HERE = Path(__file__).resolve().parent
RESULTS, LARGE = HERE / "results", HERE / "results" / "large"
EXP = HERE.parent
TRANSPORT = EXP / "transport_test" / "results"
LADDER = EXP / "task_ladder" / "results" / "large"
KV = EXP / "kv_task" / "results" / "large"
TT, BF = EXP / "teacher_training" / "results" / "large", EXP / "broad_fragility" / "results" / "large"
R1_WINDOWS = LADDER / "windows_r1_t534_v369.pt"

# Each stage's adversarial starts, in reading order, and the first Pile val row of its 256 rows.
STARTS = {"t1": ("budget", "h0", "h1"), "t1b": ("h2",), "t2": ("h0", "h1"), "t2b": ("h0", "h1"), "t2c": ("h0", "h1"),
          "t3": (), "t4": ("h0", "h1", "h2"), "t4b": ("h0", "h1", "h2"), "t5": ("budget", "h0"), "t5b": ("budget", "h0")}
ROWS = {"t2c": 256}  # every other stage: rows 0-255


@dataclass(frozen=True)
class Config:
    # W0 per harness start on rows 0-255, L40 (jobs 40166, 40195, 40497 agree to 17 digits), and the floor over h0-h2
    l40_w0: dict[str, float] = field(default_factory=lambda: {"h0": 0.6917431652545929, "h1": 0.8361621797084808,
                                                              "h2": 0.7048778533935547})
    l40_f: float = 0.20877548446862537
    f_repro_tol: float = 1e-3  # one harness start: f = l40_f when W0 reproduces its L40 KL within this, relative
    # C0, faith_score.w0_check's rule per harness start read: PGD within 10% of the reference, CI within 2%, f <= 0.25.
    # Rows 0-255: 0.69 at h0 (w0_check), the L40 values at h1, h2. Rows 256-511: broad_fragility Amendment 2 (H100).
    pgd_ref: dict[int, dict[str, float]] = field(default_factory=lambda: {
        0: {"h0": 0.69, "h1": 0.8362, "h2": 0.7049}, 256: {"h0": 0.6298, "h1": 0.6576, "h2": 0.6688}})
    ci_ref: dict[int, float] = field(default_factory=lambda: {0: 0.348, 256: 0.3331})
    pgd_tol: float = 0.10
    ci_tol: float = 0.02
    floor_max: float = 0.25
    budget_band: tuple[float, float] = (0.85, 1.15)  # C0: W0's budget KL / its h0 KL

    def __post_init__(self) -> None:
        assert set(self.l40_w0) == set(self.pgd_ref[0]) and set(ROWS.values()) | {0} == set(self.pgd_ref) == set(self.ci_ref)
        assert self.l40_f == max(self.l40_w0.values()) / min(self.l40_w0.values()) - 1


CFG = Config()
LOG: list[Path] = []


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    for p in LOG:
        with p.open("a") as fh:
            fh.write(line + "\n")


def _finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite(v) for v in o]
    return o


class Job:
    """One stage's job: the harness context on the stage's rows (W0 not yet read), its results file and its W0."""

    def __init__(self, stage: str, device: str) -> None:
        import torch

        from commute import harness

        self.stage, self.starts, self.row0 = stage, STARTS[stage], ROWS.get(stage, 0)
        self.path = RESULTS / f"{stage}.json"
        LARGE.mkdir(parents=True, exist_ok=True)
        LOG[:] = [LARGE / f"{stage}.log"]
        token = json.loads((TRANSPORT / "windows.json").read_text())["o_token_id"]
        ctx = harness.load(device, TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt", token, 1e-6, 256,
                           20, (), log)
        if self.row0:
            from commute.data import batches, pile_rows

            rows = pile_rows("val", self.row0, 256, log)
            ctx = replace(ctx, blocks=[batches(rows[s : s + 128], 8) for s in range(0, 256, 128)])
        self.ctx = ctx
        self.card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        self.out: dict[str, Any] = {"stage": stage, "card": self.card, "rows": [self.row0, self.row0 + 256],
                                    "starts": list(self.starts), "states": {}, "gates": {}}
        self.w0: dict[str, float] = {}
        self.f = math.nan

    def dump(self) -> None:
        self.path.write_text(json.dumps(_finite(self.out), indent=1, sort_keys=True))

    def halt(self, why: str) -> None:
        self.out["stopped"] = why
        self.dump()
        log(f"STOP: {why}")
        sys.exit(f"STOP: {why}")

    def fail_line(self) -> float:
        return fail_line(self.f)


def c0(job: Job) -> None:
    """W0 at every start the job reads, first and alone; the floor f; the stop rules. A rerun keeps the file's states
    only if this job's W0 equals the file's (same card, deterministic harness)."""
    from commute.faithfulness import pile_faithfulness, start_kl

    if not job.starts:
        resume(job, {"kl": {}, "note": "no adversary; the stage's control is its reproduction gate"})
        return
    m, blocks, ci = job.ctx.model, job.ctx.blocks, None
    for s in job.starts:
        t0 = time.time()
        if s == "budget":
            job.w0[s] = start_kl(m, blocks, s)
        else:
            rec = pile_faithfulness(m, blocks, int(s[1:]), 20, log)
            job.w0[s], ci = rec["pgd_kl"], rec["ci_masked_kl"]
        log(f"[C0 W0 {s}] kl {job.w0[s]:.6f} ({time.time() - t0:.0f} s)")
    hs = {s: v for s, v in job.w0.items() if s != "budget"}
    source = "this job's harness starts"
    if len(hs) == 1:
        (s, v), = hs.items()
        if job.row0 == 0 and abs(v / CFG.l40_w0[s] - 1) <= CFG.f_repro_tol:
            source = f"L40 jobs 40166/40195/40497 ({s} reproduces {CFG.l40_w0[s]:.6f})"
        else:
            extra = "h1" if s != "h1" else "h0"
            hs[extra] = pile_faithfulness(m, blocks, int(extra[1:]), 20, log)["pgd_kl"]
            source = f"this job's {s} and {extra} ({s} did not reproduce its L40 value)"
    job.f = CFG.l40_f if source.startswith("L40") else max(hs.values()) / min(hs.values()) - 1  # faith_score.seed_floor
    w0 = {"kl": job.w0, "floor_starts": hs, "ci": ci, "f": job.f, "f_source": source, "fail_line": job.fail_line(),
          "pass_line": 1.5 / (1 + job.f)}
    w0["checks"] = c0_checks(job.row0, hs, ci, job.f, job.w0)
    log(f"[C0] {json.dumps(_finite(w0))}")
    resume(job, w0)
    if not all(w0["checks"].values()):
        job.halt(f"C0: {w0['checks']}")


def c0_checks(row0: int, hs: dict[str, float], ci: float | None, f: float, w0: dict[str, float]) -> dict[str, bool]:
    ref = CFG.pgd_ref[row0]
    checks = {f"pgd_{s}": abs(v / ref[s] - 1) <= CFG.pgd_tol for s, v in hs.items()}
    checks |= {"ci": ci is not None and abs(ci / CFG.ci_ref[row0] - 1) <= CFG.ci_tol, "floor": f <= CFG.floor_max}
    if "budget" in w0 and "h0" in w0:
        lo, hi = CFG.budget_band
        checks["budget_vs_h0"] = lo <= w0["budget"] / w0["h0"] <= hi
    return checks


def resume(job: Job, w0: dict[str, Any]) -> None:
    """Continue the stage's file when its card and W0 KLs equal this job's (relative 1e-6), else refuse."""
    if job.path.is_file():
        old = json.loads(job.path.read_text())
        same = old["card"] == job.card and set(old["W0"]["kl"]) == set(job.w0) and all(
            abs(old["W0"]["kl"][s] / v - 1) <= 1e-6 for s, v in job.w0.items())
        if not same:
            sys.exit(f"{job.path} was written on {old['card']} with W0 {old['W0']['kl']}; this job: {job.card}, {job.w0}. "
                     "Move the file aside to start the stage again")
        job.out = {**old, "states": old.get("states", {}), "gates": old.get("gates", {})}
        job.out.pop("stopped", None)
        log(f"[resume] {job.path.name}: states done {sorted(job.out['states'])}")
    job.out["W0"] = w0
    job.dump()


def read(job: Job, name: str, upd: tuple[Any, Any], starts: tuple[str, ...] | None = None, stop: bool = False,
         extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One state (delta-W, alive (U', V') or None) at `starts` (default: the job's) through `multi_start`, with the fail
    line as `stop_at` if `stop`; stored, dumped and logged. A state already in the file is returned as it is."""
    from commute.faithfulness import multi_start

    if name in job.out["states"]:
        return job.out["states"][name]
    t0 = time.time()
    with at(job.ctx, *upd):
        rec = multi_start(job.ctx.model, job.ctx.blocks, list(starts or job.starts), w0=job.w0,
                          stop_at=job.fail_line() if stop else None, log=lambda s: log(f"[{name}] {s}"))
    rec |= {"seconds": time.time() - t0, **(extra or {})}
    job.out["states"][name] = rec
    job.dump()
    log(f"[{name}] ratio {json.dumps({s: round(r, 4) for s, r in rec['ratio'].items()})} max {rec['max']:.4f} mean "
        f"{rec['mean'] if rec['mean'] is None else round(rec['mean'], 4)} complete {rec['complete']} "
        f"(lines {1.5 / (1 + job.f):.3f} / {job.fail_line():.3f})" + (f" {json.dumps(_finite(extra))}" if extra else ""))
    return rec


def gate(job: Job, name: str, ok: bool, rec: dict[str, Any]) -> None:
    job.out["gates"][name] = {**rec, "pass": ok}
    log(f"[{name}] {json.dumps(_finite(job.out['gates'][name]))}")
    job.dump()
    if not ok:
        job.halt(f"{name}: {json.dumps(_finite(rec))}")


def near(x: float, ref: float, tol: float, relative: bool = False) -> bool:
    return abs(x / ref - 1) <= tol if relative else abs(x - ref) <= tol
