"""E79: is the masked fall in E76 agreement or softness? ($0, one cluster L40, `debug`.) Plan: reports/plans/2026-11-04_plan_e79-agreement-or-softness.md;
rules: rules.py, results/predictions.json (the run refuses to start without it, and while results/stopped.json exists).

Saved states W0, A, B, L, read against FT clean (W0 + u1) on M (E76's 8 reading rows) and W (FineWeb rows 32-39, in no arm's training pool). Masks
from Γ on W0's forward, frozen, at two draws; the last stage reads Γ on the state's own forward. Stages in order: control (each state reproduces
E76's (i) and (iv), or a stop), positive control, W0's reference numbers and the temperature floor, A, B, L, own masks. results/read.json after
every cell, every number in results/run.log, per-position arrays in results/large/. g per row on the node's disk, recomputed if absent.

Γ reads V rounded to its own bf16 (`qwen_vpd.readout`), so bf16 components give the g of E76's stage 0 and everything stays on the card (~44 GB).

Usage (repo root): .venv/bin/python experiments/qwen_masked_read/run.py --decomp DIR --out /ephemeral/$USER/e79
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rules
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from commute.jobs import hard_exit, load_experiment
from commute.qwen_e40_files import sums
from commute.qwen_masked import row_masks, student_masked
from commute.qwen_sft import linears, load, sha256
from commute.qwen_teacher import Offsets, change_hooks
from commute.qwen_vpd import (
    SITES,
    Components,
    Gamma,
    fp32_matmuls,
    importances,
    load_components,
    load_gamma,
    load_target,
)
from commute.vpd import kl_per_position

e76 = load_experiment("qwen_masked_teacher", "common")  # its log and stop write E76's files: this job has its own
cache_g = load_experiment("qwen_masked_teacher", "stage0").cache_g
roles = load_experiment("qwen_diagnose", "run").roles  # E73's labelling of the target token's turn
TEMPS = load_experiment("qwen_transmission", "run").C.temps  # E74's grid

HERE = Path(__file__).resolve().parent
RESULTS, LARGE, PRED, READ = HERE / "results", HERE / "results" / "large", HERE / "results" / "predictions.json", HERE / "results" / "read.json"
DEV = "cuda"
STATES = ("W0", "A", "B", "L")
T0 = time.time()


@dataclass(frozen=True)
class Cfg:
    rows: int = 8  # per row set
    web_start: int = 32  # `pools()` trains on FineWeb rows 0-31
    tol: float = 1e-3  # as E76's amendments: the same path reads ~2e-6 from rounding alone, stage 0's floor
    draw2: int = 79_200

    def __post_init__(self) -> None:
        assert self.rows == e76.C.fixed_rows and self.web_start >= e76.C.pool_each


C = Cfg()
SEEDS = {"d1": e76.C.read_seed, "d2": C.draw2}  # draw 1 is E76's


@dataclass
class Job:
    model: Any
    comps: Components  # bf16
    off: Offsets
    gamma: Gamma
    rows: dict[str, Tensor]
    g: dict[str, list[dict[str, Tensor]]]  # W0's, host
    response: dict[str, Tensor]  # bool, [rows, T]
    ft: dict[str, list[Tensor]]  # clean logits, host
    w0: dict[str, list[Tensor]]
    out: dict[str, Any]


def log(msg: str) -> None:
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    with (RESULTS / "run.log").open("a") as f:
        f.write(line + "\n")


def stop(rule: str, detail: str) -> None:
    log(f"STOP {rule}: {detail}")
    e76.write_json(RESULTS / "stopped.json", {"rule": rule, "detail": detail})
    sys.exit(f"STOP {rule}: {detail}")


def row_sets() -> dict[str, Tensor]:
    web = json.loads(e76.WEB.read_text())["rows"][C.web_start : C.web_start + C.rows]
    return {"M": e76.rows_all()[1][: C.rows], "W": torch.tensor([r[: e76.C.row_len] for r in web])}


def response_positions(tokens: Tensor) -> Tensor:
    """Positions whose next token is in an assistant turn; the last position has no next token."""
    return torch.cat([roles(tokens) == 1, torch.zeros((len(tokens), 1), dtype=torch.bool)], 1)


def load_dw() -> dict[str, Tensor]:
    got, want = sha256(e76.DW_FILE), sums(e76.DW_SUMS)[e76.DW_FILE.name]
    log(f"[dW] {e76.DW_FILE}: sha256 {got}, want {want}")
    if got != want:
        stop("dW_file", f"{got} != {want}")
    return {s: v.to(DEV) for s, v in load(e76.DW_FILE).items() if s in SITES}


def clean_logits(model: Any, rows: Tensor, change: Any) -> list[Tensor]:
    with change_hooks(model, change):
        return [model(t[None].to(DEV), use_cache=False).logits.cpu() for t in rows]


def set_state(J: Job, state: str) -> None:
    if state == "W0":
        J.off.reset()
    else:
        J.off.load_from(e76.LARGE / f"{state}_offsets.safetensors")


def clean_rows(J: Job, rs: str) -> Iterator[Tensor]:
    for t in J.rows[rs]:
        with change_hooks(J.model, J.off.change):
            q = J.model(t[None].to(DEV), use_cache=False).logits
        yield q


def masked_rows(J: Job, rs: str, seed: int, g: list[dict[str, Tensor]] | None = None) -> Iterator[Tensor]:
    """The student's masked forward, all 196 sites, one generator per pass over the rows: draw 1 on M is `arms.read_state`'s stream."""
    gen = torch.Generator(DEV).manual_seed(seed)
    for r, t in enumerate(J.rows[rs]):
        masks, deltas = row_masks((J.g[rs] if g is None else g)[r], gen)
        with student_masked(J.model, J.comps, J.off, masks, deltas):
            q = J.model(t[None].to(DEV), use_cache=False).logits
        yield q


def positions(p: Tensor, q: Tensor, temps: Sequence[float] = ()) -> dict[str, Tensor]:
    """Per position against the reference p: numbers 1-4, with `temps` also number 1 at each temperature."""
    lq = q.float().log_softmax(-1)
    out = {"fwd": kl_per_position(p, q), "rev": kl_per_position(q, p), "agree": p.argmax(-1) == q.argmax(-1), "H": -(lq.exp() * lq).sum(-1)}
    out |= {f"{rules.TEMP_KEY}{T}": kl_per_position(p, q / T) for T in temps}
    return {k: v[0].cpu() for k, v in out.items()}


def read_rows(logits: Iterable[Tensor], ref: list[Tensor], temps: Sequence[float] = ()) -> dict[str, Tensor]:
    parts = [positions(ref[r].to(DEV), q, temps) for r, q in enumerate(logits)]
    return {k: torch.stack([p[k] for p in parts]) for k in parts[0]}


def summary(pos: dict[str, Tensor], where: Tensor) -> dict[str, float]:
    n = int(where.sum())
    return {k: rules.over(float(v[where].double().sum()), n) for k, v in pos.items()} | {"n": n}


def keep(J: Job, state: str, rs: str, cell: str, pos: dict[str, Tensor]) -> dict[str, Any]:
    """The cell's arrays in small dtypes and its summary, in read.json and the log."""
    save_file({k: v.to(torch.uint8 if k == "agree" else torch.float16).contiguous() for k, v in pos.items()}, str(LARGE / f"{state}_{rs}_{cell}.safetensors"))
    s = {"all": summary(pos, torch.ones_like(J.response[rs])), "response": summary(pos, J.response[rs])}
    J.out["reads"].setdefault(state, {}).setdefault(rs, {})[cell] = s
    log(f"[{state} {rs} {cell}] {json.dumps(s)}")
    e76.write_json(READ, J.out)
    return s


def read_cell(J: Job, state: str, rs: str, cell: str) -> dict[str, Any]:
    """One state, row set and pass, against FT clean, once; W0's masked passes carry the temperatures."""
    done = J.out["reads"].get(state, {}).get(rs, {}).get(cell)
    if done:
        return done
    if cell == "clean":
        return keep(J, state, rs, cell, read_rows(clean_rows(J, rs), J.ft[rs]))
    return keep(J, state, rs, cell, read_rows(masked_rows(J, rs, SEEDS[cell]), J.ft[rs], TEMPS if state == "W0" else ()))


def stored(state: str) -> dict[str, float]:
    """E76's final reading of an arm; W0's is step 0."""
    r = e76.read_json(e76.RESULTS / f"{'B' if state == 'W0' else state}.json")["readings"][0 if state == "W0" else -1]
    return {k: r[k] for k in ("i", "iv")}


def stage_control(J: Job) -> None:
    for state in STATES:
        set_state(J, state)
        got = {"i": read_cell(J, state, "M", "clean")["all"]["fwd"], "iv": read_cell(J, state, "M", "d1")["all"]["fwd"]}
        want = stored(state)
        J.out["control"][state] = {"got": got, "want": want}
        log(f"[control {state}] {got}, E76's {want}")
        e76.write_json(READ, J.out)
        if not all(abs(got[k] - want[k]) <= C.tol for k in want):
            stop("control", f"{state}: {got} against E76's {want}, within {C.tol}")


def stage_positive(J: Job) -> None:
    """Number 7, read beside K0 0.044: the masked forward against its own clean forward. Reported; the plan files no stop on it."""
    set_state(J, "W0")
    keep(J, "W0", "M", "self", read_rows(masked_rows(J, "M", SEEDS["d1"]), J.w0["M"]))


def stage_reference(J: Job) -> None:
    """FT clean's own entropy (number 4's reference), then W0 on both row sets and draws, with number 5."""
    set_state(J, "W0")
    for rs in J.rows:
        keep(J, "FT", rs, "clean", read_rows((x.to(DEV) for x in J.ft[rs]), J.ft[rs]))
        read_cell(J, "W0", rs, "clean")
        for cell in SEEDS:
            log(f"[number 5 {rs} {cell}] {rules.floor(read_cell(J, 'W0', rs, cell)['all'])}")


def stage_arm(J: Job, state: str) -> None:
    set_state(J, state)
    for rs in J.rows:
        for cell in ("clean", *SEEDS):
            read_cell(J, state, rs, cell)


def own_g(J: Job, W0: dict[str, Tensor], rs: str = "M") -> list[dict[str, Tensor]]:
    """Γ on the state's own forward: weights W0 + change, the alive columns of the components moved; bf16, as E76 cached W0's."""
    g = []
    with J.off.state(J.model, J.comps, W0):
        for t in J.rows[rs]:
            got = importances(J.model, t[None].to(DEV), J.gamma, J.comps)[1]
            g.append({s: got[s].to("cpu", torch.bfloat16) for s in SITES})
    return g


def stage_own(J: Job) -> None:
    """Number 6, W0 first: its own masks are E76's g on the same draw, so it must read its frozen-mask number 1."""
    W0 = {s: lin.weight.data for s, lin in linears(J.model).items()}
    for state in STATES:
        set_state(J, state)
        got = keep(J, state, "M", "own", read_rows(masked_rows(J, "M", SEEDS["d1"], own_g(J, W0)), J.ft["M"]))["all"]["fwd"]
        if state == "W0" and not abs(got - (want := J.out["reads"]["W0"]["M"]["d1"]["all"]["fwd"])) <= C.tol:
            stop("own_control", f"W0 under own masks reads {got}, under E76's g {want}, within {C.tol}")


def peak() -> str:
    return f"GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"


def load_parts(decomp: Path) -> tuple[Any, Components, Offsets, Gamma, dict[str, Tensor]]:
    fp32_matmuls()
    model = load_target(DEV, torch.float32)
    rows = row_sets()
    host = load_components(decomp, "cpu", torch.float32)
    off = Offsets(host, load_file(str(e76.LARGE / "cut.safetensors")), DEV)
    comps = host.to(DEV, torch.bfloat16)
    del host
    gc.collect()
    gamma = load_gamma(decomp, DEV, torch.bfloat16)
    log(f"[load] {torch.cuda.get_device_name()}, {peak()}; config {C}, seeds {SEEDS}")
    return model, comps, off, gamma, rows


def ft_clean(model: Any, rows: dict[str, Tensor]) -> dict[str, list[Tensor]]:
    teacher = load_dw()
    return {rs: clean_logits(model, t, teacher.__getitem__) for rs, t in rows.items()}


def set_up(decomp: Path, out: Path) -> Job:
    model, comps, off, gamma, rows = load_parts(decomp)
    joined = torch.cat([rows["M"], rows["W"]])
    if missing := [i for i in range(len(joined)) if not e76.g_path(out, i).is_file()]:
        cache_g(model, joined, gamma, comps, out, missing)
    log(f"[g] {len(missing)} of {len(joined)} rows computed, {peak()}")
    g = {"M": [e76.load_g(out, i) for i in range(C.rows)], "W": [e76.load_g(out, C.rows + i) for i in range(C.rows)]}
    ft = ft_clean(model, rows)
    w0 = {rs: clean_logits(model, t, None) for rs, t in rows.items()}
    return Job(model, comps, off, gamma, rows, g, {rs: response_positions(t) for rs, t in rows.items()}, ft, w0, {"control": {}, "reads": {}})


def score() -> None:
    """From what read.json holds: a stop or a crash still scores, with "unread" for what never ran."""
    out = e76.read_json(READ)
    if out is None:
        return
    out["score"] = rules.score(out["reads"])
    e76.write_json(READ, out)
    log(f"[score] {json.dumps(out['score'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("--out", required=True, help="the node's disk: g per row, ~6 GB")
    a = ap.parse_args()
    if not PRED.is_file() or (RESULTS / "stopped.json").is_file():
        sys.exit("refusing to start: predictions.json missing, or stopped.json exists")
    LARGE.resolve().mkdir(parents=True, exist_ok=True)
    READ.unlink(missing_ok=True)  # a rerun must not score an earlier job's reads
    torch.set_grad_enabled(False)
    J = set_up(Path(a.decomp), Path(a.out))
    stage_control(J)
    stage_positive(J)
    stage_reference(J)
    for state in STATES[1:]:
        stage_arm(J, state)
    stage_own(J)
    log(f"[done] {peak()}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else 1 if e.code else 0
    except BaseException:  # noqa: BLE001  (an out-of-memory or the like: the same exit, never a hang in teardown)
        traceback.print_exc()
        code = 1
    try:
        score()
    finally:
        hard_exit(code)
