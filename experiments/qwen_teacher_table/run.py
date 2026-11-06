"""E111 (qwen_teacher_table): teacher training from the least-squares split on every state of E93's table, read by
E93's rules with Δ frozen and judged on the behaviour columns (plan reports/plans/2026-11-06_plan_e111-teacher-table.md
and its "Overrides (user, 2026-11-06)"; rules, early checks and leans: rules.py, results/predictions.json; the run
refuses to start without it, and while results/stopped.json exists). One H100 or A100 80GB pod: RUN_BRIEF.md.

  --stage 1      RL seed 70 alone, two arms (the plan's rate, and 10x as the positive control)
  --stage 2..5   the plan's later stages, one arm each; refused until stage 1's state is done
  --root DIR     the pod's disk: qwen/ (decomposition), e80_inputs/, inputs/, dW/ (fetched when missing), large/ (offsets)

Per state, each read logged and results/<state>/state.json rewritten as it ends: the δW file against E93's sha256 and
norm; least squares' split (share and size against E93's); this card's clean W0 (its drift from E93's ruler sets the
reproduction band) and FT; least squares' clean moved part against E93's; the base cells (W0 masked: its drift sets the
masked band; do-nothing), least squares' own cells at FT's weights (E93's K11, K10) and with Δ frozen. Per arm:
teacher training (teach.py), the offsets saved (bf16), then the kept state's split sums, clean moved part, gain, own
cells with Δ frozen and clean remainder, and the state's scored row printed. Exit 1: an early check missed (stopped.json).
A state or arm already written is skipped, so a rerun of the same command resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import rules
import teach
import torch
import upload
from safetensors.torch import load_file
from transformers import AutoTokenizer

from commute import qwen_d3 as d3
from commute import qwen_eval, qwen_files, qwen_tofu
from commute.jobs import EXPERIMENTS, hard_exit, load_experiment
from commute.qwen_sft import sha256
from commute.qwen_vpd import SITES

E93: Any = load_experiment("qwen_d1_table", "run")
E80: Any = load_experiment("qwen_d1_adversary", "upload")
RESULTS, PRED = rules.RESULTS, rules.RESULTS / "predictions.json"
E93.RESULTS, E93.STOPPED = RESULTS, RESULTS / "stopped.json"  # E93's log, check and memory write here, not over E93's files
log, kmean, rel = E93.log, E93.kmean, E93.rel
F64 = d3.F64


def check(name: str, kind: str, v: dict[str, Any]) -> None:
    """An early check (`rules.judge`); a miss writes stopped.json and exits 1 (E93's `check`)."""
    E93.check(name, rules.judge(kind, v), v)


def write(n: str, rec: dict[str, Any]) -> None:
    E93.write(RESULTS / n / "state.json", rec)


def host_total_gb() -> float:
    return int(next(x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemTotal")).split()[1]) / 1e6


def fetch_inputs(root: Path, names: tuple[str, ...]) -> None:
    """Everything the stage reads from W&B into `root` (each kept when complete), then the shared readers pointed at it."""
    qwen_files.fetch(root / "qwen", log)
    E80.fetch(root / "e80_inputs")
    upload.fetch(root, list(names), log)
    d3.GSTATS, d3.A1_ITEMS, qwen_tofu.DIR = root / "e80_inputs", root / "inputs" / "a1_rows.json", root / "inputs"


# ----------------------------------------------------------------------------- the reads that do not depend on the arm


def shared_reads(d: d3.Decomp, rs: d3.RowSet, n: str, e93: dict[str, Any], dW: dict[str, torch.Tensor], rec: dict[str, Any],
                 F: dict[str, Any]) -> Any:
    """Least squares' split (recomputed on a resume: the arms train from it), then, once per state, this card's reads."""
    p = d3.ls_parts(d, dW)
    S, L = p.sums, e93["split"]
    check(f"{n}_split", "split", {"share": S["left"] / S["size"], "size": S["moved"] / S["size"],
                                  "e93_share": L["left"] / L["size"], "e93_size": L["moved"] / L["size"], "seconds": p.seconds})
    if "shared" in rec:
        return p
    sh: dict[str, Any] = {"split": S, "split_s": p.seconds, "clean": {"W0": E93.clean_at(d, rs, "W0")}}
    ruler, ruler93 = kmean(sh["clean"]["W0"], rs), kmean(e93["clean"]["W0"], rs)
    check(f"{n}_card_ruler", "card", {"here": ruler, "e93": ruler93})
    sh["band"] = rules.band(rel(ruler, ruler93))
    sh["clean"]["FT"] = E93.clean_at(d, rs, "FT")
    check(f"{n}_clean_FT", "ft_zero", {"K": kmean(sh["clean"]["FT"], rs)})
    sh["clean"]["ls_moved"] = E93.clean_at(d, rs, "ls_moved", p.moved)
    check(f"{n}_ls_moved", "reproduce", {"keys": ["K"], "K": kmean(sh["clean"]["ls_moved"], rs),
                                         "e93_K": kmean(e93["clean"]["moved"], rs), "band": sh["band"]})
    E93.memory(n, F)
    sh["base"] = E93.cells_timed(d, rs, f"{n} base")
    k00, k00_93 = kmean(sh["base"], rs, "K00"), kmean(e93["base"], rs, "K00")
    check(f"{n}_card_masked", "card", {"here": k00, "e93": k00_93})
    sh["band_masked"] = rules.band(rel(k00, k00_93))
    sh["own_ft"] = E93.cells_timed(d, rs, f"{n} ls own at FT's weights", change=d3.ls_change(d, p.X), w0_v=False)
    got = {"K01": kmean(sh["base"], rs, "K01"), "K11": kmean(sh["own_ft"], rs, "K11"), "K10": kmean(sh["own_ft"], rs, "K10")}
    want = {"e93_K01": kmean(e93["base"], rs, "K01")} | {f"e93_{k}": kmean(e93["own"], rs, k) for k in ("K11", "K10")}
    check(f"{n}_ls_own_ft", "reproduce", {"keys": list(got)} | got | want | {"band": sh["band_masked"]})
    d.weights.put("ls", p.moved)
    sh["own_frozen"] = E93.cells_timed(d, rs, f"{n} ls own, Δ frozen", change=d3.ls_change(d, p.X), w0_v=False, state="ls")
    d.weights.drop("ls")
    rec["shared"] = sh
    write(n, rec)
    return p


# ----------------------------------------------------------------------------- one arm


def split_sums(dW: dict[str, torch.Tensor], mv: dict[str, torch.Tensor], S: dict[str, float]) -> dict[str, float]:
    """`d3.ls_parts`' sums for the trained moved part: size and null are δW's (the map does not enter them)."""
    out = {"size": S["size"], "null": S["null"], "left": 0.0, "moved": 0.0}
    for s in SITES:
        M, X = dW[s].to(d3.DEV, F64), mv[s].to(d3.DEV, F64)
        out["left"] += float(((M - X) ** 2).sum())
        out["moved"] += float((X * X).sum())
        del M, X
    return out


def run_arm(d: d3.Decomp, rs: d3.RowSet, seqs: list[Any], n: str, arm: str, scale: float, p: Any, dW: dict[str, torch.Tensor],
            cut: dict[str, torch.Tensor], pools: tuple[torch.Tensor, torch.Tensor], rec: dict[str, Any], root: Path) -> None:
    label, sh = rec["arm"], rec["shared"]
    lr = rules.RATE[label] * scale
    k8 = lambda sums: sum(r[d3.POOLED] for r in sums["K"][: teach.READ_ROWS]) / (teach.READ_ROWS * rs.tokens.shape[1])
    e93_k8 = k8(E93.rules.load(n)["clean"]["moved"])  # the zero-offset state's reading must reproduce it
    log(f"[{n} {arm}] lr {lr:.3g}; 8-row K of least squares, this card's clean read {k8(sh['clean']['ls_moved']):.9g}, E93's {e93_k8:.9g}")

    def early(name: str, v: dict[str, Any]) -> None:
        check(f"{n}_{arm}_{name}", name, v | ({"e93_K8": e93_k8, "band": sh["band"]} if name == "zero_offsets" else {}))

    d.use("W0")
    d.park_gamma()
    dev = {s: dW[s].to(d3.DEV, torch.float32) for s in SITES}
    off = teach.FromLS(d, cut, p.X, {s: p.moved[s].to(d3.DEV) for s in SITES})
    rows8 = [t[None].to(d3.DEV) for t in rs.tokens[: teach.READ_ROWS]]
    tr = teach.teach(d, off, dev, pools, rows8, lr, rules.STEPS, rules.EVERY, lambda st: rules.forecast(label, scale, st), early, log)
    del dev, rows8
    off.save(root / "large" / f"{n}_{arm}_offsets.safetensors", torch.bfloat16, {"state": n, "arm": arm, "lr": str(lr)}, log)
    mv, gap = teach.moved_total(d, p.X, off)
    check(f"{n}_{arm}_consistency", "consistency", {"gap": gap})
    a: dict[str, Any] = {"train": tr, "split": split_sums(dW, mv, p.sums), "consistency_gap": gap}
    log(f"[{n} {arm}] kept step {tr['kept_step']}, r2 {tr['readings'][-1]['r2']:.6g}; printed share "
        f"{a['split']['left'] / a['split']['size']:.9g} (least squares {p.sums['left'] / p.sums['size']:.9g}), squared size "
        f"{a['split']['moved'] / a['split']['size']:.9g}")
    d.weights.put("student", mv)
    a["clean"] = {"moved": E93.clean_at(d, rs, "student")}
    a["gain"] = E93.gain(d, "student", seqs)
    a["own"] = E93.cells_timed(d, rs, f"{n} {arm} own, Δ frozen", change=teach.trained_change(d, p.X, off), w0_v=False, state="student")
    d.weights.drop("student")
    del off
    torch.cuda.empty_cache()
    for s in SITES:  # the remainder δW − moved part, in place of the moved part
        mv[s].neg_().add_(dW[s].float())
    a["clean"]["remainder"] = E93.clean_at(d, rs, "remainder", mv)
    rec["arms"][arm] = a | {"done": True}
    write(n, rec)
    t = rules.score()["states"][n]
    for row in ("ls_frozen", arm):
        x = t["ls_frozen"] if row == "ls_frozen" else t["arms"][arm]
        log(f"[{n} {row}] verdict {x['verdict']} rules {json.dumps(x['rules'], default=float)} printed {json.dumps(x['printed'], default=float)}")


# ----------------------------------------------------------------------------- one state, the job


def read_state(d: d3.Decomp, rows: dict[str, d3.RowSet], seqs: dict[str, Any], n: str, stage: int, cut: dict[str, torch.Tensor],
               tok: Any, root: Path, F: dict[str, Any]) -> None:
    e93 = E93.rules.load(n)
    _, rk, label = E93.rules.STATES[n]
    rs = rows[rk]
    rec = rules.load(n) or {"name": n, "arm": label, "rows": rk, "stage": stage, "arms": {}}
    path = root / "dW" / f"{n}.safetensors"
    dW = d3.load_state({"path": path, "dtype": e93["dtype"]})
    check(f"{n}_norm", "norm", {"norm": d3.norm(dW), "e93": e93["norm"], "sha256": sha256(path), "e93_sha256": e93["sha256"]})
    d.weights.put("FT", dW)
    p = shared_reads(d, rs, n, e93, dW, rec, F)
    pools = (teach.tofu_pool(tok) if rk == "tofu" else teach.gsm8k_pool(tok), teach.web_pool())
    rec["pool"] = {"overlap16": teach.overlap(rs.tokens, pools[0]), "rows": [len(x) for x in pools]}
    log(f"[{n} pool] {json.dumps(rec['pool'])}")
    for arm, scale in rules.ARMS[stage]:
        if rec["arms"].get(arm, {}).get("done"):
            log(f"[{n} {arm}] already written")
            continue
        t0 = time.time()
        run_arm(d, rs, seqs[rk], n, arm, scale, p, dW, cut, pools, rec, root)
        log(f"[{n} {arm}] done in {time.time() - t0:.0f} s, {d3.peaks()}")
    d.weights.drop("FT")
    write(n, rec | {"done": True})


def main() -> None:
    ap = argparse.ArgumentParser(description="E111: teacher training from least squares on E93's table")
    ap.add_argument("--stage", type=int, choices=sorted(rules.STAGES), required=True)
    ap.add_argument("--root", type=Path, default=Path("/root/e111"))
    a = ap.parse_args()
    if not PRED.is_file() or "status" in json.loads(PRED.read_text()):
        sys.exit(f"refusing to start: {PRED} missing or has a 'status' key")
    if E93.STOPPED.is_file():
        sys.exit(f"refusing to start: {E93.STOPPED} exists (an early check missed; the lead decides)")
    if a.stage > 1 and not (rules.load("R_s70") or {}).get("done"):
        sys.exit("refusing: stage 1 (RL seed 70) is not done")
    P, P93 = json.loads(PRED.read_text()), json.loads((EXPERIMENTS / "qwen_d1_table" / "results" / "predictions.json").read_text())
    assert P["first_minutes"] == rules.FIRST and P["thresholds"] == rules.THRESHOLDS, "predictions.json differs from rules.py"
    names = rules.STAGES[a.stage]
    log(f"[start] {vars(a)}; states {names}; arms {rules.ARMS[a.stage]}")
    fetch_inputs(a.root, names)
    rows, seqs = E93.build(P93)
    tok = AutoTokenizer.from_pretrained(qwen_eval.MODEL, revision=qwen_eval.MODEL_REV)
    cut = load_file(str(a.root / "e80_inputs" / "cut.safetensors"))
    F = {"gpu_max_gib": rules.FIRST["gpu_max_gib"], "host_max_gb": 0.9 * host_total_gb()}
    with torch.no_grad():
        d = d3.Decomp(a.root / "qwen", log)
        d.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for n in names:
            if (rules.load(n) or {}).get("done"):
                log(f"[{n}] already written")
                continue
            t0 = time.time()
            read_state(d, rows, seqs, n, a.stage, cut, tok, a.root, F)
            log(f"[{n}] done in {time.time() - t0:.0f} s, {d3.peaks()}")
    t = rules.score()
    log(f"[leans] {json.dumps(t['leans'], default=float)}")
    log(f"[done] stage {a.stage}; FINISHED")
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e.code, flush=True)
        hard_exit(e.code if isinstance(e.code, int) else 1 if e.code else 0)
    except BaseException:  # noqa: BLE001  (an out-of-memory or the like: exit 1, never a hang in teardown)
        traceback.print_exc()
        hard_exit(1)
