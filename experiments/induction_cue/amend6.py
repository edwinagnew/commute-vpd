"""E62 Amendment 6: does W0's mask hold the answer, or open a route that copies it from the text? W0 only, at the second A (half - 1), on E62's 200 read windows:
the mask stays as read on the intact tokens, B in the text is replaced by another token of its pool, and B and the new token are read. No training, no state read,
nothing in results/large overwritten or deleted. Plan: reports/plans/2026-11-02_plan_e62-copying-heads.md, "Amendment 6 (2026-11-02)". Rules and expected values:
results/predictions_amend6.json (committed first; the run refuses otherwise; never edited), `commute.copying`. Amendment 5's filed verdict stands.

Each forward reads four cells, through one rule (copying.overwrite_reading): B and the new token, on the intact tokens and with B overwritten.
  a control, alone: every mask at 1, then the target model. Each must re-read Amendment 5's level, read mask_opens_route and pass copying.overwrite_control; a miss is exit 1.
  b the masked forward under W0's mask. Its first cell, B on the intact tokens, is the do-nothing and must re-read Amendment 5's level: a miss is exit 1, verdict reread_failed.
    A mask verdict of unreadable is exit 1 too. After the verdict, report-only: B and the new token under Γ's own mask on the overwritten tokens, and how often each is top-1.
  c the same four cells in fp32 under Γ's fp32 mask, beside the verdict. A disagreement, or a first cell off Amendment 5's, is a LOUD line, and nothing here stops the job.
Masked forward and Γ bf16 (autocast) but for (c), target model fp32, as Amendment 5. results/amend6.json and results/large/amend6_windows.pt after every stage, every number printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import torch
from amend5 import Gates, Reads5
from torch import Tensor

from commute import copying as CP
from commute import keydep as KD
from commute.open_pair import jsonable

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, BASE, RUN, A5_OUT, OUT = (RESULTS / f for f in ("predictions_amend6.json", "predictions.json", "run.json", "amend5.json", "amend6.json"))
log = KD.log
Read = Callable[[Tensor], Tensor]


@dataclass
class Job:
    R: Reads5  # at the second A
    W: dict[str, Tensor]  # on, b_over, new_intact, new_over: tokens and target last
    saved: dict[str, Tensor]  # amend5_windows.pt
    th: dict[str, Any]
    out: dict[str, Any]
    results: Path
    windows: Path
    rows: dict[str, Tensor] = field(default_factory=dict)

    def dump(self) -> None:
        self.results.write_text(json.dumps(jsonable(self.out), indent=1, sort_keys=True))
        torch.save(self.rows, self.windows)

    def note(self, key: str, rec: dict[str, Any]) -> None:
        self.out[key] = rec
        self.dump()
        log(f"[{key}] {json.dumps(jsonable(rec))}")

    def stop(self, verdict: str, why: str) -> NoReturn:
        self.out["stopped"] = why
        self.out["verdict"] = {"verdict": verdict}
        self.dump()
        sys.exit(f"STOP: {verdict}: {why}: tell the lead")


def cells(J: Job, name: str, read: Read, on_ref: Tensor, lean: Tensor) -> dict[str, Any]:
    """One forward's four cells and its reading, saved. `on_ref` is Amendment 5's row of the first cell; `lean` Amendment 5's per-window lean, the denominator."""
    n_boot = J.R.n_boot
    on, b, ni, no = (J.rows.setdefault(f"{name}_{k}", read(J.W[k]).detach().cpu()) for k in ("on", "b_over", "new_intact", "new_over"))
    d = float(lean.mean())
    reading = CP.overwrite_reading(float(on.mean()), float(b.mean()), float(ni.mean()), float(no.mean()), d, J.th)
    rec = {"on": float(on.mean()), "on_diff_a5": abs(float((on - on_ref).mean())), "on_window_max_abs_a5": float((on - on_ref).abs().max()), "b_over": float(b.mean()), "new_intact": float(ni.mean()), "new_over": float(no.mean()), "lean": d,
           **reading, "lost_sd": CP.share_sd(lean, lean - (on - b), n_boot, 0), "gained_sd": CP.share_sd(lean, lean - (no - ni), n_boot, 0), "gap_sd": CP.boot_sd(b - no, n_boot, 0)}
    J.note(name, rec)
    return rec


def reread(J: Job, name: str, rec: dict[str, Any]) -> None:
    if rec["on_diff_a5"] > J.th["reread_tol"]:
        J.stop("reread_failed", f"{name}: B on the intact tokens differs from Amendment 5's by {rec['on_diff_a5']}")


def control(J: Job, name: str, read: Read) -> None:
    s = J.saved
    rec = cells(J, name, read, s[f"{name}_on"], s[f"{name}_on"] - s[f"{name}_broken"])
    reread(J, name, rec)
    status = CP.overwrite_control(rec["on"], rec["new_over"], rec, J.th)
    J.note(f"{name}_control", {"status": status})
    if status != "pass":
        J.stop("control_failed", f"{name}: {status}; lost {rec['lost']:.3f}, gained {rec['gained']:.3f}, new token at {rec['new_over']:.3f} against {rec['on']:.3f}; the mask is not read")


def report_only(J: Job, g0: Gates) -> None:
    """After the verdict, deciding nothing: B and the new token under Γ's own mask on the overwritten tokens, how often each is top-1, and B's level as a probability."""
    R, W, r = J.R, J.W, J.rows
    b, new = W["on"][:, -1], W["new_intact"][:, -1]
    over = W["b_over"]
    g_own = R.gates(over)
    top = lambda g, w: CP.masked_top1(R.ctx.net.model, w, R.half, g, KD.BATCH, R.n)
    r["own_b_over"], r["own_new_over"] = R.masked(over, g_own, []).detach().cpu(), R.masked(W["new_over"], g_own, []).detach().cpu()
    r["top_mask_intact"], r["top_mask_over"], r["top_own_over"] = top(g0, W["on"]), top(g0, over), top(g_own, over)
    J.note("own_mask", {"b_over": float(r["own_b_over"].mean()), "new_over": float(r["own_new_over"].mean())})
    rate = lambda name, token: float((r[name] == token).double().mean())
    J.note("top1", {"mask_on": rate("top_mask_intact", b), "mask_new_intact": rate("top_mask_intact", new), "mask_b_over": rate("top_mask_over", b), "mask_new_over": rate("top_mask_over", new),
                    "own_b_over": rate("top_own_over", b), "own_new_over": rate("top_own_over", new)})
    J.note("b_probability", {k: {"at_mean_log_odds": float(torch.sigmoid(r[k].mean())), "mean_over_windows": float(torch.sigmoid(r[k]).mean())} for k in ("mask_on", "mask_b_over", "own_b_over")})


def run(J: Job) -> None:
    R, on, s = J.R, J.W["on"], J.saved
    g0 = R.gates(on)
    ones = [{k: torch.ones_like(v) for k, v in batch.items()} for batch in g0]
    control(J, "ones", lambda w: R.masked(w, ones, []))
    control(J, "target", lambda w: R.lo(w, []))
    lean = s["mask_on"] - s["mask_broken"]
    mask = cells(J, "mask", lambda w: R.masked(w, g0, []), s["mask_on"], lean)
    reread(J, "mask", mask)
    if mask["verdict"] == "unreadable":
        J.stop("unreadable", f"the mask's cells are not finite or its lean {mask['lean']} is under lean_min")
    v = {k: mask[k] for k in ("verdict", "lost", "gained", "gap")}
    J.note("verdict", v)
    log(f"[verdict] {v['verdict']} lost {v['lost']:.3f} gained {v['gained']:.3f} B minus new token {v['gap']:.3f}")
    report_only(J, g0)
    g32 = R.gates32(on)
    fp32 = cells(J, "fp32", lambda w: R.masked32(w, g32, []), s["gamma32_on"], lean)
    J.note("verdict", v | {"fp32_verdict": fp32["verdict"]})
    if fp32["on_diff_a5"] > J.th["reread_tol"]:
        log(f"*** LOUD: THE FP32 ROW'S FIRST CELL IS OFF AMENDMENT 5'S BY {fp32['on_diff_a5']}. About 0.005 means the row ran under autocast and read bf16; fp32_verdict is then no precision check. The filed verdict stands.")
    if v["verdict"] != fp32["verdict"]:
        log(f"*** LOUD: THE TWO PRECISIONS DISAGREE: bf16 {v['verdict']}, fp32 {fp32['verdict']}. The filed verdict is bf16's; the reading goes to the user with both.")


def overwrite(toks: Tensor, p: Tensor, b: Tensor, half: int, seed: int) -> tuple[Tensor, dict[str, Tensor], bool, dict[str, Any]]:
    """B overwritten by another of the read targets: the new token per window, the four target-last token sets, whether it holds as filed, and what to print."""
    pool = b.unique().tolist()
    over = CP.absent_b(toks, p, b, pool, seed)
    new = over[torch.arange(len(over)), p + 1]
    cat = lambda t, y: torch.cat([t, y[:, None]], 1)
    info = {"pool": len(pool), "seed": seed, "new_tokens": len(new.unique()), "new_elsewhere_in_window": int(((over == new[:, None]).sum(1) > 1).sum())}
    return new, {"on": cat(toks, b), "b_over": cat(over, b), "new_intact": cat(toks, new), "new_over": cat(over, new)}, CP.overwrite_holds(toks, over, p, b, half), info


def load(a: argparse.Namespace) -> Job:
    rel = str(PRED.relative_to(REPO))
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True, check=False)
    if not PRED.is_file() or not git("ls-files", rel).stdout.strip() or git("diff", "--quiet", "HEAD", "--", rel).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    large = a.large.resolve()
    windows = large / "amend6_windows.pt"
    for f in (OUT, windows):
        if f.exists():
            sys.exit(f"{f} exists: move it aside, this run deletes nothing")
    P, S, run_json, a5 = json.loads(PRED.read_text()), json.loads(BASE.read_text())["settings"], json.loads(RUN.read_text()), json.loads(A5_OUT.read_text())
    if KD.sha256(large / "windows.pt") != run_json["windows_sha256"]:
        sys.exit("windows.pt is not the one run.json recorded")
    saved = torch.load(large / "amend5_windows.pt", weights_only=True)
    if any(float(saved[k].mean()) != a5["stage_a_reread"][k] for k in ("mask_on", "mask_broken", "target_on", "target_broken")):
        sys.exit("amend5_windows.pt is not the rows amend5.json recorded")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, large)
    v = torch.load(large / "windows.pt", weights_only=False)
    half, n = S["half"], 2 * S["half"] + 1
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD").stdout.strip(), "predictions_sha256": KD.sha256(PRED), "host": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}
    new, W, holds, info = overwrite(v["read"].tokens, v["p_read"], v["target"]["copy"]["read"], half, S["control_seed"])
    means = torch.load(large / "heads.pt", weights_only=False)["means"].to(ctx.net.device)  # `Reads`' hook needs it; no head goes off here
    R = Reads5(ctx, half - 1, n, means, S["n_boot"], {"copy": W["on"]}, W["on"], {}, run_json)
    J = Job(R, W, saved, P["thresholds"], out, OUT, windows, {"new_token": new})
    J.note("overwrite", info)
    if not holds:
        J.stop("overwrite_not_as_filed", "B is not replaced on every window, or something else moved")
    return J


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large")
    J = load(ap.parse_args())
    log(f"[host] {J.out['host']}: masked forward and Γ bf16 but for the fp32 row, target model fp32")
    run(J)


if __name__ == "__main__":
    main()
