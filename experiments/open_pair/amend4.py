"""open_pair Amendment 4 (E58, 2026-10-30, after job 46398): seeds for the within-key-count reading and the shuffle's r.
Fable, reports/plans/2026-10-30_advice_e58a3-next-FABLE.md, item 2. Plan: reports/plans/2026-10-30_plan_e58-open-pair.md,
section "Amendment 4". Rules, settings and thresholds: results/predictions_amend4.json (filed and committed first).

  F  Amendment 3's split windows (<large>/amend3), sha256 checked against amend3.json; exit 1 on a mismatch.
  per seed (0, 1, 2), the cipher first and alone: fine-tune (every step 1-30), its label at step 10, then up to step 30
     until G_spec reaches the gate (`open_pair.pick_cipher_step`); exit 1 unless the control reads "pass" or "below".
     Its r and ratio at 10, 30 and the first step over the gate (item 2.3, the drift line). Seed 0 repeats Amendment 3's
     cipher (a reproduction, reported).
  seeds 1, 2: elicit, rank-1, copy fine-tuned (E54's recipe), each read at steps 10 and 30 (label, x, r, ratio, S;
     exit 1 on a block-0 shift). Seed 0's reads are Amendment 3's (amend3.json), not re-run.
  shuffle (seed 0): fine-tuned with every step 1-30 kept; read at step 30 and at its first step over the gate from 10.
  score `open_pair.score_amend4` -> results/amend4.json. Every number is also printed. No weights are saved (training
     is bitwise reproducible on this card: Amendment 3's reproduction read a max |difference| of 0).
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import amend2
import amend3
import torch
from amend1 import git, load_split
from amend3 import read

from commute import keydep as KD
from commute import open_pair as OP
from commute.finetune import FinetuneResult
from commute.heldout_keys import PARTS

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT, A3 = RESULTS / "predictions_amend4.json", RESULTS / "amend4.json", RESULTS / "amend3.json"
MAPS = ("elicit", "rank1", "copy")
STEPS = (10, 30)
log = KD.log
amend2.OUT = amend3.OUT = OUT  # their dump and stop write this run's file


def dump(out: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(OP.jsonable(out), indent=1, sort_keys=True))


def stop(out: dict[str, Any], why: str) -> NoReturn:
    out["stopped"] = why
    dump(out)
    sys.exit(f"STOP: {why}: tell the lead")


def train(ctx: KD.Ctx, d: Path, n: str, seed: int, checks: tuple[int, ...]) -> FinetuneResult:
    ft = KD.finetune(ctx, torch.load(d / f"windows_{n}_trained.pt", weights_only=False), seed, checks)
    log(f"[ft {n} s{seed}] p_heldout {({k: round(e['p_o'], 4) for k, e in ft.evals.items() if k in (10, 16, 30, 100, 300)})}")
    return ft


def over_gate(ctx: KD.Ctx, name: str, t: dict[str, Any], ft: FinetuneResult, out: dict[str, Any]) -> int | None:
    """Label reads from step 10 up to 30 until G_spec reaches the gate; the step `pick_cipher_step` picks (None if none)."""
    g_min = ctx.P["thresholds"]["G_min"]
    g, s = {10: read(ctx, name, t, ft, 10, out, vpd=False)["G_spec"]}, 10
    while OP.finite(g[s]) and g[s] < g_min and s < 30:
        s += 1
        g[s] = read(ctx, name, t, ft, s, out, vpd=False)["G_spec"]
    return OP.pick_cipher_step(g, 10, 30, g_min)


def cipher(ctx: KD.Ctx, t: dict[str, Any], d: Path, seed: int, out: dict[str, Any]) -> None:
    """The control for one seed, first and alone, then its r and ratio at 10, 30 and the first step over the gate."""
    th, name = ctx.P["thresholds"], f"cipher_s{seed}"
    ft = train(ctx, d, "cipher", seed, KD.CIPHER_CHECKS)
    at = over_gate(ctx, name, t, ft, out)
    rec: dict[str, Any] = {"step": at}
    out["control"][str(seed)] = rec
    if at is None:
        stop(out, f"{name}: under the gate {th['G_min']:.3f} at every step from 10 to 30")
    c = out["reads"][f"{name}@{at}"]
    rec["status"] = OP.elicit_control(c["x"], c["label"], th)
    log(f"[control {name}] step {at}: x {c['x']:.4f} G_spec {c['G_spec']:.4f} label {c['label']} -> {rec['status']}")
    dump(out)
    if rec["status"] not in ("pass", "below"):
        stop(out, f"{name}: the control reads {rec['status']}")
    for s in sorted({10, 30, at}):
        read(ctx, name, t, ft, s, out, vpd=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="reads <large>/amend3; writes nothing large")
    a = ap.parse_args()
    rel = str(PRED.relative_to(REPO))
    if not PRED.is_file() or not git("ls-files", rel) or subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    P, a3 = json.loads(PRED.read_text()), json.loads(A3.read_text())
    th, d = P["thresholds"], a.large.resolve() / "amend3"
    code = [str(p.relative_to(REPO)) for p in (HERE / "amend4.py", HERE / "amend3.py", HERE / "amend2.py", HERE / "amend1.py",
                                                *(REPO / "src" / "commute" / f for f in ("open_pair.py", "heldout_keys.py", "keydep.py")))]
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD"), "git_status_code": git("status", "--porcelain", "--", *code),
                           "predictions_sha256": KD.sha256(PRED), "amend3_sha256": KD.sha256(A3), "reads": {}, "control": {}}
    bad = [f"{n}_{p}" for n in (*MAPS, "cipher", "shuffle") for p in PARTS if KD.sha256(d / f"windows_{n}_{p}.pt") != a3["maps"][n]["sha256"][p]]
    if bad:
        stop(out, f"Amendment 3's windows changed: {bad}")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, d)
    out["host"] = {"hostname": socket.gethostname(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}
    log(f"[host] {out['host']}")
    tasks = {n: load_split(ctx, d, n) for n in (*MAPS, "cipher", "shuffle")}
    for seed in P["settings"]["seeds"]:
        cipher(ctx, tasks["cipher"], d, seed, out)
        if seed == 0:  # seed 0's maps are Amendment 3's reads, used only if seed 0's cipher reproduces Amendment 3's r
            pairs = [(out["reads"][f"cipher_s0@{s}"]["r"], a3["reads"][f"cipher@{s}"]["r"]) for s in STEPS]
            ok = out["seed0_comparable"] = OP.comparable(pairs, th["reproduce_tol"])
            log(f"[seed 0] cipher r here vs amend3.json at {STEPS}: {pairs} -> comparable {ok}")
            continue
        for n in MAPS:
            ft = train(ctx, d, n, seed, KD.E12_CHECKS)
            for s in STEPS:
                read(ctx, f"{n}_s{seed}", tasks[n], ft, s, out, vpd=True)
    ft = train(ctx, d, "shuffle", 0, KD.CIPHER_CHECKS)
    at = out["shuffle_gate_step"] = over_gate(ctx, "shuffle_s0", tasks["shuffle"], ft, out)
    for s in sorted({30, at if at is not None else 30}):
        read(ctx, "shuffle_s0", tasks["shuffle"], ft, s, out, vpd=True)
    del ft
    out["shuffle"] = {k: {f: out["reads"][k].get(f) for f in ("G_spec", "label", "x", "r", "ratio")} for k in out["reads"] if k.startswith("shuffle_s0@")}
    log(f"[shuffle] {json.dumps(OP.jsonable(out['shuffle']))}")
    seed_reads = lambda n, seed, s: ((a3["reads"].get(f"{n}@{s}") if out["seed0_comparable"] else None) if seed == 0 else out["reads"].get(f"{n}_s{seed}@{s}"))
    reads = {s: {n: [seed_reads(n, seed, s) for seed in P["settings"]["seeds"]] for n in MAPS} for s in STEPS}
    out["score"] = OP.score_amend4(reads, out["reads"].get("shuffle_s0@30"), th)
    out["D_r_beside_amend3_tol_r"] = {s: [w["r"].get("D"), th["amend3_tol_r"]] for s, w in out["score"]["within_key"].items()}
    out["cipher_s0_reproduces_amend3"] = {s: [out["reads"].get(f"cipher_s0@{s}", {}).get("r"), a3["reads"].get(f"cipher@{s}", {}).get("r")] for s in (10, 30)}
    dump(out)
    log(f"[score] {json.dumps(OP.jsonable(out['score']))}")


if __name__ == "__main__":
    main()
