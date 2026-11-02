"""open_pair Amendment 5 (E58, 2026-10-30, after job 46426): the shuffled elicit map at seeds 1 and 2, reported beside
seed 0 (amend4.json), not scored. Fable, reports/plans/2026-10-30_advice_e58a4-synthesis-FABLE.md, item 3. Plan:
reports/plans/2026-10-30_plan_e58-open-pair.md, section "Amendment 5". The write-up rule: results/predictions_amend5.json.

No new measurement code: Amendment 4's `train` and `over_gate` and Amendment 3's `read`, on Amendment 3's shuffle
windows (sha256 checked against amend3.json). Per seed: fine-tune (every step 1-30 kept), label reads from step 10 until
G_spec reaches the gate, then r, ratio, G_spec, x at step 30 and at that step. `open_pair.shuffle_sentence` -> results/amend5.json.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import amend2
import amend3
import amend4
import torch
from amend1 import git, load_split
from amend3 import read
from amend4 import over_gate, train

from commute import keydep as KD
from commute import open_pair as OP
from commute.heldout_keys import PARTS

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
PRED, OUT = RESULTS / "predictions_amend5.json", RESULTS / "amend5.json"
amend2.OUT = amend3.OUT = amend4.OUT = OUT  # their dump and stop write this run's file
FIELDS = ("G_spec", "label", "x", "r", "ratio")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alive", type=Path, default=HERE.parent / "transport_test" / "results" / "large" / "alive.pt")
    ap.add_argument("--large", type=Path, default=RESULTS / "large", help="reads <large>/amend3; writes nothing large")
    a = ap.parse_args()
    rel = str(PRED.relative_to(REPO))
    if not PRED.is_file() or not git("ls-files", rel) or subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", rel], check=False).returncode:
        sys.exit(f"{rel} is not committed unchanged: file and commit it before the run")
    P = json.loads(PRED.read_text())
    a3, a4 = (json.loads((RESULTS / f).read_text()) for f in ("amend3.json", "amend4.json"))
    d = a.large.resolve() / "amend3"
    out: dict[str, Any] = {"git_head": git("rev-parse", "HEAD"), "predictions_sha256": KD.sha256(PRED), "reads": {}}
    if any(KD.sha256(d / f"windows_shuffle_{p}.pt") != a3["maps"]["shuffle"]["sha256"][p] for p in PARTS):
        amend4.stop(out, "Amendment 3's shuffle windows changed")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ctx = KD.make_ctx(a.device, a.alive, P, d)
    out["host"] = {"hostname": socket.gethostname(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}
    t = load_split(ctx, d, "shuffle")
    out["seed0"] = {"gate_step": a4["shuffle_gate_step"], **{k.split("@")[1]: v for k, v in a4["shuffle"].items()}}
    for seed in P["settings"]["new_seeds"]:
        name = f"shuffle_s{seed}"
        ft = train(ctx, d, "shuffle", seed, KD.CIPHER_CHECKS)
        at = out[f"gate_step_s{seed}"] = over_gate(ctx, name, t, ft, out)
        for s in sorted({30, at if at is not None else 30}):
            read(ctx, name, t, ft, s, out, vpd=True)
        del ft
    r_at = lambda seed: (out["reads"].get(f"shuffle_s{seed}@30") or {}).get("r")
    out["table"] = {"seed0": out["seed0"], **{f"s{s}": {k.split("@")[1]: {f: v.get(f) for f in FIELDS} for k, v in out["reads"].items()
                                                         if k.startswith(f"shuffle_s{s}@")} for s in P["settings"]["new_seeds"]}}
    r30 = [out["seed0"]["30"]["r"], *(r_at(s) for s in P["settings"]["new_seeds"])]
    out["r30"], out["r30_mean"] = r30, sum(r for r in r30 if OP.finite(r)) / max(1, sum(OP.finite(r) for r in r30))
    out["sentence"] = OP.shuffle_sentence(r30[0], r30[1:], P["rule"])
    amend3.dump(out)
    KD.log(f"[amend5] r at step 30, seeds 0-2: {r30} mean {out['r30_mean']:.4f} -> {out['sentence']}")


if __name__ == "__main__":
    main()
