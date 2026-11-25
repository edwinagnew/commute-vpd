"""E306 (P2.80): Fig 4's 48 states -> (task, saved file, sha256), written into p2-80/predictions.json settings.fig4
before filing. JSON and hashing only, no model; run from the repo root:

    ~/commute-agents/fleet/cpu-run python3 experiments/weightwatch/p2-80/resolve.py

Each state is found where the readings of Fig 4 found it, and its file is read as saved (c = 1, as known_3seeds'
`Readers.read` reads it, experiments/known_3seeds/run.py:161-162, and as E294 read E62's, weightwatch/run.py:311-313):
    e58/<m>_s<k>     refit_96key/results/stage0.json .states (the file and sha256 eapig_overlap/items.py:115 and
                     cmap/job.py:114 use)
    the five seed-2 states known_3seeds trained   known_3seeds/results/stage2.json .states (file, sha256)
    every other      share_untrained/results/inventory.json .states (E136's file and task, as cmap/job.py:124)
A recorded sha256 (E147's, known_3seeds', E62's run.json) must equal the file's; the 25 without one are recorded here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path.cwd()
EXP = REPO / "experiments"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load(rel: str) -> dict:
    return json.loads((EXP / rel).read_text())


def main() -> int:
    cells = load("known_3seeds/results/known3.json")["cells"]
    states = sorted({s for c in cells for side in c["states"].values() for s in side})
    inv = load("share_untrained/results/inventory.json")["states"]
    e147 = load("refit_96key/results/stage0.json")["states"]
    k3 = load("known_3seeds/results/stage2.json")["states"]
    e62 = load("induction_cue/results/run.json")["states"]
    out, miss = {}, []
    for s in states:
        name = s.rsplit("_s", 1)[0]
        if s.startswith("e58/"):
            rec = {"task": name, "file": e147[s]["file"], "recorded": e147[s]["sha256"]}
        elif s in k3:
            task = inv[f"{name}_s0"]["task"]  # the seed-2 state's task is its seeds 0-1's (known_3seeds/run.py:67-69)
            rec = {"task": task, "file": k3[s]["file"], "recorded": k3[s]["sha256"]}
        else:
            rec = {"task": inv[s]["task"], "file": inv[s]["file"], "recorded": e62.get(s.removeprefix("e62/"), {}).get("sha256") if s.startswith("e62/") else None}
        p = REPO / rec["file"]
        if not p.is_file():
            miss.append(f"{s}: {rec['file']} missing")
            continue
        got = sha256(p)
        if rec["recorded"] not in (None, got):
            miss.append(f"{s}: sha256 {got} != recorded {rec['recorded']}")
        out[s] = {"task": rec["task"], "file": rec["file"], "sha256": got}
        print(f"{s:24s} {rec['task']:16s} {rec['file']} {got[:12]} {'checked' if rec['recorded'] else 'recorded here'}", flush=True)
    assert len(states) == 48, f"{len(states)} states, not 48"
    if miss:
        print("MISS:", *miss, sep="\n  ")
        return 1
    pred = Path(__file__).resolve().parent / "predictions.json"
    if len(sys.argv) > 1 and sys.argv[1] == "--write":
        d = json.loads(pred.read_text())
        d["settings"]["fig4"] = out
        pred.write_text(json.dumps(d, indent=2) + "\n")
        print(f"wrote settings.fig4 ({len(out)} states) into {pred}")
    else:
        print(json.dumps(out)[:200], "...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
