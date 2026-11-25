"""broad_fragility Amendment 1 (E39 A1): Test 3's SGD arm again on a lower rate grid.

In E39 the plain-SGD lowercase fine-tune diverged at every rate of its grid {0.3, 1, 3} (held-out gains -0.14, -2.38,
-3.58), so the SGD do-nothing, adjoint split and teacher training were never read and P5_sgd-P7_sgd went unscored.
This job runs run.py's Test 3 for SGD alone with the grid {0.003, 0.01, 0.03, 0.1} and the same pick rule (among rates
that did not diverge, the held-out gain nearest this job's Adam fine-tune's). Everything is run.py's, imported:

  stage 0  (i) W0 on adversary seeds 0-2: spread, and seed 0's PGD KL per block against E37's (0.6794, 0.7182);
           (ii) E37's Adam fine-tune regenerated (held-out gain 0.1301) and E37's teacher training on it read
           (check 2.2011, s 0.998), the positive control the filed P6 names. Any miss stops the job.
  adam     the Adam fine-tune's stable rank and outside share (P5's reference)
  sgd      the grid (each rate's gain logged), then the chosen rate's stable rank, outside share, and the check of
           do-nothing, the scaled adjoint split and teacher training at E37's recipe (state saved as tt_sgd_step1000.pt)

Results to results/amend1.json, log to results/large/amend1.log (E39's files are not touched). Scored: P5_sgd-P7_sgd
(the filed P5-P7 rules, copied) and P8 (the chosen rate's gain within 30% of Adam's).

Usage (repo root, project venv):  amend1.py --predictions   files results/predictions_amend1.json (once)
                                  amend1.py [--device cuda]  the run (GPU); refuses without that file
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import run
from run import (
    CFG,
    LARGE,
    RESULTS,
    TRANSPORT,
    Job,
    adam_shape,
    dump,
    log,
    other_update,
    stage0_trained,
    stage0_w0,
)

PRED, OUT = RESULTS / "predictions_amend1.json", RESULTS / "amend1.json"
run.OUT, run.LOG = OUT, LARGE / "amend1.log"  # run.read, dump, halt and log write here


@dataclass(frozen=True)
class Amend1:
    sgd_lrs: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1)
    p8_tol: float = 0.30  # P8: |gain_sgd / gain_adam - 1| <= this

    def __post_init__(self) -> None:
        assert max(self.sgd_lrs) < min(CFG.sgd_lrs)  # below E39's grid, every rate of which diverged


AM = Amend1()


def config() -> dict[str, Any]:
    return json.loads(json.dumps({"amend1": asdict(AM), "run": asdict(CFG)}))


def score(out: dict[str, Any]) -> dict[str, Any]:
    """run.score's SGD half (P5_sgd-P7_sgd, the filed rules) and P8; a prediction whose reads are missing is unscored."""
    sc = {k: v for k, v in run.score(out).items() if k.endswith("_sgd")}
    sgd = out["shape"].get("sgd", {})
    if "chosen" in sgd:
        sc["P8_ratio"] = None if sgd["chosen"] is None else sgd["gain"] / out["stage0"]["gain"]
        sc["P8"] = sc["P8_ratio"] is not None and abs(sc["P8_ratio"] - 1) <= AM.p8_tol
    return sc


def main(device: str) -> None:
    if not PRED.is_file():
        sys.exit(f"file the predictions first: {PRED} (amend1.py --predictions)")
    filed = json.loads(PRED.read_text())["config"]
    assert filed == config(), f"predictions were filed for another config: {filed}"
    if OUT.is_file():
        sys.exit(f"{OUT} exists; move it aside to start again")
    for f in (TRANSPORT / "large" / "alive.pt", TRANSPORT / "large" / "windows.pt"):
        if not f.is_file():
            sys.exit(f"missing input (gitignored): {f}")
    job = Job(device)
    out: dict[str, Any] = {"card": job.card, "reads": {}, "shape": {}, "done": [], "status": "running"}
    stage0_w0(job, out)
    stage0_trained(job, out)
    for name, fn in (("adam", lambda: adam_shape(job, out)),
                     ("sgd", lambda: other_update(job, out, "sgd", AM.sgd_lrs))):
        fn()
        out["done"].append(name)
        out["score"] = score(out)
        dump(out)
        log(f"[score after {name}] {json.dumps(out['score'], default=float)}")
    gains = {r: round(v["gain"], 4) for r, v in out["shape"]["sgd"]["grid"].items()}
    log(f"[sgd grid] held-out gain per rate {json.dumps(gains)}; Adam {out['stage0']['gain']:.4f}; "
        f"chosen {out['shape']['sgd']['chosen']}")
    out["status"] = "done"
    dump(out)
    log(f"FINISHED wrote {OUT}")


# ----------------------------------------------------------------------------- predictions


def predictions() -> dict[str, Any]:
    """P5-P7 copied from E39's filed file (rule, thresholds and confidence unchanged), scoped to SGD; P8 new."""
    e39 = json.loads(run.PRED.read_text())
    by_id = {p["id"]: p for p in e39["predictions"]}
    copied = [by_id[p] | {"id": f"{p}_sgd", "scope": "the SGD fine-tune only; E39 scored the LoRA half",
                          "copied_from": f"results/predictions.json {p} (filed {e39['filed']})"} for p in ("P5", "P6", "P7")]
    p8 = {"id": "P8",
          "rule": f"the chosen SGD rate's held-out gain within {AM.p8_tol:.0%} of this job's Adam fine-tune gain, "
                  f"|gain_sgd / gain_adam - 1| <= {AM.p8_tol} (relative); false when every grid rate diverged",
          "expected": "yes: a rate in 0.003-0.1 reaches 0.091-0.169", "confidence": "medium",
          "blind_to": "the gain is mean log-probability on 64 held-out lowercase rows only; one batch order",
          "positive_control": "the Adam fine-tune's gain regenerated in stage 0 (ii), 0.1301 in E37",
          "noise_floor": "Adam's gain reproduced to 0.0001 across jobs (0.1300, 0.1301); the batch-order spread of the "
                         "gain was not measured"}
    return {
        "plan": "E39 Amendment 1: Test 3's SGD arm on the grid {0.003, 0.01, 0.03, 0.1}, after E39's {0.3, 1, 3} all "
                "diverged (gains -0.14, -2.38, -3.58). Filed before any run; never edited after.",
        "definitions": e39["definitions"] | {"grid": f"SGD rates {list(AM.sgd_lrs)}, otherwise E39's Test 3 recipe"},
        "controls": {k: e39["controls"][k] for k in ("stage0_i", "stage0_ii")},
        "expected": {
            "stage0": "spread about 0.075, blocks 0.6794 and 0.7182, Adam gain 0.1300, trained check 2.2011, s 0.998 "
                      "(E39's H100)",
            "grid": "gain rises with the rate up to the divergence edge (lr 0.3 read -0.14); the chosen rate 0.03 or "
                    "0.1 (E39's lr 0.3 moved the weights 0.17 relative, Adam 0.049)",
            "stable rank": "SGD below Adam's 28.5 (P5)",
            "do-nothing": "check >= 1.61 (P7; Adam's read 4.87 in E37)",
            "adjoint": "check about 1.0-1.2 (Adam's scaled adjoint split read 1.05 in E37), s below 1",
            "teacher training": "check >= 1.61 (P6; Adam's 2.20)",
        },
        "predictions": [*copied, p8],
    }


def file_predictions() -> None:
    if PRED.is_file():
        sys.exit(f"{PRED} exists; filed predictions are never edited (add a dated report instead)")
    PRED.write_text(json.dumps({**predictions(), "config": config(), "filed": time.strftime("%Y-%m-%d %H:%M:%S")},
                               indent=1))
    print(f"filed {PRED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="store_true", help="file results/predictions_amend1.json and exit")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    file_predictions() if a.predictions else main(a.device)
