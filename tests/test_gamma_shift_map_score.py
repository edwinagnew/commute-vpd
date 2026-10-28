"""CPU run of experiments/gamma_shift_map/run.py's score() (and plot.py's figures) on a synthetic results dict of the
run's shape, so a scoring bug cannot crash the GPU run after its last reading."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parents[1] / "experiments" / "gamma_shift_map"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _reading(ci: float, pgd: float, sp: float) -> dict:
    return {"stochastic_kl": 0.3, "ci_masked_kl": ci, "unmasked_kl": 0.1, "pgd_kl": pgd, "sparsity": [sp, sp * 1.01, sp, sp]}


def _readout(top100: float) -> dict:
    return {"counts": {"alive": 9500}, "mass": {"by_layer": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4},
                                               "by_sign": {"up": 0.6, "down": 0.4}},
            "R_whole": 0.7, "R_by_layer": {str(i): 0.1 * i for i in range(4)}, "R_halves": {"up": 0.4, "down": 0.3},
            "R_topk": {}, "topk_share_of_whole": {"10": 0.1, "30": 0.2, "100": top100, "300": 0.8, "1000": 0.95}}


def _out() -> dict:
    pile = {f"W0/seed{s}": _reading(0.348, 0.69 + 0.01 * s, 100.0) for s in (0, 1, 2)}
    ups = {}
    for i, label in enumerate(("emoticon/sgd_0.01", "emoticon/adam_3e-05", "r1_t534_v369/sgd_0.03")):
        ups[label] = {"G1": {"ok": True, "ref": {"R_shared": 0.74 if i < 2 else 0.40}},
                      "R_everywhere": {"a_zero": 0.0, "b_shift": 0.7 if i < 2 else 0.45, "c_null": -0.2, "d_shuffled": 0.0},
                      "null_heldout": {"p_o": 0.9}, "readout": {"shift": _readout(0.3 + 0.3 * i)}}
        pile.update({f"{label}/a_zero": _reading(0.36, 0.70, 100.0), f"{label}/b_shift": _reading(0.45, 0.60, 130.0),
                     f"{label}/c_null": _reading(0.40, 0.65, 120.0), f"{label}/d_shuffled": _reading(0.5, 0.7, 125.0),
                     f"{label}/nothing": _reading(0.35, 0.72, 101.0)})
    return {"pile": pile, "updates": ups, "C1": {"w0_check": {"pgd_floor": 0.029}, "ci_floor": 0.0}}


def test_score_reads_the_rules(tmp_path):
    run = _load("gamma_shift_map_run", EXP / "run.py")
    out = _out()
    sc = run.score(out)
    assert sc["P1"]["pass"] is True
    assert sc["P2"]["reading"]["emoticon/sgd_0.01"] == "harm" and sc["P2"]["pass"] is True
    assert sc["P3"]["per"]["emoticon/sgd_0.01"]["outcome"] == "cheating"
    assert sc["P4"]["pass"] is True
    assert sc["P5"]["reading"] == {"emoticon/sgd_0.01": "many small ones act together",
                                   "emoticon/adam_3e-05": "a few larger ones", "r1_t534_v369/sgd_0.03": "a few larger ones"}
    out["score"] = sc
    plot = _load("gamma_shift_map_plot", EXP / "plot.py")
    path = tmp_path / "r.json"
    path.write_text(json.dumps(out))
    plot.OUT, plot.FIG = path, tmp_path / "figures"
    plot.main()
    # halves.png: amendment 1, drawn from the committed plot.AMEND1 (results/gamma_shift_map_amend1.json)
    assert {p.name for p in (tmp_path / "figures").iterdir()} == {"pile.png", "topk.png", "split.png", "trigger.png", "adversarial.png",
                                                                  "halves.png"}


def test_halves_score_reads_the_rules():
    """Amendment 1 (halves.py): score() on a synthetic results dict, partial and complete, against E30's committed JSON."""
    h = _load("gamma_shift_map_halves", EXP / "halves.py")
    lab = h.CFG.label
    out = {"C1": {"w0_check": {"pgd_floor": 0.035}}, "R_everywhere": {"zero": 0.0, "up": 0.40, "down": 0.25, "shift": 0.75},
           "opened": {"down": 0.9, "shuffled_down": 0.1}, "pile": {f"{lab}/zero": _reading(0.36, 0.83, 1100.0)}}
    sc = h.score(out)
    assert sc["P2"]["pass"] is True and sc["P4"]["reading"] == "premise holds" and "P1" not in sc
    out["pile"].update({f"{lab}/up": _reading(0.37, 0.80, 3400.0), f"{lab}/down": _reading(0.38, 7.0, 1000.0),
                        f"{lab}/shift": _reading(0.40, 6.7, 3420.0)})
    sc = h.score(out)
    assert sc["P1"]["outcome"] == "lowered half" and sc["P3"]["reading"] == "cheating"
    assert sc["route"] == h.PREDICTIONS["routes"]["lowered half, up cheating"]
    out["pile"][f"{lab}/shift"] = _reading(0.40, 0.9, 3420.0)  # the positive control passes: no power
    assert h.score(out)["P1"]["outcome"] == "no power"
