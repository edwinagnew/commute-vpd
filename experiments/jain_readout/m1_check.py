"""E59's P3 rule on the original model's own ranks (review reports/reviews/2026-11-02_review_e59-analysis.md, M1): P3
compares a per-window lower-median rank with values E54 picked by mean probability over 500 windows, and that statistic
was never measured on the model itself. A diagnostic of the apparatus: no training, no predictions, no state but W0.

Two arms through `readout.key_medians`, `readout.possession` and `readout.p3` with E59's filed thresholds, on E54's 97
held-out windows, elicit and cipher, at the answer position:
  model  W0's own logits
  lin    the linearised head of E59's stage 0 (`readout.losses`), at the scale job 48412 recorded in run.json

Expected:
  model  P3 passes or comes close (elicit in band and cipher above, each against 24 keys), since E54 built the values
         from these ranks. If it does not, P3 was mis-specified, which is what this checks.
  lin    the model's ranks, window for window. The final norm's per-token scale is a positive scalar and lm_head has no
         bias, so the two logit vectors at a token are positive multiples and the scale cannot change a rank; only fp32
         rounding can (`lin_differs`, expected ~0). This arm is a control on the code, not a second reading.

Beside them, for reading: the last-block readout's counts from run.json (job 48412). Writes results/m1_check.json.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import keydep as KD
from commute import readout as RO
from commute import steering as ST

HERE = Path(__file__).resolve().parent
RESULTS, E54 = HERE / "results", HERE.parent / "keydependent_elicit" / "results"
TASKS, log = ("elicit", "cipher"), KD.log


def ranks_at(logits: Tensor, tgt: Tensor) -> list[int]:
    """(N, V) logits -> rank of each window's target, 1 + the tokens scored strictly higher (`readout.target_ranks`)."""
    return (1 + (logits > logits.gather(1, tgt[:, None])).sum(-1)).tolist()


def arm(ranks: dict[str, list[int]], tasks: dict[str, ST.Task], th: dict[str, Any]) -> dict[str, Any]:
    """Per task the medians and the possession count at the output; then P3 on the two tasks' medians."""
    out: dict[str, Any] = {}
    for n, t in tasks.items():
        med = RO.key_medians(ranks[n], t.toks[:, t.half].tolist(), list(t.keys))
        poss = RO.possession(med, th["possess_rank"], th["possess_keys"], len(t.keys))
        out[n] = {"ranks": ranks[n], "medians": med, "possession": poss}
    out["P3"] = RO.p3(out["elicit"]["medians"], out["cipher"]["medians"], tuple(th["p3_band"]), th["possess_keys"], th["p3_cipher_above"])
    return out


def say(name: str, a: dict[str, Any]) -> None:
    for n in TASKS:
        log(f"[{name} {n}] possession {json.dumps(a[n]['possession'])} medians {list(a[n]['medians'].values())}")
        log(f"[{name} {n}] window ranks {a[n]['ranks']}")
    log(f"[{name} P3] {json.dumps(a['P3'])}")


@torch.no_grad()
def main() -> None:
    P, e59 = (json.loads((RESULTS / f).read_text()) for f in ("predictions.json", "run.json"))
    s1 = json.loads((E54 / "stage1.json").read_text())
    scale = e59["stage0"]["readouts"]["scale"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    net = ST.Net("cuda")
    w, W = net.net.ln_f.weight, net.net.lm_head.weight
    tasks = {n: ST.load_task(net, E54 / "large" / f"windows_{n}.pt", n, n) for n in TASKS}
    model, lin, rate_rel = {}, {}, {}
    for n, t in tasks.items():
        rate_rel[n] = KD.geo_rate(t.lp0) / s1[n]["rate"] - 1
        log(f"[{n}] {len(t.toks)} windows, {len(t.keys)} keys, W0 rate vs E54 stage1 rel {rate_rel[n]:.1e}")
        assert abs(rate_rel[n]) <= P["stage0"]["thresholds"]["rate_rel_tol"], "not E54's model and windows"
        logits, h = RO.resid(net, t.toks)
        tgt = t.tgt(t.toks).to(logits.device)
        model[n] = ranks_at(logits[:, t.half], tgt)
        lin[n] = ranks_at((h[RO.LAST][:, t.half] * scale * w) @ W.T, tgt)
    arms = {a: arm(r, tasks, P["thresholds"]) for a, r in (("model", model), ("lin", lin))}
    last = lambda n: e59["possession"][n]["all"]["possession"][RO.LAST]
    out = {"git_head": subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
           "scale": scale, "rate_rel": rate_rel, **arms,
           "lin_differs": {n: sum(a != b for a, b in zip(model[n], lin[n], strict=True)) for n in TASKS},
           "readout_e59": {"P3": e59["P3"], "possession": {n: last(n) for n in TASKS}}}  # fmt: skip
    for name, a in arms.items():
        say(name, a)
    log(f"[lin vs model] windows whose rank differs {out['lin_differs']} (scale {scale})")
    log(f"[readout, job 48412, for reading] {json.dumps(out['readout_e59'])}")
    (RESULTS / "m1_check.json").write_text(json.dumps(out, indent=1))
    log(f"[wrote] {RESULTS / 'm1_check.json'}")


if __name__ == "__main__":
    main()
