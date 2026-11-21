"""E234 check (cell P1.40, no new run): why x10's task-window FVE at resid_post3 reads -15.9 (base 0.90; next worst
item drop 0.12). x10 multiplies 8 layer-1 MLP neurons' down-projection columns by 10 (E195's C).

Per residual and block-output site of layers 1-3, for M (base) and F (x10) on x10's own 75 task windows (positions 1..):
  sq      mean |x|^2                       var     mean |x - mean x|^2 (the FVE denominator)
  err     mean |x - xhat|^2                fve     1 - err / var, as `commute.sae.read` computes it (bf16 capture)
  fve32   the same from an fp32 capture    (rules out bf16 rounding)
  err_c   mean |e - mean e|^2              fve_c   1 - err_c / var: FVE once the error's constant part is removed
  top1    share of err on the 1% of positions with the largest error
  l0      latents above threshold per position
Expected if the reading is real (a new constant offset the base SAE cannot write): fve32 = fve, fve_c near the base's,
err far above the base's while var stays near it. Expected if a bug: fve32 or the M row differ from E234's items JSON
(M's fve at resid_post3 = 0.900, x10's = -15.87).

    ~/commute-agents/fleet/cpu-run is not enough (needs a GPU): run through /pod-runner on cluster debug.
    python experiments/sae_train/check_x10.py   -> results/check_x10.json, every number also printed
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from commute import sae as S
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
SITES = ("resid_post1", "attn2", "mlp2", "resid_post2", "attn3", "mlp3", "resid_post3")


@torch.no_grad()
def capture32(net: Any, toks: torch.Tensor, sites: tuple[str, ...]) -> torch.Tensor:
    got: dict[str, torch.Tensor] = {}
    fns = {s: (lambda s: lambda x: got.__setitem__(s, x[:, 1:].float()))(s) for s in sites}
    with S.hooked(net, fns):
        net(toks.cuda())
    return torch.stack([got[s] for s in sites], 2).flatten(0, 1)


@torch.no_grad()
def numbers(ae: S.SAE, x: torch.Tensor, x32: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {}
    for tag, v in (("", x.float()), ("32", x32)):
        f = ae.encode(v)
        e = (v - ae.decode(f)).double()
        var = float((v.double() - v.double().mean(0)).pow(2).sum(1).mean())
        per = e.pow(2).sum(1)
        out[f"fve{tag}"] = 1 - float(per.mean()) / var
        if tag:
            continue
        err_c = float((e - e.mean(0)).pow(2).sum(1).mean())
        top = per.topk(max(1, len(per) // 100)).values.sum()
        out |= {"sq": float(v.double().pow(2).sum(1).mean()), "var": var, "err": float(per.mean()), "err_c": err_c,
                "fve_c": 1 - err_c / var, "top1": float(top / per.sum()), "l0": float((f > 0).sum(1).float().mean())}  # fmt: skip
    return out


def main() -> None:
    run: Any = load_experiment("sae_train", "run")
    VPD: Any = load_experiment("kstar_vpd", "run")
    cfg, _ = run.load_config("scored")
    saes = run.load_saes(cfg, [f"{s}_s0" for s in SITES], run.RESULTS / "large" / "scored")
    F, M, toks, _at, _load = VPD.World().make("x10")
    toks = toks[: cfg.task_windows]
    rec: dict[str, Any] = {"task_shape": list(toks.shape)}
    for name, net in (("M", M.net), ("x10", F.net)):
        x, x32 = S.capture(net, toks, SITES), capture32(net, toks, SITES)
        rec[name] = {s: numbers(saes[f"{s}_s0"], x[:, j], x32[:, j]) for j, s in enumerate(SITES)}
        for s, v in rec[name].items():
            print(f"{name} {s} " + " ".join(f"{k} {w:.4g}" for k, w in v.items()), flush=True)
    (HERE / "results" / "check_x10.json").write_text(json.dumps(rec, indent=1))
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
