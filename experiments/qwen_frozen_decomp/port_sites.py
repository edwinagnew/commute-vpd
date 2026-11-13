"""E175 follow-up of port_check stage 2: where our g and the library's disagree, site by site, on the first 2 qwen_port
rows (target fp32, Γ bf16, as port_check). Per site: share of values off by > 0.1, mean of ours minus the library's,
and active counts (g > 0.1) of each; also each site's mean |readout| and mean |head output| from our side, to tell a
readout (component activation) difference from a backbone one. Writes results/port_sites.json.

  sbatch --comment='E175 stage: port sites' cluster/qwen_frozen_decomp_port_check.sbatch experiments/qwen_frozen_decomp/port_sites.py
"""

from __future__ import annotations

import json

import torch
from safetensors.torch import load_file

from commute import qwen_vpd as Q
from commute.jobs import hard_exit, load_experiment

P = load_experiment("qwen_frozen_decomp", "port_check")
N_ROWS = 2


def main() -> None:
    Q.fp32_matmuls()
    rows = P.rows_of(P.PORT_ROWS)[:N_ROWS]
    out: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        model = Q.load_target(P.DEV)
        comps = Q.load_components(P.FROZEN, P.DEV)
        gamma = Q.load_gamma(P.FROZEN, P.DEV, torch.bfloat16)
        for r, t in enumerate(rows):
            clean = Q.clean_forward(model, t, gamma, comps)
            x = Q.backbone(gamma, clean.taps)
            jg = load_file(str(P.LARGE / f"jax_gamma_row{r}.safetensors"), device=P.DEV)
            for s in comps.sites:
                layer, kind = Q.parse_site(s)
                head = (x @ gamma.heads_w[kind][layer] + gamma.heads_b[kind][layer]).float()
                a = (head + clean.readouts[s].float()).clamp(0, 1)[0]
                b = jg[f"g.{s}"].float()
                d = out.setdefault(s, {"off": 0.0, "diff": 0.0, "act_ours": 0.0, "act_jax": 0.0, "readout": 0.0, "head": 0.0})
                for k, v in (("off", ((a - b).abs() > P.OFF).float().mean()), ("diff", (a - b).mean()),
                             ("act_ours", (a > P.ACTIVE).sum(-1).float().mean()),
                             ("act_jax", (b > P.ACTIVE).sum(-1).float().mean()),
                             ("readout", clean.readouts[s].float().abs().mean()), ("head", head.abs().mean())):
                    d[k] += float(v) / N_ROWS
            del clean, x, jg
    worst = sorted(out, key=lambda s: -out[s]["off"])[:12]
    by_kind = {k: {m: sum(out[s][m] for s in out if Q.parse_site(s)[1] == k) for m in ("act_ours", "act_jax")}
               for k in ("gate", "up", "down")}
    res = {"rows": N_ROWS, "by_kind": by_kind, "worst": {s: out[s] for s in worst}, "sites": out}
    (P.HERE / "results" / "port_sites.json").write_text(json.dumps(res, indent=1))
    P.log(f"STAGE sites DONE by_kind {json.dumps(by_kind)} worst {json.dumps({s: out[s] for s in worst[:6]})}")
    hard_exit(0)


if __name__ == "__main__":
    main()
