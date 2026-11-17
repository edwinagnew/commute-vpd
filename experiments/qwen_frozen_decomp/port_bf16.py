"""E175 follow-up of port_check stage 2: the same comparison with the library, but with the target in bf16 as the
library runs it (Γ bf16, neuron components bf16). If the active-set gap (Jaccard 0.979-0.981, 0.3-0.4% of g values off
by > 0.1 with the target in fp32) closes, it comes from the target's precision feeding Γ's taps and readouts, not from
the port. Writes results/port_check_bf16.json.

  sbatch --comment='E175 stage: port check bf16 target' cluster/qwen_frozen_decomp_port_check.sbatch experiments/qwen_frozen_decomp/port_bf16.py
"""

from __future__ import annotations

import json

import torch

from commute import qwen_vpd as Q
from commute.jobs import hard_exit, load_experiment

P = load_experiment("qwen_frozen_decomp", "port_check")


def main() -> None:
    Q.fp32_matmuls()
    port = P.rows_of(P.PORT_ROWS)
    with torch.no_grad():
        model = Q.load_target(P.DEV, torch.bfloat16)
        comps = Q.load_components(P.FROZEN, P.DEV, torch.bfloat16)
        gamma = Q.load_gamma(P.FROZEN, P.DEV, torch.bfloat16)
        res = P.jax_compare(model, gamma, comps, port)
    out = P.HERE / "results" / "port_check_bf16.json"
    out.write_text(json.dumps(res | {"target": "bf16", "components": "bf16"}, indent=1))
    P.log(f"STAGE jax_bf16 DONE {json.dumps(res)}")
    hard_exit(0)


if __name__ == "__main__":
    main()
