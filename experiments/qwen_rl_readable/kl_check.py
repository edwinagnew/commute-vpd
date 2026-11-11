"""P3.4 run 1 (E163), stage kl-check: E163's output change read on E69's own saved R updates (seed 0, 1e-6w10, steps 10,
50, 100), to tell a different trajectory from a wrong measure. E163's control read 0.0173, 0.0254, 0.0479 at these steps;
E93 read 0.0643 on E69's step-100 update. The read is E163's `output_change` (fp32 model, 4 rows a batch); at step 100 also
one row at a time, E93's way. Writes results/kl_check.json and prints every number.
  0.064 back on E69's step 100 = the new run is a different update; about 0.048 = the measure moved since E93.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from commute import qwen_d3
from commute.qwen_sft import linears, set_weights
from commute.qwen_vpd import load_target
from commute.vpd import kl_per_position

HERE = Path(__file__).resolve().parent
DW = HERE.parent / "qwen_rl" / "results" / "large" / "lr1e-6w10" / "dW" / "R"
OUT = HERE / "results" / "kl_check.json"


@torch.no_grad()
def kl(model, W0: dict, W: dict, rows: torch.Tensor, batch: int) -> float:
    """KL(W ‖ W0) per position, mean over every position of `rows` (E163's output_change)."""
    total = 0.0
    for chunk in rows.split(batch):
        t = chunk.to("cuda")
        with set_weights(model, W):
            after = model(t, use_cache=False).logits
        before = model(t, use_cache=False).logits  # the model holds W0
        total += float(kl_per_position(after, before).double().sum())
        del after, before
    return total / rows.numel()


def main() -> None:
    model = load_target("cuda", torch.float32)
    W0 = {s: lin.weight.data for s, lin in linears(model).items()}
    rows = qwen_d3.maths_rows().tokens
    out: dict = {"rows": list(rows.shape), "reads": {}}
    for step in (10, 50, 100):
        dW = load_file(str(DW / f"step_{step}.safetensors"))
        assert set(dW) == set(W0), sorted(set(dW) ^ set(W0))[:5]
        norm = sum(float(x.double().pow(2).sum()) for x in dW.values()) ** 0.5
        W = {s: W0[s] + dW[s].to("cuda", torch.float32) for s in W0}
        r = {"dw_norm": norm, "dtype": str(next(iter(dW.values())).dtype), "kl_batched": kl(model, W0, W, rows, 4)}
        if step == 100:
            r["kl_one_row"] = kl(model, W0, W, rows, 1)
        out["reads"][str(step)] = r
        print(f"step {step}: {json.dumps(r)}", flush=True)
        OUT.write_text(json.dumps(out, indent=1))
        del W, dW
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
