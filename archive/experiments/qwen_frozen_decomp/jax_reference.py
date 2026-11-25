"""E175 (cell P3.9): reference outputs of the authors' public library (param-decomp 7d2e5aee8, the bundle's commit)
for the frozen-neuron decomposition p-015e0710 on the 8 rows of experiments/qwen_port/results/tokens.json, which the
torch port (port_check.py) must reproduce. E131's script (experiments/qwen_new_decomp/jax_reference.py, which stays
as it ran) is loaded by path and run with this bundle's paths; its taps stage is skipped (port_check reads end to
end only). Stages: target (next-token CE), gamma (g per site saved to results/large/jax_gamma_row{r}, active count
per row at g > 0.1), masked (KL with g as masks, no delta, per row). Every scalar goes to results/jax_reference.json.

Usage: sbatch --comment='E175 stage: jax reference' cluster/qwen_frozen_decomp_jax_reference.sbatch (one L40)
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import jax

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("e131_jax", HERE.parent / "qwen_new_decomp" / "jax_reference.py")
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

RUN_ID = "p-015e0710"
base.DATA_ROOT = Path.home() / f"commute-data/artifacts/{RUN_ID}-share/data/{RUN_ID}-share/param-decomp-run"
base.RUN_DIR = base.DATA_ROOT / f"runs/{RUN_ID}"
base.SUMMARY, base.LARGE = HERE / "results/jax_reference.json", HERE / "results/large"


def main() -> None:
    tokens = json.loads(base.TOKENS.read_text())
    rows = tokens["rows"]
    assert tokens["seq_len"] == base.SEQ_LEN and all(len(r) <= base.SEQ_LEN for r in rows)
    base.LARGE.mkdir(parents=True, exist_ok=True)
    summary = {"device": str(jax.devices()[0]), "lengths": [len(r) for r in rows], "kinds": tokens["kinds"],
               "library": "param-decomp 7d2e5aee8", "run": f"{RUN_ID} step {base.STEP}",
               "target_attention": "xla (deliverable.yaml)", "active_threshold": base.ACTIVE}
    t0 = time.time()
    run = base.open_decomposition()
    base.say("load", seconds=round(time.time() - t0), step=run.step)
    for name, stage in (("target", base.stage_target), ("gamma", base.stage_gamma), ("masked", base.stage_masked)):
        t = time.time()
        with jax.set_mesh(run.mesh):
            summary[name] = stage(run, rows)
        summary[name]["seconds"] = round(time.time() - t, 1)
        base.SUMMARY.write_text(json.dumps(summary, indent=1))
        base.say(name, seconds=summary[name]["seconds"], **base.device_gib())
        print(f"STAGE {name} DONE", flush=True)
    print("FINISHED", base.SUMMARY, base.LARGE, file=sys.stdout, flush=True)


if __name__ == "__main__":
    main()
