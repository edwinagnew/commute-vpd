"""E175 (cell P3.9): convert the frozen-neuron Qwen3-1.7B decomposition's CI function (run p-015e0710, step 100000) from
its Orbax checkpoint to float32 safetensors for `commute.qwen_vpd`, once. The subcomponents are not written: the
loader builds them from the HF weights (bundle README table: gate/up V = Wᵀ, U = I; down V = I, U = Wᵀ). This script
checks that table against the checkpoint's own (frozen) factors, every layer, value for value.

E131's converter (experiments/qwen_new_decomp/convert.py, which stays as it ran) is loaded by path and its readers,
pad cut, packing, writer and read-back are reused with this bundle's paths. What differs:
  - 613 arrays: Γ (no attention sink, no final norm: launch_config sets neither, library defaults "none") and the
    three MLP component stacks; attention is not decomposed (84 sites, kinds gate, up, down).
  - Only Γ's 607 arrays are written. index.json records `kinds` (the head order, read from deliverable.yaml's site
    order) and `components: "neuron"`, which `qwen_vpd.load_components` reads.
  - The bundle has no decomposition.safetensors; the factors are compared with the HF weights instead (`check_vu`).
    Any difference stops the job (exit 1) after everything is written.

Usage (repo root; the 7d2e5aee8 JAX venv, CPU job): sbatch --comment='E175 stage: convert' cluster/qwen_frozen_decomp_convert.sbatch
"""

import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("e131_convert", HERE.parent / "qwen_new_decomp" / "convert.py")
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

RUN_ID = "p-015e0710"
BUNDLE = Path.home() / f"commute-data/artifacts/{RUN_ID}-share/data/{RUN_ID}-share"
RUN = BUNDLE / f"param-decomp-run/runs/{RUN_ID}"
OUT = Path.home() / f"commute-data/artifacts/qwen3-1.7b-{RUN_ID}-torch"
HF = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
KINDS = ("gate", "up", "down")
N_ARRAYS, N_GAMMA, D_MODEL, D_MLP = 613, 607, 2048, 6144
base.BUNDLE, base.RUN, base.STORE, base.OUT = BUNDLE, RUN, RUN / "ckpts/100000/decomposition", OUT


def ci_fn_arch() -> dict[str, str]:
    """Γ's architecture from launch_config: p-e9fcffd5's apart from the site set (no sink key, no final_norm key)."""
    cfg = yaml.safe_load((BUNDLE / "launch_config.yaml").read_text())["decomposition"]
    ci, att = cfg["ci"], cfg["ci"]["attention"]
    assert cfg["sites"]["cs"] == {k: D_MLP for k in KINDS} and cfg["sites"]["kind"] == "glu_transformer", cfg["sites"]
    assert ci["type"] == "global_transformer" and ci["d_model"] == 3840 and ci["n_blocks"] == base.N_BLOCKS, ci
    assert ci["input_tap"]["offsets"] == list(range(base.N_LAYERS + 1)), ci["input_tap"]
    assert ci["ffn"] == {"kind": "gelu", "hidden": 15360}, ci["ffn"]
    assert (att["kind"], att["mask"], att["n_heads"]) == ("mha", "causal", 32), att
    assert att["qk_norm"] is True and att["logit_softcap"] == 30.0, att
    assert "sink" not in att and "final_norm" not in ci, (att, ci)
    return {"sink": "none", "final_norm": "none"}


def head_kinds() -> list[str]:
    """Γ's head stacks follow the order in which each kind first appears among the run's sites (core/components.py
    vu_groups), and sites run layer-ascending then KIND_ORDER (targets/transformer.py:214-221, glu_site_specs), so
    with cs = {gate, up, down} the heads are gate, up, down. deliverable.yaml lists cs, not site names."""
    cs = yaml.safe_load((RUN / "deliverable.yaml").read_text())["decomposition"]["sites"]["cs"]
    assert set(cs) == set(KINDS), cs
    return list(KINDS)


def hf_tensors() -> dict[str, tuple[Path, dict]]:
    """Every tensor of the HF snapshot: name -> (file, header entry)."""
    out = {}
    for path in sorted(HF.glob("*.safetensors")):
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        out |= {k: (path, v | {"start": 8 + n}) for k, v in header.items() if k != "__metadata__"}
    return out


def hf_matrix(tensors: dict, key: str) -> np.ndarray:
    """A bf16 HF matrix as float32, exactly (bf16 is the top half of float32)."""
    path, entry = tensors[key]
    assert entry["dtype"] == "BF16", (key, entry["dtype"])
    a, b = entry["data_offsets"]
    with path.open("rb") as f:
        f.seek(entry["start"] + a)
        raw = np.frombuffer(f.read(b - a), dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(entry["shape"])


def check_vu(stored: dict) -> dict:
    """The checkpoint's V and U against the README table, built from the HF weights, every layer."""
    tensors, eye = hf_tensors(), np.eye(D_MLP, dtype=np.float32)
    unequal, max_abs = [], 0.0
    for kind in KINDS:
        V, U = stored[f"components.stacks.{kind}.0"], stored[f"components.stacks.{kind}.1"]
        for layer in range(base.N_LAYERS):
            W = hf_matrix(tensors, f"model.layers.{layer}.mlp.{kind}_proj.weight")
            want = (W.T, eye) if kind in ("gate", "up") else (eye, W.T)
            for name, arr, w in (("V", V, want[0]), ("U", U, want[1])):
                got = arr[layer : layer + 1].read().result()[0]
                assert got.shape == w.shape, (kind, layer, name, got.shape, w.shape)
                if not np.array_equal(got, w):
                    unequal.append(f"{kind}.{layer}.{name}")
                    max_abs = max(max_abs, float(np.abs(got - w).max()))
        print(f"V, U of {kind}: checked {base.N_LAYERS} layers, {len(unequal)} unequal so far", flush=True)
    res = {"n_compared": 2 * base.N_LAYERS * len(KINDS), "n_unequal": len(unequal), "unequal": unequal[:20],
           "max_abs_diff": max_abs, "against": "HF Qwen/Qwen3-1.7B@70d244cc, bundle README table"}
    print(f"V, U against the HF weights: {res}", flush=True)
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arch, kinds = ci_fn_arch(), head_kinds()
    keys = base.array_keys()
    assert len(keys) == N_ARRAYS, len(keys)
    stored = {k: base.open_array(k) for k in keys}
    assert all(a.dtype.numpy_dtype == np.float32 for a in stored.values())
    gamma = {k: a for k, a in stored.items() if not k.startswith("components.")}
    assert len(gamma) == N_GAMMA, len(gamma)
    arrays, pads = base.cut(gamma)
    total = sum(int(np.prod(a.shape)) for a in arrays.values())
    print(f"{len(arrays)} Γ arrays, {total:,} float32 values kept, {total * 4 / 2**30:.1f} GiB; heads {kinds}", flush=True)
    index = base.write(arrays)
    index |= {"n_values": total, "ci_fn_arch": arch, "kinds": kinds, "components": "neuron", "pads": pads,
              "pads_cut_by": "convert (checked zero)"}
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    shutil.copy(BUNDLE / "launch_config.yaml", OUT / "launch_config.yaml")
    base.read_back(arrays, index)
    index["neuron_vu_check"] = check_vu(stored)
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    if index["neuron_vu_check"]["n_unequal"]:
        print("V/U differ from the README table: stopping (exit 1)", flush=True)
        sys.exit(1)
    print("FINISHED", OUT, flush=True)


if __name__ == "__main__":
    main()
