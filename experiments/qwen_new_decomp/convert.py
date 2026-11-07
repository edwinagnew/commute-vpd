"""Convert Goodfire's second Qwen3-1.7B decomposition (run p-9707d6ff, step 100000) from its Orbax checkpoint to
float32 safetensors, once, for the torch port (plan reports/plans/2026-11-07_plan_e131-new-decomposition.md; the
pattern of experiments/qwen_port/convert.py, which converted p-e9fcffd5 and stays as it ran).

What differs from that conversion:
  - 1414 arrays (p-e9fcffd5: 1413): the learned attention sink `ci_fn.backbone.backbone.blocks.attention.sink_logits`
    is new; the block gate and norm scales are None in this run (GELU FFN, weightless norms) and hold no array.
  - The persist stacks are saved padded with trailing all-zero slots (README: 32 layer slots, 28-31 zero). The pads
    are cut here, once: components and Γ heads to their 28 layers, Γ blocks (and the sink) to their 32 blocks. Each
    cut slice is read and must be exactly zero, else the job stops. index.json records every cut as
    `pads[key] = [stored length, kept length]` (the loader then asserts 28/32-long stacks).
  - index.json carries `ci_fn_arch = {"sink", "final_norm"}` from the bundle's launch_config, which
    `commute.qwen_vpd.load_gamma` reads; the converter also asserts the rest of Γ's architecture is p-e9fcffd5's
    (width 3840, 32 blocks, 32 heads, q/k RMS norm, softcap 30, GELU 15360, taps 0-28).
  - After writing, V and U of every one of the 196 sites are compared value for value with the bundle's
    decomposition.safetensors (written by the authors from the same run); the counts of unequal sites and the
    largest absolute difference go in index.json, and any difference stops the job (exit 1) after everything is
    written, so the conversion is not lost.
Tensors are packed into files of at most ~5 GB in sorted key order; the read-back checks every key's shape and dtype
and the first and last kept slice of every array against the store, and re-hashes each file.

Memory: one file group (<= 5 GB) or one oversized tensor (blocks.w1 is 7.5 GB) plus its pad slice; ~20 GB peak.

Usage (repo root; the 7d2e5aee8 JAX venv, CPU job): sbatch cluster/qwen_new_decomp_convert.sbatch
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorstore as ts
import yaml
from safetensors import safe_open
from safetensors.numpy import save_file

BUNDLE = Path.home() / "commute-data/artifacts/p-9707d6ff-share/data/p-9707d6ff-share"
RUN = BUNDLE / "param-decomp-run/runs/p-9707d6ff"
STORE = RUN / "ckpts/100000/decomposition"
DEC = BUNDLE / "decomposition.safetensors"
OUT = Path.home() / "commute-data/artifacts/qwen3-1.7b-p-9707d6ff-torch"
FILE_BYTES = 5 * 2**30
N_KEYS, N_LAYERS, N_BLOCKS = 1414, 28, 32
KINDS = ("q", "k", "v", "o", "gate", "up", "down")
C = {"q": 2048, "k": 1024, "v": 1024, "o": 2048, "gate": 6144, "up": 6144, "down": 6144}
STACKED = {"components.stacks.": N_LAYERS, "ci_fn.backbone.backbone.heads.": N_LAYERS,
           "ci_fn.backbone.backbone.blocks.": N_BLOCKS}  # key prefix -> real leading length
CTX = ts.Context({"cache_pool": {"total_bytes_limit": 0}, "data_copy_concurrency": {"limit": 8}})


def array_keys() -> list[str]:
    kv = ts.KvStore.open({"driver": "ocdbt", "base": f"file://{STORE.resolve()}"}, context=CTX).result()
    keys = [k.decode() for k in kv.list().result()]
    return sorted(k[: -len("/.zarray")] for k in keys if k.endswith("/.zarray"))


def open_array(key: str) -> ts.TensorStore:
    spec = {"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": f"file://{STORE.resolve()}", "path": key + "/"}}
    return ts.open(spec, open=True, read=True, context=CTX).result()


def ci_fn_arch() -> dict[str, str]:
    """Γ's sink and final norm from the bundle's launch_config; every other setting must equal p-e9fcffd5's."""
    cfg = yaml.safe_load((BUNDLE / "launch_config.yaml").read_text())["decomposition"]
    ci, att = cfg["ci"], cfg["ci"]["attention"]
    assert cfg["sites"]["cs"] == C and cfg["sites"]["kind"] == "glu_transformer", cfg["sites"]
    assert ci["type"] == "global_transformer" and ci["d_model"] == 3840 and ci["n_blocks"] == N_BLOCKS, ci
    assert ci["input_tap"]["offsets"] == list(range(N_LAYERS + 1)), ci["input_tap"]
    assert ci["ffn"] == {"kind": "gelu", "hidden": 15360}, ci["ffn"]
    assert (att["kind"], att["mask"], att["n_heads"]) == ("mha", "causal", 32), att
    assert att["qk_norm"] == "rms" and att["logit_softcap"] == 30.0, att
    arch = {"sink": att["sink"], "final_norm": ci["final_norm"]}
    assert arch == {"sink": "learned", "final_norm": "rms"}, arch
    return arch


def real_length(key: str, shape: tuple[int, ...]) -> int | None:
    """The kept leading length of a stacked key (None: not stacked, kept whole)."""
    for prefix, n in STACKED.items():
        if key.startswith(prefix):
            assert shape[0] >= n, (key, shape)
            return n
    return None


def cut(arrays: dict[str, ts.TensorStore]) -> tuple[dict[str, ts.TensorStore], dict[str, list[int]]]:
    """Each stacked array sliced to its real length, after checking that its pad slots read exactly zero."""
    kept, pads = {}, {}
    for key, arr in arrays.items():
        n = real_length(key, tuple(arr.shape))
        if n is None:
            kept[key] = arr
            continue
        if arr.shape[0] > n:
            for i in range(n, arr.shape[0]):  # one slot at a time: host memory holds one slice
                pad = arr[i : i + 1].read().result()
                assert not np.any(pad), f"{key} pad slot {i} is not zero (max |x| {np.abs(pad).max()})"
        pads[key] = [int(arr.shape[0]), n]
        kept[key] = arr[:n]
    print(f"pads cut and checked zero: {sum(a != b for a, b in pads.values())} of {len(pads)} stacked keys padded; "
          f"stored lengths {sorted({a for a, _ in pads.values()})}", flush=True)
    return kept, pads


def pack(arrays: dict[str, ts.TensorStore]) -> list[list[str]]:
    """Consecutive keys into files of at most FILE_BYTES; an oversized tensor alone."""
    files, current, size = [], [], 0
    for key, arr in arrays.items():
        nbytes = int(np.prod(arr.shape)) * 4
        if current and size + nbytes > FILE_BYTES:
            files.append(current)
            current, size = [], 0
        current.append(key)
        size += nbytes
    return files + [current]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(64 * 2**20):
            h.update(chunk)
    return h.hexdigest()


def write(arrays: dict[str, ts.TensorStore]) -> dict:
    tensors, files = {}, {}
    groups = pack(arrays)
    for i, keys in enumerate(groups):
        name = f"decomposition-{i + 1:05d}-of-{len(groups):05d}.safetensors"
        values = {k: np.ascontiguousarray(arrays[k].read().result()) for k in keys}
        for k, v in values.items():
            assert v.dtype == np.float32, (k, v.dtype)
            tensors[k] = {"file": name, "shape": list(v.shape), "dtype": "float32"}
        save_file(values, str(OUT / name))
        del values
        files[name] = {"sha256": sha256(OUT / name), "bytes": (OUT / name).stat().st_size}
        print(f"wrote {name}: {len(keys)} tensors, {files[name]['bytes'] / 2**30:.2f} GiB", flush=True)
    return {"source": f"{STORE} (Orbax, zarr v2 <f4)", "tensors": tensors, "files": files}


def read_back(arrays: dict[str, ts.TensorStore], index: dict) -> None:
    assert set(index["tensors"]) == set(arrays), "key sets differ"
    handles = {name: safe_open(str(OUT / name), framework="numpy") for name in index["files"]}
    for key, arr in arrays.items():
        entry = index["tensors"][key]
        got = handles[entry["file"]].get_slice(key)
        assert list(got.get_shape()) == list(arr.shape) == entry["shape"], (key, got.get_shape(), arr.shape)
        assert got.get_dtype() == "F32" and arr.dtype.numpy_dtype == np.float32, (key, got.get_dtype(), arr.dtype)
        for i in (0, arr.shape[0] - 1):
            assert np.array_equal(got[i : i + 1], arr[i : i + 1].read().result()), (key, i)
    for name, meta in index["files"].items():
        assert sha256(OUT / name) == meta["sha256"], name
    print(f"read-back ok: {len(arrays)} keys, shapes/dtypes match the store, first/last slices equal, "
          f"{len(index['files'])} file hashes re-checked", flush=True)


def compare_bundle_vu(index: dict) -> dict:
    """V and U per site, ours (from the written files) against decomposition.safetensors, one layer at a time."""
    ours = {name: safe_open(str(OUT / name), framework="numpy") for name in index["files"]}
    dec = safe_open(str(DEC), framework="numpy")
    unequal, max_abs = [], 0.0
    for layer in range(N_LAYERS):
        for kind in KINDS:
            module = f"model.layers.{layer}.{'self_attn' if kind in ('q', 'k', 'v', 'o') else 'mlp'}.{kind}_proj"
            for part, suffix in ((0, "V"), (1, "U")):
                key = f"components.stacks.{kind}.{part}"
                mine = ours[index["tensors"][key]["file"]].get_slice(key)[layer : layer + 1][0]
                theirs = dec.get_tensor(f"{module}.{suffix}")
                assert mine.shape == theirs.shape, (module, suffix, mine.shape, theirs.shape)
                if not np.array_equal(mine, theirs):
                    unequal.append(f"{module}.{suffix}")
                    max_abs = max(max_abs, float(np.abs(mine - theirs).max()))
    res = {"n_compared": 2 * N_LAYERS * len(KINDS), "n_unequal": len(unequal), "unequal": unequal[:20],
           "max_abs_diff": max_abs}
    print(f"V, U against decomposition.safetensors: {res}", flush=True)
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arch = ci_fn_arch()
    keys = array_keys()
    assert len(keys) == N_KEYS, len(keys)
    stored = {k: open_array(k) for k in keys}
    assert all(a.dtype.numpy_dtype == np.float32 for a in stored.values())
    arrays, pads = cut(stored)
    total = sum(int(np.prod(a.shape)) for a in arrays.values())
    print(f"{len(keys)} arrays, {total:,} float32 values kept, {total * 4 / 2**30:.1f} GiB", flush=True)
    index = write(arrays)
    index |= {"n_values": total, "ci_fn_arch": arch, "pads": pads, "pads_cut_by": "convert (checked zero)"}
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    shutil.copy(BUNDLE / "launch_config.yaml", OUT / "launch_config.yaml")
    read_back(arrays, index)
    index["bundle_vu_check"] = compare_bundle_vu(index)
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    if index["bundle_vu_check"]["n_unequal"]:
        print("V/U differ from decomposition.safetensors: stopping (exit 1)", flush=True)
        sys.exit(1)
    print("FINISHED", OUT, flush=True)


if __name__ == "__main__":
    main()
