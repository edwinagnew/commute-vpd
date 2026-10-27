"""Convert the Qwen3-1.7B decomposition (Goodfire run p-e9fcffd5, step 100000) from its Orbax checkpoint to float32
safetensors, once, so the torch port can read it (plan reports/plans/2026-10-27_plan_qwen-port-and-baseline.md,
item 2; checkpoint tree in reports/analyses/2026-10-27_analysis_qwen-loader.md section 1).

Each of the 1413 arrays is read with tensorstore straight from the OCDBT/zarr store and written unchanged: float32
in, float32 out, keys = the Orbax tree path joined with "." (e.g. `components.stacks.q.0`,
`ci_fn.backbone.backbone.blocks.w1`). Tensors are packed into files of at most ~5 GB in sorted key order; a tensor
larger than that gets a file of its own. `index.json` maps each key to its file, shape and dtype and each file to
its sha256; `launch_config.yaml` is copied beside it. The read-back then checks every key's shape and dtype against
the zarr metadata and compares the first and last slice along axis 0 of every array value for value.

Usage (repo root; JAX venv, CPU job):
  ~/commute-agents/fleet/cpu-run -c 8 --mem 96G -t 120 \
    ~/commute-data/param-decomp-177d0cce8/param_decomp/.venv/bin/python experiments/qwen_port/convert.py
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorstore as ts
from safetensors import safe_open
from safetensors.numpy import save_file

RUN = Path.home() / "commute-data/artifacts/qwen3-1.7b-p-e9fcffd5/runs/p-e9fcffd5"
STORE = RUN / "ckpts/100000/decomposition"
OUT = Path.home() / "commute-data/artifacts/qwen3-1.7b-p-e9fcffd5-torch"
FILE_BYTES = 5 * 2**30
CTX = ts.Context({"cache_pool": {"total_bytes_limit": 0}, "data_copy_concurrency": {"limit": 8}})


def array_keys() -> list[str]:
    kv = ts.KvStore.open({"driver": "ocdbt", "base": f"file://{STORE.resolve()}"}, context=CTX).result()
    keys = [k.decode() for k in kv.list().result()]
    return sorted(k[: -len("/.zarray")] for k in keys if k.endswith("/.zarray"))


def open_array(key: str) -> ts.TensorStore:
    spec = {"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": f"file://{STORE.resolve()}", "path": key + "/"}}
    return ts.open(spec, open=True, read=True, context=CTX).result()


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
        values = {k: arrays[k].read().result() for k in keys}
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
    print(f"read-back ok: {len(arrays)} keys, shapes/dtypes match zarr metadata, first/last slices equal, "
          f"{len(index['files'])} file hashes re-checked", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = array_keys()
    assert len(keys) == 1413, len(keys)
    arrays = {k: open_array(k) for k in keys}
    assert all(a.dtype.numpy_dtype == np.float32 for a in arrays.values())
    total = sum(int(np.prod(a.shape)) for a in arrays.values())
    print(f"{len(keys)} arrays, {total:,} float32 values, {total * 4 / 2**30:.1f} GiB", flush=True)
    index = write(arrays)
    index["n_values"] = total
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    shutil.copy(RUN / "launch_config.yaml", OUT / "launch_config.yaml")
    read_back(arrays, index)
    print("FINISHED", OUT, file=sys.stdout, flush=True)


if __name__ == "__main__":
    main()
