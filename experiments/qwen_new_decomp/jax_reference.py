"""Reference outputs of the authors' public library (param-decomp 7d2e5aee8) for Goodfire's second Qwen3-1.7B
decomposition (p-9707d6ff, step 100000) on the 8 rows of experiments/qwen_port/results/tokens.json, which the torch
port must reproduce (control (b) of reports/plans/2026-11-07_plan_e131-new-decomposition.md). The pattern of
experiments/qwen_port/jax_reference.py (p-e9fcffd5, param-decomp 177d0cce8), which stays as it ran.

The run is opened as `open_run(..., ONE_DEVICE_OWNER_LAYOUT)` does it (load_run.py:434-498, unchanged between the two
commits): target and decomposition in bf16, target attention XLA as the bundle's deliverable.yaml sets, Γ's own
attention XLA as its launch_config sets. One change, altering no value: the prepare step runs as two jits that donate
the restored arrays, so restored and prepared copies (~25 GiB each) are never on the device together (the one-jit
prepare needed 53.7 GB for p-e9fcffd5, past an L40, job 38761). Each row is one document padded to 512 with document
id -1, and run as a batch of two copies (the bundle's compute_ci.py: the sequence layout loses its batch sharding on
a batch of one); only the first copy's positions below the row's length are read or saved.

Stages, cheapest first, each written as soon as it finishes (big arrays in results/large/, safetensors):
  1. target: next-token CE per row, the control (text rows ~2-3 nats; qwen_port read the same rows).
  2. taps: the 29 raw residual boundaries resid.0-resid.28 (bf16) -> jax_taps_row{r}.
  3. gamma: g (lower importance, bf16) for every site on all 8 rows -> jax_gamma_row{r}; per row the mean count of
     subcomponents with g > 0.1 and with g > 0 per position, summed over sites.
  4. masked: per-position KL(clean || masked) with m = g and no delta (the CI-masked KL), per row -> jax_masked_row{r}.
Every scalar is printed and kept in results/jax_reference.json.

Usage: sbatch cluster/qwen_new_decomp_jax_reference.sbatch (one L40, ~15 min).
"""

import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.model import CaptureRequest, MaterializedMasking
from param_decomp.core.placement import batch_axes
from param_decomp.core.sharding import mesh_for_shape
from param_decomp.experiments.lm import load_run
from param_decomp.experiments.lm.resolved import require_unrouted_ci_fn_arch
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.sequence import SequenceLayout
from safetensors.numpy import save_file

HERE = Path(__file__).resolve().parent
TOKENS = HERE.parent / "qwen_port/results/tokens.json"
SUMMARY, LARGE = HERE / "results/jax_reference.json", HERE / "results/large"
DATA_ROOT = Path.home() / "commute-data/artifacts/p-9707d6ff-share/data/p-9707d6ff-share/param-decomp-run"
RUN_DIR = DATA_ROOT / "runs/p-9707d6ff"
STEP, SEQ_LEN, PAD_ID, N_BOUNDARIES, ACTIVE = 100000, 512, 151643, 29, 0.1
TAPS = frozenset(f"resid.{b}" for b in range(N_BOUNDARIES))


def say(stage: str, **values) -> None:
    for k, v in values.items():
        print(f"[{stage}] {k} = {v}", flush=True)


@eqx.filter_jit(donate="all-except-first")
def prepare_weights(placed, components):
    return placed.prepare_compute_weights(components)


@eqx.filter_jit(donate="all")
def prepare_ci_fn(ci_fn):
    return ci_fn.prepare()


def tree_gib(tree) -> float:
    return sum(x.nbytes for x in jax.tree.leaves(tree) if isinstance(x, jax.Array) and not x.is_deleted()) / 2**30


def device_gib() -> dict:
    stats = jax.devices()[0].memory_stats() or {}
    return {k: round(stats[k] / 2**30, 2) for k in ("bytes_in_use", "peak_bytes_in_use") if k in stats}


def open_decomposition() -> load_run.LoadedRun:
    """`open_run(..., ONE_DEVICE_OWNER_LAYOUT)` step by step, with the prepare split into two donating jits."""
    layout = load_run.ONE_DEVICE_OWNER_LAYOUT
    mesh = mesh_for_shape(layout.mesh)
    deliverable = load_run.load_deliverable(RUN_DIR, DATA_ROOT)
    assert deliverable.target.attention_implementation == "xla", deliverable.target
    placed = load_run.build_target(deliverable.target, mesh, DATA_ROOT, layout.sharding, sequence_sharding="replicate")
    with jax.set_mesh(mesh):
        restored, step = load_run._restore_decomposition(
            require_unrouted_ci_fn_arch(deliverable.ci_fn), placed, mesh, RUN_DIR, STEP
        )
        say("load", restored_gib=round(tree_gib(restored), 2), **device_gib())
        weights = jax.block_until_ready(prepare_weights(placed, restored.components))
        say("load", prepared_components_gib=round(tree_gib(weights), 2), **device_gib())
        ci_fn = jax.block_until_ready(prepare_ci_fn(restored.ci_fn))
        say("load", prepared_ci_fn_gib=round(tree_gib(ci_fn), 2), **device_gib())
    del restored
    return load_run.LoadedRun(
        run_id=RUN_DIR.name, step=step, placed=placed, deliverable=deliverable,
        prepared_weights=weights, prepared_ci_fn=ci_fn, mesh=mesh,
    )


def batch_of(run, row: list[int]) -> LMBatchWithDocuments:
    """The row as one document from position 0, right-padded with document id -1, twice (a batch of two)."""
    ids = np.full((2, SEQ_LEN), PAD_ID, np.int32)
    docs = np.full((2, SEQ_LEN), -1, np.int32)
    ids[:, : len(row)], docs[:, : len(row)] = row, 0
    sharding = NamedSharding(run.mesh, P(batch_axes(run.mesh), None))
    ids_dev, docs_dev = (jax.device_put(jnp.asarray(a), sharding) for a in (ids, docs))
    return LMBatchWithDocuments(LMBatch(ids_dev), SequenceLayout(docs_dev))


def logits_of(output) -> jax.Array:
    """fp32-accumulated logits from the streamed output edge (`targets.losses._chunk_logits`)."""
    return jnp.dot(output.activations, output.head.T, preferred_element_type=jnp.float32)


@eqx.filter_jit
def clean(placed, batch):
    r = placed.clean_forward(batch, TAPS)
    return logits_of(r.output), r.captures.read(CaptureRequest(TAPS, jax.Array))


@eqx.filter_jit
def importance(placed, weights, ci_fn, batch):
    r = placed.clean_forward(batch, ci_fn.capture_keys)
    ci = evaluate_ci_from_captures(ci_fn, r.captures, r.conditioning, weights, sequence=r.sequence, remat=False)
    return ci.lower


@eqx.filter_jit
def masked(placed, weights, batch, masks):
    masking = placed.model.prepare_masking(MaterializedMasking(component_masks=masks, weight_delta_masks=None))
    r = placed.masked_forward(weights, batch, masking=masking, routes=None, remat=False)
    return logits_of(r.output)


@jax.jit
def kl_per_position(clean_logits, masked_logits):
    log_p = jax.nn.log_softmax(clean_logits, axis=-1)
    log_q = jax.nn.log_softmax(masked_logits, axis=-1)
    return jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1)


def next_token_ce(logits: jax.Array, row: list[int]) -> float:
    log_p = jax.nn.log_softmax(logits[0, : len(row) - 1], axis=-1)
    return float(-jnp.mean(jnp.take_along_axis(log_p, jnp.asarray(row[1:])[:, None], axis=-1)))


def stage_target(run, rows) -> dict:
    res = {"next_token_ce": []}
    for r, row in enumerate(rows):
        logits, _ = clean(run.placed, batch_of(run, row))
        res["next_token_ce"].append(next_token_ce(logits, row))
        say("target", row=r, length=len(row), next_token_ce=res["next_token_ce"][-1])
    return res


def stage_taps(run, rows) -> dict:
    res = {"rms_resid28": []}
    for r, row in enumerate(rows):
        _, taps = clean(run.placed, batch_of(run, row))
        n = len(row)
        save_file({k: np.asarray(v[0, :n]) for k, v in taps.items()}, str(LARGE / f"jax_taps_row{r}.safetensors"))
        res["rms_resid28"].append(float(jnp.sqrt(jnp.mean(taps["resid.28"][0, :n].astype(jnp.float32) ** 2))))
        say("taps", row=r, rms_resid28=res["rms_resid28"][-1])
    return res


def stage_gamma(run, rows) -> dict:
    res = {"active_0.1": [], "nonzero": []}
    for r, row in enumerate(rows):
        g = importance(run.placed, run.prepared_weights, run.prepared_ci_fn, batch_of(run, row))
        n = len(row)
        save_file({f"g.{s}": np.asarray(v[0, :n]) for s, v in g.items()}, str(LARGE / f"jax_gamma_row{r}.safetensors"))
        res["active_0.1"].append(sum(float((v[0, :n] > ACTIVE).sum(-1).mean()) for v in g.values()))
        res["nonzero"].append(sum(float((v[0, :n] > 0).sum(-1).mean()) for v in g.values()))
        say("gamma", row=r, active=res["active_0.1"][-1], nonzero=res["nonzero"][-1])
        del g
    return res


def stage_masked(run, rows) -> dict:
    res = {"ci_kl": []}
    for r, row in enumerate(rows):
        batch, n = batch_of(run, row), len(row)
        clean_logits, _ = clean(run.placed, batch)
        g = importance(run.placed, run.prepared_weights, run.prepared_ci_fn, batch)
        kl = kl_per_position(clean_logits, masked(run.placed, run.prepared_weights, batch, g))[0, :n]
        save_file({"ci.kl": np.asarray(kl)}, str(LARGE / f"jax_masked_row{r}.safetensors"))
        res["ci_kl"].append(float(kl.mean()))
        say("masked", row=r, ci_kl=res["ci_kl"][-1])
        del g
    return res


def main() -> None:
    tokens = json.loads(TOKENS.read_text())
    rows = tokens["rows"]
    assert tokens["seq_len"] == SEQ_LEN and all(len(r) <= SEQ_LEN for r in rows)
    LARGE.mkdir(parents=True, exist_ok=True)
    summary = {"device": str(jax.devices()[0]), "lengths": [len(r) for r in rows], "kinds": tokens["kinds"],
               "library": "param-decomp 7d2e5aee8", "run": "p-9707d6ff step 100000",
               "target_attention": "xla (deliverable.yaml)", "active_threshold": ACTIVE}
    t0 = time.time()
    run = open_decomposition()
    say("load", seconds=round(time.time() - t0), step=run.step)
    stages = [("target", stage_target), ("taps", stage_taps), ("gamma", stage_gamma), ("masked", stage_masked)]
    for name, stage in stages:
        t = time.time()
        with jax.set_mesh(run.mesh):  # the owner layout's capture buffers name mesh axes
            summary[name] = stage(run, rows)
        summary[name]["seconds"] = round(time.time() - t, 1)
        SUMMARY.write_text(json.dumps(summary, indent=1))
        say(name, seconds=summary[name]["seconds"], **device_gib())
    print("FINISHED", SUMMARY, LARGE, file=sys.stdout, flush=True)


if __name__ == "__main__":
    main()
