"""Reference outputs of the authors' JAX code for the Qwen3-1.7B decomposition (p-e9fcffd5, step 100000) on fixed
tokens, which the torch port (src/commute/qwen_vpd.py, experiments/qwen_port/torch_check.py) must reproduce. Plan
reports/plans/2026-10-27_plan_qwen-port-and-baseline.md, item 3; the JAX entry points are described in
reports/analyses/2026-10-27_analysis_qwen-loader.md sections 2-3.

The run is opened as `open_run(..., layout=ONE_DEVICE_OWNER_LAYOUT)` does it: target and decomposition in bf16, as
the authors evaluate it. Two changes, neither altering a value: the target's attention runs on XLA instead of cuDNN
flash (same arithmetic; cuDNN flash is unproven on an L40), and the prepare step donates the restored arrays so it
fits one L40 (`open_decomposition`). Rows come from results/tokens.json (make_tokens.py); each row
is one document, padded to 512 with document id -1, and only positions below its length are read or saved.

Stages, cheapest first, each written as soon as it finishes (all big arrays in results/large/, safetensors):
  1. target: logits (fp32, from the bf16 final hidden state and head with fp32 accumulation) and the final hidden
     state per row -> target_row{r}.safetensors; next-token CE per row printed as the control (text rows ~2-3 nats).
  2. taps: the 29 raw residual boundaries resid.0-resid.28 (resid.28 before the final norm), bf16 -> taps_row{r}.
  3. gamma: g (lower importance) and p (preactivation) for every site on rows DETAIL_ROWS (bf16 -> gamma_row{r});
     per-site mean g, mean count of g > 0 per position, and mean p on every row (in the summary).
  4. masked: the masked forward per row under m = g without delta, m = 1 without delta, and m = g + (1 - g) a with
     d = a_delta for the fixed source array a (source.safetensors, numpy seed SOURCE_SEED); per-position KL from the
     clean logits and the final hidden state for every row (masked_row{r}), logits for DETAIL_ROWS.
  5. adversary: the config's PGDReconLoss (random source shared over rows and positions, 20 sign steps of 0.1,
     KL objective, auxiliary coefficient 0) on the four text rows as one batch, for each seed in PGD_SEEDS.
Every scalar is printed and kept in results/jax_reference.json.

Usage: sbatch cluster/qwen_jax_reference.sbatch (one L40, about 10-20 min).
"""

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from param_decomp.core.adversary import SiteSource
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.masking import (
    materialize_masking,
    source_masking,
    unmasked_no_delta_masking,
)
from param_decomp.core.model import CaptureRequest, MaterializedMasking
from param_decomp.core.recon_eval import FreshPGDAttack, fresh_pgd_read_outs
from param_decomp.core.sharding import mesh_for_shape
from param_decomp.experiments.lm import load_run
from param_decomp.experiments.lm.resolved import require_unrouted_ci_fn_arch
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.sequence import SequenceLayout
from safetensors.numpy import save_file

HERE = Path(__file__).parent
TOKENS, SUMMARY, LARGE = HERE / "results/tokens.json", HERE / "results/jax_reference.json", HERE / "results/large"
DATA_ROOT = Path.home() / "commute-data/artifacts/qwen3-1.7b-p-e9fcffd5"
RUN_DIR = DATA_ROOT / "runs/p-e9fcffd5"
STEP, SEQ_LEN, PAD_ID = 100000, 512, 151643
N_BOUNDARIES = 29
DETAIL_ROWS = (0, 4)  # one text row, one GSM8K row
SOURCE_SEED, PGD_SEEDS, PGD_STEPS, PGD_STEP_SIZE = 0, (0, 1, 2), 20, 0.1
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


def open_decomposition() -> load_run.LoadedRun:
    """`open_run(..., ONE_DEVICE_OWNER_LAYOUT)` step by step (load_run.py:434-498), with two changes. The target's
    attention runs on XLA. The prepare step (`_prepare_read_only_consumer`: components relayout, Γ cast to bf16 and
    gathered to its compute layout) runs as two jits that donate the restored arrays, so the restored and prepared
    copies (25 GiB each) are never on the device together: one jit for both needs 53.7 GB, past an L40 (job 38761)."""
    layout = load_run.ONE_DEVICE_OWNER_LAYOUT
    mesh = mesh_for_shape(layout.mesh)
    deliverable = load_run.load_deliverable(RUN_DIR, DATA_ROOT)
    deliverable = replace(deliverable, target=replace(deliverable.target, attention_implementation="xla"))
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


def tree_gib(tree) -> float:
    return sum(x.nbytes for x in jax.tree.leaves(tree) if isinstance(x, jax.Array) and not x.is_deleted()) / 2**30


def device_gib() -> dict:
    """In-use and peak memory of the device, where the backend reports it (GPU yes, CPU no)."""
    stats = jax.devices()[0].memory_stats() or {}
    return {k: round(stats[k] / 2**30, 2) for k in ("bytes_in_use", "peak_bytes_in_use") if k in stats}


def batch_of(rows: list[list[int]], seq_len: int) -> LMBatchWithDocuments:
    """Each row one document from position 0, right-padded with document id -1."""
    ids = np.full((len(rows), seq_len), PAD_ID, np.int32)
    docs = np.full((len(rows), seq_len), -1, np.int32)
    for i, row in enumerate(rows):
        ids[i, : len(row)], docs[i, : len(row)] = row, 0
    return LMBatchWithDocuments(LMBatch(jnp.asarray(ids)), SequenceLayout(jnp.asarray(docs)))


def logits_of(output) -> jax.Array:
    """fp32-accumulated logits from the streamed output edge (`targets.losses._chunk_logits`)."""
    return jnp.dot(output.activations, output.head.T, preferred_element_type=jnp.float32)


@eqx.filter_jit
def clean(placed, batch):
    r = placed.clean_forward(batch, TAPS)
    return logits_of(r.output), r.output.activations, r.captures.read(CaptureRequest(TAPS, jax.Array))


@eqx.filter_jit
def importance(placed, weights, ci_fn, batch):
    r = placed.clean_forward(batch, ci_fn.capture_keys)
    ci = evaluate_ci_from_captures(ci_fn, r.captures, r.conditioning, weights, sequence=r.sequence, remat=False)
    return ci.lower, ci.preactivations


@eqx.filter_jit
def masked(placed, weights, batch, masks, deltas):
    """`MaterializedMasking` is not a pytree, so its arrays cross the jit boundary as dicts."""
    masking = placed.model.prepare_masking(MaterializedMasking(component_masks=masks, weight_delta_masks=deltas))
    r = placed.masked_forward(weights, batch, masking=masking, routes=None, remat=False)
    return logits_of(r.output), r.output.activations


@jax.jit
def kl_per_position(clean_logits, masked_logits):
    """KL(softmax(clean) || softmax(masked)) at every position, fp32."""
    log_p = jax.nn.log_softmax(clean_logits, axis=-1)
    log_q = jax.nn.log_softmax(masked_logits, axis=-1)
    return jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1)


@eqx.filter_jit
def adversary_kl(placed, weights, batch, g, key):
    """The eval's fresh PGD (`recon_eval.make_fresh_pgd_eval_step`) with the KL objective alone."""
    r = placed.clean_forward(batch)

    def loss(masking):
        m = placed.masked_forward(weights, batch, masking=placed.model.prepare_masking(masking), routes=None, remat=True)
        return placed.recon_loss_fn(m.output, r.output)

    attack = FreshPGDAttack(step_size=PGD_STEP_SIZE, read_out_steps=(PGD_STEPS,))
    (kl,) = fresh_pgd_read_outs(placed.model.sites, g, r.leading_shape, key, attack, loss, loss)
    return kl


def next_token_ce(logits: jax.Array, row: list[int]) -> float:
    n = len(row)
    log_p = jax.nn.log_softmax(logits[0, : n - 1], axis=-1)
    return float(-jnp.mean(jnp.take_along_axis(log_p, jnp.asarray(row[1:])[:, None], axis=-1)))


def stage_target(run, rows, seq_len, out: Path) -> dict:
    res = {"next_token_ce": [], "logit_mean": [], "logit_std": []}
    for r, row in enumerate(rows):
        logits, hidden, _ = clean(run.placed, batch_of([row], seq_len))
        n = len(row)
        save_file({"logits": np.asarray(logits[0, :n]), "final_hidden": np.asarray(hidden[0, :n])},
                  str(out / f"target_row{r}.safetensors"))
        res["next_token_ce"].append(next_token_ce(logits, row))
        res["logit_mean"].append(float(logits[0, :n].mean()))
        res["logit_std"].append(float(logits[0, :n].std()))
        say("target", row=r, length=n, next_token_ce=res["next_token_ce"][-1], logit_std=res["logit_std"][-1])
    return res


def stage_taps(run, rows, seq_len, out: Path) -> dict:
    res = {"rms": []}
    for r, row in enumerate(rows):
        _, _, taps = clean(run.placed, batch_of([row], seq_len))
        n = len(row)
        save_file({k: np.asarray(v[0, :n]) for k, v in taps.items()}, str(out / f"taps_row{r}.safetensors"))
        rms = [float(jnp.sqrt(jnp.mean(taps[f"resid.{b}"][0, :n].astype(jnp.float32) ** 2)))
               for b in range(N_BOUNDARIES)]
        res["rms"].append(rms)
        say("taps", row=r, rms_resid0=rms[0], rms_resid28=rms[-1])
    return res


def site_stats(g: dict, p: dict, n: int) -> dict:
    return {
        site: {
            "mean_g": float(g[site][0, :n].astype(jnp.float32).mean()),
            "l0": float((g[site][0, :n] > 0).sum(-1).mean()),
            "mean_p": float(p[site][0, :n].astype(jnp.float32).mean()),
        }
        for site in g
    }


def stage_gamma(run, rows, seq_len, out: Path, detail_rows) -> dict:
    res = {"per_site": [], "l0_total": [], "mean_g": []}
    for r, row in enumerate(rows):
        g, p = importance(run.placed, run.prepared_weights, run.prepared_ci_fn, batch_of([row], seq_len))
        n = len(row)
        if r in detail_rows:
            save_file({f"g.{s}": np.asarray(v[0, :n]) for s, v in g.items()}
                      | {f"p.{s}": np.asarray(v[0, :n]) for s, v in p.items()}, str(out / f"gamma_row{r}.safetensors"))
        stats = site_stats(g, p, n)
        n_components = sum(v.shape[-1] for v in g.values())
        res["per_site"].append(stats)
        res["l0_total"].append(sum(s["l0"] for s in stats.values()))
        res["mean_g"].append(sum(s["mean_g"] * g[k].shape[-1] for k, s in stats.items()) / n_components)
        say("gamma", row=r, l0_total=res["l0_total"][-1], mean_g=res["mean_g"][-1])
        del g, p
    return res


def fixed_source(sites, out: Path) -> dict[str, SiteSource]:
    """One U(0,1) value per component and one delta value per site, numpy seed SOURCE_SEED, in site order."""
    rng = np.random.default_rng(SOURCE_SEED)
    arrays = {}
    for site in sites:
        arrays[f"{site.name}.components"] = rng.random(site.C, dtype=np.float32)
        arrays[f"{site.name}.delta"] = rng.random(1, dtype=np.float32)
    save_file(arrays, str(out / "source.safetensors"))
    return {
        site.name: SiteSource(
            components=jnp.asarray(arrays[f"{site.name}.components"]).reshape(1, 1, site.C),
            delta=jnp.asarray(arrays[f"{site.name}.delta"]).reshape(1, 1),
        )
        for site in sites
    }


def stage_masked(run, rows, seq_len, out: Path, detail_rows) -> dict:
    sources = fixed_source(run.placed.model.sites, out)
    res = {name: [] for name in ("ci", "ones", "source")}
    for r, row in enumerate(rows):
        batch, n = batch_of([row], seq_len), len(row)
        clean_logits, _, _ = clean(run.placed, batch)
        g, _ = importance(run.placed, run.prepared_weights, run.prepared_ci_fn, batch)
        maskings = {
            "ci": MaterializedMasking(component_masks=g, weight_delta_masks=None),
            "ones": unmasked_no_delta_masking(g),
            "source": materialize_masking(source_masking(g, sources)),
        }
        arrays = {}
        for name, masking in maskings.items():
            logits, hidden = masked(
                run.placed, run.prepared_weights, batch, masking.component_masks, masking.weight_delta_masks
            )
            kl = kl_per_position(clean_logits, logits)[0, :n]
            arrays |= {f"{name}.kl": np.asarray(kl), f"{name}.final_hidden": np.asarray(hidden[0, :n])}
            if r in detail_rows:
                arrays[f"{name}.logits"] = np.asarray(logits[0, :n])
            res[name].append(float(kl.mean()))
            say("masked", row=r, mask=name, mean_kl=res[name][-1])
            del logits
        save_file(arrays, str(out / f"masked_row{r}.safetensors"))
        del g, maskings
    return res


def stage_adversary(run, rows, seq_len) -> dict:
    """Γ is evaluated a row at a time and stacked, so its transients never hold four rows at once. The last stage:
    Γ's 16 GiB are freed once g is computed (the attack reads g, never Γ), leaving ~28 GiB of an L40 for the
    attack's masks, their cotangents and the recomputed layers."""
    gs = [importance(run.placed, run.prepared_weights, run.prepared_ci_fn, batch_of([row], seq_len))[0]
          for row in rows]
    g = jax.block_until_ready({site: jnp.concatenate([gi[site] for gi in gs]) for site in gs[0]})
    del gs
    for leaf in jax.tree.leaves(run.prepared_ci_fn):
        if isinstance(leaf, jax.Array):
            leaf.delete()
    say("adversary", g_gib=round(tree_gib(g), 2), **device_gib())
    batch = batch_of(rows, seq_len)
    res = {"kl": [], "seeds": list(PGD_SEEDS), "rows": len(rows)}
    for seed in PGD_SEEDS:
        kl = adversary_kl(run.placed, run.prepared_weights, batch, g, jax.random.PRNGKey(seed))
        res["kl"].append(float(kl))
        say("adversary", seed=seed, kl=res["kl"][-1])
    return res


def main() -> None:
    tokens = json.loads(TOKENS.read_text())
    rows, seq_len = tokens["rows"], tokens["seq_len"]
    text_rows = [row for row, kind in zip(rows, tokens["kinds"], strict=True) if kind == "text"]
    LARGE.mkdir(parents=True, exist_ok=True)
    summary = {"device": str(jax.devices()[0]), "lengths": [len(r) for r in rows], "kinds": tokens["kinds"],
               "target_attention": "xla (stored config: flash)", "detail_rows": list(DETAIL_ROWS),
               "source_seed": SOURCE_SEED, "pgd": {"steps": PGD_STEPS, "step_size": PGD_STEP_SIZE}}
    t0 = time.time()
    run = open_decomposition()
    say("load", seconds=round(time.time() - t0), device=summary["device"])
    stages = [
        ("target", lambda: stage_target(run, rows, seq_len, LARGE)),
        ("taps", lambda: stage_taps(run, rows, seq_len, LARGE)),
        ("gamma", lambda: stage_gamma(run, rows, seq_len, LARGE, DETAIL_ROWS)),
        ("masked", lambda: stage_masked(run, rows, seq_len, LARGE, DETAIL_ROWS)),
        ("adversary", lambda: stage_adversary(run, text_rows, seq_len)),
    ]
    for name, stage in stages:
        t = time.time()
        with jax.set_mesh(run.mesh):  # the owner layout's capture buffers name mesh axes
            summary[name] = stage()
        summary[name]["seconds"] = round(time.time() - t, 1)
        SUMMARY.write_text(json.dumps(summary, indent=1))
        say(name, seconds=summary[name]["seconds"], **device_gib())
    print("FINISHED", SUMMARY, LARGE, file=sys.stdout, flush=True)


if __name__ == "__main__":
    main()
