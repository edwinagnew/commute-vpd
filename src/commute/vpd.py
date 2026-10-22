"""The released VPD decomposition: fetch, key-free load, causal importances, alive set, accessors.

Layout under `env.VPD_DIR` (as vpd-audit lays it out):
    s-55ea3f9b/  model_400000.pth  final_config.yaml            the decomposition
    t-9d2b8f02/  final_config.yaml model_config.yaml ckpt/model_step_99999.pt   its target model

Conventions. A decomposed matrix `name` (e.g. `h.2.mlp.down_proj`) is an nn.Linear with weight
W of shape (d_out, d_in). Its components hold V (d_in, C) and U (C, d_out); subcomponent c is the
rank-1 matrix B_c = U[c]^T V[:, c]^T, i.e. B_c[o, i] = U[c, o] V[i, c], and sum_c B_c + Delta = W.
The causal-importance function Gamma is read through `compute_ci`; g is `lower_leaky` in [0, 1].
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from commute import env
from commute.vpd_files import (  # noqa: F401  (fetch and the W&B names are re-exported)
    DECOMP_FILES,
    DECOMP_RUN,
    TARGET_CKPT,
    TARGET_FILES,
    TARGET_RUN,
    WANDB_PROJECT,
    fetch,
)

SEQ_LEN = 512
CI_CHUNK = 16  # sequences per Gamma pass; the (B, 512, 38912) tensors are the memory hog

EMOTICON_KEY = "h.2.mlp.down_proj:2359"
EMOTICON_SIBLINGS = tuple(f"h.2.mlp.down_proj:{c}" for c in (1672, 2623, 3290, 3327, 3382))
PAPER_FAITHFULNESS = {"stochastic_kl": 0.2381, "pgd_kl": 0.9268}  # eval table, VPD blog post


# ----------------------------------------------------------------------------- fetch / load


def load_component_model(device: str = "cuda") -> Any:
    """Key-free load from the local files; the target model loads by its local branch too."""
    from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo

    decomp_ckpt = env.VPD_DIR / DECOMP_RUN / DECOMP_FILES[0]
    target = env.VPD_DIR / TARGET_RUN / "ckpt" / TARGET_CKPT
    for p in (decomp_ckpt, decomp_ckpt.parent / "final_config.yaml", target,
              target.parent.parent / "final_config.yaml", target.parent.parent / "model_config.yaml"):
        assert p.is_file(), f"missing {p}; run `fetch` first"
    run_info = ParamDecompRunInfo.from_path(decomp_ckpt)
    run_info.config = run_info.config.model_copy(update={"pretrained_model_name": str(target)})
    model = ComponentModel.from_run_info(run_info).to(device).eval()
    assert len(model.components) == 24 and sum(model.module_to_c.values()) == 38_912
    assert all(not p.requires_grad for p in model.target_model.parameters())
    model.requires_grad_(False)
    return model


def run_config() -> Any:
    """The released decomposition's `Config` (param-decomp's pydantic object, from its final_config.yaml): the loss
    stack with its coefficients, the sampling, the optimizer settings. Read from the file, never copied by hand."""
    from param_decomp.models.component_model import ParamDecompRunInfo

    return ParamDecompRunInfo.from_path(env.VPD_DIR / DECOMP_RUN / DECOMP_FILES[0]).config


def tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


def sparsity_config() -> dict[str, float]:
    """The released run's importance-minimality loss as it stood at the END of its training (`final_config.yaml`):
    p is the annealed final value when annealing was on (2.0 -> 0.4 over the run), beta and eps as configured.
    Read from the file so the numbers are never copied by hand."""
    import yaml

    cfg = yaml.safe_load((env.VPD_DIR / DECOMP_RUN / "final_config.yaml").read_text())
    im = next(c for c in cfg["loss_metric_configs"] if c["classname"] == "ImportanceMinimalityLoss")
    annealed = im.get("p_anneal_final_p") is not None and im.get("p_anneal_start_frac", 1.0) < 1.0
    return {"pnorm": float(im["p_anneal_final_p"] if annealed else im["pnorm"]), "beta": float(im["beta"]),
            "eps": float(im.get("eps", 1e-12))}


# ----------------------------------------------------------------------------- accessors

_LAYER_RE = re.compile(r"^h\.(\d+)\.")


def module_keys(model: Any) -> list[str]:
    return sorted(model.module_to_c)


def layer_of(key: str) -> int:
    m = _LAYER_RE.match(key)
    assert m, key
    return int(m.group(1))


def parse_key(key: str) -> tuple[str, int]:
    name, idx = key.rsplit(":", 1)
    return name, int(idx)


def target_linear(model: Any, name: str) -> nn.Linear:
    mod = model.target_model.get_submodule(name)
    assert isinstance(mod, nn.Linear) and mod.bias is None
    return mod


def target_weights(model: Any) -> dict[str, Tensor]:
    """Detached fp32 clones of the 24 decomposed weights, (d_out, d_in)."""
    return {k: target_linear(model, k).weight.detach().float().clone() for k in module_keys(model)}


def set_target_weights(model: Any, weights: dict[str, Tensor]) -> None:
    with torch.no_grad():
        for k, w in weights.items():
            lin = target_linear(model, k)
            lin.weight.copy_(w.to(lin.weight.dtype))


def uv(model: Any, name: str) -> tuple[Tensor, Tensor]:
    """(U (C, d_out), V (d_in, C)) of a decomposed matrix, detached."""
    comp = model.components[name]
    return comp.U.detach(), comp.V.detach()


def component_weights(model: Any) -> dict[str, tuple[Tensor, Tensor]]:
    return {k: (model.components[k].U.detach().clone(), model.components[k].V.detach().clone()) for k in module_keys(model)}


def set_component_weights(model: Any, weights: dict[str, tuple[Tensor, Tensor]]) -> None:
    with torch.no_grad():
        for k, (u, v) in weights.items():
            model.components[k].U.copy_(u)
            model.components[k].V.copy_(v)


def weight_deltas(model: Any) -> dict[str, Tensor]:
    """Delta = W_target - sum_c B_c per matrix, as the authors compute it."""
    with torch.no_grad():
        return model.calc_weight_deltas()


# ----------------------------------------------------------------------------- forward helpers


def target_logits(model: Any, batch: Tensor) -> Tensor:
    """Logits of the target model as it currently stands (bf16 autocast on CUDA)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=batch.is_cuda):
        return model(batch)


@dataclass
class CI:
    g: dict[str, Tensor]  # module -> (B, T, C) lower-leaky causal importances in [0, 1]
    logits: Tensor  # (B, T, V) target logits of the same forward


def compute_ci(model: Any, batch: Tensor, chunk: int = CI_CHUNK) -> CI:
    """g and the target logits for a batch of token ids (B, T), T <= 512, exactly as the authors do:
    clean forward caching the inputs of the 24 matrices, then Gamma with `sampling='continuous'`.
    Gamma reads each position's own pre-weight activations, so a 41-token window is as valid an input as a
    512-token row (the model is causal with rotary positions; `n_ctx` 512 is a maximum). Relaxed from `== SEQ_LEN`
    on 2026-10-15 for the task-local harness (`local_faith.py`)."""
    assert batch.ndim == 2 and 1 <= batch.shape[1] <= SEQ_LEN, tuple(batch.shape)
    keys = module_keys(model)
    g_parts: dict[str, list[Tensor]] = {k: [] for k in keys}
    logit_parts: list[Tensor] = []
    device = next(model.parameters()).device
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for s in range(0, batch.shape[0], chunk):
            sub = batch[s : s + chunk].to(device)
            out = model(sub, cache_type="input")
            ci = model.calc_causal_importances(out.cache, sampling="continuous", detach_inputs=False)
            for k in keys:
                g_parts[k].append(ci.lower_leaky[k])
            logit_parts.append(out.output)
            del out, ci
    return CI(g={k: torch.cat(g_parts[k]) for k in keys}, logits=torch.cat(logit_parts))


def mean_ci(model: Any, batches: list[Tensor]) -> dict[str, Tensor]:
    """Mean g per subcomponent over all positions of the given batches, float64 on CPU."""
    keys = module_keys(model)
    sums = {k: torch.zeros(model.module_to_c[k], dtype=torch.float64) for k in keys}
    n = 0
    for b in batches:
        ci = compute_ci(model, b)
        for k in keys:
            sums[k] += ci.g[k].sum(dim=(0, 1), dtype=torch.float64).cpu()
        n += b.shape[0] * b.shape[1]
        del ci
    return {k: s / n for k, s in sums.items()}


def alive_masks(mean_g: dict[str, Tensor], threshold: float = 1e-6) -> dict[str, Tensor]:
    """The paper's alive rule: mean causal importance above 1e-6."""
    return {k: (m > threshold) for k, m in mean_g.items()}


def kl_per_position(p_logits: Tensor, q_logits: Tensor) -> Tensor:
    """KL(P || Q) per position in float32, shapes (..., V) -> (...)."""
    lp = torch.log_softmax(p_logits.float(), dim=-1)
    lq = torch.log_softmax(q_logits.float(), dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1)
