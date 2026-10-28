"""SFT of Qwen3-1.7B on GSM8K's human-written solutions, and the pieces that read and save its update (E40, plan
reports/plans/2026-10-28_plan_qwen-sft.md, stage F).

Trainable: the 196 decomposed matrices only (`qwen_vpd.SITES`); embedding, tied head and norm gains stay frozen, so δW
lies in those matrices. fp32 master weights, bf16 autocast forward, gradient checkpointing (`use_reentrant=False`),
fused AdamW. A sequence is the baseline's chat prompt (thinking off, `qwen_eval.prompt`) + `gold_as_response` + the end
token, cut at 512; the loss is the mean token log-loss over the response tokens of the whole batch.

Weights are swapped by pointer: `set_weights` assigns `.data` of each decomposed nn.Linear and restores the previous
tensor on exit, so a state costs one set of matrices and restoring is exact. Matrices are stored in safetensors files
keyed by site, in HF layout [d_out, d_in].
"""

from __future__ import annotations

import math
import os
import re
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import Tensor

from commute.gsm8k import gold_as_response
from commute.qwen_eval import MAX_NEW, prompt
from commute.qwen_files import ENTITY, PROJECT
from commute.qwen_vpd import SITES
from commute.vpd import kl_per_position

Seq = tuple[list[int], int]  # token ids, number of prompt tokens (the response is ids[n_prompt:])
Item = tuple[list[int], int | list[bool]]  # a Seq, or token ids with a per-token target flag (a packed row)
Weights = Mapping[str, Tensor]  # site -> [d_out, d_in], fp32 on the model's device
DISK_MARGIN = 2 * 10**9  # bytes left free on the volume after any save


def linears(model: Any) -> dict[str, torch.nn.Linear]:
    return {s: model.get_submodule("model." + s) for s in SITES}


@contextmanager
def set_weights(model: Any, W: Weights | None) -> Iterator[None]:
    """The decomposed matrices set to W for the block (None: unchanged); the previous tensors back on exit."""
    if W is None:
        yield
        return
    lins = linears(model)
    before = {s: lin.weight.data for s, lin in lins.items()}
    try:
        for s, lin in lins.items():
            lin.weight.data = W[s]
        yield
    finally:
        for s, lin in lins.items():
            lin.weight.data = before[s]


def plus(W0: Weights, change: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """W0 + change in fp32 per site (change in any dtype, on any device)."""
    return {s: W0[s] + change[s].to(W0[s].device, torch.float32) for s in SITES}


# ----------------------------------------------------------------------------- files


_VOLUME: dict[str, Any] = {"root": None, "bytes": None}


def set_volume(root: Path | None, gb: float | None) -> None:
    """A quota'd network volume mounted at `root`: free space under it is `gb` minus the bytes of its files, because
    `df` there reports the shared backend (RunPod's MooseFS: 121 TB free on a 150 GB volume). None: `df` is right."""
    _VOLUME.update(root=None if root is None else Path(root).resolve(), bytes=None if gb is None else gb * 1e9)


def du(root: Path) -> int:
    """Bytes of the regular files under `root` (as `du -sb`, symlinks not followed; a file that vanishes during the
    walk, as a temporary one may, is skipped)."""
    total = 0
    for d, _, files in os.walk(root):
        for f in files:
            p = Path(d) / f
            try:
                if not p.is_symlink():
                    total += p.stat().st_size
            except FileNotFoundError:
                continue
    return total


def free_bytes(path: Path) -> int:
    """Free space where `path` lives: under a `set_volume` root the smaller of the quota minus its contents and `df`,
    else `df`."""
    path, root = Path(path).resolve(), _VOLUME["root"]
    df = shutil.disk_usage(path).free
    if root is not None and path.is_relative_to(root):
        return min(int(_VOLUME["bytes"] - du(root)), df)
    return df


def nbytes(tensors: Mapping[str, Tensor], dtype: torch.dtype) -> int:
    return sum(t.numel() for t in tensors.values()) * torch.empty((), dtype=dtype).element_size()


def save(path: Path, tensors: Mapping[str, Tensor], dtype: torch.dtype, droppable: bool, log: Any = print,
         meta: dict[str, str] | None = None) -> bool:
    """Write `tensors` cast to `dtype` (CPU, contiguous) to a safetensors file if the volume has room for it plus
    DISK_MARGIN. No room: a droppable item is skipped (False, logged); any other raises."""
    path.parent.mkdir(parents=True, exist_ok=True)
    need = nbytes(tensors, dtype)
    free = free_bytes(path.parent) + (path.stat().st_size if path.is_file() else 0)  # an overwrite
    if free < need + DISK_MARGIN:
        msg = f"{path.name}: {need / 1e9:.1f} GB does not fit ({free / 1e9:.1f} GB free)"
        if not droppable:
            raise RuntimeError(msg)
        log(f"[save] DROPPED {msg}")
        return False
    tmp = path.with_name(path.name + ".tmp")  # written whole, then renamed: a lost pod never leaves half a file
    save_file({k: v.detach().to("cpu", dtype).contiguous() for k, v in tensors.items()}, str(tmp), metadata=meta)
    os.replace(tmp, path)
    log(f"[save] {path} ({need / 1e9:.2f} GB)")
    return True


def load(path: Path, device: str = "cpu") -> dict[str, Tensor]:
    return load_file(str(path), device=device)


# ----------------------------------------------------------------------------- sequences and rows


def sft_seq(tok: Any, question: str, solution: str, end_id: int, seq_len: int) -> Seq:
    p = tok(prompt(tok, question), add_special_tokens=False)["input_ids"]
    r = tok(gold_as_response(solution), add_special_tokens=False)["input_ids"] + [end_id]
    ids = (p + r)[:seq_len]
    assert len(ids) > len(p), "the prompt fills the window"
    return ids, len(p)


def pack(seqs: list[list[int]], n_rows: int | None, seq_len: int) -> list[list[int]]:
    """Sequences joined end to end and cut into rows of seq_len; n_rows None: every whole row."""
    flat = [t for s in seqs for t in s]
    n = len(flat) // seq_len if n_rows is None else n_rows
    assert n * seq_len <= len(flat), f"{len(flat)} tokens make {len(flat) // seq_len} rows, {n} needed"
    return [flat[i * seq_len : (i + 1) * seq_len] for i in range(n)]


@torch.no_grad()
def answer_ids(tok: Any, gen: Any, texts: list[str], batch: int) -> list[list[int]]:
    """Greedy answers as token ids, `qwen_eval.generate`'s settings (left padding, MAX_NEW new tokens): per prompt,
    its ids followed by the response up to and including the first end token (all MAX_NEW if none came)."""
    eos = torch.tensor(gen.generation_config.eos_token_id).flatten()
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s : s + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(gen.device)
        full = gen.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False).cpu()
        P = enc["input_ids"].shape[1]
        for ids, mask, row in zip(enc["input_ids"].cpu(), enc["attention_mask"].cpu(), full, strict=True):
            resp = row[P:]
            hits = torch.isin(resp, eos).nonzero()
            n = int(hits[0]) + 1 if len(hits) else len(resp)
            out.append(ids[mask.bool()].tolist() + resp[:n].tolist())
    return out


# ----------------------------------------------------------------------------- the trainable copy, the rate


@contextmanager
def trainable(model: Any, W0: Weights) -> Iterator[dict[str, Tensor]]:
    """The decomposed matrices set to fp32 copies of W0 that require grad (the fp32 master weights); yields site ->
    Parameter, whose .data is the copy. On exit W0 is back in place and nothing requires grad."""
    with set_weights(model, {s: w.clone() for s, w in W0.items()}):
        params = {s: lin.weight.requires_grad_(True) for s, lin in linears(model).items()}
        try:
            yield params
        finally:
            for p in params.values():
                p.requires_grad_(False)
                p.grad = None


def adamw(params: Mapping[str, Tensor], lr: float) -> torch.optim.AdamW:
    """AdamW, betas (0.9, 0.999), no weight decay, fused; the rate is set per step by the caller (`lr_at`)."""
    return torch.optim.AdamW(list(params.values()), lr=lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)


def lr_at(t: int, peak: float, warmup: int, steps: int, schedule: str) -> float:
    """The rate of optimiser step t (1-based). "constant": peak. "hf_cosine": the rate
    `transformers.get_cosine_schedule_with_warmup(opt, warmup, steps)` gives step t, written out to check the library
    against (`lr_schedule`): its counter k = t − 1 starts at 0, so step 1 runs at rate 0; peak·k/warmup while k < warmup,
    then peak·½(1 + cos(π(k − warmup)/(steps − warmup))). Amendment 1's recipe (warm-up 10, cosine over 100, peak 1e-5)
    is LLaMA-Factory's documented full-SFT setting for Qwen (lr 1e-5, cosine, warm-up ratio 0.1)."""
    if schedule == "constant":
        return peak
    assert schedule == "hf_cosine", schedule
    k = t - 1
    if k < warmup:
        return peak * k / max(1, warmup)
    return peak * max(0.0, 0.5 * (1 + math.cos(math.pi * (k - warmup) / max(1, steps - warmup))))


def lr_schedule(opt: torch.optim.Optimizer, warmup: int, steps: int, kind: str) -> Any:
    """The library's scheduler for "hf_cosine" (stepped after every optimiser step); None for "constant"."""
    if kind == "constant":
        return None
    from transformers import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=steps)


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def meta_of(path: Path) -> dict[str, str]:
    """A safetensors file's metadata (empty if none)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return f.metadata() or {}


# ----------------------------------------------------------------------------- loss and step


def targets(seqs: Sequence[Item], pad: int) -> tuple[Tensor, Tensor]:
    """Right-padded tokens [b, T] and the mask [b, T - 1] of logit positions whose next token is a response token
    (a flag list marks the target tokens themselves; its first entry has no logit position and must be False)."""
    T = max(len(ids) for ids, _ in seqs)
    toks = torch.full((len(seqs), T), pad, dtype=torch.long)
    mask = torch.zeros(len(seqs), T - 1, dtype=torch.bool)
    for i, (ids, p) in enumerate(seqs):
        toks[i, : len(ids)] = torch.tensor(ids)
        if isinstance(p, int):
            mask[i, p - 1 : len(ids) - 1] = True
        else:
            assert len(p) == len(ids) and not p[0], "one flag per token, the first never a target"
            mask[i, : len(ids) - 1] = torch.tensor(p[1:], dtype=torch.bool)
    return toks, mask


def n_targets(seqs: Sequence[Item]) -> int:
    return sum(len(ids) - p if isinstance(p, int) else sum(p[1:]) for ids, p in seqs)


def nll_sum(model: Any, seqs: Sequence[Item], device: str) -> Tensor:
    """Summed log-loss of the response tokens; causal attention, so right padding changes no real position."""
    toks, mask = targets(seqs, pad=0)
    toks, mask = toks.to(device), mask.to(device)
    logits = model(toks, use_cache=False).logits[:, :-1]
    nll = F.cross_entropy(logits.float().flatten(0, 1), toks[:, 1:].flatten(), reduction="none")
    return (nll * mask.flatten()).sum()


def sft_step(model: Any, seqs: Sequence[Item], micro: int, device: str) -> float:
    """Gradients of the batch's mean response-token log-loss accumulated in micro-batches (bf16 autocast); returns
    the loss. The caller zeroes gradients before and steps the optimiser after."""
    n, total = n_targets(seqs), 0.0
    for i in range(0, len(seqs), micro):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = nll_sum(model, seqs[i : i + micro], device)
        (s / n).backward()
        total += float(s)
    return total / n


@torch.no_grad()
def gold_loss(model: Any, seqs: Sequence[Item], batch: int, device: str) -> float:
    """Mean response-token log-loss in fp32 (no autocast), pooled over every response token of `seqs`."""
    total = sum(float(nll_sum(model, seqs[i : i + batch], device).double()) for i in range(0, len(seqs), batch))
    return total / n_targets(seqs)


@torch.no_grad()
def mean_kl(model: Any, rows: list[Tensor], p: Weights | None, q: Weights | None) -> float:
    """Mean over every position of KL(model at p ‖ model at q) on `rows` (None: the weights as they stand), in the
    model's dtype (fp32 here)."""
    total, n = 0.0, 0
    for tokens in rows:
        with set_weights(model, p):
            lp = model(tokens, use_cache=False).logits
        with set_weights(model, q):
            lq = model(tokens, use_cache=False).logits
        total += float(kl_per_position(lp, lq).double().sum())
        n += tokens.numel()
    return total / n


def adam_state(opt: torch.optim.Optimizer, params: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], int]:
    """The two Adam moments per site (fp32) and the step count."""
    out, steps = {}, set()
    for s, p in params.items():
        st = opt.state[p]
        out[f"{s}.exp_avg"], out[f"{s}.exp_avg_sq"] = st["exp_avg"], st["exp_avg_sq"]
        steps.add(int(st["step"]))
    assert len(steps) == 1, steps
    return out, steps.pop()


class Uploader:
    """Each saved file logged to W&B as its own artifact, for a pod whose disk dies with it (`qwen_files`' entity and
    project; hashed in place, `policy="immutable"`, `skip_cache=True`, as experiments/qwen_port/upload.py, so a file
    must not change once added). `log_artifact` returns at once and wandb uploads in the background; `finish()` waits
    for every upload. A file over `max_bytes` is skipped and logged (the rehearsal's cap). Needs WANDB_API_KEY."""

    def __init__(self, prefix: str, max_bytes: int | None, wandb_dir: Path, log: Any = print):
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is not set in the environment")
        import wandb

        wandb_dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.max_bytes, self.log, self.sent = prefix, max_bytes, log, []
        self.run: Any = wandb.init(entity=ENTITY, project=PROJECT, job_type=prefix, dir=str(wandb_dir))

    def add(self, path: Path, wait: bool = False) -> None:
        """Queue `path` as an artifact; with `wait`, block until it is uploaded (the benchmark's timing)."""
        import wandb

        size = path.stat().st_size
        if self.max_bytes is not None and size > self.max_bytes:
            self.log(f"[upload] skipped {path} ({size / 1e9:.2f} GB, over the cap)")
            return
        name = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{self.prefix}-{path.parent.name}-{path.name}")
        art = wandb.Artifact(name, type=self.prefix)
        art.add_file(str(path), skip_cache=True, policy="immutable")
        logged = self.run.log_artifact(art)
        if wait:
            logged.wait()
        self.sent.append(name)
        self.log(f"[upload] queued {name} ({size / 1e9:.2f} GB)")

    def finish(self) -> None:
        self.run.finish()
        self.log(f"[upload] finished: {len(self.sent)} artifacts")
