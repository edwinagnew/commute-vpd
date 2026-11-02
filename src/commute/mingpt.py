"""Karpathy's minGPT (github.com/karpathy/minGPT, `mingpt/model.py`) in plain PyTorch, as used by Jain et al. 2024 for
the PCFG setting (L1160-1163: 6 blocks, 6 heads, d 192, two-layer MLP, ~3M parameters).

Kept from minGPT: pre-LayerNorm blocks, learned position embeddings, 4x MLP with tanh GELU, untied output head without
bias, dropout 0.1 on embeddings, attention weights and residual branches (`GPT.get_default_config`), N(0, 0.02) init
with the residual projections at 0.02 / sqrt(2 n_layer), and weight decay on Linear weights only
(`configure_optimizers`). Attention uses `scaled_dot_product_attention` (same function, fused kernel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class GPTConfig:
    vocab: int
    block: int  # longest input
    n_layer: int = 6
    n_head: int = 6
    d: int = 192
    dropout: float = 0.1


class SelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head, self.p = cfg.n_head, cfg.dropout
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d)
        self.proj = nn.Linear(cfg.d, cfg.d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = (z.view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for z in self.qkv(x).split(d, dim=2))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        return self.drop(self.proj(y.transpose(1, 2).reshape(b, t, d)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d), nn.LayerNorm(cfg.d)
        self.attn = SelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d, 4 * cfg.d), nn.GELU(approximate="tanh"), nn.Linear(4 * cfg.d, cfg.d), nn.Dropout(cfg.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte, self.wpe = nn.Embedding(cfg.vocab, cfg.d), nn.Embedding(cfg.block, cfg.d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.apply(_init)
        for name, w in self.named_parameters():  # minGPT's c_proj rule: residual-branch outputs
            if name.endswith(("attn.proj.weight", "mlp.2.weight")):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device)))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def _init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, std=0.02)
    if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.zeros_(m.bias)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW with minGPT's trainer defaults (betas 0.9, 0.95) and decay on Linear weights only."""
    decay = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    params = list(model.parameters())
    groups = [
        {"params": [p for p in params if id(p) in decay], "weight_decay": weight_decay},
        {"params": [p for p in params if id(p) not in decay], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
