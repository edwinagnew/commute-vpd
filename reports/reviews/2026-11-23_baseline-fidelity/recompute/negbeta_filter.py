"""Recompute the specific part with Minder et al. App. E.2's exclusion (drop a latent whose beta_base < 0 for the
error or the reconstruction target) added to the saved fine-tune-specific mask; s, bias and everything else as saved.
Reads results-large .pt files with weights_only=True. Usage: python negbeta_filter.py"""
import sys
from pathlib import Path

import torch

L = Path.home() / "commute-data" / "results-large"
ITEMS = [
    ("E235 B head c0", "crosscoder_proper/run1/head.B.c0.pt"),
    ("E235 B head c1", "crosscoder_proper/run1/head.B.c1.pt"),
    ("E235 B head c2", "crosscoder_proper/run1/head.B.c2.pt"),
    ("E235 A head c0", "crosscoder_proper/run1/head.A.c0.pt"),
    ("E235 A head c1", "crosscoder_proper/run1/head.A.c1.pt"),
    ("E235 A head c2", "crosscoder_proper/run1/head.A.c2.pt"),
    ("E243 rung1 c0", "crosscoder_proper/run2/head-r1.A.c0.pt"),
    ("E243 rung2 c0", "crosscoder_proper/run2/head-r2.A.c0.pt"),
    ("E243 rung4 c0", "crosscoder_proper/run2/head-r4.A.c0.pt"),
    ("E243 rung4 c1", "crosscoder_proper/run2/head-r4.A.c1.pt"),
    ("E243 rung5 c0", "crosscoder_proper/run2/head-r5.A.c0.pt"),
    ("E203 e58 copy", "crosscoder/run2/e58__copy_s0.pt"),
    *[(f"E235 B gauss c{c}", f"crosscoder_proper/run1/gauss.B.c{c}.pt") for c in range(3)],
    *[(f"E235 B {t}", f"crosscoder_proper/run1/e54__{t}.B.c0.pt") for t in ("elicit_s0", "elicit_s1", "elicit_s2", "cipher_s0", "cipher_s1", "cipher_s2")],
    ("E235 B self", "crosscoder_proper/run1/self.B.c0.pt"),
]


def one(name: str, rel: str) -> None:
    d = torch.load(L / rel, map_location="cpu", weights_only=True)
    s, J, b = d["s"].double(), d["scaled"].long(), d["betas"]
    g = d["groups"]
    spec = g["ft_specific"].bool()
    neg = torch.zeros_like(spec)
    neg[J] = (b["eps_base"] < 0) | (b["r_base"] < 0)
    keep = spec & ~neg
    # check the saved mask is the thresholds on the saved betas (sanity)
    thr = torch.zeros_like(spec)
    thr[J] = (b["nu_r"] < 0.5) & (b["nu_eps"] < 0.2)
    other = {k: float(s[m.bool()].sum()) for k, m in g.items() if k != "ft_specific"}
    print(f"{name:16s} specific {float(s[spec].sum()):.6g} -> filtered {float(s[keep].sum()):.6g}  "
          f"count {int(spec.sum())} -> {int(keep.sum())}  dropped {int((spec & neg).sum())}  "
          f"mask==thresholds&top: {bool((spec & ~thr).sum() == 0)}  "
          f"neg_eps {int((b['eps_base'] < 0).sum())} neg_r {int((b['r_base'] < 0).sum())} of {len(J)}  "
          f"shared {other.get('shared', float('nan')):.4f} artefact {other.get('ft_artefact', float('nan')):.4f} "
          f"sum_s {float(s.nansum()):.4f}", flush=True)


for n, r in ITEMS:
    try:
        one(n, r)
    except Exception as e:  # report and go on
        print(f"{n}: ERROR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
