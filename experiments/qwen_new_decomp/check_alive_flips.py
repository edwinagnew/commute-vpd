# cpu-run --mem 6G python /mnt/nw/home/e.agnew/commute-data/scratch/qwen-decomp-lead/w0-refs-lead/flips.py
# Compares the W0 g statistics rebuilt for the old decomposition (job 56661, L40) with the stored ones (H100): for each
# column whose alive status flipped, how many positions held it alive, on the side where it was alive.
from pathlib import Path

import torch
from safetensors.torch import load_file

HOME = Path("/mnt/nw/home/e.agnew/commute-data/results-large")
OLD = HOME / "qwen_sft/nowarmup/gstats"
NEW = HOME / "qwen_new_decomp/w0_refs/p-e9fcffd5/gstats"

BLOCKS = ("maths", "fineweb")
old = {b: load_file(OLD / f"W0_{b}.safetensors") for b in BLOCKS}
new = {b: load_file(NEW / f"W0_{b}.safetensors") for b in BLOCKS}
sites = sorted(k[: -len(".active")] for k in old["maths"] if k.endswith(".active"))

positions = {}
for b in BLOCKS:
    smallest = min(v[v > 0].double().min().item() for k, v in old[b].items() if k.endswith(".active"))
    positions[b] = round(1 / smallest)
    print(f"{b}: smallest non-zero active share {smallest:.4e} = 1 / {positions[b]} positions")

flips, sym = [], 0
for s in sites:
    a_old = torch.stack([old[b][f"{s}.active"].double() > 0 for b in BLOCKS]).any(0)
    a_new = torch.stack([new[b][f"{s}.active"].double() > 0 for b in BLOCKS]).any(0)
    diff = (a_old ^ a_new).nonzero().flatten()
    sym += len(diff)
    for c in diff.tolist():
        n = sum(round(max(old[b][f"{s}.active"][c].item(), new[b][f"{s}.active"][c].item()) * positions[b]) for b in BLOCKS)
        mean = max(max(old[b][f"{s}.mean_g"][c].item(), new[b][f"{s}.mean_g"][c].item()) for b in BLOCKS)
        flips.append((n, mean, bool(a_new[c]), s, c))

print(f"alive-status flips (symmetric difference): {sym}; alive only in fresh {sum(f[2] for f in flips)}, only in stored {sum(not f[2] for f in flips)}")
ns = torch.tensor([f[0] for f in flips], dtype=torch.double)
print("positions holding a flipped column alive: max", ns.max().item(), "median", ns.median().item(), "counts 1,2,3,>3:", [(ns == k).sum().item() for k in (1, 2, 3)], (ns > 3).sum().item())
print("largest mean_g among flipped columns:", max(f[1] for f in flips))
for f in sorted(flips, reverse=True)[:5]:
    print("  ", f)

for b in BLOCKS:
    deltas, totals = [], []
    for s in sites:
        d = (new[b][f"{s}.active"].double() - old[b][f"{s}.active"].double()).abs() * positions[b]
        deltas.append(d)
        totals.append(old[b][f"{s}.active"].double() * positions[b])
    d, t = torch.cat(deltas), torch.cat(totals)
    print(f"{b}: active-count change in positions: max {d.max().item():.1f}, share of columns changed {(d > 0.5).double().mean().item():.4f}, sum |change| / sum count {d.sum().item() / t.sum().item():.6f}, rounding of F16 alone at 2^-11 {2 ** -11:.6f}")
