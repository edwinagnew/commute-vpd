import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
items = ["case", "twin", "fact", "kv", "repeat"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, (title, kv) in zip(axes, [("rival A: K like K*", [0.9995] * 5), ("rival B: K lower", [0.9995, 0.97, 0.999, 0.93, 0.9993])]):
    x = range(len(items))
    ax.scatter(x, [0.9995] * 5, marker="o", label="over K* (saved)")
    ax.scatter(x, kv, marker="x", label="over K (made up)")
    ax.axhline(0.999, ls=":", c="grey")
    ax.set_xticks(list(x), items)
    ax.set_title(title)
axes[0].set_ylabel("graded cos^2 reading")
axes[0].legend(fontsize=7)
fig.savefig(f"{out}/2026-11-26_sketch_p165-run1.png", dpi=80, bbox_inches="tight")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
feats = ["VPD s0", "VPD s1", "VPD s2", "SAE s0", "SAE s1", "SAE s2"]
for ax, (title, g) in zip(axes, [("rival A: as low as full", [0.04, 0.05, 0.04, 0.04, 0.03, 0.05]), ("rival B: control high", [0.04, 0.3, 0.35, 0.4, 0.38, 0.42])]):
    x = range(len(feats))
    ax.scatter(x, [0.033, 0.033, 0.033, 0.04, 0.04, 0.04], marker="o", label="full fine-tunes")
    ax.scatter(x, g, marker="x", label="rescale-only (made up)")
    ax.set_xticks(list(x), feats, fontsize=7)
    ax.set_title(title)
axes[0].set_ylabel("E_b over K")
axes[0].legend(fontsize=7)
fig.savefig(f"{out}/2026-11-26_sketch_p165-run2.png", dpi=80, bbox_inches="tight")
