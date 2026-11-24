"""The run's figure (the card's sketch with the real numbers): per fit in the filed order, each candidate's steering
suppression against its pool's maximum, the label above. Usage: python plot.py [tag] (default run1) ->
figures/<tag>_labels.png."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else "run1"
order = json.loads((HERE / "results" / "predictions.json").read_text())["settings"]["run"]["order"]
items = {k: json.loads(p.read_text()) for k in order if (p := HERE / "results" / tag / "items" / f"{k}.json").is_file()}
rows = [(k, r) for k, d in items.items() if d.get("done") for r in d["read"].get("rows", [])]
fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.5), 3.5))
x = range(len(rows))
ax.bar([i - 0.2 for i in x], [r["steer"] for _, r in rows], 0.4, label="candidate")
ax.bar([i + 0.2 for i in x], [r["pool_max"]["steer"] for _, r in rows], 0.4, color="gray", label="its pool's maximum")
ax.set_xticks(list(x), [f"{k} #{r['latent']}" for k, r in rows], rotation=60, fontsize=7)
for k, d in items.items():
    first = next((i for i, (kk, _) in enumerate(rows) if kk == k), None)
    if first is not None:
        ax.annotate(d["read"]["label"], (first, ax.get_ylim()[1]), fontsize=7, va="top")
ax.set_ylabel("suppression of g (nats)")
ax.legend(fontsize=7)
fig.tight_layout()
(HERE / "figures").mkdir(exist_ok=True)
fig.savefig(HERE / "figures" / f"{tag}_labels.png", dpi=100)
