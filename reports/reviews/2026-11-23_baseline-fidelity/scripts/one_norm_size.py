"""Read-only: the size column under one norm (E187's ||dW||_F/||W0|| of the as-saved states, m67.json .pairs.<p>.sizes)
beside final.json's size cells, for the nine built pairs. Gap log10(reuse mean / new mean); call = seed ranges."""
import json, math, statistics as S
from pathlib import Path
R = Path("/mnt/nw/home/e.agnew/commute-agents/redteam-lead/experiments")
m = json.loads((R/"size_matched/results/m67.json").read_text())["pairs"]
F = {c["pair"]: c for c in json.loads((R/"method_table/results/final.json").read_text())["cells"] if c["method"] == "size"}
sh = {c["pair"]: c for c in json.loads((R/"method_table/results/final.json").read_text())["cells"] if c["method"] == "share"}
def rd(a, b):
    g = math.log10(S.fmean(a) / S.fmean(b))
    if min(len(a), len(b)) < 2: c = "one value a side"
    elif min(a) > max(b) or max(a) < min(b): c = "apart"
    else: c = "no call"
    return g, c
def verdict(g, c, share_g):
    if c != "apart": return "no call"
    return "agree" if (g > 0) == (share_g > 0) else "reverse"
tally = {"one_norm": {}, "scorecard": {}}
for p in m:
    a, b = m[p]["sizes"]["reuse"], m[p]["sizes"]["new"]
    g1, c1 = rd(a, b)
    f = F[p]
    v1, v0 = verdict(g1, c1, sh[p]["gap"]), verdict(f["gap"], f["call"], sh[p]["gap"])
    tally["one_norm"][v1] = tally["one_norm"].get(v1, 0) + 1
    tally["scorecard"][v0] = tally["scorecard"].get(v0, 0) + 1
    margin = (min(b) - max(a)) if S.fmean(b) > S.fmean(a) else (min(a) - max(b))
    print(f"{p}: one-norm n={len(a)}/{len(b)} gap {g1:+.4f} {c1} [{v1}]; reuse {min(a):.8f}-{max(a):.8f} new {min(b):.8f}-{max(b):.8f}; "
          f"separation margin {margin:+.3e} (={margin / S.fmean(a + b) * 100:+.3f}% of the size); mean diff {(S.fmean(b) / S.fmean(a) - 1) * 100:+.2f}% | "
          f"scorecard n={f['n_reuse']}/{f['n_new']} gap {f['gap']:+.4f} {f['call']} [{v0}] form '{f['form']}'")
print(tally)
