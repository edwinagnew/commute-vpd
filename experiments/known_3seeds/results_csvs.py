"""P2.77: the data behind the paper's Fig 3a (share gap per pair) and Fig 4 (share gap against prior and steering gap),
rebuilt so every point on an axis is one form of its reading. $0, CPU, saved files only:

    ~/commute-agents/fleet/cpu-run --mem 8G python experiments/known_3seeds/results_csvs.py

Writes results/csv/paper_fig3a_share.csv and paper_fig4_share_vs_prior_steering.csv with the columns and rows of
experiments/method_table/paper_figs.py's CSVs, plus a sibling .forms.json per CSV naming the form of each column. Prints
old (method_table/figures) against new per row, and the steering reversal count in the pooled and per-key forms.

Forms (redteam review 2026-11-24_review_scorecard-forms.md, MUST 1 and 2):
- share: E136 re-gating share of the trained update, log-odds, on every pair. Built pairs from known3.json; the four
  twins from final.json's re-emitted share cell, which must carry a unit and must not be the label contrast.
- steering (Fig 4b x): E143 pooled share on every pair. Built pairs from known3.json "steer" (holes_fill / refit_96key
  `.steering.pooled.share`; e62 has no E143 reading, so E135 steering_copy_lookup `.pooled.share_lo`); twins from
  final.json's re-emitted pooled column (POOLED below). Per-key (PERKEY) is read only for the second reversal count.
- prior: log10 of the original model's probability of the trained answer, one value per task (gap in decades).
- built pairs read exactly known3.json's states (seeds 0-2 a side; e46 ident32 vs rand32, e53 lowercase vs cipher).
  A null there (E299's five seed-2 states, until its stage 2 lands) is dropped; rerun this file once known3.json fills.
- "matched" rows: E187 size_matched/summary.json as before (its own seeds); its matched steering point is kept only
  where its form is pooled, so e54_matched and e54_own (per-key there) lose their matched steering point.
- the starred pair (E199): its own share and strength-1 pooled steering (E151's reader), as before.

A reversal is the scorecard's "opposite to the share": the steering seed ranges are apart (every reuse seed beyond every
new seed, or below every one) and point the other way from the share's gap. The paper counts it over the nine built pairs.

Until compare-lead's re-emitted final.json (log-odds share on every pair, steering as POOLED and PERKEY cells, a "unit"
per cell) is on main, the twin rows and the per-key count print as WAITING and the script exits 1 after writing.
"""

import csv
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
KNOWN3 = "experiments/known_3seeds/results/known3.json"
FINAL = "experiments/method_table/results/final.json"
MATCHED = "experiments/size_matched/results/summary.json"
E199 = "experiments/disagreeing_pairs2/results/stage2.json"
OLD = ROOT / "experiments/method_table/figures"
OUT = ROOT / "experiments/known_3seeds/results/csv"
POOLED, PERKEY = (
    "steering_pooled",
    "steering_perkey",
)  # re-emitted final.json method names; check once pushed
E199_NAME = "nearest-word copy vs cipher, built so prior and steering disagree"
FIG3A, FIG4 = "paper_fig3a_share.csv", "paper_fig4_share_vs_prior_steering.csv"
Point = dict[str, Any]
WAITING: list[str] = []
FORMS = {
    "y_share_gap": "E136 re-gating share, trained update, log-odds; reuse-side mean minus new-side mean (twins: real minus twin)",
    "y_lo/y_hi": "lowest reuse seed minus highest new seed / highest reuse seed minus lowest new seed",
    "x_gap (panel prior)": "log10 prior, one value per task, gap in decades",
    "x_gap (panel steering)": "E143 pooled steering share, filed selection rule (e62: E135 pooled share_lo, no E143 reading; "
    "E199 star: strength-1 pooled, E151 reader)",
    "states": "built pairs: known3.json seeds 0-2 a side (nulls dropped); twins: final.json cell; matched: E187's seeds",
}


def load(path: str) -> Any:
    return json.loads((ROOT / path).read_text())


def reading(
    reuse: list[float | None], new: list[float | None], src: str, key: str
) -> Point | None:
    """One gap with its seed range, nulls dropped; None when a side has no value."""
    r, n = [v for v in reuse if v is not None], [v for v in new if v is not None]
    if not r or not n:
        return None
    return {"gap": fmean(r) - fmean(n), "lo": min(r) - max(n), "hi": max(r) - min(n), "n": (len(r), len(n)),
            "source": src, "key": key}  # fmt: skip


def from_final(
    final: dict[tuple[str, str], Point], pair: str, method: str
) -> Point | None:
    """A re-emitted final.json cell, or None (logged as WAITING) while it lacks a unit or is the label contrast."""
    c = final.get((pair, method))
    if c is None or "unit" not in c:
        WAITING.append(f"{pair} {method}: no cell with a 'unit' field in {FINAL}")
        return None
    if "contrast" in f"{c['form']} {c['unit']}":
        WAITING.append(
            f"{pair} {method}: still the label contrast ({c['form']!r}, {c['unit']!r})"
        )
        return None
    key = f'.cells[] | select(.pair=="{pair}" and .method=="{method}")'
    return reading(c["reuse"], c["new"], FINAL, key)


def from_known3(
    known3: dict[tuple[str, str], Point], pair: str, method: str
) -> Point | None:
    c = known3[(pair, method)]
    return reading(
        c["reuse"],
        c["new"],
        KNOWN3,
        f'.cells[] | select(.pair=="{pair}" and .method=="{method}")',
    )


def pairs() -> list[Point]:
    final = {(c["pair"], c["method"]): c for c in load(FINAL)["cells"]}
    known3 = {(c["pair"], c["method"]): c for c in load(KNOWN3)["cells"]}
    matched = {
        (c["pair"], c["reading"]): c
        for c in load(MATCHED)["cells"]
        if c["model"] == "67M"
    }
    out = []
    for pair in dict.fromkeys(p for p, _ in final):
        c = final[(pair, "prior")]
        rec: Point = {
            "pair": pair,
            "name": c["pair_name"],
            "group": c["group"],
            "perkey": from_final(final, pair, PERKEY),
        }
        if (pair, "share") in known3:
            rec |= {
                m: from_known3(known3, pair, k)
                for m, k in (
                    ("share", "share"),
                    ("steering", "steer"),
                    ("prior", "prior"),
                )
            }
        else:
            rec |= {"share": from_final(final, pair, "share"), "steering": from_final(final, pair, POOLED),
                    "prior": reading(c["reuse"], c["new"], FINAL, f'.cells[] | select(.pair=="{pair}" and .method=="prior")')}  # fmt: skip
        for m in ("share", "steering"):
            mc = matched.get((pair, m))
            if mc is not None and (m == "share" or mc["form"] == "pooled"):
                v, key = (
                    mc["matched_values"],
                    f'.cells[] | select(.model=="67M" and .pair=="{pair}" and .reading=="{m}") | .matched_values',
                )
                rec[f"{m}_matched"] = reading(v["reuse"], v["new"], MATCHED, key)
        out.append(rec)
    return out


def e199() -> Point:
    s = load(E199)["inputs"]
    rec: Point = {
        "pair": "e199_neighbour_prior_cipher",
        "name": E199_NAME,
        "group": "built, prior and steering disagree",
    }
    rec |= {
        m: reading(s[m][0], s[m][1], E199, f".inputs.{m}")
        for m in ("share", "steering")
    }
    rec["prior"] = {"gap": s["prior_gap"], "source": E199, "key": ".inputs.prior_gap"}
    return rec


def row(
    fig: str,
    panel: str,
    p: Point,
    series: str,
    x: Point | None,
    y: Point | None,
    with_x: bool,
) -> Point:
    out = {
        "figure": fig,
        "panel": panel,
        "pair": p["pair"],
        "pair_name": p["name"],
        "group": p["group"],
        "series": series,
    }
    y = y or {}
    out |= {
        "y_share_gap": y.get("gap"),
        "y_lo": y.get("lo"),
        "y_hi": y.get("hi"),
        "y_source": y.get("source"),
        "y_key": y.get("key"),
    }
    if with_x:
        x = x or {}
        out |= {
            "x_gap": x.get("gap"),
            "x_source": x.get("source"),
            "x_key": x.get("key"),
        }
    return out


def fig3a_rows(recs: list[Point]) -> list[Point]:
    return [row("3a", "share", r, s, None, r[k], False) for r in recs
            for k, s in (("share", "unscaled"), ("share_matched", "matched")) if k in r]  # fmt: skip


def fig4_rows(recs: list[Point], extra: Point) -> list[Point]:
    out = []
    for x in ("prior", "steering"):
        for r in recs:
            out.append(row("4", x, r, "unscaled", r[x], r["share"], True))
            if (
                "share_matched" in r
                and (mx := r.get(f"{x}_matched", r[x] if x == "prior" else None))
                is not None
            ):
                out.append(row("4", x, r, "matched", mx, r["share_matched"], True))
        out.append(row("4", x, extra, "unscaled", extra[x], extra["share"], True))
    return out


def write(name: str, rows: list[Point]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    forms = {
        k: v for k, v in FORMS.items() if name == FIG4 or not k.startswith("x_gap")
    }
    (OUT / name).with_suffix(".forms.json").write_text(
        json.dumps(forms, indent=1) + "\n"
    )


def diff(name: str, rows: list[Point]) -> None:
    """Old (method_table/figures) against new, per row and plotted column, largest move first."""
    with (OLD / name).open() as f:
        old = {(r["panel"], r["pair"], r["series"]): r for r in csv.DictReader(f)}
    lines = []
    for r in rows:
        o = old.pop((r["panel"], r["pair"], r["series"]), {})
        for col in ("y_share_gap", "x_gap"):
            if col not in r:
                continue
            a, b = float(o[col]) if o.get(col) else None, r[col]
            if a is None or b is None or abs(a - b) > 1e-9:
                d = abs(a - b) if a is not None and b is not None else float("inf")
                lines.append(
                    (
                        d,
                        f"{r['pair']:<28} {r['panel']:<9} {r['series']:<9} {col:<12} {a!s:>22} {b!s:>22}",
                    )
                )
    print(
        f"\n{name}: {len(lines)} changed values ({len(old)} old rows with no new row: {sorted(old)})"
    )
    print(
        f"{'pair':<28} {'panel':<9} {'series':<9} {'column':<12} {'old':>22} {'new':>22}"
    )
    for _, line in sorted(lines, reverse=True):
        print(line)


def against_share(steer: Point | None, share: Point | None) -> str:
    if steer is None or share is None:
        return "missing"
    if not (steer["lo"] > 0 or steer["hi"] < 0):
        return "no call"
    if not (share["lo"] > 0 or share["hi"] < 0):
        return "apart where the share is not"
    return (
        "same direction as the share"
        if (steer["gap"] > 0) == (share["gap"] > 0)
        else "opposite to the share"
    )


def reversals(recs: list[Point]) -> None:
    final = {(c["pair"], c["method"]): c for c in load(FINAL)["cells"]}
    for group in ("built", "twin"):
        print(f"\nsteering against the share, {group} pairs")
        for form in ("steering", "perkey", "as drawn before"):
            calls = {}
            for r in (r for r in recs if r["group"] == group):
                old = final.get((r["pair"], "steering"))
                calls[r["pair"]] = (
                    old["against_share"]
                    if form == "as drawn before" and old
                    else against_share(r.get(form), r["share"])
                )
            rev = sorted(p for p, c in calls.items() if c == "opposite to the share")
            label = {
                "steering": "E143 pooled",
                "perkey": "per-key",
                "as drawn before": "old final.json (mixed forms)",
            }[form]
            print(
                f"  {label:<30} reverse {len(rev)} of {len(calls)}: {rev}; missing: {sorted(p for p, c in calls.items() if c == 'missing')}"
            )


def main() -> None:
    recs = pairs()
    a, b = fig3a_rows(recs), fig4_rows(recs, e199())
    write(FIG3A, a)
    write(FIG4, b)
    diff(FIG3A, a)
    diff(FIG4, b)
    reversals(recs)
    print("\nseeds a side per pair (share; steering): " + "; ".join(
        f"{r['pair']} {r['share'] and r['share']['n']} {r['steering'] and r['steering']['n']}" for r in recs))  # fmt: skip
    if WAITING:
        print(
            "\nWAITING (rerun once these land):\n  "
            + "\n  ".join(dict.fromkeys(WAITING))
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
