"""Comparison table across both networks. Shared and frozen.

Reads every results/<network>.csv (one row per evaluated run) and prints them in one
table, best survival first. Used on comparison day and for Report 2's results table.

    python -m common.compare               all rows
    python -m common.compare --split test  held-out ground only

Uses: the CSV files written by evaluate.evaluate_and_record.
"""
import argparse
import csv

from common.evaluate import RESULTS

COLS = ["network", "run", "split", "seed", "steps", "params", "survival_pct", "survival_ci",
        "falls_by_reason", "toe_peak_bw", "mean_K", "mean_theta_deg", "mean_return",
        "ceiling_pct", "minus_reference"]


def main():
    """Print the table; say so if there are no results yet."""
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=None)
    a = p.parse_args()
    rows = []
    for f in sorted(RESULTS.glob("*.csv")):
        if f.name.startswith("reference_"):
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                r["network"] = f.stem
                if a.split is None or r.get("split") == a.split:
                    rows.append(r)
    if not rows:
        print("No results yet. Each run writes a row to results/<network>.csv.")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in COLS}
    print("  ".join(c.ljust(widths[c]) for c in COLS))
    for r in sorted(rows, key=lambda r: -float(r.get("survival_pct", 0))):
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in COLS))


if __name__ == "__main__":
    main()
