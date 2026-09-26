"""Exit 0 when a judged cell is complete: every pair has a verdict and the leaderboard counts them all.

A judge outage in the middle of a run leaves pairs without a verdict, and AlpacaEval does not retry
them when the run resumes, so an incomplete cell has to be judged again from scratch.

Usage:
  python -m src.eval.check_cell <result folder> [--expected N]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cell", type=Path, help="Folder with annotations.json and leaderboard.csv")
    ap.add_argument("--expected", type=int, default=0, help="Minimum number of judged pairs")
    args = ap.parse_args()
    try:
        annotations = json.loads((args.cell / "annotations.json").read_text(encoding="utf-8"))
        with open(args.cell / "leaderboard.csv", newline="", encoding="utf-8") as f:
            n_total = int(float(next(csv.DictReader(f))["n_total"]))
    except (OSError, StopIteration, KeyError, ValueError):
        print(f"{args.cell}: no complete result")
        return 1
    missing = sum(
        1 for r in annotations
        if not isinstance(r.get("preference"), (int, float)) or math.isnan(r["preference"])
    )
    print(f"{args.cell}: pairs={len(annotations)} missing_verdicts={missing} n_total={n_total}")
    return 0 if missing == 0 and n_total == len(annotations) and n_total >= args.expected else 1


if __name__ == "__main__":
    sys.exit(main())
