"""Compute CometKiwi (Unbabel/wmt22-cometkiwi-da, reference-free) on FLORES / WMT24++ predictions.

Reference-free quality estimation: scores each translation against its source only, so it is
less biased against paraphrastic translations than reference-based metrics.

Run in the chindamt-comet environment: unbabel-comet 2.2.7 requires transformers<5.

Usage:
  python -m src.eval.compute_cometkiwi --dataset flores --models <generator>
  python -m src.eval.compute_cometkiwi --dataset wmt24pp --models <generator>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def strip_prefix(s: str) -> str:
    for p in ("TH: ", "EN: ", "Thai: ", "English: ", "TH:", "EN:", "Thai:", "English:"):
        if s.startswith(p):
            return s[len(p):].strip()
    return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["flores", "wmt24pp"])
    ap.add_argument("--models", nargs="+", required=True, help="Generator names, read from predictions/<generator>-<dataset>-plain/")
    ap.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")
    args = ap.parse_args()

    dataset_dir = PROJECT / "data/eval" / args.dataset
    pred_dir = PROJECT / "predictions"

    suite = [json.loads(l) for l in open(dataset_dir / "plain.jsonl")]
    by_instr = {r["instruction"]: r for r in suite}

    from comet import download_model, load_from_checkpoint
    print("Loading CometKiwi (Unbabel/wmt22-cometkiwi-da)...")
    ckpt = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(ckpt)

    results = []
    models_to_run = args.models
    for gen in models_to_run:
        pred_file = pred_dir / f"{gen}-{args.dataset}-plain" / "model_outputs.json"
        if not pred_file.exists():
            print(f"[skip] {gen}: no predictions")
            continue
        preds = json.load(pred_file.open())
        for direction in ("en->th", "th->en"):
            data = []
            for p in preds:
                rec = by_instr.get(p["instruction"])
                if rec is None or rec["direction"] != direction:
                    continue
                data.append({
                    "src": rec["source_text"],
                    "mt": strip_prefix(p["output"]),
                })
            if not data:
                continue
            out = model.predict(data, batch_size=32, gpus=1, progress_bar=False)
            score = out.system_score * 100
            results.append({"model": gen, "direction": direction, "n": len(data), "cometkiwi": round(score, 2)})
            print(f"  {gen:<36} {direction:<7} n={len(data):>4} CometKiwi={score:>6.2f}")

    out_csv = PROJECT / f"output/metrics/{args.dataset}-cometkiwi.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.append and out_csv.exists():
        with out_csv.open("r", newline="") as f:
            existing = [row for row in csv.DictReader(f) if row["model"] not in {r["model"] for r in results}]
        results = existing + results
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "direction", "n", "cometkiwi"])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved to {out_csv} ({len(results)} rows total)")


if __name__ == "__main__":
    main()
