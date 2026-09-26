"""Compute MetricX-24 (reference-based) on FLORES / WMT24++ predictions.

Uses the official google-research/metricx24 implementation (MT5ForRegression).
The model output is an ERROR SCORE in [0, 25] where 0 = perfect. For a
higher-is-better view alongside COMET and GEMBA, it also reports
(25 - error) as `metricx_quality`.

H100 setup: batch_size=64 with bfloat16 fits comfortably (~25GB peak),
~3000 rows/min on a single H100.

Usage:
  python -m src.eval.compute_metricx --dataset flores --models <generator>
  python -m src.eval.compute_metricx --dataset wmt24pp --models <gen> --append
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]

# Add the cloned metricx repo to sys.path so we can import its custom MT5ForRegression
METRICX_REPO = Path("/tmp/metricx")
if METRICX_REPO.exists():
    sys.path.insert(0, str(METRICX_REPO))


def strip_prefix(s: str) -> str:
    for p in ("TH: ", "EN: ", "Thai: ", "English: ", "TH:", "EN:", "Thai:", "English:"):
        if s.startswith(p):
            return s[len(p):].strip()
    return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["flores", "wmt24pp"])
    ap.add_argument("--models", nargs="+", required=True, help="Generator names, read from predictions/<generator>-<dataset>-plain/")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="H100 80GB easily handles batch 64 with bf16 (uses ~25GB peak).")
    ap.add_argument("--max-input-length", type=int, default=1536)
    args = ap.parse_args()

    dataset_dir = PROJECT / "data/eval" / args.dataset
    pred_dir = PROJECT / "predictions"
    suite = [json.loads(l) for l in open(dataset_dir / "plain.jsonl")]
    by_inst = {r["instruction"]: r for r in suite}

    import torch
    from transformers import AutoTokenizer
    from metricx24 import models

    model_id = "google/metricx-24-hybrid-large-v2p6-bfloat16"
    tokenizer_id = "google/mt5-xl"
    print(f"Loading MetricX-24 from {model_id} ...", flush=True)
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, legacy=False)
    model = models.MT5ForRegression.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Loaded in {time.time()-t_load:.1f}s on {device}", flush=True)

    @torch.no_grad()
    def score_batch(items):
        # MetricX reference-based input format: "source: <src> candidate: <mt> reference: <ref>"
        inputs = [
            f"source: {it['src']} candidate: {it['mt']} reference: {it['ref']}"
            for it in items
        ]
        enc = tokenizer(
            inputs,
            max_length=args.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        # The metricx custom model strips the trailing EOS in its training pipeline; mirror that
        # by truncating input_ids/attention_mask by 1 only when they currently end with eos.
        eos_id = tokenizer.eos_token_id
        last_col_is_eos = (enc["input_ids"][:, -1] == eos_id).all().item() if eos_id is not None else False
        if last_col_is_eos:
            enc["input_ids"] = enc["input_ids"][:, :-1]
            enc["attention_mask"] = enc["attention_mask"][:, :-1]
        out = model(**enc)
        # MT5ForRegression returns a regression head output (single scalar per example)
        # Output is in `.logits` or `.prediction` depending on the model class
        if hasattr(out, "predictions"):
            scores = out.predictions
        elif hasattr(out, "logits"):
            scores = out.logits
        else:
            scores = out
        scores = scores.detach().float().cpu().tolist()
        # Flatten if nested
        if isinstance(scores, list) and scores and isinstance(scores[0], list):
            scores = [s[0] for s in scores]
        return scores

    results = []
    models_to_run = args.models
    for gen in models_to_run:
        pred_file = pred_dir / f"{gen}-{args.dataset}-plain" / "model_outputs.json"
        if not pred_file.exists():
            print(f"[skip] {gen}: no predictions")
            continue
        preds = json.load(pred_file.open())
        for direction in ("en->th", "th->en"):
            items = []
            for p in preds:
                rec = by_inst.get(p["instruction"])
                if rec is None or rec["direction"] != direction:
                    continue
                items.append({
                    "src": rec["source_text"],
                    "mt": strip_prefix(p["output"]),
                    "ref": rec["reference_text"],
                })
            if not items:
                continue
            scores = []
            t0 = time.time()
            for i in range(0, len(items), args.batch_size):
                batch = items[i:i + args.batch_size]
                batch_scores = score_batch(batch)
                scores.extend(batch_scores)
                if (i // args.batch_size + 1) % 5 == 0:
                    dt = time.time() - t0
                    rate = (i + len(batch)) / dt
                    print(f"    {gen} {direction}: {i+len(batch)}/{len(items)}  rate={rate:.1f}/s", flush=True)
            avg = sum(scores) / max(len(scores), 1)
            quality = max(0.0, 25 - avg)
            dt = time.time() - t0
            print(f"  {gen:<40} {direction:<7} n={len(scores):>4}  error={avg:>5.2f}  quality={quality:>5.2f}  {dt:.0f}s", flush=True)
            results.append({
                "model": gen,
                "direction": direction,
                "n": len(scores),
                "metricx_error": round(avg, 2),
                "metricx_quality": round(quality, 2),
            })

    out_csv = PROJECT / f"output/metrics/{args.dataset}-metricx.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.append and out_csv.exists():
        with out_csv.open("r", newline="") as f:
            existing = [row for row in csv.DictReader(f) if row["model"] not in {r["model"] for r in results}]
        results = existing + results
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "direction", "n", "metricx_error", "metricx_quality"])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved to {out_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
