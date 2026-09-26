"""GEMBA-DA and GEMBA-MQM scoring with an OpenAI-compatible judge (the paper used Qwen3.6-35B-A3B-FP8).

GEMBA-DA: reference-based direct assessment, 0-100 scalar.
GEMBA-MQM: reference-free error-category annotation (Accuracy/Fluency/Style/Terminology) +
           severity (Minor/Major/Critical) -> aggregated score.

Reads judge config from $OPENAI_BASE_URL / $OPENAI_API_KEY / $JUDGE_MODEL_NAME (.env).

Usage:
  python -m src.eval.compute_gemba --dataset flores --mode da --models <generator>
  python -m src.eval.compute_gemba --dataset flores --mode mqm --models <generator>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def strip_prefix(s: str) -> str:
    for p in ("TH: ", "EN: ", "Thai: ", "English: ", "TH:", "EN:", "Thai:", "English:"):
        if s.startswith(p):
            return s[len(p):].strip()
    return s.strip()


DA_PROMPT = """You are evaluating the quality of a machine translation.

Source ({src_lang}):
{src}

Reference translation ({tgt_lang}):
{ref}

Candidate translation ({tgt_lang}):
{mt}

Rate the candidate's overall translation quality on a scale from 0 to 100, where:
- 0-30: major meaning errors, poor fluency, unusable
- 30-60: partial meaning preserved, noticeable errors
- 60-80: good translation, minor issues
- 80-100: excellent translation, near-perfect

Consider accuracy, fluency, and faithfulness to the source. The reference is one acceptable translation; equally valid alternatives should not be penalized.

Output only a single integer between 0 and 100, nothing else.""".strip()


MQM_PROMPT = """You are performing a Multidimensional Quality Metrics (MQM) evaluation.

Source ({src_lang}):
{src}

Candidate translation ({tgt_lang}):
{mt}

Identify translation errors in the candidate. For each error, output one line in the exact format:
  CATEGORY / SEVERITY / short-description

Categories: Accuracy, Fluency, Terminology, Style, Locale, Other
Severities: Minor, Major, Critical

If there are no errors, output the single line:
  NO_ERRORS

Do not include any text other than the error lines.""".strip()


MQM_PENALTY = {"Minor": 1, "Major": 5, "Critical": 10}


def score_mqm_response(resp: str) -> float:
    """Aggregate MQM errors into a 0-100 quality score.
    No errors = 100; each error subtracts its severity penalty (capped at 0).
    """
    if "NO_ERRORS" in resp.upper():
        return 100.0
    total = 0
    for line in resp.splitlines():
        parts = [x.strip() for x in line.split("/")]
        if len(parts) < 2:
            continue
        sev = parts[1]
        for k in MQM_PENALTY:
            if k.lower() in sev.lower():
                total += MQM_PENALTY[k]
                break
    return max(0.0, 100.0 - total)


def score_da_response(resp: str) -> float | None:
    """Extract the first integer 0-100 from the judge response."""
    nums = re.findall(r"\b(\d{1,3})\b", resp)
    for n in nums:
        v = int(n)
        if 0 <= v <= 100:
            return float(v)
    return None


def call_judge_one(client, model_name, prompt, timeout_s=120, max_tokens=2048):
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                timeout=timeout_s,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            if attempt == 4:
                return f"__ERROR__{e}"
            time.sleep(min(2 ** attempt, 8))
    return ""


def call_judge_with_score_retry(client, model_name, prompt, mode, max_attempts=4):
    """Call judge and retry with progressively stricter prompts if parse fails."""
    resp = call_judge_one(client, model_name, prompt)
    if resp.startswith("__ERROR__"):
        return resp, None
    score = score_da_response(resp) if mode == "da" else score_mqm_response(resp)
    if score is not None and mode == "da":
        return resp, score
    if mode == "mqm" and (score is not None and (score > 0 or "NO_ERRORS" in resp.upper() or any(s.lower() in resp.lower() for s in ("minor", "major", "critical")))):
        return resp, score
    # Parse failed -> retry with stricter follow-up
    for attempt in range(max_attempts):
        if mode == "da":
            retry_prompt = prompt + "\n\nReminder: output ONLY a single integer between 0 and 100. No explanation, no commentary."
        else:
            retry_prompt = prompt + "\n\nReminder: output ONLY the error lines (CATEGORY / SEVERITY / desc), or the single token NO_ERRORS. No commentary, no headers."
        resp = call_judge_one(client, model_name, retry_prompt)
        if resp.startswith("__ERROR__"):
            continue
        s = score_da_response(resp) if mode == "da" else score_mqm_response(resp)
        if s is not None and (mode == "da" or s > 0 or "NO_ERRORS" in resp.upper() or any(k.lower() in resp.lower() for k in ("minor", "major", "critical"))):
            return resp, s
    # Final fallback: ask judge directly for just the score
    if mode == "da":
        final_prompt = "Given this evaluation context, what is your 0-100 quality score? Output ONLY the integer.\n\n" + prompt
        resp = call_judge_one(client, model_name, final_prompt, max_tokens=64)
        s = score_da_response(resp)
        if s is not None:
            return resp, s
    return resp, None


def run_one(gen_name, dataset, dataset_dir, pred_dir, mode, client, model_name, max_workers):
    pred_file = pred_dir / f"{gen_name}-{dataset}-plain" / "model_outputs.json"
    if not pred_file.exists():
        print(f"[skip] {gen_name}: no predictions")
        return []

    suite = [json.loads(l) for l in open(dataset_dir / "plain.jsonl")]
    by_instr = {r["instruction"]: r for r in suite}
    preds = json.load(pred_file.open())

    by_dir = {"en->th": [], "th->en": []}
    for p in preds:
        rec = by_instr.get(p["instruction"])
        if rec is None:
            continue
        by_dir[rec["direction"]].append((rec, strip_prefix(p["output"])))

    template = DA_PROMPT if mode == "da" else MQM_PROMPT
    results = []
    for direction, items in by_dir.items():
        if not items:
            continue
        src_lang = "English" if direction == "en->th" else "Thai"
        tgt_lang = "Thai" if direction == "en->th" else "English"

        prompts = []
        for rec, mt in items:
            fmt = {"src_lang": src_lang, "tgt_lang": tgt_lang,
                   "src": rec["source_text"], "ref": rec["reference_text"], "mt": mt}
            prompts.append(template.format(**fmt))

        scores = []
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i, (resp, s) in enumerate(pool.map(lambda p: call_judge_with_score_retry(client, model_name, p, mode), prompts), 1):
                if s is not None:
                    scores.append(s)
                if i % 100 == 0:
                    print(f"    {direction}: {i}/{len(prompts)}  parsed={len(scores)}  avg={sum(scores)/max(len(scores),1):.1f}", flush=True)
        dur = time.time() - t0
        mean_score = sum(scores) / max(len(scores), 1)
        coverage = 100 * len(scores) / max(len(prompts), 1)
        print(f"  {gen_name:<36} {direction:<7} n={len(scores):>4}/{len(prompts):>4}  ({coverage:.1f}%)  mean={mean_score:>6.2f}  {dur:.0f}s", flush=True)
        results.append({"model": gen_name, "direction": direction, "n": len(scores), "score": round(mean_score, 2)})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["flores", "wmt24pp"])
    ap.add_argument("--mode", required=True, choices=["da", "mqm"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--models", nargs="+", required=True, help="Generator names, read from predictions/<generator>-<dataset>-plain/")
    ap.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    model_name = os.environ["JUDGE_MODEL_NAME"]
    print(f"Judge: {model_name} @ {os.environ['OPENAI_BASE_URL']}  (mode={args.mode})")

    dataset_dir = PROJECT / "data/eval" / args.dataset
    pred_dir = PROJECT / "predictions"

    all_results = []
    models_to_run = args.models
    for gen in models_to_run:
        all_results.extend(run_one(gen, args.dataset, dataset_dir, pred_dir, args.mode, client, model_name, args.workers))

    out_csv = PROJECT / f"output/metrics/{args.dataset}-gemba-{args.mode}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.append and out_csv.exists():
        with out_csv.open("r", newline="") as f:
            existing = [row for row in csv.DictReader(f) if row["model"] not in {r["model"] for r in all_results}]
        all_results = existing + all_results
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "direction", "n", "score"])
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved to {out_csv} ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
