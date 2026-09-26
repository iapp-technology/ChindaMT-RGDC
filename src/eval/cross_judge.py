"""Re-judge a primary-judge cell with a second judge, as in the cross-judge tables of the paper.

Reads the pairs and primary verdicts that AlpacaEval cached in a result folder of
scripts/04_evaluate.sh, sends every pair to a second OpenAI-compatible judge with the same rubric,
and reports per-pair agreement, Cohen's kappa, and the win rate of the target under both judges.

The protocol is fixed to match the paper:
- the system and user parts of config/evaluation/judge_prompt.txt go out as one user message
- the order of the two outputs is flipped per pair from one random stream seeded with --seed
- temperature 0 and 1024 new tokens with thinking off; GPT-OSS models get low reasoning effort and
  1536 tokens instead
- a reply that does not parse is retried with a format reminder, then read from its prose
- win rate counts a tie as half a win; AlpacaEval puts the target in output_2

Results go to <cell>/cross_judge/<judge>.csv (per pair) and <judge>.json (summary).

Usage:
  python -m src.eval.cross_judge --cell output/<judge>/chindamt-4b-vs-hymt-1.5-7b-own/coreeval-plain \\
      --base-url http://localhost:8001/v1 --api-key <your-api-key> --model gpt-oss-120b
  python -m src.eval.cross_judge --summary output/<judge>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path

from src.eval.judge_id import judge_id

ANNOTATIONS = "annotations_seed0_judge_config.resolved.json"
REMINDER_PROMPT = (
    "\n\nYour previous response was not a valid Python list. Return ONLY the "
    "raw Python list, with no reasoning, no markdown fences, no explanation:\n"
    "[{'model': 'model_1', 'rank': 1}, {'model': 'model_2', 'rank': 2}]\n"
    "Replace the ranks 1/2 with your actual preference."
)


def split_template(template: str) -> tuple[str, str]:
    """Return the system and user parts of the ChatML-tagged judge prompt."""
    m_sys = re.search(r"<\|im_start\|>system\s*(.*?)<\|im_end\|>", template, re.DOTALL)
    m_usr = re.search(r"<\|im_start\|>user\s*(.*?)<\|im_end\|>", template, re.DOTALL)
    if m_sys and m_usr:
        return m_sys.group(1).strip(), m_usr.group(1).strip()
    return "", template


def parse_ranking(raw: str) -> str | None:
    """Parse the ranking list into 'model_1', 'model_2', 'tie', or None."""
    if not raw:
        return None
    txt = re.sub(r"^```(?:python|json)?\s*", "", raw.strip())
    txt = re.sub(r"\s*```$", "", txt)
    pairs = re.findall(r"['\"]model['\"]\s*:\s*['\"](model_\d)['\"][^}]*?['\"]rank['\"]\s*:\s*(\d)", txt)
    if len(pairs) < 2:
        return None
    ranks = {m: int(r) for m, r in pairs}
    if ranks.get("model_1") == ranks.get("model_2"):
        return "tie"
    return "model_1" if ranks.get("model_1", 2) < ranks.get("model_2", 2) else "model_2"


def parse_ranking_prose(raw: str) -> str | None:
    """Read a preference from prose, used only when the reply is not a ranking list."""
    if not raw:
        return None
    txt = raw.lower()
    patterns_1 = [
        r"model[\s_]*1\s+(?:is|wins|would win|is preferred|is better|is the better|is the best|ranks first)",
        r"(?:i\s+)?(?:prefer|choose|select)\s+model[\s_]*1",
        r"model[\s_]*1\s+rank(?:s|ed)?\s+(?:1|first|higher)",
        r"\brank\s*1\s*[:=]\s*model[\s_]*1",
    ]
    patterns_2 = [
        r"model[\s_]*2\s+(?:is|wins|would win|is preferred|is better|is the better|is the best|ranks first)",
        r"(?:i\s+)?(?:prefer|choose|select)\s+model[\s_]*2",
        r"model[\s_]*2\s+rank(?:s|ed)?\s+(?:1|first|higher)",
        r"\brank\s*1\s*[:=]\s*model[\s_]*2",
    ]
    tie_patterns = [
        r"\b(?:tied|equivalent|equally (?:good|preferred))\b",
        r"(?:cannot|can(?:'|no)t)\s+(?:decide|choose)",
        r"both\s+(?:are\s+)?(?:equally|comparable|similar)",
    ]
    hit_1 = any(re.search(p, txt) for p in patterns_1)
    hit_2 = any(re.search(p, txt) for p in patterns_2)
    if hit_1 and not hit_2:
        return "model_1"
    if hit_2 and not hit_1:
        return "model_2"
    if any(re.search(p, txt) for p in tie_patterns):
        return "tie"
    return None


def call_judge(client, model: str, prompt: str, max_tokens: int = 1024, retries: int = 5) -> str:
    """Return the judge reply, or a string starting with __ERROR__ after the last retry."""
    is_reasoning = "gpt-oss" in model.lower() or model.lower().startswith(("o1", "o3"))
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
                  max_tokens=max(max_tokens, 1536) if is_reasoning else max_tokens)
    # Templates without a thinking switch ignore enable_thinking, so it only affects models that think by default
    kwargs["extra_body"] = ({"reasoning_effort": "low"} if is_reasoning
                            else {"chat_template_kwargs": {"enable_thinking": False}})
    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs).choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                return f"__ERROR__ {type(e).__name__}: {e}"
            time.sleep(2.0 * (2 ** attempt))
    return ""


def cohens_kappa(paired: list[tuple[float, float]]) -> float:
    cats = [1.0, 1.5, 2.0]
    n = len(paired)
    po = sum(1 for a, b in paired if a == b) / n
    pe = sum((sum(1 for a, _ in paired if a == c) / n) * (sum(1 for _, b in paired if b == c) / n) for c in cats)
    if abs(1 - pe) < 1e-9:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def win_rate(prefs: list[float]) -> float:
    return sum(1 if p == 2.0 else 0.5 if p == 1.5 else 0 for p in prefs) / len(prefs) * 100 if prefs else 0.0


def bootstrap_se(paired: list[tuple[float, float]], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(paired)
    first, second = [], []
    for _ in range(n_boot):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        first.append(win_rate([p for p, _ in sample]))
        second.append(win_rate([c for _, c in sample]))

    def std(xs):
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    return std(first), std(second)


def judge_cell(args) -> int:
    cell = args.cell
    suite, _, split = cell.name.partition("-")
    suite_file = args.suite_file or Path("data/eval") / suite / f"{split}.jsonl"
    name = judge_id(args.model, prompt=args.prompt_template)
    out_csv = cell / "cross_judge" / f"{name}.csv"
    if out_csv.exists() and not args.force:
        print(f"[skip] {out_csv} exists")
        return 0

    random.seed(args.seed)
    annotations = json.loads((cell / ANNOTATIONS).read_text(encoding="utf-8"))
    direction = {r["instruction"]: r["direction"] for r in map(json.loads, open(suite_file, encoding="utf-8"))}
    items = [{**a, "_direction": direction[a["instruction"]]} for a in annotations
             if a.get("instruction") in direction and a.get("preference") in (1.0, 1.5, 2.0)]
    by_dir = {d: [x for x in items if x["_direction"] == d] for d in ("en->th", "th->en")}
    if args.n_samples:
        items = (random.sample(by_dir["en->th"], min((args.n_samples + 1) // 2, len(by_dir["en->th"])))
                 + random.sample(by_dir["th->en"], min(args.n_samples // 2, len(by_dir["th->en"]))))
    else:
        items = by_dir["en->th"] + by_dir["th->en"]
    flips = [random.random() < 0.5 for _ in items]

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    system_text, user_text = split_template(args.prompt_template.read_text(encoding="utf-8"))
    print(f"{cell}: {len(items)} pairs, judge {args.model}")

    def judge_one(i: int) -> dict:
        item, flip = items[i], flips[i]
        out_a, out_b = (item["output_2"], item["output_1"]) if flip else (item["output_1"], item["output_2"])
        user = user_text.replace("{instruction}", item["instruction"]).replace("{output_1}", out_a).replace("{output_2}", out_b)
        prompt = f"{system_text}\n\n{user}" if system_text else user
        winner, method, raw, raw2 = None, "FAILED", "", ""
        for attempt in range(6):
            if attempt:
                time.sleep(min(2 ** attempt, 20))
            raw = call_judge(client, args.model, prompt)
            if raw.startswith("__ERROR__"):
                continue
            winner = parse_ranking(raw)
            if winner:
                method = "strict" if attempt == 0 else f"strict-attempt{attempt + 1}"
                break
            raw2 = call_judge(client, args.model, prompt + REMINDER_PROMPT)
            winner = None if raw2.startswith("__ERROR__") else parse_ranking(raw2)
            if winner:
                method = "retry-reminder"
                break
            winner = parse_ranking_prose(raw)
            if winner:
                method = "prose-fallback"
                break
        pref = {"model_1": 2.0 if flip else 1.0, "model_2": 1.0 if flip else 2.0, "tie": 1.5}.get(winner)
        return {"direction": item["_direction"], "primary_pref": item["preference"], "cross_pref": pref,
                "parse_method": method, "flipped": flip, "cross_raw": raw[:400], "cross_raw2": raw2[:200]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(judge_one, range(len(items))))

    failed = sum(1 for r in rows if r["cross_pref"] is None)
    if failed > args.max_fail_rate * len(rows):
        print(f"ABORT: {failed}/{len(rows)} pairs got no verdict; nothing written, rerun when the judge is stable")
        return 2
    paired = [(r["primary_pref"], r["cross_pref"]) for r in rows if r["cross_pref"] is not None]
    se_primary, se_cross = bootstrap_se(paired) if len(paired) >= 10 else (float("nan"), float("nan"))
    summary = {
        "comparison": cell.parent.name, "cell": cell.name, "judge": args.model, "judge_id": name,
        "n_valid": len(paired), "n_total": len(rows),
        "agreement": round(sum(1 for p, c in paired if p == c) / len(paired) * 100, 2),
        "kappa": round(cohens_kappa(paired), 4),
        "primary_wr": round(win_rate([p for p, _ in paired]), 2), "primary_se": round(se_primary, 2),
        "cross_wr": round(win_rate([c for _, c in paired]), 2), "cross_se": round(se_cross, 2),
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"agreement {summary['agreement']}%  kappa {summary['kappa']}  "
          f"primary WR {summary['primary_wr']}  {args.model} WR {summary['cross_wr']}  -> {out_csv}")
    return 0


def summarize(root: Path) -> int:
    cells: dict[tuple[str, str], dict] = {}
    judges: list[str] = []
    for path in sorted(root.glob("*/*/cross_judge/*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        row = cells.setdefault((s["comparison"], s["cell"]), {"primary": s["primary_wr"]})
        row[s["judge"]] = s["cross_wr"]
        if s["judge"] not in judges:
            judges.append(s["judge"])
    if not cells:
        print(f"no cross-judge results under {root}")
        return 1
    print("| Comparison | Cell | Primary | " + " | ".join(judges) + " | Mean |")
    print("|---|---|---:|" + "---:|" * len(judges) + "---:|")
    for (comparison, cell), row in sorted(cells.items()):
        values = [row["primary"]] + [row.get(j) for j in judges]
        shown = [f"{v:.1f}" if v is not None else "-" for v in values]
        mean = sum(v for v in values if v is not None) / sum(v is not None for v in values)
        print(f"| {comparison} | {cell} | " + " | ".join(shown) + f" | {mean:.1f} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", type=Path, help="Result folder of scripts/04_evaluate.sh, named <suite>-<split>")
    ap.add_argument("--summary", type=Path, help="Print the WR table for every re-judged cell under this folder")
    ap.add_argument("--suite-file", type=Path, help="Suite file for directions (default: data/eval/<suite>/<split>.jsonl)")
    ap.add_argument("--prompt-template", type=Path, default=Path("config/evaluation/judge_prompt.txt"))
    ap.add_argument("--model", default=os.environ.get("CROSS_JUDGE_MODEL"))
    ap.add_argument("--base-url", default=os.environ.get("CROSS_JUDGE_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("CROSS_JUDGE_API_KEY", "EMPTY"))
    ap.add_argument("--n-samples", type=int, help="Judge a direction-balanced sample instead of every pair")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-fail-rate", type=float, default=0.05)
    ap.add_argument("--force", action="store_true", help="Judge again even if results exist")
    args = ap.parse_args()
    if args.summary:
        return summarize(args.summary)
    if not (args.cell and args.model and args.base_url):
        ap.error("--cell, --model, and --base-url are required (or set CROSS_JUDGE_MODEL and CROSS_JUDGE_BASE_URL)")
    return judge_cell(args)


if __name__ == "__main__":
    raise SystemExit(main())
