"""Convert a parallel corpus into the pool's minimal ShareGPT files, one per direction.

Input: a UTF-8 CSV or TSV with a header and two columns holding the English and Thai segments
(column names given by --en-col and --th-col). Rows are kept as they are, including empty sides.
Validated byte-for-byte against the paper's pool on wikimedia (32,999 rows per direction), except
one row whose missing English cell the original tool rendered as the string "None" on the source
side in both directions; this converter swaps sides faithfully instead. Output: two folders named
{name}_en2th_minimal_zero-shot and {name}_th2en_minimal_zero-shot under --out-dir, each with a
{name}_{file}_minimal_zero-shot_sharegpt.json (list of {conversations, system}) and a metadata
file, in the exact layout the RGDC pool uses. Run src/data/generate_mid_zero_shot.py on each
folder afterwards to produce the mid_zero-shot variant that Phase 1 consumes.

Usage:
  python -m src.data.prepare_corpus --input wikimedia.csv --name wikimedia \
      --en-col en --th-col th --out-dir data/source/sharegpt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SYSTEM = {"en2th": "You are a professional English-Thai translator.", "th2en": "You are a professional Thai-English translator."}


def write_direction(name: str, fname: str, pairs: list[tuple[str, str]], direction: str, out_dir: Path) -> Path:
    folder = out_dir / f"{name}_{direction}_minimal_zero-shot"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for en, th in pairs:
        src, tgt = (en, th) if direction == "en2th" else (th, en)
        records.append({"conversations": [{"from": "human", "value": src}, {"from": "gpt", "value": tgt}], "system": SYSTEM[direction]})
    out = folder / f"{name}_{fname}_minimal_zero-shot_sharegpt.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {"dataset_name": name, "prompt_style": "zero-shot", "few_shot_k": 0, "context_pool_size": 1000,
            "files_processed": {fname: len(records)}, "total_samples": len(records)}
    (folder / f"{name}_{direction}_minimal_zero-shot_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", required=True, help="corpus name, e.g. wikimedia")
    ap.add_argument("--en-col", default="en")
    ap.add_argument("--th-col", default="th")
    ap.add_argument("--out-dir", default="data/source/sharegpt")
    args = ap.parse_args()
    path = Path(args.input)
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    pairs = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=delim):
            pairs.append(((row.get(args.en_col) or ""), (row.get(args.th_col) or "")))
    for d in ("en2th", "th2en"):
        out = write_direction(args.name, path.name, pairs, d, Path(args.out_dir))
        print(f"wrote {len(pairs)} records to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
