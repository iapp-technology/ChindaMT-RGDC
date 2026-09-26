"""Helpers for AlpacaEval output format and output cleaning.

AlpacaEval format: [{"instruction": ..., "output": ..., "generator": ...}]
"""

import json
import re
from pathlib import Path

SPECIAL_TOKENS = [
    "<bos>", "<eos>", "<pad>", "<unk>",
    "<start_of_turn>", "<end_of_turn>",
    "<|im_start|>", "<|im_end|>",
    "<|startoftext|>", "<|endoftext|>",
    "<|user|>", "<|assistant|>", "<|system|>",
]

ROLE_PREFIXES = ["Human:", "Assistant:", "System:", "User:"]

THINK_PATTERN = re.compile(r"<think>\s*.*?\s*</think>\s*", flags=re.DOTALL | re.IGNORECASE)


def clean_output(text: str) -> str:
    """Strip special tokens, role prefixes, thinking blocks. Collapse whitespace."""
    if not text:
        return ""

    text = THINK_PATTERN.sub("", text)

    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")

    for prefix in ROLE_PREFIXES:
        if text.lstrip().startswith(prefix):
            text = text.lstrip()[len(prefix):]

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_model_outputs(path: Path, records: list[dict]) -> None:
    """Write records in AlpacaEval format (JSON array)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def partial_path(final_path: Path) -> Path:
    """Per-run resumable checkpoint path (JSONL)."""
    return final_path.with_suffix(".partial.jsonl")


def load_partial(path: Path) -> list[dict]:
    """Load already-processed records from partial JSONL (resume)."""
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_partial(path: Path, record: dict) -> None:
    """Append one record to partial JSONL (flushed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def finalize_partial(partial_path: Path, final_path: Path) -> int:
    """Convert partial JSONL to final JSON array; delete partial. Returns count."""
    records = load_partial(partial_path)
    write_model_outputs(final_path, records)
    partial_path.unlink(missing_ok=True)
    return len(records)


def load_suite(path: Path) -> list[dict]:
    """Load a suite JSONL file."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_instruction(record: dict) -> str:
    """Return the instruction string to send to the model.

    Suite rows carry a prepared 'instruction' field, with a Rules block in the
    constrained split. Fall back to a default translation prompt if missing.
    """
    if record.get("instruction"):
        return record["instruction"]

    direction = record.get("direction", "")
    source = record.get("source_text", "")
    if direction == "en->th":
        return f"Translate English to Thai.\n\nEN: {source}"
    if direction == "th->en":
        return f"Translate Thai to English.\n\nTH: {source}"
    return source
