"""Per-model prompt adapters for fair comparison with MT specialists.

Some baselines (Typhoon-Translate, GemmaX2) are trained on a rigid prompt
template that differs from the shared prompt scaffold. This module rewrites the
instruction into each model's native format from the raw (direction, source_text)
fields. Registered by name; the per-model YAML sets `prompt_adapter: <name>`.

Plain split only: the constrained split carries rules that these native formats do not expose.
"""


def _direction(record: dict) -> tuple[str, str]:
    d = record.get("direction", "")
    if d == "en->th":
        return "English", "Thai"
    if d == "th->en":
        return "Thai", "English"
    raise ValueError(f"unknown direction: {d!r}")


def typhoon_native(record: dict) -> str:
    """Typhoon-Translate 1.5 native format (README, Prompting section)."""
    src_lang, tgt_lang = _direction(record)
    source = record.get("source_text", "")
    return (
        f"Translate the following {src_lang} text into {tgt_lang}, "
        f"return only the translated text.\n\n"
        f"Source Text:\n\n{source}"
    )


def gemmax2_native(record: dict) -> str:
    """GemmaX2-28 native format (README, Run the model)."""
    src_lang, tgt_lang = _direction(record)
    source = record.get("source_text", "")
    return f"Translate this from {src_lang} to {tgt_lang}:\n{src_lang}: {source}\n{tgt_lang}:"


def hymt_native(record: dict) -> str:
    """HY-MT 1.5 native format for non-Chinese pairs (README, Prompts section)."""
    _, tgt_lang = _direction(record)
    source = record.get("source_text", "")
    return f"Translate the following segment into {tgt_lang}, without additional explanation.\n\n{source}"


ADAPTERS = {
    "typhoon_native": typhoon_native,
    "gemmax2_native": gemmax2_native,
    "hymt_native": hymt_native,
}


def apply_adapter(adapter_name: str | None, record: dict) -> str | None:
    """Return adapter-transformed instruction, or None if no adapter set."""
    if not adapter_name:
        return None
    fn = ADAPTERS.get(adapter_name)
    if fn is None:
        raise ValueError(f"unknown prompt_adapter: {adapter_name!r}; registered: {list(ADAPTERS)}")
    return fn(record)
