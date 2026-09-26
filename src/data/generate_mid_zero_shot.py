#!/usr/bin/env python3
"""
Convert minimal_zero-shot ShareGPT data to mid_zero-shot format.

Applies instruction diversification by randomly assigning A (opener),
C (input format), and D (output cue) template modules to each record.

Usage:
    python src/data/generate_mid_zero_shot.py \
        --input data/source/sharegpt/wikimedia_en2th_minimal_zero-shot/wikimedia_wikimedia.csv_minimal_zero-shot_sharegpt.json \
        --output-dir data/source/pool/wikimedia_en2th_mid_zero-shot/ \
        --source-lang English --target-lang Thai \
        --seed 42

    python src/data/generate_mid_zero_shot.py \
        --input data/source/sharegpt/wikimedia_th2en_minimal_zero-shot/wikimedia_wikimedia.csv_minimal_zero-shot_sharegpt.json \
        --output-dir data/source/pool/wikimedia_th2en_mid_zero-shot/ \
        --source-lang Thai --target-lang English \
        --seed 42
"""

import json
import random
import argparse
import os
from pathlib import Path


OPENERS = {
    "en2th": [
        "Translate this from English to Thai.",
        "Convert the following text from English to Thai.",
        "Please provide the Thai translation for the following English text.",
        "Translate the given content into Thai.",
        "You are translating between English and Thai.",
        "Task: Translate English -> Thai.",
        "Perform a translation task from English to Thai.",
    ],
    "th2en": [
        "Translate this from Thai to English.",
        "Convert the following text from Thai to English.",
        "Please provide the English translation for the following Thai text.",
        "Translate the given content into English.",
        "You are translating between Thai and English.",
        "Task: Translate Thai -> English.",
        "Perform a translation task from Thai to English.",
    ],
}

SYSTEM_MESSAGES = {
    "en2th": "You are a professional English-Thai translator.",
    "th2en": "You are a professional Thai-English translator.",
}


def get_input_formats(source_lang):
    """Return C-module templates. {text} is replaced with the source text."""
    lang_label = source_lang  # e.g. "English" or "Thai"
    return [
        lambda t, l=lang_label: f"{l}: {t}",                          # C1
        lambda t: f"Text to translate:\n{t}",                         # C2
        lambda t: f"Input text:\n---\n{t}\n---",                      # C3
        lambda t: f"Source:\n{t}",                                    # C4
        lambda t: f"Original text:\n{t}",                             # C5
        lambda t: t,                                                  # C6 (bare)
        lambda t: f"### Text: {t}",                                   # C7
        lambda t: f"Start: {t}",                                      # C8
        lambda t: f"> {t}",                                           # C9
    ]


def get_output_cues(target_lang):
    """Return D-module cue strings."""
    return [
        f"{target_lang}:",
        f"Translated ({target_lang}):",
        f"Output in {target_lang}:",
        "Translation:",
        "Result:",
        "Target:",
        "### Response:",
        "=>",
    ]


def build_mid_zero_shot(opener, input_formatted, output_cue):
    """Assemble mid_zero-shot prompt: A + C + D."""
    if output_cue:
        return f"{opener}\n{input_formatted}\n{output_cue}"
    return f"{opener}\n{input_formatted}"


def convert_record(record, direction, source_lang, target_lang, rng):
    """Convert a single minimal_zero-shot record to mid_zero-shot."""
    convs = record["conversations"]
    source_text = convs[0]["value"]
    target_text = convs[1]["value"]

    opener = rng.choice(OPENERS[direction])
    input_fn = rng.choice(get_input_formats(source_lang))
    output_cue = rng.choice(get_output_cues(target_lang))

    input_formatted = input_fn(source_text)
    human_value = build_mid_zero_shot(opener, input_formatted, output_cue)

    return {
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": target_text},
        ],
        "system": SYSTEM_MESSAGES[direction],
    }


def detect_direction(source_lang, target_lang):
    source_lower = source_lang.lower()
    target_lower = target_lang.lower()
    if source_lower == "english" and target_lower == "thai":
        return "en2th"
    elif source_lower == "thai" and target_lower == "english":
        return "th2en"
    else:
        raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert minimal_zero-shot ShareGPT to mid_zero-shot with template diversification"
    )
    parser.add_argument("--input", required=True, help="Input minimal_zero-shot ShareGPT JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for mid_zero-shot data")
    parser.add_argument("--source-lang", required=True, help="Source language (English or Thai)")
    parser.add_argument("--target-lang", required=True, help="Target language (English or Thai)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-canary", action="store_true", default=True,
                        help="Skip CANARY GUID records (default: True)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    direction = detect_direction(args.source_lang, args.target_lang)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.skip_canary:
        original_count = len(data)
        data = [r for r in data if "CANARY" not in r["conversations"][0]["value"]]
        skipped = original_count - len(data)
        if skipped > 0:
            print(f"Skipped {skipped} CANARY records")

    converted = []
    for record in data:
        converted.append(convert_record(record, direction, args.source_lang, args.target_lang, rng))

    os.makedirs(args.output_dir, exist_ok=True)

    input_stem = Path(args.input).stem.replace("minimal_zero-shot", "mid_zero-shot")
    output_file = os.path.join(args.output_dir, f"{input_stem}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    dir_name = os.path.basename(args.output_dir.rstrip("/"))
    metadata = {
        "dataset_name": dir_name,
        "prompt_style": "mid_zero-shot",
        "few_shot_k": 0,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "direction": direction,
        "total_samples": len(converted),
        "seed": args.seed,
        "source_file": args.input,
        "template_modules": {
            "A_count": len(OPENERS[direction]),
            "C_count": len(get_input_formats(args.source_lang)),
            "D_count": len(get_output_cues(args.target_lang)),
        },
    }
    metadata_file = os.path.join(args.output_dir, f"{dir_name}_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} records -> {output_file}")
    print(f"Metadata -> {metadata_file}")
    print(f"Direction: {direction}, Templates: {metadata['template_modules']}")


if __name__ == "__main__":
    main()
