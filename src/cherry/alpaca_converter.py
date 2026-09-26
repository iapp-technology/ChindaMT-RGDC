#!/usr/bin/env python3
"""
Convert JSON/JSONL files from ShareGPT format to Alpaca format.

ShareGPT format:
{
    "conversations": [
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}
    ],
    "system": "..." (optional)
}

Alpaca format:
{
    "instruction": "...",
    "input": "...",
    "output": "..."
}
"""

import json
import random
import argparse
import sys
from pathlib import Path


def convert_sharegpt_to_alpaca(sharegpt_item):
    """
    Convert a single ShareGPT format item to Alpaca format.

    Args:
        sharegpt_item: Dictionary with ShareGPT format

    Returns:
        Dictionary with Alpaca format
    """
    conversations = sharegpt_item.get("conversations", [])
    system_message = sharegpt_item.get("system", "")

    # Extract human and assistant messages
    human_messages = []
    assistant_messages = []

    for conv in conversations:
        role = conv.get("from", "").lower()
        value = conv.get("value")

        # Skip entire item if any value is None
        if value is None:
            return None

        if role == "human":
            human_messages.append(value)
        elif role in ["gpt", "assistant", "gpt-4", "gpt-3.5-turbo"]:
            assistant_messages.append(value)

    # Use human message as instruction, input as empty string
    # split by newline between intstruction and input
    human_text = "\n".join(human_messages) if human_messages else ""

    # Split by newline: first line is instruction, rest is input
    # instruction = human_text.split("\n")[0] if human_text else ""
    # input_text = "\n".join(human_text.split("\n")[1:]) if human_text else ""

    instruction = human_text if human_text else ""
    input_text = ""

    # Join all assistant messages
    output_text = "\n".join(assistant_messages) if assistant_messages else ""

    result = {"instruction": instruction, "input": input_text, "output": output_text}

    # Add system message if present
    if system_message:
        result["system"] = system_message

    return result


def process_json_file(input_path, output_path, number_samples):
    """Process a JSON file (array format)."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]
    

    if number_samples is not None:
        if number_samples > len(data):
            print(f"Warning: Requested {number_samples} samples, but only {len(data)} available. Using all {len(data)} samples.")
            number_samples = len(data)
        data = random.sample(data, number_samples)

    alpaca_data = [convert_sharegpt_to_alpaca(item) for item in data]
    # Filter out None items (skipped due to invalid data)
    alpaca_data = [item for item in alpaca_data if item is not None]
    print(alpaca_data[0])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    skipped = len(data) - len(alpaca_data)
    print(f"Converted {len(alpaca_data)} items from {input_path} to {output_path} (skipped {skipped} invalid)")


def process_jsonl_file(input_path, output_path, number_samples):
    """Process a JSONL file (one JSON object per line)."""
    alpaca_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                if number_samples is not None and len(alpaca_data) >= number_samples:
                    break

                sharegpt_item = json.loads(line)

                alpaca_item = convert_sharegpt_to_alpaca(sharegpt_item)
                # Skip None items (invalid data)
                if alpaca_item is not None:
                    alpaca_data.append(alpaca_item)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Skipping line {line_num} due to JSON error: {e}",
                    file=sys.stderr,
                )

    # Write as JSON array
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(alpaca_data)} items from {input_path} to {output_path}")


def detect_format(input_path):
    """Detect if file is JSON or JSONL format."""
    with open(input_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            return "json"
        else:
            # Check first line to see if it's valid JSON
            first_line = f.readline().strip()
            try:
                json.loads(first_line)
                return "jsonl"
            except json.JSONDecodeError:
                # Try as JSON array
                f.seek(0)
                try:
                    json.load(f)
                    return "json"
                except json.JSONDecodeError:
                    return "jsonl"  # Default to jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT format to Alpaca format"
    )
    parser.add_argument(
        "input_file", type=str, help="Input file in ShareGPT format (JSON or JSONL)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path (default: input_file with '_alpaca' suffix)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl", "auto"],
        default="auto",
        help="Input file format (default: auto-detect)",
    )
    parser.add_argument(
        "--number-samples",
        type=int,
        default=None,
        help="Number of samples to convert (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random sampling (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    number_samples = args.number_samples

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_alpaca{input_path.suffix}"

    # Detect format
    if args.format == "auto":
        file_format = detect_format(input_path)
        print(f"Detected format: {file_format}")
    else:
        file_format = args.format

    # Process file
    try:
        if file_format == "json":
            process_json_file(input_path, output_path, number_samples)
        else:
            process_jsonl_file(input_path, output_path, number_samples)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
