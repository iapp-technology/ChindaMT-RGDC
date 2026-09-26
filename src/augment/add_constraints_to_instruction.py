#!/usr/bin/env python3
"""
Add constraints as rules to instruction field.

Transforms:
    instruction: "Perform a translation task..."
    constraints: ["Rule 1", "Rule 2", "Rule 3"]

Into:
    instruction: "Perform a translation task...

    Rules:
    - Rule 1
    - Rule 2
    - Rule 3"

Usage:
    python add_constraints_to_instruction.py --input augmented_sft_dataset.jsonl --output augmented_sft_with_rules.jsonl
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def add_constraints_to_instruction(item: dict) -> dict:
    """Add constraints as rules to the instruction field."""
    new_item = item.copy()

    instruction = item.get("instruction", "")
    constraints = item.get("constraints", [])

    if constraints:
        # Format constraints as bullet points
        rules_text = "\n".join(f"- {constraint}" for constraint in constraints)
        new_instruction = f"{instruction}\n\nRules:\n{rules_text}"
        new_item["instruction"] = new_instruction

    return new_item


def main():
    parser = argparse.ArgumentParser(
        description="Add constraints as rules to instruction field"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output JSONL file"
    )
    parser.add_argument(
        "--keep-constraints",
        action="store_true",
        help="Keep the constraints field in output (default: remove it)"
    )
    parser.add_argument(
        "--keep-eval-fields",
        action="store_true",
        help="Keep evaluation fields (eval_questions, eval_result, etc.)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Fields to remove for clean SFT dataset
    fields_to_remove = []
    if not args.keep_constraints:
        fields_to_remove.append("constraints")
    if not args.keep_eval_fields:
        fields_to_remove.extend([
            "eval_questions",
            "eval_result",
            "all_passed",
            "passed_count",
            "total_count",
            "variant",
            "source_idx"
        ])

    total_lines = sum(1 for line in open(input_path, "r", encoding="utf-8") if line.strip())

    processed_count = 0
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, total=total_lines, desc="Processing"):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            new_item = add_constraints_to_instruction(item)

            for field in fields_to_remove:
                new_item.pop(field, None)

            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            processed_count += 1

    print(f"Processed {processed_count} items")
    print(f"Output saved to: {output_path}")

    # Show example
    print("\n--- Example output ---")
    with open(output_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        example = json.loads(first_line)
        print(f"instruction:\n{example['instruction'][:500]}...")
        print(f"\nFields: {list(example.keys())}")


if __name__ == "__main__":
    main()
