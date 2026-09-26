"""
merge_final_dataset.py

Merge prompts, responses, and evaluations to create final augmented SFT dataset.
Filter based on evaluation scores (YES/NO).
"""

import json
import jsonlines
import argparse
import re
import ast
from tqdm import tqdm
from collections import defaultdict


def parse_evaluation_json(output_str):
    """Parse evaluation JSON from LLM output."""
    try:
        pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        matches = pattern.findall(output_str)
        if matches:
            json_str = matches[0]
        else:
            pattern2 = re.compile(r"(\{.*\})", re.DOTALL)
            matches2 = pattern2.findall(output_str)
            if matches2:
                json_str = matches2[0]
            else:
                json_str = output_str

        result = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            result = ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            return None

    try:
        json.dumps(result)
        return result
    except (TypeError, ValueError):
        return None


def check_all_passed(eval_result):
    """Check if all questions passed (score = YES)."""
    if not eval_result or not isinstance(eval_result, dict):
        return False

    for key, value in eval_result.items():
        if "Question" in key or key.startswith("Question"):
            if not isinstance(value, dict):
                return False
            score = value.get("score", "").upper()
            if score != "YES":
                return False

    return True


def count_passed(eval_result):
    """Count number of passed questions."""
    if not eval_result or not isinstance(eval_result, dict):
        return 0, 0

    passed = 0
    total = 0
    for key, value in eval_result.items():
        if "Question" in key or key.startswith("Question"):
            total += 1
            if isinstance(value, dict):
                score = value.get("score", "").upper()
                if score == "YES":
                    passed += 1

    return passed, total


def load_prompts_metadata(prompts_file):
    """Load prompts and return metadata map."""
    prompts_map = {}
    with jsonlines.open(prompts_file, "r") as reader:
        for row in tqdm(reader, desc="Loading prompts"):
            custom_id = row.get("custom_id", "")
            metadata = row.get("metadata", {})
            body = row.get("body", {})
            messages = body.get("messages", [])

            # Extract user message (the query)
            query = ""
            for msg in messages:
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break

            prompts_map[custom_id] = {
                "query": query,
                "metadata": metadata
            }
    return prompts_map


def load_responses(responses_file):
    """Load responses map."""
    responses_map = {}
    with jsonlines.open(responses_file, "r") as reader:
        for row in tqdm(reader, desc="Loading responses"):
            custom_id = row.get("custom_id", "")
            output = row.get("output", "")
            error = row.get("error", None)
            responses_map[custom_id] = {
                "output": output,
                "error": error
            }
    return responses_map


def load_evaluations(evaluations_file):
    """Load evaluations map."""
    eval_map = {}
    with jsonlines.open(evaluations_file, "r") as reader:
        for row in tqdm(reader, desc="Loading evaluations"):
            custom_id = row.get("custom_id", "")
            output = row.get("output", "")
            error = row.get("error", None)

            # Extract original custom_id from eval-{custom_id}
            if custom_id.startswith("eval-"):
                original_id = custom_id[5:]  # Remove "eval-" prefix
            else:
                original_id = custom_id

            eval_map[original_id] = {
                "output": output,
                "error": error
            }
    return eval_map


def main():
    parser = argparse.ArgumentParser(description="Merge final augmented SFT dataset")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSONL")
    parser.add_argument("--responses", required=True, help="Path to responses JSONL")
    parser.add_argument("--evaluations", required=True, help="Path to evaluations JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL file for final dataset")
    parser.add_argument("--filter-mode", choices=["all_pass", "any_pass", "none"],
                        default="all_pass", help="Filter mode: all_pass, any_pass, or none")
    parser.add_argument("--include-failed", action="store_true",
                        help="Also output failed records to separate file")
    args = parser.parse_args()

    # Load all data
    print("Loading data...")
    prompts_map = load_prompts_metadata(args.prompts)
    responses_map = load_responses(args.responses)
    eval_map = load_evaluations(args.evaluations)

    print(f"  Prompts: {len(prompts_map)}")
    print(f"  Responses: {len(responses_map)}")
    print(f"  Evaluations: {len(eval_map)}")

    # Stats
    stats = defaultdict(int)
    passed_count = 0
    failed_count = 0

    failed_output = args.output.replace(".jsonl", "_failed.jsonl")
    passed_fp = open(args.output, "w", encoding="utf-8")
    passed_writer = jsonlines.Writer(passed_fp)
    failed_fp = None
    failed_writer = None
    if args.include_failed:
        failed_fp = open(failed_output, "w", encoding="utf-8")
        failed_writer = jsonlines.Writer(failed_fp)

    try:
        for custom_id, prompt_data in tqdm(prompts_map.items(), desc="Merging"):
            metadata = prompt_data.get("metadata", {})

            if custom_id not in responses_map:
                stats["no_response"] += 1
                continue

            response_data = responses_map[custom_id]
            if response_data.get("error"):
                stats["response_error"] += 1
                continue

            response = response_data.get("output", "")
            if not response.strip():
                stats["empty_response"] += 1
                continue

            if custom_id not in eval_map:
                stats["no_evaluation"] += 1
                if args.filter_mode == "none":
                    record = {
                        "instruction": metadata.get("original_instruction", ""),
                        "input": metadata.get("original_input", ""),
                        "output": response,
                        "constraints": metadata.get("constraints", []),
                        "eval_questions": metadata.get("eval_questions", []),
                        "variant": metadata.get("variant", ""),
                        "source_idx": metadata.get("source_idx"),
                        "eval_result": None,
                        "all_passed": None,
                    }
                    passed_writer.write(record)
                    passed_count += 1
                continue

            eval_data = eval_map[custom_id]
            if eval_data.get("error"):
                stats["eval_error"] += 1
                continue

            eval_output = eval_data.get("output", "")
            parsed_eval = parse_evaluation_json(eval_output)

            if parsed_eval is None:
                stats["eval_parse_error"] += 1
                continue

            all_passed = check_all_passed(parsed_eval)
            pc, tc = count_passed(parsed_eval)

            record = {
                "instruction": metadata.get("original_instruction", ""),
                "input": metadata.get("original_input", ""),
                "output": response,
                "constraints": metadata.get("constraints", []),
                "eval_questions": metadata.get("eval_questions", []),
                "variant": metadata.get("variant", ""),
                "source_idx": metadata.get("source_idx"),
                "eval_result": parsed_eval,
                "all_passed": all_passed,
                "passed_count": pc,
                "total_count": tc,
            }

            if args.filter_mode == "all_pass":
                if all_passed:
                    passed_writer.write(record)
                    passed_count += 1
                    stats["passed_all"] += 1
                else:
                    if failed_writer:
                        failed_writer.write(record)
                    failed_count += 1
                    stats["failed_some"] += 1
            elif args.filter_mode == "any_pass":
                if pc > 0:
                    passed_writer.write(record)
                    passed_count += 1
                    stats["passed_any"] += 1
                else:
                    if failed_writer:
                        failed_writer.write(record)
                    failed_count += 1
                    stats["failed_all"] += 1
            else:
                passed_writer.write(record)
                passed_count += 1
                stats["included"] += 1
    finally:
        passed_fp.close()
        if failed_fp:
            failed_fp.close()

    print(f"\n=== Statistics ===")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")

    print(f"\n=== Summary ===")
    print(f"  Total prompts: {len(prompts_map)}")
    print(f"  Passed records: {passed_count}")
    print(f"  Failed records: {failed_count}")
    print(f"  Filter mode: {args.filter_mode}")


if __name__ == "__main__":
    main()
