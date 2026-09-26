"""
pack_augmented_prompts.py

Pack queries with constraints from extract_constraint_results_qwen.jsonl
and create prompts for LLM inference.

Format:
    [Original Instruction]

    Rules:
    1. [Constraint 1]
    2. [Constraint 2]

    [Original Input]
"""

import json
import jsonlines
import argparse
import random
import re
from tqdm import tqdm


def parse_constraint_json(output_str):
    """Parse constraint JSON from LLM output (handles ```json...``` blocks)."""
    try:
        # Extract JSON from ```json...``` block
        pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        matches = pattern.findall(output_str)
        if matches:
            json_str = matches[0]
        else:
            json_str = output_str

        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_eval_question_json(output_str):
    """Parse evaluation question JSON from LLM output.
    Handles both batched format {"1": "q1", "2": "q2"} and
    legacy single format {"question": "..."}."""
    try:
        pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        matches = pattern.findall(output_str)
        json_str = matches[0] if matches else output_str
        data = json.loads(json_str)
        if "question" in data:
            return data.get("question", "")
        return data
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


def flatten_constraints(constraint_dict):
    """Flatten constraint dict into list of (constraint_text, constraint_index)."""
    constraints = []
    cnt = 0
    for category, items in constraint_dict.items():
        if isinstance(items, list):
            for item in items:
                cnt += 1
                if isinstance(item, dict) and "constraint" in item:
                    constraints.append((item["constraint"], cnt))
                elif isinstance(item, str):
                    constraints.append((item, cnt))
    return constraints


def load_constraints(constraints_file):
    """Load constraints and build mapping by request index."""
    constraint_map = {}
    with jsonlines.open(constraints_file, "r") as reader:
        for row in tqdm(reader, desc="Loading constraints"):
            custom_id = row.get("custom_id", "")
            output = row.get("output", "")

            # Extract index from custom_id (e.g., "request-11" -> 11)
            match = re.match(r"request-(\d+)", custom_id)
            if match:
                idx = int(match.group(1))
                parsed = parse_constraint_json(output)
                if parsed:
                    constraint_map[idx] = parsed
    return constraint_map


def load_eval_questions(eval_questions_file):
    """Load evaluation questions and build mapping {idx: {cnt: question}}.
    Supports batched format (custom_id=request-{idx}, output={"1":"q1","2":"q2"})
    and legacy format (custom_id=request-{idx}-{cnt}, output={"question":"..."})."""
    eval_map = {}
    with jsonlines.open(eval_questions_file, "r") as reader:
        for row in tqdm(reader, desc="Loading eval questions"):
            custom_id = row.get("custom_id", "")
            output = row.get("output", "")

            parsed = parse_eval_question_json(output)
            if not parsed:
                continue

            match_batched = re.match(r"request-(\d+)$", custom_id)
            match_legacy = re.match(r"request-(\d+)-(\d+)", custom_id)

            if match_batched:
                idx = int(match_batched.group(1))
                if isinstance(parsed, dict):
                    eval_map[idx] = {}
                    for key, value in parsed.items():
                        try:
                            cnt = int(key)
                        except ValueError:
                            continue
                        if isinstance(value, str) and value.strip():
                            eval_map[idx][cnt] = value.strip()
            elif match_legacy:
                idx = int(match_legacy.group(1))
                cnt = int(match_legacy.group(2))
                if isinstance(parsed, str) and parsed:
                    if idx not in eval_map:
                        eval_map[idx] = {}
                    eval_map[idx][cnt] = parsed
    return eval_map


def format_augmented_prompt(instruction, input_text, constraints):
    """
    Format prompt with instruction, rules, and input.

    Format:
        [Original Instruction]

        Rules:
        1. [Constraint 1]
        2. [Constraint 2]

        [Original Input]
    """
    # Format rules
    rules_text = "Rules:\n"
    for i, (constraint, _) in enumerate(constraints, 1):
        rules_text += f"{i}. {constraint}\n"

    # Combine
    if input_text.strip():
        prompt = f"{instruction}\n\n{rules_text}\n{input_text}"
    else:
        prompt = f"{instruction}\n\n{rules_text}"

    return prompt.strip()


def create_request(custom_id, prompt, system_prompt, model="Qwen/Qwen3.5-35B-A3B-FP8", metadata=None):
    """Create request in call_endpoint format."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    }

    if metadata:
        request["metadata"] = metadata

    return request


def main():
    parser = argparse.ArgumentParser(description="Pack augmented prompts with constraints")
    parser.add_argument("--input", required=True, help="Path to source queries JSON file")
    parser.add_argument("--constraints", required=True, help="Path to constraints JSONL file")
    parser.add_argument("--eval-questions", required=True, help="Path to eval questions JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file for prompts")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process")
    parser.add_argument("--min-sample", type=int, default=1, help="Min constraints to sample")
    parser.add_argument("--max-sample", type=int, default=3, help="Max constraints to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8", help="Model name for requests")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load source data
    print(f"Loading source data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        source_data = json.load(f)
    print(f"Loaded {len(source_data)} source records")

    # Load constraints
    print(f"Loading constraints from {args.constraints}...")
    constraint_map = load_constraints(args.constraints)
    print(f"Loaded constraints for {len(constraint_map)} records")

    # Load evaluation questions
    print(f"Loading eval questions from {args.eval_questions}...")
    eval_map = load_eval_questions(args.eval_questions)
    print(f"Loaded eval questions for {len(eval_map)} records")

    # System prompt
    system_prompt = "You are an expert tasked with answering the given query. Please provide a clear and concise response directly, without introductory phrases such as 'What a great question,' 'Here is the answer,' or similar expressions. Focus solely on addressing the query while strictly following its inside constraints."

    # Process and create prompts
    prompts_written = 0
    with jsonlines.open(args.output, "w") as writer:
        for idx, data in enumerate(tqdm(source_data, desc="Creating prompts")):
            if args.limit and idx >= args.limit:
                break

            # Check if this record has constraints
            if idx not in constraint_map:
                continue

            instruction = data.get("instruction", "")
            input_text = data.get("input", "")
            system_text = data.get("system", "")

            # Get and flatten constraints
            constraint_dict = constraint_map[idx]
            constraints = flatten_constraints(constraint_dict)

            if not constraints:
                continue

            # Get eval questions for this record
            eval_questions = eval_map.get(idx, {})

            # Variant 1: Sampled constraints (1-3 random)
            num_sample = min(random.randint(args.min_sample, args.max_sample), len(constraints))
            sampled_constraints = random.sample(constraints, num_sample)

            sampled_prompt = format_augmented_prompt(instruction, input_text, sampled_constraints)
            sampled_constraint_indices = [cnt for _, cnt in sampled_constraints]
            sampled_eval_questions = [eval_questions.get(cnt, "") for _, cnt in sampled_constraints]
            sampled_eval_questions = [q for q in sampled_eval_questions if q]  # Filter empty

            sampled_request = create_request(
                custom_id=f"request-{idx}-sampled",
                prompt=sampled_prompt,
                system_prompt=system_text if system_text else system_prompt,
                model=args.model,
                metadata={
                    "source_idx": idx,
                    "variant": "sampled",
                    "constraint_indices": sampled_constraint_indices,
                    "constraints": [c for c, _ in sampled_constraints],
                    "eval_questions": sampled_eval_questions,
                    "original_instruction": instruction,
                    "original_input": input_text,
                }
            )
            writer.write(sampled_request)
            prompts_written += 1

            # Variant 2: All constraints
            all_prompt = format_augmented_prompt(instruction, input_text, constraints)
            all_constraint_indices = [cnt for _, cnt in constraints]
            all_eval_questions = [eval_questions.get(cnt, "") for _, cnt in constraints]
            all_eval_questions = [q for q in all_eval_questions if q]  # Filter empty

            all_request = create_request(
                custom_id=f"request-{idx}-all",
                prompt=all_prompt,
                system_prompt=system_text if system_text else system_prompt,
                model=args.model,
                metadata={
                    "source_idx": idx,
                    "variant": "all",
                    "constraint_indices": all_constraint_indices,
                    "constraints": [c for c, _ in constraints],
                    "eval_questions": all_eval_questions,
                    "original_instruction": instruction,
                    "original_input": input_text,
                }
            )
            writer.write(all_request)
            prompts_written += 1

    print(f"\nDone! Wrote {prompts_written} prompts to {args.output}")


if __name__ == "__main__":
    main()
