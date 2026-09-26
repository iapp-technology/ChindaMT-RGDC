import jsonlines
import os
from tqdm import tqdm
import json
import re

generate_prompt_template = """You are an expert in crafting questions to evaluate whether a response to a query adheres to specific constraints.

For each constraint below, design a question that human evaluators can use to assess if the response meets that constraint. Each question should focus solely on its given constraint.

If a constraint is meaningless or is part of the content itself (e.g., descriptions, scenarios, examples), respond with an empty string for that constraint.

Example:
Query: Recommend books.
Constraints:
1. Use bullet points in your answer.
2. Recommend exactly ten books.

Response:
```json
{{
    "1": "Does the response use bullet points?",
    "2": "Does the response recommend exactly ten books?"
}}
```

Now generate evaluation questions for:
Query: {query}

Constraints:
{constraints_list}

Respond only in JSON format mapping constraint numbers to questions:
```json
{{
    "1": "question for constraint 1",
    "2": "question for constraint 2"
}}
```
""".strip()


def parse_constraint_json(output_str):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", output_str.strip(), re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(output_str.strip())
    except (json.JSONDecodeError, AttributeError):
        return None


def packing(data_path, batch_call_results, output_path, model="Qwen/Qwen3.5-35B-A3B-FP8"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    constraint_dict = {}
    with jsonlines.open(batch_call_results, "r") as reader:
        for row in tqdm(reader, desc="Loading constraints"):
            constraint_dict[row.get("custom_id", "")] = row.get("output", "")

    existing_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_count = sum(1 for _ in f)
        if existing_count > 0:
            print(f"Resuming from record {existing_count}")

    skipped = 0
    written = 0
    with jsonlines.open(output_path, "a") as writer:
        for idx, d in enumerate(tqdm(data, desc="Generating eval question prompts")):
            if idx < existing_count:
                continue

            cid = f"request-{idx}"
            output = constraint_dict.get(cid, "")
            if not output:
                skipped += 1
                continue

            parsed = parse_constraint_json(output)
            if not parsed:
                skipped += 1
                continue

            constraints_list = ""
            cnt = 0
            for key, value in parsed.items():
                if not isinstance(value, list):
                    continue
                for item in value:
                    cnt += 1
                    if isinstance(item, dict) and "constraint" in item:
                        constraints_list += f"{cnt}. {item['constraint']}\n"
                    elif isinstance(item, str):
                        constraints_list += f"{cnt}. {item}\n"

            if cnt == 0:
                skipped += 1
                continue

            simplified_instruction = "\n".join([d["instruction"], d["input"]])
            messages = [
                {
                    "role": "user",
                    "content": generate_prompt_template.format(
                        query=simplified_instruction,
                        constraints_list=constraints_list.strip(),
                    ),
                }
            ]

            writer.write(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages,
                        "max_tokens": 512,
                        "temperature": 0,
                        "top_p": 1.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                }
            )
            written += 1

    print(f"Done: {written} prompts written, {skipped} skipped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation question prompts")
    parser.add_argument("--input", required=True, help="Input data file (JSON)")
    parser.add_argument("--constraints", required=True, help="Constraint results file (JSONL)")
    parser.add_argument("--output", required=True, help="Output prompts file (JSONL)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    args = parser.parse_args()

    packing(args.input, args.constraints, args.output, model=args.model)
