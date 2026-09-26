"""
pack_evaluation_prompts.py

Create evaluation prompts to assess if responses meet constraints.
Uses evaluation questions from metadata in prompts file.
"""

import json
import jsonlines
import argparse
from tqdm import tqdm


EVALUATE_PROMPT = """You are an expert that is good at judging whether the response to a given query meets the specified evaluator questions.
Your task is to carefully examine the response to determine if it adheres to each requirement outlined in the evaluator questions.

[Query] {query}
[Response] {response}
[Evaluator Question] {question}

For each question, please provide a justification for your evaluation, explaining how the response does or does not satisfy the criteria and a score ('YES' or 'NO') indicating whether the answer satisfies each constraint.
You should only respond in the following JSON format:
```json
{{
    "Question 1": {{
        "explanation": "",
        "score": "YES" or "NO"
    }},
    "Question 2": {{
        "explanation": "",
        "score": "YES" or "NO"
    }},
}}
```
""".strip()


def load_prompts_with_metadata(prompts_file):
    """Load prompts and extract metadata."""
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
    """Load responses and build mapping by custom_id."""
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


def format_questions(eval_questions):
    """Format evaluation questions as numbered list."""
    if not eval_questions:
        return ""

    formatted = ""
    for i, q in enumerate(eval_questions, 1):
        formatted += f"{i}. {q}\n"
    return formatted.strip()


def create_eval_request(custom_id, query, response, questions, model="Qwen/Qwen3.5-35B-A3B-FP8", metadata=None):
    """Create evaluation request."""
    prompt = EVALUATE_PROMPT.format(
        query=query,
        response=response,
        question=questions
    )

    messages = [{"role": "user", "content": prompt}]

    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0,
            "top_p": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    }

    if metadata:
        request["metadata"] = metadata

    return request


def main():
    parser = argparse.ArgumentParser(description="Pack evaluation prompts")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSONL with metadata")
    parser.add_argument("--responses", required=True, help="Path to responses JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL file for evaluation prompts")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8", help="Model for evaluation")
    args = parser.parse_args()

    # Load prompts with metadata
    print(f"Loading prompts from {args.prompts}...")
    prompts_map = load_prompts_with_metadata(args.prompts)
    print(f"Loaded {len(prompts_map)} prompts")

    # Load responses
    print(f"Loading responses from {args.responses}...")
    responses_map = load_responses(args.responses)
    print(f"Loaded {len(responses_map)} responses")

    # Create evaluation prompts
    eval_count = 0
    skipped_no_response = 0
    skipped_error = 0
    skipped_no_questions = 0

    with jsonlines.open(args.output, "w") as writer:
        for custom_id, prompt_data in tqdm(prompts_map.items(), desc="Creating eval prompts"):
            # Get response
            if custom_id not in responses_map:
                skipped_no_response += 1
                continue

            response_data = responses_map[custom_id]

            # Skip if error
            if response_data.get("error"):
                skipped_error += 1
                continue

            response = response_data.get("output", "")
            if not response.strip():
                skipped_error += 1
                continue

            # Get metadata
            metadata = prompt_data.get("metadata", {})
            eval_questions = metadata.get("eval_questions", [])

            if not eval_questions:
                skipped_no_questions += 1
                continue

            # Format questions
            formatted_questions = format_questions(eval_questions)

            # Create evaluation request
            eval_request = create_eval_request(
                custom_id=f"eval-{custom_id}",
                query=prompt_data.get("query", ""),
                response=response,
                questions=formatted_questions,
                model=args.model,
                metadata={
                    "original_custom_id": custom_id,
                    "source_idx": metadata.get("source_idx"),
                    "variant": metadata.get("variant"),
                    "constraints": metadata.get("constraints", []),
                    "eval_questions": eval_questions,
                }
            )

            writer.write(eval_request)
            eval_count += 1

    print(f"\nDone!")
    print(f"  Created {eval_count} evaluation prompts")
    print(f"  Skipped (no response): {skipped_no_response}")
    print(f"  Skipped (error): {skipped_error}")
    print(f"  Skipped (no eval questions): {skipped_no_questions}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
