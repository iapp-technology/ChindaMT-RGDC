"""vLLM-based inference on evaluation test suites.

Faster than HF for 400+ sample suites. Falls back to HF instructions on failure.
Requires: pip install vllm
"""

import argparse
from pathlib import Path
from typing import Any

from src.eval.format_outputs import build_instruction, clean_output, load_suite, write_model_outputs
from src.eval.model_registry import get_family_defaults


def run_inference(
    records: list[dict],
    model_path: str,
    generator_name: str,
    chat_template_kwargs: dict,
    sampling: dict[str, Any],
    dtype: str,
    tensor_parallel_size: int | None = None,
) -> list[dict]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError("vllm is not installed. `pip install vllm` in the chindamt environment.") from e

    llm = LLM(
        model=model_path,
        dtype=dtype,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size or 1,
    )
    tokenizer = llm.get_tokenizer()

    prompts = []
    instructions = []
    for rec in records:
        instruction = build_instruction(rec)
        instructions.append(instruction)
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        prompts.append(prompt)

    params = SamplingParams(
        max_tokens=sampling.get("max_new_tokens", 1024),
        temperature=sampling.get("temperature", 0.01),
        top_p=sampling.get("top_p", 0.7),
        top_k=sampling.get("top_k", 20),
        repetition_penalty=sampling.get("repetition_penalty", 1.05),
    )

    results = llm.generate(prompts, params)

    outputs = []
    for instruction, result in zip(instructions, results):
        raw = result.outputs[0].text
        outputs.append({
            "instruction": instruction,
            "output": clean_output(raw),
            "generator": generator_name,
        })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM inference on eval suites")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--generator-name", required=True)
    parser.add_argument("--model-family", default=None)
    parser.add_argument("--suite-dir", required=True, type=Path, help="Directory with <split>.jsonl files")
    parser.add_argument("--splits", nargs="+", default=["plain", "constrained"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite", default="coreeval", help="Suite name, used in the output folder name")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    args = parser.parse_args()

    family_defaults = get_family_defaults(args.model_family)
    dtype = args.dtype or family_defaults["dtype"]
    chat_template_kwargs = family_defaults["chat_template_kwargs"]

    sampling = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }

    all_records = {}
    for split in args.splits:
        input_path = args.suite_dir / f"{split}.jsonl"
        if not input_path.exists():
            print(f"Skipping missing: {input_path}")
            continue
        records = load_suite(input_path)
        if args.limit:
            records = records[:args.limit]
        all_records[split] = records

    # Run inference per split (separate LLM invocations not required, but keeps
    # memory usage predictable and matches HF behavior).
    for split, records in all_records.items():
        generator = f"{args.generator_name}-{args.suite}-{split}"
        print(f"\nRunning {split} ({len(records)} records) -> {generator}")

        outputs = run_inference(
            records=records,
            model_path=args.model_name_or_path,
            generator_name=generator,
            chat_template_kwargs=chat_template_kwargs,
            sampling=sampling,
            dtype=dtype,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        out_dir = args.output_dir / generator
        write_model_outputs(out_dir / "model_outputs.json", outputs)
        print(f"Wrote {len(outputs)} records to {out_dir}/model_outputs.json")


if __name__ == "__main__":
    main()
