"""HF transformers inference on evaluation test suites.

Model-agnostic: uses chat_template_kwargs from the model family registry,
overridable per model via config.
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

# Configure logging before transformers imports
_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_level, force=True)
if _level >= logging.WARNING:
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.format_outputs import (
    append_partial,
    build_instruction,
    clean_output,
    finalize_partial,
    load_partial,
    load_suite,
    partial_path,
    write_model_outputs,
)
from src.eval.model_registry import get_family_defaults


def load_model(model_path: str, dtype: str, device: str = "auto"):
    torch_dtype = getattr(torch, dtype) if dtype else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def render_prompt(tokenizer, instruction: str, chat_template_kwargs: dict) -> str:
    """Return the tokenizer-rendered prompt string (for batched tokenization)."""
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )


def run_inference(
    records: list[dict],
    model,
    tokenizer,
    generator_name: str,
    chat_template_kwargs: dict,
    sampling: dict[str, Any],
    partial_file: Path,
    start_idx: int = 0,
    batch_size: int = 8,
    prompt_adapter: str | None = None,
) -> int:
    """Run batched inference, appending each output to partial_file. Returns count written."""
    device = next(model.parameters()).device
    count = 0

    total = len(records) + start_idx
    pbar = tqdm(desc="inference", initial=start_idx, total=total)

    from src.eval.prompt_adapters import apply_adapter

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        # Standard instruction (saved in output for alpaca_eval join-by-instruction);
        # model_input may be adapter-rewritten for baselines trained on a different format.
        standard_instructions = [build_instruction(rec) for rec in batch]
        model_inputs = [apply_adapter(prompt_adapter, rec) or ins for rec, ins in zip(batch, standard_instructions)]
        prompts = [render_prompt(tokenizer, ins, chat_template_kwargs) for ins in model_inputs]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(device)

        # Use model's generation_config eos_token_id (may be a list of stop tokens,
        # e.g. Gemma-4 uses [<eos>=1, <turn|>=106, 50]); fall back to tokenizer.eos_token_id.
        eos_ids = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=sampling.get("max_new_tokens", 1024),
                temperature=sampling.get("temperature", 0.01),
                top_p=sampling.get("top_p", 0.7),
                top_k=sampling.get("top_k", 20),
                repetition_penalty=sampling.get("repetition_penalty", 1.05),
                do_sample=sampling.get("temperature", 0.01) > 0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )
        # Left-padded batch: sliced at input length for each row
        input_len = enc["input_ids"].shape[1]
        for instruction, row in zip(standard_instructions, gen):
            new_tokens = row[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            cleaned = clean_output(text)
            record = {
                "instruction": instruction,
                "output": cleaned,
                "generator": generator_name,
            }
            append_partial(partial_file, record)
            count += 1
            pbar.update(1)

    pbar.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="HF transformers inference on eval suites")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--generator-name", required=True)
    parser.add_argument("--model-family", default=None, help="Key in MODEL_REGISTRY (qwen3_5, hymt, gemma3, ...)")
    parser.add_argument("--suite-dir", required=True, type=Path, help="Directory with <split>.jsonl files")
    parser.add_argument("--splits", nargs="+", default=["plain", "constrained"])
    parser.add_argument("--output-dir", required=True, type=Path, help="e.g. predictions")
    parser.add_argument("--suite", default="coreeval", help="Suite name, used in the output folder name")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size (increase if GPU has room)")
    parser.add_argument("--dtype", default=None, help="bfloat16 / float16 / float32 (defaults to family)")
    parser.add_argument("--prompt-adapter", default=None, help="Name in prompt_adapters.ADAPTERS (e.g., typhoon_native, gemmax2_native)")
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

    model = None
    tokenizer = None

    for split in args.splits:
        input_path = args.suite_dir / f"{split}.jsonl"
        if not input_path.exists():
            print(f"Skipping missing: {input_path}")
            continue

        records = load_suite(input_path)
        if args.limit:
            records = records[:args.limit]

        generator = f"{args.generator_name}-{args.suite}-{split}"
        out_dir = args.output_dir / generator
        final_file = out_dir / "model_outputs.json"
        partial_file = partial_path(final_file)

        if final_file.exists():
            print(f"[{split}] Skipped, final output exists: {final_file}")
            continue

        already = load_partial(partial_file)
        start_idx = len(already)
        if start_idx > 0:
            print(f"[{split}] Resuming from record {start_idx}/{len(records)}")
        to_run = records[start_idx:]

        if to_run:
            if model is None:
                print(f"Loading model: {args.model_name_or_path} (dtype={dtype}, family={args.model_family})")
                model, tokenizer = load_model(args.model_name_or_path, dtype)
            print(f"\nRunning {split} ({len(to_run)} remaining records) -> {generator}")
            run_inference(to_run, model, tokenizer, generator, chat_template_kwargs, sampling, partial_file, start_idx=start_idx, batch_size=args.batch_size, prompt_adapter=args.prompt_adapter)

        count = finalize_partial(partial_file, final_file)
        print(f"Wrote {count} records to {final_file}")


if __name__ == "__main__":
    main()
