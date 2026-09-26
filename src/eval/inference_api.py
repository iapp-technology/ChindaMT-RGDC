"""Async inference via OpenAI-compatible API endpoint.

Used when the fine-tuned model is deployed (e.g., to a vLLM serving endpoint).
Reuses the pattern from src/augment/call_endpoint.py: async, resumable, retry,
429 handling.
"""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
from tqdm import tqdm

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


async def call_one(session, base_url: str, api_key: str, model: str, instruction: str,
                   sampling: dict, chat_template_kwargs: dict, max_retries: int = 5,
                   retry_backoff: float = 2.0) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": instruction}],
        "max_tokens": sampling.get("max_new_tokens", 1024),
        "temperature": sampling.get("temperature", 0.01),
        "top_p": sampling.get("top_p", 0.7),
    }
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(max_retries + 1):
        try:
            async with session.post(
                f"{base_url}/chat/completions",
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else retry_backoff ** attempt
                    if attempt < max_retries:
                        await asyncio.sleep(wait)
                        continue
                    return ""
                if response.status in (500, 502, 503, 504):
                    if attempt < max_retries:
                        await asyncio.sleep(retry_backoff ** attempt)
                        continue
                    return ""
                response.raise_for_status()
                data = await response.json()
                return data["choices"][0]["message"]["content"]
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt < max_retries:
                await asyncio.sleep(retry_backoff ** attempt)
                continue
            return ""
    return ""


async def run_inference_async(records, base_url, api_key, model, generator_name,
                              sampling, chat_template_kwargs, partial_file,
                              done_indices, max_concurrent=24):
    """Run inference, appending completed records to partial_file as they finish.
    Skips any indices already present in done_indices.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    write_lock = asyncio.Lock()
    pending = [(i, rec) for i, rec in enumerate(records) if i not in done_indices]
    pbar = tqdm(total=len(pending), desc="api inference")

    async def process(session, idx, rec):
        async with semaphore:
            instruction = build_instruction(rec)
            raw = await call_one(session, base_url, api_key, model, instruction,
                                 sampling, chat_template_kwargs)
            record = {
                "_idx": idx,
                "instruction": instruction,
                "output": clean_output(raw),
                "generator": generator_name,
            }
            async with write_lock:
                append_partial(partial_file, record)
            pbar.update(1)

    async with aiohttp.ClientSession() as session:
        tasks = [process(session, idx, rec) for idx, rec in pending]
        await asyncio.gather(*tasks)

    pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="API endpoint inference on eval suites")
    parser.add_argument("--base-url", required=True, help="e.g. https://<your-vllm-endpoint>/v1")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True, help="Model name on the endpoint")
    parser.add_argument("--generator-name", required=True)
    parser.add_argument("--suite-dir", required=True, type=Path, help="Directory with <split>.jsonl files")
    parser.add_argument("--splits", nargs="+", default=["plain", "constrained"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite", default="coreeval", help="Suite name, used in the output folder name")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-concurrent", type=int, default=24)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()

    sampling = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    chat_template_kwargs = {"enable_thinking": False} if args.disable_thinking else {}

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
        done_indices = {rec.get("_idx") for rec in already if "_idx" in rec}
        if done_indices:
            print(f"[{split}] Resuming: {len(done_indices)}/{len(records)} already done")

        print(f"\nRunning {split} -> {generator}")

        asyncio.run(run_inference_async(
            records=records,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            generator_name=generator,
            sampling=sampling,
            chat_template_kwargs=chat_template_kwargs,
            partial_file=partial_file,
            done_indices=done_indices,
            max_concurrent=args.max_concurrent,
        ))

        # Reload, sort by _idx, strip _idx, write final
        all_records = load_partial(partial_file)
        all_records.sort(key=lambda r: r.get("_idx", 0))
        clean_records = [{k: v for k, v in r.items() if k != "_idx"} for r in all_records]
        write_model_outputs(final_file, clean_records)
        partial_file.unlink(missing_ok=True)
        print(f"Wrote {len(clean_records)} records to {final_file}")


if __name__ == "__main__":
    main()
