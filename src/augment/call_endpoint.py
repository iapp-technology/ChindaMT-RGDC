# call_endpoint.py
import jsonlines
import re
import os
import asyncio
import aiohttp
from tqdm import tqdm
from collections import defaultdict


def strip_thinking(text):
    """Remove <think>...</think> blocks from output."""
    return re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)


def load_processed_ids(output_file):
    """Load already processed custom_ids from output file."""
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            for row in jsonlines.open(output_file, "r"):
                cid = row.get("custom_id")
                if cid is not None:
                    processed_ids.add(cid)
        except Exception as e:
            print(f"Warning: couldn't read existing output for resume: {e}")
    return processed_ids


async def call_endpoint_async(input_file, output_file, base_url, api_key, max_concurrent=20, batch_size=20, delay_between_requests=0.0, resume=True, max_retries=3, retry_backoff=2.0):
    """
    Async concurrent requests using aiohttp.

    Args:
        input_file: Path to input JSONL file with requests
        output_file: Path to output JSONL file for results
        base_url: Base URL of the API endpoint
        api_key: API key for authorization
        max_concurrent: Maximum number of concurrent requests (default: 20)
        batch_size: Number of results to buffer before writing to disk (default: 10)
        delay_between_requests: Seconds to wait between each request start (default: 0.0, no delay)
        resume: If True, skip already processed custom_ids (default: True)
        max_retries: Maximum number of retry attempts for failed requests (default: 3)
        retry_backoff: Backoff multiplier for retry delays (default: 2.0)
    """
    requests_data = list(jsonlines.open(input_file, "r"))
    print(f"Total requests: {len(requests_data)}")

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    processed_ids = load_processed_ids(output_file) if resume else set()
    pending_requests = [r for r in requests_data if r.get("custom_id") not in processed_ids]
    print(f"Pending requests: {len(pending_requests)} (skipped {len(processed_ids)} already processed)")

    if not pending_requests:
        print("All requests already processed.")
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    results_buffer = []
    buffer_lock = asyncio.Lock()
    error_counts = defaultdict(int)
    success_count = 0
    pbar = tqdm(total=len(pending_requests), desc="Async requests")

    # Open file for incremental writing
    fp = open(output_file, "a", encoding="utf-8")
    writer = jsonlines.Writer(fp)

    async def flush_buffer():
        async with buffer_lock:
            if results_buffer:
                writer.write_all(results_buffer)
                fp.flush()
                results_buffer.clear()

    async def process_single(session, req):
        cid = req.get("custom_id")
        async with semaphore:
            # Optional delay after acquiring semaphore (rate limiting)
            if delay_between_requests > 0:
                await asyncio.sleep(delay_between_requests)
            
            # Retry logic for handling 502 and other retryable errors
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    async with session.post(
                        f"{base_url}{req['url']}",
                        json=req["body"],
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as response:
                        # Check for retryable status codes
                        if response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            wait_time = float(retry_after) if retry_after else retry_backoff ** attempt
                            if attempt < max_retries:
                                await asyncio.sleep(wait_time)
                                last_error = "HTTP 429: Rate limited"
                                continue
                            else:
                                result = {"custom_id": cid, "error": "HTTP 429: Rate limited (exhausted retries)"}
                                break
                        if response.status in (500, 502, 503, 504):
                            if attempt < max_retries:
                                wait_time = retry_backoff ** attempt
                                await asyncio.sleep(wait_time)
                                last_error = f"HTTP {response.status}: {response.reason}"
                                continue
                            else:
                                last_error = f"HTTP {response.status}: {response.reason}"
                                result = {"custom_id": cid, "error": last_error}
                                break
                        
                        response.raise_for_status()
                        data = await response.json()
                        output = data["choices"][0]["message"]["content"]
                        content = strip_thinking(output)
                        result = {"custom_id": cid, "output": content}
                        break
                        
                except aiohttp.ClientResponseError as e:
                    if e.status in (429, 500, 502, 503, 504) and attempt < max_retries:
                        wait_time = retry_backoff ** attempt
                        await asyncio.sleep(wait_time)
                        last_error = str(e)
                        continue
                    else:
                        result = {"custom_id": cid, "error": f"HTTP {e.status}: {e.message}"}
                        break
                        
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    # Retry on network errors and timeouts
                    if attempt < max_retries:
                        wait_time = retry_backoff ** attempt
                        await asyncio.sleep(wait_time)
                        last_error = str(e)
                        continue
                    else:
                        result = {"custom_id": cid, "error": str(e)}
                        break
                        
                except Exception as e:
                    # Don't retry on other exceptions (e.g., JSON parsing errors)
                    result = {"custom_id": cid, "error": str(e)}
                    break
            else:
                # If all retries exhausted
                result = {"custom_id": cid, "error": f"Max retries exceeded. Last error: {last_error}"}

            # Track errors
            nonlocal success_count
            if "error" in result and result["error"]:
                error_counts[str(result["error"])[:80]] += 1
            else:
                success_count += 1

            # Add to buffer and flush if needed
            async with buffer_lock:
                results_buffer.append(result)
                if len(results_buffer) >= batch_size:
                    writer.write_all(results_buffer)
                    fp.flush()
                    results_buffer.clear()

            pbar.update(1)
            return result

    try:
        async with aiohttp.ClientSession() as session:
            tasks = [process_single(session, req) for req in pending_requests]
            await asyncio.gather(*tasks)

        # Flush remaining results
        await flush_buffer()
    finally:
        pbar.close()
        fp.close()

    total_errors = sum(error_counts.values())
    print(f"\n=== Summary ===")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {total_errors}")
    if error_counts:
        for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    [{cnt}x] {err}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Async API endpoint caller for JSONL batch processing")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSONL file with requests")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSONL file for results")
    parser.add_argument("--base-url", "-u", required=True, help="Base URL of the API endpoint")
    parser.add_argument("--api-key", "-k", required=True, help="API key for authorization")
    parser.add_argument("--max-concurrent", "-c", type=int, default=30, help="Maximum concurrent requests (default: 20)")
    parser.add_argument("--batch-size", "-b", type=int, default=30, help="Results buffer size before writing (default: 20)")
    parser.add_argument("--delay", "-d", type=float, default=0.2, help="Delay between requests in seconds (default: 0.0)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing output file")
    parser.add_argument("--max-retries", "-r", type=int, default=3, help="Max retry attempts for failed requests (default: 3)")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Backoff multiplier for retries (default: 2.0)")

    args = parser.parse_args()

    asyncio.run(call_endpoint_async(
        input_file=args.input,
        output_file=args.output,
        base_url=args.base_url,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
        delay_between_requests=args.delay,
        resume=not args.no_resume,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff
    ))


if __name__ == "__main__":
    main()
