"""Dispatcher: loads a model config YAML and invokes the right backend.

Usage:
    python -m src.eval.run_inference --config config/evaluation/default.yaml \\
        --suite coreeval [--mode hf|vllm|api] [--splits plain constrained] [--limit N]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def expand_env(value):
    """Expand ${VAR} references in strings."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def build_hf_cmd(cfg: dict, args) -> list[str]:
    inf = cfg.get("inference", {})
    cmd = [
        sys.executable, "-m", "src.eval.inference_hf",
        "--model-name-or-path", cfg["model_name_or_path"],
        "--generator-name", cfg["generator_name"],
        "--suite-dir", str(args.suite_dir),
        "--output-dir", str(args.output_dir),
        "--suite", args.suite,
    ]
    if cfg.get("model_family"):
        cmd += ["--model-family", cfg["model_family"]]
    if args.splits:
        cmd += ["--splits"] + list(args.splits)
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    for k, flag in [
        ("max_new_tokens", "--max-new-tokens"),
        ("temperature", "--temperature"),
        ("top_p", "--top-p"),
        ("top_k", "--top-k"),
        ("repetition_penalty", "--repetition-penalty"),
        ("batch_size", "--batch-size"),
    ]:
        if k in inf:
            cmd += [flag, str(inf[k])]
    if cfg.get("dtype"):
        cmd += ["--dtype", cfg["dtype"]]
    if cfg.get("prompt_adapter"):
        cmd += ["--prompt-adapter", cfg["prompt_adapter"]]
    return cmd


def build_vllm_cmd(cfg: dict, args) -> list[str]:
    inf = cfg.get("inference", {})
    cmd = [
        sys.executable, "-m", "src.eval.inference_vllm",
        "--model-name-or-path", cfg["model_name_or_path"],
        "--generator-name", cfg["generator_name"],
        "--suite-dir", str(args.suite_dir),
        "--output-dir", str(args.output_dir),
        "--suite", args.suite,
    ]
    if cfg.get("model_family"):
        cmd += ["--model-family", cfg["model_family"]]
    if args.splits:
        cmd += ["--splits"] + list(args.splits)
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    for k, flag in [
        ("max_new_tokens", "--max-new-tokens"),
        ("temperature", "--temperature"),
        ("top_p", "--top-p"),
        ("top_k", "--top-k"),
        ("repetition_penalty", "--repetition-penalty"),
    ]:
        if k in inf:
            cmd += [flag, str(inf[k])]
    if cfg.get("dtype"):
        cmd += ["--dtype", cfg["dtype"]]
    if cfg.get("tensor_parallel_size"):
        cmd += ["--tensor-parallel-size", str(cfg["tensor_parallel_size"])]
    return cmd


def build_api_cmd(cfg: dict, args) -> list[str]:
    inf = cfg.get("inference", {})
    cmd = [
        sys.executable, "-m", "src.eval.inference_api",
        "--base-url", cfg["api_base_url"],
        "--api-key", cfg["api_key"],
        "--model", cfg["api_model_name"],
        "--generator-name", cfg["generator_name"],
        "--suite-dir", str(args.suite_dir),
        "--output-dir", str(args.output_dir),
        "--suite", args.suite,
    ]
    if args.splits:
        cmd += ["--splits"] + list(args.splits)
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    for k, flag in [
        ("max_new_tokens", "--max-new-tokens"),
        ("temperature", "--temperature"),
        ("top_p", "--top-p"),
        ("max_concurrent", "--max-concurrent"),
    ]:
        if k in inf:
            cmd += [flag, str(inf[k])]
    family = cfg.get("model_family", "")
    if family in ("qwen3_5", "qwen3"):
        cmd += ["--disable-thinking"]
    return cmd


def resolve_model_config(main_cfg: dict, model_ref: str) -> dict:
    """Model reference may be a YAML path or an already-loaded dict."""
    if isinstance(model_ref, dict):
        return expand_env(model_ref)
    cfg = load_yaml(Path(model_ref))
    return expand_env(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for a model on eval suites")
    parser.add_argument("--config", required=True, type=Path, help="Path to main eval config OR a model config")
    parser.add_argument("--model-ref", default=None, help="Specific model config path (defaults to target_model from main config)")
    parser.add_argument("--mode", choices=["hf", "vllm", "api"], default=None, help="Override mode from config")
    parser.add_argument("--suite", default="coreeval", help="Suite name under `suites` in the config, e.g. coreeval or broadeval")
    parser.add_argument("--splits", nargs="+", default=None, help="Override the split list")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    main_cfg = load_yaml(args.config)

    if args.model_ref:
        model_cfg = resolve_model_config(main_cfg, args.model_ref)
    elif "target_model" in main_cfg:
        model_cfg = resolve_model_config(main_cfg, main_cfg["target_model"])
    else:
        model_cfg = expand_env(main_cfg)

    mode = args.mode or model_cfg.get("mode", "hf")

    suite_cfg = main_cfg.get("suites", {}).get(args.suite, {})
    suite_dir = Path(suite_cfg.get("path", f"data/eval/{args.suite}"))
    if not args.splits:
        args.splits = suite_cfg.get("splits", ["plain", "constrained"])

    output_dir = Path(main_cfg.get("predictions_dir", "predictions"))

    args.suite_dir = suite_dir
    args.output_dir = output_dir

    if mode == "hf":
        cmd = build_hf_cmd(model_cfg, args)
    elif mode == "vllm":
        cmd = build_vllm_cmd(model_cfg, args)
    elif mode == "api":
        cmd = build_api_cmd(model_cfg, args)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
