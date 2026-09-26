"""Name the result folder of a judge: its served model name plus a fingerprint of the model name,
the judge prompt, and the verdict-relevant judge settings.

AlpacaEval caches verdicts by prompt and outputs only, so a new judge or an edited rubric must write
to a new folder to be judged afresh. Concurrency and retry settings are left out of the fingerprint
because they do not change verdicts.

Usage:
  python -m src.eval.judge_id --model "$JUDGE_MODEL_NAME" --config config/evaluation/judge_config.yaml
  python -m src.eval.judge_id --model gpt-oss-120b --prompt config/evaluation/judge_prompt.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

OPERATIONAL_KEYS = {"num_procs", "n_retries", "client_kwargs"}


def judge_id(model: str, config: Path | None = None, prompt: Path | None = None) -> str:
    h = hashlib.sha256(model.encode("utf-8"))
    prompts = [prompt] if prompt else []
    if config:
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        for annot in cfg.values():
            if not isinstance(annot, dict):
                continue
            kwargs = annot.get("completions_kwargs", {})
            for key in OPERATIONAL_KEYS:
                kwargs.pop(key, None)
            if "prompt_template" in annot:
                prompts.append(Path(annot["prompt_template"]))
        h.update(json.dumps(cfg, sort_keys=True).encode("utf-8"))
    for p in prompts:
        h.update(p.read_bytes())
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")
    return f"{name}-{h.hexdigest()[:8]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="Judge model name as served")
    ap.add_argument("--config", type=Path, help="AlpacaEval judge config; its prompt_template is included")
    ap.add_argument("--prompt", type=Path, help="Judge prompt file, when no config is used")
    args = ap.parse_args()
    print(judge_id(args.model, args.config, args.prompt))


if __name__ == "__main__":
    main()
