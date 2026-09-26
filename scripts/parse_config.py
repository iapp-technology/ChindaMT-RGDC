#!/usr/bin/env python3
"""Parse a YAML config file and output key=value pairs for bash consumption.

Usage in bash:
    eval "$(python3 scripts/parse_config.py config/rgdc_qwen35.yaml phase1a)"
    echo $source_data_dir
"""

import sys
import yaml


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix=f"{prefix}{k}_"))
        elif isinstance(v, list):
            out[f"{prefix}{k}"] = " ".join(str(x) for x in v)
        else:
            out[f"{prefix}{k}"] = str(v) if v is not None else ""
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_config.py <config.yaml> [section]", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    section = sys.argv[2] if len(sys.argv) > 2 else None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if section:
        config = config.get(section, {})

    for k, v in flatten(config).items():
        print(f'{k}="{v}"')


if __name__ == "__main__":
    main()
