#!/bin/bash
# Run Phase 1A, 1C, and 2 from a config; Phase 1B is trained in LLaMA-Factory between 1A and 1C.
#
# Usage: bash scripts/run_all.sh config/rgdc_qwen35.yaml

set -e

CONFIG="${1:?Usage: $0 <config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Phase 1A: Pre-Experience Sampling"
bash "$SCRIPT_DIR/01_pre_experience.sh" "$CONFIG"

echo ""
echo "Phase 1C: IFD Cherry Selection"
bash "$SCRIPT_DIR/02_cherry_selection.sh" "$CONFIG"

echo ""
echo "Phase 2: Constraint Augmentation"
bash "$SCRIPT_DIR/03_constraint_augmentation.sh" "$CONFIG" all

echo ""
echo "Pipeline complete."
