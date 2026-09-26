#!/bin/bash
# Phase 1A: Pre-Experience Sampling
# Computes embeddings + perplexity per corpus, K-Means clusters, selects middle-confidence samples.
#
# Usage: bash scripts/01_pre_experience.sh config/rgdc_qwen35.yaml

set -e

CONFIG="${1:?Usage: $0 <config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

eval "$(python3 scripts/parse_config.py "$CONFIG" phase1a)"

export OPENBLAS_NUM_THREADS=4

mkdir -p "$output_dir"

find -L "$source_data_dir" -name "*_sharegpt.json" 2>/dev/null | grep -E "_${template_filter}" | while read -r sharegpt_file; do
    folder_name=$(basename "$(dirname "$sharegpt_file")")
    file_name=$(basename "$sharegpt_file" .json)
    dataset_id="${folder_name}__${file_name}"

    alpaca_file="$output_dir/${dataset_id}_alpaca.json"
    pt_file="$output_dir/${dataset_id}_pre.pt"
    out_file="$output_dir/${dataset_id}_pre_1k.json"

    if [ -f "$out_file" ]; then
        echo "Skipping $dataset_id (already processed)"
        continue
    fi

    echo "Processing: $dataset_id"

    python3 src/cherry/alpaca_converter.py "$sharegpt_file" \
        -o "$alpaca_file" --format json \
        --number-samples "$max_samples_per_corpus" --seed "$seed"

    python3 src/cherry/data_analysis.py \
        --data_path "$alpaca_file" \
        --save_path "$pt_file" \
        --model_name_or_path "$base_model" \
        --max_length "$max_length" \
        --prompt alpaca \
        --mod pre

    python3 src/cherry/data_by_cluster.py \
        --pt_data_path "$pt_file" \
        --json_data_path "$alpaca_file" \
        --json_save_path "$out_file" \
        --sample_num "$samples_per_cluster" \
        --kmeans_num_clusters "$kmeans_clusters" \
        --low_th "$low_th" \
        --up_th "$up_th"

    rm -f "$alpaca_file" "$pt_file"
    echo "Completed: $out_file"
done

echo "Concatenating pre-experience datasets..."
python3 src/cherry/concat_dataset.py \
    --root-dir "$output_dir" \
    --save-path "$output_concat" \
    --file-suffix "_pre_1k.json"

echo "Done: $output_concat"
